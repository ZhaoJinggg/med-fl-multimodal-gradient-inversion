"""Image attack: simulation-based update matching on the ResNet-18 + head part of the update.

The victim's u local SGD steps on its batch are replayed on dummy images x' (BN in training mode,
known labels, text features of the attacker's condition), and x' is optimized so that the simulated
update matches the observed one in cosine distance, with TV, clipping (x' in [0, 1]) and BN priors.
At u = 1 this is the gradient matching of Geiping et al.

The victim update itself is produced here from the saved global model (round t), so the pilot can
vary u and the batch size without re-running the federation.

Protocols (--proto; paper Table I, Secs. "Threat Model" and "Attacks"): the victim's round is
protocols.client_upload, the attacker's replay protocols.simulate_image_steps (proximal term / SCAFFOLD
correction inside the unrolled steps). Auxiliary state comes from the warm-up run directory (the folder
of --global) via train_fl.load_aux for round --aux-round (default: the k of global_round{k}.pt):
SCAFFOLD control variates c, c_k (the server holds both); the victim's local BN state (FedBN, or
--bn-stats local), which only the client uses. When no BN statistics are uploaded (FedBN, --bn-stats
local) the BN prior is off; under FedBN the simulation keeps the BN entries of the broadcast file (the
server's pretrained ones) and the BN parameters, which are not uploaded, drop out of the distance.

Text conditions (--text; paper Sec. "Attacks", last paragraph):
  oracle     the victim's true per-step text features (client side; reference only)
  recovered  the text attack's result for the same victim (--text-from): recovered token ids as stored
             (at most B), sign-rule labels, label counts repaired by |coef_ls|, each text paired with
             one dummy image of its label; step j uses GPT-2 weights w_txt^t + (j-1)/u * dw_txt. Slots
             without a text are treated as under `none`.
  none       the text's contribution to the logit of each pair is one unknown scalar per pair and step,
             optimized with x'; the text part of the head is left out of the distance.
For a victim the three conditions use the same initialization of x' (seed 1000 + v + 100 r) and budget.

--text-from: a directory D (file D/v{v}_B{B}_u{u}{suffix}.json, attack_text.py's name for the same cell
with an empty --tag: suffix '' for FedAvg, _fedprox_mu1, _scaffold_optII, _fedbn, plus _bnlocal), a pattern
with the placeholders {v} {B} {u} {seed} {proto} {suffix} (e.g. "runs/text/seed1/v{v}_B{B}_u{u}{suffix}.json"),
or one file (single victim only). Relative paths are taken from the repository root. Identifiers the text
result carries (victim, batch, u, seed, proto, mu, scaffold_option, bn_stats) must match this run.

Every cell must be written to its own --out directory (score_images.py groups the mismatched control by
directory and tag, and the file name v{v}_B{B}_u{u}_{init}{tag} does not carry protocol or condition).

usage: python src/attack_image.py --global runs/pilot/seed0_fedavg/global_round9.pt --seed 0
          --victims 0 1 2 --batch 8 --u 1 --iters 10000 --text oracle --out runs/pilot/attacks
"""
import argparse
import copy
import hashlib
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity
from torch.func import functional_call

import protocols as P
from common import MAX_TOKENS, ROOT, FusionModel, PairData, is_image_param, load_seed

MOM = 0.1
EOS = 50256          # GPT-2 <|endoftext|>, also the pad token of common.tokenize


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--global", dest="glob", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--victims", type=int, nargs="+", default=[0])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--u", type=int, default=1)
    ap.add_argument("--lr-img", type=float, default=0.01)
    ap.add_argument("--lr-txt", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 2000, 5000, 10000])
    ap.add_argument("--opt-lr", type=float, default=0.1)
    ap.add_argument("--tv", type=float, default=1e-4)
    ap.add_argument("--bn", type=float, default=1e-2)
    ap.add_argument("--init", choices=["random", "mean", "near"], default="random",
                    help="mean: mean image of data outside the federation; near: true images + noise (control only)")
    ap.add_argument("--cos-w", type=float, default=1.0, help="weight of the matching loss (0 = BN statistics only)")
    ap.add_argument("--bn-mode", choices=["train", "eval"], default="train", help="eval: BN uses running statistics (control only)")
    ap.add_argument("--signed", action="store_true", help="use the sign of the gradient (Geiping et al.; images only)")
    ap.add_argument("--restarts", type=int, default=1, help="keep the restart with the lowest final matching loss")
    ap.add_argument("--text", choices=["oracle", "recovered", "none"], default="oracle")
    ap.add_argument("--text-from", default=None,
                    help="recovered: text attack result (directory, pattern with {v} {B} {u} {seed} {proto}, or file)")
    ap.add_argument("--lr-scalar", type=float, default=None,
                    help="Adam learning rate of the per-pair, per-step text scalars (none / missing texts); default --opt-lr")
    # protocols
    ap.add_argument("--proto", choices=P.PROTOCOLS, default="fedavg")
    ap.add_argument("--mu", type=float, default=1.0, help="FedProx proximal weight (paper: 1)")
    ap.add_argument("--scaffold-option", choices=["I", "II"], default="II", help="paper: Option II")
    ap.add_argument("--bn-stats", choices=["upload", "local"], default="upload",
                    help="local: BN running statistics not uploaded (BN prior off); FedBN is always local")
    ap.add_argument("--aux-round", type=int, default=None,
                    help="round of the auxiliary state in the --global folder (default: k of global_round{k}.pt)")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/pilot/attacks")
    ap.add_argument("--tag", default="")
    return ap.parse_args(argv)


# ------------------------------------------------------------ helpers (no global state)
def bn_targets(rs0, rs1, u):
    """Weighted-average batch statistics over the u steps, recovered from the running statistics."""
    c = (1 - MOM) ** u
    return {n: ((rs1[n][0] - c * rs0[n][0]) / (1 - c), (rs1[n][1] - c * rs0[n][1]) / (1 - c)) for n in rs0}


def cosine_dist(dw_sim, dw):
    num = sum((dw_sim[n] * dw[n]).sum() for n in dw)
    na = torch.sqrt(sum(dw_sim[n].pow(2).sum() for n in dw))
    nb = torch.sqrt(sum(dw[n].pow(2).sum() for n in dw))
    return 1 - num / (na * nb + 1e-12)


def drop_text_head(d):
    """The distance without the text part of the head (columns 512: of head.weight)."""
    d = dict(d)
    d["head.weight"] = d["head.weight"][:, :512]
    return d


def tv(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def ssim_matrix(rec, orig):
    r = rec.detach().cpu().numpy()[:, 0]; o = orig.cpu().numpy()[:, 0]
    return np.array([[structural_similarity(o[i], r[j], data_range=1.0, gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False) for j in range(len(r))] for i in range(len(o))])


def score(rec, orig):
    """One-to-one assignment maximizing mean SSIM (the paper's matching rule)."""
    S = ssim_matrix(rec, orig)
    ri, ci = linear_sum_assignment(-S)
    return float(S[ri, ci].mean()), S


def resolve_aux_round(glob_path, aux_round):
    if aux_round is not None:
        return aux_round
    m = re.search(r"global_round(\d+)\.pt$", os.path.basename(glob_path))
    return int(m.group(1)) if m else None


def text_cell_suffix(proto, mu, scaffold_option, bn_stats):
    """The file-name suffix attack_text.py (cell_suffix) gives a non-exact result of this cell: '' for
    FedAvg with default options, else _fedprox_mu{mu}, _scaffold_opt{I|II}, _fedbn, and _bnlocal."""
    parts = []
    if proto == "fedprox":
        parts.append(f"fedprox_mu{mu:g}")
    elif proto == "scaffold":
        parts.append(f"scaffold_opt{scaffold_option}")
    elif proto == "fedbn":
        parts.append("fedbn")
    if bn_stats == "local" and proto != "fedbn":
        parts.append("bnlocal")
    return "".join("_" + p for p in parts)


def text_result_path(spec, v, B, u, seed, proto, n_victims, suffix=""):
    """File-name convention of --text-from (see the module docstring)."""
    path = spec if os.path.isabs(spec) else os.path.join(ROOT, spec)
    if "{" in spec:
        return path.format(v=v, B=B, u=u, seed=seed, proto=proto, suffix=suffix)
    if os.path.isdir(path):
        return os.path.join(path, f"v{v}_B{B}_u{u}{suffix}.json")
    assert n_victims == 1, "--text-from names one file but several victims are attacked; give a directory or pattern"
    return path


def read_text_result(path, v, B, u, seed, proto, opts=None):
    """The fields of an attack_text.py result that the recovered-text condition uses. Fails on a missing
    field instead of inventing one; checks the identifiers that the file carries (opts: further cell
    options to compare, e.g. {"scaffold_option": "II", "bn_stats": "upload", "mu": 1.0})."""
    t = json.load(open(path))
    for k, want in (opts or {}).items():
        if k in t and t[k] is not None and want is not None:
            assert t[k] == want, f"{path}: {k} = {t[k]}, expected {want}"
    for k, want in (("victim", v), ("batch", B), ("u", u), ("seed", seed)):
        if k in t:
            assert int(t[k]) == want, f"{path}: {k} = {t[k]}, expected {want}"
    for k in ("proto", "protocol"):
        if k in t:
            assert str(t[k]) == proto, f"{path}: {k} = {t[k]}, expected {proto}"
    assert "rec_ids" in t, f"{path}: no 'rec_ids' (text result written before the full output format)"
    rec_ids = t["rec_ids"]
    labels = t.get("labels_sign")
    coefs = t.get("coef_ls")
    if rec_ids:
        assert labels is not None and len(labels) == len(rec_ids), f"{path}: 'labels_sign' missing or of wrong length"
        assert coefs is not None and len(coefs) == len(rec_ids), f"{path}: 'coef_ls' missing or of wrong length"
    if t.get("exact"):
        print(f"WARNING: {path} is the exact-update (float64) reference text result, not the main one", flush=True)
    return {"rec_ids": rec_ids, "labels": labels or [], "coefs": coefs or [], "exact": bool(t.get("exact", False)),
            "checked": [k for k in ("victim", "batch", "u", "seed", "proto", "protocol", *(opts or {})) if k in t]}


def clean_ids(t):
    """A recovered token sequence in the victim's format: no trailing EOS/pad, at most MAX_TOKENS tokens."""
    t = [int(i) for i in t]
    while t and t[-1] == EOS:
        t.pop()
    return t[:MAX_TOKENS]


def assign_texts(rec_ids, labels, coefs, y):
    """The paper's rules for the recovered texts (Sec. "Attacks"; revised 2026-11-22: the texts are taken
    exactly as the text attack returns them, at most B; the former "keep the B largest" rule is dropped):
      1. more than B sequences is an error (the text attack returns at most B);
      2. inferred label counts that do not fit the known counts -> the texts with the smallest |coefficient|
         of the over-represented label are reassigned to the other label;
      3. each text is paired with one dummy image (slot) of its label; slots of a label are filled in
         index order by decreasing |coefficient|; slots left over get no text (treated as under `none`).
    y: the known labels of the batch (list of 0/1); the slots are the dummy images 0..B-1 with these labels.
    Returns (slot_ids: list of B token lists or None, info). Empty sequences are dropped first."""
    B = len(y)
    items = [(clean_ids(t), int(lab), float(c), k) for k, (t, lab, c) in enumerate(zip(rec_ids, labels, coefs))]
    n_returned = len(items)
    items = [it for it in items if len(it[0]) > 0]
    n_empty = n_returned - len(items)
    if n_returned > B:
        raise ValueError(f"the text result holds {n_returned} sequences for a batch of {B}; the text attack "
                         f"returns at most B and no selection rule is applied here")
    labs = [it[1] for it in items]
    known = {l: sum(1 for t in y if int(t) == l) for l in (0, 1)}
    reassigned = []
    for l in (0, 1):
        excess = sum(1 for t in labs if t == l) - known[l]
        if excess > 0:
            cand = sorted([i for i in range(len(items)) if labs[i] == l], key=lambda i: (abs(items[i][2]), i))
            for i in cand[:excess]:
                labs[i] = 1 - l
                reassigned.append(items[i][3])
    slot_ids, slot_src = [None] * B, [None] * B
    for l in (0, 1):
        slots = [i for i in range(B) if int(y[i]) == l]
        texts = sorted([i for i in range(len(items)) if labs[i] == l], key=lambda i: (-abs(items[i][2]), i))
        assert len(texts) <= len(slots)
        for s, i in zip(slots, texts):
            slot_ids[s], slot_src[s] = items[i][0], items[i][3]
    info = {"n_returned": n_returned, "n_empty": n_empty, "reassigned": reassigned,
            "labels_used": {str(items[i][3]): labs[i] for i in range(len(items))},
            "slot_source": slot_src, "n_slots_with_text": sum(s is not None for s in slot_ids)}
    return slot_ids, info


def pad_ids(tok_lists, device):
    ids = torch.full((len(tok_lists), MAX_TOKENS), EOS, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, t in enumerate(tok_lists):
        ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        mask[i, :len(t)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def interpolated_text_features(model, ids, mask, dtxt, u):
    """Per-step text features of the recovered texts: step j (1-based) uses the GPT-2 weights
    w_txt^t + (j - 1)/u * dw_txt (linear interpolation between broadcast and uploaded weights; paper Sec.
    "Attacks"). Step 1 is the broadcast model itself (exactly what the victim's first step sees)."""
    base = {n: p.detach() for n, p in model.txt.named_parameters()}
    last = mask.sum(1) - 1
    out = []
    for j in range(u):
        if j == 0:
            h = model.text_features(ids, mask)
        else:
            params = {n: base[n] + (j / u) * dtxt["txt." + n] for n in base}
            hs = functional_call(model.txt, params, (), {"input_ids": ids, "attention_mask": mask}).last_hidden_state
            h = hs[torch.arange(hs.shape[0], device=hs.device), last]
        out.append(h.detach())
    return out


class SplitTextView(nn.Module):
    """The fusion model with the text contribution to the logit split off, for pairs without text.
    Shares every parameter and buffer with `fm` under the same names, so protocols.simulate_image_steps
    can drive it with functional_call. h_txt = (h, s): h (n, 768) text features, set to the feature mean
    txt_mu for pairs without text (their standardized text input is then exactly 0, so they add nothing
    through the text half of the head); s (n,) the unknown scalar of each pair at this step (0 for pairs
    with text). logit = head([z_img, z_txt]) + s. With record=[], each forward also appends the text
    contribution w_txt . z_txt of every pair (used only for the check with the true scalars)."""

    def __init__(self, fm):
        super().__init__()
        self.img, self.txt, self.head = fm.img, fm.txt, fm.head
        for k in ("im_mean", "im_std", "img_mu", "img_sd", "txt_mu", "txt_sd"):
            self.register_buffer(k, getattr(fm, k), persistent=False)
        self.record = None

    def forward(self, x, ids, mask, h_txt):
        h, s = h_txt
        h_img = self.img((x.repeat(1, 3, 1, 1) - self.im_mean) / self.im_std)
        z_txt = (h - self.txt_mu) / self.txt_sd
        z = torch.cat([(h_img - self.img_mu) / self.img_sd, z_txt], 1)
        if self.record is not None:
            self.record.append((z_txt @ self.head.weight[0, 512:]).detach())
        return self.head(z).squeeze(1) + s


def init_hash(xd):
    return hashlib.sha1(xd.detach().cpu().numpy().tobytes()).hexdigest()[:16]


# ------------------------------------------------------------ main
def main(argv=None):
    a = get_args(argv)
    dev = a.device
    sync = (lambda: torch.cuda.synchronize()) if dev.startswith("cuda") else (lambda: None)
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    if a.text == "recovered":
        assert a.text_from, "--text recovered needs --text-from"

    data = PairData(a.res)
    split = load_seed(a.seed)
    model = FusionModel().to(dev)
    if a.glob == "random":        # control: randomly initialized ResNet-18
        import torchvision
        r = torchvision.models.resnet18(weights=None); r.fc = torch.nn.Identity()
        model.img = r.to(dev)
    elif a.glob != "pretrained":  # "pretrained" = round 0
        model.load_state_dict(torch.load(os.path.join(ROOT, a.glob), map_location=dev))
    model.train()
    bn_w = a.bn
    if a.bn_mode == "eval":
        model.img.eval()
        bn_w = 0.0

    img_names = [n for n, _ in model.named_parameters() if is_image_param(n)]
    bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]

    # protocol and the auxiliary state of the warm-up run (folder of --global)
    proto = P.Protocol(a.proto, mu=a.mu, option=a.scaffold_option, bn_stats=a.bn_stats)
    aux_round = resolve_aux_round(a.glob, a.aux_round)
    run_dir = os.path.dirname(os.path.join(ROOT, a.glob))
    needs_aux = a.proto in ("scaffold", "fedbn") or a.bn_stats == "local"
    if needs_aux and a.glob in ("random", "pretrained"):
        print(f"WARNING: --proto {a.proto} with --global {a.glob}: no auxiliary state; SCAFFOLD fails, the "
              f"victim keeps the broadcast BN entries", flush=True)
    if not proto.bn_stats_uploaded:
        bn_w = 0.0                # no uploaded statistics -> no BN prior (paper Sec. "Attacks")
    use_bn = bn_w > 0
    view = SplitTextView(model)
    bn_stats_eff = "upload" if proto.bn_stats_uploaded else "local"
    tsuffix = text_cell_suffix(a.proto, a.mu, a.scaffold_option, a.bn_stats)
    topts = {"bn_stats": bn_stats_eff, "scaffold_option": a.scaffold_option if a.proto == "scaffold" else None,
             "mu": proto.mu if a.proto == "fedprox" else None}
    lr_scalar = a.opt_lr if a.lr_scalar is None else a.lr_scalar

    # guessing baseline: mean image of the seed's client (training) data, as defined in the paper
    all_rows = torch.cat([data.rows(v) for v in split["clients"].values()])
    mean_img = data.x[all_rows].mean(0, keepdim=True)
    # initialization prior: mean image of pairs that no client holds (validation + test split)
    pub_img = data.x[torch.cat([data.split("val"), data.split("test")])].mean(0, keepdim=True)

    # ------------------------------------------------------------ attack (server side)
    cur = {"proto": proto}          # the protocol state of the victim being attacked (set in the loop)

    def simulate(xd, y, cond, collect_bn, S=None):
        """Unrolled u steps on dummy images under the protocol; returns the simulated image-side delta
        and BN batch stats. cond["h"]: per-step text inputs; cond["miss"]: None (every pair has a text:
        the plain fusion model) or a 0/1 float mask of the pairs without text (SplitTextView, S (n, u))."""
        if cond["miss"] is None:
            return P.simulate_image_steps(model, img_names, bn_mods, xd, y, cond["h"], a.lr_img, cur["proto"], collect_bn)
        hs = [(cond["h"][j], S[:, j] * cond["miss"]) for j in range(a.u)]
        return P.simulate_image_steps(view, img_names, bn_mods, xd, y, hs, a.lr_img, cur["proto"], collect_bn)

    def attack(x, y, cond, dw, tgt, m0, v, seed):
        """One optimization run; returns the dummy images, the final matching loss, the curve, the
        minutes, the scalars (or None) and a hash of the initial x'."""
        torch.manual_seed(seed)
        n = len(y)
        if a.init == "mean":
            xd = (pub_img.repeat(n, 1, 1, 1) + 0.05 * torch.randn(n, 1, a.res, a.res)).clamp(0, 1).to(dev)
        elif a.init == "near":
            xd = (x.cpu() + 0.05 * torch.randn(n, 1, a.res, a.res)).clamp(0, 1).to(dev)
        else:
            xd = torch.rand(n, 1, a.res, a.res, device=dev)
        h0 = init_hash(xd)
        xd.requires_grad_(True)
        dist = cond["dist"]
        if cond["miss"] is None:
            S = None
            opt = torch.optim.Adam([xd], lr=a.opt_lr)
        else:                     # unknown text contributions start at 0 (the text adds nothing)
            S = torch.zeros(n, a.u, device=dev, requires_grad=True)
            opt = torch.optim.Adam([{"params": [xd]}, {"params": [S], "lr": lr_scalar}], lr=a.opt_lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(a.iters * f) for f in (0.375, 0.625, 0.875)], gamma=0.1)
        sync(); t0 = time.time()
        curve = []
        for it in range(1, a.iters + 1):
            opt.zero_grad()
            dw_sim, bn_avg = simulate(xd, y, cond, use_bn, S)
            cos = cosine_dist(dist(dw_sim), dw)
            loss = a.cos_w * cos + a.tv * tv(xd)
            if use_bn:
                loss = loss + bn_w * sum((bn_avg[k][0] - tgt[k][0]).norm() + (bn_avg[k][1] - tgt[k][1]).norm() for k in tgt)
            loss.backward()
            if a.signed:
                xd.grad.sign_()
            opt.step(); sched.step()
            with torch.no_grad():
                xd.clamp_(0, 1)
            if it in a.checkpoints:
                sync()
                s, _ = score(xd, x)
                curve.append({"iter": it, "ssim": round(s, 4), "cos": round(cos.item(), 5), "loss": round(loss.item(), 5),
                              "minutes": round((time.time() - t0) / 60, 2)})
                print(f"victim {v} B={n} u={a.u} it={it} ssim={s:.4f} (m0={m0:.4f}) cos={cos.item():.4f} "
                      f"loss={loss.item():.4f} t={curve[-1]['minutes']}min", flush=True)
        final = cosine_dist(dist(simulate(xd, y, cond, False, S)[0]), dw).item()
        return xd.detach(), final, curve, (time.time() - t0) / 60, (None if S is None else S.detach()), h0

    results = []
    for v in a.victims:
        studies = split["victim_batches"][str(v)][: a.batch]
        rows = data.rows(studies)
        x, ids, mask, y = data.batch(rows, dev)
        B = len(rows)

        # the victim's round (client side) under the protocol
        aux = load_aux_state(run_dir, aux_round, v, dev) if needs_aux and a.glob not in ("random", "pretrained") else \
            {"c": None, "c_k": None, "local": None}
        pr = copy.copy(proto)      # this victim's protocol state (SCAFFOLD: its c - c_k); used by simulate
        cur["proto"] = pr
        if a.proto == "scaffold":
            assert aux["c"] is not None, f"scaffold: no control variates for round {aux_round} in {run_dir}"
            pr.set_controls(aux["c"], aux["c_k"])
        bn_local = aux["local"] if P.local_state_keys(model, pr) else None
        torch.manual_seed(1000 + v)
        up, client = P.client_upload(model, x, ids, mask, y, a.u, a.lr_img, a.lr_txt, pr, bn_local=bn_local,
                                     bn_eval=a.bn_mode == "eval")
        if a.proto == "scaffold":  # the replay needs only c - c_k of the simulated (image) parameters
            pr.corr = {n: pr.corr[n] for n in img_names}
            pr.c = pr.c_k = None
            del aux
        dw = {n: up["delta"][n] for n in img_names if n in up["delta"]}
        h_oracle = client["h_txt"]
        tgt = bn_targets(up["bn_before"], up["bn_after"], a.u) if up["bn_before"] is not None else None
        m0 = score(mean_img.repeat(len(rows), 1, 1, 1), x)[0]

        # the attacker's text inputs for this condition
        text_info, checks = None, {}
        y_list = [int(t) for t in y.tolist()]
        if a.text == "oracle":
            cond = {"h": h_oracle, "miss": None, "dist": lambda d: d}
        else:
            if a.text == "recovered":
                tpath = text_result_path(a.text_from, v, B, a.u, a.seed, a.proto, len(a.victims), suffix=tsuffix)
                tr = read_text_result(tpath, v, B, a.u, a.seed, a.proto, opts=topts)
                slot_ids, text_info = assign_texts(tr["rec_ids"], tr["labels"], tr["coefs"], y_list)
                text_info.update({"file": os.path.relpath(tpath, ROOT).replace("\\", "/"), "fields_checked": tr["checked"],
                                  "text_exact": tr["exact"]})
            else:
                slot_ids = [None] * B
            have = [i for i in range(B) if slot_ids[i] is not None]
            h_steps = [model.txt_mu.expand(B, -1).clone() for _ in range(a.u)]
            if have:
                tids, tmask = pad_ids([slot_ids[i] for i in have], dev)
                dtxt = {n: d for n, d in up["delta"].items() if n.startswith("txt.")}
                h_rec = interpolated_text_features(model, tids, tmask, dtxt, a.u)
                for j in range(a.u):
                    h_steps[j][have] = h_rec[j]
                # check (client side): distance of each recovered feature to the nearest true feature
                err = [max(min((h_rec[j][q] - h_oracle[j][k]).abs().max().item() for k in range(B))
                           for q in range(len(have))) for j in range(a.u)]
                checks["text_feat_err"] = err
                checks["text_feat_relerr"] = [e / h_oracle[j].abs().max().item() for j, e in enumerate(err)]
                # the same in the head's input space (standardized features), relative L2 error
                zs = lambda h: (h - model.txt_mu) / model.txt_sd
                checks["text_feat_zerr"] = [max(min(((zs(h_rec[j][q]) - zs(h_oracle[j][k])).norm()
                                                     / zs(h_oracle[j][k]).norm()).item() for k in range(B))
                                                for q in range(len(have))) for j in range(a.u)]
            if len(have) == B:     # every pair has a text: the plain fusion model, full distance
                cond = {"h": h_steps, "miss": None, "dist": lambda d: d}
            else:
                miss = torch.ones(B, device=dev)
                miss[have] = 0.0
                cond = {"h": h_steps, "miss": miss, "dist": drop_text_head}

        # exactness check: at the true images the simulation must reproduce the observed update (FedBN:
        # off by the BN-parameter mismatch; recovered: off by the text-feature error)
        dw_true, bn_true = simulate(x, y, {"h": h_oracle, "miss": None}, collect_bn=tgt is not None)
        d_true = cosine_dist(dw_true, dw).item()
        checks["cos_true_oracle"] = d_true
        bn_err = None
        if tgt is not None:
            bn_err = max(((bn_true[n][0] - tgt[n][0]).abs().max().item(), (bn_true[n][1] - tgt[n][1]).abs().max().item())
                         for n in tgt)
            checks["bn_err_mean"], checks["bn_err_var"] = bn_err
        del dw_true, bn_true
        if a.text != "oracle":
            S_true = None
            if cond["miss"] is not None:
                # true scalars (check only, client side): the text contribution w_txt^(j) . z_txt^(j) of each
                # pair along the replay at the true images with the true features (view with s = 0)
                view.record = []
                simulate(x, y, {"h": h_oracle, "miss": torch.zeros(B, device=dev)}, False, torch.zeros(B, a.u, device=dev))
                S_true = torch.stack(view.record, 1)
                view.record = None
            dws, _ = simulate(x, y, cond, False, S_true)
            checks["cos_true"] = cosine_dist(cond["dist"](dws), cond["dist"](dw)).item()
            del dws
        else:
            checks["cos_true"] = d_true
        dw_att = cond["dist"](dw)
        checks["dist_tensors"] = len(dw_att)
        checks["dist_numel"] = int(sum(t.numel() for t in dw_att.values()))
        print(f"victim {v}: check at true images: cosine distance {d_true:.2e}"
              + (f", max BN target error mean {bn_err[0]:.2e} var {bn_err[1]:.2e}" if bn_err else ", no BN statistics")
              + ("" if a.text == "oracle" else f"; with the {a.text}-text inputs {checks['cos_true']:.2e}"),
              flush=True)

        # restarts are chosen by the final matching loss only (never by SSIM, which needs the truth)
        runs = [attack(x, y, cond, dw_att, tgt, m0, v, 1000 + v + 100 * r) for r in range(a.restarts)]
        xd, cos_final, curve, minutes, S_fin, h0 = min(runs, key=lambda t: t[1])
        s, S = score(xd, x)
        rec = {"victim": v, "batch": len(rows), "u": a.u, "init": a.init, "text": a.text, "res": a.res,
               "signed": a.signed, "cos_w": a.cos_w, "bn_mode": a.bn_mode, "global": a.glob, "opt_lr": a.opt_lr, "tv": a.tv, "bn": bn_w, "iters": a.iters, "restarts": a.restarts,
               "ssim": round(s, 4), "m0": round(m0, 4), "cos": round(cos_final, 5), "cos_all": [round(t[1], 5) for t in runs],
               "minutes": round(sum(t[3] for t in runs), 2), "curve": curve,
               # cell identifiers (score_images.py reads "seed", "protocol", "condition", "global")
               "seed": a.seed, "proto": a.proto, "protocol": a.proto, "condition": a.text,
               "mu": a.mu if a.proto == "fedprox" else None,
               "scaffold_option": a.scaffold_option if a.proto == "scaffold" else None,
               "bn_stats": "upload" if proto.bn_stats_uploaded else "local", "bn_requested": a.bn,
               "bn_prior": use_bn, "aux_round": aux_round if needs_aux else None,
               "global_path": os.path.relpath(os.path.join(ROOT, a.glob), ROOT).replace("\\", "/")
               if a.glob not in ("random", "pretrained") else a.glob,
               "lr_img": a.lr_img, "lr_txt": a.lr_txt, "lr_scalar": lr_scalar if cond["miss"] is not None else None,
               "dist_drops_text_head": cond["miss"] is not None, "init_sha1": h0, "checks": checks,
               "text_assign": text_info,
               "scalars": None if S_fin is None else [[round(t, 5) for t in row] for row in S_fin.cpu().tolist()],
               "labels": y_list, "device": dev}
        results.append(rec)
        print(f"victim {v} FINAL ssim={s:.4f} (m0={m0:.4f}) cos={cos_final:.4f}{a.tag}", flush=True)
        name = f"v{v}_B{len(rows)}_u{a.u}_{a.init}{a.tag}"
        jpath = os.path.join(out, name + ".json")
        if os.path.exists(jpath):  # never overwrite the result of another cell
            old = json.load(open(jpath))
            for k in ("seed", "protocol", "condition"):
                assert k not in old or old[k] == rec[k], \
                    f"{jpath} holds {k}={old[k]} (this run: {rec[k]}); use another --out or --tag per cell"
        torch.save({"rec": xd.cpu(), "orig": x.cpu(), "labels": y.cpu(), "studies": studies}, os.path.join(out, name + ".pt"))
        json.dump(rec, open(jpath, "w"), indent=1)
    print("SUMMARY" + a.tag, json.dumps([{k: r[k] for k in ("victim", "batch", "u", "init", "ssim", "m0", "cos", "minutes")} for r in results]))
    return results


def load_aux_state(run_dir, r, client, dev):
    """train_fl.load_aux, imported lazily (train_fl parses nothing on import, but keeps this module light)."""
    import train_fl
    assert r is not None, "cannot infer --aux-round from the --global file name; pass --aux-round"
    return train_fl.load_aux(run_dir, r, client, device=dev)


if __name__ == "__main__":
    main()
