"""Diagnostic for the label step of the text attack (CPU only, no DAGER run).

Separates the three possible causes of poor label accuracy:
  (1) noise in the float32 head update, (2) wrong feature vectors of partly recovered texts,
  (3) texts missing from the design matrix,
and compares label rules on the same update. Partial recovery is simulated the way it shows up under
float32: the texts the model already fits well (smallest |p - y|) are cut to a prefix or dropped.

The server-side quantities used by the rules: the head update (text weights and bias), the broadcast
model, the (simulated) recovered token ids, B, u and the head learning rate. p_true (full model with the
image) is used only for the identity check, never by a rule.

usage (CPU; does not touch the GPU):
  python src/calibration/label_diag.py --global runs/pilot/seed0_fedavg/global_round9.pt --seed 0 --victims 0 1 2 \
      --batch 8 --u 1 --partial 3 --keep 0.5 --drop 2
"""
import argparse
import itertools
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch is imported
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ (common.py)
from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--victims", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--u", type=int, default=1)
ap.add_argument("--lr-img", type=float, default=0.01)
ap.add_argument("--lr-txt", type=float, default=1e-4)
ap.add_argument("--partial", type=int, default=3, help="the k best-fit texts are cut to a prefix")
ap.add_argument("--keep", type=float, default=0.5, help="prefix length as a fraction of the tokens")
ap.add_argument("--drop", type=int, default=2, help="the d best-fit texts are missing (scenario 'drop')")
ap.add_argument("--ridge", type=float, default=1e-2, help="ridge strength relative to mean diag of Z Z^T")
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--out", default="")
a = ap.parse_args()
torch.manual_seed(0)
dev = "cpu"

data = PairData(a.res)
split = load_seed(a.seed)
model = FusionModel().to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, a.glob), map_location=dev))
model.train()


def lr_of(n):
    return a.lr_img if is_image_param(n) else a.lr_txt


def victim_update(x, ids, mask, y):
    """Same as attack_text.victim_update; also returns the step-0 logits (for the identity check)."""
    m = FusionModel().to(dev)
    m.load_state_dict(model.state_dict())
    m.train()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    acc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in m.named_parameters()}
    logit0 = None
    for t in range(a.u):
        logit = m(x, ids, mask)
        if t == 0:
            logit0 = logit.detach()
        loss = bce(logit, y)
        grads = torch.autograd.grad(loss, list(m.parameters()))
        with torch.no_grad():
            for (n, p), g in zip(m.named_parameters(), grads):
                p -= lr_of(n) * g
                acc[n] -= lr_of(n) * g.double()
    f32 = {n: (p.detach() - w0[n]) for n, p in m.named_parameters()}
    return f32, {n: v for n, v in acc.items()}, logit0


@torch.no_grad()
def feats(seqs):
    """Standardized last-token text features and the text-only prediction p_hat of the broadcast model
    (image part at its training mean, i.e. 0 after standardization)."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), 50256, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
    z = (model.text_features(ids, mask) - model.txt_mu) / model.txt_sd
    w = model.head.weight[0, 512:]
    p_hat = torch.sigmoid(z @ w + model.head.bias[0])
    return z.double().numpy(), p_hat.double().numpy()


def rules(Z, p_hat, g, db, B, n_all):
    """Label rules. Z (n, 768) features of the returned texts, g (768,) text-weight update, db bias update."""
    s = a.lr_img * a.u / B            # |coef| <= s, since coef_i = lr*u*(y_i - p_i)/B with p_i in (0, 1)
    out = {}
    c_ls = np.linalg.lstsq(Z.T, g, rcond=None)[0]                    # rank-revealing (SVD), float64
    out["sign_ls"] = (c_ls > 0).astype(int)
    G = Z @ Z.T
    lam = a.ridge * np.trace(G) / len(G)
    c_r = np.linalg.solve(G + lam * np.eye(len(G)), Z @ g)
    out["sign_ridge"] = (c_r > 0).astype(int)
    c_box = lsq_linear(Z.T, g, bounds=(-s, s)).x
    out["sign_box"] = (c_box > 0).astype(int)
    out["model_only"] = (p_hat > 0.5).astype(int)
    out["combined_ls"] = (p_hat + c_ls / s > 0.5).astype(int)        # y = p + B*coef/(lr*u)
    out["combined_box"] = (p_hat + c_box / s > 0.5).astype(int)
    best, best_r = None, np.inf                                       # 2^n binary fit with p = p_hat
    for yy in itertools.product((0, 1), repeat=len(p_hat)):
        r = np.linalg.norm(g - Z.T @ (s * (np.array(yy) - p_hat)))
        if r < best_r:
            best, best_r = np.array(yy), r
    out["binary_enum"] = best
    if len(p_hat) == n_all:                                           # number of positives from the bias
        K = int(np.clip(round(db / s + p_hat.sum()), 0, n_all))
        score = p_hat + c_box / s
        yk = np.zeros(n_all, int); yk[np.argsort(-score)[:K]] = 1
        out["bias_count_box"] = yk
    stats = {"r2_ls": float(1 - np.sum((g - Z.T @ c_ls) ** 2) / np.sum(g ** 2)),
             "cond_Z": float(np.linalg.cond(Z)),
             "max_cos": float(np.max(np.abs(np.triu((Z / np.linalg.norm(Z, axis=1, keepdims=True)) @
                                                     (Z / np.linalg.norm(Z, axis=1, keepdims=True)).T, 1)))),
             "coef_ls_scaled": np.round(c_ls / s, 3).tolist(), "sum_coef_vs_db": [float(c_ls.sum()), float(db)]}
    return out, stats


results = []
for v in a.victims:
    rows = data.rows(split["victim_batches"][str(v)][: a.batch])
    x, ids, mask, y = data.batch(rows, dev)
    B = len(rows)
    torch.manual_seed(1000 + v)
    f32, exact, logit0 = victim_update(x, ids, mask, y)
    yn = y.numpy().astype(int)
    p_true = torch.sigmoid(logit0).double().numpy()
    seqs = [ids[i][: int(mask[i].sum())].tolist() for i in range(B)]
    Z_true, p_hat_true = feats(seqs)
    rec = {"victim": v, "B": B, "u": a.u, "y": yn.tolist(), "p_true": np.round(p_true, 3).tolist(),
           "p_hat_text_only": np.round(p_hat_true, 3).tolist(), "always_abnormal_acc": float(yn.mean())}
    # identity check (exact for u = 1): head update = lr*u/B * sum (y - p) [z; 1]
    pred = a.lr_img * a.u / B * (yn - p_true) @ Z_true
    for nm, dw in (("f32", f32), ("exact", exact)):
        g = dw["head.weight"][0, 512:].double().numpy()
        rec[f"identity_relerr_{nm}"] = float(np.linalg.norm(g - pred) / np.linalg.norm(g))
    g = f32["head.weight"][0, 512:].double().numpy()
    db = float(f32["head.bias"][0])
    order = np.argsort(np.abs(p_true - yn))                           # best-fit first
    part = set(order[: a.partial].tolist())
    seqs_part = [s[: max(1, int(round(a.keep * len(s))))] if i in part else s for i, s in enumerate(seqs)]
    keep_idx = sorted(set(range(B)) - set(order[: a.drop].tolist()))
    scen = {"true_all": (seqs, list(range(B))), "partial": (seqs_part, list(range(B))),
            "drop": ([seqs[i] for i in keep_idx], keep_idx),
            "partial_drop": ([seqs_part[i] for i in keep_idx], keep_idx)}
    rec["scenarios"] = {}
    for name, (ss, idx) in scen.items():
        Z, p_hat = feats(ss)
        out, stats = rules(Z, p_hat, g, db, B, B)
        acc = {k: float(np.mean(lab == yn[idx])) for k, lab in out.items()}
        acc["always_abnormal_same_subset"] = float(yn[idx].mean())  # the majority (prior) label is 1
        rec["scenarios"][name] = {"n": len(idx), "acc": acc, **stats}
        print(f"victim {v} {name:13s} n={len(idx)} " + " ".join(f"{k}={val:.2f}" for k, val in acc.items())
              + f" | R2={stats['r2_ls']:.3f} cond={stats['cond_Z']:.1f} maxcos={stats['max_cos']:.3f}", flush=True)
    print(f"victim {v} identity rel. err f32 {rec['identity_relerr_f32']:.2e} exact {rec['identity_relerr_exact']:.2e}"
          f" | y {rec['y']} p_true {rec['p_true']} p_hat {rec['p_hat_text_only']}", flush=True)
    results.append(rec)

names = list(results[0]["scenarios"]["true_all"]["acc"])
for sc in results[0]["scenarios"]:
    print(f"MEAN {sc:13s} " + " ".join(f"{k}={np.mean([r['scenarios'][sc]['acc'].get(k, np.nan) for r in results]):.3f}"
                                      for k in names))
if a.out:
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    json.dump(results, open(os.path.join(ROOT, a.out, f"label_diag_u{a.u}.json"), "w"), indent=1)
