"""Image attack: simulation-based update matching on the ResNet-18 + head part of the update.

The victim's u local SGD steps on its batch are replayed on dummy images x' (BN in training mode,
known labels, given text features), and x' is optimized so that the simulated update matches the
observed one in cosine distance, with TV, clipping (x' in [0, 1]) and BN priors.
At u = 1 this is the gradient matching of Geiping et al.

The victim update itself is produced here from the saved global model (round t), so the pilot can
vary u and the batch size without re-running the federation.

usage: python src/attack_image.py --global runs/pilot/seed0_fedavg/global_round9.pt --seed 0
          --victims 0 1 2 --batch 8 --u 1 --iters 10000 --text oracle --out runs/pilot/attacks
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity
from torch.func import functional_call

from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed

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
ap.add_argument("--signed", action="store_true", help="use the sign of the gradient (Geiping et al.)")
ap.add_argument("--restarts", type=int, default=1, help="keep the restart with the lowest final matching loss")
ap.add_argument("--text", choices=["oracle"], default="oracle")
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--out", default="runs/pilot/attacks")
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = "cuda"
out = os.path.join(ROOT, a.out)
os.makedirs(out, exist_ok=True)

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
if a.bn_mode == "eval":
    model.img.eval()
    a.bn = 0.0

img_names = [n for n, _ in model.named_parameters() if is_image_param(n)]
bn_mods = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
MOM = 0.1


def lr_of(name):
    return a.lr_img if is_image_param(name) else a.lr_txt


# ------------------------------------------------------------ victim update (client side)
def victim_update(x, ids, mask, y):
    """u local SGD steps of the full model on one batch; returns the image-side delta, the BN
    running statistics before/after, and the per-step text features (for the oracle condition)."""
    m = FusionModel().to(dev)
    m.load_state_dict(model.state_dict())
    m.train()
    if a.bn_mode == "eval":
        m.img.eval()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    rs0 = {n: (mod.running_mean.clone(), mod.running_var.clone()) for n, mod in m.named_modules()
           if isinstance(mod, torch.nn.BatchNorm2d)}
    h_steps = []
    for _ in range(a.u):
        with torch.no_grad():
            h_steps.append(m.text_features(ids, mask).detach())
        loss = bce(m(x, ids, mask), y)
        names = [n for n, _ in m.named_parameters()]
        grads = torch.autograd.grad(loss, list(m.parameters()))
        with torch.no_grad():
            for n, p, g in zip(names, m.parameters(), grads):
                p -= lr_of(n) * g
    dw = {n: (p.detach() - w0[n]) for n, p in m.named_parameters() if is_image_param(n)}
    rs1 = {n: (mod.running_mean.clone(), mod.running_var.clone()) for n, mod in m.named_modules()
           if isinstance(mod, torch.nn.BatchNorm2d)}
    return dw, rs0, rs1, h_steps


def bn_targets(rs0, rs1):
    """Weighted-average batch statistics over the u steps, recovered from the running statistics."""
    c = (1 - MOM) ** a.u
    return {n: ((rs1[n][0] - c * rs0[n][0]) / (1 - c), (rs1[n][1] - c * rs0[n][1]) / (1 - c)) for n in rs0}


# ------------------------------------------------------------ attack (server side)
def simulate(xd, y, h_steps, collect_bn):
    """Unrolled u steps on dummy images; returns the simulated image-side delta and BN batch stats."""
    params = {n: p.detach() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    p_img = {n: params[n].clone().requires_grad_(True) for n in img_names}
    start = p_img
    stats = {n: [] for n, _ in bn_mods}
    weights = [MOM * (1 - MOM) ** (a.u - j) for j in range(1, a.u + 1)]
    for j in range(a.u):
        hooks = []
        if collect_bn:
            for n, mod in bn_mods:
                def hk(module, inp, outp, n=n):
                    t = inp[0]
                    stats[n].append((t.mean((0, 2, 3)), t.var((0, 2, 3), unbiased=True)))
                hooks.append(mod.register_forward_hook(hk))
        full = dict(params); full.update(p_img)
        logit = functional_call(model, (full, buffers), (xd, None, None), {"h_txt": h_steps[j]})
        for h in hooks:
            h.remove()
        loss = bce(logit, y)
        g = torch.autograd.grad(loss, list(p_img.values()), create_graph=True)
        p_img = {n: v - a.lr_img * gi for (n, v), gi in zip(p_img.items(), g)}
    dw_sim = {n: p_img[n] - start[n] for n in img_names}
    bn_avg = {}
    if collect_bn:
        tot = sum(weights)
        for n in stats:
            bn_avg[n] = (sum(w * s[0] for w, s in zip(weights, stats[n])) / tot,
                         sum(w * s[1] for w, s in zip(weights, stats[n])) / tot)
    return dw_sim, bn_avg


def cosine_dist(dw_sim, dw):
    num = sum((dw_sim[n] * dw[n]).sum() for n in dw)
    na = torch.sqrt(sum(dw_sim[n].pow(2).sum() for n in dw))
    nb = torch.sqrt(sum(dw[n].pow(2).sum() for n in dw))
    return 1 - num / (na * nb + 1e-12)


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


# guessing baseline: mean image of the seed's client (training) data, as defined in the paper
all_rows = torch.cat([data.rows(v) for v in split["clients"].values()])
mean_img = data.x[all_rows].mean(0, keepdim=True)
# initialization prior: mean image of pairs that no client holds (validation + test split)
pub_img = data.x[torch.cat([data.split("val"), data.split("test")])].mean(0, keepdim=True)


def attack(x, y, h_steps, dw, tgt, m0, v, seed):
    """One optimization run; returns the dummy images, the final matching loss and the curve."""
    torch.manual_seed(seed)
    n = len(y)
    if a.init == "mean":
        xd = (pub_img.repeat(n, 1, 1, 1) + 0.05 * torch.randn(n, 1, a.res, a.res)).clamp(0, 1).to(dev)
    elif a.init == "near":
        xd = (x.cpu() + 0.05 * torch.randn(n, 1, a.res, a.res)).clamp(0, 1).to(dev)
    else:
        xd = torch.rand(n, 1, a.res, a.res, device=dev)
    xd.requires_grad_(True)
    opt = torch.optim.Adam([xd], lr=a.opt_lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(a.iters * f) for f in (0.375, 0.625, 0.875)], gamma=0.1)
    torch.cuda.synchronize(); t0 = time.time()
    curve = []
    for it in range(1, a.iters + 1):
        opt.zero_grad()
        dw_sim, bn_avg = simulate(xd, y, h_steps, collect_bn=a.bn > 0)
        cos = cosine_dist(dw_sim, dw)
        loss = a.cos_w * cos + a.tv * tv(xd)
        if a.bn > 0:
            loss = loss + a.bn * sum((bn_avg[k][0] - tgt[k][0]).norm() + (bn_avg[k][1] - tgt[k][1]).norm() for k in tgt)
        loss.backward()
        if a.signed:
            xd.grad.sign_()
        opt.step(); sched.step()
        with torch.no_grad():
            xd.clamp_(0, 1)
        if it in a.checkpoints:
            torch.cuda.synchronize()
            s, _ = score(xd, x)
            curve.append({"iter": it, "ssim": round(s, 4), "cos": round(cos.item(), 5), "loss": round(loss.item(), 5),
                          "minutes": round((time.time() - t0) / 60, 2)})
            print(f"victim {v} B={n} u={a.u} it={it} ssim={s:.4f} (m0={m0:.4f}) cos={cos.item():.4f} "
                  f"loss={loss.item():.4f} t={curve[-1]['minutes']}min", flush=True)
    final = cosine_dist(simulate(xd, y, h_steps, collect_bn=False)[0], dw).item()
    return xd.detach(), final, curve, (time.time() - t0) / 60

results = []
for v in a.victims:
    studies = split["victim_batches"][str(v)][: a.batch]
    rows = data.rows(studies)
    x, ids, mask, y = data.batch(rows, dev)
    torch.manual_seed(1000 + v)
    dw, rs0, rs1, h_steps = victim_update(x, ids, mask, y)
    tgt = bn_targets(rs0, rs1)
    m0 = score(mean_img.repeat(len(rows), 1, 1, 1), x)[0]
    # exactness check: at the true images the simulation must reproduce the observed update
    dw_true, bn_true = simulate(x, y, h_steps, collect_bn=True)
    d_true = cosine_dist(dw_true, dw).item()
    bn_err = max(((bn_true[n][0] - tgt[n][0]).abs().max().item(), (bn_true[n][1] - tgt[n][1]).abs().max().item())
                 for n in tgt)
    print(f"victim {v}: check at true images: cosine distance {d_true:.2e}, max BN target error "
          f"mean {bn_err[0]:.2e} var {bn_err[1]:.2e}", flush=True)
    del dw_true, bn_true

    # restarts are chosen by the final matching loss only (never by SSIM, which needs the truth)
    runs = [attack(x, y, h_steps, dw, tgt, m0, v, 1000 + v + 100 * r) for r in range(a.restarts)]
    xd, cos_final, curve, minutes = min(runs, key=lambda t: t[1])
    s, S = score(xd, x)
    rec = {"victim": v, "batch": len(rows), "u": a.u, "init": a.init, "text": a.text, "res": a.res,
           "signed": a.signed, "cos_w": a.cos_w, "bn_mode": a.bn_mode, "global": a.glob, "opt_lr": a.opt_lr, "tv": a.tv, "bn": a.bn, "iters": a.iters, "restarts": a.restarts,
           "ssim": round(s, 4), "m0": round(m0, 4), "cos": round(cos_final, 5), "cos_all": [round(t[1], 5) for t in runs],
           "minutes": round(sum(t[3] for t in runs), 2), "curve": curve}
    results.append(rec)
    print(f"victim {v} FINAL ssim={s:.4f} (m0={m0:.4f}) cos={cos_final:.4f}{a.tag}", flush=True)
    name = f"v{v}_B{len(rows)}_u{a.u}_{a.init}{a.tag}"
    torch.save({"rec": xd.cpu(), "orig": x.cpu(), "labels": y.cpu(), "studies": studies}, os.path.join(out, name + ".pt"))
    json.dump(rec, open(os.path.join(out, name + ".json"), "w"), indent=1)
print("SUMMARY" + a.tag, json.dumps([{k: r[k] for k in ("victim", "batch", "u", "init", "ssim", "m0", "cos", "minutes")} for r in results]))
