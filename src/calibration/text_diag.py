"""Diagnostic for the text attack (uses the true texts, so it is calibration, not an attack result).

For the first-layer span check of DAGER: how far are the true (token, position) inputs from the span
recovered from the update, and how many wrong tokens pass at a given threshold? Compared for the exact
gradient (autograd) and the gradient the server derives from the float32 weight difference.

usage: python src/calibration/text_diag.py --global runs/pilot/seed0_fedavg/global_round9.pt --lr-txt 1e-2 --victims 0 1 2
"""
import argparse
import os

import torch

import os as _os, sys as _sys  # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # src/ (common.py)
from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed

ap = argparse.ArgumentParser()
ap.add_argument("--global", dest="glob", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--victims", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--u", type=int, default=1)
ap.add_argument("--lr-img", type=float, default=0.01)
ap.add_argument("--lr-txt", type=float, default=1e-2)
ap.add_argument("--res", type=int, default=128)
a = ap.parse_args()
dev = "cuda"
data = PairData(a.res)
split = load_seed(a.seed)
model = FusionModel().to(dev)
model.load_state_dict(torch.load(os.path.join(ROOT, a.glob), map_location=dev))
model.train()
gpt = model.txt
KEY = "txt.h.0.attn.c_attn.weight"
THR = (5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2)


def dist(R, v):
    v = v / v.norm(dim=-1, keepdim=True)
    return (v - (v @ R.T) @ R).norm(dim=-1)


for v in a.victims:
    studies = split["victim_batches"][str(v)][: a.batch]
    x, ids, mask, y = data.batch(data.rows(studies), dev)
    n_tok = int(mask.sum())
    pairs = {(p, int(ids[b, p])) for b in range(len(ids)) for p in range(int(mask[b].sum()))}
    # exact gradient at the broadcast weights (first step)
    g_exact = torch.autograd.grad(bce(model(x, ids, mask), y), dict(model.named_parameters())[KEY])[0].detach()
    # the update the server sees: u SGD steps in float32, then the weight difference
    m = FusionModel().to(dev); m.load_state_dict(model.state_dict()); m.train()
    w0 = dict(m.named_parameters())[KEY].detach().clone()
    for _ in range(a.u):
        names = [n for n, _ in m.named_parameters()]
        grads = torch.autograd.grad(bce(m(x, ids, mask), y), list(m.parameters()))
        with torch.no_grad():
            for n, p, g in zip(names, m.parameters(), grads):
                p -= (a.lr_img if is_image_param(n) else a.lr_txt) * g
    g_upd = -(dict(m.named_parameters())[KEY].detach() - w0) / (a.lr_txt * a.u)
    rel = ((g_upd - g_exact).norm() / g_exact.norm()).item()
    print(f"\nvictim {v}: {n_tok} tokens, {len(pairs)} distinct (token, position) pairs; "
          f"|update-derived - exact| / |exact| = {rel:.2e}")
    positions = sorted({p for p, _ in pairs})
    for name, g in (("exact", g_exact), ("update", g_upd)):
        U, S, Vh = torch.linalg.svd(g.T.double(), full_matrices=False)
        k_true = len(pairs)
        floor = S[-200:].median()
        k_noise = int((S > 5 * floor).sum())
        print(f"  [{name}] s[0]={S[0]:.2e} s[k-1]={S[k_true - 1]:.2e} s[k]={S[k_true]:.2e} floor={floor:.2e} "
              f"rank(5x floor)={k_noise}; share of singular values above 5x floor among the first {k_true}: "
              f"{(S[:k_true] > 5 * floor).float().mean():.2f}")
        for kname, k in (("k=pairs", k_true), ("k=5x floor", k_noise), ("k=pairs+40", min(k_true + 40, 748))):
            R = Vh[:k]
            d_true, n_false = [], {t: 0 for t in THR}
            for p in positions:
                with torch.no_grad():
                    d = dist(R, gpt.h[0].ln_1(gpt.wte.weight + gpt.wpe.weight[p]).double())
                tid = torch.tensor([t for pp, t in pairs if pp == p], device=dev)
                d_true.append(d[tid])
                d[tid] = 9.0
                for t in THR:
                    n_false[t] += int((d < t).sum())
            d_true = torch.cat(d_true)
            q = [d_true.quantile(x).item() for x in (0.5, 0.9, 0.99, 1.0)]
            miss = {t: int((d_true >= t).sum()) for t in THR}
            print(f"    {kname:11s} ({k}): true-token distance median {q[0]:.1e} 90% {q[1]:.1e} 99% {q[2]:.1e} max {q[3]:.1e}")
            print("       threshold: " + "  ".join(f"{t:g}: miss {miss[t]}/{len(d_true)}, false {n_false[t]}" for t in THR))
