"""Identification score A of saved image-attack results (any batch size).

Reconstructions are matched to the originals by the one-to-one assignment that maximizes mean SSIM
(the paper's rule). For each matched pair, r is the rank of the original among the candidates ordered
by decreasing SSIM with the reconstruction (ties at mid-rank); the candidates are all pairs at the
attack resolution minus the other images of the attacked batch. A = 1 - (r - 1) / (N - 1); the
victim's score is the mean over its batch. Also printed: top-1 and top-1% rates, mean SSIM, and the
same score for the mean image (the free guess, about 0.5).

usage: python src/identify.py runs/pilot/attrib128/*.pt
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ssim(a, b):
    return structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def rank_score(q, pool, true, exclude):
    s = np.array([ssim(p, q) for p in pool])
    keep = np.ones(len(pool), bool); keep[exclude] = False
    others = s[keep]
    r = 1 + (others > s[true]).sum() + 0.5 * (others == s[true]).sum()
    n = keep.sum() + 1
    return 1 - (r - 1) / (n - 1), r


cache = {}
for f in sys.argv[1:]:
    d = torch.load(f)
    rec = d["rec"][:, 0].numpy().astype(np.float64)
    orig = d["orig"][:, 0].numpy().astype(np.float64)
    B, res = len(rec), rec.shape[-1]
    if res not in cache:
        z = np.load(os.path.join(ROOT, "data", "cache", f"images{res}.npz"))
        cache[res] = z["x"].astype(np.float64) / 255
    pool = cache[res]
    idx = [int(np.abs(pool - o).reshape(len(pool), -1).sum(1).argmin()) for o in orig]
    S = np.array([[ssim(orig[i], rec[j]) for j in range(B)] for i in range(B)])
    ri, ci = linear_sum_assignment(-S)
    mean_img = pool.mean(0)
    A, R, Am = [], [], []
    for i, j in zip(ri, ci):
        ex = [k for k in idx if k != idx[i]] + [idx[i]]
        a_, r_ = rank_score(rec[j], pool, idx[i], ex)
        A.append(a_); R.append(r_)
        Am.append(rank_score(mean_img, pool, idx[i], ex)[0])
    n = len(pool) - B + 1
    row = {"file": os.path.basename(f), "B": B, "N": n, "A": round(float(np.mean(A)), 4),
           "top1": round(float(np.mean(np.array(R) == 1)), 3), "top1pct": round(float(np.mean(np.array(R) <= 0.01 * n)), 3),
           "ranks": [int(r) for r in R], "ssim": round(float(S[ri, ci].mean()), 4),
           "A_mean_image": round(float(np.mean(Am)), 4), "ssim_mean_image": round(float(np.mean([ssim(o, mean_img) for o in orig])), 4)}
    print(json.dumps(row), flush=True)
