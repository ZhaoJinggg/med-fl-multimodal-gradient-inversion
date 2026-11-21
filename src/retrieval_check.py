"""Retrieval check: does a reconstruction identify its own victim image among all pairs?

For each saved attack result (batch 1), compare the reconstruction with every image in the pool
(all pairs at the same resolution) and report the rank of the true image (1 = best) under SSIM and
under Pearson pixel correlation. The same is done for the mean image (the attacker's free guess), whose
expected rank is about N/2. Runs on CPU from the image cache; no model needed.

usage: python src/retrieval_check.py runs/pilot/b1_long/*_best_r224.pt
"""
import json
import os
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ssim(a, b):
    return structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def corr(q, pool):
    qc = q.ravel() - q.mean()
    pc = pool.reshape(len(pool), -1)
    pc = pc - pc.mean(1, keepdims=True)
    return pc @ qc / (np.linalg.norm(pc, axis=1) * np.linalg.norm(qc) + 1e-12)


def rank_of(scores, i):
    return int((scores > scores[i]).sum()) + 1


cache = {}
out = []
for f in sys.argv[1:]:
    d = torch.load(f)
    rec = d["rec"][0, 0].numpy().astype(np.float64)
    orig = d["orig"][0, 0].numpy().astype(np.float64)
    res = rec.shape[-1]
    if res not in cache:
        z = np.load(os.path.join(ROOT, "data", "cache", f"images{res}.npz"))
        cache[res] = z["x"].astype(np.float64) / 255
    pool = cache[res]
    true = int(np.abs(pool - orig).reshape(len(pool), -1).sum(1).argmin())
    mean_img = pool.mean(0)
    row = {"file": os.path.basename(f), "pool": len(pool)}
    for name, q in (("rec", rec), ("mean", mean_img)):
        s = np.array([ssim(p, q) for p in pool])
        c = corr(q, pool)
        row[name] = {"ssim_true": round(float(s[true]), 4), "ssim_rank": rank_of(s, true),
                     "ssim_others_median": round(float(np.median(np.delete(s, true))), 4),
                     "corr_true": round(float(c[true]), 4), "corr_rank": rank_of(c, true)}
    out.append(row)
    print(json.dumps(row), flush=True)
