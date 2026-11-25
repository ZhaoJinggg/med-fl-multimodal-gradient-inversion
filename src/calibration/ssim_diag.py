"""Diagnostic (uses the ground truth, so it is not an attack result): how much of the SSIM deficit
of a reconstruction comes from global intensity (brightness / contrast) and from pixel noise?

For each saved result, SSIM against the original is recomputed after
  - a least-squares intensity alignment  a * rec + b  (fitted to the original),
  - a Gaussian blur of the reconstruction (sigma 1 and 2 pixels),
  - both.
The mean image of the pool is listed for comparison.

usage: python src/calibration/ssim_diag.py runs/pilot/b1_long/*.pt
"""
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ssim(a, b):
    return structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


def align(r, o):
    A = np.stack([r.ravel(), np.ones(r.size)], 1)
    (s, b), *_ = np.linalg.lstsq(A, o.ravel(), rcond=None)
    return np.clip(s * r + b, 0, 1), s, b


def tv(x):
    return np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean()


cache = {}
print("file | raw | aligned (a, b) | blur1 | blur2 | aligned+blur1 | aligned+blur2 | mean image | TV rec / orig")
for f in sys.argv[1:]:
    d = torch.load(f)
    for i in range(d["rec"].shape[0]):
        rec = d["rec"][i, 0].numpy().astype(np.float64)
        orig = d["orig"][i, 0].numpy().astype(np.float64)
        res = rec.shape[-1]
        if res not in cache:
            z = np.load(os.path.join(ROOT, "data", "cache", f"images{res}.npz"))
            cache[res] = (z["x"].astype(np.float64) / 255).mean(0)
        al, s, b = align(rec, orig)
        row = [ssim(orig, rec), ssim(orig, al), ssim(orig, gaussian_filter(rec, 1)), ssim(orig, gaussian_filter(rec, 2)),
               ssim(orig, align(gaussian_filter(rec, 1), orig)[0]), ssim(orig, align(gaussian_filter(rec, 2), orig)[0]),
               ssim(orig, cache[res])]
        print(f"{os.path.basename(f)}[{i}] | {row[0]:.3f} | {row[1]:.3f} ({s:.2f}, {b:+.2f}) | {row[2]:.3f} | {row[3]:.3f} | "
              f"{row[4]:.3f} | {row[5]:.3f} | {row[6]:.3f} | {tv(rec):.4f} / {tv(orig):.4f}", flush=True)
