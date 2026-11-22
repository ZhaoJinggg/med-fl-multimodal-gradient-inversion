"""Score saved image-attack results: identification score A (primary), top-1 and top-1% rates, SSIM, PSNR,
LPIPS, the mean-image control, and the mismatched control. CPU only.

Writes one tidy row per result file (= one victim under one setting) to OUT and one row per matched pair
to OUT with the suffix "_pairs" (column "kind" = attack | mean_image | mismatched).

Definitions (paper, Section "Evaluation Metrics"):
  * Matching: the reconstructions of a batch are matched to the originals by the one-to-one assignment
    (Hungarian, scipy.optimize.linear_sum_assignment) that maximizes the mean SSIM.
  * Identification score: for a reconstruction matched to original x_i, all candidates (the radiographs of
    the dataset at the attack resolution, minus the OTHER images of the attacked batch; N = 3,199 - B + 1)
    are ordered by decreasing SSIM to the reconstruction; r_i is the rank of x_i (ties at mid-rank) and
    A_i = 1 - (r_i - 1) / (N - 1). The victim's A is the mean over its batch. top1 = share with r_i = 1,
    top1pct = share with r_i <= 0.01 N (as in src/identify.py).
  * SSIM: skimage.metrics.structural_similarity(data_range=1, gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False) - the 11 x 11 Gaussian window of the paper. Here it is computed by a
    vectorized torch implementation (see `ssim_vs_pool`) that reproduces skimage: skimage filters with
    reflect padding and then crops (win_size - 1) / 2 = 5 pixels on each side before averaging, and the
    Gaussian kernel has radius 5, so no cropped-in pixel depends on the padding and a 'valid' filter gives
    the same numbers up to float rounding. Every file also records `ssim_check_maxdiff`, the largest
    difference between the torch and the skimage SSIM on its B x B (original, reconstruction) matrix.
  * PSNR = 10 log10(1 / MSE) per matched pair (data range 1); LPIPS = AlexNet LPIPS (lpips package) with the
    gray image replicated to 3 channels and scaled to [-1, 1]; both are means over the matched pairs.
  * Mean-image control ("_mean_image" columns): every reconstruction is replaced by the mean of the
    validation + test images (the public mean radiograph that initializes the attack) at the attack
    resolution, and the same scores are computed (the matching is then trivial).
  * Mismatched control ("_mismatched" columns; the paper's definition): the ORIGINALS of victim v are held
    fixed and scored against the RECONSTRUCTIONS of every other victim w of the same group: rec_w is matched to
    v's originals by the same max-mean-SSIM assignment, and each original of v is ranked among the candidate
    pool of v's real score (the dataset minus the other images of v's batch) by SSIM to its matched rec_w.
    A_mismatched(v) is the mean of these A over all w; A_mismatched_sd is their spread over partners, and
    n_partners their number. GROUP: same (directory, batch size, u, init, tag, resolution); the seed is NOT
    required to match, so the 20 test victims of one factor combination form one group. A group with a single
    victim gets NaN. Because v's images are fixed, A(v) - A_mismatched(v) removes how typical v's radiographs
    are. The per-partner pair rows are in the pairs file (kind = mismatched, partner_file, partner_victim). The
    paper's zero point m_0 for images is the mean of A_mismatched over the FedAvg, u = 1, recovered-text cell.
  * A_mismatched_next (old rule, kept for comparison because it is free): the reconstructions of v scored
    against the originals of the next victim (by victim id, cyclic) of the same group AND seed, pool minus the
    other images of that victim's batch (kind = mismatched_next in the pairs file).
  * LPIPS is not computed for the two mismatched controls (NaN in the pairs file).

Identifiers: victim, batch, u, init and tag come from the file name v{victim}_B{batch}_u{u}_{init}{tag};
the seed from the JSON key "seed" if present, else --seed, else "seed<k>" in the JSON "global" path, else
"seed<k>" in the directory path (column seed_source says which); the protocol from the JSON key
"protocol" if present, else from "seed<k>_<protocol>" in the global path. All scalar JSON fields are copied
with the prefix "json_".

usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python src/score_images.py runs/pilot/push_b8 \
      "runs/pilot/push_b24/*.pt" --out runs/scores_test/pilot.csv [--seed 0] [--no-lpips]
  python src/score_images.py --verify-ssim 20      # torch vs skimage on 20 random pool pairs, then exit
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCH_HOME", os.path.join(ROOT, "pretrained", "torch"))  # cached AlexNet for LPIPS

NAME_RE = re.compile(r"^v(\d+)_B(\d+)_u(\d+)_(random|mean|near)(.*)$")
C1, C2 = 0.01 ** 2, 0.03 ** 2


# ---------------------------------------------------------------- SSIM (vectorized, skimage-equivalent)
def gauss_band(n, sigma=1.5, truncate=3.5, dtype=torch.float64):
    """(n - 2r) x n matrix G such that G @ x is the 'valid' 1-D Gaussian filter of scipy/skimage."""
    r = int(truncate * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=torch.float64)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    G = torch.zeros(n - 2 * r, n, dtype=torch.float64)
    for i in range(n - 2 * r):
        G[i, i:i + 2 * r + 1] = k
    return G.to(dtype)


class SSIMPool:
    """Candidate pool (N, R, R) with precomputed local means and variances; SSIM of a query to all of it."""

    def __init__(self, pool, dtype=torch.float64, chunk=400):
        self.dtype, self.chunk = dtype, chunk
        self.pool = pool.to(dtype)
        R = pool.shape[-1]
        self.G = gauss_band(R, dtype=dtype)
        self.mu, self.var = [], []
        for s in range(0, len(self.pool), chunk):
            p = self.pool[s:s + chunk]
            mu = self.filt(p)
            self.mu.append(mu)
            self.var.append(self.filt(p * p) - mu * mu)

    def filt(self, x):
        return self.G @ x @ self.G.T

    def ssim_pair(self, a, b):
        """SSIM of two single images (R, R), same implementation."""
        a, b = a.to(self.dtype), b.to(self.dtype)
        ma, mb = self.filt(a), self.filt(b)
        va, vb = self.filt(a * a) - ma * ma, self.filt(b * b) - mb * mb
        vab = self.filt(a * b) - ma * mb
        S = (2 * ma * mb + C1) * (2 * vab + C2) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
        return float(S.mean())

    def ssim_vs_pool(self, q):
        """SSIM of query q (R, R) with every candidate; returns a float64 numpy array (N,)."""
        q = q.to(self.dtype)
        mq = self.filt(q)
        vq = self.filt(q * q) - mq * mq
        out = []
        for c, s in enumerate(range(0, len(self.pool), self.chunk)):
            p, mp, vp = self.pool[s:s + self.chunk], self.mu[c], self.var[c]
            vpq = self.filt(p * q) - mp * mq
            S = (2 * mp * mq + C1) * (2 * vpq + C2) / ((mp * mp + mq * mq + C1) * (vp + vq + C2))
            out.append(S.mean(dim=(1, 2)))
        return torch.cat(out).double().numpy()


def sk_ssim(a, b):
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)


# ---------------------------------------------------------------- data
class Resolution:
    """Everything that depends only on the attack resolution: pool, study index, mean image, its SSIMs."""

    def __init__(self, res, dtype, chunk):
        z = np.load(os.path.join(ROOT, "data", "cache", f"images{res}.npz"))
        self.ids = [str(s) for s in z["ids"]]
        self.index = {s: i for i, s in enumerate(self.ids)}
        self.x = torch.from_numpy(z["x"].astype(np.float64) / 255)  # (N, R, R), exactly as identify.py
        with open(os.path.join(ROOT, "data", "splits", "pairs.csv"), encoding="utf-8") as f:
            pairs = list(csv.DictReader(f))
        if [p["study"] for p in pairs] != self.ids:
            raise SystemExit(f"images{res}.npz ids do not match data/splits/pairs.csv")
        pub = [i for i, p in enumerate(pairs) if p["split"] in ("val", "test")]
        self.mean_img = self.x[pub].mean(0)
        self.pool = SSIMPool(self.x, dtype, chunk)
        self.s_mean = self.pool.ssim_vs_pool(self.mean_img)  # mean image vs every candidate (cached)


def pool_indices(res_data, orig, studies):
    """Pool row of each original: by study id, checked against the pixels; nearest image as fallback."""
    idx, how = [], "study"
    for k, o in enumerate(orig):
        i = res_data.index.get(str(studies[k])) if studies is not None else None
        if i is None or float((res_data.x[i] - o).abs().max()) > 1e-4:
            i = int((res_data.x - o).abs().flatten(1).sum(1).argmin())
            how = "nearest"
        idx.append(i)
    return idx, how


# ---------------------------------------------------------------- scores
def rank_scores(svecs, idx_batch, assign):
    """A and r of each matched pair. svecs[j] = SSIM of reconstruction j to every candidate;
    assign = list of (i, j): original i of the scored batch matched to reconstruction j."""
    out = []
    for i, j in assign:
        s = svecs[j]
        true = idx_batch[i]
        keep = np.ones(len(s), bool)
        keep[idx_batch] = False
        others = s[keep]
        r = 1 + (others > s[true]).sum() + 0.5 * (others == s[true]).sum()
        n = keep.sum() + 1
        out.append((1 - (r - 1) / (n - 1), float(r), int(n)))
    return out


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


class LPIPS:
    def __init__(self, enabled):
        self.model = None
        if enabled:
            try:
                import lpips
                self.model = lpips.LPIPS(net="alex", verbose=False).eval()
            except Exception as e:  # missing package or weights: keep going without LPIPS
                print(f"WARNING: LPIPS disabled ({e})", file=sys.stderr)

    @torch.no_grad()
    def __call__(self, a, b):
        """a, b: (R, R) in [0, 1] -> LPIPS distance (float), NaN if disabled."""
        if self.model is None:
            return float("nan")
        t = lambda x: (x.float() * 2 - 1).view(1, 1, *x.shape).repeat(1, 3, 1, 1)
        return float(self.model(t(a), t(b)).item())


def score_set(rec, orig_scored, idx_scored, svecs, lp, kind):
    """Match rec (B,R,R) to orig_scored (B,R,R) by max mean SSIM, then all metrics; returns summary, pairs."""
    B = len(rec)
    S = np.array([[svecs[j][idx_scored[i]] for j in range(B)] for i in range(B)])  # S[i, j] = SSIM(orig i, rec j)
    ri, ci = linear_sum_assignment(-S)
    assign = list(zip(ri.tolist(), ci.tolist()))
    rk = rank_scores(svecs, idx_scored, assign)
    pairs = []
    for (i, j), (a_, r_, n_) in zip(assign, rk):
        pairs.append({"kind": kind, "pair": i, "rec_index": j, "pool_index": idx_scored[i], "rank": r_, "N": n_,
                      "A": a_, "ssim": S[i, j], "psnr": psnr(rec[j], orig_scored[i]), "lpips": lp(rec[j], orig_scored[i])})
    n = pairs[0]["N"]
    R = np.array([p["rank"] for p in pairs])
    summ = {"A": np.mean([p["A"] for p in pairs]), "top1": float(np.mean(R == 1)), "top1pct": float(np.mean(R <= 0.01 * n)),
            "ssim": float(np.mean([p["ssim"] for p in pairs])), "psnr": float(np.mean([p["psnr"] for p in pairs])),
            "lpips": float(np.mean([p["lpips"] for p in pairs]))}
    return summ, pairs, S


# ---------------------------------------------------------------- identifiers
def identifiers(path, meta, cli_seed):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = NAME_RE.match(stem)
    if not m:
        raise ValueError(f"file name does not match v{{victim}}_B{{batch}}_u{{u}}_{{init}}{{tag}}: {stem}")
    row = {"file": stem, "dir": os.path.relpath(os.path.dirname(os.path.abspath(path)), ROOT).replace("\\", "/"),
           "victim": int(m.group(1)), "batch": int(m.group(2)), "u": int(m.group(3)), "init": m.group(4), "tag": m.group(5)}
    glob_path = str(meta.get("global", ""))
    gm = re.search(r"seed(\d+)(?:_([A-Za-z]+))?", glob_path)
    dm = re.search(r"seed(\d+)", row["dir"])
    if "seed" in meta:
        row["seed"], row["seed_source"] = int(meta["seed"]), "json"
    elif cli_seed is not None:
        row["seed"], row["seed_source"] = cli_seed, "cli"
    elif gm:
        row["seed"], row["seed_source"] = int(gm.group(1)), "global_path"
    elif dm:
        row["seed"], row["seed_source"] = int(dm.group(1)), "dir"
    else:
        row["seed"], row["seed_source"] = "", "unknown"
    row["protocol"] = meta.get("protocol", gm.group(2).lower() if gm and gm.group(2) else "")
    row["condition"] = meta.get("condition", meta.get("text", ""))
    return row


def collect(inputs):
    files = []
    for s in inputs:
        if os.path.isdir(s):
            files += sorted(glob.glob(os.path.join(s, "*.pt")))
        else:
            files += sorted(glob.glob(s)) if any(ch in s for ch in "*?[") else [s]
    files = [f for f in files if NAME_RE.match(os.path.splitext(os.path.basename(f))[0])]
    return list(dict.fromkeys(os.path.abspath(f) for f in files))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def verify_ssim(n, res, dtype, chunk, seed=0):
    """Largest |torch - skimage| SSIM difference on n random candidate pairs and on the mean image."""
    rd = Resolution(res, dtype, chunk)
    g = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        q = int(g.integers(len(rd.x)))
        s = rd.pool.ssim_vs_pool(rd.x[q])
        for c in g.integers(len(rd.x), size=5).tolist() + [q]:
            worst = max(worst, abs(s[c] - sk_ssim(rd.x[c].numpy(), rd.x[q].numpy())))
    for c in g.integers(len(rd.x), size=20).tolist():
        worst = max(worst, abs(rd.s_mean[c] - sk_ssim(rd.x[c].numpy(), rd.mean_img.numpy())))
    print(f"verify-ssim res={res} dtype={dtype}: max |torch - skimage| = {worst:.3e} over {n * 6 + 20} pairs")
    return worst


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="result directories, .pt files, or quoted globs")
    ap.add_argument("--out", default="runs/scores/images.csv")
    ap.add_argument("--seed", type=int, default=None, help="seed for files whose JSON has none")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-skimage-check", action="store_true", help="skip the per-file skimage cross-check")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--chunk", type=int, default=400, help="candidates per vectorized block")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    ap.add_argument("--verify-ssim", type=int, default=0, metavar="N", help="check torch vs skimage SSIM and exit")
    ap.add_argument("--verify-res", type=int, default=128)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dtype = getattr(torch, a.dtype)
    if a.verify_ssim:
        verify_ssim(a.verify_ssim, a.verify_res, dtype, a.chunk)
        return
    files = collect(a.inputs)
    if not files:
        raise SystemExit("no result files found")
    lp = LPIPS(not a.no_lpips)
    by_res, done = {}, []
    t_all = time.time()
    # phase 1: attack scores and mean-image control; keep each file's SSIM-to-pool vectors
    for path in files:
        t0 = time.time()
        d = torch.load(path, map_location="cpu")
        jp = os.path.splitext(path)[0] + ".json"
        meta = json.load(open(jp)) if os.path.exists(jp) else {}
        row = identifiers(path, meta, a.seed)
        rec = d["rec"][:, 0].double().clamp(0, 1)
        orig = d["orig"][:, 0].double()
        B, res = len(rec), rec.shape[-1]
        if B != row["batch"]:
            print(f"WARNING {row['file']}: batch {B} in tensor, {row['batch']} in name", file=sys.stderr)
        if res not in by_res:
            by_res[res] = Resolution(res, dtype, a.chunk)
        rd = by_res[res]
        idx, how = pool_indices(rd, orig, d.get("studies"))
        svecs = [rd.pool.ssim_vs_pool(rec[j]) for j in range(B)]
        orig_pool = rd.x[idx]
        summ, pairs, S = score_set(rec, orig_pool, idx, svecs, lp, "attack")
        row.update({"res": res, "N": pairs[0]["N"], "pool_match": how})
        row.update(summ)
        row["ranks"] = json.dumps([p["rank"] for p in pairs])
        # mean-image control: the same mean image for every original, so its SSIM vector is shared
        mvecs = [rd.s_mean] * B
        msumm, mpairs, _ = score_set(rd.mean_img.expand(B, res, res), orig_pool, idx, mvecs, lp, "mean_image")
        row.update({k + "_mean_image": v for k, v in msumm.items()})
        if not a.no_skimage_check:
            Ssk = np.array([[sk_ssim(orig[i].numpy(), rec[j].numpy()) for j in range(B)] for i in range(B)])
            row["ssim_check_maxdiff"] = float(np.abs(Ssk - S).max())
        row.update({"json_" + k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))})
        row["seconds"] = round(time.time() - t0, 1)
        studies = list(d.get("studies", [""] * B))
        labels = d.get("labels", torch.full((B,), float("nan"))).tolist()
        for p in pairs + mpairs:
            p.update({"file": row["file"], "victim": row["victim"], "seed": row["seed"], "study": studies[p["pair"]],
                      "label": labels[p["pair"]], "partner_file": ""})
        done.append({"row": row, "pairs": pairs + mpairs, "svecs": svecs, "rec": rec, "idx": idx, "orig": orig_pool,
                     "studies": studies, "labels": labels})
        print(f"{row['file']}: A={row['A']:.4f} top1={row['top1']:.3f} ssim={row['ssim']:.4f} psnr={row['psnr']:.2f} "
              f"lpips={row['lpips']:.4f} | mean image A={row['A_mean_image']:.4f} | {row['seconds']}s", flush=True)
    # phase 2: mismatched controls (reuse the SSIM-to-pool vectors of phase 1, so they cost no SSIM evaluations;
    # LPIPS is not computed for them)
    no_lp = lambda x, y: float("nan")  # noqa: E731
    groups, groups_next = {}, {}
    for k, e in enumerate(done):
        r = e["row"]
        groups.setdefault((r["dir"], r["batch"], r["u"], r["init"], r["tag"], r["res"]), []).append(k)
        groups_next.setdefault((r["dir"], r["batch"], r["u"], r["init"], r["tag"], r["res"], r["seed"]), []).append(k)
    order = lambda k: (str(done[k]["row"]["seed"]), done[k]["row"]["victim"])  # noqa: E731
    # (a) A_mismatched: originals of v held fixed, scored against the reconstructions of every other victim w
    for key, ks in groups.items():
        ks = sorted(ks, key=order)
        for k in ks:
            e = done[k]
            per = []
            for m in ks:
                if m == k:
                    continue
                w = done[m]
                assert len(w["rec"]) == len(e["idx"]), f"batch size differs: {w['row']['file']} vs {e['row']['file']}"
                summ, pairs, _ = score_set(w["rec"], e["orig"], e["idx"], w["svecs"], no_lp, "mismatched")
                per.append(summ)
                for p in pairs:
                    p.update({"file": e["row"]["file"], "victim": e["row"]["victim"], "seed": e["row"]["seed"],
                              "study": e["studies"][p["pair"]], "label": e["labels"][p["pair"]],
                              "partner_file": w["row"]["file"], "partner_victim": w["row"]["victim"]})
                e["pairs"] += pairs
            e["row"]["n_partners"] = len(per)
            for k2 in ("A", "top1", "top1pct", "ssim", "psnr"):
                e["row"][k2 + "_mismatched"] = float(np.mean([s[k2] for s in per])) if per else float("nan")
            e["row"]["A_mismatched_sd"] = float(np.std([s["A"] for s in per], ddof=1)) if len(per) > 1 else float("nan")
    # (b) A_mismatched_next (old single-partner rule, kept for comparison): reconstructions of v scored against
    # the originals of the next victim of the same seed (cyclic), pool minus the other images of that batch
    for key, ks in groups_next.items():
        ks = sorted(ks, key=order)
        for n, k in enumerate(ks):
            e = done[k]
            if len(ks) < 2:
                e["row"].update({"partner_next": "", "A_mismatched_next": float("nan")})
                continue
            w = done[ks[(n + 1) % len(ks)]]
            summ, pairs, _ = score_set(e["rec"], w["orig"], w["idx"], e["svecs"], no_lp, "mismatched_next")
            e["row"]["partner_next"] = w["row"]["file"]
            e["row"]["A_mismatched_next"] = summ["A"]
            for p in pairs:
                p.update({"file": e["row"]["file"], "victim": e["row"]["victim"], "seed": e["row"]["seed"],
                          "study": w["studies"][p["pair"]], "label": w["labels"][p["pair"]],
                          "partner_file": w["row"]["file"], "partner_victim": w["row"]["victim"]})
            e["pairs"] += pairs
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_csv(out, [e["row"] for e in done])
    write_csv(os.path.splitext(out)[0] + "_pairs.csv", [p for e in done for p in e["pairs"]])
    print(f"wrote {out} ({len(done)} files) and its _pairs.csv in {time.time() - t_all:.0f}s")
    for key, ks in groups.items():
        rows = [done[k]["row"] for k in ks]
        mm = [r.get("A_mismatched", float("nan")) for r in rows]
        print(f"group dir={key[0]} B={key[1]} u={key[2]} init={key[3]} tag={key[4]!r} res={key[5]}: n={len(rows)} "
              f"mean A={np.mean([r['A'] for r in rows]):.4f} mean A_mismatched={np.nanmean(mm) if len(rows) > 1 else float('nan'):.4f} "
              f"mean A_mean_image={np.mean([r['A_mean_image'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
