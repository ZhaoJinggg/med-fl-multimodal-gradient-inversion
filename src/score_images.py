"""Score saved image-attack results: identification score A (primary, matching-free max rule), top-1 and
top-1% shares, the same-label score A_lab, the old matched score A_matched, SSIM, PSNR, LPIPS, the
mean-image control, the mismatched control, and the full cross matrices A(v, w) and A_lab(v, w) of every
group. CPU only.

Outputs (OUT = --out):
  OUT              one row per result file (= one victim under one setting)
  OUT_pairs.csv    one row per original: kind = attack (max-rule rank and A, same-label rank_lab, N_lab and
                   A_lab, plus the matched pair's A_matched, SSIM, PSNR, LPIPS), mean_image, and mismatched (one
                   row per original of v and partner w: rank, A, rank_lab, A_lab of v's original against w's
                   reconstruction batch). Label columns on attack and mismatched rows: label (from the .pt),
                   label_dataset (pairs.csv), rec_n_abnormal and rec_batch (label count and size of the
                   reconstruction batch scored: v's own on attack rows, w's on mismatched rows), x_share = share
                   of that batch with the label of this original (the regressor of analyze.label_regression),
                   group (as in OUT_cross.csv).
  OUT_cross.csv    the cross matrix of every group, long format: one row per (v, w), including the diagonal
                   A(v, v) = the real score; columns = group fields + v_seed, v_victim, v_file, v_dir,
                   v_n_abnormal, w_seed, w_victim, w_file, w_dir, w_n_abnormal, diagonal, A, top1, top1pct,
                   A_lab, top1_lab, top1pct_lab, scope. The A_lab matrix is the extra column A_lab of the
                   same file (no second cross file). src/analyze.py reads it (gate, floor tests; the strata
                   of the stratified permutation test are v_n_abnormal).

Definitions (paper, Section "Evaluation Metrics"):
  * Primary score A (no matching): for the reconstruction batch x^_1..x^_B every candidate radiograph c gets
    s(c) = max_j SSIM(c, x^_j). For original x_i the candidates are all radiographs of the dataset at the
    attack resolution without the OTHER originals of the attacked batch (N = 3,199 - B + 1); r_i is the rank
    of x_i by decreasing s (ties at mid-rank) and A_i = 1 - (r_i - 1) / (N - 1). The victim's A is the mean
    over its B originals; top1 = share of r_i = 1, top1pct = share of r_i <= 0.01 N. At B = 1 the max rule
    and the matched rule coincide.
  * Same-label score A_lab (label control, second review): the same max rule, but original x_i is ranked
    only among the candidates with its own label (label column of data/splits/pairs.csv): the dataset images
    of that label minus the other originals of the attacked batch (N_lab_i = their number + 1). The
    attacker is given the batch labels, so a reconstruction batch can carry label information alone; under
    "label information only" s(c) does not separate candidates of one label and E[A_lab] = 0.5 for every
    (v, w). r_lab_i <= r_i always (fewer competitors). top1_lab / top1pct_lab as above (1% of N_lab_i).
    The label of an original is its pairs.csv label (label_dataset); label_mismatch counts originals whose
    .pt label differs (expected 0). n_abnormal = number of abnormal pairs of the attacked batch, from the
    .pt labels (= the first B pairs actually attacked); pairs.csv labels if the .pt has none.
    A_lab_mismatched(v) = mean over w != v of A_lab(v, w) (same partners as A_mismatched).
  * A_matched / top1_matched / top1pct_matched (old rule, for the record): reconstructions matched to the
    originals by the one-to-one assignment maximizing mean SSIM (Hungarian), each original ranked by SSIM to
    its matched reconstruction (identical to src/identify.py). This assignment is still used for SSIM, PSNR
    and LPIPS, which are means over the matched pairs.
  * SSIM: skimage.metrics.structural_similarity(data_range=1, gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False), computed by a vectorized torch implementation (`SSIMPool`): skimage filters
    with reflect padding and crops 5 px before averaging; the Gaussian kernel has radius 5, so a 'valid'
    filter gives the same numbers (max difference 5.6e-15 in float64). `ssim_check_maxdiff` records the
    largest difference to skimage on each file's B x B (original, reconstruction) matrix.
  * PSNR = 10 log10(1 / MSE) (data range 1); LPIPS = AlexNet LPIPS (lpips package), gray image replicated to
    3 channels and scaled to [-1, 1].
  * Mean-image control ("_mean_image"): the public mean image (validation + test pairs) repeated B times as
    the reconstruction batch, same max rule (s(c) is then SSIM(c, mean image)).
  * Mismatched control: for victim v and every other victim w of the same group, A(v, w) = the max-rule
    score of v's originals (candidate pool of v) against w's reconstruction batch. A_mismatched(v) = mean
    over w != v; A_mismatched_sd = spread over partners; n_partners. LPIPS is not computed for controls.
  * GROUP (control and cross matrix), --scope directory (default): same (directory, protocol, condition,
    batch, u, init, tag, resolution); --scope cell: the same fields without the directory, so the result
    files of one cell spread over several directories (e.g. one per seed: the main experiment has 4 victims
    from each of seeds 1-5 per cell) form ONE group and one cross matrix; the cross file's "dir" column is
    then the ";"-joined list of the group's directories (v_dir / w_dir per row). Victims are identified by
    (seed, victim) (client 1 is a victim in four seeds); the same (seed, victim) twice in a group is an
    error. A victim's own reconstructions from another protocol, u, text condition or tag never enter its
    control. A group with one victim gets NaN.

Identifiers: victim, batch, u, init and tag from the file name [s{seed}_]v{victim}_B{batch}_u{u}_{init}{tag}
(attack_image.py writes the seed prefix since 2026-11-22; older names without it are read too); seed from
JSON "seed", else the file-name prefix s{seed} (an error if both exist and differ), else --seed, else
"seed<k>" in the JSON "global" path, else in the directory path (column seed_source); protocol from JSON
"proto" or "protocol", else "seed<k>_<protocol>" in the global path, else --protocol (default empty);
condition from JSON "condition", else JSON "text" (old files: "oracle" -> oracle), else empty. All scalar
JSON fields are copied with the prefix "json_".

usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python src/score_images.py runs/pilot/push_b8 \
      "runs/pilot/push_b24/*.pt" --out runs/scores_test/pilot.csv [--seed 0] [--no-lpips] [--scope cell]
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

NAME_RE = re.compile(r"^(?:s(\d+)_)?v(\d+)_B(\d+)_u(\d+)_(random|mean|near)(.*)$")
C1, C2 = 0.01 ** 2, 0.03 ** 2
GROUP_FIELDS = ("dir", "protocol", "condition", "batch", "u", "init", "tag", "res")
CELL_FIELDS = GROUP_FIELDS[1:]  # --scope cell: across directories
COND_ALIASES = {"oracle": "oracle", "recovered": "recovered", "rec": "recovered", "none": "none",
                "notext": "none", "no_text": "none", "no-text": "none"}


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
        self.G = gauss_band(pool.shape[-1], dtype=dtype)
        self.mu, self.var = [], []
        for s in range(0, len(self.pool), chunk):
            p = self.pool[s:s + chunk]
            mu = self.filt(p)
            self.mu.append(mu)
            self.var.append(self.filt(p * p) - mu * mu)

    def filt(self, x):
        return self.G @ x @ self.G.T

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
        self.labels = np.array([int(p["label"]) for p in pairs])  # 0 normal, 1 abnormal (A_lab pools)
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
def rank_of(s, true, idx_batch, allowed=None):
    """Mid-rank of candidate `true` by decreasing s among the pool minus the other originals of the batch;
    with `allowed` (boolean mask over the pool) only the allowed candidates compete (A_lab: same label)."""
    keep = np.ones(len(s), bool) if allowed is None else np.asarray(allowed, bool).copy()
    keep[idx_batch] = False
    others = s[keep]
    r = 1 + (others > s[true]).sum() + 0.5 * (others == s[true]).sum()
    n = int(keep.sum()) + 1
    return 1 - (r - 1) / (n - 1), float(r), n


def rank_scores(svecs, idx_batch, assign):
    """Old matched rule for given (i, j) pairs: original i ranked by SSIM to reconstruction j. Kept because
    scratch scripts import it."""
    return [rank_of(svecs[j], idx_batch[i], idx_batch) for i, j in assign]


def max_rule(svecs, idx_batch, labels=None):
    """Primary score: s(c) = max_j SSIM(c, rec_j); per original (A_i, r_i, N). With `labels` (label of every
    pool candidate) the same-label score A_lab: original i competes only with candidates of its own label."""
    s = np.max(np.stack(svecs), 0)
    if labels is None:
        return [rank_of(s, true, idx_batch) for true in idx_batch]
    return [rank_of(s, true, idx_batch, labels == labels[true]) for true in idx_batch]


def summarize(ranks, suffix=""):
    A = np.array([a for a, _, _ in ranks])
    R = np.array([r for _, r, _ in ranks])
    n = np.array([k for _, _, k in ranks])  # N differs per original only for A_lab
    return {"A" + suffix: float(A.mean()), "top1" + suffix: float(np.mean(R == 1)),
            "top1pct" + suffix: float(np.mean(R <= 0.01 * n))}


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
        if self.model is None:
            return float("nan")
        t = lambda x: (x.float() * 2 - 1).view(1, 1, *x.shape).repeat(1, 3, 1, 1)  # noqa: E731
        return float(self.model(t(a), t(b)).item())


def matched(rec, orig, idx, svecs, lp):
    """Old rule: Hungarian assignment maximizing mean SSIM; per original i: (j, A, r, N, ssim, psnr, lpips)."""
    B = len(rec)
    S = np.array([[svecs[j][idx[i]] for j in range(B)] for i in range(B)])  # S[i, j] = SSIM(orig i, rec j)
    ri, ci = linear_sum_assignment(-S)
    out = {}
    for i, j in zip(ri.tolist(), ci.tolist()):
        (a_, r_, n_), = rank_scores(svecs, idx, [(i, j)])
        out[i] = (j, a_, r_, n_, S[i, j], psnr(rec[j], orig[i]), lp(rec[j], orig[i]))
    return out, S


# ---------------------------------------------------------------- identifiers
def identifiers(path, meta, cli_seed, cli_protocol):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = NAME_RE.match(stem)
    if not m:
        raise ValueError(f"file name does not match [s{{seed}}_]v{{victim}}_B{{batch}}_u{{u}}_{{init}}{{tag}}: {stem}")
    row = {"file": stem, "dir": os.path.relpath(os.path.dirname(os.path.abspath(path)), ROOT).replace("\\", "/"),
           "victim": int(m.group(2)), "batch": int(m.group(3)), "u": int(m.group(4)), "init": m.group(5), "tag": m.group(6)}
    name_seed = int(m.group(1)) if m.group(1) is not None else None
    glob_path = str(meta.get("global", ""))
    gm = re.search(r"seed(\d+)(?:_([A-Za-z]+))?", glob_path)
    dm = re.search(r"seed(\d+)", row["dir"])
    if "seed" in meta:
        row["seed"], row["seed_source"] = int(meta["seed"]), "json"
        if name_seed is not None and name_seed != row["seed"]:
            raise ValueError(f"{stem}: seed {name_seed} in the file name, {row['seed']} in the JSON")
    elif name_seed is not None:
        row["seed"], row["seed_source"] = name_seed, "file_name"
    elif cli_seed is not None:
        row["seed"], row["seed_source"] = cli_seed, "cli"
    elif gm:
        row["seed"], row["seed_source"] = int(gm.group(1)), "global_path"
    elif dm:
        row["seed"], row["seed_source"] = int(dm.group(1)), "dir"
    else:
        row["seed"], row["seed_source"] = "", "unknown"
    proto = meta.get("proto", meta.get("protocol"))
    if proto is None:
        proto = gm.group(2) if gm and gm.group(2) else cli_protocol
    row["protocol"] = str(proto).lower()
    cond = str(meta.get("condition", meta.get("text", ""))).strip().lower()
    row["condition"] = COND_ALIASES.get(cond, cond)
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
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="result directories, .pt files, or quoted globs")
    ap.add_argument("--out", default="runs/scores/images.csv")
    ap.add_argument("--scope", choices=["directory", "cell"], default="directory",
                    help="group (control, cross matrix) per directory, or per cell across directories")
    ap.add_argument("--seed", type=int, default=None, help="seed for files whose JSON has none")
    ap.add_argument("--protocol", default="", help="protocol for files whose JSON and global path have none")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-skimage-check", action="store_true", help="skip the per-file skimage cross-check")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--chunk", type=int, default=400, help="candidates per vectorized block")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    ap.add_argument("--verify-ssim", type=int, default=0, metavar="N", help="check torch vs skimage SSIM and exit")
    ap.add_argument("--verify-res", type=int, default=128)
    a = ap.parse_args(argv)
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
    # phase 1: own scores and mean-image control; keep each file's SSIM-to-pool vectors
    for path in files:
        t0 = time.time()
        d = torch.load(path, map_location="cpu")
        jp = os.path.splitext(path)[0] + ".json"
        meta = json.load(open(jp)) if os.path.exists(jp) else {}
        row = identifiers(path, meta, a.seed, a.protocol)
        rec = d["rec"][:, 0].double().clamp(0, 1)
        orig = d["orig"][:, 0].double()
        B, res = len(rec), rec.shape[-1]
        if B != row["batch"] or len(orig) != B:
            raise SystemExit(f"{row['file']}: {B} reconstructions, {len(orig)} originals, batch {row['batch']} in name")
        if res not in by_res:
            by_res[res] = Resolution(res, dtype, a.chunk)
        rd = by_res[res]
        idx, how = pool_indices(rd, orig, d.get("studies"))
        svecs = [rd.pool.ssim_vs_pool(rec[j]) for j in range(B)]
        orig_pool = rd.x[idx]
        own = max_rule(svecs, idx)
        own_lab = max_rule(svecs, idx, rd.labels)
        mt, S = matched(rec, orig_pool, idx, svecs, lp)
        mm = max_rule([rd.s_mean], idx)
        mm_lab = max_rule([rd.s_mean], idx, rd.labels)
        lab_ds = rd.labels[idx]  # pairs.csv label of each original (defines its A_lab pool)
        labels = [float(x) for x in d["labels"].tolist()] if d.get("labels") is not None else [float("nan")] * B
        lab_batch = np.array([lab_ds[i] if math.isnan(labels[i]) else labels[i] for i in range(B)])  # labels given to the attack
        row.update({"res": res, "N": own[0][2], "pool_match": how, "n_abnormal": int(round(lab_batch.sum())),
                    "label_mismatch": int(sum(not math.isnan(labels[i]) and int(labels[i]) != int(lab_ds[i]) for i in range(B)))})
        row.update(summarize(own))
        row["ranks"] = json.dumps([r for _, r, _ in own])
        row.update(summarize(own_lab, "_lab"))
        row["ranks_lab"] = json.dumps([r for _, r, _ in own_lab])
        row["N_lab"] = json.dumps([n for _, _, n in own_lab])
        mtr = [(mt[i][1], mt[i][2], mt[i][3]) for i in range(B)]
        row.update({k + "_matched": v for k, v in summarize(mtr).items()})
        row["ranks_matched"] = json.dumps([r for _, r, _ in mtr])
        row.update(ssim=float(np.mean([mt[i][4] for i in range(B)])), psnr=float(np.mean([mt[i][5] for i in range(B)])),
                   lpips=float(np.mean([mt[i][6] for i in range(B)])))
        row.update({k + "_mean_image": v for k, v in summarize(mm).items()})
        row.update({k + "_mean_image": v for k, v in summarize(mm_lab, "_lab").items()})
        row.update(ssim_mean_image=float(np.mean([rd.s_mean[i] for i in idx])),
                   psnr_mean_image=float(np.mean([psnr(rd.mean_img, o) for o in orig_pool])),
                   lpips_mean_image=float(np.mean([lp(rd.mean_img, o) for o in orig_pool])))
        if not a.no_skimage_check:
            Ssk = np.array([[sk_ssim(orig[i].numpy(), rec[j].numpy()) for j in range(B)] for i in range(B)])
            row["ssim_check_maxdiff"] = float(np.abs(Ssk - S).max())
        row.update({"json_" + k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))})
        row["seconds"] = round(time.time() - t0, 1)
        studies = list(d.get("studies", [""] * B))
        base = {"file": row["file"], "seed": row["seed"], "victim": row["victim"]}
        pairs = []
        for i in range(B):
            j, a_m, r_m, _, s_, p_, l_ = mt[i]
            lab = {"label": labels[i], "label_dataset": int(lab_ds[i])}
            pairs.append({**base, "kind": "attack", "pair": i, "study": studies[i], **lab, "pool_index": idx[i],
                          "rank": own[i][1], "N": own[i][2], "A": own[i][0],
                          "rank_lab": own_lab[i][1], "N_lab": own_lab[i][2], "A_lab": own_lab[i][0],
                          "rec_batch": B, "rec_n_abnormal": row["n_abnormal"], "x_share": float(np.mean(lab_batch == lab_ds[i])),
                          "rec_index_matched": j, "rank_matched": r_m,
                          "A_matched": a_m, "ssim": s_, "psnr": p_, "lpips": l_})
            pairs.append({**base, "kind": "mean_image", "pair": i, "study": studies[i], **lab, "pool_index": idx[i],
                          "rank": mm[i][1], "N": mm[i][2], "A": mm[i][0],
                          "rank_lab": mm_lab[i][1], "N_lab": mm_lab[i][2], "A_lab": mm_lab[i][0], "ssim": rd.s_mean[idx[i]]})
        done.append({"row": row, "pairs": pairs, "svecs": svecs, "idx": idx, "studies": studies, "labels": labels,
                     "lab_ds": lab_ds, "lab_batch": lab_batch, "pool_labels": rd.labels})
        print(f"{row['file']}: A={row['A']:.4f} A_lab={row['A_lab']:.4f} top1={row['top1']:.3f} A_matched={row['A_matched']:.4f} "
              f"ssim={row['ssim']:.4f} psnr={row['psnr']:.2f} lpips={row['lpips']:.4f} | mean image A={row['A_mean_image']:.4f} "
              f"| {row['seconds']}s", flush=True)
    # phase 2: cross matrices A(v, w) and A_lab(v, w) per group (reuses the SSIM-to-pool vectors: no new SSIM)
    fields = GROUP_FIELDS if a.scope == "directory" else CELL_FIELDS
    groups, cross = {}, []
    for k, e in enumerate(done):
        groups.setdefault(tuple(e["row"][f] for f in fields), []).append(k)
    nan = float("nan")
    for gid, (key, ks) in enumerate(groups.items()):
        ks = sorted(ks, key=lambda k: (str(done[k]["row"]["seed"]), done[k]["row"]["victim"]))
        vids = [(str(done[k]["row"]["seed"]), done[k]["row"]["victim"]) for k in ks]
        if len(set(vids)) != len(vids):
            dup = sorted({v for v in vids if vids.count(v) > 1})
            raise SystemExit(f"group {dict(zip(fields, key))}: the same (seed, victim) appears twice: {dup} "
                             f"(files {[done[k]['row']['dir'] + '/' + done[k]['row']['file'] for k in ks if vids[ks.index(k)] in dup]})")
        gfields = dict(zip(fields, key))
        if a.scope == "cell":
            gfields = {"dir": ";".join(sorted({done[k]["row"]["dir"] for k in ks})), **gfields}
        for k in ks:
            e, per, per_lab = done[k], [], []
            for p in e["pairs"]:
                p["group"] = gid
            for m in ks:
                w = done[m]
                assert len(w["svecs"]) == len(e["idx"]), f"batch size differs: {w['row']['file']} vs {e['row']['file']}"
                rk = max_rule(w["svecs"], e["idx"])
                rk_lab = max_rule(w["svecs"], e["idx"], e["pool_labels"])
                sm, sm_lab = summarize(rk), summarize(rk_lab, "_lab")
                cross.append({"group": gid, **gfields, "v_seed": e["row"]["seed"], "v_victim": e["row"]["victim"],
                              "v_file": e["row"]["file"], "v_dir": e["row"]["dir"], "v_n_abnormal": e["row"]["n_abnormal"],
                              "w_seed": w["row"]["seed"], "w_victim": w["row"]["victim"], "w_file": w["row"]["file"],
                              "w_dir": w["row"]["dir"], "w_n_abnormal": w["row"]["n_abnormal"], "diagonal": m == k,
                              **sm, **sm_lab, "scope": a.scope})
                if m == k:
                    continue
                per.append(sm["A"])
                per_lab.append(sm_lab["A_lab"])
                for i, ((a_, r_, n_), (al_, rl_, nl_)) in enumerate(zip(rk, rk_lab)):
                    e["pairs"].append({"file": e["row"]["file"], "seed": e["row"]["seed"], "victim": e["row"]["victim"],
                                       "kind": "mismatched", "pair": i, "study": e["studies"][i], "label": e["labels"][i],
                                       "label_dataset": int(e["lab_ds"][i]), "pool_index": e["idx"][i],
                                       "rank": r_, "N": n_, "A": a_, "rank_lab": rl_, "N_lab": nl_, "A_lab": al_,
                                       "rec_batch": len(w["lab_batch"]), "rec_n_abnormal": w["row"]["n_abnormal"],
                                       "x_share": float(np.mean(w["lab_batch"] == e["lab_ds"][i])),
                                       "partner_file": w["row"]["file"], "partner_seed": w["row"]["seed"],
                                       "partner_victim": w["row"]["victim"], "partner_dir": w["row"]["dir"], "group": gid})
            e["row"].update(group=gid, scope=a.scope, n_partners=len(per),
                            A_mismatched=float(np.mean(per)) if per else nan,
                            A_mismatched_sd=float(np.std(per, ddof=1)) if len(per) > 1 else nan,
                            A_lab_mismatched=float(np.mean(per_lab)) if per_lab else nan,
                            A_lab_mismatched_sd=float(np.std(per_lab, ddof=1)) if len(per_lab) > 1 else nan)
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stem = os.path.splitext(out)[0]
    write_csv(out, [e["row"] for e in done])
    write_csv(stem + "_pairs.csv", [p for e in done for p in e["pairs"]])
    write_csv(stem + "_cross.csv", cross)
    print(f"wrote {out} ({len(done)} files), {stem}_pairs.csv, {stem}_cross.csv in {time.time() - t_all:.0f}s")
    for key, ks in groups.items():
        rows = [done[k]["row"] for k in ks]
        many = len(rows) > 1
        print(f"group {dict(zip(fields, key))}: n={len(rows)} mean A={np.mean([r['A'] for r in rows]):.4f} "
              f"A_mismatched={np.nanmean([r['A_mismatched'] for r in rows]) if many else float('nan'):.4f} "
              f"A_lab={np.mean([r['A_lab'] for r in rows]):.4f} "
              f"A_lab_mismatched={np.nanmean([r['A_lab_mismatched'] for r in rows]) if many else float('nan'):.4f} "
              f"A_matched={np.mean([r['A_matched'] for r in rows]):.4f} A_mean_image={np.mean([r['A_mean_image'] for r in rows]):.4f}"
              f" n_abnormal={[r['n_abnormal'] for r in rows]}")


if __name__ == "__main__":
    main()
