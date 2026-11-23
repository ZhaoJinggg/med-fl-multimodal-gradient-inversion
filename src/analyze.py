"""Decision rules of the paper's hypothesis table (Table "Hypothesis Testing Framework and Decision Rules",
Sections "Evaluation Metrics" and "Hypothesis Testing") and the calibration gate.

INPUT
  Tidy CSVs, one row per victim, cell and modality:
    seed, victim, protocol, u, modality, value          (required)
    batch                                               (optional; default 8)
    condition                                           (optional; recovered | oracle | none; default recovered)
    m0                                                  (optional; the victim's zero-point score in that cell)
  - a CELL is (protocol, u, batch, condition); victims are paired by (seed, victim);
    protocol: fedavg | fedprox | scaffold | fedbn;
  - a batch size written into the condition (the rows of src/score_text.py: "B4", "exact_B4") is moved to
    the batch column ("B4" -> recovered at batch 4); it must agree with a batch column if both are given;
  - any other condition ("exact", "optI", ...) is a cell of its own that enters no test;
  - modality "image" (value = identification score A) and "text" (value = ROUGE-L) are primary. Any other
    modality gets the directional comparisons of its family only ("text..." H1; "image..." H5a and H6),
    "reported, decides nothing";
  - m0: image = A_mismatched of src/score_images.py (the victim's originals against the reconstructions of
    every other victim of the cell); text = column m0 of src/score_text.py (the B most frequent findings
    reports of the public pairs, matched and scored like the attack).
  Cross matrices (--cross, the *_cross.csv of src/score_images.py --scope cell): A(v, w) of v's originals
  against w's reconstructions, per cell, with v_n_abnormal (the label count of v's attacked batch; the strata
  of the stratified test). Needed for every image floor test.
  `python src/analyze.py tidy-images scores.csv` converts a src/score_images.py CSV (value = A, m0 =
  A_mismatched, batch and condition as scored).

RULES IMPLEMENTED (the paper's closed design of 2026-11-23: H1, H2a, H2b, H3, H4, H5a, H5b, H6)
  Reference cells: text FedAvg, u = 1, batch 8; image FedAvg, u = 1, batch 2, recovered text.
  f(c, v) = (m(c, v) - m_0) / (mean_v m(ref, v) - m_0): m_0 = mean of m0 over the victims of the reference
     cell (--m0 image=0.5 overrides it); one denominator per modality. f of a modality is computed only if
     its reference cell is above the floor; otherwise the hypotheses that use f are "not decidable" for the
     modality (H2a, H2b for the images; H3, H4, H5b for the text). H1, H5a and H6 use the primary metric and
     do not depend on the reference cell.
  Floor test of a cell (one-sided, alpha = 0.05): images, the stratified permutation test of the cell's
     cross matrix (below); text, a one-sided Wilcoxon signed-rank test of the victims' ROUGE-L against their
     m0 in that cell. A cell that cannot be tested (no cross matrix, no m0) is treated as not above the
     floor and the output says so; no point rule replaces the test.
  Comparisons (differences are treatment - reference, paired by victim):
     H1   text ROUGE-L, FedAvg, batch 8: u = 2, 4, 8, 16 against u = 1; one-sided Wilcoxon (decrease);
          supported if all 4 are significant, otherwise not supported. No floor rule.
     H2a  image f, FedAvg, u = 1, batch 2: recovered - oracle; two-sided 95% t interval against +-0.05.
     H2b  image f, FedAvg, u = 1, batch 2: recovered - none; one-sided Wilcoxon (gain); supported if
          significant; if not and the 95% t interval lies within +-0.05: "text does not matter within the
          margin"; otherwise inconclusive.
     H3   text f, batch 8, u = 4, 8: SCAFFOLD - FedAvg; two-sided 95% t intervals against +-0.05.
     H4   text f, batch 8, u = 4, 8: FedProx - FedAvg; supported if every one-sided 95% t lower bound is at
          or above -0.05; contradicted if a two-sided 95% interval lies entirely below -0.05; else inconclusive.
     H5a  image A, u = 1, recovered, batch 1, 2: FedBN - FedAvg; one-sided Wilcoxon (decrease);
          supported if both are significant, otherwise not supported.
     H5b  text f, batch 8, u = 4, 8: FedBN - FedAvg; two-sided 95% t intervals against +-0.05.
     H6   image A, FedAvg, u = 1, recovered: batch 2, 4, 8 against batch 1; one-sided Wilcoxon (decrease);
          supported if all 3 are significant, otherwise not supported. No floor rule.
  Floor rule (H2a to H5b): a comparison is decidable only if the FedAvg cell that serves as its control
     (recovered condition, same modality, u and batch) is above the floor. A comparison whose control cell
     is at the floor is "not decidable" and leaves its family; the hypothesis is decided on the
     remaining comparisons ("not decidable" if none remains).
  Multiplicity: every Wilcoxon test is run at alpha = 0.05 without adjustment. H1, H5a and H6 are supported
     only if all their comparisons are significant (intersection-union rule), which keeps the probability
     of a false "supported" at or below alpha. The columns p_holm and mixed_p_holm are reported for
     information and decide nothing.
  Interval rules (H2a, H3, H5b): supported if every interval lies within the margin; contradicted if at
     least one lies entirely outside it; otherwise inconclusive.
  Zero differences: Wilcoxon zero_method="wilcox" (all zero -> p = 1); a zero-width interval is [mean, mean].
  Margins are inclusive. Every comparison also gets the per-seed means and the mixed-model sensitivity
  analysis d ~ 1 + (1 | seed) (statsmodels MixedLM, REML, Wald intervals, one-sided p for directional tests,
  unadjusted like the main tests; column mixed_agrees). ANOVA: text f ~ C(protocol) * C(u) + C(victim) on the
  protocol experiment (batch 8, the u levels at which the other protocols were run), type II; decides nothing.
  Output cells.csv: n, mean, SD, mean m0, mean f and the floor test of every cell of a primary modality.

PERMUTATION TESTS
  Permutation test of a cross matrix M (n victims): statistic T = mean_v M[v, v]; null = mean_v M[v, pi(v)]
     over all permutations pi (exact for n <= 8, else 100,000 Monte Carlo draws with seed 0 plus the
     identity); p = share of permutations with a statistic >= T (identity included).
  LABEL CONTROLS. The attacker is given the batch labels and every batch is reconstructed with its own
     labels, so a reconstruction that reflects only the labels beats the unstratified permutation test. Hence:
     - Stratified permutation test (stratified_perm_test): only permutations within strata of victims with
       the same number of abnormal pairs in their attacked batch (v_n_abnormal of the cross file); exact if
       there are <= 1,000,000 such permutations (the 20 test victims at batch 8: 552,960, so every batch-8
       cell is enumerated and its p does not depend on a seed), else 100,000 Monte Carlo draws (seed 0) plus
       the identity (batch sizes 4, 2, 1).
       Reported: number of permutations, smallest attainable p (1 / number evaluated), p, and the effect size
       mean over victims in strata of size >= 2 of [A(v, v) - mean within-stratum off-diagonal A(v, w)] with
       the number of contributing victims. With few victims the test may be unable to reject at all (the 7
       gate clients of seed 0, label counts 0, 0, 8, 8, 8, 3, 6, have 2! 3! = 12 permutations: p >= 1/12).
     - It is THE image floor test of `test` (the unstratified p is kept as floor_p_unstratified /
       ref_test_p_unstratified; --no-stratify switches back, for comparison only; a cross file without
       v_n_abnormal falls back to the unstratified test and floor_method says so).
     - `gate` reports, per group: the unstratified test (informative, the paper's gate definition), the
       stratified test of A (p_strat, ..., informative_strat), both tests on the same-label score A_lab
       (*_lab; E[A_lab] = 0.5 for every (v, w) under label information only), the one-sided sign-flip test
       (exact for n <= 20) and Wilcoxon test of D_v = A_lab(v, v) - mean_{w != v} A_lab(v, w), and with
       --regression the label-distance regression (label_regression): A_i(v, w) on the share x of w's batch
       with the label of original i, fixed effect per original, fitted on the diagonal and the mismatched
       rows (reg_*; reg_offdiag_*: fitted on the mismatched rows only); slope = size of the label channel,
       excess = mean over victims of (diagonal value - fitted value at its own share), 95% percentile
       bootstrap over victims (2,000 resamples, seed 0).

GATE (calibration, command `gate`): per group of a score_images.py cross file, restricted to --victims
  (and --seed) BEFORE the control is computed: informative = permutation-test p <= 0.05; strongly_informative
  = informative and mean A >= 0.90; also the share of originals ranked first, the control (mean over
  w != v within the restricted group) and, for the record, the old one-sided Wilcoxon of A against it;
  plus the label controls above (stratified, A_lab, sign-flip, --regression).

READINGS (most literal reading implemented)
  1. The floor test is on the FedAvg control cell only, even when the treatment cell is itself at chance.
  2. H2a and H2b have the image reference cell as their control cell, so one test serves both conditions.
  3. A hypothesis for which fewer comparisons are present than the paper plans is decided on those present
     and the reason says so.
  4. H2b's third outcome is output verbatim.
  5. `gate`: "informative" keeps the paper's (unstratified) definition; the label controls are extra columns.
     At the gate (7 victims) the stratified test cannot reject at batch 8 (12 permutations); the sign-flip
     test of A_lab (p >= 1/128) can.
  6. Regression fitted WITH the diagonal (as specified): a true image effect raises the diagonal, whose own
     label share is the highest of its row, so part of it is absorbed into the slope and the excess shrinks
     (synthetic, 7 victims: slope 0.327 / excess 0.082 for true 0.30 / 0.10); reg_offdiag_* (fit on w != v
     only, diagonal predicted out of sample) recovers both and is reported next to it.
  7. Stratified effect size: per victim, A(v, v) minus the mean of its within-stratum row entries; then the
     mean over contributing victims (not the mean of all within-stratum entries pooled).

usage:
  python src/analyze.py test tidy.csv [more.csv] --cross scores_cross.csv --out runs/analysis [--m0 image=0.5]
         [--no-stratify]
  python src/analyze.py tidy-images runs/scores/images.csv --out tidy_images.csv [--condition recovered]
  python src/analyze.py gate runs/scores/images.csv [--cross ..._cross.csv] [--victims 3 4 5 6 7 8 9] [--seed 0]
         [--regression [--pairs ..._pairs.csv] [--reg-score A]] [--no-stratify] [--out gate.csv]
"""
import argparse
import collections
import itertools
import math
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ALPHA, MARGIN, GATE_A = 0.05, 0.05, 0.90
MAIN_BATCH = 8
PRIMARY = ("image", "text")
# cells are (protocol, u, batch, condition)
REF = {"text": ("fedavg", 1, 8, "recovered"), "image": ("fedavg", 1, 2, "recovered")}
H1_U, PROTO_U, H2_B, H5A_B, H6_B = (2, 4, 8, 16), (4, 8), 2, (1, 2), (2, 4, 8)
PLANNED = {"H1": 4, "H2a": 1, "H2b": 1, "H3": 2, "H4": 2, "H5a": 2, "H5b": 2, "H6": 3}
COND_ALIASES = {"": "recovered", "nan": "recovered", "recovered": "recovered", "rec": "recovered",
                "oracle": "oracle", "none": "none", "notext": "none", "no_text": "none", "no-text": "none"}


# ---------------------------------------------------------------- statistics
def t_interval(d, conf=0.95):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return m, m
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return m - h, m + h


def t_lower(d, conf=0.95):
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return np.nan
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    return m if se == 0 else m - stats.t.ppf(conf, n - 1) * se


def wilcoxon(d, alternative):
    """Signed-rank test; zero differences dropped (zero_method='wilcox'); all zero -> p = 1."""
    d = np.asarray(d, float)
    nz = int((d != 0).sum())
    if nz == 0:
        return np.nan, 1.0, 0
    r = stats.wilcoxon(d, zero_method="wilcox", alternative=alternative)
    return float(r.statistic), float(r.pvalue), nz


def holm(p):
    p = np.asarray(p, float)
    adj = np.full(len(p), np.nan)
    ok = np.where(~np.isnan(p))[0]
    m, run = len(ok), 0.0
    for k, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def perm_test(M, n_mc=100_000, seed=0, exact_max=8):
    """Permutation test of a cross matrix M[v, w] (rows: originals of v, columns: reconstructions of w).
    Returns (T, p, n_permutations, exact)."""
    M = np.asarray(M, float)
    n = len(M)
    if n == 0:
        return np.nan, np.nan, 0, True
    T = float(np.trace(M) / n)
    rows = np.arange(n)
    if n <= exact_max:
        P = np.array(list(itertools.permutations(range(n))))
        null = M[rows, P].mean(1)
        exact = True
    else:
        g = np.random.default_rng(seed)
        P = np.argsort(g.random((n_mc, n)), axis=1)
        null = np.concatenate([[T], M[rows, P].mean(1)])  # identity included
        exact = False
    p = float(np.mean(null >= T - 1e-12))
    return T, p, len(null), exact


def _strata_groups(strata):
    strata = np.asarray(strata)
    return [np.where(strata == s)[0] for s in sorted(set(strata.tolist()), key=str)]


def n_stratified_perms(strata):
    """Number of permutations that keep every victim within its stratum: prod over strata of size!."""
    return int(np.prod([math.factorial(len(g)) for g in _strata_groups(strata)], dtype=object))


def strata_label(strata):
    """'k:count' per stratum, e.g. '0:2,3:1,6:1,8:3'."""
    c = collections.Counter(np.asarray(strata).tolist())
    return ",".join(f"{k}:{c[k]}" for k in sorted(c, key=lambda x: (str(type(x)), x)))


def stratified_perm_test(M, strata, n_mc=100_000, seed=0, exact_max=1_000_000):
    """Permutation test of a cross matrix M[v, w] with the permutations restricted to victims of the same
    stratum (here: the number of abnormal pairs of the attacked batch), so that a reconstruction batch is only
    ever compared with batches carrying the same label information. Statistic T = mean_v M[v, v].
    Exact enumeration of all stratified permutations if their number is <= exact_max, else n_mc Monte Carlo
    draws (seed `seed`) plus the identity. p = share of evaluated permutations with a statistic >= T.
    Victims alone in their stratum are fixed points of every permutation: they shift T and the null alike and
    do not affect p. Effect size = mean over the victims of a stratum of size >= 2 (n_contrib of them) of
    M[v, v] - mean_{w in stratum(v), w != v} M[v, w] (mean diagonal minus mean within-stratum off-diagonal,
    row by row). Returns a dict: T, p, n_perm (number of stratified permutations), n_eval (evaluated),
    exact, p_min (= 1 / n_eval, the smallest p the test can return), effect, n_contrib, strata."""
    M = np.asarray(M, float)
    n = len(M)
    strata = np.asarray(strata)
    assert len(strata) == n, "one stratum per victim"
    groups = _strata_groups(strata)
    total = n_stratified_perms(strata)
    T = float(np.trace(M) / n) if n else np.nan
    rows = np.arange(n)
    if total <= exact_max:
        per = [list(itertools.permutations(g.tolist())) for g in groups]
        P = np.tile(rows, (total, 1))
        for r, combo in enumerate(itertools.product(*per)):
            for g, perm in zip(groups, combo):
                P[r, g] = perm
        null = M[rows, P].mean(1)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        keys = rng.random((n_mc, n))
        P = np.tile(rows, (n_mc, 1))
        for g in groups:
            if len(g) > 1:
                P[:, g] = g[np.argsort(keys[:, g], axis=1)]
        null = np.concatenate([[T], M[rows, P].mean(1)])  # identity included
        exact = False
    p = float(np.mean(null >= T - 1e-12))
    D = []
    for g in groups:
        if len(g) < 2:
            continue
        for v in g:
            others = [w for w in g if w != v]
            D.append(M[v, v] - M[v, others].mean())
    return {"T": T, "p": p, "n_perm": total, "n_eval": len(null), "exact": exact, "p_min": 1.0 / len(null),
            "effect": float(np.mean(D)) if D else np.nan, "n_contrib": len(D), "strata": strata_label(strata)}


def sign_flip_test(D, n_mc=100_000, seed=0, exact_max=20):
    """One-sided sign-flip (randomization) test of mean(D) > 0: null = mean(s * D) over sign vectors s in
    {-1, +1}^n, all 2^n of them for n <= exact_max (identity included), else n_mc Monte Carlo draws plus the
    identity. Returns (mean D, p, n_eval, exact)."""
    D = np.asarray(D, float)
    D = D[~np.isnan(D)]
    n = len(D)
    if n == 0:
        return np.nan, np.nan, 0, True
    T = D.sum()
    tol = 1e-12 * max(1.0, np.abs(D).sum())
    if n <= exact_max:
        total, hits, bits = 2 ** n, 0, np.arange(n)
        for s in range(0, total, 1 << 16):
            k = np.arange(s, min(total, s + (1 << 16)))
            signs = 1 - 2 * ((k[:, None] >> bits) & 1)
            hits += int(((signs @ D) >= T - tol).sum())
        return float(D.mean()), hits / total, total, True
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_mc, n))
    null = np.concatenate([[T], signs @ D])
    return float(D.mean()), float(np.mean(null >= T - tol)), len(null), False


def label_regression(pairs, score="A", include_diagonal=True, n_boot=2000, seed=0, conf=0.95):
    """Label-distance regression (third label control) on the pairs rows of ONE group (src/score_images.py
    *_pairs.csv, kinds 'attack' = diagonal w = v and 'mismatched' = w != v; columns seed, victim, pair, kind,
    x_share, `score`). Model: A_i(v, w) = alpha_i + beta * x_iw + e, x_iw = share of w's reconstruction batch
    that has the label of original i, alpha_i a fixed effect per original (within estimator). Fitted on the
    diagonal and the mismatched rows (include_diagonal=True, as specified) or on the mismatched rows only.
    beta = the size of the label channel (score per unit of label share). excess = mean over victims of the
    mean over their originals of (diagonal value - alpha_i - beta * x_ii), what the own reconstructions give
    beyond what their label share predicts. Interval: percentile bootstrap over victims (n_boot resamples,
    seed), refitting beta in every resample (alpha_i stays the within mean of each original).
    Returns a dict (NaN entries if there is no variation in x)."""
    p = pairs[pairs.kind.isin(["attack", "mismatched"])].dropna(subset=[score, "x_share"]).copy()
    out = {"score": score, "include_diagonal": include_diagonal, "n_rows": len(p)}
    if p.empty:
        return {**out, "slope": np.nan, "excess": np.nan}
    p["vk"] = p["seed"].astype(str) + ":" + p["victim"].astype(str)
    p["ok"] = p["vk"] + ":" + p["pair"].astype(str)
    fit = p if include_diagonal else p[p.kind == "mismatched"]
    mx, my = fit.groupby("ok")["x_share"].mean(), fit.groupby("ok")[score].mean()
    fit = fit.assign(xc=fit["x_share"] - fit["ok"].map(mx), yc=fit[score] - fit["ok"].map(my))
    by_v = fit.assign(sxy=fit.xc * fit.yc, sxx=fit.xc ** 2).groupby("vk")[["sxy", "sxx"]].sum()
    dg = p[p.kind == "attack"].copy()
    dg = dg[dg["ok"].isin(mx.index)]
    dg["xc"], dg["yc"] = dg["x_share"] - dg["ok"].map(mx), dg[score] - dg["ok"].map(my)
    dv = dg.groupby("vk")[["xc", "yc"]].mean().reindex(by_v.index)  # per-victim mean over its originals
    keep = dv.notna().all(axis=1)
    by_v, dv = by_v[keep], dv[keep]
    out.update(n_victims=len(by_v), n_originals=int(dg["ok"].nunique()), mean_diag=float(dg[score].mean()),
               mean_x_diag=float(dg["x_share"].mean()), mean_x_mismatched=float(p[p.kind == "mismatched"]["x_share"].mean()))
    if by_v.sxx.sum() <= 0:
        return {**out, "slope": np.nan, "excess": np.nan, "note": "no variation of x_share within originals"}

    def est(w):
        b = (w * by_v.sxy).sum() / (w * by_v.sxx).sum()
        return b, (w * (dv.yc - b * dv.xc)).sum() / w.sum()

    b, ex = est(np.ones(len(by_v)))
    rng = np.random.default_rng(seed)
    nv = len(by_v)
    bs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(nv, size=nv), minlength=nv).astype(float)
        if (w * by_v.sxx.to_numpy()).sum() > 0:
            bs.append(est(w))
    bs = np.array(bs)
    q = [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100]
    out.update(slope=float(b), excess=float(ex), mean_fitted_diag=float(dg[score].mean() - ex) if nv else np.nan,
               slope_lo=float(np.percentile(bs[:, 0], q[0])), slope_hi=float(np.percentile(bs[:, 0], q[1])),
               excess_lo=float(np.percentile(bs[:, 1], q[0])), excess_hi=float(np.percentile(bs[:, 1], q[1])),
               n_boot=len(bs))
    return out


def mixed(d, seeds):
    """d ~ 1 + (1 | seed): returns estimate, se, two-sided p, note."""
    d = np.asarray(d, float)
    if len(set(seeds)) < 2 or np.allclose(d, d[0]):
        return np.nan, np.nan, np.nan, "skipped (fewer than 2 seeds or constant differences)"
    import statsmodels.formula.api as smf
    df = pd.DataFrame({"d": d, "seed": [str(s) for s in seeds]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm("d ~ 1", df, groups=df["seed"]).fit(reml=True)
        except Exception as e:  # singular fits etc.
            return np.nan, np.nan, np.nan, f"failed: {e}"
    note = "; ".join(sorted({str(x.message).split("\n")[0][:80] for x in w}))
    return float(fit.fe_params["Intercept"]), float(fit.bse_fe["Intercept"]), float(fit.pvalues["Intercept"]), note


def _control(M):
    n = len(M)
    return np.array([(M[v].sum() - M[v, v]) / (n - 1) for v in range(n)]) if n > 1 else np.full(n, np.nan)


def _strat_cols(M, strata, suffix):
    if strata is None:
        return {}
    r = stratified_perm_test(M, strata)
    return {f"p_strat{suffix}": r["p"], f"n_perm_strat{suffix}": r["n_perm"], f"n_eval_strat{suffix}": r["n_eval"],
            f"exact_strat{suffix}": r["exact"], f"p_min_strat{suffix}": r["p_min"], f"effect_strat{suffix}": r["effect"],
            f"n_contrib_strat{suffix}": r["n_contrib"]}


def gate(M, top1=None, alpha=ALPHA, threshold=GATE_A, strata=None, M_lab=None):
    """Calibration gate on a cross matrix M[v, w] (diagonal = real scores) of the gate clients only.
    informative = permutation-test p <= alpha (unstratified, as in the paper); strongly_informative =
    informative and mean A >= threshold. The old one-sided Wilcoxon of A against the control (mean over
    w != v) is returned for the record.
    Label controls (second review): with `strata` (n_abnormal per victim) the stratified permutation test of
    A (columns *_strat, informative_strat); with `M_lab` (the A_lab cross matrix) its unstratified and
    stratified permutation tests (*_lab), and the sign-flip and Wilcoxon tests of D_v = A_lab(v, v) - mean_w
    A_lab(v, w) (one-sided, mean_D_lab, p_signflip_lab, p_wilcoxon_lab)."""
    M = np.asarray(M, float)
    n = len(M)
    A = np.diag(M)
    ctrl = _control(M)
    T, p, nperm, exact = perm_test(M)
    _, pw, nz = wilcoxon(A - ctrl, "greater") if n > 1 else (np.nan, np.nan, 0)
    informative = bool(p <= alpha)
    out = {"n": n, "mean_A": float(A.mean()) if n else np.nan, "mean_control": float(np.nanmean(ctrl)) if n > 1 else np.nan,
           "top1_share": float(np.mean(top1)) if top1 is not None and len(top1) else np.nan,
           "p_perm": p, "n_perm": nperm, "exact": exact, "informative": informative,
           "strongly_informative": bool(informative and A.mean() >= threshold), "p_wilcoxon_old": pw}
    if strata is not None:
        out["strata"] = strata_label(strata)
        out.update(_strat_cols(M, strata, ""))
        out["informative_strat"] = bool(out["p_strat"] <= alpha)
    if M_lab is not None:
        L = np.asarray(M_lab, float)
        Dl = np.diag(L) - _control(L)
        _, pl, npl, exl = perm_test(L)
        out.update(mean_A_lab=float(np.diag(L).mean()), mean_control_lab=float(np.nanmean(_control(L))) if n > 1 else np.nan,
                   p_perm_lab=pl, n_perm_lab=npl)
        out.update(_strat_cols(L, strata, "_lab"))
        if n > 1:
            mD, psf, nsf, exsf = sign_flip_test(Dl)
            _, pwl, _ = wilcoxon(Dl, "greater")
            out.update(mean_D_lab=mD, p_signflip_lab=psf, n_signflip_lab=nsf, exact_signflip_lab=exsf, p_wilcoxon_lab=pwl)
    return out


def cell_perm_test(M, strata, stratify=True):
    """The image floor / reference-cell test of `test`: the stratified permutation test (strata = n_abnormal)
    unless --no-stratify or the cross file has no n_abnormal; the unstratified p is always reported."""
    _, p_u, _, _ = perm_test(M)
    out = {"p_unstratified": p_u}
    if stratify and strata is not None:
        r = stratified_perm_test(M, strata)
        out.update(p=r["p"], n_perm=r["n_perm"], p_min=r["p_min"], effect=r["effect"], n_contrib=r["n_contrib"],
                   method=f"stratified permutation test (n={len(M)}, {r['n_perm']} permutations"
                          f"{'' if r['exact'] else ', Monte Carlo'}, p_min={r['p_min']:.3g}, strata {r['strata']})")
    else:
        why = "--no-stratify" if not stratify else "no n_abnormal in the cross file"
        out.update(p=p_u, method=f"permutation test (n={len(M)}; unstratified: {why})")
    return out


# ---------------------------------------------------------------- data
_BTOK = re.compile(r"(?:^|_)b(\d+)(?=_|$)")


def _norm_ids(df, cols):
    for c in cols:  # 0, 0.0 and "0" must pair: integers where possible
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all() and (num == num.round()).all():
            df[c] = num.astype(int)
    return df


def split_batch(cond):
    """Batch part of a lower-case condition label: 'b4' -> ('', 4); 'oracle_b2' -> ('oracle', 2);
    'no_text' -> ('no_text', None)."""
    m = _BTOK.search(cond)
    if not m:
        return cond, None
    return (cond[:m.start()] + cond[m.end():]).strip("_"), int(m.group(1))


def _norm_cell(df):
    parts = [split_batch(c) for c in df["condition"].fillna("").astype(str).str.strip().str.lower()]
    tok = pd.Series([np.nan if b is None else b for _, b in parts], index=df.index, dtype=float)
    col = pd.to_numeric(df["batch"], errors="coerce") if "batch" in df else pd.Series(np.nan, index=df.index)
    clash = tok.notna() & col.notna() & (tok != col)
    if clash.any():
        raise SystemExit(f"{int(clash.sum())} rows whose batch column and condition label disagree")
    df["batch"] = col.fillna(tok).fillna(MAIN_BATCH).astype(int)
    df["condition"] = [COND_ALIASES.get(c, c) for c, _ in parts]
    df["protocol"] = df["protocol"].fillna("").astype(str).str.strip().str.lower().replace({"": "fedavg", "nan": "fedavg"})
    df["u"] = df["u"].astype(int)
    return df


def load_tidy(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    need = {"seed", "victim", "protocol", "u", "modality", "value"}
    if need - set(df.columns):
        raise SystemExit(f"missing columns: {sorted(need - set(df.columns))}")
    if "condition" not in df:
        df["condition"] = "recovered"
    df = _norm_cell(df)
    df["modality"] = df["modality"].astype(str).str.strip().str.lower()
    df = _norm_ids(df, ("seed", "victim"))
    df["key"] = df["seed"].astype(str) + ":" + df["victim"].astype(str)
    dup = df.duplicated(["key", "protocol", "u", "batch", "condition", "modality"])
    if dup.any():
        raise SystemExit(f"{dup.sum()} duplicate rows (same victim, cell and modality)")
    return df


def load_cross(paths):
    if not paths:
        return None
    c = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    c = _norm_cell(c)
    c = _norm_ids(c, ("v_seed", "v_victim", "w_seed", "w_victim"))
    c["v_key"] = c["v_seed"].astype(str) + ":" + c["v_victim"].astype(str)
    c["w_key"] = c["w_seed"].astype(str) + ":" + c["w_victim"].astype(str)
    return c


def cross_matrix_strata(cross, keys, protocol="fedavg", u=1, batch=MAIN_BATCH, condition="recovered", value="A"):
    """Square A(v, w) of one cell over the victims in `keys` that have a complete row and column, with the
    strata of the stratified test (v_n_abnormal per victim; None if the cross file has no such column).
    Returns (M, keys, strata); M is None if the cell has no cross matrix."""
    if cross is None:
        return None, [], None
    c = cross[(cross.protocol == protocol) & (cross.u == u) & (cross.batch == batch) & (cross.condition == condition)]
    if "group" in c and c.group.nunique() > 1:
        raise SystemExit(f"cross file has {c.group.nunique()} groups for cell {protocol} u={u} B={batch} {condition}: "
                         "score the cell as one group (score_images.py --scope cell)")
    keys = [k for k in keys if k in set(c.v_key)]
    c = c[c.v_key.isin(keys) & c.w_key.isin(keys)]
    if c.empty:
        return None, [], None
    M = c.pivot_table(index="v_key", columns="w_key", values=value, aggfunc="mean").reindex(index=keys, columns=keys)
    full = [k for k in keys if M.loc[k].notna().all() and M[k].notna().all()]
    if len(full) < len(keys):
        M = M.loc[full, full]
    strata = None
    if "v_n_abnormal" in c and len(full) and c["v_n_abnormal"].notna().all():
        na = c.groupby("v_key")["v_n_abnormal"].agg(lambda s: set(s.astype(int)))
        if any(len(s) != 1 for s in na):
            raise SystemExit("cross file: a victim has more than one n_abnormal in one cell")
        strata = np.array([next(iter(na[k])) for k in full])
    return (M.to_numpy() if len(full) else None), full, strata


def cell(df, mod, protocol="fedavg", u=1, batch=MAIN_BATCH, condition="recovered"):
    return df[(df.modality == mod) & (df.protocol == protocol) & (df.u == u) & (df.batch == batch)
              & (df.condition == condition)]


def cell_name(c):
    proto, u, batch, cond = c
    return f"{proto} u={u} B={batch}" + ("" if cond == "recovered" else f" {cond}")


def floor_test(df, cross, mod, c, m0_override=None, stratify=True, cache=None):
    """Is the cell c = (protocol, u, batch, condition) of a primary modality above the floor?
    Images: (stratified) permutation test of the cell's cross matrix. Text: one-sided Wilcoxon of the
    victims' ROUGE-L against their m0 in the cell (or the --m0 override).
    Returns (above, p, method, extra): above is None if the test cannot be run (method says why); extra holds
    the unstratified p and the details of the stratified test for images."""
    ck = (mod, c, stratify)
    if cache is not None and ck in cache:
        return cache[ck]
    g = cell(df, mod, *c)
    if g.empty:
        out = (None, np.nan, "not tested: no such cell", {})
    elif mod == "image":
        M, keys, strata = cross_matrix_strata(cross, list(g.key), *c)
        if M is None or len(keys) < 2:
            out = (None, np.nan, "not tested: no cross matrix for the cell", {})
        else:
            r = cell_perm_test(M, strata, stratify)
            out = (bool(r["p"] <= ALPHA), r["p"], r["method"], r)
    else:
        if m0_override and mod in m0_override:
            m0 = np.full(len(g), float(m0_override[mod]))
        else:
            m0 = g["m0"].to_numpy(float) if "m0" in g else np.full(len(g), np.nan)
        ok = ~np.isnan(m0)
        if ok.sum() < 2:
            out = (None, np.nan, "not tested: no m0 in the cell", {})
        else:
            _, p, _ = wilcoxon(g.value.to_numpy(float)[ok] - m0[ok], "greater")
            out = (bool(p <= ALPHA), p, f"Wilcoxon vs m0 (n={int(ok.sum())})", {})
    if cache is not None:
        cache[ck] = out
    return out


def add_f(df, m0_override=None, cross=None, stratify=True, cache=None):
    """Adds column f (primary modalities); returns (df, info per modality). f of a modality is computed only
    if its reference cell (REF) is above the floor; info[mod]['ref_passes'] is True, False (at the floor) or
    None (the reference cell could not be tested); f stays NaN unless it is True."""
    m0_override = m0_override or {}
    df = df.copy()
    df["f"] = np.nan
    info = {}
    for mod in PRIMARY:
        if not (df.modality == mod).any():
            continue
        ref = cell(df, mod, *REF[mod])
        inf = {"ref_cell": cell_name(REF[mod])}
        if ref.empty:
            info[mod] = {**inf, "ref_passes": None, "note": "no reference cell in the data: f undefined"}
            continue
        above, p, how, x = floor_test(df, cross, mod, REF[mod], m0_override, stratify, cache)
        inf.update(ref_test_p=p, ref_test_method=how, ref_passes=above, n_ref=len(ref))
        if "p_unstratified" in x:
            inf["ref_test_p_unstratified"] = x["p_unstratified"]
        if "effect" in x:
            inf.update(ref_test_p_min=x["p_min"], ref_effect_within_stratum=x["effect"], ref_n_contrib=x["n_contrib"])
        if above is not True:
            why = "is at the floor" if above is False else "could not be tested"
            info[mod] = {**inf, "note": f"reference cell {why}: {mod} f not computed, its hypotheses on f are not decidable"}
            continue
        if mod in m0_override:
            m0, src = float(m0_override[mod]), "cli"
        elif "m0" in df and ref["m0"].notna().any():
            m0, src = float(ref["m0"].mean()), "mean m0 of the reference cell"
        else:
            info[mod] = {**inf, "ref_passes": None, "note": "no m0 (column m0 or --m0): f undefined"}
            continue
        den = float(ref["value"].mean()) - m0
        sel = df.modality == mod
        df.loc[sel, "f"] = (df.loc[sel, "value"] - m0) / den
        info[mod] = {**inf, "m0": m0, "m0_source": src, "ref_mean": den + m0, "denominator": den}
    return df, info


def paired(a, b, col):
    m = a[["key", "seed", col]].merge(b[["key", col]], on="key", suffixes=("_t", "_r"))
    m = m.dropna(subset=[col + "_t", col + "_r"])
    return m.assign(d=m[col + "_t"] - m[col + "_r"])


# ---------------------------------------------------------------- the hypothesis table
ROW_KEYS = ("hypothesis", "modality", "u", "batch", "metric", "treatment", "reference", "kind", "alternative")


def comparisons(df):
    """Every planned comparison whose two cells are present in the data: list of dicts (without results).
    t, r: treatment and reference cell; control: the FedAvg cell of the floor rule (None: no floor rule)."""
    out = []

    def has(mod, c):
        return not cell(df, mod, *c).empty

    def add(h, mod, u, batch, metric, t, r, kind, alt, control):
        if has(mod, t) and has(mod, r):
            out.append(dict(hypothesis=h, modality=mod, u=u, batch=batch, metric=metric, treatment=cell_name(t),
                            reference=cell_name(r), t=t, r=r, kind=kind, alternative=alt,
                            control=control if mod in PRIMARY else None))

    for mod in sorted(df.modality.unique()):
        if mod.startswith("text"):
            for u in H1_U:
                add("H1", mod, u, MAIN_BATCH, "value", ("fedavg", u, MAIN_BATCH, "recovered"),
                    ("fedavg", 1, MAIN_BATCH, "recovered"), "wilcoxon", "less", None)
        if mod.startswith("image"):
            for b in H5A_B:
                fa = ("fedavg", 1, b, "recovered")
                add("H5a", mod, 1, b, "value", ("fedbn", 1, b, "recovered"), fa, "wilcoxon", "less", fa)
            for b in H6_B:
                add("H6", mod, 1, b, "value", ("fedavg", 1, b, "recovered"), ("fedavg", 1, 1, "recovered"),
                    "wilcoxon", "less", None)
    rec = ("fedavg", 1, H2_B, "recovered")
    add("H2a", "image", 1, H2_B, "f", rec, ("fedavg", 1, H2_B, "oracle"), "equivalence", None, rec)
    add("H2b", "image", 1, H2_B, "f", rec, ("fedavg", 1, H2_B, "none"), "wilcoxon", "greater", rec)
    for proto, h, kind in (("scaffold", "H3", "equivalence"), ("fedprox", "H4", "noninferiority"), ("fedbn", "H5b", "equivalence")):
        for u in PROTO_U:
            fa = ("fedavg", u, MAIN_BATCH, "recovered")
            add(h, "text", u, MAIN_BATCH, "f", (proto, u, MAIN_BATCH, "recovered"), fa, kind, None, fa)
    order = {h: i for i, h in enumerate(PLANNED)}
    return sorted(out, key=lambda c: (order[c["hypothesis"]], c["modality"] not in PRIMARY, c["modality"]))


def evaluate(df, cross=None, m0_override=None, info=None, stratify=True, cache=None):
    """Returns (comparisons, decisions, per-seed means)."""
    m0_override = m0_override or {}
    info = info or {}
    cache = {} if cache is None else cache
    comps, per_seed = [], []
    for c in comparisons(df):
        mod, metric = c["modality"], c["metric"]
        row = {k: c[k] for k in ROW_KEYS}
        row["ref_floor"] = bool(metric == "f" and info.get(mod, {}).get("ref_passes") is not True)
        m = paired(cell(df, mod, *c["t"]), cell(df, mod, *c["r"]), metric)
        d, seeds = m["d"].to_numpy(), m["seed"].tolist()
        lo, hi = t_interval(d)
        row.update(n=len(d), mean_diff=d.mean() if len(d) else np.nan, sd_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
                   ci_low=lo, ci_high=hi)
        if c["control"] is not None:
            above, fp, how, fx = floor_test(df, cross, mod, c["control"], m0_override, stratify, cache)
            row.update(control=cell_name(c["control"]), control_mean=cell(df, mod, *c["control"])["value"].mean(),
                       floor_p=fp, floor_method=how, floor_p_unstratified=fx.get("p_unstratified", np.nan),
                       floor_p_min=fx.get("p_min", np.nan), floor_effect_within_stratum=fx.get("effect", np.nan),
                       at_floor=above is not True)
        else:
            row.update(control="", control_mean=np.nan, floor_p=np.nan, floor_method="", at_floor=False)
        row["decidable"] = not row["ref_floor"] and not row["at_floor"] and len(d) >= 2
        if c["kind"] == "wilcoxon":
            row["W"], row["p"], row["n_nonzero"] = wilcoxon(d, c["alternative"]) if len(d) else (np.nan, np.nan, 0)
        elif c["kind"] == "equivalence":
            row["within_margin"] = bool(lo >= -MARGIN and hi <= MARGIN) if not np.isnan(lo) else False
            row["outside_margin"] = bool(lo > MARGIN or hi < -MARGIN) if not np.isnan(lo) else False
        else:
            row["lower_bound"] = t_lower(d)
            row["within_margin"] = bool(row["lower_bound"] >= -MARGIN) if not np.isnan(row["lower_bound"]) else False
            row["outside_margin"] = bool(hi < -MARGIN) if not np.isnan(hi) else False
        est, se, p2, note = mixed(d, seeds) if len(d) > 2 else (np.nan, np.nan, np.nan, "skipped (n < 3)")
        row.update(mixed_est=est, mixed_se=se, mixed_note=note)
        if not np.isnan(est):
            z = stats.norm.ppf(0.975)
            row.update(mixed_ci_low=est - z * se, mixed_ci_high=est + z * se)
            if c["kind"] == "wilcoxon":
                row["mixed_p"] = stats.norm.cdf(est / se) if c["alternative"] == "less" else stats.norm.sf(est / se)
            elif c["kind"] == "noninferiority":
                row["mixed_lower_bound"] = est - stats.norm.ppf(0.95) * se
        comps.append(row)
        for s, g in m.groupby("seed"):
            per_seed.append({"hypothesis": c["hypothesis"], "modality": mod, "u": c["u"], "batch": c["batch"],
                             "treatment": row["treatment"], "reference": row["reference"], "seed": s, "n": len(g),
                             "mean_diff": g["d"].mean()})
    comp = pd.DataFrame(comps)
    if comp.empty:
        return comp, pd.DataFrame(), pd.DataFrame(per_seed)
    for col in ("p", "W", "n_nonzero", "within_margin", "outside_margin", "lower_bound", "mixed_est", "mixed_p",
                "mixed_ci_low", "mixed_ci_high", "mixed_lower_bound", "floor_p_unstratified", "floor_p_min",
                "floor_effect_within_stratum"):
        if col not in comp:
            comp[col] = np.nan
    comp["p_holm"], comp["mixed_p_holm"] = np.nan, np.nan
    for (h, mod), g in comp[comp.kind == "wilcoxon"].groupby(["hypothesis", "modality"]):
        g = g[g.decidable.astype(bool)]
        if len(g):
            comp.loc[g.index, "p_holm"] = holm(g["p"].to_numpy(float))
            comp.loc[g.index, "mixed_p_holm"] = holm(g["mixed_p"].to_numpy(float))
    comp["significant"] = np.where((comp.kind == "wilcoxon") & comp.decidable.astype(bool), comp["p"] < ALPHA, np.nan)
    agree = []
    for _, r in comp.iterrows():
        if not r["decidable"] or pd.isna(r["mixed_est"]):
            agree.append(np.nan)
        elif r["kind"] == "wilcoxon":
            agree.append(bool((r["mixed_p"] < ALPHA) == (r["p"] < ALPHA)))
        elif r["kind"] == "equivalence":
            agree.append(bool((r["mixed_ci_low"] >= -MARGIN and r["mixed_ci_high"] <= MARGIN) == bool(r["within_margin"])))
        else:
            agree.append(bool((r["mixed_lower_bound"] >= -MARGIN) == bool(r["within_margin"])))
    comp["mixed_agrees"] = agree
    return comp, decide(comp), pd.DataFrame(per_seed)


def decide(comp):
    out = []

    def add(h, mod, decision, reason):
        out.append({"hypothesis": h, "modality": mod, "decision": decision, "reason": reason})

    for (h, mod), g in comp.groupby(["hypothesis", "modality"], sort=False):
        if mod not in PRIMARY:
            add(h, mod, "reported, decides nothing", f"{int(np.nansum(g.significant))}/{len(g)} significant"); continue
        plan = "" if len(g) == PLANNED[h] else f" (the paper plans {PLANNED[h]} comparison(s), the data has {len(g)})"
        if g.ref_floor.astype(bool).all():
            add(h, mod, "not decidable", f"reference cell of the {mod} ({cell_name(REF[mod])}) is not above the floor: "
                                         f"f not computed" + plan); continue
        dec = g[g.decidable.astype(bool)]
        nd = len(g) - len(dec)
        tail = (f"; {nd} comparison(s) with the control cell at the floor, not decidable" if nd else "") + plan
        if dec.empty:
            add(h, mod, "not decidable", "no decidable comparison" + tail); continue
        k = len(dec)
        kind = g.kind.iloc[0]
        if kind == "wilcoxon":
            s = int(np.nansum(dec.significant))
            if h == "H2b":
                inside = bool(((dec.ci_low >= -MARGIN) & (dec.ci_high <= MARGIN)).all())
                if s == k:
                    add(h, mod, "supported", f"{s}/{k} significant" + tail)
                elif s == 0 and inside:
                    add(h, mod, "text does not matter within the margin", f"0/{k} significant, t interval within +-{MARGIN}" + tail)
                else:
                    add(h, mod, "inconclusive", f"{s}/{k} significant, t interval within the margin: {inside}" + tail)
            else:
                add(h, mod, "supported" if s == k else "not supported", f"{s}/{k} significant" + tail)
        else:
            w = int(dec.within_margin.astype(bool).sum())
            o = int(dec.outside_margin.astype(bool).sum())
            if kind == "noninferiority":
                what = f"{w}/{k} lower bounds >= -{MARGIN}, {o}/{k} two-sided intervals entirely below -{MARGIN}"
            else:
                what = f"{w}/{k} intervals within +-{MARGIN}, {o}/{k} entirely outside"
            d = "supported" if w == k else ("contradicted" if o > 0 else "inconclusive")
            add(h, mod, d, what + tail)
    return pd.DataFrame(out)


def cells_table(df, cross=None, m0_override=None, stratify=True, cache=None):
    """Every cell: n, mean, SD, mean m0, mean f; for the primary modalities also its floor test."""
    rows = []
    for (mod, proto, u, b, cond), g in df.groupby(["modality", "protocol", "u", "batch", "condition"], sort=True):
        r = {"modality": mod, "protocol": proto, "u": u, "batch": b, "condition": cond, "n": len(g),
             "mean": g.value.mean(), "sd": g.value.std(ddof=1) if len(g) > 1 else np.nan,
             "mean_m0": g["m0"].mean() if "m0" in g else np.nan, "mean_f": g["f"].mean() if "f" in g else np.nan}
        if mod in PRIMARY:
            above, p, how, x = floor_test(df, cross, mod, (proto, u, b, cond), m0_override, stratify, cache)
            r.update(above_floor=above, floor_p=p, floor_p_min=x.get("p_min", np.nan),
                     floor_effect_within_stratum=x.get("effect", np.nan),
                     floor_p_unstratified=x.get("p_unstratified", np.nan), floor_method=how)
        rows.append(r)
    return pd.DataFrame(rows)


def anova(df):
    """Two-way ANOVA of the text f with victims as blocks on the protocol experiment; decides nothing."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    rec = df[(df.modality == "text") & (df.condition == "recovered") & (df.batch == MAIN_BATCH)]
    # the protocol experiment: only the u levels at which EVERY protocol was run (FedBN at u = 1 is
    # descriptive and FedAvg has further levels for H1; including them makes the design rank-deficient)
    per = [set(x.u) for _, x in rec.groupby("protocol")]
    us = sorted(set.intersection(*per)) if per else []
    g = rec[rec.u.isin(us)].dropna(subset=["f"])
    if g.protocol.nunique() < 2 or len(us) < 1:
        return pd.DataFrame(rows)
    form = "f ~ C(protocol) * C(u) + C(key)" if len(us) > 1 else "f ~ C(protocol) + C(key)"
    tab = anova_lm(smf.ols(form, g).fit(), typ=2)
    tab["share_SS"] = tab["sum_sq"] / tab["sum_sq"].sum()
    tab.index = [i.replace("C(key)", "victim (block)") for i in tab.index]
    for name, r in tab.iterrows():
        rows.append({"modality": "text", "source": name, "df": r["df"], "sum_sq": r["sum_sq"], "share_SS": r["share_SS"],
                     "F": r["F"], "p": r["PR(>F)"], "n_rows": len(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- commands
def cmd_test(a):
    df = load_tidy(a.inputs)
    cross = load_cross(a.cross)
    m0 = {}
    for s in a.m0 or []:
        k, v = s.split("=")
        m0[k.strip().lower()] = float(v)
    cache, strat = {}, not a.no_stratify
    df, info = add_f(df, m0, cross, strat, cache)
    comp, dec, per_seed = evaluate(df, cross, m0, info, strat, cache)
    cells = cells_table(df, cross, m0, strat, cache)
    an = anova(df)
    os.makedirs(a.out, exist_ok=True)
    for name, t in (("cells", cells), ("comparisons", comp), ("decisions", dec), ("per_seed", per_seed), ("anova", an)):
        t.to_csv(os.path.join(a.out, name + ".csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 110)
    print("reference cells, zero points, denominators:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("\ncells:\n" + cells.drop(columns=["floor_method"], errors="ignore").round(4).to_string(index=False))
    cols = ["hypothesis", "modality", "u", "batch", "treatment", "reference", "n", "mean_diff", "sd_diff", "ci_low", "ci_high",
            "lower_bound", "p", "p_holm", "significant", "within_margin", "outside_margin", "floor_p", "decidable",
            "mixed_est", "mixed_agrees"]
    if comp.empty:
        print("\nno planned comparison is present in the data")
    else:
        print("\ncomparisons:\n" + comp[[c for c in cols if c in comp]].round(4).to_string(index=False))
        print("\ndecisions:\n" + dec.to_string(index=False))
        missing = [h for h in PLANNED if h not in set(comp.hypothesis)]
        if missing:
            print(f"\nno data for: {', '.join(missing)}")
    if not an.empty:
        print("\nANOVA of the text f (type II, decides nothing):\n" + an.round(4).to_string(index=False))
    print(f"\nwrote {a.out}/cells.csv, comparisons.csv, decisions.csv, per_seed.csv, anova.csv")


def tidy_from_image_scores(scores, condition=None):
    """Tidy rows of a src/score_images.py CSV: value = A, m0 = A_mismatched, batch and condition as scored."""
    s = pd.read_csv(scores)
    cond = s["condition"] if condition is None and "condition" in s else pd.Series(condition or "recovered", index=s.index)
    cond = cond.fillna("").astype(str).str.strip().str.lower().map(lambda c: COND_ALIASES.get(c, c))
    out = pd.DataFrame({"seed": s["seed"], "victim": s["victim"], "protocol": s.get("protocol", "fedavg"),
                        "u": s["u"], "batch": s["batch"].astype(int), "condition": cond, "modality": "image",
                        "value": s["A"], "m0": s.get("A_mismatched", np.nan), "file": s["file"]})
    for k in ("n_abnormal", "A_lab", "A_lab_mismatched", "top1", "top1pct", "dir"):  # carried along, used by no test
        if k in s:
            out[k] = s[k]
    out["protocol"] = out["protocol"].fillna("fedavg").replace("", "fedavg")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    if out["seed"].isna().any():
        print(f"WARNING: {int(out['seed'].isna().sum())} rows without a seed (rerun score_images.py with --seed)", file=sys.stderr)
    return out


def cmd_tidy(a):
    t = tidy_from_image_scores(a.scores, a.condition)
    t.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(t)} rows)")


def cmd_gate(a):
    s = pd.read_csv(a.scores)
    cross_path = a.cross or os.path.splitext(a.scores)[0] + "_cross.csv"
    c = pd.read_csv(cross_path)
    s = _norm_ids(s, ("seed", "victim"))
    c = _norm_ids(c, ("v_seed", "v_victim", "w_seed", "w_victim"))
    if a.victims:  # restrict BEFORE the control is computed: both the originals and the partners
        c = c[c.v_victim.isin(a.victims) & c.w_victim.isin(a.victims)]
    if a.seed is not None:
        c = c[(c.v_seed == a.seed) & (c.w_seed == a.seed)]
    if "group" in s:  # (group, seed, victim) identifies a row also when file names repeat across directories
        top1 = {(g_, sd, v): t for g_, sd, v, t in zip(s["group"], s["seed"], s["victim"], s["top1"])}
    else:
        top1 = dict(zip(s["file"], s["top1"]))
    pairs = None
    if a.regression:
        pp = a.pairs or os.path.splitext(a.scores)[0] + "_pairs.csv"
        pairs = _norm_ids(pd.read_csv(pp), ("seed", "victim", "partner_seed", "partner_victim"))
        if "group" not in pairs or "x_share" not in pairs:
            raise SystemExit(f"{pp} has no group / x_share column: rescore with the current src/score_images.py")
    rows = []
    fields = ["dir", "protocol", "condition", "batch", "u", "init", "tag", "res"]
    for gid, g in c.groupby("group"):
        keys = sorted(set(zip(g.v_seed, g.v_victim)), key=lambda t: (str(t[0]), t[1]))
        g = g.assign(vk=list(zip(g.v_seed, g.v_victim)), wk=list(zip(g.w_seed, g.w_victim)))
        M = g.pivot_table(index="vk", columns="wk", values="A", aggfunc="mean").reindex(index=keys, columns=keys)
        if M.isna().any().any():
            print(f"WARNING group {gid}: incomplete cross matrix, skipped", file=sys.stderr)
            continue
        M_lab = None
        if "A_lab" in g and g["A_lab"].notna().all():
            M_lab = g.pivot_table(index="vk", columns="wk", values="A_lab", aggfunc="mean").reindex(index=keys, columns=keys).to_numpy()
        strata = None
        if not a.no_stratify and "v_n_abnormal" in g and g["v_n_abnormal"].notna().all():
            strata = g.groupby("vk")["v_n_abnormal"].first().reindex(keys).astype(int).to_numpy()
        diag = g[g.diagonal.astype(bool)].set_index("vk").loc[keys]
        t1 = [top1.get((gid, k[0], k[1]), np.nan) for k in keys] if "group" in s else [top1.get(f, np.nan) for f in diag["v_file"]]
        r = gate(M.to_numpy(), t1, strata=strata, M_lab=M_lab)
        if pairs is not None:
            pg = pairs[pairs.group == gid]
            vset = set(keys)
            own = pd.Series([t in vset for t in zip(pg.seed, pg.victim)], index=pg.index, dtype=bool)
            par = pd.Series([t in vset for t in zip(pg.partner_seed, pg.partner_victim)], index=pg.index, dtype=bool) \
                if "partner_seed" in pg else pd.Series(False, index=pg.index)
            pg = pg[own & ((pg.kind == "attack") | ((pg.kind == "mismatched") & par))]
            for tag_, incl in (("reg", True), ("reg_offdiag", False)):
                lr = label_regression(pg, a.reg_score, incl, a.n_boot)
                for k_ in ("slope", "slope_lo", "slope_hi", "excess", "excess_lo", "excess_hi"):
                    r[f"{tag_}_{k_}"] = lr.get(k_, np.nan)
                r[f"{tag_}_n_victims"] = lr.get("n_victims", 0)
            r["reg_score"] = a.reg_score
            r["reg_mean_x_diag"], r["reg_mean_x_mismatched"] = lr.get("mean_x_diag", np.nan), lr.get("mean_x_mismatched", np.nan)
        rows.append({**{f: g[f].iloc[0] for f in fields if f in g}, **r})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 80)
    show = out.drop(columns=[x for x in ("dir", "protocol", "condition") if x in out and out[x].nunique() <= 1])
    print(show.round(4).T.to_string() if len(show) <= 12 else show.round(4).to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test", help="decision rules of the hypothesis table")
    t.add_argument("inputs", nargs="+")
    t.add_argument("--cross", nargs="*", help="score_images.py *_cross.csv files (image floor tests)")
    t.add_argument("--out", default="runs/analysis")
    t.add_argument("--m0", nargs="*", help="override zero points, e.g. image=0.5 text=0.21")
    t.add_argument("--no-stratify", action="store_true",
                   help="image floor tests unstratified (comparison only; default: stratified by n_abnormal)")
    t.set_defaults(fn=cmd_test)
    ti = sub.add_parser("tidy-images", help="score_images.py CSV -> tidy rows (value = A, m0 = A_mismatched)")
    ti.add_argument("scores")
    ti.add_argument("--out", required=True)
    ti.add_argument("--condition", default=None, help="force a condition label (default: column 'condition')")
    ti.set_defaults(fn=cmd_tidy)
    g = sub.add_parser("gate", help="calibration gate (permutation test) on a score_images.py CSV and its cross file")
    g.add_argument("scores")
    g.add_argument("--cross", default=None, help="default: <scores stem>_cross.csv")
    g.add_argument("--victims", type=int, nargs="*", help="gate clients only; restricts the group before the control")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--out", default=None)
    g.add_argument("--no-stratify", action="store_true", help="skip the stratified permutation tests")
    g.add_argument("--regression", action="store_true", help="label-distance regression (needs the *_pairs.csv)")
    g.add_argument("--pairs", default=None, help="default: <scores stem>_pairs.csv")
    g.add_argument("--reg-score", default="A", choices=["A", "A_lab"], help="score regressed on the label share")
    g.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples over victims (regression)")
    g.set_defaults(fn=cmd_gate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
