"""Decision rules of the paper's hypothesis table (Table "Hypothesis Testing Framework and Decision Rules",
Sections "Evaluation Metrics" and "Hypothesis Testing") and the calibration gate.

INPUT
  Tidy CSVs, one row per victim, cell and modality:
    seed, victim, protocol, u, modality, value          (required)
    condition                                           (optional; recovered | oracle | none; default recovered)
    m0                                                  (optional; the victim's zero-point score)
  - victims are paired by (seed, victim); protocol: fedavg | fedprox | scaffold | fedbn;
  - modality "image" (value = identification score A, max rule) and "text" (value = ROUGE-L) are primary.
    Any other modality (e.g. "image_ssim") gets the H1 comparisons only, "reported, decides nothing";
  - m0: image = A_mismatched of src/score_images.py (the victim's originals against the reconstructions of
    every other victim of the cell); text = column m0 of src/score_text.py: the B most frequent findings
    reports of the PUBLIC pairs (validation + test), matched to the victim's B texts by the ROUGE-L
    assignment and scored like the attack (the single most frequent report is the earlier definition,
    m0_single there). The modality's m_0 = mean of m0 over the reference cell (FedAvg, u = 1, recovered);
    --m0 image=0.5 overrides it.
  Cross matrices (--cross, the *_cross.csv of src/score_images.py): A(v, w) of v's originals against w's
  reconstructions, per cell (protocol, condition, u, batch), with v_n_abnormal (the label count of v's
  attacked batch; the strata of the stratified test) and A_lab. Needed for the image floor tests and the
  reference-cell test; restricted to the victims of the tidy cell before testing. For the 20 victims of a
  main-experiment cell (4 per seed, seeds 1-5) score with `score_images.py --scope cell`.
  `python src/analyze.py tidy-images scores.csv` converts a src/score_images.py CSV (value = A, m0 =
  A_mismatched; batch sizes other than --main-batch 8 get condition "B<k>" / "<cond>_B<k>", like score_text.py).

RULES IMPLEMENTED (literal reading of the paper as revised on 2026-11-22)
  Permutation test of a cross matrix M (n victims): statistic T = mean_v M[v, v]; null = mean_v M[v, pi(v)]
     over all permutations pi (exact for n <= 8, else 100,000 Monte Carlo draws with seed 0 plus the
     identity); p = share of permutations with a statistic >= T (identity included).
  LABEL CONTROLS (second review, 2026-11-22). The attacker is given the batch labels and every batch is
     reconstructed with its own labels, so a reconstruction that reflects only the labels beats the
     unstratified permutation test. Hence:
     - Stratified permutation test (stratified_perm_test): only permutations within strata of victims with
       the same number of abnormal pairs in their attacked batch (v_n_abnormal of the cross file); exact if
       there are <= 50,000 such permutations, else 100,000 Monte Carlo draws (seed 0) plus the identity.
       Reported: number of permutations, smallest attainable p (1 / number evaluated), p, and the effect size
       mean over victims in strata of size >= 2 of [A(v, v) - mean within-stratum off-diagonal A(v, w)] with
       the number of contributing victims. With few victims the test may be unable to reject at all (the 7
       gate clients of seed 0, label counts 0, 0, 8, 8, 8, 3, 6, have 2! 3! = 12 permutations: p >= 1/12).
     - It is THE image floor and reference-cell test of `test` (the unstratified p is kept as
       floor_p_unstratified / ref_test_p_unstratified; --no-stratify switches back, for comparison only;
       a cross file without v_n_abnormal falls back to the unstratified test and floor_method says so).
     - `gate` reports, per group: the unstratified test (informative, the paper's definition), the
       stratified test of A (p_strat, ..., informative_strat), both tests on the same-label score A_lab
       (*_lab; E[A_lab] = 0.5 for every (v, w) under label information only), the one-sided sign-flip test
       (exact for n <= 20) and Wilcoxon test of D_v = A_lab(v, v) - mean_{w != v} A_lab(v, w), and with
       --regression the label-distance regression (label_regression): A_i(v, w) on the share x of w's batch
       with the label of original i, fixed effect per original, fitted on the diagonal and the mismatched
       rows (reg_*; reg_offdiag_*: fitted on the mismatched rows only); slope = size of the label channel,
       excess = mean over victims of (diagonal value - fitted value at its own share), 95% percentile
       bootstrap over victims (2,000 resamples, seed 0).
  Reference-cell test: image f is computed only if the FedAvg, u = 1, recovered cell passes the stratified
     permutation test (p <= 0.05); otherwise every image decision (H1 included) is "not tested". Without a
     cross matrix for that cell f is computed and the output says the test was not run. Text f is computed
     only if the FedAvg, u = 1 text cell passes its floor test (one-sided Wilcoxon of ROUGE-L vs m0);
     otherwise the text parts of H2-H5 are "not tested" (H1 text, decided on ROUGE-L, is still tested; paper
     Section "Evaluation Metrics": "H2 to H5 for the text").
  f_P(u, v) = (m_P(u, v) - m_0) / (mean_v m_FedAvg(1, v) - m_0), one denominator per modality.
  Floor (H2-H5 only): a comparison is decidable only if its FedAvg cell (recovered condition, same modality
     and u) is above the floor by a test at one-sided alpha = 0.05: images, the permutation test on that
     cell's cross matrix; text, a one-sided Wilcoxon signed-rank test of the victims' ROUGE-L against their m0.
     A cell that fails is at the floor; its comparisons are "not decidable" and leave their Holm family; the
     hypothesis is decided on the remaining comparisons ("undecided" for a modality if none remains). If an
     image cell has no cross matrix the old point rule (group mean f <= 0.1) is used and floor_method says so.
  Differences are treatment - reference: H1 m(u) - m(1); H2a f(rec) - f(oracle); H2b f(rec) - f(none);
     H3/H4/H5 f(protocol) - f(FedAvg).
  H1: one-sided Wilcoxon (m(u) < m(1)), Holm over the comparisons of a modality; supported if all significant.
  Interval rules (H2a, H3, H5 text; two-sided 95% t interval vs +-0.05): supported if every interval lies
     within the margin; contradicted if at least one lies entirely outside it; otherwise inconclusive.
  H4: supported if every one-sided 95% t lower bound >= -0.05; contradicted if the upper end of a two-sided
     95% interval lies below -0.05; otherwise inconclusive.
  H2b: one-sided Wilcoxon (f(rec) > f(none)), Holm over 2; supported if all significant; if none is and all
     t intervals lie within the margin -> "text does not matter within the margin"; otherwise undecided.
  H5 image: one-sided Wilcoxon (f(FedBN) < f(FedAvg)), Holm over 2, all significant -> supported.
  Overall H3-H5 (both modalities): decided on the modality parts that are decided: supported if all are
     supported; contradicted if any is contradicted; not supported if any Wilcoxon part is not supported;
     otherwise inconclusive.
  Zero differences: Wilcoxon zero_method="wilcox" (all zero -> p = 1); a zero-width interval is [mean, mean].
  Margins are inclusive. Every comparison also gets the per-seed means and the mixed-model sensitivity
  analysis d ~ 1 + (1 | seed) (statsmodels MixedLM, REML, Wald intervals, one-sided p for directional tests,
  Holm in the same family; column mixed_agrees). ANOVA per modality: f ~ C(protocol) * C(u) + C(victim) on the
  protocol experiment, type II; decides nothing.
  --image-gate not_informative (calibration gate failed at B = 8): all image decisions "undecided (gate)".

GATE (calibration, command `gate`): per group of a score_images.py cross file, restricted to --victims
  (and --seed) BEFORE the control is computed: informative = permutation-test p <= 0.05; strongly_informative
  = informative and mean A >= 0.90; also the share of originals ranked first, the control (mean over
  w != v within the restricted group) and, for the record, the old one-sided Wilcoxon of A against it;
  plus the label controls above (stratified, A_lab, sign-flip, --regression).

AMBIGUITIES (most literal reading implemented)
  1. "Every image decision is not tested" when the reference cell fails is applied to H1 too, although H1 is
     decided on A, not f.
  2. Overall H3-H5 when the modality parts disagree in kind (e.g. image "not supported", text "inconclusive")
     is not defined by the paper; the order above is ours.
  3. The floor test is on the FedAvg cell only, even when the treatment cell is itself at chance.
  4. H2b's third outcome is output verbatim.
  5. `gate`: "informative" keeps the paper's (unstratified) definition; the label controls are extra columns
     (informative_strat etc.) until the paper says which test decides. At the gate (7 victims) the stratified
     test cannot reject at batch 8 (12 permutations); the sign-flip test of A_lab (p >= 1/128) can.
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
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ALPHA, MARGIN, FLOOR, GATE_A = 0.05, 0.05, 0.1, 0.90
PRIMARY = ("image", "text")
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


def stratified_perm_test(M, strata, n_mc=100_000, seed=0, exact_max=50_000):
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
def _norm_ids(df, cols):
    for c in cols:  # 0, 0.0 and "0" must pair: integers where possible
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all() and (num == num.round()).all():
            df[c] = num.astype(int)
    return df


def _norm_cell(df):
    df["condition"] = df["condition"].fillna("").astype(str).str.strip().str.lower().map(lambda c: COND_ALIASES.get(c, c))
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
    dup = df.duplicated(["key", "protocol", "u", "condition", "modality"])
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


def cross_matrix(cross, keys, protocol="fedavg", u=1, condition="recovered", batch=None):
    """Square A(v, w) over the victims in `keys` that have a complete row in the cell; None if unavailable."""
    M, keys, _ = cross_matrix_strata(cross, keys, protocol, u, condition, batch)
    return M, keys


def cross_matrix_strata(cross, keys, protocol="fedavg", u=1, condition="recovered", batch=None, value="A"):
    """cross_matrix plus the strata of the stratified test (v_n_abnormal per victim; None if the cross file
    has no such column)."""
    if cross is None:
        return None, [], None
    c = cross[(cross.protocol == protocol) & (cross.u == u) & (cross.condition == condition)]
    if batch is not None and "batch" in c:
        c = c[c.batch == batch]
    if "group" in c and c.group.nunique() > 1:
        raise SystemExit(f"cross file has {c.group.nunique()} groups for cell {protocol} u={u} {condition}; use --batch")
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


def cell(df, mod, protocol="fedavg", u=1, condition="recovered"):
    return df[(df.modality == mod) & (df.protocol == protocol) & (df.u == u) & (df.condition == condition)]


def victim_m0(df, mod):
    """Per-victim zero point (m0 is a property of the victim); NaN if absent."""
    if "m0" not in df:
        return {}
    g = df[(df.modality == mod) & df["m0"].notna()]
    return g.groupby("key")["m0"].mean().to_dict()


def floor_test(df, cross, mod, u, m0_override, batch=None, stratify=True):
    """Is the FedAvg recovered cell at u above the floor? Returns (above, p, method, extra); extra holds the
    unstratified p and the stratified test's details for images (empty for text)."""
    c = cell(df, mod, "fedavg", u)
    if c.empty:
        return False, np.nan, "no FedAvg cell", {}
    if mod == "image":
        M, keys, strata = cross_matrix_strata(cross, list(c.key), "fedavg", u, "recovered", batch)
        if M is not None and len(keys) > 1:
            r = cell_perm_test(M, strata, stratify)
            return bool(r["p"] <= ALPHA), r["p"], r["method"], r
        fm = c["f"].mean()
        return bool(fm > FLOOR) if not np.isnan(fm) else False, np.nan, "point rule mean f > 0.1 (no cross matrix)", {}
    if mod in m0_override:
        m0 = np.full(len(c), m0_override[mod])
    else:
        mp = victim_m0(df, mod)
        m0 = c.key.map(mp).to_numpy(float)
    ok = ~np.isnan(m0)
    if not ok.any():
        fm = c["f"].mean()
        return bool(fm > FLOOR) if not np.isnan(fm) else False, np.nan, "point rule mean f > 0.1 (no m0)", {}
    _, p, _ = wilcoxon(c.value.to_numpy(float)[ok] - m0[ok], "greater")
    return bool(p <= ALPHA), p, f"Wilcoxon vs m0 (n={int(ok.sum())})", {}


def add_f(df, m0_override, cross=None, batch=None, stratify=True):
    """Adds column f; returns (df, info per modality). Image f only if the reference cell passes the
    (stratified) permutation test; text f only if the text reference cell passes its floor test (one-sided
    Wilcoxon of ROUGE-L vs m0). info[mod]['ref_passes'] False -> f not computed."""
    df = df.copy()
    df["f"] = np.nan
    info = {}
    for mod in df.modality.unique():
        ref = cell(df, mod)
        if ref.empty:
            info[mod] = {"note": "no FedAvg u=1 recovered cell: f undefined"}
            continue
        inf = {}
        if mod == "image":
            M, keys, strata = cross_matrix_strata(cross, list(ref.key), batch=batch)
            if M is not None and len(keys) > 1:
                r = cell_perm_test(M, strata, stratify)
                inf.update(ref_test_p=r["p"], ref_test_p_unstratified=r["p_unstratified"], ref_test_method=r["method"],
                           ref_test_n=len(keys), ref_passes=bool(r["p"] <= ALPHA))
                if "effect" in r:
                    inf.update(ref_test_p_min=r["p_min"], ref_effect_within_stratum=r["effect"], ref_n_contrib=r["n_contrib"])
            else:
                inf.update(ref_test_p=np.nan, ref_passes=None, ref_note="reference cell not tested (no cross matrix)")
            if inf["ref_passes"] is False:
                info[mod] = {**inf, "note": "reference cell fails the permutation test: image f not computed"}
                continue
        if mod in m0_override:
            m0, src = m0_override[mod], "cli"
        elif "m0" in df and ref["m0"].notna().any():
            m0, src = float(ref["m0"].mean()), "mean m0 of reference cell"
        else:
            info[mod] = {**inf, "note": "no m0 (column m0 or --m0): f undefined"}
            continue
        if mod == "text":  # reference-cell precondition for the text (paper: H2-H5 text not tested)
            above, p, how, _ = floor_test(df, cross, mod, 1, m0_override, batch, stratify)
            inf.update(ref_test_p=p, ref_test_method=how, ref_passes=bool(above))
            if not above:
                info[mod] = {**inf, "note": "text reference cell fails its floor test: text f not computed, H2-H5 text not tested"}
                continue
        den = float(ref["value"].mean()) - m0
        sel = df.modality == mod
        df.loc[sel, "f"] = (df.loc[sel, "value"] - m0) / den
        info[mod] = {**inf, "m0": m0, "m0_source": src, "ref_mean": den + m0, "denominator": den, "n_ref": len(ref)}
    return df, info


def paired(a, b, col):
    m = a[["key", "seed", col]].merge(b[["key", col]], on="key", suffixes=("_t", "_r"))
    m = m.dropna(subset=[col + "_t", col + "_r"])
    return m.assign(d=m[col + "_t"] - m[col + "_r"])


# ---------------------------------------------------------------- the hypothesis table
def comparisons(df):
    """Every planned comparison present in the data: list of dicts (without results)."""
    out = []
    fa = df[(df.protocol == "fedavg") & (df.condition == "recovered")]
    for mod in sorted(df.modality.unique()):
        for u in sorted(set(fa[fa.modality == mod].u) - {1}):
            out.append(dict(hypothesis="H1", modality=mod, u=u, metric="value", treatment=f"fedavg u={u}",
                            reference="fedavg u=1", t=("fedavg", u, "recovered"), r=("fedavg", 1, "recovered"),
                            kind="wilcoxon", alternative="less", floor_rule=False))
    for cond, h, kind, alt in (("oracle", "H2a", "equivalence", None), ("none", "H2b", "wilcoxon", "greater")):
        us = sorted(set(df[(df.modality == "image") & (df.protocol == "fedavg") & (df.condition == cond)].u))
        for u in us:
            out.append(dict(hypothesis=h, modality="image", u=u, metric="f", treatment=f"recovered u={u}",
                            reference=f"{cond} u={u}", t=("fedavg", u, "recovered"), r=("fedavg", u, cond),
                            kind=kind, alternative=alt, floor_rule=True))
    for proto, h in (("scaffold", "H3"), ("fedprox", "H4"), ("fedbn", "H5")):
        for mod in PRIMARY:
            for u in sorted(set(df[(df.modality == mod) & (df.protocol == proto) & (df.condition == "recovered")].u)):
                if h == "H3" or (h == "H5" and mod == "text"):
                    kind, alt = "equivalence", None
                elif h == "H4":
                    kind, alt = "noninferiority", None
                else:
                    kind, alt = "wilcoxon", "less"
                out.append(dict(hypothesis=h, modality=mod, u=u, metric="f", treatment=f"{proto} u={u}",
                                reference=f"fedavg u={u}", t=(proto, u, "recovered"), r=("fedavg", u, "recovered"),
                                kind=kind, alternative=alt, floor_rule=True))
    return out


def evaluate(df, image_gate="unknown", cross=None, m0_override=None, info=None, batch=None, stratify=True):
    m0_override = m0_override or {}
    info = info or {}
    image_not_tested = info.get("image", {}).get("ref_passes") is False
    text_not_tested = info.get("text", {}).get("ref_passes") is False
    comps, per_seed, floors = [], [], {}
    for c in comparisons(df):
        mod, metric = c["modality"], c["metric"]
        if (mod.startswith("image") and image_not_tested) or (mod == "text" and text_not_tested and c["floor_rule"]):
            comps.append({k: c[k] for k in ("hypothesis", "modality", "u", "metric", "treatment", "reference", "kind",
                                            "alternative")} | {"n": 0, "decidable": False, "not_tested": True})
            continue
        m = paired(cell(df, mod, *c["t"]), cell(df, mod, *c["r"]), metric)
        d, seeds = m["d"].to_numpy(), m["seed"].tolist()
        row = {k: c[k] for k in ("hypothesis", "modality", "u", "metric", "treatment", "reference", "kind", "alternative")}
        lo, hi = t_interval(d)
        row.update(n=len(d), mean_diff=d.mean() if len(d) else np.nan, sd_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
                   ci_low=lo, ci_high=hi, not_tested=False)
        if c["floor_rule"]:
            if (mod, c["u"]) not in floors:
                floors[(mod, c["u"])] = floor_test(df, cross, mod, c["u"], m0_override, batch, stratify)
            above, fp, how, fx = floors[(mod, c["u"])]
            row.update(fedavg_mean_f=cell(df, mod, "fedavg", c["u"])["f"].mean(), floor_p=fp, floor_method=how,
                       floor_p_unstratified=fx.get("p_unstratified", np.nan), floor_p_min=fx.get("p_min", np.nan),
                       floor_effect_within_stratum=fx.get("effect", np.nan), at_floor=not above)
        else:
            row.update(fedavg_mean_f=np.nan, floor_p=np.nan, floor_method="", at_floor=False)
        row["decidable"] = not row["at_floor"] and len(d) >= 2
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
            per_seed.append({"hypothesis": c["hypothesis"], "modality": mod, "u": c["u"], "treatment": c["treatment"],
                             "reference": c["reference"], "seed": s, "n": len(g), "mean_diff": g["d"].mean()})
    comp = pd.DataFrame(comps)
    if comp.empty:
        return comp, pd.DataFrame(), pd.DataFrame(per_seed)
    for col in ("p", "W", "n_nonzero", "within_margin", "outside_margin", "lower_bound", "mixed_est", "mixed_p",
                "mixed_ci_low", "mixed_ci_high", "mixed_lower_bound", "at_floor", "floor_p", "floor_method", "ci_low", "ci_high"):
        if col not in comp:
            comp[col] = np.nan
    comp["p_holm"], comp["mixed_p_holm"] = np.nan, np.nan
    for (h, mod), g in comp[comp.kind == "wilcoxon"].groupby(["hypothesis", "modality"]):
        g = g[g.decidable.astype(bool)]
        if len(g):
            comp.loc[g.index, "p_holm"] = holm(g["p"].to_numpy(float))
            comp.loc[g.index, "mixed_p_holm"] = holm(g["mixed_p"].to_numpy(float))
    comp["significant"] = np.where((comp.kind == "wilcoxon") & comp.decidable.astype(bool), comp["p_holm"] < ALPHA, np.nan)
    agree = []
    for _, r in comp.iterrows():
        if not r["decidable"] or pd.isna(r["mixed_est"]):
            agree.append(np.nan)
        elif r["kind"] == "wilcoxon":
            agree.append(bool((r["mixed_p_holm"] < ALPHA) == (r["p_holm"] < ALPHA)))
        elif r["kind"] == "equivalence":
            agree.append(bool((r["mixed_ci_low"] >= -MARGIN and r["mixed_ci_high"] <= MARGIN) == bool(r["within_margin"])))
        else:
            agree.append(bool((r["mixed_lower_bound"] >= -MARGIN) == bool(r["within_margin"])))
    comp["mixed_agrees"] = agree
    return comp, decide(comp, image_gate), pd.DataFrame(per_seed)


def decide(comp, image_gate):
    out = []

    def add(h, mod, decision, reason):
        out.append({"hypothesis": h, "modality": mod, "decision": decision, "reason": reason})

    for (h, mod), g in comp.groupby(["hypothesis", "modality"], sort=False):
        if mod.startswith("image") and image_gate == "not_informative":
            add(h, mod, "undecided (gate)", "image attack not informative at B = 8"); continue
        if g.not_tested.astype(bool).all():
            add(h, mod, "not tested", "FedAvg u=1 recovered cell fails the permutation test" if mod.startswith("image")
                else "FedAvg u=1 recovered cell fails its floor test (Wilcoxon vs m0)"); continue
        dec = g[g.decidable.astype(bool)]
        nd = len(g) - len(dec)
        tail = f"; {nd} comparison(s) at the floor, not decidable" if nd else ""
        if h == "H1" and mod not in PRIMARY:
            add(h, mod, "reported, decides nothing", f"{int(np.nansum(g.significant))}/{len(g)} Holm-significant"); continue
        if dec.empty:
            add(h, mod, "undecided", "no decidable comparison" + tail); continue
        k = len(dec)
        kind = g.kind.iloc[0]
        if kind == "wilcoxon":
            s = int(np.nansum(dec.significant))
            if h == "H2b":
                inside = bool(((dec.ci_low >= -MARGIN) & (dec.ci_high <= MARGIN)).all())
                if s == k:
                    add(h, mod, "supported", f"{s}/{k} Holm-significant" + tail)
                elif s == 0 and inside:
                    add(h, mod, "text does not matter within the margin", f"0/{k} significant, all t intervals within +-{MARGIN}" + tail)
                else:
                    add(h, mod, "undecided", f"{s}/{k} significant, intervals within margin: {inside}" + tail)
            else:
                note = "" if h != "H1" or k == 4 else f" (paper plans 4 comparisons, data has {k})"
                add(h, mod, "supported" if s == k else "not supported", f"{s}/{k} Holm-significant{note}" + tail)
        else:
            w = int(dec.within_margin.astype(bool).sum())
            o = int(dec.outside_margin.astype(bool).sum())
            if kind == "noninferiority":
                what = f"{w}/{k} lower bounds >= -{MARGIN}, {o}/{k} upper bounds < -{MARGIN}"
            else:
                what = f"{w}/{k} intervals within +-{MARGIN}, {o}/{k} entirely outside"
            d = "supported" if w == k else ("contradicted" if o > 0 else "inconclusive")
            add(h, mod, d, what + tail)
    dec = pd.DataFrame(out)
    for h in ("H3", "H4", "H5"):
        g = dec[(dec.hypothesis == h) & dec.modality.isin(PRIMARY)]
        if g.empty:
            continue
        parts = g[~g.decision.str.startswith("undecided") & (g.decision != "not tested")]
        missing = sorted(set(PRIMARY) - set(g.modality))
        if parts.empty:
            d = "undecided"
        elif (parts.decision == "supported").all():
            d = "supported"
        elif (parts.decision == "contradicted").any():
            d = "contradicted"
        elif (parts.decision == "not supported").any():
            d = "not supported"
        else:
            d = "inconclusive"
        why = ", ".join(f"{r.modality}: {r.decision}" for r in g.itertuples())
        if missing:
            why += f"; no data for {missing}"
        out.append({"hypothesis": h, "modality": "overall", "decision": d, "reason": why})
    return pd.DataFrame(out)


def anova(df):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    rec = df[df.condition == "recovered"]
    for mod in PRIMARY:
        us = sorted(set(rec[(rec.modality == mod) & (rec.protocol != "fedavg")].u))
        g = rec[(rec.modality == mod) & rec.u.isin(us)].dropna(subset=["f"])
        if g.protocol.nunique() < 2 or len(us) < 1:
            continue
        form = "f ~ C(protocol) * C(u) + C(key)" if len(us) > 1 else "f ~ C(protocol) + C(key)"
        tab = anova_lm(smf.ols(form, g).fit(), typ=2)
        tab["share_SS"] = tab["sum_sq"] / tab["sum_sq"].sum()
        tab.index = [i.replace("C(key)", "victim (block)") for i in tab.index]
        for name, r in tab.iterrows():
            rows.append({"modality": mod, "source": name, "df": r["df"], "sum_sq": r["sum_sq"], "share_SS": r["share_SS"],
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
    df, info = add_f(df, m0, cross, a.batch, not a.no_stratify)
    comp, dec, per_seed = evaluate(df, a.image_gate, cross, m0, info, a.batch, not a.no_stratify)
    an = anova(df)
    os.makedirs(a.out, exist_ok=True)
    for name, t in (("comparisons", comp), ("decisions", dec), ("per_seed", per_seed), ("anova", an)):
        t.to_csv(os.path.join(a.out, name + ".csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 90)
    print("zero points, denominators, reference-cell test:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if a.image_gate == "unknown":
        print("  (calibration gate not given: image hypotheses are tested as if the attack were informative at B = 8)")
    cols = ["hypothesis", "modality", "u", "treatment", "reference", "n", "mean_diff", "ci_low", "ci_high", "lower_bound",
            "p", "p_holm", "significant", "within_margin", "outside_margin", "floor_p", "floor_p_unstratified", "decidable",
            "mixed_est", "mixed_agrees"]
    if comp.empty:
        print("\nno planned comparison is present in the data")
    else:
        print("\ncomparisons:\n" + comp[[c for c in cols if c in comp]].round(4).to_string(index=False))
        print("\ndecisions:\n" + dec.to_string(index=False))
    if not an.empty:
        print("\nANOVA (type II, decides nothing):\n" + an.round(4).to_string(index=False))
    print(f"\nwrote {a.out}/comparisons.csv, decisions.csv, per_seed.csv, anova.csv")


def tidy_from_image_scores(scores, condition=None, main_batch=8):
    """Batch sizes other than main_batch get the part "B<k>" in the condition, joined with "_" to any
    condition other than recovered, as in src/score_text.py ("B4" for recovered, "oracle_B4"): such rows are
    never a main-experiment cell and cannot collide with the reference cell."""
    s = pd.read_csv(scores)
    cond = s["condition"] if condition is None and "condition" in s else pd.Series(condition or "recovered", index=s.index)
    cond = cond.fillna("").astype(str).str.strip().str.lower().map(lambda c: COND_ALIASES.get(c, c))
    parts = [[] if c == "recovered" else [c] for c in cond]
    for p, b in zip(parts, s["batch"].astype(int)):
        if b != main_batch:
            p.append(f"B{b}")
    cond = ["_".join(p) if p else "recovered" for p in parts]
    out = pd.DataFrame({"seed": s["seed"], "victim": s["victim"], "protocol": s.get("protocol", "fedavg"),
                        "u": s["u"], "condition": cond, "modality": "image", "value": s["A"],
                        "m0": s.get("A_mismatched", np.nan), "batch": s["batch"], "file": s["file"]})
    for k in ("n_abnormal", "A_lab", "A_lab_mismatched", "dir"):  # carried along, used by no test
        if k in s:
            out[k] = s[k]
    out["protocol"] = out["protocol"].fillna("fedavg").replace("", "fedavg")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    if out["seed"].isna().any():
        print(f"WARNING: {int(out['seed'].isna().sum())} rows without a seed (rerun score_images.py with --seed)", file=sys.stderr)
    return out


def cmd_tidy(a):
    t = tidy_from_image_scores(a.scores, a.condition, a.main_batch)
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
    t.add_argument("--cross", nargs="*", help="score_images.py *_cross.csv files (image floor and reference tests)")
    t.add_argument("--batch", type=int, default=None, help="batch size of the cross matrices to use (if several)")
    t.add_argument("--out", default="runs/analysis")
    t.add_argument("--m0", nargs="*", help="override zero points, e.g. image=0.5 text=0.21")
    t.add_argument("--image-gate", choices=["informative", "not_informative", "unknown"], default="unknown")
    t.add_argument("--no-stratify", action="store_true",
                   help="image floor / reference-cell tests unstratified (comparison only; default: stratified by n_abnormal)")
    t.set_defaults(fn=cmd_test)
    ti = sub.add_parser("tidy-images", help="score_images.py CSV -> tidy rows (value = A, m0 = A_mismatched)")
    ti.add_argument("scores")
    ti.add_argument("--out", required=True)
    ti.add_argument("--condition", default=None, help="force a condition label (default: column 'condition')")
    ti.add_argument("--main-batch", type=int, default=8, help="other batch sizes get condition part B<k>")
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
