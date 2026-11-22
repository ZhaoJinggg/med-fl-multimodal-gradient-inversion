"""Decision rules of the paper's hypothesis table (Table "Hypothesis Testing Framework and Decision Rules",
Sections "Evaluation Metrics" and "Hypothesis Testing") and the calibration gate.

INPUT: one or more tidy CSVs with one row per victim, cell and modality:
    seed, victim, protocol, u, modality, value          (required)
    condition                                           (optional; recovered | oracle | none; default recovered)
    m0                                                  (optional; the victim's zero-point score, see below)
  - victims are paired by (seed, victim);
  - protocol: fedavg | fedprox | scaffold | fedbn (case-insensitive);
  - modality: "image" (value = identification score A) and "text" (value = ROUGE-L) are the primary
    metrics. Any other modality name (e.g. "image_ssim") gets the H1 comparisons only and is labelled
    "reported, decides nothing" (the registered H1 test on SSIM);
  - m0: per-victim zero point (image: A_mismatched of src/score_images.py, the victim's originals scored
    against the reconstructions of all other victims of the cell; text: ROUGE-L of the most frequent
    training report against the victim's texts). The modality's m_0 is the mean of m0 over the victims of
    the reference cell (FedAvg, u = 1, recovered); --m0 image=0.5 overrides it.
  `python src/analyze.py tidy-images scores.csv` converts a src/score_images.py CSV (value = A, m0 = A_mismatched).

RULES IMPLEMENTED (literal reading of the paper):
  f_P(u, v) = (m_P(u, v) - m_0) / (mean_v m_FedAvg(1, v) - m_0), one denominator per modality, taken from
     FedAvg, u = 1, recovered text.
  Floor: a group mean of f <= 0.1. H2-H5 are decided only in cells whose FedAvg group mean of f (recovered
     condition, same modality and u) lies above the floor; a comparison in a floor cell is "not decidable",
     leaves its Holm family, and the hypothesis is decided on the remaining comparisons ("undecided" for a
     modality if none remains). H1 (on the primary metric, not on f) has no floor rule.
  Differences are always treatment - reference: H1 m(u) - m(1); H2a f(rec) - f(oracle); H2b f(rec) - f(none);
     H3/H4/H5 f(protocol) - f(FedAvg).
  H1: one-sided Wilcoxon signed-rank (alternative: m(u) < m(1)) for u = 2, 4, 8, 16 vs 1, Holm over the
     comparisons of a modality; supported for a modality if all its comparisons are significant.
  H2a: two-sided 95% t interval of the paired differences, supported if both (all decidable) intervals lie
     within [-0.05, 0.05].
  H2b: one-sided Wilcoxon (f(rec) > f(none)), Holm over 2; supported if both significant; if neither is and
     both t intervals lie within the margin -> "text does not matter within the margin"; otherwise undecided.
  H3: two-sided 95% t interval within +-0.05 for every comparison (both modalities, both u).
  H4: one-sided 95% t lower bound (mean - t_{0.95, n-1} se) >= -0.05 for every comparison.
  H5: image: one-sided Wilcoxon (f(FedBN) < f(FedAvg)), Holm over 2, both significant; text: two-sided 95% t
     interval within +-0.05 for both u. Supported if both parts hold.
  Zero differences: Wilcoxon zero_method="wilcox" (zeros dropped; all zero -> p = 1); a zero-width interval
     (all differences equal) is [mean, mean], so an all-zero difference lies within the margin.
  "Within the margin" and "at or above -0.05" are inclusive.
  Every comparison also gets the paired two-sided 95% t interval, a per-seed table, and the sensitivity
  analysis: a linear mixed model d ~ 1 + (1 | seed) (statsmodels MixedLM, REML), whose estimate, 95% Wald
  interval (one-sided lower bound for H4) and p value (one-sided for directional tests; Holm within the same
  family) give a decision that is compared with the main decision (column mixed_agrees).
  ANOVA: per modality, f ~ C(protocol) * C(u) + C(victim) on the protocol experiment (recovered condition,
  the u levels at which a non-FedAvg protocol was run), type II sums of squares; decides nothing.
  Gate: if the image attack is not informative at B = 8 (--image-gate not_informative), all image parts of
  H1-H5 are "undecided (gate)".

AMBIGUITIES (the most literal reading is implemented; see also the report of the agent that wrote this):
  1. H3-H5 span both modalities but the floor rule speaks of "a modality": decisions are given per modality
     and overall; overall = decided on the remaining (decidable) modality parts.
  2. "Not supported" for an interval rule does not distinguish an interval that excludes the margin from
     one that is merely too wide (both are reported via ci_low / ci_high).
  3. The floor is checked on the FedAvg cell only, even when the treatment cell (e.g. FedBN images, which
     H5 expects to drop) is itself at the floor.
  4. The image m_0 is the mean of the mismatched control (src/score_images.py column A_mismatched: the
     victim's originals scored against the reconstructions of every other victim of the same factor
     combination) over the reference cell (FedAvg, u = 1, recovered). Override with --m0.
  5. H2b's third outcome ("text does not matter within the margin") is not a "supported/not supported"
     decision; it is output verbatim.

usage:
  python src/analyze.py test tidy.csv [more.csv] --out runs/analysis [--m0 image=0.5] [--image-gate informative]
  python src/analyze.py tidy-images runs/scores/images.csv --out tidy_images.csv [--condition recovered]
  python src/analyze.py gate runs/scores/images.csv [--by batch u init tag res] [--victims 3 4 5 6 7 8 9]
    (one gate row per group; the paper's gate uses the 7 seed-0 clients not used for tuning, at B = 8, 4, 2, 1)
"""
import argparse
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


def gate(A, A_mismatched, alpha=ALPHA, threshold=GATE_A):
    """Calibration gate of the paper. 'informative': one-sided Wilcoxon signed-rank test of A against the
    mismatched control (A_mismatched: the victim's own originals scored against the other victims'
    reconstructions), paired by victim, at alpha; 'identifies the patients': informative and mean A >= 0.90."""
    A, M = np.asarray(A, float), np.asarray(A_mismatched, float)
    ok = ~(np.isnan(A) | np.isnan(M))
    A, M = A[ok], M[ok]
    stat, p, nz = wilcoxon(A - M, "greater")
    informative = bool(p < alpha)
    return {"n": int(len(A)), "n_nonzero": nz, "mean_A": float(A.mean()) if len(A) else np.nan,
            "mean_A_mismatched": float(M.mean()) if len(M) else np.nan, "W": stat, "p": p,
            "informative": informative, "identifies": bool(informative and A.mean() >= threshold)}


# ---------------------------------------------------------------- data
def load_tidy(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    need = {"seed", "victim", "protocol", "u", "modality", "value"}
    if need - set(df.columns):
        raise SystemExit(f"missing columns: {sorted(need - set(df.columns))}")
    if "condition" not in df:
        df["condition"] = "recovered"
    df["condition"] = df["condition"].fillna("").astype(str).str.strip().str.lower().map(lambda c: COND_ALIASES.get(c, c))
    df["protocol"] = df["protocol"].astype(str).str.strip().str.lower()
    df["modality"] = df["modality"].astype(str).str.strip().str.lower()
    df["u"] = df["u"].astype(int)
    for c in ("seed", "victim"):  # 0, 0.0 and "0" must pair: integers where possible
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all() and (num == num.round()).all():
            df[c] = num.astype(int)
    df["key"] = df["seed"].astype(str) + ":" + df["victim"].astype(str)
    dup = df.duplicated(["key", "protocol", "u", "condition", "modality"])
    if dup.any():
        raise SystemExit(f"{dup.sum()} duplicate rows (same victim, cell and modality)")
    return df


def cell(df, mod, protocol="fedavg", u=1, condition="recovered"):
    return df[(df.modality == mod) & (df.protocol == protocol) & (df.u == u) & (df.condition == condition)]


def add_f(df, m0_override):
    """Adds column f; returns (df, info per modality)."""
    df = df.copy()
    df["f"] = np.nan
    info = {}
    for mod in df.modality.unique():
        ref = cell(df, mod)
        if ref.empty:
            info[mod] = {"note": "no FedAvg u=1 recovered cell: f undefined"}
            continue
        if mod in m0_override:
            m0, src = m0_override[mod], "cli"
        elif "m0" in df and ref["m0"].notna().any():
            m0, src = float(ref["m0"].mean()), "mean m0 of reference cell"
        else:
            info[mod] = {"note": "no m0 (column m0 or --m0): f undefined"}
            continue
        den = float(ref["value"].mean()) - m0
        sel = df.modality == mod
        df.loc[sel, "f"] = (df.loc[sel, "value"] - m0) / den
        info[mod] = {"m0": m0, "m0_source": src, "ref_mean": den + m0, "denominator": den, "n_ref": len(ref)}
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


def evaluate(df, image_gate="unknown"):
    comps, per_seed = [], []
    for c in comparisons(df):
        mod, metric = c["modality"], c["metric"]
        a = cell(df, mod, *c["t"])
        b = cell(df, mod, *c["r"])
        m = paired(a, b, metric)
        d, seeds = m["d"].to_numpy(), m["seed"].tolist()
        row = {k: c[k] for k in ("hypothesis", "modality", "u", "metric", "treatment", "reference", "kind", "alternative")}
        lo, hi = t_interval(d)
        row.update(n=len(d), mean_diff=d.mean() if len(d) else np.nan, sd_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
                   ci_low=lo, ci_high=hi)
        if c["floor_rule"]:
            fmean = cell(df, mod, "fedavg", c["u"], "recovered")["f"].mean()
            row.update(fedavg_mean_f=fmean, at_floor=bool(fmean <= FLOOR) if not np.isnan(fmean) else True)
        else:
            row.update(fedavg_mean_f=np.nan, at_floor=False)
        row["decidable"] = not row["at_floor"] and len(d) >= 2
        if c["kind"] == "wilcoxon":
            row["W"], row["p"], row["n_nonzero"] = wilcoxon(d, c["alternative"]) if len(d) else (np.nan, np.nan, 0)
        elif c["kind"] == "equivalence":
            row["within_margin"] = bool(lo >= -MARGIN and hi <= MARGIN) if not np.isnan(lo) else False
        else:
            row["lower_bound"] = t_lower(d)
            row["within_margin"] = bool(row["lower_bound"] >= -MARGIN) if not np.isnan(row["lower_bound"]) else False
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
    for col in ("p", "W", "n_nonzero", "within_margin", "lower_bound", "mixed_p", "mixed_ci_low", "mixed_ci_high",
                "mixed_lower_bound"):
        if col not in comp:
            comp[col] = np.nan
    # Holm within each family (hypothesis x modality), over the decidable comparisons only
    comp["p_holm"], comp["mixed_p_holm"] = np.nan, np.nan
    for (h, mod), g in comp[comp.kind == "wilcoxon"].groupby(["hypothesis", "modality"]):
        g = g[g.decidable]
        if len(g):
            comp.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
            comp.loc[g.index, "mixed_p_holm"] = holm(g["mixed_p"].to_numpy())
    comp["significant"] = np.where((comp.kind == "wilcoxon") & comp.decidable, comp["p_holm"] < ALPHA, np.nan)
    # sensitivity: does the mixed model reach the same comparison-level decision?
    agree = []
    for _, r in comp.iterrows():
        if not r["decidable"] or (np.isnan(r["mixed_est"]) if isinstance(r["mixed_est"], float) else False):
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

    def gated(mod):
        return mod.startswith("image") and image_gate == "not_informative"

    for (h, mod), g in comp.groupby(["hypothesis", "modality"], sort=False):
        dec = g[g.decidable]
        nd = len(g) - len(dec)
        tail = f"; {nd} comparison(s) at the floor, not decidable" if nd else ""
        if gated(mod):
            add(h, mod, "undecided (gate)", "image attack not informative at B = 8"); continue
        if h == "H1" and mod not in PRIMARY:
            add(h, mod, "reported, decides nothing", f"{int(g.significant.sum())}/{len(g)} Holm-significant"); continue
        if dec.empty:
            add(h, mod, "undecided", "no decidable comparison" + tail); continue
        k = len(dec)
        if g.kind.iloc[0] == "wilcoxon":
            s = int(dec.significant.sum())
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
            what = "lower bounds >= -0.05" if g.kind.iloc[0] == "noninferiority" else f"intervals within +-{MARGIN}"
            add(h, mod, "supported" if w == k else "not supported", f"{w}/{k} {what}" + tail)
    dec = pd.DataFrame(out)
    # overall decisions for hypotheses that span both modalities, on the remaining (decided) modality parts
    for h in ("H3", "H4", "H5"):
        g = dec[(dec.hypothesis == h) & dec.modality.isin(PRIMARY)]
        if g.empty:
            continue
        parts = g[~g.decision.str.startswith("undecided")]
        missing = sorted(set(PRIMARY) - set(g.modality))
        if parts.empty:
            d, why = "undecided", "no modality decidable"
        else:
            d = "supported" if (parts.decision == "supported").all() else "not supported"
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
    m0 = {}
    for s in a.m0 or []:
        k, v = s.split("=")
        m0[k.strip().lower()] = float(v)
    df, info = add_f(df, m0)
    comp, dec, per_seed = evaluate(df, a.image_gate)
    an = anova(df)
    os.makedirs(a.out, exist_ok=True)
    for name, t in (("comparisons", comp), ("decisions", dec), ("per_seed", per_seed), ("anova", an)):
        t.to_csv(os.path.join(a.out, name + ".csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 90)
    print("zero points and denominators:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if a.image_gate == "unknown":
        print("  (image gate not given: image hypotheses are tested as if the attack were informative at B = 8)")
    cols = ["hypothesis", "modality", "u", "treatment", "reference", "n", "mean_diff", "ci_low", "ci_high", "lower_bound",
            "p", "p_holm", "significant", "within_margin", "fedavg_mean_f", "decidable", "mixed_est", "mixed_agrees"]
    if comp.empty:
        print("\nno planned comparison is present in the data")
    else:
        print("\ncomparisons:\n" + comp[[c for c in cols if c in comp]].round(4).to_string(index=False))
        print("\ndecisions:\n" + dec.to_string(index=False))
    if not an.empty:
        print("\nANOVA (type II, decides nothing):\n" + an.round(4).to_string(index=False))
    print(f"\nwrote {a.out}/comparisons.csv, decisions.csv, per_seed.csv, anova.csv")


def tidy_from_image_scores(scores, condition=None):
    s = pd.read_csv(scores)
    cond = s["condition"] if condition is None and "condition" in s else (condition or "recovered")
    out = pd.DataFrame({"seed": s["seed"], "victim": s["victim"], "protocol": s.get("protocol", "fedavg"),
                        "u": s["u"], "condition": cond, "modality": "image", "value": s["A"],
                        "m0": s.get("A_mismatched", np.nan), "batch": s["batch"], "file": s["file"]})
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
    if a.victims:
        s = s[s.victim.isin(a.victims)]
    rows = []
    for key, g in s.groupby(a.by, dropna=False):
        r = gate(g["A"], g["A_mismatched"])
        rows.append({**dict(zip(a.by, key if isinstance(key, tuple) else (key,))), **r})
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test", help="decision rules of the hypothesis table")
    t.add_argument("inputs", nargs="+")
    t.add_argument("--out", default="runs/analysis")
    t.add_argument("--m0", nargs="*", help="override zero points, e.g. image=0.5 text=0.21")
    t.add_argument("--image-gate", choices=["informative", "not_informative", "unknown"], default="unknown")
    t.set_defaults(fn=cmd_test)
    ti = sub.add_parser("tidy-images", help="score_images.py CSV -> tidy rows (value = A, m0 = A_mismatched)")
    ti.add_argument("scores")
    ti.add_argument("--out", required=True)
    ti.add_argument("--condition", default=None, help="force a condition label (default: column 'condition')")
    ti.set_defaults(fn=cmd_tidy)
    g = sub.add_parser("gate", help="calibration gate on a score_images.py CSV")
    g.add_argument("scores")
    g.add_argument("--by", nargs="+", default=["batch", "u", "init", "tag", "res"])
    g.add_argument("--victims", type=int, nargs="*", help="gate clients only (e.g. 3 4 5 6 7 8 9)")
    g.set_defaults(fn=cmd_gate)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
