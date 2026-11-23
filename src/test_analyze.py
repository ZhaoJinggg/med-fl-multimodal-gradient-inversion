"""Synthetic tests with known answers for src/analyze.py (no data files needed).

usage: python src/test_analyze.py      (prints the tables and "ALL TESTS PASSED")
"""
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an  # noqa: E402

rng = np.random.default_rng(0)
VICTIMS = [(s, v) for s in range(1, 6) for v in range(4)]  # 20 victims, 4 per seed
TMP = tempfile.gettempdir()


def rows(mod, protocol, u, values, condition="recovered", m0=np.nan, batch=8):
    return [{"seed": s, "victim": v, "protocol": protocol, "u": u, "batch": batch, "condition": condition,
             "modality": mod, "value": float(x), "m0": m0} for (s, v), x in zip(VICTIMS, values)]


def noise(sd, n=len(VICTIMS)):
    return rng.normal(0, sd, n)


def cross_rows(values, protocol="fedavg", u=1, condition="recovered", off=0.5, off_sd=0.05, batch=8, group=None):
    """Cross matrix with diagonal = the victims' A and off-diagonal = chance-level scores."""
    out = []
    for i, (vs, vv) in enumerate(VICTIMS):
        for j, (ws, wv) in enumerate(VICTIMS):
            a = values[i] if i == j else off + rng.normal(0, off_sd)
            out.append({"group": group if group is not None else f"{protocol}{u}{condition}B{batch}", "protocol": protocol,
                        "condition": condition, "u": u, "batch": batch, "v_seed": vs, "v_victim": vv,
                        "w_seed": ws, "w_victim": wv, "diagonal": i == j, "A": a})
    return out


def synthetic():
    """The closed design. Text (batch 8): m0 = 0.2, FedAvg u=1 mean ~0.8 (f = 1 per 0.6 of ROUGE-L).
    Image (u = 1): m0 = 0.5, FedAvg batch 2 mean ~0.85 (reference cell; f = 1 per 0.35 of A)."""
    R, C = [], []
    # text, FedAvg: u=2 exactly equal to u=1 (zero differences); u=8 at the floor (ROUGE-L ~ m0)
    base_txt = 0.8 + noise(0.02)
    txt = {1: base_txt, 2: base_txt, 4: base_txt + noise(0.004), 8: 0.2 + noise(0.01), 16: base_txt - 0.3 + noise(0.01)}
    for u, v in txt.items():
        R += rows("text", "fedavg", u, v, m0=0.2)
    for u in (4, 8):
        # H3: SCAFFOLD equal within noise; u=8 is a floor cell -> decided on u=4 only
        R += rows("text", "scaffold", u, txt[u] + noise(0.002))
        # H4: FedProx lower by 0.12 ROUGE-L at u=4 (f -0.2) -> contradicted
        R += rows("text", "fedprox", u, txt[u] - (0.12 if u == 4 else 0) + noise(0.002))
        # H5b: FedBN equal -> supported
        R += rows("text", "fedbn", u, txt[u] + noise(0.002))
    # text at the other batch sizes and the exact reference, as src/score_text.py labels them: in no test
    R += rows("text", "fedavg", 1, base_txt + 0.1, condition="B2", batch=2, m0=0.2)
    for r in rows("text", "fedavg", 1, base_txt + 0.15, condition="exact_B4", m0=0.2):
        r.pop("batch")
        R.append(r)
    # image, FedAvg, recovered text: known decline with the batch size -> H6 supported
    b2 = 0.85 + noise(0.02)
    img = {1: b2 + 0.08 + noise(0.01), 2: b2, 4: b2 - 0.15 + noise(0.01), 8: b2 - 0.25 + noise(0.01)}
    for b, v in img.items():
        R += rows("image", "fedavg", 1, v, m0=0.5, batch=b)
        C += cross_rows(v, batch=b)
    # H2a: oracle == recovered at batch 2 (zero width) -> supported; H2b: no text lower by 0.1 A -> supported
    R += rows("image", "fedavg", 1, b2, condition="oracle", batch=2)
    R += rows("image", "fedavg", 1, b2 - 0.1 + noise(0.01), condition="no_text", batch=2)
    # H5a: FedBN lower by 0.1 A at batches 1 and 2 -> supported
    for b in (1, 2):
        R += rows("image", "fedbn", 1, img[b] - 0.1 + noise(0.01), batch=b)
    # a batch-8 oracle cell (calibration-style): in no test
    R += rows("image", "fedavg", 1, img[8], condition="oracle", batch=8)
    s1 = 0.6 + noise(0.02)  # secondary metric: reported only
    for b in (1, 2, 4, 8):
        R += rows("image_ssim", "fedavg", 1, s1 - 0.01 * b + noise(0.005), batch=b)
    return pd.DataFrame(R), pd.DataFrame(C)


def run(df, cross, m0=None, stratify=True):
    p, pc = os.path.join(TMP, "test_analyze_tidy.csv"), os.path.join(TMP, "test_analyze_cross.csv")
    df.to_csv(p, index=False)
    cross.to_csv(pc, index=False)
    d = an.load_tidy([p])
    c = an.load_cross([pc])
    cache = {}
    d, info = an.add_f(d, m0 or {}, c, stratify, cache)
    comp, dec, ps = an.evaluate(d, c, m0 or {}, info, stratify, cache)
    return d, info, comp, dec, ps


def decision(dec, h, mod):
    return dec[(dec.hypothesis == h) & (dec.modality == mod)].decision.iloc[0]


def reason(dec, h, mod):
    return dec[(dec.hypothesis == h) & (dec.modality == mod)].reason.iloc[0]


def check(dec, exp):
    for (h, mod), e in exp.items():
        got = decision(dec, h, mod)
        assert got == e, f"{h} {mod}: expected {e!r}, got {got!r}"


COUNTS20 = [0] * 2 + [1] + [3] * 2 + [4] * 5 + [6] * 2 + [7] * 4 + [8] * 4  # 20 victims, 552,960 stratified perms
GATE_COUNTS = [0, 0, 8, 8, 8, 3, 6]  # seed 0, gate clients 3-9


def label_matrix(counts, same=0.7, other=0.5, sd=0.0, diag=None, g=None):
    """Cross matrix with a pure label effect: A(v, w) = `same` if v and w have the same label count, else
    `other` (+ noise); `diag` overrides the diagonal (a true image effect)."""
    g = g if g is not None else rng
    c = np.asarray(counts)
    M = np.where(c[:, None] == c[None, :], same, other) + g.normal(0, sd, (len(c), len(c)))
    if diag is not None:
        np.fill_diagonal(M, diag)
    return M


def cross_from_matrix(M, counts, protocol="fedavg", u=1, condition="recovered", M_lab=None, batch=8):
    out = []
    for i, (vs, vv) in enumerate(VICTIMS):
        for j, (ws, wv) in enumerate(VICTIMS):
            out.append({"group": f"{protocol}{u}{condition}B{batch}", "protocol": protocol, "condition": condition, "u": u,
                        "batch": batch, "v_seed": vs, "v_victim": vv, "v_n_abnormal": counts[i], "w_seed": ws, "w_victim": wv,
                        "w_n_abnormal": counts[j], "diagonal": i == j, "A": M[i, j],
                        "A_lab": M_lab[i, j] if M_lab is not None else np.nan})
    return out


def reg_pairs(nv=20, B=8, beta=0.3, delta=0.0, sd=0.02, seed=1):
    """Pairs rows (attack + mismatched) with A_i(v, w) = alpha_i + beta * x_iw + delta * [w = v] + noise."""
    g = np.random.default_rng(seed)
    counts = g.integers(0, B + 1, nv)
    labs = [np.array([1] * k + [0] * (B - k)) for k in counts]
    rows = []
    for v in range(nv):
        for i in range(B):
            alpha = g.normal(0.5, 0.1)
            for w in range(nv):
                x = float(np.mean(labs[w] == labs[v][i]))
                rows.append({"group": 0, "seed": 0, "victim": v, "pair": i, "kind": "attack" if w == v else "mismatched",
                             "partner_seed": 0 if w != v else np.nan, "partner_victim": w if w != v else np.nan,
                             "x_share": x, "A": alpha + beta * x + (delta if w == v else 0) + g.normal(0, sd)})
    return pd.DataFrame(rows), counts


def no_ref_cross(cross):
    """The cross rows without those of the image reference cell (FedAvg, u = 1, batch 2, recovered)."""
    return cross[~((cross.protocol == "fedavg") & (cross.u == 1) & (cross.batch == 2) & (cross.condition == "recovered"))]


def label_controls():
    """Stratified permutation test, sign-flip test, label regression, reference-cell preconditions, tidy."""
    # stratified enumeration counts
    assert an.n_stratified_perms([0, 0, 1, 1, 1, 2, 3]) == 12
    assert an.n_stratified_perms([5, 5, 5, 5, 9, 9, 9]) == 144
    assert an.n_stratified_perms(COUNTS20) == 552_960
    Mr = 0.5 + rng.normal(0, 0.05, (7, 7))
    r = an.stratified_perm_test(Mr, [0, 0, 1, 1, 1, 2, 3])
    assert r["exact"] and r["n_perm"] == 12 and r["n_eval"] == 12 and abs(r["p_min"] - 1 / 12) < 1e-12 and r["n_contrib"] == 5
    r = an.stratified_perm_test(Mr, [5, 5, 5, 5, 9, 9, 9])
    assert r["exact"] and r["n_eval"] == 144 and r["n_contrib"] == 7
    # one stratum = the unstratified exact test
    assert abs(an.stratified_perm_test(Mr, [0] * 7)["p"] - an.perm_test(Mr)[1]) < 1e-12
    # gate clients: even a perfect diagonal cannot reach p <= 0.05 (12 permutations)
    Mg = label_matrix(GATE_COUNTS, diag=0.95)
    rg = an.stratified_perm_test(Mg, GATE_COUNTS)
    assert rg["n_perm"] == 12 and abs(rg["p"] - 1 / 12) < 1e-12 and rg["n_contrib"] == 5
    print("gate clients (counts 0,0,8,8,8,3,6), perfect diagonal: stratified p =", round(rg["p"], 4), "(= 1/12)")
    # 20 victims: all 552,960 stratified permutations are enumerated (the paper's batch-8 floor test)
    r20 = an.stratified_perm_test(label_matrix(COUNTS20, diag=0.9), COUNTS20)
    assert r20["exact"] and r20["n_perm"] == 552_960 and r20["n_eval"] == 552_960 and abs(r20["p"] - 1 / 552_960) < 1e-12
    assert r20["n_contrib"] == 19  # victim with count 1 is alone in its stratum
    # Monte Carlo path (used when there are more than exact_max permutations: batch sizes 4, 2, 1)
    rmc = an.stratified_perm_test(label_matrix(COUNTS20, diag=0.9), COUNTS20, exact_max=50_000)
    assert not rmc["exact"] and rmc["n_eval"] == 100_001 and abs(rmc["p"] - 1 / 100_001) < 1e-12 and rmc["n_contrib"] == 19
    # permutations stay within strata on both paths: a matrix that is +inf off-stratum would break the mean
    Minf = label_matrix(COUNTS20, same=0.5, other=np.inf)
    for kw in ({}, {"exact_max": 50_000}):
        rinf = an.stratified_perm_test(Minf, COUNTS20, **kw)
        assert np.isfinite(rinf["T"]) and rinf["p"] == 1.0

    # pure label effect: passes the unstratified test, FAILS the stratified one
    for sd, name in ((0.0, "exact"), (0.02, "noisy")):
        ML = label_matrix(COUNTS20, sd=sd, g=np.random.default_rng(7))
        pu = an.perm_test(ML)[1]
        rs = an.stratified_perm_test(ML, COUNTS20)
        print(f"pure label effect ({name}): unstratified p = {pu:.2e}, stratified p = {rs['p']:.3f}, effect {rs['effect']:.4f}")
        assert pu <= 0.05 and rs["p"] > 0.05 and abs(rs["effect"]) < 0.03
        assert not an.gate(ML, strata=np.array(COUNTS20))["informative_strat"] and an.gate(ML)["informative"]
    # true diagonal effect on top of the label effect: passes both
    MD = label_matrix(COUNTS20, sd=0.02, diag=0.9 + rng.normal(0, 0.02, 20))
    rs = an.stratified_perm_test(MD, COUNTS20)
    assert an.perm_test(MD)[1] <= 0.05 and rs["p"] <= 0.05 and abs(rs["effect"] - 0.2) < 0.03

    # sign-flip test
    assert abs(an.sign_flip_test(np.full(7, 0.1))[1] - 1 / 128) < 1e-12
    m, p, n, ex = an.sign_flip_test(rng.uniform(0.01, 0.2, 20))
    assert ex and n == 2 ** 20 and abs(p - 2 ** -20) < 1e-15
    Ds = rng.normal(0, 1, 10)
    import itertools
    brute = np.mean([np.dot(s, Ds) >= Ds.sum() - 1e-12 for s in itertools.product([-1, 1], repeat=10)])
    assert abs(an.sign_flip_test(Ds)[1] - brute) < 1e-12
    assert an.sign_flip_test(np.array([0.1, -0.1, 0.1, -0.1]))[1] > 0.5
    m, p, n, ex = an.sign_flip_test(rng.normal(0.5, 0.1, 25))
    assert not ex and n == 100_001 and p == 1 / 100_001
    # gate with A_lab: D_v positive for all 7 -> sign-flip p = 1/128
    Ll = label_matrix(GATE_COUNTS, same=0.5, other=0.5, sd=0.01, diag=0.8)
    gl = an.gate(Mg, strata=np.array(GATE_COUNTS), M_lab=Ll)
    assert abs(gl["p_signflip_lab"] - 1 / 128) < 1e-12 and gl["exact_signflip_lab"] and abs(gl["p_wilcoxon_lab"] - 1 / 128) < 1e-12
    assert gl["n_perm_strat_lab"] == 12 and abs(gl["p_perm_lab"] - 1 / 5040) < 1e-12 and not gl["informative_strat"]
    print({k: v for k, v in gl.items() if "lab" in k or "strat" in k})

    # label-distance regression recovers a known slope
    P0, _ = reg_pairs(beta=0.3, delta=0.0)
    for incl in (True, False):
        lr = an.label_regression(P0, include_diagonal=incl, n_boot=500)
        assert abs(lr["slope"] - 0.3) < 0.01 and abs(lr["excess"]) < 0.01 and lr["slope_lo"] < 0.3 < lr["slope_hi"], lr
        assert lr["excess_lo"] < 0 < lr["excess_hi"] and lr["n_victims"] == 20
    P1, _ = reg_pairs(beta=0.3, delta=0.2)
    lo = an.label_regression(P1, include_diagonal=False, n_boot=500)
    li = an.label_regression(P1, include_diagonal=True, n_boot=500)
    print("regression delta=0.2: offdiag slope %.4f excess %.4f [%.4f, %.4f]; with diagonal slope %.4f excess %.4f"
          % (lo["slope"], lo["excess"], lo["excess_lo"], lo["excess_hi"], li["slope"], li["excess"]))
    assert abs(lo["slope"] - 0.3) < 0.01 and abs(lo["excess"] - 0.2) < 0.01 and lo["excess_lo"] < 0.2 < lo["excess_hi"]
    assert 0.15 < li["excess"] < 0.2  # the diagonal enters its own fixed effect (shrinks by about 1/20)
    P2, _ = reg_pairs(beta=0.0, delta=0.0)
    assert abs(an.label_regression(P2, n_boot=200)["slope"]) < 0.01

    # `test`: pure label effect in the image reference cell (batch 2) -> the stratified test fails -> image f not
    # computed, H2a and H2b "not decidable"; H5a at batch 2 has the same control cell (not decidable) and is
    # decided on batch 1; H6 does not depend on the floor. --no-stratify (unstratified) passes the cell.
    df, cross = synthetic()
    rest = no_ref_cross(cross)
    ref_vals = df[(df.modality == "image") & (df.protocol == "fedavg") & (df.u == 1) & (df.batch == 2)
                  & (df.condition == "recovered")].value.to_numpy()
    MLr = label_matrix(COUNTS20, same=ref_vals.mean(), other=0.5, sd=0.0)
    np.fill_diagonal(MLr, ref_vals.mean())
    cross_l = pd.concat([rest, pd.DataFrame(cross_from_matrix(MLr, COUNTS20, batch=2))])
    d_s, info_s, comp_s, dec_s, _ = run(df, cross_l)
    assert info_s["image"]["ref_passes"] is False and info_s["image"]["ref_test_p_unstratified"] <= 0.05
    assert info_s["image"]["ref_test_method"].startswith("stratified") and d_s[d_s.modality == "image"].f.isna().all()
    check(dec_s, {("H2a", "image"): "not decidable", ("H2b", "image"): "not decidable", ("H5a", "image"): "supported",
                  ("H6", "image"): "supported", ("H3", "text"): "supported"})
    assert "not decidable" in reason(dec_s, "H5a", "image")
    _, info_u, _, dec_u, _ = run(df, cross_l, stratify=False)
    assert info_u["image"]["ref_passes"] and "unstratified: --no-stratify" in info_u["image"]["ref_test_method"]
    assert decision(dec_u, "H2a", "image") == "supported"
    # a true image effect in the reference cell with strata: passes the stratified test (enumerated: 552,960)
    MDr = label_matrix(COUNTS20, same=0.55, other=0.5, sd=0.02)
    np.fill_diagonal(MDr, ref_vals)
    cross_d = pd.concat([rest, pd.DataFrame(cross_from_matrix(MDr, COUNTS20, batch=2))])
    _, info_d, comp_d, dec_d, _ = run(df, cross_d)
    assert info_d["image"]["ref_passes"] and info_d["image"]["ref_test_p"] <= 1e-4 and info_d["image"]["ref_n_contrib"] == 19
    check(dec_d, {("H2a", "image"): "supported", ("H2b", "image"): "supported", ("H5a", "image"): "supported"})
    h2 = comp_d[comp_d.hypothesis.isin(["H2a", "H2b"])]
    assert h2.floor_p_unstratified.notna().all() and h2.floor_method.str.startswith("stratified").all()

    # text reference cell (FedAvg u=1, batch 8) at its m0 -> text f not computed; H3, H4, H5b "not decidable";
    # H1 (ROUGE-L, no floor rule) is still decided; the image hypotheses are untouched
    df_t = df.copy()
    sel = (df_t.modality == "text") & (df_t.protocol == "fedavg") & (df_t.u == 1) & (df_t.batch == 8) & (df_t.condition == "recovered")
    df_t.loc[sel, "value"] = 0.2 + rng.normal(0, 0.01, sel.sum())
    d_t, info_t, comp_t, dec_t, _ = run(df_t, cross)
    assert info_t["text"]["ref_passes"] is False and d_t[d_t.modality == "text"].f.isna().all()
    for h in ("H3", "H4", "H5b"):
        assert decision(dec_t, h, "text") == "not decidable", (h, decision(dec_t, h, "text"))
    assert decision(dec_t, "H1", "text") == "not supported" and decision(dec_t, "H2a", "image") == "supported"
    assert info_d["text"]["ref_passes"] is True

    # tidy-images: batch and condition as scored
    ps = os.path.join(TMP, "tidy_scores.csv")
    pd.DataFrame({"seed": [1, 1, 1, 1], "victim": [0, 0, 0, 0], "protocol": "fedavg", "u": 1, "batch": [8, 4, 8, 2],
                  "condition": ["recovered", "recovered", "oracle", "notext"], "A": 0.9, "A_mismatched": 0.5,
                  "file": ["a", "b", "c", "d"], "n_abnormal": [3, 1, 3, 1]}).to_csv(ps, index=False)
    t = an.tidy_from_image_scores(ps)
    assert list(t.condition) == ["recovered", "recovered", "oracle", "none"] and list(t.batch) == [8, 4, 8, 2] and "n_abnormal" in t
    pt = os.path.join(TMP, "tidy_scores_tidy.csv")
    t.to_csv(pt, index=False)
    assert len(an.load_tidy([pt])) == 4  # four different cells: no duplicate

    # gate command with A_lab, strata and --regression (7 gate clients, file names repeat across "dirs")
    P7, cnt7 = reg_pairs(nv=7, beta=0.3, delta=0.1, seed=3)
    P7["victim"] += 3
    P7.loc[P7.kind == "mismatched", "partner_victim"] += 3
    cr, sc = [], []
    for v in range(3, 10):
        sc.append({"file": "same_name", "seed": 0, "victim": v, "top1": 0.5, "group": 0})
        for w in range(3, 10):
            a_ = P7[(P7.victim == v) & ((P7.kind == "attack") if v == w else (P7.partner_victim == w))]["A"].mean()
            cr.append({"group": 0, "dir": "x", "protocol": "fedavg", "condition": "oracle", "batch": 8, "u": 1, "init": "mean",
                       "tag": "_gate", "res": 128, "v_seed": 0, "v_victim": v, "v_file": "same_name", "v_n_abnormal": cnt7[v - 3],
                       "w_seed": 0, "w_victim": w, "w_file": "same_name", "w_n_abnormal": cnt7[w - 3], "diagonal": v == w,
                       "A": a_, "A_lab": a_})
    ps_, pc_, pp_, po_ = (os.path.join(TMP, n) for n in ("g7.csv", "g7_cross.csv", "g7_pairs.csv", "g7_out.csv"))
    pd.DataFrame(sc).to_csv(ps_, index=False)
    pd.DataFrame(cr).to_csv(pc_, index=False)
    P7.to_csv(pp_, index=False)
    an.main(["gate", ps_, "--victims", *map(str, range(3, 10)), "--seed", "0", "--regression", "--n-boot", "300", "--out", po_])
    go = pd.read_csv(po_).iloc[0]
    ref = an.label_regression(P7, n_boot=300)
    assert go.n == 7 and abs(go.reg_slope - ref["slope"]) < 1e-12 and abs(go.reg_excess - ref["excess"]) < 1e-12
    assert abs(go.top1_share - 0.5) < 1e-12 and "p_strat" in go and "p_signflip_lab" in go and "reg_offdiag_slope" in go
    print("gate --regression:", {k: round(float(go[k]), 4) for k in ("reg_slope", "reg_excess", "reg_excess_lo", "reg_excess_hi",
                                                                     "reg_offdiag_slope", "reg_offdiag_excess", "p_strat", "n_perm_strat")})


def hypothesis_table():
    """The decision rules of the closed design (H1, H2a, H2b, H3, H4, H5a, H5b, H6) on synthetic cells."""
    # condition labels of src/score_text.py: the batch part moves to the batch column
    assert an.split_batch("b4") == ("", 4) and an.split_batch("oracle_b2") == ("oracle", 2)
    assert an.split_batch("exact_b4") == ("exact", 4) and an.split_batch("no_text") == ("no_text", None)
    assert an.split_batch("bnlocal") == ("bnlocal", None) and an.split_batch("exact_b1_opti") == ("exact_opti", 1)

    df, cross = synthetic()
    d, info, comp, dec, ps = run(df, cross)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(info)
    print(comp[["hypothesis", "modality", "u", "batch", "n", "mean_diff", "ci_low", "ci_high", "lower_bound", "p", "p_holm",
                "significant", "within_margin", "outside_margin", "floor_p", "decidable", "mixed_agrees"]].round(4).to_string(index=False))
    print(dec.to_string(index=False))
    # cells: "B2" -> recovered at batch 2; "exact_B4" -> exact at batch 4; neither enters a test
    tb = d[d.modality == "text"]
    assert len(tb[(tb.batch == 2) & (tb.condition == "recovered")]) == 20 and len(tb[(tb.batch == 4) & (tb.condition == "exact")]) == 20
    # reference cells and f
    ref_i = d[(d.modality == "image") & (d.protocol == "fedavg") & (d.u == 1) & (d.batch == 2) & (d.condition == "recovered")]
    ref_t = d[(d.modality == "text") & (d.protocol == "fedavg") & (d.u == 1) & (d.batch == 8) & (d.condition == "recovered")]
    assert abs(ref_i.f.mean() - 1) < 1e-12 and abs(info["image"]["m0"] - 0.5) < 1e-12 and info["image"]["ref_passes"] is True
    assert abs(ref_t.f.mean() - 1) < 1e-12 and abs(info["text"]["m0"] - 0.2) < 1e-12 and info["text"]["ref_passes"] is True
    assert info["image"]["ref_cell"] == "fedavg u=1 B=2" and info["text"]["ref_cell"] == "fedavg u=1 B=8"
    assert d[d.modality == "image_ssim"].f.isna().all()
    check(dec, {("H1", "text"): "not supported", ("H2a", "image"): "supported", ("H2b", "image"): "supported",
                ("H3", "text"): "supported", ("H4", "text"): "contradicted", ("H5a", "image"): "supported",
                ("H5b", "text"): "supported", ("H6", "image"): "supported", ("H6", "image_ssim"): "reported, decides nothing"})
    assert set(dec.hypothesis) == set(an.PLANNED) and "overall" not in set(dec.modality)
    # exactly the planned comparisons, in the planned cells
    n_prim = comp[comp.modality.isin(an.PRIMARY)].groupby("hypothesis").size().to_dict()
    assert n_prim == an.PLANNED, n_prim
    assert set(comp[comp.hypothesis == "H1"].u) == {2, 4, 8, 16} and (comp[comp.hypothesis == "H1"].batch == 8).all()
    assert set(comp[(comp.hypothesis == "H6") & (comp.modality == "image")].batch) == {2, 4, 8}
    assert set(comp[(comp.hypothesis == "H5a") & (comp.modality == "image")].batch) == {1, 2}
    assert (comp[comp.hypothesis.isin(["H2a", "H2b"])].batch == 2).all()
    assert (comp[comp.hypothesis.isin(["H1", "H5a", "H6"])].metric == "value").all()
    assert (comp[comp.hypothesis.isin(["H2a", "H2b", "H3", "H4", "H5b"])].metric == "f").all()
    # text floor cell (u=8): Wilcoxon of ROUGE-L vs m0 fails -> not decidable, leaves the families
    t8 = comp[(comp.modality == "text") & (comp.u == 8) & comp.hypothesis.isin(["H3", "H4", "H5b"])]
    assert len(t8) == 3 and (~t8.decidable.astype(bool)).all() and (t8.floor_p > 0.05).all()
    assert "not decidable" in reason(dec, "H3", "text")
    # image floor tests use the permutation test of the FedAvg control cell and pass
    i5 = comp[(comp.modality == "image") & (comp.hypothesis == "H5a")]
    assert i5.floor_method.str.startswith("permutation").all() and (i5.floor_p <= 0.05).all() and i5.significant.all()
    assert list(i5.control) == ["fedavg u=1 B=1", "fedavg u=1 B=2"]
    # H1 and H6 have no floor rule
    h1 = comp[comp.hypothesis == "H1"]
    assert h1.decidable.astype(bool).all() and h1.p_holm.notna().all() and h1.floor_p.isna().all()
    assert h1[h1.u == 2].p.iloc[0] == 1.0
    h6 = comp[(comp.hypothesis == "H6") & (comp.modality == "image")]
    assert h6.decidable.astype(bool).all() and h6.significant.all() and h6.floor_p.isna().all()
    # no multiplicity adjustment: a Wilcoxon comparison is significant iff its own p is below alpha
    wd = comp[(comp.kind == "wilcoxon") & comp.decidable.astype(bool)]
    assert (wd.significant.astype(bool) == (wd.p < 0.05)).all() and (wd.p_holm >= wd.p).all()
    # H2a: zero differences -> zero-width interval within the margin; H2b: one comparison, Holm = p
    z = comp[comp.hypothesis == "H2a"].iloc[0]
    assert z.ci_low == 0 and z.ci_high == 0 and z.within_margin
    b = comp[comp.hypothesis == "H2b"].iloc[0]
    assert b.p == b.p_holm and b.significant

    # image FedAvg batch 1 at chance in its cross matrix -> H5a at batch 1 not decidable, decided on batch 2;
    # H6 (batch 1 is its reference level) does not depend on the floor
    at1 = (cross.protocol == "fedavg") & (cross.batch == 1)
    c2 = pd.concat([cross[~at1], pd.DataFrame(cross_rows(0.45 + noise(0.01), batch=1))])
    _, _, comp2, dec2, _ = run(df, c2)
    e1 = comp2[(comp2.modality == "image") & (comp2.hypothesis == "H5a")]
    assert list(e1.decidable.astype(bool)) == [False, True], e1[["batch", "floor_p"]]
    assert e1.p_holm.isna().tolist() == [True, False] and e1.p_holm.iloc[1] == e1.p.iloc[1]  # family of one
    assert decision(dec2, "H5a", "image") == "supported" and "not decidable" in reason(dec2, "H5a", "image")
    assert decision(dec2, "H6", "image") == "supported"

    # image reference cell (batch 2) at chance -> image f not computed; H2a and H2b not decidable;
    # H5a decided on batch 1 alone; H6 and the text hypotheses untouched
    c3 = pd.concat([no_ref_cross(cross), pd.DataFrame(cross_rows(0.45 + noise(0.01), batch=2))])
    d3, info3, comp3, dec3, _ = run(df, c3)
    assert info3["image"]["ref_passes"] is False and d3[d3.modality == "image"].f.isna().all()
    check(dec3, {("H2a", "image"): "not decidable", ("H2b", "image"): "not decidable", ("H5a", "image"): "supported",
                 ("H6", "image"): "supported", ("H3", "text"): "supported", ("H4", "text"): "contradicted"})
    # no cross matrix at all: no image floor test can be run and no point rule replaces it
    d3b, info3b, comp3b, dec3b, _ = run(df, cross[cross.batch == 99])
    assert info3b["image"]["ref_passes"] is None and d3b[d3b.modality == "image"].f.isna().all()
    check(dec3b, {("H2a", "image"): "not decidable", ("H5a", "image"): "not decidable", ("H6", "image"): "supported"})
    assert comp3b[comp3b.hypothesis == "H5a"].floor_method.str.startswith("not tested").all()

    # both text control cells at the floor -> H3 and H5b not decidable
    df4 = df.copy()
    sel = (df4.modality == "text") & (df4.protocol == "fedavg") & (df4.u == 4)
    df4.loc[sel, "value"] = 0.2 + rng.normal(0, 0.01, sel.sum())
    _, _, _, dec4, _ = run(df4, cross)
    assert decision(dec4, "H3", "text") == "not decidable" and decision(dec4, "H5b", "text") == "not decidable"

    # equal groups, large noise -> inconclusive; a real shift beyond the margin -> contradicted
    df5 = df.copy()
    sel = (df5.modality == "text") & (df5.protocol == "scaffold")
    df5.loc[sel, "value"] += rng.normal(0, 0.1, sel.sum())
    _, _, _, dec5, _ = run(df5, cross)
    assert decision(dec5, "H3", "text") == "inconclusive", decision(dec5, "H3", "text")
    df6 = df.copy()
    df6.loc[sel, "value"] += 0.1
    _, _, _, dec6, _ = run(df6, cross)
    assert decision(dec6, "H3", "text") == "contradicted"
    # H4 inconclusive: lower bound below -0.05 but the upper end of the two-sided interval above -0.05
    df7 = df.copy()
    sel7 = (df7.modality == "text") & (df7.protocol == "fedprox") & (df7.u == 4)
    fa4 = df7[(df7.modality == "text") & (df7.protocol == "fedavg") & (df7.u == 4)].value.to_numpy()
    zz = np.linspace(-1, 1, len(VICTIMS))
    zz = (zz - zz.mean()) / zz.std(ddof=1)  # standardized offsets: f difference mean -0.05, sd 0.1
    den = info["text"]["denominator"]
    df7.loc[sel7, "value"] = fa4 + den * (-0.05 + 0.1 * zz)
    _, _, comp7, dec7, _ = run(df7, cross)
    r7 = comp7[(comp7.hypothesis == "H4") & (comp7.u == 4)].iloc[0]
    assert abs(r7.mean_diff + 0.05) < 1e-9 and abs(r7.sd_diff - 0.1) < 1e-9 and r7.lower_bound < -0.05 < r7.ci_high
    assert decision(dec7, "H4", "text") == "inconclusive"
    # H4 supported: FedProx equal to FedAvg at u=4
    df7b = df.copy()
    df7b.loc[sel7, "value"] = fa4 + rng.normal(0, 0.002, len(fa4))
    assert decision(run(df7b, cross)[3], "H4", "text") == "supported"

    # H2b third outcome and "inconclusive"
    df8 = df.copy()
    s8 = (df8.modality == "image") & (df8.batch == 2) & (df8.condition == "no_text")
    df8.loc[s8, "value"] = ref_i.value.to_numpy() + rng.normal(0, 0.002, s8.sum())
    assert decision(run(df8, cross)[3], "H2b", "image") == "text does not matter within the margin"
    df8.loc[s8, "value"] = ref_i.value.to_numpy() + rng.normal(0, 0.1, s8.sum())
    assert decision(run(df8, cross)[3], "H2b", "image") == "inconclusive"
    # H5a not supported if FedBN lowers nothing at batch 2; H6 not supported if batch 4 does not decline
    df9 = df.copy()
    s9 = (df9.modality == "image") & (df9.protocol == "fedbn") & (df9.batch == 2)
    df9.loc[s9, "value"] = ref_i.value.to_numpy() + rng.normal(0, 0.01, s9.sum())
    s9b = (df9.modality == "image") & (df9.protocol == "fedavg") & (df9.batch == 4) & (df9.condition == "recovered")
    b1 = df9[(df9.modality == "image") & (df9.protocol == "fedavg") & (df9.batch == 1) & (df9.condition == "recovered")].value.to_numpy()
    df9.loc[s9b, "value"] = b1 + rng.normal(0, 0.01, s9b.sum())
    dec9 = run(df9, cross)[3]
    assert decision(dec9, "H5a", "image") == "not supported" and decision(dec9, "H6", "image") == "not supported"
    # a hypothesis with fewer comparisons than planned says so
    dec10 = run(df[~((df.modality == "text") & (df.u == 16))], cross)[3]
    assert "plans 4" in reason(dec10, "H1", "text")
    # --m0 override
    _, info11, _, _, _ = run(df, cross, m0={"text": 0.0})
    assert info11["text"]["m0"] == 0.0 and info11["text"]["m0_source"] == "cli"
    # cells table: every cell once, floor tests of the primary modalities
    ct = an.cells_table(d, an.load_cross([os.path.join(TMP, "test_analyze_cross.csv")]))
    assert not ct.duplicated(["modality", "protocol", "u", "batch", "condition"]).any()
    row = lambda mod, proto, u, b, cond="recovered": ct[(ct.modality == mod) & (ct.protocol == proto) & (ct.u == u)
                                                        & (ct.batch == b) & (ct.condition == cond)].iloc[0]
    assert row("text", "fedavg", 8, 8).above_floor == False and row("text", "fedavg", 4, 8).above_floor == True  # noqa: E712
    assert row("image", "fedavg", 1, 8).above_floor == True and abs(row("image", "fedavg", 1, 2).mean_f - 1) < 1e-12  # noqa: E712
    assert row("image", "fedbn", 1, 1).floor_method.startswith("not tested")  # no cross matrix in the synthetic data
    # ANOVA: the text of the protocol experiment only
    tab = an.anova(d)
    print(tab.round(4).to_string(index=False))
    assert set(tab.modality) == {"text"} and tab[tab.source == "C(protocol)"].p.iloc[0] < 1e-6 and tab.n_rows.iloc[0] == 160
    # the command itself
    out = os.path.join(TMP, "test_analyze_out")
    an.main(["test", os.path.join(TMP, "test_analyze_tidy.csv"), "--cross", os.path.join(TMP, "test_analyze_cross.csv"), "--out", out])
    assert all(os.path.exists(os.path.join(out, n + ".csv")) for n in ("cells", "comparisons", "decisions", "per_seed", "anova"))


def main():
    # unit checks
    assert np.allclose(an.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert an.t_interval(np.zeros(20)) == (0.0, 0.0)
    assert an.wilcoxon(np.zeros(20), "greater")[1] == 1.0
    lo, hi = an.t_interval(np.array([1.0, 2.0, 3.0]))
    assert abs(lo - (2 - 4.302652729911275 / np.sqrt(3))) < 1e-9 and abs(hi - (2 + 4.302652729911275 / np.sqrt(3))) < 1e-9
    # permutation test: dominant diagonal -> only the identity reaches T (exact, n = 7)
    M = 0.5 + rng.normal(0, 0.02, (7, 7))
    np.fill_diagonal(M, 0.95)
    T, p, nperm, exact = an.perm_test(M)
    assert exact and nperm == math.factorial(7) and abs(p - 1 / 5040) < 1e-12 and abs(T - 0.95) < 1e-12
    # constant matrix: every permutation ties with T -> p = 1
    assert an.perm_test(np.full((5, 5), 0.6))[1] == 1.0
    # Monte Carlo path (n = 20): identity included, p small, reproducible
    M20 = 0.5 + rng.normal(0, 0.02, (20, 20))
    np.fill_diagonal(M20, 0.9)
    T20, p20, n20, ex20 = an.perm_test(M20)
    assert not ex20 and n20 == 100_001 and abs(p20 - 1 / 100_001) < 1e-12 and an.perm_test(M20)[1] == p20
    # null matrix: p is roughly uniform, not tiny
    Mn = 0.5 + rng.normal(0, 0.05, (8, 8))
    print("null 8x8 permutation p:", round(an.perm_test(Mn)[1], 3))

    hypothesis_table()

    # calibration gate function
    g1 = an.gate(M, top1=[1, 1, 0, 1, 1, 1, 1])
    assert g1["informative"] and g1["strongly_informative"] and abs(g1["p_perm"] - 1 / 5040) < 1e-12
    assert abs(g1["top1_share"] - 6 / 7) < 1e-12 and abs(g1["mean_A"] - 0.95) < 1e-12
    M2 = M.copy()
    np.fill_diagonal(M2, 0.75)
    g2 = an.gate(M2)
    assert g2["informative"] and not g2["strongly_informative"]
    g3 = an.gate(np.full((7, 7), 0.5) + np.diag(rng.normal(0, 0.001, 7)) * 0)
    assert not g3["informative"]
    print(g1, g2, g3, sep="\n")

    # gate command: --victims restricts the group before the control (victim 3 is an outlier partner)
    cr, sc = [], []
    for v in range(4):
        sc.append({"file": f"v{v}", "seed": 0, "victim": v, "top1": 1.0})
        for w in range(4):
            a = 0.95 if v == w else (0.1 if w == 3 else 0.5)
            cr.append({"group": 0, "dir": "x", "protocol": "fedavg", "condition": "oracle", "batch": 8, "u": 1,
                       "init": "mean", "tag": "", "res": 128, "v_seed": 0, "v_victim": v, "v_file": f"v{v}",
                       "w_seed": 0, "w_victim": w, "w_file": f"v{w}", "diagonal": v == w, "A": a})
    ps_, pc_, po_ = (os.path.join(TMP, n) for n in ("gate_scores.csv", "gate_scores_cross.csv", "gate_out.csv"))
    pd.DataFrame(sc).to_csv(ps_, index=False)
    pd.DataFrame(cr).to_csv(pc_, index=False)
    an.main(["gate", ps_, "--victims", "0", "1", "2", "--out", po_])
    go = pd.read_csv(po_).iloc[0]
    assert go.n == 3 and abs(go.mean_control - 0.5) < 1e-12 and abs(go.p_perm - 1 / 6) < 1e-12 and not go.informative
    an.main(["gate", ps_, "--out", po_])
    go = pd.read_csv(po_).iloc[0]
    assert go.n == 4 and abs(go.mean_control - (3 * (0.5 + 0.5 + 0.1) / 3 + 1.5 / 3) / 4) < 1e-12 and abs(go.p_perm - 1 / 24) < 1e-12

    label_controls()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
