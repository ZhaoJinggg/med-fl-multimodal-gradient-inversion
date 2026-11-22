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


def rows(mod, protocol, u, values, condition="recovered", m0=np.nan):
    return [{"seed": s, "victim": v, "protocol": protocol, "u": u, "condition": condition, "modality": mod,
             "value": float(x), "m0": m0} for (s, v), x in zip(VICTIMS, values)]


def noise(sd, n=len(VICTIMS)):
    return rng.normal(0, sd, n)


def cross_rows(values, protocol="fedavg", u=1, condition="recovered", off=0.5, off_sd=0.05, batch=8, group=None):
    """Cross matrix with diagonal = the victims' A and off-diagonal = chance-level scores."""
    out = []
    for i, (vs, vv) in enumerate(VICTIMS):
        for j, (ws, wv) in enumerate(VICTIMS):
            a = values[i] if i == j else off + rng.normal(0, off_sd)
            out.append({"group": group if group is not None else f"{protocol}{u}{condition}", "protocol": protocol,
                        "condition": condition, "u": u, "batch": batch, "v_seed": vs, "v_victim": vv,
                        "w_seed": ws, "w_victim": wv, "diagonal": i == j, "A": a})
    return out


def synthetic():
    """Image: m0 = 0.5, FedAvg u=1 mean ~0.9 (f = 1 per 0.4 of A). Text: m0 = 0.2, FedAvg u=1 mean ~0.8."""
    R, C = [], []
    base_img = 0.9 + noise(0.02)
    R += rows("image", "fedavg", 1, base_img, m0=0.5)
    C += cross_rows(base_img, u=1)
    img = {1: base_img}
    for k, u in enumerate([2, 4, 8, 16], 1):  # known decline of 0.05 A per level -> H1 image supported
        img[u] = base_img - 0.05 * k + noise(0.01)
        R += rows("image", "fedavg", u, img[u])
        C += cross_rows(img[u], u=u)
    base_txt = 0.8 + noise(0.02)
    txt = {1: base_txt, 2: base_txt, 4: base_txt + noise(0.004), 8: 0.2 + noise(0.01), 16: base_txt - 0.3 + noise(0.01)}
    for u, v in txt.items():  # u=2 exactly equal (zero differences); u=8 at the floor (ROUGE-L ~ m0)
        R += rows("text", "fedavg", u, v, m0=0.2)
    # H2a: oracle == recovered at u=1 (zero width), oracle ~ recovered at u=4 -> supported
    R += rows("image", "fedavg", 1, base_img, condition="oracle")
    R += rows("image", "fedavg", 4, img[4] + noise(0.002), condition="oracle")
    # H2b: no text lower by 0.1 A (0.25 f) at both levels -> supported
    R += rows("image", "fedavg", 1, base_img - 0.1 + noise(0.01), condition="notext")
    R += rows("image", "fedavg", 4, img[4] - 0.1 + noise(0.01), condition="no_text")
    for u in (4, 8):
        # H3: SCAFFOLD equal within noise; text u=8 is a floor cell -> decided on u=4 only
        R += rows("image", "scaffold", u, img[u] + noise(0.002))
        R += rows("text", "scaffold", u, txt[u] + noise(0.002))
        # H4: FedProx image slightly higher (supported); text u=4 lower by 0.12 ROUGE (f -0.2) -> contradicted
        R += rows("image", "fedprox", u, img[u] + 0.02 + noise(0.002))
        R += rows("text", "fedprox", u, txt[u] - (0.12 if u == 4 else 0) + noise(0.002))
        # H5: FedBN image lower by 0.1 A (significant), text equal -> supported
        R += rows("image", "fedbn", u, img[u] - 0.1 + noise(0.01))
        R += rows("text", "fedbn", u, txt[u] + noise(0.002))
    s1 = 0.4 + noise(0.02)  # secondary metric: H1 reported only
    R += rows("image_ssim", "fedavg", 1, s1)
    for u in (2, 4, 8, 16):
        R += rows("image_ssim", "fedavg", u, s1 - 0.01 * u + noise(0.01))
    return pd.DataFrame(R), pd.DataFrame(C)


def run(df, cross, gate="unknown", m0=None):
    p, pc = os.path.join(TMP, "test_analyze_tidy.csv"), os.path.join(TMP, "test_analyze_cross.csv")
    df.to_csv(p, index=False)
    cross.to_csv(pc, index=False)
    d = an.load_tidy([p])
    c = an.load_cross([pc])
    d, info = an.add_f(d, m0 or {}, c)
    comp, dec, ps = an.evaluate(d, gate, c, m0 or {}, info)
    return d, info, comp, dec, ps


def decision(dec, h, mod):
    return dec[(dec.hypothesis == h) & (dec.modality == mod)].decision.iloc[0]


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


def cross_from_matrix(M, counts, protocol="fedavg", u=1, condition="recovered", M_lab=None):
    out = []
    for i, (vs, vv) in enumerate(VICTIMS):
        for j, (ws, wv) in enumerate(VICTIMS):
            out.append({"group": f"{protocol}{u}{condition}", "protocol": protocol, "condition": condition, "u": u,
                        "batch": 8, "v_seed": vs, "v_victim": vv, "v_n_abnormal": counts[i], "w_seed": ws, "w_victim": wv,
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


def label_controls():
    """Second review: stratified permutation test, sign-flip test, label regression, text precondition, tidy."""
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
    # 20 victims: Monte Carlo path
    r20 = an.stratified_perm_test(label_matrix(COUNTS20, diag=0.9), COUNTS20)
    assert not r20["exact"] and r20["n_perm"] == 552_960 and r20["n_eval"] == 100_001 and abs(r20["p"] - 1 / 100_001) < 1e-12
    assert r20["n_contrib"] == 19  # victim with count 1 is alone in its stratum
    # MC permutations stay within strata: a matrix that is +inf off-stratum would break the mean
    Minf = label_matrix(COUNTS20, same=0.5, other=np.inf)
    assert np.isfinite(an.stratified_perm_test(Minf, COUNTS20)["T"]) and an.stratified_perm_test(Minf, COUNTS20)["p"] == 1.0

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

    # `test`: pure label effect in the reference cell -> stratified test fails -> image "not tested";
    # --no-stratify (unstratified) passes it
    df, cross = synthetic()
    cross = cross[~((cross.protocol == "fedavg") & (cross.u == 1))]
    ref_vals = df[(df.modality == "image") & (df.protocol == "fedavg") & (df.u == 1) & (df.condition == "recovered")].value.to_numpy()
    MLr = label_matrix(COUNTS20, same=ref_vals.mean(), other=0.5, sd=0.0)
    np.fill_diagonal(MLr, ref_vals.mean())
    cross_l = pd.concat([cross, pd.DataFrame(cross_from_matrix(MLr, COUNTS20))])
    d_s, info_s, comp_s, dec_s, _ = run(df, cross_l)
    assert info_s["image"]["ref_passes"] is False and info_s["image"]["ref_test_p_unstratified"] <= 0.05
    assert info_s["image"]["ref_test_method"].startswith("stratified") and (dec_s[dec_s.modality == "image"].decision == "not tested").all()
    p, pc = os.path.join(TMP, "test_analyze_tidy.csv"), os.path.join(TMP, "test_analyze_cross.csv")
    d = an.load_tidy([p])
    c = an.load_cross([pc])
    d, info_u = an.add_f(d, {}, c, stratify=False)
    assert info_u["image"]["ref_passes"] and "unstratified: --no-stratify" in info_u["image"]["ref_test_method"]
    # a true image effect at u=1 with strata: passes the stratified test, floor columns filled for u=4 (no strata there)
    MDr = label_matrix(COUNTS20, same=0.55, other=0.5, sd=0.02)
    np.fill_diagonal(MDr, ref_vals)
    cross_d = pd.concat([cross, pd.DataFrame(cross_from_matrix(MDr, COUNTS20))])
    _, info_d, comp_d, dec_d, _ = run(df, cross_d)
    assert info_d["image"]["ref_passes"] and info_d["image"]["ref_test_p"] <= 1e-4 and info_d["image"]["ref_n_contrib"] == 19
    assert decision(dec_d, "H3", "image") == "supported"
    assert comp_d[(comp_d.modality == "image") & (comp_d.hypothesis == "H3")].floor_p_unstratified.notna().all()

    # text reference precondition: text FedAvg u=1 at its m0 -> text f not computed, H2-H5 text "not tested", H1 text tested
    df_t = df.copy()
    sel = (df_t.modality == "text") & (df_t.protocol == "fedavg") & (df_t.u == 1)
    df_t.loc[sel, "value"] = 0.2 + rng.normal(0, 0.01, sel.sum())
    d_t, info_t, comp_t, dec_t, _ = run(df_t, pd.concat([cross, pd.DataFrame(cross_from_matrix(MDr, COUNTS20))]))
    assert info_t["text"]["ref_passes"] is False and d_t[d_t.modality == "text"].f.isna().all()
    for h in ("H3", "H4", "H5"):
        assert decision(dec_t, h, "text") == "not tested", (h, decision(dec_t, h, "text"))
    assert decision(dec_t, "H1", "text") != "not tested" and decision(dec_t, "H3", "overall") == "supported"
    assert info_d["text"]["ref_passes"] is True

    # tidy-images: batch sizes other than 8 get B<k>
    ps = os.path.join(TMP, "tidy_scores.csv")
    pd.DataFrame({"seed": [1, 1, 1, 1], "victim": [0, 0, 0, 0], "protocol": "fedavg", "u": 1, "batch": [8, 4, 8, 2],
                  "condition": ["recovered", "recovered", "oracle", "oracle"], "A": 0.9, "A_mismatched": 0.5,
                  "file": ["a", "b", "c", "d"], "n_abnormal": [3, 1, 3, 1]}).to_csv(ps, index=False)
    t = an.tidy_from_image_scores(ps)
    assert list(t.condition) == ["recovered", "B4", "oracle", "oracle_B2"] and "n_abnormal" in t

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

    df, cross = synthetic()
    d, info, comp, dec, ps = run(df, cross)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(info)
    print(comp[["hypothesis", "modality", "u", "n", "mean_diff", "ci_low", "ci_high", "lower_bound", "p", "p_holm",
                "significant", "within_margin", "outside_margin", "floor_p", "decidable", "mixed_agrees"]].round(4).to_string(index=False))
    print(dec.to_string(index=False))
    ref = d[(d.modality == "image") & (d.protocol == "fedavg") & (d.u == 1) & (d.condition == "recovered")]
    assert abs(ref.f.mean() - 1) < 1e-12 and abs(info["image"]["m0"] - 0.5) < 1e-12 and info["image"]["ref_passes"]
    check(dec, {("H1", "image"): "supported", ("H1", "text"): "not supported", ("H1", "image_ssim"): "reported, decides nothing",
                ("H2a", "image"): "supported", ("H2b", "image"): "supported",
                ("H3", "image"): "supported", ("H3", "text"): "supported", ("H3", "overall"): "supported",
                ("H4", "image"): "supported", ("H4", "text"): "contradicted", ("H4", "overall"): "contradicted",
                ("H5", "image"): "supported", ("H5", "text"): "supported", ("H5", "overall"): "supported"})
    # text floor cell (u=8): Wilcoxon of ROUGE-L vs m0 fails -> not decidable, leaves the families
    t8 = comp[(comp.modality == "text") & (comp.u == 8) & comp.hypothesis.isin(["H3", "H4", "H5"])]
    assert len(t8) == 3 and (~t8.decidable.astype(bool)).all() and (t8.floor_p > 0.05).all()
    assert "not decidable" in dec[(dec.hypothesis == "H3") & (dec.modality == "text")].reason.iloc[0]
    # image floor tests use the permutation test and pass
    i4 = comp[(comp.modality == "image") & (comp.hypothesis == "H3")]
    assert i4.floor_method.str.startswith("permutation").all() and (i4.floor_p <= 0.05).all()
    # H1 has no floor rule
    h1t = comp[(comp.hypothesis == "H1") & (comp.modality == "text")]
    assert h1t.decidable.astype(bool).all() and h1t.p_holm.notna().all()
    assert h1t[h1t.u == 2].p.iloc[0] == 1.0
    z = comp[(comp.hypothesis == "H2a") & (comp.u == 1)].iloc[0]
    assert z.ci_low == 0 and z.ci_high == 0 and z.within_margin
    assert comp[(comp.hypothesis == "H1") & (comp.modality == "image")].significant.all()
    assert comp[(comp.hypothesis == "H5") & (comp.modality == "image")].significant.all()

    # image FedAvg u=8 at chance in its cross matrix (diagonal ~ off-diagonal) -> image u=8 comparisons not decidable
    c2 = cross[~((cross.protocol == "fedavg") & (cross.u == 8))]
    c2 = pd.concat([c2, pd.DataFrame(cross_rows(0.45 + noise(0.01), u=8))])
    _, _, comp2, dec2, _ = run(df, c2)
    e8 = comp2[(comp2.modality == "image") & (comp2.u == 8) & comp2.hypothesis.isin(["H3", "H4", "H5"])]
    assert len(e8) == 3 and (~e8.decidable.astype(bool)).all(), e8[["hypothesis", "floor_p"]]
    assert decision(dec2, "H3", "image") == "supported"  # decided on u=4 alone
    assert "not decidable" in dec2[(dec2.hypothesis == "H3") & (dec2.modality == "image")].reason.iloc[0]

    # reference cell at chance -> image f not computed, every image decision "not tested"
    c3 = cross[~((cross.protocol == "fedavg") & (cross.u == 1))]
    c3 = pd.concat([c3, pd.DataFrame(cross_rows(0.45 + noise(0.01), u=1))])
    d3, info3, comp3, dec3, _ = run(df, c3)
    assert info3["image"]["ref_passes"] is False and d3[d3.modality == "image"].f.isna().all()
    assert (dec3[dec3.modality == "image"].decision == "not tested").all()
    assert decision(dec3, "H3", "overall") == "supported"  # text part only
    assert decision(dec3, "H1", "image_ssim") == "not tested"  # literal: every image decision

    # every text FedAvg cell at the floor -> H3 text undecided; overall decided on the image part
    df4 = df.copy()
    sel = (df4.modality == "text") & (df4.protocol == "fedavg") & (df4.u == 4)
    df4.loc[sel, "value"] = 0.2 + rng.normal(0, 0.01, sel.sum())
    _, _, _, dec4, _ = run(df4, cross)
    assert decision(dec4, "H3", "text") == "undecided" and decision(dec4, "H3", "overall") == "supported"

    # equal groups, large noise -> inconclusive; a real shift beyond the margin -> contradicted
    df5 = df.copy()
    sel = (df5.modality == "image") & (df5.protocol == "scaffold")
    df5.loc[sel, "value"] += rng.normal(0, 0.1, sel.sum())
    _, _, _, dec5, _ = run(df5, cross)
    assert decision(dec5, "H3", "image") == "inconclusive", decision(dec5, "H3", "image")
    df6 = df.copy()
    df6.loc[sel, "value"] += 0.1
    _, _, _, dec6, _ = run(df6, cross)
    assert decision(dec6, "H3", "image") == "contradicted" and decision(dec6, "H3", "overall") == "contradicted"
    # H4 inconclusive: lower bound below -0.05 but upper end above -0.05
    df7 = df.copy()
    sel7 = (df7.modality == "image") & (df7.protocol == "fedprox")
    zz = np.linspace(-1, 1, len(VICTIMS))
    zz = (zz - zz.mean()) / zz.std(ddof=1)  # fixed standardized offsets: f difference mean -0.05, sd 0.1
    df7.loc[sel7, "value"] += -0.04 + 0.04 * np.tile(zz, sel7.sum() // len(VICTIMS))
    _, _, comp7, dec7, _ = run(df7, cross)
    assert decision(dec7, "H4", "image") == "inconclusive", comp7[comp7.hypothesis == "H4"][["lower_bound", "ci_high"]]

    # H2b third outcome
    df8 = df.copy()
    for u in (1, 4):
        rec = df8[(df8.modality == "image") & (df8.protocol == "fedavg") & (df8.u == u) & (df8.condition == "recovered")]
        s8 = (df8.modality == "image") & (df8.u == u) & df8.condition.isin(["notext", "no_text"])
        df8.loc[s8, "value"] = rec.value.to_numpy() + rng.normal(0, 0.002, s8.sum())
    _, _, _, dec8, _ = run(df8, cross)
    assert decision(dec8, "H2b", "image") == "text does not matter within the margin"

    # calibration gate not informative -> every image decision undecided (gate)
    _, _, _, dec9, _ = run(df, cross, gate="not_informative")
    assert (dec9[dec9.modality.str.startswith("image")].decision == "undecided (gate)").all()
    # --m0 override
    _, info10, _, _, _ = run(df, cross, m0={"text": 0.0})
    assert info10["text"]["m0"] == 0.0 and info10["text"]["m0_source"] == "cli"
    # ANOVA
    tab = an.anova(d)
    print(tab.round(4).to_string(index=False))
    assert set(tab.modality) == {"image", "text"} and tab[(tab.modality == "image") & (tab.source == "C(protocol)")].p.iloc[0] < 1e-6

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
