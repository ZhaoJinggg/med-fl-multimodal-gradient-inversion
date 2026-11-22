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
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
