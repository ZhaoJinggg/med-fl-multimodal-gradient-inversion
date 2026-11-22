"""Synthetic tests with known answers for src/analyze.py (no data files needed).

usage: python src/test_analyze.py      (prints the tables and "ALL TESTS PASSED")
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an  # noqa: E402

rng = np.random.default_rng(0)
VICTIMS = [(s, v) for s in range(1, 6) for v in range(4)]  # 20 victims, 4 per seed


def rows(mod, protocol, u, values, condition="recovered", m0=np.nan):
    return [{"seed": s, "victim": v, "protocol": protocol, "u": u, "condition": condition, "modality": mod,
             "value": float(x), "m0": m0} for (s, v), x in zip(VICTIMS, values)]


def noise(sd):
    return rng.normal(0, sd, len(VICTIMS))


def synthetic():
    """Image: m0 = 0.5, FedAvg u=1 mean ~0.9 (f = 1 per 0.4 of A). Text: m0 = 0.2, FedAvg u=1 mean ~0.8."""
    R = []
    base_img = 0.9 + noise(0.02)
    R += rows("image", "fedavg", 1, base_img, m0=0.5)
    for k, u in enumerate([2, 4, 8, 16], 1):  # known decline of 0.05 A per level -> H1 image supported
        R += rows("image", "fedavg", u, base_img - 0.05 * k + noise(0.01))
    base_txt = 0.8 + noise(0.02)
    R += rows("text", "fedavg", 1, base_txt, m0=0.2)
    R += rows("text", "fedavg", 2, base_txt)                       # exactly equal: zero differences
    R += rows("text", "fedavg", 4, base_txt + noise(0.004))        # no decline
    R += rows("text", "fedavg", 8, 0.2 + noise(0.01))              # at the floor (f ~ 0)
    R += rows("text", "fedavg", 16, base_txt - 0.3 + noise(0.01))  # decline
    img4 = [r["value"] for r in R if r["modality"] == "image" and r["u"] == 4 and r["protocol"] == "fedavg"]
    img8 = [r["value"] for r in R if r["modality"] == "image" and r["u"] == 8 and r["protocol"] == "fedavg"]
    txt4 = [r["value"] for r in R if r["modality"] == "text" and r["u"] == 4 and r["protocol"] == "fedavg"]
    txt8 = [r["value"] for r in R if r["modality"] == "text" and r["u"] == 8 and r["protocol"] == "fedavg"]
    # H2a: oracle == recovered at u=1 (zero width), oracle ~ recovered at u=4 -> supported
    R += rows("image", "fedavg", 1, base_img, condition="oracle")
    R += rows("image", "fedavg", 4, np.array(img4) + noise(0.002), condition="oracle")
    # H2b: no text lower by 0.1 A (0.25 f) at both levels -> supported
    R += rows("image", "fedavg", 1, base_img - 0.1 + noise(0.01), condition="notext")
    R += rows("image", "fedavg", 4, np.array(img4) - 0.1 + noise(0.01), condition="no_text")
    # H3: SCAFFOLD equal within noise; text u=8 is a floor cell -> decided on u=4 only
    for u, iv, tv in ((4, img4, txt4), (8, img8, txt8)):
        R += rows("image", "scaffold", u, np.array(iv) + noise(0.002))
        R += rows("text", "scaffold", u, np.array(tv) + noise(0.002))
    # H4: FedProx image slightly higher (supported); text at u=4 lower by 0.12 ROUGE (f -0.2) -> not supported
        R += rows("image", "fedprox", u, np.array(iv) + 0.02 + noise(0.002))
        R += rows("text", "fedprox", u, np.array(tv) - (0.12 if u == 4 else 0) + noise(0.002))
    # H5: FedBN image lower by 0.1 A (significant), text equal -> supported
        R += rows("image", "fedbn", u, np.array(iv) - 0.1 + noise(0.01))
        R += rows("text", "fedbn", u, np.array(tv) + noise(0.002))
    # secondary metric: H1 reported only
    s1 = 0.4 + noise(0.02)
    R += rows("image_ssim", "fedavg", 1, s1)
    for u in (2, 4, 8, 16):
        R += rows("image_ssim", "fedavg", u, s1 - 0.01 * u + noise(0.01))
    return pd.DataFrame(R)


def run(df, gate="unknown", m0=None):
    path = os.path.join(tempfile.gettempdir(), "test_analyze_tidy.csv")
    df.to_csv(path, index=False)
    d = an.load_tidy([path])
    d, info = an.add_f(d, m0 or {})
    comp, dec, ps = an.evaluate(d, gate)
    return d, info, comp, dec, ps


def decision(dec, h, mod):
    return dec[(dec.hypothesis == h) & (dec.modality == mod)].decision.iloc[0]


def main():
    # unit checks
    assert np.allclose(an.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert an.t_interval(np.zeros(20)) == (0.0, 0.0)
    assert an.wilcoxon(np.zeros(20), "greater")[1] == 1.0
    lo, hi = an.t_interval(np.array([1.0, 2.0, 3.0]))
    assert abs(lo - (2 - 4.302652729911275 * 1 / np.sqrt(3))) < 1e-9 and abs(hi - (2 + 4.302652729911275 / np.sqrt(3))) < 1e-9

    df = synthetic()
    d, info, comp, dec, ps = run(df)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(info)
    print(comp[["hypothesis", "modality", "u", "n", "mean_diff", "ci_low", "ci_high", "lower_bound", "p", "p_holm",
                "significant", "within_margin", "fedavg_mean_f", "decidable", "mixed_agrees"]].round(4).to_string(index=False))
    print(dec.to_string(index=False))
    # f is 1 on average in the reference cell
    ref = d[(d.modality == "image") & (d.protocol == "fedavg") & (d.u == 1) & (d.condition == "recovered")]
    assert abs(ref.f.mean() - 1) < 1e-12 and abs(info["image"]["m0"] - 0.5) < 1e-12
    exp = {("H1", "image"): "supported", ("H1", "text"): "not supported", ("H1", "image_ssim"): "reported, decides nothing",
           ("H2a", "image"): "supported", ("H2b", "image"): "supported",
           ("H3", "image"): "supported", ("H3", "text"): "supported", ("H3", "overall"): "supported",
           ("H4", "image"): "supported", ("H4", "text"): "not supported", ("H4", "overall"): "not supported",
           ("H5", "image"): "supported", ("H5", "text"): "supported", ("H5", "overall"): "supported"}
    for (h, mod), e in exp.items():
        got = decision(dec, h, mod)
        assert got == e, f"{h} {mod}: expected {e!r}, got {got!r}"
    # the text floor cell (u=8) is reported as not decidable and leaves its families
    t8 = comp[(comp.modality == "text") & (comp.u == 8) & comp.hypothesis.isin(["H3", "H4", "H5"])]
    assert len(t8) == 3 and (~t8.decidable).all() and t8.at_floor.all()
    assert "not decidable" in dec[(dec.hypothesis == "H3") & (dec.modality == "text")].reason.iloc[0]
    # H1 has no floor rule: the text u=8 comparison stays in the H1 family
    h1t = comp[(comp.hypothesis == "H1") & (comp.modality == "text")]
    assert h1t.decidable.all() and h1t.p_holm.notna().all()
    # zero differences: H1 text u=2 has p = 1; H2a u=1 interval is [0, 0] and within the margin
    assert h1t[h1t.u == 2].p.iloc[0] == 1.0
    z = comp[(comp.hypothesis == "H2a") & (comp.u == 1)].iloc[0]
    assert z.ci_low == 0 and z.ci_high == 0 and z.within_margin
    # known shifts are significant
    assert comp[(comp.hypothesis == "H1") & (comp.modality == "image")].significant.all()
    assert comp[(comp.hypothesis == "H5") & (comp.modality == "image")].significant.all()

    # a cell entirely at the floor: H3 text undecided when every text FedAvg cell is at the floor
    df2 = df.copy()
    sel = (df2.modality == "text") & (df2.protocol == "fedavg") & (df2.u.isin([4]))
    df2.loc[sel, "value"] = 0.2 + rng.normal(0, 0.01, sel.sum())
    _, _, comp2, dec2, _ = run(df2)
    assert decision(dec2, "H3", "text") == "undecided", decision(dec2, "H3", "text")
    assert decision(dec2, "H3", "overall") == "supported"  # decided on the remaining (image) part

    # equal groups with large noise: interval too wide -> H3 image not supported
    df3 = df.copy()
    sel = (df3.modality == "image") & (df3.protocol == "scaffold")
    df3.loc[sel, "value"] += rng.normal(0, 0.1, sel.sum())
    _, _, _, dec3, _ = run(df3)
    assert decision(dec3, "H3", "image") == "not supported"

    # H2b third outcome: no text equal to recovered -> "text does not matter within the margin"
    df4 = df.copy()
    for u in (1, 4):
        rec = df4[(df4.modality == "image") & (df4.protocol == "fedavg") & (df4.u == u) & (df4.condition == "recovered")]
        sel = (df4.modality == "image") & (df4.u == u) & df4.condition.isin(["notext", "no_text"])
        df4.loc[sel, "value"] = rec.value.to_numpy() + rng.normal(0, 0.002, sel.sum())
    _, _, _, dec4, _ = run(df4)
    assert decision(dec4, "H2b", "image") == "text does not matter within the margin", decision(dec4, "H2b", "image")

    # gate not informative -> every image decision undecided
    _, _, _, dec5, _ = run(df, gate="not_informative")
    assert (dec5[dec5.modality.str.startswith("image")].decision == "undecided (gate)").all()

    # --m0 override
    _, info6, _, _, _ = run(df, m0={"text": 0.0})
    assert info6["text"]["m0"] == 0.0 and info6["text"]["m0_source"] == "cli"

    # ANOVA runs on the protocol experiment
    tab = an.anova(d)
    print(tab.round(4).to_string(index=False))
    assert set(tab.modality) == {"image", "text"} and "C(protocol)" in set(tab.source)
    assert tab[(tab.modality == "image") & (tab.source == "C(protocol)")].p.iloc[0] < 1e-6

    # calibration gate
    A_mm = 0.5 + rng.normal(0, 0.05, 7)
    g1 = an.gate(np.full(7, 0.95) + rng.normal(0, 0.01, 7), A_mm)
    g2 = an.gate(np.full(7, 0.75) + rng.normal(0, 0.01, 7), A_mm)
    g3 = an.gate(A_mm + np.array([0.01, -0.01, 0.02, -0.02, 0.005, -0.005, 0.0]), A_mm)
    print(g1, g2, g3, sep="\n")
    assert g1["informative"] and g1["identifies"]
    assert g2["informative"] and not g2["identifies"]
    assert not g3["informative"] and not g3["identifies"] and g3["n_nonzero"] == 6
    assert abs(g1["p"] - 1 / 128) < 1e-12  # exact one-sided p for 7 positive differences
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
