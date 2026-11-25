"""CPU tests for the protocol wiring of attack_text.py and for score_text.py (tiny settings, real model and data).

T1  fedavg: attack_text.victim_update (through protocols) == the former victim_update (copied verbatim from
    commit 31f2344 below), bit for bit over every parameter, with and without --exact, u = 1 and u = 3; same
    logits; same CPU random state afterwards (DAGER's svd_lowrank draws from it); the scale factors (dager's
    1/(lr_txt u), the label bound lr_img u / B) are bit for bit the former ones
T2  scaffold, random nonzero control variates: server_delta at u = 1 vs the fedavg update, element-wise in
    units of the float32 spacing of the weights; singular values / span rank of the two DAGER layers for the
    fedavg update, the raw scaffold upload and server_delta (rank rule of dager(), noise factors 1.5 and 5),
    and principal angles of the token spans against the exact fedavg span; same at u = 3
T3  auxiliary metrics on hand-made texts with known answers (exact recovery, one wrong token, an unmatched
    original, a shifted text, more recovered than originals)
T4  score_text: m0 == src/calibration/text_baseline.py (executed) for seed 0; the existing runs/test/text result; synthetic
    result JSON for seeds 1-2 over all cells + exact + batch 4 -> tidy CSV -> analyze.py test; the exact and
    batch-4 rows change nothing in the comparisons; duplicate files are rejected
T5  end-to-end on the CPU (--device cpu): attack_text.main for fedavg (exact and float32), and, on tiny
    train_fl warm-ups (2 clients x 4 pairs, 2 rounds), scaffold (Options II and I), fedprox, fedbn; then
    score_text on the outputs and analyze.load_tidy

usage (from mmgia/, no GPU): CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe tests/test_attack_text.py --out <scratch dir>
       [--tests T1 T2 ...]
"""
import argparse
import copy
import glob
import json
import os
import random
import sys
import time

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # CPU only (an empty value is dropped on Windows)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rouge_score import rouge_scorer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # noqa: E402
import analyze  # noqa: E402
import attack_text as AT  # noqa: E402
import protocols as P  # noqa: E402
import score_text  # noqa: E402
import train_fl  # noqa: E402
from common import ROOT, FusionModel, PairData, bce, is_image_param, load_seed, tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="scratch directory (absolute)")
ap.add_argument("--tests", nargs="*", default=None)
ap.add_argument("--B", type=int, default=2)
ap.add_argument("--res", type=int, default=224, help="224 reads the existing image cache (no new files)")
a = ap.parse_args()
dev = "cpu"
assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=-1 (CPU only)"
LR_IMG, LR_TXT = 0.01, 1e-2
RUN = lambda t: a.tests is None or t in a.tests
t_start = time.time()
os.makedirs(a.out, exist_ok=True)
FAILS = []


def report(name, ok, msg):
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}  ({time.time() - t_start:.0f} s)", flush=True)


def same(d1, d2):
    return set(d1) == set(d2) and all(torch.equal(d1[k], d2[k]) for k in d1)


need_model = any(RUN(t) for t in ("T1", "T2"))
if need_model:
    data = PairData(a.res)
    split = load_seed(0)
    rows = data.rows(split["victim_batches"]["0"][: a.B])
    x, ids, mask, y = data.batch(rows, dev)
    torch.manual_seed(0)
    model = FusionModel().to(dev)
    model.train()
    fedavg = P.Protocol("fedavg")


def lr_of(name):
    return LR_IMG if is_image_param(name) else LR_TXT


# ------------------------------------------------ the former victim_update (attack_text.py at 31f2344), verbatim
# except a.u -> u, a.exact -> exact
def victim_update_old(x, ids, mask, y, u, exact):
    m = FusionModel().to(dev)
    m.load_state_dict(model.state_dict())
    m.train()
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    acc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in m.named_parameters()}
    logit0 = None
    for t in range(u):
        logit = m(x, ids, mask)
        if t == 0:
            logit0 = logit.detach()
        loss = bce(logit, y)
        names = [n for n, _ in m.named_parameters()]
        grads = torch.autograd.grad(loss, list(m.parameters()))
        with torch.no_grad():
            for n, p, g in zip(names, m.parameters(), grads):
                p -= lr_of(n) * g
                acc[n] -= lr_of(n) * g.double()
    if exact:
        return {n: v.float() for n, v in acc.items()}, logit0
    return {n: (p.detach() - w0[n]) for n, p in m.named_parameters()}, logit0


# ------------------------------------------------ T1
if RUN("T1"):
    for u in (1, 3):
        for exact in (False, True):
            torch.manual_seed(1000)
            d_old, l_old = victim_update_old(x, ids, mask, y, u, exact)
            rng_old = torch.get_rng_state()
            torch.manual_seed(1000)
            d_new, l_new, view = AT.victim_update(model, x, ids, mask, y, u, LR_IMG, LR_TXT, fedavg, exact=exact, names=None)
            rng_new = torch.get_rng_state()
            torch.manual_seed(1000)
            d_need, _, _ = AT.victim_update(model, x, ids, mask, y, u, LR_IMG, LR_TXT, fedavg, exact=exact)
            ok = (same(d_old, d_new) and torch.equal(l_old, l_new) and torch.equal(rng_old, rng_new)
                  and same({n: d_old[n] for n in AT.NEEDED}, d_need) and view == {"attack_input": "delta", "u_view": u})
            report(f"T1 fedavg u={u} exact={exact}: new victim_update == former", ok,
                   f"{len(d_new)} tensors bit for bit: {same(d_old, d_new)}; logit0 {torch.equal(l_old, l_new)}; "
                   f"CPU RNG state after the update {torch.equal(rng_old, rng_new)}; attack subset {sorted(d_need)}")
    ok = True
    for u in (1, 2, 4, 8, 16):
        st, si = P.effective_steps(fedavg, u, LR_TXT), P.effective_steps(fedavg, u, LR_IMG)
        for B in (1, 2, 4, 8):
            ok &= (-1.0 / (LR_TXT * st) == -1.0 / (LR_TXT * u)) and (LR_IMG * si / B == LR_IMG * u / B)
    report("T1 fedavg scale factors == former", ok, "dager scale and label bound bit for bit for u in 1..16, B in 1..8")

# ------------------------------------------------ T2
def spacing(t):
    t = t.abs().float()
    return torch.nextafter(t, torch.full_like(t, float("inf"))) - t


def rank_report(dw, k_true, ref_V, nfs=(1.5, 5.0)):
    g = [dw[n].T.double() for n in AT.DAGER_LAYERS]
    ranks = {nf: AT.span_rank([t.float() for t in g], nf) for nf in nfs}
    out = []
    for li, t in enumerate(g):
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        floor = S[-200:].median()
        cos = torch.linalg.svdvals(Vh[:k_true] @ ref_V[li].T).clamp(max=1.0)
        out.append({"s_k/floor": (S[k_true - 1] / floor).item(), "s_k+1/floor": (S[k_true] / floor).item(),
                    "max_sin": torch.sqrt(1 - cos.min() ** 2).item()})
    return ranks, out


def rand_controls(scale_from):
    """Nonzero control variates of the size of a gradient (c and c_k differ), as in test_protocols.py."""
    torch.manual_seed(1)
    c = {n: 0.5 * g + 0.1 * g.abs().mean() * torch.randn_like(g) for n, g in scale_from.items()}
    c_k = {n: -0.3 * g + 0.1 * g.abs().mean() * torch.randn_like(g) for n, g in scale_from.items()}
    return c, c_k


if RUN("T2"):
    m0 = copy.deepcopy(model).train()
    names_all = [n for n, p in m0.named_parameters() if p.requires_grad]
    g0 = dict(zip(names_all, torch.autograd.grad(bce(m0(x, ids, mask), y), [p for p in m0.parameters() if p.requires_grad])))
    del m0
    c, c_k = rand_controls(g0)
    n_tok = int(mask.sum())
    w0 = {n: p.detach() for n, p in model.named_parameters()}
    for u in (1, 3):
        sc = P.Protocol("scaffold", c=c, c_k=c_k)
        d_f, _, _ = AT.victim_update(model, x, ids, mask, y, u, LR_IMG, LR_TXT, fedavg)
        d_fx, _, _ = AT.victim_update(model, x, ids, mask, y, u, LR_IMG, LR_TXT, fedavg, exact=True)
        d_s, _, view = AT.victim_update(model, x, ids, mask, y, u, LR_IMG, LR_TXT, sc)
        d_sx, _, _ = AT.victim_update(model, x, ids, mask, y, u, LR_IMG, LR_TXT, sc, exact=True)
        up_raw, _ = P.client_upload(model, x, ids, mask, y, u, LR_IMG, LR_TXT, sc)
        raw = {n: up_raw["delta"][n] for n in AT.NEEDED}
        del up_raw
        if u == 1:
            msg, ok = [], view["attack_input"] == "delta_minus_scaffold_correction"
            for n in AT.NEEDED:
                # float32 spacing of the largest weight either client passes through (w^t, w^t + delta)
                sp = torch.maximum(torch.maximum(spacing(w0[n]), spacing(w0[n] + d_f[n])), spacing(w0[n] + raw[n]))
                ex = d_fx[n].double()                                          # the exact u = 1 update -eta g
                r_sf = ((d_s[n] - d_f[n]).abs() / sp).max().item()             # scaffold after removal vs fedavg
                r_s = ((d_s[n].double() - ex).abs() / sp).max().item()         # scaffold after removal vs exact
                r_f = ((d_f[n].double() - ex).abs() / sp).max().item()         # fedavg's own rounding vs exact
                rel_s = ((d_s[n].double() - ex).norm() / ex.norm()).item()
                rel_f = ((d_f[n].double() - ex).norm() / ex.norm()).item()
                corr = ((raw[n] - d_f[n]).norm() / d_f[n].norm()).item()
                # each float32 update is within ~2 spacings of the exact one (rounding of w^t - eta g, of the
                # product eta g and of the difference); scaffold may not be worse than that, nor than fedavg
                ok &= r_s <= max(2.0, r_f) and r_sf <= 4.0 and rel_s <= 2 * rel_f
                msg.append(f"{n.replace('txt.', '').replace('.attn.c_attn.weight', '')}: max error vs exact {r_s:.2f} "
                           f"spacings (fedavg {r_f:.2f}), rel norm error {rel_s:.1e} (fedavg {rel_f:.1e}), max |d_s - d_f| "
                           f"{r_sf:.2f} spacings; |correction| / |gradient part| {corr:.2f}")
            report("T2a scaffold u=1: server_delta == fedavg update within float32 rounding", ok, "; ".join(msg))
        # token span of the exact fedavg update = reference
        ref_V = [torch.linalg.svd(d_fx[n].T.double(), full_matrices=False)[2][:n_tok] for n in AT.DAGER_LAYERS]
        rows_ = {}
        for name, d in (("fedavg f32", d_f), ("fedavg exact", d_fx), ("scaffold raw upload f32", raw),
                        ("scaffold server_delta f32", d_s), ("scaffold server_delta exact", d_sx)):
            rows_[name] = rank_report(d, n_tok, ref_V)
        lines = [f"{k}: rank(nf 1.5) {v[0][1.5]}, rank(nf 5) {v[0][5.0]}, layer0/1 s_k/floor "
                 f"{v[1][0]['s_k/floor']:.1f}/{v[1][1]['s_k/floor']:.1f}, s_k+1/floor {v[1][0]['s_k+1/floor']:.2f}/"
                 f"{v[1][1]['s_k+1/floor']:.2f}, max sin angle to exact fedavg span {v[1][0]['max_sin']:.2e}/{v[1][1]['max_sin']:.2e}"
                 for k, v in rows_.items()]
        fa, sd, rw = rows_["fedavg f32"], rows_["scaffold server_delta f32"], rows_["scaffold raw upload f32"]
        fx, sx = rows_["fedavg exact"], rows_["scaffold server_delta exact"]
        # the raw upload (correction not removed) loses the token span: no gap after the k-th singular value
        # and a token span far from the true one; server_delta restores it
        raw_lost = all(rw[1][i]["max_sin"] > 0.9 and rw[1][i]["s_k/floor"] < 1.2 * rw[1][i]["s_k+1/floor"] for i in (0, 1))
        if u == 1:   # same gradient: same rank, gap and span as fedavg (up to rounding)
            ok = (raw_lost and abs(sd[0][1.5] - fa[0][1.5]) <= 2 and abs(sd[0][5.0] - fa[0][5.0]) <= 2
                  and abs(sx[0][5.0] - fx[0][5.0]) <= 2
                  and all(sd[1][i]["max_sin"] <= 2 * fa[1][i]["max_sin"] + 0.02 for i in (0, 1)))
        else:        # different trajectories at u > 1: rank numbers informational
            ok = raw_lost
        report(f"T2b u={u} rank behaviour of the DAGER layers ({n_tok} tokens in the batch)", ok, "\n    " + "\n    ".join(lines))

# ------------------------------------------------ T3
if RUN("T3"):
    tok = tokenizer()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    def run(originals, recovered):
        o_ids = [tok.encode(t) for t in originals]
        r_ids = [tok.encode(t) for t in recovered]
        rl, r1, pairs, ri, ci = AT.match_texts(originals, recovered, scorer)
        means, per = AT.aux_text_metrics(originals, recovered, o_ids, r_ids, ri, ci, scorer)
        return float(np.mean(rl)), float(r1), means, per, rl

    o1, o2 = "The heart is normal", "No pleural effusion or pneumothorax."
    assert len(tok.encode(o1)) == 4 and len(tok.encode("The heart is large")) == 4
    close = lambda u, v: abs(u - v) < 1e-12
    rl, r1, m, per, _ = run([o1, o2], [o2, o1])
    report("T3a exact recovery (shuffled) -> 1.0", close(rl, 1) and close(r1, 1) and all(close(v, 1) for v in m.values()),
           f"ROUGE-L {rl}, ROUGE-1 {r1}, {m}")
    rl, r1, m, per, _ = run([o1], ["The heart is large"])
    ok = close(rl, 0.75) and close(r1, 0.75) and close(m["rouge1"], 0.75) and close(m["rouge2"], 2 / 3) and close(m["token_acc"], 0.75)
    report("T3b one wrong token of 4", ok, f"ROUGE-L {rl} (0.75), ROUGE-1 {m['rouge1']} (0.75), ROUGE-2 {m['rouge2']:.4f} (2/3), "
           f"token acc {m['token_acc']} (0.75)")
    rl, r1, m, per, rls = run([o1, o2], [o1])
    ok = close(rl, 0.5) and close(r1, 0.5) and close(m["rouge2"], 0.5) and close(m["token_acc"], 0.5) and per["token_acc"] == [1.0, 0.0] and rls == [1.0, 0.0]
    report("T3c unmatched original scores 0, mean over all B", ok, f"ROUGE-L per original {rls}, mean {rl}; ROUGE-1 {r1}; {m}")
    rl, r1, m, per, _ = run([o1], ["Normal. The heart is normal"])
    ok = m["token_acc"] == 0.0 and rl > 0.8
    report("T3d shifted text: token accuracy by position 0, ROUGE high", ok, f"ROUGE-L {rl:.4f}, ROUGE-1 {m['rouge1']:.4f}, token acc {m['token_acc']}")
    rl, r1, m, per, _ = run([o1], ["The heart", o1])
    rl2, _, m2, _, _ = run([o1], ["The heart is normal and the lungs are clear"])
    ok = close(rl, 1) and close(m["token_acc"], 1) and close(m2["token_acc"], 1.0) and rl2 < 1
    report("T3e more recovered than originals; longer recovered text", ok,
           f"best of two picked: ROUGE-L {rl}, token acc {m['token_acc']}; longer text: token acc {m2['token_acc']} "
           f"(extra tokens not counted), ROUGE-L {rl2:.4f}")

# ------------------------------------------------ T4
SYN_SEEDS = (1, 2)


def synthetic_result(seed, victim, proto, u, exact=False, batch=8, rng=None):
    base = {"fedavg": 0.6, "scaffold": 0.6, "fedprox": 0.62, "fedbn": 0.6}[proto]
    r = float(np.clip(base - 0.05 * np.log2(u) + (0.3 if exact else 0) + rng.normal(0, 0.05), 0, 1))
    return {"victim": victim, "batch": batch, "u": u, "rougeL": round(r, 4), "rouge1": round(min(1, r + 0.05), 4),
            "n_recovered": batch, "label_acc": 0.875, "minutes": 1.0, "rank": 480, "B_used": 480, "positions": 64,
            "seed": seed, "proto": proto, "exact": exact, "mu": 1.0 if proto == "fedprox" else 0.0,
            "scaffold_option": "II" if proto == "scaffold" else None, "bn_stats": "local" if proto == "fedbn" else "upload",
            "aux_round": 9 if proto in ("scaffold", "fedbn") else None, "noise_factor": 1.5, "l1_thresh": 5e-2,
            "l2_thresh": 1.5e-2, "rank_cutoff": 20, "max_beams": 500, "lr_img": 0.01, "lr_txt": 0.01, "res": 128,
            "global": f"runs/fl/seed{seed}_{proto}/global_round9.pt", "eta_mu_u_img": 0.01 * u if proto == "fedprox" else 0.0,
            "eta_mu_u_txt": 0.01 * u if proto == "fedprox" else 0.0, "attack_input": "delta", "u_view": u,
            "rouge2": round(max(0, r - 0.1), 4), "token_acc": round(max(0, r - 0.2), 4),
            "label_scores": {"sign": {"acc_all_fallback1": 0.875, "acc_matched": 1.0, "n_matched": 5},
                             "combined": {"acc_all_fallback1": 0.75, "acc_matched": 0.8, "n_matched": 5},
                             "model": {"acc_all_fallback1": 0.75, "acc_matched": 0.6, "n_matched": 5},
                             "always_abnormal": {"acc_all_fallback1": 0.625, "acc_matched": 0.6, "n_matched": 5}}}


if RUN("T4"):
    # m0 against text_baseline.py itself (executed with the cached 224 px images instead of building a 128 px cache)
    bdir = os.path.join(a.out, "baseline")
    os.makedirs(bdir, exist_ok=True)
    bpath = os.path.join(ROOT, "src", "calibration", "text_baseline.py")
    src = open(bpath, encoding="utf-8").read()
    assert src.count("data = PairData(128)") == 1
    src = src.replace("data = PairData(128)", f"data = PairData({a.res})")
    argv = sys.argv
    sys.argv = ["text_baseline.py", "--seed", "0", "--victims", "0", "1", "2", "--batch", "8", "--out", bdir]
    exec(compile(src, "text_baseline.py", "exec"), {"__name__": "__main__", "__file__": bpath})
    sys.argv = argv
    ref = json.load(open(os.path.join(bdir, "baseline_seed0_B8.json")))
    td = score_text.TextData()
    got = [td(0, r["victim"], 8) for r in ref]
    ok = all(round(g["m0_single"], 4) == r["rougeL_frequent"] and round(g["label_acc_majority"], 3) == r["label_acc_majority"]
             and g["m0_single_count"] == r["frequent_count"] for g, r in zip(got, ref))
    report("T4a score_text m0_single == text_baseline.py (seed 0, victims 0-2, B = 8)", ok,
           f"m0_single {[round(g['m0_single'], 4) for g in got]} vs {[r['rougeL_frequent'] for r in ref]}; majority-label acc "
           f"{[round(g['label_acc_majority'], 3) for g in got]} vs {[r['label_acc_majority'] for r in ref]}")
    # the existing local result (old format: no seed, no cell fields)
    olds = sorted(glob.glob(os.path.join(ROOT, "runs", "**", "text", "*.json"), recursive=True))
    if olds:
        df_old = score_text.main(olds + ["--out", os.path.join(a.out, "tidy_old.csv"), "--default-seed", "0"])
        report("T4b score_text on the existing runs/*/text results", len(df_old) == len(olds) and df_old.m0.notna().all(),
               f"{len(df_old)} row(s): " + df_old[["seed", "victim", "protocol", "u", "condition", "value", "m0", "rouge1",
                                                     "rouge2", "token_acc", "label_acc_returned"]].to_string(index=False).replace("\n", " | "))
    # synthetic results of the new format over all cells
    sdir = os.path.join(a.out, "synthetic")
    rng = np.random.default_rng(0)
    cells = [("fedavg", u) for u in (1, 2, 4, 8, 16)] + [(p, u) for p in ("scaffold", "fedprox", "fedbn") for u in (4, 8)]
    files = []
    for s in SYN_SEEDS:
        for v in sorted(int(k) for k in load_seed(s)["victim_batches"]):
            for proto, u in cells:
                files.append((f"seed{s}/{proto}/v{v}_B8_u{u}.json", synthetic_result(s, v, proto, u, rng=rng)))
            for u in (1, 4):
                files.append((f"seed{s}/fedavg_exact/v{v}_B8_u{u}_exact.json", synthetic_result(s, v, "fedavg", u, exact=True, rng=rng)))
            files.append((f"seed{s}/fedavg_b4/v{v}_B4_u1.json", synthetic_result(s, v, "fedavg", 1, batch=4, rng=rng)))
    for f, rec in files:
        os.makedirs(os.path.dirname(os.path.join(sdir, f)), exist_ok=True)
        json.dump(rec, open(os.path.join(sdir, f), "w"))
    json.dump([{"victim": 0}], open(os.path.join(sdir, "baseline_seed1_B8.json"), "w"))    # must be skipped
    tidy_all = os.path.join(a.out, "tidy_synthetic.csv")
    df = score_text.main([sdir, "--out", tidy_all])
    n_v = sum(len(load_seed(s)["victim_batches"]) for s in SYN_SEEDS)
    conds = df.condition.value_counts().to_dict()
    ok = len(df) == len(files) and conds == {"recovered": 11 * n_v, "exact": 2 * n_v, "B4": n_v} and df.m0.notna().all()
    report("T4c score_text on synthetic results (2 seeds, all 11 cells + exact + B = 4)", ok,
           f"{len(df)} rows, conditions {conds}, m0 range {df.m0.min():.3f}-{df.m0.max():.3f}")
    # analyze: with and without the exact / batch-4 rows -> identical comparisons
    tidy_main = os.path.join(a.out, "tidy_synthetic_main.csv")
    df[df.condition == "recovered"].to_csv(tidy_main, index=False)
    analyze.main(["test", tidy_all, "--out", os.path.join(a.out, "an_all")])
    analyze.main(["test", tidy_main, "--out", os.path.join(a.out, "an_main")])
    c_all = pd.read_csv(os.path.join(a.out, "an_all", "comparisons.csv"))
    c_main = pd.read_csv(os.path.join(a.out, "an_main", "comparisons.csv"))
    hyps = sorted(set(c_all.hypothesis))
    ok = c_all.equals(c_main) and set(c_all.n) == {n_v} and hyps == ["H1", "H3", "H4", "H5b"]
    report("T4d analyze.py test accepts the tidy CSV; exact / B = 4 rows never enter a comparison", ok,
           f"{len(c_all)} comparisons ({hyps}), n per comparison {sorted(set(c_all.n))}, identical with and without "
           f"the extra rows: {c_all.equals(c_main)}")
    # duplicates are rejected
    json.dump(files[0][1], open(os.path.join(sdir, "seed1", "dup.json"), "w"))
    try:
        score_text.main([sdir, "--out", os.path.join(a.out, "dup.csv")])
        ok = False
    except SystemExit as e:
        ok = "same victim and cell" in str(e)
    os.remove(os.path.join(sdir, "seed1", "dup.json"))
    report("T4e duplicate result files rejected", ok, "SystemExit with the list of files")

    # revised m0: the B most frequent public reports, matched; independent computation with attack_text.match_texts
    sc3 = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    pub = [i for i, p in enumerate(td.pairs) if p["split"] in ("val", "test")]
    cnt = {}
    for i in pub:
        cnt[td.text(i)] = cnt.get(td.text(i), 0) + 1
    order = sorted(cnt, key=lambda t: -cnt[t])                  # stable: first occurrence among equal counts
    vals, ok = [], True
    for s_, v_ in ((0, 0), (0, 1), (0, 2), (1, 0), (2, 5)):
        orig = td.originals(s_, v_, 8)
        rl, _, _, _, _ = AT.match_texts(orig, order[:8], sc3)
        got_ = td(s_, v_, 8)["m0"]
        ok &= abs(float(np.mean(rl)) - got_) < 1e-12
        vals.append((s_, v_, round(got_, 4), round(td(s_, v_, 8)["m0_single"], 4)))
    # match_rougeL == attack_text.match_texts on fewer, equal and more texts than originals
    rs = random.Random(0)
    for n_h in (3, 8, 11):
        orig = td.originals(1, 0, 8)
        hyps = [td.text(rs.randrange(len(td.pairs))) for _ in range(n_h)]
        ok &= score_text.match_rougeL(orig, hyps, td.scorer) == AT.match_texts(orig, hyps, sc3)[0]
    report("T4f m0 = B most frequent public reports, matched (== independent computation); match_rougeL == match_texts", ok,
           f"{len(pub)} public pairs, top counts {[cnt[t] for t in order[:8]]}; (seed, victim, m0, m0_single): {vals}")

    # mismatched control, recomputed ROUGE-L, label metrics: seed 0 victims 0-2 'recovered' exactly (rec_ids = their own
    # token ids), one copy of victim 2 in another directory (same cell suffix)
    mdir = os.path.join(a.out, "mismatch")
    recs_m = {}
    for v_ in (0, 1, 2):
        rows_v = td.victim_rows(0, v_, 8)
        lab_v = td(0, v_, 8)["labels"]
        rid = [td.ids[i][: int(td.mask[i].sum())].tolist() for i in rows_v][::-1]      # reversed order
        sign_v = list(lab_v[::-1]); sign_v[0] = 1 - sign_v[0]                          # one wrong label
        recs_m[v_] = {"victim": v_, "batch": 8, "u": 1, "rougeL": 1.0, "rouge1": 1.0, "seed": 0, "proto": "fedavg",
                      "exact": False, "n_recovered": 8, "label_acc": 0.875, "rec_ids": rid, "labels_true": lab_v,
                      "labels_sign": sign_v, "matching": [{"orig": 7 - j, "rec": j, "rougeL": 1.0} for j in range(8)],
                      "label_scores": {"sign": {"acc_all_fallback1": 0.875, "acc_matched": 0.875, "n_matched": 8},
                                       "always_abnormal": {"acc_all_fallback1": 0.5, "acc_matched": 0.5, "n_matched": 8}}}
    os.makedirs(os.path.join(mdir, "a"), exist_ok=True); os.makedirs(os.path.join(mdir, "b"), exist_ok=True)
    for v_ in (0, 1, 2):
        json.dump(recs_m[v_], open(os.path.join(mdir, "a", f"v{v_}_B8_u1_t.json"), "w"))
    json.dump({**recs_m[2], "seed": 3, "victim": 1, "labels_true": td(3, 1, 8)["labels"]},
              open(os.path.join(mdir, "b", "v1_B8_u1_t.json"), "w"))
    dm = score_text.main([mdir, "--out", os.path.join(a.out, "tidy_mismatch.csv")])
    da = dm[dm.file.str.contains("/a/")].set_index("victim")
    exp0 = np.mean([np.mean(score_text.match_rougeL(td.originals(0, 0, 8), [td.tok.decode(t) for t in recs_m[w]["rec_ids"]], td.scorer))
                    for w in (1, 2)])
    k0 = td(0, 0, 8)["n_abnormal"]
    ok = (np.allclose(da.rougeL_recomputed, 1.0) and abs(da.loc[0, "rougeL_mismatched"] - exp0) < 1e-12
          and set(da.n_mismatched) == {2} and np.allclose(da.label_acc_returned, 0.875)
          and abs(da.loc[0, "label_acc_random_counts"] - (k0 ** 2 + (8 - k0) ** 2) / 64) < 1e-12
          and dm[dm.file.str.contains("/b/")].n_mismatched.iloc[0] == 0)
    dm2 = score_text.main([mdir, "--out", os.path.join(a.out, "tidy_mismatch2.csv"), "--mismatch-scope", "cell"])
    ok &= set(dm2.n_mismatched) == {3}
    report("T4g mismatched control, recomputed ROUGE-L, label metrics (a)-(c)", ok,
           "\n    " + da[["value", "rougeL_recomputed", "m0", "m0_single", "rougeL_mismatched", "n_mismatched",
                              "label_acc_returned", "label_acc_matched", "label_acc_random_counts", "labels_mixed",
                              "n_abnormal", "label_acc_oracle_fallback"]].round(4).to_string().replace("\n", "\n    ")
           + f"\n    expected mismatched for victim 0: {exp0:.4f}; scope cell: n_mismatched {sorted(set(dm2.n_mismatched))}")

# ------------------------------------------------ T5
if RUN("T5"):
    edir = os.path.join(a.out, "e2e")
    common_args = ["--seed", "0", "--victims", "0", "--batch", str(a.B), "--res", str(a.res), "--lr-img", str(LR_IMG),
                   "--lr-txt", str(LR_TXT), "--device", "cpu", "--out", edir, "--parallel", "100"]
    cal = ["--noise-factor", "1.5", "--l1-thresh", "5e-2", "--l2-thresh", "1.5e-2"]
    g_init = os.path.join(ROOT, "runs", "test", "global_init.pt")

    def attack(extra, label):
        t0 = time.time()
        AT.main(common_args + extra)
        print(f"  ({label}: {time.time() - t0:.0f} s)", flush=True)

    attack(["--global", g_init, "--u", "1", "--exact"] + cal, "fedavg u=1 exact")
    attack(["--global", g_init, "--u", "1"] + cal, "fedavg u=1 float32")
    fl = os.path.join(a.out, "fl")
    shared = PairData(a.res)

    def warmup(proto, extra=()):
        args = train_fl.get_args(["--proto", proto, "--seed", "0", "--rounds", "2", "--batch", "2", "--lr-img", str(LR_IMG),
                                  "--lr-txt", str(LR_TXT), "--device", "cpu", "--out", fl, "--clients", "0", "1",
                                  "--pairs-per-client", "4", "--eval-pairs", "4", "--aux-clients", "0", *extra])
        _, out = train_fl.run(args, data=shared)
        return os.path.join(out, "global_round2.pt")

    g_sc, g_sc1 = warmup("scaffold"), warmup("scaffold", ["--scaffold-option", "I"])
    g_px, g_bn = warmup("fedprox", ["--mu", "1"]), warmup("fedbn")
    attack(["--global", g_sc, "--proto", "scaffold", "--u", "2", "--exact"] + cal, "scaffold II u=2 exact")
    attack(["--global", g_sc, "--proto", "scaffold", "--u", "2"] + cal, "scaffold II u=2 float32")
    attack(["--global", g_sc1, "--proto", "scaffold", "--scaffold-option", "I", "--u", "2"] + cal, "scaffold I u=2")
    attack(["--global", g_px, "--proto", "fedprox", "--u", "2", "--exact"] + cal, "fedprox u=2 exact")
    attack(["--global", g_bn, "--proto", "fedbn", "--u", "2", "--exact"] + cal, "fedbn u=2 exact")
    outs = sorted(glob.glob(os.path.join(edir, "*.json")))
    recs = {os.path.basename(f): json.load(open(f)) for f in outs}
    for f, r in recs.items():
        print(f"  {f}: proto {r['proto']} exact {r['exact']} u {r['u']} input {r['attack_input']} steps_txt "
              f"{r['steps_eff_txt']:.4f} rank {r['rank']} returned {r['n_recovered']}/{r['batch']} ROUGE-L {r['rougeL']} "
              f"ROUGE-1 {r['rouge1']} ROUGE-2 {r['rouge2']} token acc {r['token_acc']} label all/matched "
              f"{r['label_scores'].get('sign', {}).get('acc_all_fallback1')}/{r['label_scores'].get('sign', {}).get('acc_matched')} "
              f"aux round {r['aux_round']} bn_stats {r['bn_stats']}", flush=True)
    want = {"v0_B2_u1_exact.json", "v0_B2_u1.json", "v0_B2_u2_scaffold_optII_exact.json", "v0_B2_u2_scaffold_optII.json",
            "v0_B2_u2_scaffold_optI.json", "v0_B2_u2_fedprox_mu1_exact.json", "v0_B2_u2_fedbn_exact.json"}
    keys = {"seed", "proto", "u", "batch", "exact", "mu", "scaffold_option", "bn_stats", "aux_round", "noise_factor",
            "l1_thresh", "l2_thresh", "global", "rouge2", "token_acc", "per_original", "rougeL", "rouge1", "label_scores",
            "rec_ids", "matching", "calib_abs_residual", "args"}
    ok = set(recs) == want and all(keys <= set(r) for r in recs.values())
    ok &= recs["v0_B2_u2_fedprox_mu1_exact.json"]["steps_eff_txt"] < 2 and recs["v0_B2_u1.json"]["steps_eff_txt"] == 1.0
    ok &= recs["v0_B2_u2_scaffold_optI.json"]["attack_input"] == "scaffold_option1_dc_plus_ck"
    ok &= all(abs(np.mean(r["per_original"]["rougeL"]) - r["rougeL"]) < 1e-3 for r in recs.values())
    report("T5a attack_text.main on the CPU for every protocol", ok, f"{len(recs)} result files with all cell fields")
    df = score_text.main([edir, "--out", os.path.join(a.out, "tidy_e2e.csv")])
    t = analyze.load_tidy([os.path.join(a.out, "tidy_e2e.csv")])
    report("T5b score_text on the CPU results; analyze.load_tidy accepts them", len(t) == len(recs),
           "\n    " + df[["protocol", "u", "condition", "value", "m0", "rouge1", "rouge2", "token_acc", "n_recovered",
                          "label_acc_returned", "label_acc_matched", "label_acc_random_counts", "label_acc_majority"]].to_string(index=False).replace("\n", "\n    "))

print(f"done in {time.time() - t_start:.0f} s; {len(FAILS)} failure(s): {FAILS}")
