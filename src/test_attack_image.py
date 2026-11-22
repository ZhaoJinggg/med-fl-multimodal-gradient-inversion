"""CPU tests for the protocol and text-condition wiring of attack_image.py (tiny settings: 224 px cache, real
FusionModel, victim 0 of seed 0 with B = 2, small u, a few iterations).

A1  FedAvg / oracle with default options == the pre-change script (exec'd from a reference copy with
    dev = "cpu"), bit for bit: .pt tensors and every JSON field of the old script except timings
A2  each protocol runs; cosine distance of the simulation at the true images to the upload (JSON
    checks.cos_true_oracle): ~0 for fedavg, fedprox, scaffold, fedavg --bn-stats local, fedbn when the
    victim's BN equals the server's; > 0 for fedbn with the victim's own BN from the warm-up (expected
    mismatch); BN prior off and BN parameters out of the distance under fedbn
A3  recovered text, exact texts with correct labels: u = 1 per-step features == oracle and the attack ==
    the oracle attack bit for bit; u = 3: step 1 exact, errors of the interpolated steps reported
A4  the label / selection rules of assign_texts and read_text_result's checks (pure functions)
A5  none: runs, the scalars receive gradients (move from 0), the split model with the true scalars
    reproduces the upload (cos ~0), text part of the head out of the distance; recovered with one text
    missing: only that slot gets scalars
A6  the three conditions share the initialization (JSON init_sha1)

usage (from mmgia/): python src/test_attack_image.py --ref <copy of the old attack_image.py> --out <scratch dir>
"""
import argparse
import json
import os
import random
import sys
import time

import torch

import attack_image as AI
import train_fl
from common import PairData, load_seed

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="copy of the pre-change attack_image.py")
ap.add_argument("--out", required=True, help="scratch directory (absolute)")
ap.add_argument("--tests", nargs="*", default=None)
ap.add_argument("--iters", type=int, default=3)
a = ap.parse_args()
RUN = lambda t: a.tests is None or t in a.tests
t_start = time.time()
os.makedirs(a.out, exist_ok=True)
DATA = PairData(224)
AI.PairData = lambda res: DATA           # main() loads the data once per call otherwise
split = load_seed(0)
fails = []

# the calibrated setting (paper TODO): both terms, BN weight 1, TV 1e-2, Adam lr 0.01; init mean
COMMON = ["--seed", "0", "--victims", "0", "--batch", "2", "--lr-img", "0.01", "--lr-txt", "0.01",
          "--iters", str(a.iters), "--checkpoints", str(a.iters), "--opt-lr", "0.01", "--tv", "1e-2", "--bn", "1.0",
          "--init", "mean", "--res", "224", "--device", "cpu"]


def report(name, ok, msg):
    if not ok:
        fails.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}  ({time.time() - t_start:.0f} s)", flush=True)


def run_new(out, extra, u):
    torch.manual_seed(0)   # FusionModel() draws the head at random; --global pretrained keeps that head
    res = AI.main(COMMON + ["--u", str(u), "--out", out] + extra)
    name = f"v0_B2_u{u}_mean"
    return res[0], torch.load(os.path.join(out, name + ".pt")), json.load(open(os.path.join(out, name + ".json")))


def same_pt(p1, p2):
    return all(torch.equal(p1[k], p2[k]) for k in ("rec", "orig", "labels")) and p1["studies"] == p2["studies"]


# ------------------------------------------------ A1
def run_ref(out, u, extra=()):
    src = open(a.ref, encoding="utf-8").read()
    for old, new in [('dev = "cuda"', 'dev = "cpu"'), ("data = PairData(a.res)", "data = _DATA"),
                     ("torch.cuda.synchronize()", "None")]:
        assert old in src, old
        src = src.replace(old, new)
    argv = sys.argv
    sys.argv = ["attack_image.py", "--global", "pretrained", *COMMON[:-2], "--u", str(u), "--out", out, *extra]
    torch.manual_seed(0)
    exec(compile(src, "attack_image_ref", "exec"), {"__name__": "__main__", "_DATA": DATA})
    sys.argv = argv
    name = f"v0_B2_u{u}_mean"
    return torch.load(os.path.join(out, name + ".pt")), json.load(open(os.path.join(out, name + ".json")))


def strip_time(j):
    j = dict(j)
    j.pop("minutes", None)
    j["curve"] = [{k: v for k, v in c.items() if k != "minutes"} for c in j["curve"]]
    return j


if RUN("A1"):
    for u, extra in ((2, []), (1, ["--restarts", "2"])):
        pr, jr = run_ref(os.path.join(a.out, f"A1_ref_u{u}"), u, extra)
        _, pn, jn = run_new(os.path.join(a.out, f"A1_new_u{u}"), ["--global", "pretrained"] + extra, u)
        jr_, jn_ = strip_time(jr), strip_time(jn)
        diff = [k for k in jr_ if jr_[k] != jn_.get(k)]
        report(f"A1 fedavg/oracle == old script (u={u}{', 2 restarts' if extra else ''})", same_pt(pr, pn) and not diff,
               f".pt bit for bit: {same_pt(pr, pn)}; old JSON fields that differ: {diff}; "
               f"cos old {jr['cos']} new {jn['cos']}, ssim {jr['ssim']} / {jn['ssim']}")

# ------------------------------------------------ warm-up runs for A2 (2 clients x 4 pairs, B = 2, 2 rounds)
def tfl(proto, extra=()):
    args = train_fl.get_args(["--proto", proto, "--seed", "0", "--rounds", "2", "--batch", "2", "--lr-img", "0.01",
                              "--lr-txt", "0.01", "--device", "cpu", "--out", os.path.join(a.out, "fl"), "--clients", "0", "1",
                              "--pairs-per-client", "4", "--eval-pairs", "4", "--aux-clients", "0", "1", *extra])
    _, out = train_fl.run(args, data=DATA)
    return os.path.join(out, "global_round2.pt")


if RUN("A2"):
    g = {"fedavg": tfl("fedavg"), "scaffold": tfl("scaffold"), "fedbn": tfl("fedbn"),
         "bnlocal": tfl("fedavg", ["--bn-stats", "local"])}
    n_img = None
    cases = [("fedavg", ["--global", g["fedavg"], "--proto", "fedavg"], True),
             ("fedprox mu=1", ["--global", g["fedavg"], "--proto", "fedprox", "--mu", "1"], True),
             ("scaffold II", ["--global", g["scaffold"], "--proto", "scaffold"], True),
             ("scaffold I", ["--global", g["scaffold"], "--proto", "scaffold", "--scaffold-option", "I"], True),
             ("fedavg --bn-stats local", ["--global", g["bnlocal"], "--proto", "fedavg", "--bn-stats", "local"], True),
             ("fedbn, victim BN == server BN (pretrained)", ["--global", "pretrained", "--proto", "fedbn"], True),
             ("fedbn, victim's own BN from the warm-up", ["--global", g["fedbn"], "--proto", "fedbn"], False)]
    for name, extra, exact in cases:
        tag = name.split(",")[0].replace(" ", "_").replace("=", "").replace("-", "")
        _, pn, j = run_new(os.path.join(a.out, "A2_" + tag + ("_pre" if "pretrained" in name else "")), extra, 3)
        if name == "fedavg":     # bit identity also from a saved broadcast model (head loaded, not drawn)
            pr, jr = run_ref(os.path.join(a.out, "A2_ref_fedavg"), 3, ["--global", g["fedavg"]])
            jr_, jn_ = strip_time(jr), strip_time(j)
            diff = [k for k in jr_ if jr_[k] != jn_.get(k)]
            report("A1c fedavg/oracle == old script from a saved global (u=3)", same_pt(pr, pn) and not diff,
                   f".pt bit for bit: {same_pt(pr, pn)}; old JSON fields that differ: {diff}; cos {jr['cos']} / {j['cos']}")
        c = j["checks"]
        if name == "fedavg":
            n_img = c["dist_tensors"]
        d = c["cos_true_oracle"]
        ok = d < 1e-5 if exact else d > 1e-4
        msg = f"cos at true images {d:.2e}; BN prior {j['bn_prior']}, bn {j['bn']}; distance over {c['dist_tensors']} tensors"
        if "bn_err_mean" in c:
            msg += f"; BN target err {c['bn_err_mean']:.1e}/{c['bn_err_var']:.1e}"
        if "fedbn" in name or "local" in name:
            ok = ok and not j["bn_prior"] and j["bn"] == 0.0 and "bn_err_mean" not in c
        if "fedbn" in name:
            ok = ok and n_img is not None and c["dist_tensors"] == n_img - 40
        else:
            ok = ok and (n_img is None or c["dist_tensors"] == n_img) and (j["bn_prior"] == ("local" not in name))
        ok = ok and j["proto"] == extra[extra.index("--proto") + 1] and j["condition"] == "oracle"
        report(f"A2 {name}", ok, msg + f"; JSON proto={j['proto']} mu={j['mu']} option={j['scaffold_option']} "
               f"bn_stats={j['bn_stats']} aux_round={j['aux_round']}")

# ------------------------------------------------ A3 / A5 / A6: text conditions
rows = DATA.rows(split["victim_batches"]["0"][:2])
_, ids0, mask0, y0 = DATA.batch(rows, "cpu")
true_ids = [ids0[i][: mask0[i].sum()].tolist() for i in range(2)]
y_true = [int(t) for t in y0.tolist()]
print("victim 0, B = 2: labels", y_true, "token counts", [len(t) for t in true_ids], flush=True)


def text_json(path, rec, labels, coefs, u, extra=None):
    d = {"victim": 0, "batch": 2, "u": u, "rec_ids": rec, "labels_sign": labels, "coef_ls": coefs}
    d.update(extra or {})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, "w"))


if RUN("A3") or RUN("A5") or RUN("A6"):
    G = ["--global", "pretrained"]
    # exact texts, correct labels, listed in reverse order; |coefficient| decreasing in the batch order, so
    # that the slots are filled in the victim's order (then the recovered attack must equal the oracle one)
    coef = [(1 if y_true[i] else -1) * (2.0 - i) * 1e-3 for i in range(2)]
    tdir = os.path.join(a.out, "text_exact")
    for u in (1, 3):
        text_json(os.path.join(tdir, f"v0_B2_u{u}.json"), true_ids[::-1], y_true[::-1], coef[::-1], u)
    res = {}
    for u in (1, 3):
        res[("oracle", u)] = run_new(os.path.join(a.out, f"T_oracle_u{u}"), G + ["--text", "oracle"], u)
        res[("recovered", u)] = run_new(os.path.join(a.out, f"T_rec_u{u}"), G + ["--text", "recovered", "--text-from", tdir], u)
        res[("none", u)] = run_new(os.path.join(a.out, f"T_none_u{u}"), G + ["--text", "none"], u)

    if RUN("A3"):
        _, po, jo = res[("oracle", 1)]
        _, pr_, jr = res[("recovered", 1)]
        c = jr["checks"]
        ok = c["text_feat_err"] == [0.0] and same_pt(po, pr_) and jr["cos"] == jo["cos"] and not jr["dist_drops_text_head"]
        report("A3a recovered (exact texts) == oracle at u = 1", ok,
               f"feature err per step {c['text_feat_err']}; attack .pt bit for bit {same_pt(po, pr_)}; "
               f"cos {jr['cos']} vs {jo['cos']}; slots {jr['text_assign']['slot_source']}")
        _, _, j3 = res[("recovered", 3)]
        c3 = j3["checks"]
        # pass criterion: step 1 exact and the replay at the true images unaffected; the feature errors of
        # steps 2..u are REPORTED, not tested: in this tiny setting (random head, both labels 0, lr_txt 0.01)
        # the logits saturate after step 1, so GPT-2 moves almost only in step 1 and the linear
        # interpolation (j-1)/u is far from the true trajectory at j >= 2 (see agent_notes/attack_image_wiring.md)
        ok = c3["text_feat_err"][0] == 0.0 and c3["cos_true"] < 1e-4
        report("A3b recovered (exact texts), u = 3: step 1 exact; errors of the interpolated steps reported", ok,
               f"max abs err per step {[f'{e:.2e}' for e in c3['text_feat_err']]}, relative "
               f"{[f'{e:.2e}' for e in c3['text_feat_relerr']]} (of the largest feature); relative L2 error of the "
               f"standardized features {[f'{e:.2e}' for e in c3['text_feat_zerr']]}; cos at true images {c3['cos_true']:.2e} "
               f"(oracle {c3['cos_true_oracle']:.2e})")
        # pattern form of --text-from and the file-name check
        pat = os.path.join(tdir, "v{v}_B{B}_u{u}.json")
        _, pp, _ = run_new(os.path.join(a.out, "T_rec_pattern"), G + ["--text", "recovered", "--text-from", pat], 1)
        report("A3c --text-from as a pattern", same_pt(pp, pr_), "same result as the directory form")

    if RUN("A5"):
        for u in (1, 3):
            _, _, jn = res[("none", u)]
            sc = torch.tensor(jn["scalars"])
            c = jn["checks"]
            ok = sc.abs().min().item() > 0 and c["cos_true"] < 1e-5 and jn["dist_drops_text_head"] \
                and c["dist_tensors"] == res[("oracle", u)][2]["checks"]["dist_tensors"] \
                and c["dist_numel"] == res[("oracle", u)][2]["checks"]["dist_numel"] - 768
            report(f"A5a none, u = {u}", ok,
                   f"scalars after {a.iters} iterations {jn['scalars']} (start 0); cos at true images with the true "
                   f"scalars {c['cos_true']:.2e}; distance drops 768 head weights: {c['dist_numel']}; final cos {jn['cos']}")
        # recovered with the second text missing
        tdir2 = os.path.join(a.out, "text_one_missing")
        text_json(os.path.join(tdir2, "v0_B2_u1.json"), true_ids[:1], y_true[:1], coef[:1], 1)
        _, _, jm = run_new(os.path.join(a.out, "T_rec_missing"), G + ["--text", "recovered", "--text-from", tdir2], 1)
        sc = jm["scalars"]
        slots = jm["text_assign"]["slot_source"]
        miss = [i for i, s in enumerate(slots) if s is None]
        ok = len(miss) == 1 and all(v == 0 for i, r in enumerate(sc) if i not in miss for v in r) \
            and all(v != 0 for i in miss for v in sc[i]) and jm["checks"]["cos_true"] < 1e-5 and jm["dist_drops_text_head"]
        report("A5b recovered with one text missing", ok,
               f"slots {slots}, scalars {sc}; cos at true images (recovered text + true scalar) {jm['checks']['cos_true']:.2e}")

    if RUN("A6"):
        for u in (1, 3):
            h = {k: res[(k, u)][2]["init_sha1"] for k in ("oracle", "recovered", "none")}
            report(f"A6 shared initialization, u = {u}", len(set(h.values())) == 1, str(h))

# ------------------------------------------------ A4: rules (pure functions)
if RUN("A4"):
    y = [1, 1, 0, 0, 0]
    T = lambda k: [100 + k, 200 + k]
    # (a) more than B returned: an error (revised spec: texts taken as stored, no selection rule)
    try:
        AI.assign_texts([T(k) for k in range(7)], [1, 1, 0, 0, 0, 1, 0], [0.9, 0.8, -0.7, -0.6, -0.5, 0.01, -0.02], y)
        ok_a = False
    except ValueError:
        ok_a = True
    s, info = AI.assign_texts([T(k) for k in range(5)], [1, 1, 0, 0, 0], [0.9, 0.8, -0.7, -0.6, -0.5], y)
    ok_a = ok_a and sorted(info["slot_source"]) == [0, 1, 2, 3, 4] and not info["reassigned"]
    # (b) label counts off: three 1s inferred, two known -> the 1 with the smallest |c| becomes 0
    labs = [1, 1, 1, 0, 0]
    co = [0.9, 0.8, 0.05, -0.7, -0.6]
    s, info = AI.assign_texts([T(k) for k in range(5)], labs, co, y)
    ok_b = info["reassigned"] == [2] and info["labels_used"]["2"] == 0 and s[0] == T(0) and s[1] == T(1) \
        and s[2] == T(3) and s[3] == T(4) and s[4] == T(2)
    # (c) fewer texts than B, one label over-represented: 3 zeros inferred among 3 texts, 3 known zeros, fine;
    #     2 texts both inferred 1 with 1 known... -> y = [1, 0, 0]
    s, info = AI.assign_texts([T(0), T(1)], [1, 1], [0.3, 0.1], [1, 0, 0])
    ok_c = info["reassigned"] == [1] and s == [T(0), T(1), None] and info["n_slots_with_text"] == 2
    # (d) trailing EOS stripped, empty sequences dropped, truncation to 64 tokens
    s, info = AI.assign_texts([[5, 6, AI.EOS, AI.EOS], [AI.EOS], list(range(70))], [0, 1, 1], [-0.2, 0.5, 0.4], [1, 0, 0])
    ok_d = s == [list(range(64)), [5, 6], None] and info["n_empty"] == 1
    # (e) no text at all
    s, info = AI.assign_texts([], [], [], [1, 0])
    ok_e = s == [None, None]
    # (f) read_text_result: missing coefficients / wrong identifiers are errors, not guesses
    p = os.path.join(a.out, "A4", "bad.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    errs = []
    for d in ({"victim": 0, "batch": 2, "u": 1, "rec_ids": [[1]], "labels_sign": [1]},
              {"victim": 0, "batch": 2, "u": 1, "labels_sign": [1]},
              {"victim": 0, "batch": 2, "u": 2, "rec_ids": [], "labels_sign": [], "coef_ls": []},
              {"victim": 0, "batch": 2, "u": 1, "proto": "fedbn", "rec_ids": [], "labels_sign": [], "coef_ls": []}):
        json.dump(d, open(p, "w"))
        try:
            AI.read_text_result(p, 0, 2, 1, 0, "fedavg")
            errs.append(False)
        except AssertionError:
            errs.append(True)
    ok_f = all(errs)
    # file-name convention
    root_rel = os.path.relpath(os.path.join(a.out, "A4"), AI.ROOT)
    ok_g = AI.text_result_path(root_rel, 3, 8, 4, 1, "fedavg", 2).endswith(os.path.join("A4", "v3_B8_u4.json")) and \
        AI.text_result_path("runs/t/seed{seed}_{proto}/v{v}_B{B}_u{u}.json", 3, 8, 4, 1, "scaffold", 2).replace("\\", "/") \
        .endswith("runs/t/seed1_scaffold/v3_B8_u4.json")
    ok_g = ok_g and AI.text_cell_suffix("fedavg", 1.0, "II", "upload") == ""         and AI.text_cell_suffix("fedprox", 1.0, "II", "upload") == "_fedprox_mu1"         and AI.text_cell_suffix("scaffold", 1.0, "II", "upload") == "_scaffold_optII"         and AI.text_cell_suffix("fedbn", 1.0, "II", "local") == "_fedbn"         and AI.text_cell_suffix("fedavg", 1.0, "II", "local") == "_bnlocal"         and AI.text_result_path(root_rel, 3, 8, 4, 1, "scaffold", 2, suffix="_scaffold_optII").endswith("v3_B8_u4_scaffold_optII.json")
    json.dump({"victim": 0, "batch": 2, "u": 1, "proto": "scaffold", "scaffold_option": "I", "rec_ids": [],
               "labels_sign": [], "coef_ls": []}, open(p, "w"))
    try:
        AI.read_text_result(p, 0, 2, 1, 0, "scaffold", opts={"scaffold_option": "II"})
        errs.append(False)
    except AssertionError:
        errs.append(True)
    ok_f = all(errs)
    report("A4 assign_texts / read_text_result / file names", all([ok_a, ok_b, ok_c, ok_d, ok_e, ok_f, ok_g]),
           f"more than B -> error, B as stored {ok_a}; reassign smallest |c| {ok_b}; fewer texts {ok_c}; EOS/empty/truncate {ok_d}; "
           f"no text {ok_e}; missing field / wrong identifier raise {errs}; names {ok_g}")

print(f"done in {time.time() - t_start:.0f} s; failures: {fails if fails else 'none'}")
