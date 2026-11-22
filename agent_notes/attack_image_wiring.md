# attack_image.py wiring log (protocols + three text conditions) -- 2026-11-22

Brief: extend src/attack_image.py to every cell of the main experiment (proto x u x text condition);
edit only src/attack_image.py, new src/test_attack_image.py, this file. Local CPU only, no commit,
no server. Do not touch attack_text.py (another agent), common.py, protocols.py, train_fl.py,
score_images.py, analyze.py.

## Milestone 0 (start): reading done
- Spec: paper/main.tex Threat Model (l. 300-310), Attacks (l. 341-357), tables matrix / cells.
- Reference copy of the pre-change script for the bit-identity test:
  %SCRATCH%/attack_image_ref.py (scratchpad of this session; identical to git HEAD src/attack_image.py,
  commit 8780f61). If the scratchpad is gone: `git show 8780f61:src/attack_image.py`.
- Python: mmgia/.venv/Scripts/python.exe (torch 2.14.1, transformers 5.23.0).
- Text JSON (attack_text.py, current): file `<out>/v{v}_B{B}_u{u}{tag}.json`; fields used here:
  rec_ids (token id lists), labels_sign (per recovered text), coef_ls (per recovered text, sign rule's
  coefficient), victim, batch, u. Not present: seed, protocol, global path -> cannot be cross-checked.
- score_images.py reads JSON keys "seed", "protocol" (not "proto"), "condition" (else "text"), "global";
  file name must match v{v}_B{B}_u{u}_{init}{tag}; mismatched-control group = (dir, B, u, init, tag, res),
  so every cell needs its own --out directory (or tag).

## Milestone 1: code written (src/attack_image.py restructured: helpers + main(argv); test file written)
- Script now: get_args(argv), pure helpers (bn_targets(rs0, rs1, u), cosine_dist, drop_text_head, tv,
  ssim_matrix, score, resolve_aux_round, text_result_path, read_text_result, clean_ids, assign_texts,
  pad_ids, interpolated_text_features, SplitTextView, init_hash), main(argv) with the closures simulate /
  attack, load_aux_state (lazy train_fl.load_aux). --device added (default cuda; cpu for tests).
- Oracle/FedAvg path: client_upload replaces victim_update, simulate_image_steps replaces simulate; the
  plain FusionModel is used whenever every pair has a text, so the computation is unchanged.
- No-text pairs: SplitTextView (same parameter/buffer names, shares tensors) with h_txt = (h, s):
  h = txt_mu for pairs without text (standardized text input exactly 0), logit = head(z) + s.
  S (n, u) zeros, extra Adam group (lr --lr-scalar, default --opt-lr), same MultiStepLR schedule;
  distance drops head.weight[:, 512:]. Init of x' unchanged and drawn before S (S needs no RNG).
- Checks in JSON "checks": cos_true_oracle (true images, true features), cos_true (condition inputs;
  for none/missing with the TRUE scalars recorded along the oracle replay), bn_err_*, text_feat_err /
  relerr (per step, recovered feature vs nearest true feature), dist_tensors, dist_numel.
- Gotcha: PowerShell Set-Content wrote a BOM + CRLF; fixed with python (LF, no BOM). Use python/bash for
  rewriting files, never Set-Content.

## Milestone 2: first bit-identity run FAILED -> the CPU is not reproducible across model instances
- A1 (u=2): old cos 0.2366 vs new 0.1412 after 3 iterations. Probe: the OLD script run twice in one
  process also differs (check at true images -1.2e-7 vs 6.0e-8, final cos 0.334 vs 0.422, max |rec diff|
  2e-5). probe2 (same model object reused 3x): bitwise identical; probe3 (fresh FusionModel each time):
  delta, BN stats, d/dx all differ -> CPU kernels depend on memory addresses/alignment (MKL/oneDNN).
  Also: tiny numerical noise changes the early cos a lot (Adam's first steps follow the sign of tiny
  gradients), so "close" is not a usable criterion; need a reproducible backend for the test.
- Trying: torch.backends.mkldnn.enabled=False and/or MKL_CBWR=COMPATIBLE with fresh models.
- CORRECTION (real cause): not the CPU. With --global pretrained, FusionModel() draws the head at
  random from the global RNG, and the second run in one process starts from another RNG state -> other
  head. (probe3 also built fresh unseeded models.) Fix in the test: torch.manual_seed(0) before every
  run (old and new); plus A1c compares old vs new from a saved train_fl global (head loaded).
  Production cells always load a saved global, so this does not affect them.

## Milestone 3: A1 PASS (bit for bit, u=2; u=1 with 2 restarts; A1c u=3 from a saved global)
- A2 fedavg 0.0, fedprox -1.2e-7 PASS. scaffold FAILED (cos 0.648): bug, simulate() used the base
  Protocol without c - c_k (set only on the per-victim copy). Fixed: simulate reads cur["proto"].

- A2 fedavg --bn-stats local 0.0; fedbn victim BN == server BN 6e-8 (22 tensors = 62 - 40 BN);
  fedbn victim's own BN 3.3e-2 (expected mismatch). PASS.
- Coordinator spec change (2026-11-22, independent paper review): (1) the "keep the 8 largest |coef|"
  rule is dropped; texts are taken as stored in rec_ids with labels_sign; label-count repair by
  smallest |coef_ls| stays; missing text -> no-text treatment stays. Implemented: more than B
  sequences now raises ValueError. (2) JSON must carry condition in {oracle, recovered, none}, proto,
  seed (already so); "text" key kept (task said keep existing fields; score_images only falls back to it).

## Milestone 4: full suite (scratch t2) after the SCAFFOLD fix: everything PASS except A3b
- A2 scaffold II / I now -1.2e-7 (BN target err 1.2e-6). fedbn own BN 3.6e-2 (expected).
- A3b finding (real, not a bug): u = 3, exact texts, pretrained random head, labels [0, 0], lr 0.01:
  standardized text features of the interpolated steps 2 and 3 are off by 35% and 17% (rel. L2); text
  logit contribution oracle -1.33/-1.77 at steps 2-3 vs interpolated -0.42/-0.80 (step 2). Cause
  (dbg_interp.py): logits go to -10 after step 1 (p - y ~ 5e-5), so GPT-2 moves almost only in step 1
  and the uploaded delta ~ the step-1 change; linear interpolation (j-1)/u then lags. Here it had no
  effect on the replay (cos at true images -1.2e-7) because steps 2..u carry ~no gradient. On the real
  round-9 globals the size of this error is UNKNOWN (lr_txt = 0.01 is large for GPT-2); the JSON records
  checks.text_feat_zerr per victim so it can be read off the real runs. A3b now passes on step-1
  exactness + replay, and reports the errors.
- attack_text.py (other agent) now writes v{v}_B{B}_u{u}{cell_suffix}{tag}.json with suffix
  _fedprox_mu{mu}, _scaffold_opt{opt}, _fedbn, _bnlocal, _exact, and stores seed, proto, mu,
  scaffold_option, bn_stats, exact. Adopted: directory form looks for the same suffix (text_cell_suffix),
  pattern gets {suffix}, these options are cross-checked, an exact-reference text file gives a warning.
- Revised spec: more than B texts -> ValueError (A4 updated).

## Milestone 5 (final): all tests PASS on the final code
- Full suite scratch t2 (A1, A1c, A2 x7, A3a, A3c, A5a x2, A5b, A6 x2) + rerun of A3/A4 (scratch t3)
  after the last edits (suffix convention, more-than-B error, zerr). ~4 min on CPU.
- Reference copy deleted from the scratchpad; to rerun A1: `git show 8780f61:src/attack_image.py > ref.py`
  then `python src/test_attack_image.py --ref ref.py --out <scratch>`.
- Untested: CUDA, 128 px (no local cache; would write data/cache/images128.npz), full budgets, B = 8,
  mixed labels in an integrated run (victim 0's first two labels are both 0; the label rules are unit-
  tested in A4), real round-9 globals / real text results, several victims per call.

## Readings chosen where the paper is not specific
1. Slots: dummy image i keeps the known label y_i (batch order); recovered texts of label l fill the
   slots of label l in index order by decreasing |coef|. Dummies are exchangeable within a label, so the
   order carries no information about the originals; .pt "labels" unchanged.
2. Rules order: drop empty sequences (after stripping EOS); more than B -> error (revised spec); then
   reassign the over-represented label's smallest |coef| texts (with fewer than B texts, only when a
   label exceeds its known count). Coefficient = coef_ls (the one whose sign gives labels_sign).
3. A pair without text -> scalar per step for that pair, and the text part of the head is dropped from
   the distance for the whole batch (its update needs every pair's text features).
4. Interpolation uses the uploaded GPT-2 delta as is (includes the SCAFFOLD correction / proximal
   effect: "the uploaded weights").
5. Recovered ids: trailing EOS stripped, truncated to 64 tokens, padded like common.tokenize.
6. Scalars: start 0, Adam lr --lr-scalar (default --opt-lr), same MultiStepLR, no clipping, --signed
   applies to x' only. BN prior kept under `none` (statistics are uploaded).
7. FedBN: BN params are stepped in the replay (as on the client) with the broadcast (pretrained) values,
   and are out of the distance because they are not in the upload. --bn-stats local: prior off, BN
   params in the distance.
8. SCAFFOLD Option I: same weight delta as Option II, so the attack is unchanged; an "Option I view"
   (single-step gradient dc + c_k) is NOT implemented here.
9. Script default --init stays "random"; the paper's cells need --init mean (calibrated setting).

## Plan
1. Restructure into helpers + main(argv) (importable for tests), same operations on the oracle path.
2. Protocol wiring via protocols.client_upload / simulate_image_steps / train_fl.load_aux.
3. Text conditions oracle / recovered / none (NoText view of the model for scalar contributions).
4. Tests in src/test_attack_image.py.
