# attack_text wiring work log (src/attack_text.py, src/score_text.py, src/test_attack_text.py) -- 2026-11-22

Brief: protocols into attack_text.py (fedavg bit-identical), cell identification + auxiliary text metrics
in the result JSON, score_text.py -> tidy CSV for analyze.py; CPU tests only (no server, no GPU; the
laptop has a GPU too: tests run with CUDA_VISIBLE_DEVICES="" and --device cpu). No commit. Do not touch
attack_image.py, common.py, protocols.py, train_fl.py, score_images.py, analyze.py.

## Milestone 1: reading done
Findings that shape the code:
- DAGER's get_layer_decomp uses torch.svd_lowrank (random projection, CPU RNG). The old victim_update
  built a fresh FusionModel() after torch.manual_seed(1000 + v), which consumes CPU RNG (weight init),
  so the search's random state depends on it. protocols.client_upload deep-copies instead. To keep the
  FedAvg search reproducible, the new code builds (and discards) one FusionModel() on the CPU at the
  same place; the test checks that the CPU RNG state after the update equals the old one.
- DAGER is scale-invariant (check_if_in_span normalizes v; span from SVD; rank threshold relative to
  the noise floor), so the FedProx scale factor of dager() only changes nothing but the printed scale;
  the label bound s = lr_img * steps / B matters for c_box / the combined rule only (sign rule unaffected).
- ROUGE-L: per original, assignment value S[i, j], unmatched originals 0, mean over len(originals) = B.
  ROUGE-1 likewise (matched list + zeros). Both over all B originals.
- text_baseline.py: per victim, pool = all client rows of the seed without the victim's first-B batch rows,
  most frequent decoded (truncated) report, ROUGE-L against the victim texts, mean. Needs only text, so
  score_text.py recomputes it without loading images.
- analyze.py: cells filtered on condition == "recovered"; any other condition string stays apart from all
  tests and from the ANOVA. Exact-update rows -> condition "exact".
- DAGER code takes args.device everywhere -> --device (default cuda) is a trivial option.

## Milestone 2: code written (not yet fully tested)
- src/attack_text.py: get_args(argv) with --proto/--mu/--scaffold-option/--bn-stats/--aux-round/--device;
  script body moved into main(argv) (module importable for tests, functions read globals set by main).
  victim_update(model, ..., proto, exact, bn_local, names) = client_upload + server_delta (or option1_update
  for Option I) on NEEDED = two c_attn weights + head.weight; builds one FusionModel() first (RNG compat).
  span_rank() = the former rank lines; dager(dw, B, steps, multi); infer_labels(..., steps);
  match_texts() = the former inline matching (unchanged arithmetic); token_accuracy(), aux_text_metrics();
  cell_suffix() for file names (empty for default fedavg); JSON gains cell fields, rouge2, token_acc,
  per_original, steps_eff_*, attack_input, u_view, args. Under SCAFFOLD the correction is removed from the
  head too (label regression), via the same server_delta call.
- src/score_text.py: TextBaseline (text_baseline.py logic without images), row_of(), tidy(), main().
- src/test_attack_text.py: T1-T5 (see docstring). CUDA_VISIBLE_DEVICES=-1 (empty value is dropped on Windows).

## Milestone 3: T1 + T3 PASS (CPU, 265 s)
- T1 fedavg u=1,3 x exact/float32: 210 tensors bit for bit, logit0 equal, CPU RNG state after update equal; scale factors equal.
- T3 metrics: exact 1.0; one wrong token of 4: RL .75 R1 .75 R2 2/3 tokacc .75; unmatched -> 0, mean .5; shifted -> tokacc 0, RL .889.

## Milestone 4: T4 PASS, T2 criteria revised (T2 first run: my pass criteria were wrong, numbers fine)
- T4a m0 == text_baseline.py (0.2298/0.2063/0.221; majority acc .375/1/.875); T4b old local file -> 1 row;
  T4c 112 synthetic rows (88 recovered, 16 exact, 8 B4); T4d analyze: 10 comparisons H1/H3/H4/H5, n = 8, identical
  with/without exact + B4 rows; T4e duplicates rejected.
- T2 (B = 2, 79 tokens, random controls 0.8 g + noise): u=1 float32 error vs exact <= 2 spacings for fedavg AND
  scaffold-after-removal (each), so |d_s - d_f| <= 4 spacings; rel norm error 1.1e-3 on the DAGER layers both.
  Raw scaffold upload: no singular-value gap (s_k/floor 2.4 = s_k+1/floor), span angle 1.0 -> unusable; after
  server_delta: rank 501 vs 502 (nf 1.5), 79 vs 79 (nf 5), gap 8.4/15.8 = fedavg.
- FINDING u=3: scaffold server_delta float32 gap 4.3/7.4 vs fedavg 8.3/15.8, span angle .43/.15 vs .08/.07, while
  exact server_delta is fine (4e-4). Float32 application of the full-rank correction leaves rounding noise that the
  exact removal cannot take out -> possible H3 effect at u > 1 (open question for the author).
- Observation: nf 1.5 gives rank ~500 even for 79 tokens (the float32 noise spectrum, not the tokens, sets it).

## Milestone 5: T2 + T5 PASS; coordinator additions (m0 revised, mismatched control, label metrics) in score_text
- T5: CPU end-to-end DAGER works (--device cpu): fedavg u1 exact/f32, scaffold II u2 exact/f32, scaffold I u2,
  fedprox u2 exact (steps_txt 1.99), fedbn u2 exact (aux local loaded): B = 2, all 2/2 exact recoveries, 34-116 s each.
- score_text rewritten: m0 = B most frequent public (val+test) reports matched; m0_single (old); rougeL_mismatched
  (other victims of same dir/proto/u/batch/exact/file suffix, --mismatch-scope cell for across dirs);
  rougeL_recomputed; label (a) returned, (b) matched >= 0.5, (c) random with known counts, labels_mixed;
  JSON "all B" -> *_oracle_fallback. Tests T4a (m0_single), T4f, T4g added.
- scp of the server _lab files was DENIED by the permission system -> not run; command left for the user.

## Milestone 6: all tests PASS (T1-T5; T4 rerun with the revised score_text: T4a-T4g)
- T4f: m0 == independent computation (960 public pairs, top counts 18,18,15,12,11,10,10,7); seed 0 victims 0-2:
  m0 0.3407/0.3081/0.4403 vs m0_single 0.2298/0.2063/0.2210.
- T4g: mismatched control (exact-recovery synthetic seed-0 files) 0.309/0.311/0.311; label (a) 0.875, (c) random
  with counts 0.531/1.0/0.781 (victim 1 all abnormal: labels_mixed False).
- score_text on the T5 CPU outputs: rougeL_recomputed == stored ROUGE-L for real DAGER output.
- Pending: user must copy the server _lab files (scp blocked for me); then run score_text --default-seed 0.
