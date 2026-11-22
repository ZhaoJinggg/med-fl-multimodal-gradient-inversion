# Label rule work log (attack_text.py infer_labels) -- 2026-11-22

## Milestone 1: review of the previous agent's uncommitted diff
- Previous agent had implemented essentially all of the target state in src/attack_text.py:
  infer_labels(dw, texts_ids, B) (dedup of exact duplicates, CPU float64 np.linalg.lstsq rcond=None,
  bvls lsq_linear with |c| <= lr_img*u/B, p_hat text-only with image part = 0, labels_sign/combined/model,
  label_fit stats), score_labels (all-B with fallback 1, matched ROUGE-L >= 0.5 with count), always_abnormal,
  JSON fields rec_ids, matching, labels_true, coef_ls, coef_box, p_hat_text_only, calib_abs_residual
  (from victim_update's first-step logits), --gpu-mem default 0.5.
- Checked against common.py: head input = cat[(h_img-img_mu)/img_sd, (h_txt-txt_mu)/txt_sd]; GPT-2 dropout 0,
  so train mode is harmless for p_hat. Head is "image" group (lr_img). DAGER path and victim_update RNG untouched.
- My changes: cond_Z guarded against sv[-1] == 0 (would write Infinity), spacing in --gpu-mem line. Compiles.
- Next: check server (_lab files, running processes), copy file, run u=1, u=4, exact u=1.

## Milestone 2: server state at 17:43 (server clock)
- Server src/attack_text.py (17:35) == my local version except the two cosmetic lines (cond_Z guard, spacing).
- _lab results for u=1 and u=4 (clients 0-2) already exist (17:37-17:42, logs/lab_u1.log, logs/lab_u4.log),
  produced by the previous agent's runs with this code; no attack_text process left running.
- u=1 _lab reproduces the frozen setting exactly: ROUGE-L 0.700/0.4883/0.539, 8/8 returned, rank 504/500/502,
  old label_acc 0.75/0.625/0.375 (identical to v*_B8_u1_g2_nf1.5_l1_5e-2_l2_1.5e-2.json) -> text recovery unchanged.
- u=4 _lab: ROUGE-L 0.579/0.4905/0.3552, 8/8 returned, rank 494/488/490 (no earlier u=4 run at the frozen setting
  to compare with; thresholds of that run not recorded in the JSON). Copies saved locally in scratchpad prev_lab/.
- Next: copy file, run --exact --u 1 (tag _lab_exact), then rerun u=4 to confirm thresholds/determinism.

## Milestone 3: exact reference run done (logs/lab_exact.log, tag _lab_exact)
- ROUGE-L 0.875/1.000/1.000, returned 7/8/8, label_acc 1.0 x3 (= earlier exact runs). GPU peak 73 C during run.
- sign: all 0.875/1.0/1.0, matched 7/7, 8/8, 8/8; combined: all 0.75/1.0/1.0, matched 6/7, 8/8, 8/8;
  model: all 0.75/0.75/0.75, matched 6/7, 6/8, 6/8; always-abnormal all 0.375/1.0/0.875.
- Next: rerun u=4 (tag _lab) to confirm thresholds/determinism vs backup in scratchpad prev_lab/.

## Milestone 4: u=4 rerun done (logs/lab_u4_r2.log; overwrote the _lab u4 JSONs, old copies in scratchpad prev_lab/)
- Label scores of all three rules identical to the previous agent's u=4 run. Text recovery is NOT bit-identical
  at u=4 run to run: v1 rank 488->489 (ROUGE-L 0.4905 both), v2 rank 490->492, ROUGE-L 0.3552->0.3597,
  old label_acc 0.5->0.625 (GPU float32 nondeterminism compounding over 4 steps; DAGER path untouched by the edit).
- GPU peaked 75 C during my runs (watchdog stops at 78). No attack_text process of mine left; image job untouched.

## Results (counts; "all" = 8 per client with fallback 1, "m" = matched ROUGE-L >= 0.5)
| setting | rule | all (v0,v1,v2) | matched (v0,v1,v2) |
| u=1 f32 | sign | 23/24 (8,8,7) | 11/11 (5/5,3/3,3/3) |
| u=1 f32 | combined | 23/24 (8,8,7) | 11/11 |
| u=1 f32 | model | 20/24 (8,6,6) | 8/11 (5,1,2) |
| u=1 f32 | always 1 | 18/24 (3,8,7) | 6/11 (0,3,3) |
| u=4 f32 | sign | 21/24 (6,8,7) | 8/9 (3/4,3/3,2/2) |
| u=4 f32 | combined | 20/24 (7,7,6) | 7/9 (4,2,1) |
| u=4 f32 | model | 19/24 (7,6,6) | 6/9 (4,1,1) |
| u=4 f32 | always 1 | 18/24 (3,8,7) | 5/9 (0,3,2) |
| exact u=1 | sign | 23/24 (7,8,8) | 23/23 (7/7,8/8,8/8) |
| exact u=1 | combined | 22/24 (6,8,8) | 22/23 (6,8,8) |
| exact u=1 | model | 18/24 (6,6,6) | 18/23 (6,6,6) |
| exact u=1 | always 1 | 18/24 (3,8,7) | 18/23 (3,8,7) |
Reading: combined never beats sign (ties u=1, worse u=4 and exact); its errors come from p_hat (image part zeroed)
differing from the full-model p (exact v0 orig 5: c/s = 0.145 = y - p exactly, but p_hat 0.31 vs p 0.855), and at
u=4 c/s shrinks so combined ~ model. Sign errors sit on garbage recoveries; on matched texts sign is 11/11, 8/9, 23/23.
