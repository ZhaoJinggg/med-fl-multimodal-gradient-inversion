# Label controls agent log (2026-11-22)

Task: label controls after the second review (A_lab, stratified permutation test, sign-flip test, label-distance
regression, cells across seeds). Files edited: src/score_images.py, src/analyze.py, src/test_analyze.py, and the
file-name line of src/attack_image.py. Local only (server down), nothing committed. Python: mmgia/.venv.

## Milestones
- [x] score_images.py: A_lab (rank_of(..., allowed), max_rule(..., labels)), n_abnormal, label_mismatch,
      x_share / rec_n_abnormal / group in pairs, A_lab + v/w_n_abnormal in the cross file (same file, extra
      columns), A_lab_mismatched, --scope cell (CELL_FIELDS), NAME_RE accepts optional s{seed}_ prefix.
- [x] analyze.py: stratified_perm_test, n_stratified_perms, sign_flip_test, label_regression, cell_perm_test,
      cross_matrix_strata; gate() new columns; floor/reference test stratified (--no-stratify); text
      reference precondition; tidy-images B<k>; docstring (text m0, label controls). Old tests still pass.
- [x] attack_image.py line 542: name = f"s{a.seed}_v{v}_B..." (nothing else; docstring line 37 now stale;
      src/test_attack_image.py lines 62/82 read "v0_B2_u{u}_mean" and will fail - not mine to edit).
- [x] test_analyze.py new tests (label_controls()): ALL TESTS PASSED locally (.venv, ~25 s).
      Counts 12 / 144 / 552,960; gate clients 0,0,8,8,8,3,6 -> 12 perms, p_min 1/12 = 0.083 > 0.05: the
      stratified gate test CANNOT reject at n = 7 whatever the data. Pure label effect: unstratified p 1e-5,
      stratified p 1.0 (exact) / 0.745 (noisy). Sign-flip exact = brute force. Regression: offdiag fit
      recovers slope 0.300 and excess 0.201 (true 0.2); fit WITH the diagonal (as specified) gives slope 0.320,
      excess 0.187 at n = 20, and 0.327 / 0.082 (true 0.30 / 0.10) at n = 7: the diagonal's high own label
      share (0.79 vs 0.46) lets an image effect leak into the slope. Both variants are output.
- [x] synthetic score_images check (scratchpad syn_score.py, 224 px cache, --no-lpips, ~1 min): two victims of
      B = 2 in two directories: seed 1 with the new name s1_v0_B2_u1_mean_syn (1 normal + 1 abnormal,
      originals + N(0, 0.05) noise) and seed 2 with the old name v0_B2_u1_mean_syn (2 abnormal, rec =
      0.12 orig + 0.88 another image + noise -> ranks 978 / 3). Checked: --scope cell -> 1 group, 4 cross
      rows, dir = both dirs joined; --scope directory -> 2 groups, controls NaN; n_abnormal 1 / 2 from the .pt,
      label_mismatch 0; every attack/mismatched row: rank_lab <= rank, N_lab = #same label - #other batch
      originals of that label, A_lab = 1 - (rank_lab - 1)/(N_lab - 1); x_share values; cross diagonal = per
      victim A / A_lab; INDEPENDENT recomputation of victim 2's A_lab with skimage SSIM over all 2,035
      same-label candidates: ranks 609 and 2 and A_lab identical to score_images.py. Wrong-victim A_lab: 100
      random B = 2 dataset batches against each of the 2 reconstruction batches -> mean 0.501 (SE 0.014).
      identifiers(): seed from s3_ prefix without JSON seed (seed_source file_name); JSON/name conflict raises.
      `analyze.py gate --regression` and `tidy-images` (condition oracle_B2) run on the synthetic output.

## Commands for the server (after the code reaches it via git)
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 python src/score_images.py runs/gate/image --seed 0 \
      --out runs/scores/gate_images_v2.csv
  python src/analyze.py gate runs/scores/gate_images_v2.csv --victims 3 4 5 6 7 8 9 --seed 0 --regression \
      --out runs/scores/gate_v2.csv

DONE (2026-11-22).
