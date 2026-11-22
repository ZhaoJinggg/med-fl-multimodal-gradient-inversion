# Scoring/statistics agent log (2026-11-22) - DONE

Task: new files src/score_images.py, src/analyze.py, src/test_analyze.py. No existing file edited; nothing committed.
Server copies (scp, new files only): ~/jacob/mmgia/src/{score_images,analyze,test_analyze}.py.
Server outputs only in runs/scores_test/ (pilot.csv, pilot_pairs.csv, score.log, identify.log, tidy_pilot.csv,
synthetic_tidy.csv, analysis_synth/). runs/gate/ was not touched.

## Results
- torch SSIM (banded-matrix 'valid' Gaussian filter, float64) vs skimage: max diff 5.6e-15 (80 pool pairs, 128 px);
  per-file B x B check vs skimage on the .pt originals (float32): max 7.5e-9 over all 30 pilot files (128 and 224).
- A and ranks identical to src/identify.py on 4 files (v0/v1 B8 both_bn1_12k, v2 B4 both_bn1_12k, v0 B1 r224):
  A 0.6734/0.6917/0.7298/0.9991, all ranks equal. Mean-image control differs by design (val+test mean vs all 3,199).
- CPU time (8 threads, nice 10, GPU job running): ~14 s per B=8 file at 128 px, ~7 s B=4, ~3.6 s B=2, ~2 s B=1;
  224 px B=1 ~6 s. 30 files in 278 s total including LPIPS.
- test_analyze.py: ALL TESTS PASSED (known shift significant, equal groups within margin, floor cell not decidable,
  zero differences, H2b third outcome, gate-not-informative, m0 override, ANOVA, gate p = 1/128 for n=7).

## Follow-up (coordinator): all-partner mismatched control
- A_mismatched now = originals of v held fixed vs reconstructions of every other victim w in the group
  (dir, B, u, init, tag, res; seed NOT required); pool = dataset minus v's other images. Mean over w;
  A_mismatched_sd, n_partners added. Old single-partner value kept as A_mismatched_next (free).
- Pairs file: kind = mismatched (per partner, partner_file/partner_victim) and mismatched_next. No LPIPS for controls.
- analyze.py docstrings updated (gate/tidy use A_mismatched unchanged). Uploaded to server src/.
- Rerun DONE: runs/scores_test/pilot_v2.csv (+_pairs), 269 s for 30 files (~14 s per B=8 file, control is free).
  test_analyze still passes; server src copies verified identical by md5.
- B=8 both_bn1_12k (v0,v1,v2): A .673/.692/.681, new ctrl .551/.464/.613, old next .498/.570/.535.
  B=2 both_bn1_12k: A .9995/.9923/.9434, new ctrl .792/.453/.465, old next .469/.555/.761.
  Only 2 partners in the pilot; the new control follows v's typicality (correlates with mean-image A).

## Design decisions
- Mismatched partner: group by (dir, B, u, init, tag, seed, res), sort by victim, partner = next victim (cyclic).
  Pool excludes the partner batch's other images. Reuses the SSIM-to-pool vectors, so costs nothing extra.
- analyze.py m0 = mean of per-victim m0 column over the reference cell (FedAvg u=1 recovered) unless --m0.
