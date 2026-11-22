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

## Follow-up 2 (coordinator): max rule, cross matrix, permutation gate, 3-outcome decisions
- score_images.py rewritten: A = max rule (s(c) = max_j SSIM(c, rec_j), no matching); A_matched/top1_matched kept;
  assignment only for SSIM/PSNR/LPIPS. Cross matrix A(v, w) per group -> OUT_cross.csv (diagonal = real score).
  Group = (dir, protocol, condition, batch, u, init, tag, res); victims (seed, victim). JSON proto/protocol,
  condition, text("oracle") read. rank_scores() kept for the reviewer's scratch scripts.
- analyze.py rewritten: perm_test (exact n<=8, else 100k MC seed 0 + identity), gate on cross (--victims
  restricts before control), floor by test (image perm test on cross; text Wilcoxon vs m0), reference-cell test
  -> image "not tested", interval rules supported/contradicted/inconclusive, --cross/--batch options.
- test_analyze.py rewritten; ALL TESTS PASSED on server.
- Rerun: runs/scores_test/pilot_v3.csv (+_pairs, _cross), dirs push_b8 push_b24 b1_long attrib128 tune128,
  --seed 0 --protocol fedavg; log score_v3.log. DONE: 69 files in 434 s (median 1.8/3.5/7.1/14.1 s at B=1/2/4/8).
  Table (clients 0-2) in runs/scores_test/pilot_v3_table.csv, gate in gate_pilot_v3.csv (script runs/scores_test/code/pilot_table.py).
  both_bn1_12k / b1_long r128: max-rule A 0.996 / 0.891 / 0.729 / 0.560 at B=1/2/4/8 (matched 0.996/0.978/0.803/0.682);
  control (2 partners) 0.43 / 0.46 / 0.45 / 0.43 - below 0.5. Server src copies verified by md5.

## Design decisions
- Mismatched partner: group by (dir, B, u, init, tag, seed, res), sort by victim, partner = next victim (cyclic).
  Pool excludes the partner batch's other images. Reuses the SSIM-to-pool vectors, so costs nothing extra.
- analyze.py m0 = mean of per-victim m0 column over the reference cell (FedAvg u=1 recovered) unless --m0.
