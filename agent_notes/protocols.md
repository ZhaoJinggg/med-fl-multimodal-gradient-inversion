# Protocols work log (src/protocols.py, src/train_fl.py) -- 2026-11-22

Brief: implement FedProx (mu = 1), SCAFFOLD (Option II, Option I as an extra option), FedBN next to
FedAvg; local CPU smoke tests only; no commit; do not edit attack_image.py / attack_text.py / common.py.

## Milestone 1 (about 18:00): review of the previous agent's uncommitted files
Both files were essentially complete (no truncated code). Read critically against paper/main.tex,
progress.md, design_review_2026-11-21.md, common.py, warmup.py, attack_image.py, attack_text.py.

Paper passages that fix the specification:
- Table I (protocols): FedAvg weight difference; FedProx proximal term; FedBN difference without BN
  parameters; SCAFFOLD difference plus change of the client's control variate.
- Sec. II-A (l. 63): FedBN keeps BN layers local, excludes their parameters from averaging.
- Threat model (l. 306): server knows mu and the control variates; reconstructs c_k by accumulating the
  uploaded changes (full participation); FedAvg/FedProx/SCAFFOLD upload BN statistics, FedBN does not.
- Image attack (l. 346): proximal term / SCAFFOLD correction enter the unrolled steps.
- BN prior (l. 353): FedBN -> BN prior off, simulation uses the pretrained BN parameters and running
  statistics, BN parameters left out of the distance.
- Text attack (l. 355): SCAFFOLD correction -u eta (c - c_k) removed from the GPT-2 part before DAGER;
  FedProx needs no step (only rescales earlier gradients).
- Calibration (l. 477): with mu = 1 the proximal term rescales step j by (1 - eta mu)^(u-j); eta mu u
  reported per parameter group -> the proximal term uses each group's own learning rate.
- Table fixed settings: mu = 1; SCAFFOLD Option II. Warm-up (l. 297): 9 rounds per protocol, 1 epoch,
  B = 32; attacked round 10 replayed from the saved round-9 state.

Kept from the previous agent (checked line by line):
- protocols.local_step == common.sgd_step operation for operation under fedavg; direction() adds
  mu (w - w^t) and/or (c - c_k) before the per-group learning rate.
- client_upload == attack_image/attack_text victim_update under fedavg (same op order); exact=True
  reproduces attack_text's float64 accumulator.
- simulate_image_steps == attack_image.simulate under fedavg; adds the protocol term per step.
- SCAFFOLD Option II with per-group eta: c_k+ = c_k - c + (w^t - w^(K)) / (K eta_group).
- FedBN: BN selected by module type (includes downsample.1); all BN state (weight, bias, running
  stats, num_batches_tracked) local; server BN entries stay pretrained.
- train_fl: FedAvg path reproduces warmup.py's RNG order, permutation generator and averaging order.

Planned changes:
- protocols: remove unused rs0; remove SCAFFOLD correction in float64 (avoid adding rounding);
  helper for Option I (server gets the single-step gradient dc + c_k); flag for "BN statistics
  uploaded" (paper Sec. impl. l. 555 names this choice); assert controls set under scaffold.
- train_fl: strict check of missing keys under FedBN instead of strict=False; optional
  --bn-stats local (running statistics not uploaded, for FedAvg-family); FedBN also logs the
  accuracy of the broadcast model with the server's (pretrained) BN.

## Milestone 2 (about 18:40): code changes done
- protocols.py: Protocol(bn_stats="upload"|"local"); local_state_keys(); client_upload asserts the
  controls under scaffold and loads bn_local for any protocol with local keys; correction in float64
  (scaffold_correction / remove_correction); option1_update() (Option I single-step view);
  effective_steps() (sum (1 - eta mu)^(u-j), scale factor for the text attack under FedProx).
- train_fl.py: --bn-stats; generic local keys (FedBN: all BN; bn-stats local: BN buffers); strict
  missing-key check (BatchNorm silently refills num_batches_tracked, so that key is excluded from the
  check); local state saved as local_round{r}.pt {"keys", "clients"}; FedBN also logs
  test_acc_server_state; load_aux(run_dir, r, client) -> {"c", "c_k", "local"}.
- src/test_protocols.py: smoke tests T1-T9 (CPU, real FusionModel + data at 224 px from the existing
  cache, B = 2, u = 3; train_fl with 2 clients x 4 pairs, B = 2, 2 rounds).

## Milestone 3 (about 18:50): all 25 checks PASS (CPU, about 85 s + 125 s)
- T1 fedavg client_upload == verbatim copies of attack_image / attack_text victim_update: image delta,
  BN stats, h_txt, full delta, logit0, exact delta all bit for bit.
- T2 simulate_image_steps(fedavg) == attack_image.simulate: delta, BN stats, d/dx' bit for bit.
- T3 simulation at the true images vs upload, cosine distance: fedavg 0, fedprox 0, scaffold 2e-16,
  fedbn with the victim's BN 0; fedbn with the server's BN (victim BN = pretrained x (1 + 0.1 N(0,1)))
  0.108 -> the BN-parameter mismatch alone moves the image-side direction noticeably.
- T4 fedprox mu=0 == fedavg bit for bit; mu=1 exact upload vs -eta sum (1-eta mu)^(u-j) g_j: 7.6e-5
  max rel err; unweighted sum is off by 2.0e-2 (= eta mu (u-1)).
- T5 scaffold c = c_k = 0 == fedavg bit for bit.
- T6 scaffold: exact upload minus correction == -eta sum g_j (corrected trajectory) 2.9e-7; float32
  upload 9.1e-4 on the DAGER layers (|correction|/|gradient part| up to 2.4); u=1 minus correction vs
  fedavg 6.0e-4; Option II dc == -c - delta/(u eta) 2.2e-7 (no information beyond delta); Option I
  dc + c_k vs gradient at w^t 5.6e-8 on the DAGER layers, against 3.1e-4 for the float32 fedavg u=1
  delta -> Option I hands the server an almost rounding-free single-step gradient.
- T7 fedbn: exactly the 40 BN tensors (20 layers incl. downsample.1) are left out, no statistics, rest
  == fedavg; fedavg --bn-stats local: all parameters, no statistics.
- T8 train_fl: scaffold round 1 == fedavg round 1 bit for bit, round 2 differs; fedprox mu 0 == fedavg
  after 2 rounds; fedprox mu 1 differs; fedbn server BN == pretrained, clients' BN moved and differ;
  c == mean_k c_k (2.3e-7); Option I and --bn-stats local run; load_aux works.
- T9 train_fl fedavg == warmup.py itself (exec'd with dev cpu, 2 clients x 4 pairs, 4 eval pairs):
  global_round1.pt and global_round2.pt bit for bit.
Not exercised: CUDA, 128 px (would write a new image cache), full-size runs, the attack scripts with
the wiring applied, bn_local with --bn-stats local in client_upload.

## Wiring (not applied; attack_text.py belongs to another agent)
See the final report; summary: client_upload replaces victim_update, simulate_image_steps replaces
simulate, server_delta before dager, BN prior off when up["bn_before"] is None, Option I via
option1_update with u = 1, FedProx scale factors via effective_steps.

## Open questions for the author (most important first)
1. Option I cell: not in the paper; if added, say that it gives a single-step AND rounding-free gradient.
2. BN running statistics uploaded and averaged under FedAvg/FedProx/SCAFFOLD (as warmup.py); no control
   variates for buffers.
3. FedBN accuracy: mean of client models (test_acc) or broadcast model with pretrained BN
   (test_acc_server_state)? Paper says "test accuracy of each broadcast model".
4. Control variates on all parameters, per-group eta in Option II, server step 1, K = local steps.
5. Proximal term scaled by each group's eta; mu on all parameters incl. BN affine, not on buffers.
6. Label bound / DAGER scale under FedProx: effective_steps instead of u (3.4% at u = 8).
7. SCAFFOLD float32: correction removal exact, residual keeps the weight-difference rounding.
