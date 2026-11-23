# F110 — HiSA (Hidden Sparse Attention) on F107b_e5 (one seed per arm)

Paper: "HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs", Borbely, Salzmann, Verscheure, 2026, arXiv 2611.26363.
Code: github.com/ambroise-b/hisa (commit 0fae2ce, Apache-2.0).
The project owner approved the run on 2026-11-23 10:10 UTC. One epoch = the standard 5100-step BlaGPT full run.

**Push pending.** The mission sandbox cannot push. Commits 18f26c33..HEAD are on local `main` of push-repo.
The parent session must push them (plain fast-forward on origin/main 187cb7de, no force).

## Verdict

- **A (HiSA, base 1e6): keep by the rule, but the second seed is not run.** A gives 3.1124. The bar is 3.1153.
  A is 0.0029 lower. Seed noise is 0.0024, so A passes by 0.0005 only. A seed-2 run is a decision for the project owner. I did not run it.
- **Confound for A.** The F107b_e5 run (3.1153) ran at commit d8fd8233. That commit has no `lattice_scan_impl` and no
  `lattice_film_impl`, so the baseline ran with the Hillis-Steele scan and fp32 FiLM. The F110 arms ran at 187cb7de with
  `best_model_config.py` defaults `triton` and `bf16cat`. Thus "A minus 3.1153" holds the HiSA change AND the speed-option change.
  bf16cat changes numerics. No dense run at 187cb7de with base 1e6 exists, so I cannot split the two effects.
  A clean test needs one dense base-1e6 run at 187cb7de (same seed). That is a decision for the project owner.
- **B vs C (base 1e4): HiSA does not help and does not hurt at the end.** B 3.1243 = C 3.1243 (difference 0.0000).
  At equal wall time, B is 0.0034 lower than C, because B runs 1.2 % faster per step (see below).
- **Base effect:** base 1e4 is worse than base 1e6. C (dense, 1e4) is 3.1243 vs 3.1153 (+0.0090), with the same confound as above.
  A (1e6) vs B (1e4), both HiSA and both at 187cb7de: +0.0119 for base 1e4.

## Results per arm

Numbers come from the run logs (`tables.py`, not typed by hand).

| Run | RoPE base | HiSA | rho(1024) | Final val_loss | Δ vs 3.1153 | step_avg ms | train_time ms | Peak MiB | Params |
|---|---|---|---|---|---|---|---|---|---|
| F107b_e5 (base, d8fd8233) | 1e6 | off | 1.0 | 3.1153 | 0 | 732.44 | 3728122 | 70569 | 404636689 |
| F110A | 1e6 | on | 0.6241 | 3.1124 | -0.0029 | 663.60 | 3377744 | 60237 | 404636689 |
| F110B | 1e4 | on | 0.4438 | 3.1243 | +0.0090 | 676.91 | 3445456 | 60237 | 404636689 |
| F110C | 1e4 | off | 1.0 | 3.1243 | +0.0090 | 685.27 | 3488042 | 58478 | 404636689 |

rho = kept fraction of the causal pair products (paper eq. 5, T=1024, d=64). The estimates were 0.62 and 0.44.
At T=384 (short-sequence phase): 0.6949 (1e6) and 0.5503 (1e4). Reach max(L_l): 1020047 (1e6) and 11780 (1e4).
HiSA adds 0 parameters.

### Step time (honest view)

- A 663.60 ms vs the base 732.44 ms looks like -9.4 %. **Most of this gap is not HiSA.** The base ran without the
  Triton scan and bf16cat FiLM (see confound). The F107e notes give these options as speed options.
- The fair speed pair is B vs C (same commit, same base): 676.91 vs 685.27 ms, HiSA is **1.2 % faster**.
  A vs C (different base only, and the base does not change the compute): 663.60 vs 685.27 ms, **3.2 % faster**.
- Peak memory: HiSA 60237 MiB vs dense 58478 MiB at the same commit (+1759 MiB, +3.0 %).
  The 70569 MiB of the base is again the old fp32 FiLM path.

## Val loss by step vs F107b_e5

| Step | F107b_e5 | F110A | F110B | F110C | A − base | B − C |
|---|---|---|---|---|---|---|
| 125 | 4.8352 | 4.7553 | 4.7950 | 4.8296 | -0.0799 | -0.0346 |
| 250 | 4.3953 | 4.2695 | 4.2893 | 4.4104 | -0.1258 | -0.1211 |
| 500 | 4.1647 | 3.9660 | 3.9847 | 4.1893 | -0.1987 | -0.2046 |
| 1000 | 4.0701 | 3.7705 | 3.7851 | 4.1248 | -0.2996 | -0.3397 |
| 1125 | 4.0764 | 3.7452 | 3.7549 | 4.1157 | -0.3312 | -0.3608 |
| 1250 | 3.6401 | 3.6259 | 3.6432 | 3.6561 | -0.0142 | -0.0129 |
| 1500 | 3.5276 | 3.5231 | 3.5343 | 3.5382 | -0.0045 | -0.0039 |
| 2000 | 3.4508 | 3.4484 | 3.4594 | 3.4588 | -0.0024 | +0.0006 |
| 2500 | 3.4080 | 3.4081 | 3.4196 | 3.4192 | +0.0001 | +0.0004 |
| 3000 | 3.3854 | 3.3838 | 3.4046 | 3.3951 | -0.0016 | +0.0095 |
| 3500 | 3.3242 | 3.3223 | 3.3346 | 3.3336 | -0.0019 | +0.0010 |
| 4000 | 3.2592 | 3.2564 | 3.2677 | 3.2686 | -0.0028 | -0.0009 |
| 4500 | 3.1893 | 3.1866 | 3.1984 | 3.1983 | -0.0027 | +0.0001 |
| 5000 | 3.1214 | 3.1184 | 3.1304 | 3.1304 | -0.0030 | +0.0000 |
| 5100 | 3.1153 | 3.1124 | 3.1243 | 3.1243 | -0.0029 | +0.0000 |

Observation: until step 1125 the step_avg is ~490 ms, and at step 1250 it jumps. The training sequence length becomes
longer at that point (inferred from the step-time jump; I did not read the schedule code). Before the jump, the dense
runs stall at ~4.07-4.12 while both HiSA runs keep falling (gap up to 0.36). A likely cause: validation runs at the full
length, and HiSA, which keeps fast RoPE pairs only at short distances, extrapolates better past the short training length.
I did not test this cause. After the jump, the gap closes in about 250 steps.

## Equal-time comparison

val_loss of each run at the train_time where the named arm ended (linear interpolation between val points):

| Reference time | ms | F107b_e5 | F110A | F110B | F110C |
|---|---|---|---|---|---|
| end of F110A | 3377744 | 3.1690 | 3.1124 | 3.1299 | 3.1355 |
| end of F110B | 3445456 | 3.1557 | 3.1124 | 3.1243 | 3.1277 |
| end of F110C | 3488042 | 3.1477 | 3.1124 | 3.1243 | 3.1243 |

The base column looks bad at equal time, but only because the base ran at 732 ms/step without the speed options.
The fair equal-time pair is B vs C: at the end of B, C is at 3.1277, so B is 0.0034 lower.

## Method

- Rule (eq. 3): s_ij = (1/√d) Σ_l 1[(i−j)·θ_l < Φ] ⟨rot q_i^(l), rot k_j^(l)⟩, θ_l = b^(−2l/d), Φ = 90°, causal i ≥ j.
  Only q·k changes. Q/K projections, values, softmax, RoVE, XSA, the gate and the GOAT sink stay as they are.
- Exact per-pair rule, not the Flash block rule. Integer windows L_l = ceil(Φ/θ_l) in float64; pair l counts iff D < L_l.
- RoPE layout: BlaGPT `apply_rotary_emb` (class `Rotary`) is half-split, so pair l = dims (l, l+32). The thresholds use this map.
- Best-config attention: `ComposableGatedAttention` (XSA + GOAT sink + gate, a subclass of `GOATSinkAttention`),
  n_head 12, head_dim 64, `pos_encoding="rotary"`, `rope_variant="standard"`. HiSA acts on this path.
- Kernel: the upstream exact-rule Triton kernel (`triton_exact`), adapted as a torch custom op (`hisa_triton.py`,
  with the Apache-2.0 notice), so torch.compile + DDP run it as in the full run.
- GOAT sink: the kernel does not add the per-head sink logit u at key 0. A correction after the kernel adds it with the kernel's
  log-sum-exp: with p0 = exp(s_i0 − lse_i), o' = (o + (e^u − 1)·p0·v0) / (1 + (e^u − 1)·p0). This is exact.
- Arms: configs differ from `ar/full_runs/F107b_e5/config.json` only in `hisa`, `hisa_angle_deg` and `rope_theta`
  (checked with a dict diff). Seed 1337, block_size 1024, 5100 steps, 8x H200, torchrun.

## Code (commit 18f26c33 on 187cb7de)

- `bla_gpt/bla_gpt.py`: `GPTConfig.hisa` (False) and `hisa_angle_deg` (90.0). +4 lines.
- `bla_gpt/attentions.py` (+81): `hisa_thresholds`, `hisa_rho`, `hisa_reference` (readable eq. 3), `hisa_attention`
  (kernel + sink correction); hooks in `Attention._flash_attention` and `GOATSinkAttention._flash_attention`
  (XSA exclusion still runs after HiSA). Flag off: no buffer, no new code path.
- `bla_gpt/hisa_triton.py` (+550): exact-rule forward/backward Triton kernels, custom op.
- `tests/test_hisa_f110.py` (+197). `ar/full_train/configs/F110{A,B,C}.json` + `_smoke.json`.
- Run records: commits 3d796e71 (F110A), 47fb38a3 (F110B), 62a614df (F110C) from `ar/record_run.py`
  (`ar/full_results.tsv` rows + `ar/full_runs/F110*`). One more commit adds the README row.
  Note: the tsv code-commit column for F110B/F110C shows the previous record commit, as `record_run.py` writes HEAD;
  all three arms ran the code of 18f26c33.
- `ar/best_config.json` is not changed.

## Tests on the pod (H200, torch 2.8.0+cu129, triton 3.4.0)

`pytest tests/test_hisa_f110.py`: **11 passed in 414.63 s, PYTEST_EXIT=0** (log `/data/tts/erogol/blagpt_f110_20261123/test.log`).

- `test_flag_off_bit_identical_to_base`: same state_dict, logits and loss as a `git archive` of 187cb7de.
- `test_phi_inf_equals_dense`: Φ = ∞ gives dense attention.
- `test_reference_and_kernel_vs_naive_loop` (base 1e6 and 1e4, with and without sink): reference and kernel vs a naive (i, j, l) loop.
- `test_kernel_grads_vs_reference`, `test_rho_matches_eq5` (2 bases, vs a brute-force sum), `test_thresholds_exact`,
  `test_hisa_model_compiled_bf16_train_step` (compiled bf16 forward + backward, finite).

Smoke runs (20 steps, 8 GPU, compile + DDP): A 8.1459, B 8.1459, C 8.1705 val_loss, all finite.

## Files

- Vast: `/data/tts/erogol/blagpt_f110_20261123/` (full_runs/F110A,B,C, smokes, queue.log, test.log, versions.txt, f110.bundle md5 3ace3503…).
- Local: `f110/F110{A,B,C}_run.log`, `f110/F107b_e5_run.log` (copy of `/data/tts/erogol/blagpt_f107b_e5_20261120/full_runs/F107b_e5/run.log`),
  `tables.py`, pod scripts. Checkpoints were deleted after each run.

## Open decisions for the project owner

1. Seed 2 of F110A (the rule says confirm before a new crown).
2. One dense base-1e6 run at 187cb7de (same seed), to separate the HiSA effect from the speed-option change.
