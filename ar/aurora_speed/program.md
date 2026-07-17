# Aurora-Speed Autoresearch Track (S-series) — v1.0

## Goal
Reduce Aurora optimizer wall-clock cost at strict val-loss parity. "Faster" = lower
optimizer-step time (and thus step_avg) with final_val_loss within EPSILON of the
parent run. This is a SPEED track: val-loss improvements are welcome but not the target.

## Ground rules (inherited from ar/ program v2.1)
- One GPU run at a time. NEVER launch while the F100/F101 queue (tmux ar_queue) or any
  train.py is running — check `pgrep -f train.py` first.
- Ledger: ar/aurora_speed/results.tsv — every experiment gets a row, keeps AND discards.
- Commit code + ledger after every experiment. Push milestones via bundle relay to
  GitHub autoresearch/jul17.
- Discard discipline: `git checkout <best_commit> -- bla_gpt/` (no reset --hard).

## Protocol per candidate (two gates)
1. **Microbench gate** (cheap, minutes): `python ar/aurora_speed/bench_optimizer.py`
   times optimizer.step() in isolation (100 steps, synthetic grads, real param shapes
   from the best config, CUDA events, warmup 10). Candidate must show >=10% reduction
   in optimizer step time vs baseline Aurora. If not -> discard, record row, next.
2. **Parity gate** (full run, ~66 min): standard 5100-step full run via
   `bash ar/run_full_experiment.sh S<N> <config>`. Accept iff:
   - final_val_loss <= parent + 0.0030 (parity epsilon), AND
   - step_avg improves vs parent run on the same stack.
   Record both numbers in the ledger row.

## Baseline first (S0)
Before any candidate: run bench_optimizer.py on current Aurora (rms_match config from
F100 sweep winner) AND Muon, to measure the optimizer's absolute share of the 720ms
step. Record in ledger as S0. If Aurora share < 5% of step time, notify Eren with the
numbers and pause the track for re-scoping (speed work would be better spent elsewhere).

## Candidate queue (ranked; from the 2026-07-17 brainstorm, grounded in arXiv:2606.27715)
- S1 batched-bmm: stack same-shape tall matrices (10 layers x up/gate 3072x768) into one
  batched Newton-Schulz (torch.bmm). Also batch attention matrices per shape group.
- S2 gram-space NS: run inverse-sqrt iterations on the n x n (768^2) Gram matrix
  M^T M, then form M (M^T M)^{-1/2} — ~4x fewer FLOPs/iter at 4:1 aspect ratio.
- S3 warm-start NS: cache previous polar factor per param, refine with 2-4 NS steps
  instead of full iteration count (momentum beta=0.95 -> polar drifts slowly).
- S4 adaptive pp_iterations: compute row-norm CV of the polar factor (one reduction);
  skip row-normalize->reorthogonalize passes when CV < 0.05 (paper Fig 7: momentum
  converges to uniform rows, pp passes become no-ops late in training).
- S5 fused pp (Dykstra-style): interleave row normalization INSIDE the NS recurrence
  (alternating projection onto orthogonal ∩ uniform-row-norm manifolds) instead of
  (full NS -> rownorm) x pp_iterations. Research-flavored; needs parity watch.
- S6 CUDA-graph capture of the whole optimizer step (launch-bound small matmuls at 768 dim).
- S7 DDP sharding: each rank orthogonalizes 1/8 of params, all_gather updates
  (modded-nanoGPT Muon pattern) — only if optimizer share is large after S1-S6.

## Chain rule
If a candidate keeps, it becomes the new baseline for subsequent candidates (stack wins).
Map interaction only when two kept candidates touch the same code path (e.g. S1+S2).

## Ledger format (TSV)
id  commit  parent  opt_ms_before  opt_ms_after  step_avg  final_val_loss  verdict  slug  note
