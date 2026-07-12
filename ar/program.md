# BlaGPT autoresearch — program.md (v3 — FULL-TRAINING RESEARCH)

## Coordination note (2026-07-12, from a sibling research agent -- READ ONCE, then keep)

Two parallel missions were briefly active on this pod. The sibling has STOOD
DOWN after: recording F78 (PoPE, discard 3.2299, commit 1046299) and F79 (GOAT
relative spectral prior, discard 3.2257, commits ff3e595/04feba1). Best remains
F77 = 3.2230. Handoff notes for the active mission:
- ar/record_run.py (committed) appends the ledger row + commits from a run.log;
  it is dedup-guarded. CHAIN IT after run_full_experiment.sh in every tmux launch
  so no completed run is left unrecorded. An insurance recorder for F80 is already
  running in tmux ar_record_F80 (harmless if you record first).
- F80 NAG readout pitfall: exp(s_L) must actually reach the logits -- RMSNorm-family
  ln_f erases scale. Verify the implementation. NAG also attenuates attention sinks,
  possibly redundant with the kept GOAT sink prior (F74).
- Lipschitz training (2507.13338): deprioritize -- the paper itself matches baseline
  accuracy only at a vacuous 1e264 Lipschitz bound; it is a robustness method, not a
  val-loss improver.
- Branch pushed to GitHub through cf64a6a (bundle relay, procedure
  an external relay host). Push after every keep and
  periodically regardless -- pod TTL expires ~Jul 13-15.
- Protocol: never two GPU experiments concurrently.

## v3 protocol — FULL TRAINING ONLY (2026-07-12)

This section overrides every conflicting 600-second/time-budget rule below.

- Every clean candidate is trained with normal `bla_gpt/train.py` for exactly
  **5100 iterations** using 8 GPUs, global batch 512, device batch 32, sequence
  length 1024, fixed validation data/tokens, and the normal LR schedule.
- Baseline/crowned parent: combined-keeps full run, final val_loss **3.2354**,
  checkpoint `bla_gpt/logs/combined_keeps_full_5100_0/state_step005100.pt`.
  Experiments still start from RANDOM INIT; this checkpoint is comparison only.
- Launch: `/nvme/BlaGPT/ar/run_full_experiment.sh <id> <config.json>` inside a
  durable tmux session. Artifacts live in `ar/full_runs/<id>/`.
- Keep iff final step-5100 validation loss is lower than the current full-run
  best. Improvements below 0.003 require one full confirmation rerun. There is
  no reuse of the old 600-second EPSILON.
- Debugging may use at most a 60-second `train_ar.py` smoke run, but smoke
  metrics NEVER enter the ledger and NEVER decide keep/discard.
- Checkpointing stays enabled. Preserve the final/best checkpoint path in the
  full-run ledger. Never run two GPU experiments concurrently.
- Use `ar/full_results.tsv` as the canonical full-training ledger; do not mix
  these results with `ar/results.tsv` (the old proxy ledger).
- Paper implementations must be minimal and native to repo structure: smallest
  viable diff, config-gated default-off, existing registries/classes, no broad
  refactors. Read the actual paper before implementation.
- Continue the Notion BlaGPT paper queue in `ideas.md`; prioritize mechanisms
  with a real full-horizon transfer story. No schedule-budget gaming.

## Harness location (moved into repo 2026-07-10)

The harness lives IN-REPO at `/nvme/BlaGPT/ar/` (`/nvme/ar` is a symlink to
it, all old paths still work). Every experiment commit MUST include the
updated `ar/results.tsv`, `ar/ideas.md`, and `ar/runs/<n>/` so the full
tried-ideas history is durable in git and survives pod death. Never commit
`*.bundle` files (gitignored).

## Mission

Do not merely enumerate configurations — INVENT. The goal is to discover
mechanisms that are not in this repo yet. A night that produces one genuinely
new mechanism that works is worth more than twenty knob tweaks. Sources of
invention: recent papers, mechanisms from other fields (signal processing,
memory systems, optimization theory), hybrids of existing techniques, and
original ideas. Implementing something that might not work is the job, not
a risk.

Run tag: `jul8` · branch `autoresearch/jul8` · an 8× H100 pod
Spec: maintained off-repo (v0.2)
Metric: `final_val_loss` from `/nvme/ar/runs/<n>/run.log` (lower is better).
Budget: AR_TIME_BUDGET=600 s pure training time (compile + val excluded).
Current best: 3.4047 (exp 35, warmup100 + curriculum 1200@384 + 11 layers + no PLTE + rope_theta 1e6 + MHA, commit 52df558). EPSILON = 0.0064.

## Editable surface — everything is fair game

You may edit ANY code in `bla_gpt/`: model architecture, attention variants,
optimizers, LR schedules (including the time-based `ar_lr_mult` default),
initialization, normalization, training-loop logic inside `train_ar.py`,
batch/microbatch strategy, new modules and new files. Rewriting whole
components is allowed and encouraged. Config JSONs remain supported but are
no longer the only lever.

## FROZEN — touching any of these invalidates the run

1. Time-budget machinery in `train_ar.py`: AR_TIME_BUDGET accounting
   (clock starts after step 10, val time excluded), the all-rank stop
   allreduce, and the NaN fast-fail exit.
2. Final validation + summary block: fixed val shard
   (`fineweb_val_000000.bin`), val_tokens=10485760, GPT-2 tokenization,
   and the exact greppable summary print format (`final_val_loss:` etc.).
3. `/nvme/ar/run_experiment.sh`, `/nvme/ar/results.tsv` history
   (append-only), and past `runs/<n>/` directories.
4. Fixed data-order seed and the train shard list (model init seed is free;
   init noise is handled by EPSILON).
5. RANDOM INIT ONLY: loading pretrained weights, checkpoints from previous
   runs, pretrained embeddings, or any downloaded artifact is CHEATING
   (torch.load / from_pretrained / hf_hub / safetensors are banned in
   experiment diffs; frozen_check greps for them).
6. `val_tokens=10485760` — fixed in code AND may not be overridden via
   config.json.
7. CLOCK INTEGRITY: mid-training validation windows are excluded from the
   time budget; NO model-updating compute may run while the clock is
   stopped. Any diff touching `training_time_ms`/`t0` accounting is flagged
   by frozen_check and gets mandatory driver review. Recorded
   training_seconds must land within ~±5% of AR_TIME_BUDGET.
   Note: growing/shrinking the MODEL is not cheating — the wall-clock
   budget makes size a legitimate tradeoff (bigger = fewer tokens).

Guard: after each experiment, run `/nvme/ar/frozen_check.sh` BEFORE
recording the result — a violation means the run is void regardless of
val_loss. The driver independently reviews `git diff` (editor ≠ grader).
Any improvement > 5×EPSILON is bug-until-proven: inspect the diff for eval
leakage before keeping.

## Loop rules

- One idea per experiment. Commit each experiment on the branch
  (`exp N: slug`). Save `runs/<n>/diff.patch` (git diff vs best commit) —
  required now that code changes, it is the reproducibility record.
- Keep if `final_val_loss < best − EPSILON`. Borderline (within ±EPSILON):
  status `rerun?`, rerun once, keep only if both runs beat best.
- Discard: `git checkout <best_commit> -- bla_gpt/` — NEVER `git reset --hard`:
  the ledger (`ar/results.tsv`, `ar/ideas.md`, `ar/runs/`) is tracked in the
  repo now and uncommitted rows must survive a discard.
- **Research chains (invention support):** a novel-mechanism idea is NOT
  judged by its first run. It gets a chain of up to 4 runs (debug → tune →
  tune) before the keep/discard verdict; only the chain's final result
  enters results.tsv as the idea's verdict (intermediate rows logged with
  status `chain`). A chain may continue past a worse-than-best result if
  the trajectory across runs is improving. Chains are for NEW mechanisms
  only — config mutations of existing techniques stay one-shot.
- **Debugging ≠ experimenting:** crashes during implementation of new code
  are debug iterations, not discards — up to 3 debug cycles allowed, each
  verified with a 60s smoke run (AR_TIME_BUDGET=60) before spending a real
  600s run. Only clean (non-crash) results count toward family statistics.
  OOM → halve device batch + grad-accum compensate; NaN → 0.5× LR once.
- Scheduling: **~30% exploit** (mutate current best), **~40% invent**
  (implement mechanisms that do NOT yet exist in the repo — from papers,
  cross-domain analogies, or original design), **~30% explore** (untried
  `techniques/`, near-miss variants: anything that lost by <2×EPSILON gets
  a variants slot). Family ban only after 5 consecutive CLEAN discards.
- **Idea generation discipline:** before each invent-slot experiment, write
  3 candidate ideas to `ideas.md`, each with its core mechanism named in
  ONE WORD; if two share the mechanism word, replace one. Pick the one with
  the best expected-value×novelty. No idea may be a pure re-parameterization
  of a previous experiment.
- Driver review is LIMITED to: frozen-zone integrity (frozen_check.sh) and
  eval-leakage inspection on >5×EPSILON jumps. The driver must NOT veto or
  discourage ideas for being unusual, ambitious, or unlikely to work.
- Log every run: results.tsv row + runs/<n>/{diff.patch,config.json,run.log}.
- Report every iteration to the owner (change + result + decision).
- NEVER STOP until manually interrupted.

## Launch

/nvme/ar/run_experiment.sh <n> best [/nvme/ar/runs/<n>/config.json]

## Keep classification (added 2026-07-10, per owner)
- Every KEEP must be classified in its results.tsv note: [class=TRANSFERABLE] (plausibly helps at real horizons: architecture, capacity/step trades, data ordering) or [class=BUDGET-ARTIFACT] (only exploits the fixed 600s budget: warmup/warmdown shape, schedule-integral tweaks).
- Budget-artifact keeps update the running best (the metric is the metric) but do NOT count as invent-slot wins; their families get at most one mapping experiment, then close. No further LR-schedule-integral mining.
- Invent slots must have a transfer story.
- Final report: best_config deltas listed in the two classes separately.
