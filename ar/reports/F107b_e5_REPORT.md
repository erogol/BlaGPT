# F107b_e5: F107b with Engram x5 (engram_vocab_mult 20 -> 5), two seeds

Date: 2026-11-20. Pod erogol-f107be5-20261120 (node g272, 8x H200, tts-queue, train-high).
Branch f107b-e5 (base origin/main a0d9d3ee). Config commit d9e5378c (pod: d8fd823, same content via git am).
Configs F107b_e5.json / F107b_e5_s2.json / F107b_e5_smoke.json are exact copies of the F107b configs;
a script check found one changed key only: engram_vocab_mult 20 -> 5. No lattice_scan_impl / lattice_film_impl.

Question: how much of the F107b result comes from the large Engram table?

## Short answer

- x5 is worse than x20 at equal steps on both seeds: +0.0086 (seed 1337) and +0.0195 (seed 2).
- Both gaps are larger than the seed spread (x5 0.0077, x20 0.0032). Pre-set rule: **"x20 needed"** (step-matched).
- At equal train time, x5 wins by a large margin: at x5's end (~3.6-3.7 ks), x20 is still at ~3.33-3.34 val.
- x5 costs 404.64M params (vs 984.14M), 707 ms/step (vs ~1031-1044), 70569 MiB peak (vs 94160).
- x5 still beats E1c (3.1487, last x5 best) by 0.026-0.033, so Lattice/RoPE/RoVE gains hold with x5.

## Smoke

20 steps, step:20/20 val_loss 8.1610 (finite), 404.64M params, peak 32070 MiB. SMOKE_OK 12:16:04Z.

## Final val

| Run | Final val | vs F107b (x20) | vs E7b | vs E1c 3.1487 |
|---|---|---|---|---|
| F107b_e5 (seed 1337) | 3.1153 | +0.0086 (F107b 3.1067) | +0.0040 (E7b 3.1113) | -0.0334 |
| F107b_e5_s2 (seed 2) | 3.1230 | +0.0195 (F107b_s2 3.1035) | -0.0010 (E7b_s2 3.1240) | -0.0257 |

record_run.py rows (branch f107b-e5): F107b_e5 3.1153 discard, F107b_e5_s2 3.1230 discard.

## Step-matched val (x5 - x20)

| Step | F107b_e5 | F107b | diff | F107b_e5_s2 | F107b_s2 | diff |
|---|---|---|---|---|---|---|
| 125 | 4.8352 | 4.8720 | -0.0368 | 4.8567 | 4.8532 | +0.0035 |
| 250 | 4.3953 | 4.3972 | -0.0019 | 4.4603 | 4.4186 | +0.0417 |
| 500 | 4.1647 | 4.1790 | -0.0143 | 4.2775 | 4.1736 | +0.1039 |
| 1000 | 4.0701 | 4.1050 | -0.0349 | 4.1879 | 4.0491 | +0.1388 |
| 1500 | 3.5276 | 3.5229 | +0.0047 | 3.5346 | 3.5182 | +0.0164 |
| 2000 | 3.4508 | 3.4455 | +0.0053 | 3.4566 | 3.4383 | +0.0183 |
| 2500 | 3.4080 | 3.4012 | +0.0068 | 3.4155 | 3.3990 | +0.0165 |
| 3000 | 3.3854 | 3.3766 | +0.0088 | 3.3935 | 3.3846 | +0.0089 |
| 3500 | 3.3242 | 3.3153 | +0.0089 | 3.3320 | 3.3131 | +0.0189 |
| 4000 | 3.2592 | 3.2498 | +0.0094 | 3.2662 | 3.2469 | +0.0193 |
| 4500 | 3.1893 | 3.1804 | +0.0089 | 3.1966 | 3.1774 | +0.0192 |
| 5000 | 3.1214 | 3.1127 | +0.0087 | 3.1290 | 3.1094 | +0.0196 |
| 5100 | 3.1153 | 3.1067 | +0.0086 | 3.1230 | 3.1035 | +0.0195 |

After the step 1200 switch the gap is stable: ~+0.009 (seed 1337) and ~+0.019 (seed 2) from step 3500 on.
Seed 2 x5 was much worse before the switch (+0.10 to +0.14 at steps 500-1000), then recovered most of it.

## Time-matched (train_time column, linear interpolation between val points)

- F107b_e5 ends at 3728 s with 3.1153. F107b at 3728 s: 3.3441. x5 - x20 = -0.2288.
- F107b_e5_s2 ends at 3607 s with 3.1230. F107b_s2 at 3607 s: 3.3282. x5 - x20 = -0.2052.
- F107b ends at 5853 s (F107b_s2 at 5579 s). x5 has no point there: it ended 1.57x / 1.55x earlier.
  The val of x5 at x20's final time is not measured. A time-matched x5 run (~8000+ steps, new LR schedule) would be needed.
- Caution: the interpolated x20 values are mid-schedule (LR not yet decayed), so this view favours the shorter run.
  It shows that x5 reaches a good final val in 64% of the time; it does not show x5 beats x20 at equal compute.

## Cost

| Run | Params (log) | Median ms/step, steps >1201 | Peak memory MiB | Final train_time s |
|---|---|---|---|---|
| F107b_e5 | 404.64M | 707 | 70569 | 3728 |
| F107b_e5_s2 | 404.64M | 707 | 70569 | 3607 |
| F107b | 984.14M | 1044 | 94160 | 5853 |
| F107b_s2 | 984.14M | 1031 | 94160 | 5579 |

- Median ms/step: train_time deltas of steps >1201, steps after a val step and step 1201 excluded.
- Params: 404.64M matches the parent CPU count (~404M). The x20 log shows 984.14M (goal text said 959M).
- Optimizer share: not measured (no profiler in these runs; not cheap to get).
- Disk: checkpoints (state_step005100.pt, best_model_5100.pt) deleted after each run; /nvme 20G after each.

## Seed spread

- x5: |3.1153 - 3.1230| = 0.0077
- x20: |3.1067 - 3.1035| = 0.0032
- x5 - x20: +0.0086 (seed 1337), +0.0195 (seed 2); mean +0.0141.

## Verdict (stated, not applied)

- Step-matched rule: x5 is worse than F107b by more than the seed spread (0.0077, the larger one) on both seeds -> **"x20 needed"**.
- Time-matched view: x5 finishes 5100 steps in ~64% of x20's time and ends far below x20's val at that time.
  The equal-compute question (x5 trained for x20's wall time) stays open.
- Note: only 2 seeds per arm; the seed-2 gap (0.0195) is twice the seed-1337 gap (0.0086).

## Files

- Run dirs: /data/tts/erogol/blagpt_f107b_e5_20261120/full_runs/{F107b_e5,F107b_e5_s2}, smokes/, f107b_e5.bundle.
- Local: f107b-e5/pod_out/ (step/val lines of all 4 runs, queue.log, setup.log, smoke log, ledger rows, bundle),
  analyze.py (all numbers above), analysis.md.
- Branch f107b-e5 commits: d9e5378c configs, 09276fd2 record F107b_e5, 62716816 record F107b_e5_s2, plus this report.
  Not pushed. README.md not edited.
