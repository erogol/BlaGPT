# BlaGPT Leaderboard — FineWeb 10B, 5100 steps, 8×H100/H200

| Rank | Run | Val loss | Config changes vs B0 | Seed | Status |
|---|---|---|---|---|---|
| 1 | E7b | 3.1106 | rotary + RoVE | 1337 | keep (best) |
| 2 | Combined | 3.1113 | rotary + RoVE + MHAR H8 | 1337 | discard (+0.0007 vs E7b) |
| 3 | E7a | 3.1134 | rotary | 1337 | keep |
| 4 | E6b_s2 | 3.1203 | MHAR H8 | 2 | keep |
| 5 | E6b | 3.1238 | MHAR H8 | 1337 | keep (borderline) |
| 6 | E7b_s2 | 3.1240 | rotary + RoVE | 2 | keep_pending_confirmation |
| 7 | B0s2 | 3.1241 | baseline | 2 | baseline seed 2 |
| 8 | E7a_s2 | 3.1250 | rotary | 2 | discard (no gain on seed 2) |
| 9 | E7c | 3.1262 | GRAPE + RoVE | 1337 | discard (no gain) |
| 10 | B0 | 3.1265 | baseline | 1337 | baseline |

## Best model

- **Run:** E7b (rotary + RoVE)
- **Validation loss:** 3.1106 at step 5100
- **Seed:** 1337
- **Config:** `ar/best_config.json`
- **Checkpoint:** `bla_gpt/logs/ar_full_E7b_0/state_step005100.pt`
- **Step time:** 1018 ms (seed-1337 run, 8×H100)

## Notes

- Rotary and RoVE are the winning techniques. MHAR H8 adds cost without benefit on top of rotary + RoVE.
- E7a rotary showed a large gain on seed 1337 (−0.0131) but no gain on seed 2 (+0.0009). The E7b combination is more stable.
- Two seeds do not establish statistical significance. All gains below 0.003 are tentative.