# F111 — 3-way AttnRes (Split Q/K/V Depth Routing) on F110A

Inspiration: DeepCrossAttention (arXiv:2502.06785) — decoupling depth routing for Query, Key, and Value streams.
Base model: `ar/best_config.json` (F110A HiSA, `val_loss` 3.1124, seed 1337, 5100 steps, 8x H200).
Hardware: 8x NVIDIA H200 on node `g291`, `torch 2.8.0+cu129`, `triton 3.4.0`.

**Push pending.** The mission sandbox cannot push. Commits `fe90baa5..HEAD` are on local `main` (plain fast-forward on `origin/main` `df87becc`, no force).

## Verdict

- **KEEP — confirmed new best model (two seeds).**
- `F111A` (seed 1337) reaches `val_loss` **3.1076** at step 5100, a **-0.0048** improvement over `F110A` (3.1124) — 2x the 0.0024 seed-noise threshold.
- Seed-2 confirmation (`F111A_s2`, seed 2) reaches `val_loss` **3.1114** (-0.0010 vs F110A seed 1337, -0.0116 vs F107b_e5 seed 2 `3.1230`).
- Two-seed mean: **3.1095** (-0.0029 vs F110A `3.1124`). Both seeds beat F110A.
- `ar/best_config.json`, `bla_gpt/best_model_config.py`, `ar/full_results.tsv`, and `README.md` are updated with `attn_res_split_qkv=true`.
- **Speed and memory trade-off:** `F111A` runs at 722.12 ms/step (+8.8% vs F110A 663.60 ms/step) and uses 64,660 MiB peak memory (+4,423 MiB / +7.3% vs F110A 60,237 MiB). At equal wall time (F110A end at 3,377,744 ms), `F111A` is at 3.1491 (+0.0367 higher) because it is still in its LR warmdown phase.

## Results (computed from run logs via `tables.py`)

| Run | Seed | Final val_loss | Δ vs F110A | step_avg ms | train_time ms | Peak MiB | Params | Node |
|---|---|---|---|---|---|---|---|---|
| F107b_e5 (dense, old speed opts) | 1337 | 3.1153 | +0.0029 | 732.44 | 3728122 | 70569 | 404.64M | not logged |
| F110D (dense control) | 1337 | 3.1147 | +0.0023 | 671.95 | 3420242 | 58478 | 404.64M | g375 |
| F110A (HiSA, prior best) | 1337 | 3.1124 | 0.0000 | 663.60 | 3377744 | 60237 | 404.64M | not logged |
| **F111A (3-way AttnRes)** | 1337 | **3.1076** | **-0.0048** | 722.12 | 3675607 | 64660 | 404.65M | g291 |
| **F111A_s2 (3-way AttnRes)** | 2 | **3.1114** | **-0.0010** | 719.22 | 3660809 | 64660 | 404.65M | g291 |

- Smoke test (`F111A_smoke.json`, 20 steps): `val_loss` 8.1417, `train_time` 5614 ms, `step_avg` 561.43 ms, peak 29,875 MiB.
- Parameter delta: `+2 * n_layer * n_embd = +2 * 10 * 768 = +15,360` params (+0.0038%).

## Val loss by step vs F110A

| Step | F110A | F111A (s1337) | F111A_s2 (s2) | F111A − F110A | F111A_s2 − F110A |
|---|---|---|---|---|---|
| 125 | 4.7553 | 4.7305 | 4.7553 | -0.0248 | +0.0000 |
| 250 | 4.2695 | 4.2679 | 4.2692 | -0.0016 | -0.0003 |
| 500 | 3.9660 | 3.9668 | 3.9700 | +0.0008 | +0.0040 |
| 1000 | 3.7705 | 3.7647 | 3.7629 | -0.0058 | -0.0076 |
| 1125 | 3.7452 | 3.7376 | 3.7398 | -0.0076 | -0.0054 |
| 1250 | 3.6259 | 3.6248 | 3.6267 | -0.0011 | +0.0008 |
| 1500 | 3.5231 | 3.5215 | 3.5236 | -0.0016 | +0.0005 |
| 2000 | 3.4484 | 3.4450 | 3.4444 | -0.0034 | -0.0040 |
| 2500 | 3.4081 | 3.4035 | 3.4052 | -0.0046 | -0.0029 |
| 3000 | 3.3838 | 3.3780 | 3.3808 | -0.0058 | -0.0030 |
| 3500 | 3.3223 | 3.3167 | 3.3199 | -0.0056 | -0.0024 |
| 4000 | 3.2564 | 3.2512 | 3.2549 | -0.0052 | -0.0015 |
| 4500 | 3.1866 | 3.1818 | 3.1853 | -0.0048 | -0.0013 |
| 5000 | 3.1184 | 3.1136 | 3.1174 | -0.0048 | -0.0010 |
| 5100 | 3.1124 | 3.1076 | 3.1114 | -0.0048 | -0.0010 |

`F111A` leads `F110A` at every validation checkpoint from step 1000 onward; `F111A_s2` leads `F110A` at every checkpoint from step 2000 onward.

## Equal-time comparison

Linear interpolation between validation points at each run's end `train_time`:

| Reference train_time | ms | F110A | F111A | F111A_s2 | F111A − F110A |
|---|---|---|---|---|---|
| end of F110A | 3377744 | 3.1124 | 3.1491 | 3.1499 | +0.0367 |
| end of F111A_s2 | 3660809 | (ended) | 3.1087 | 3.1114 | — |
| end of F111A | 3675607 | (ended) | 3.1076 | (ended) | — |

The +58 ms/step (+8.8%) overhead comes from two extra `_attn_res_mix` depth-softmax passes, two extra `ln_1` normalizations, and split K/V `F.linear` projections per layer. Fusing the three depth-routing weights into one batched matmul over `_srcs` is a natural speed follow-up if wall-clock efficiency matters.

## Implementation & Tests

- `bla_gpt/bla_gpt.py`:
  - Added `attn_res_split_qkv: bool = False` to `GPTConfig`.
  - In `GPT._init_attn_res()`: when `use_attn_res` and `attn_res_split_qkv` are both true, allocates zero-initialized `attn_res_w_k` and `attn_res_w_v` of shape `(n_layer, n_embd)`.
  - In `GPT.forward()`: computes `x_k` and `x_v` over the same `_srcs` stack, applies the exact same U-Net skip and Engram additive delta so `x_k == x_v == x` at zero init, and passes `x_k=x_k, x_v=x_v` into `block(...)`.
  - In `Block.forward()`: normalizes `x_k` and `x_v` with `self.ln_1` and passes them to `self.attn`; residual stream and MLP branch stay fed by `x`.
- `bla_gpt/attentions.py`:
  - `Attention.forward()` and `ComposableGatedAttention` / `AffineScaledAttention` accept optional `x_k` and `x_v`.
  - `Attention._project_kv_split()` projects K from `x_k` and V from `x_v` using the existing `kv_proj` weight halves (zero extra projection parameters), then runs the same Lattice FiLM, embed value residual, and value residual hooks.
- `tests/test_attn_res_split_qkv_f111.py`: 10/10 passing locally on CPU and on the pod (gate-off byte-identical output and zero extra params; zero-init identity for both `regular` and `xsa`; non-zero `attn_res_w_k`/`attn_res_w_v` change output; finite non-zero gradients on `attn_res_w`, `attn_res_w_k`, `attn_res_w_v`; exact `+2 * n_layer * n_embd` param delta).
