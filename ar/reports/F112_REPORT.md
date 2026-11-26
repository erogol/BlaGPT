# F112: LoomFormer (arXiv:2611.30634) Evaluation on BlaGPT

**Verdict:** Discard (`❌`) — `F112A` finished cleanly at `val_loss = 3.1124` (seed 1337), which is `+0.0048` behind current best `F111A` (`3.1076`, seed-2 `3.1114`, mean `3.1095`). `best_config.json` and `best_model_config.py` are unchanged.

---

## 1. Architecture & Implementation

Implemented LoomFormer (*"Rewirable Circuits in Transformers with Loom Attention"*, Xiao et al. 2026, arXiv:2611.30634, `Caiyun-AI/LoomFormer`) in `bla_gpt/loom.py`, wired into `bla_gpt/bla_gpt.py`, `bla_gpt/mlps.py`, and `bla_gpt/optimizers/__init__.py` behind `use_loom: bool = False`:

- **Dual residual stream:** Vector stream `x` (`D = 768`) + per-token matrix memory `M` (`d_k x d = 24 x 64 = 2D`, with `d_r = 144`, `rope_dim = 16`).
- **Components:** `LoomEmbedWrite` (initializes `M_0`), `MShift` (per-slot shift gate), whole-matrix `MNorm`, 4-arm `MRead` (`q`, `k`, `v`, `o` bypass), `LoomMHA` (strictly causal `s < t` attention with `Q/K` head dim `64 + 16 = 80`, `V` head dim `64`, and local bypass `o_m`), `MWrite`, and `FFN_M` write branch on layers `0..8` (layer `9` omits unused writes to `M_L`).
- **Fast strictly causal unequal-width attention:** `strict_causal_attention` shifts `(q[:, 1:], k[:, :-1], v[:, :-1])` and pads `v` from `64` to `80` with zeros, then calls `F.scaled_dot_product_attention(..., is_causal=True, scale=1/sqrt(64))` and slices back to `64` with zero at `t = 0`. This avoids an `O(T^2)` mask, avoids empty-row `NaN` at `t = 0`, and hits FlashAttention/cuDNN directly on H200.
- **Appendix D.1 initialization:** Gate projection weights initialized to `0`; biases set initial gate openings to `0.05` (dynamic read), `0.9` / effective `1.0` (static read), `0.1` (write), and `0.5` (`MShift`); `q_bar_v ~ N(0, 1/sqrt(d_k))`, `q_bar_{q,k,o} = 0`; `W_r2_v` cloned from `W_r2_o`.
- **Optimizer routing (`loom_adamw = True`):** Main projection matrices (`W_e`, `W_q`, `W_k`, `W_o`, `ffn_x`, `ffn_m`) stay on Aurora (`lr = 0.03`, `wd = 0.025`). Small port/gate parameters (`W_sg`, `W_r1`, `W_r2_*`, `W_rg_*`, `W_rg_bar_*`, `q_bar_*`, `c_qk`, `c_vo`, `W_w1/w2/wg`, `ffn_W_w1/w2/wg`) use AdamW (`loom_lr = 1e-3`, `loom_wd = 0.1` on 2D weights, `0.0` on biases/scales/`q_bar`).
- **Parameter budget match:** `FFN_x = 6784`, `FFN_M = 3392` (exact `2:1` split on layers `0..8`), `FFN_x(last) = 10112`.
  - `F112A`: `404,786,697` total params (`211,617,800` non-Engram + `193,168,897` Engram x5)
  - `F111A`: `404,652,049` total params (`211,483,152` non-Engram + `193,168,897` Engram x5)
  - Non-Engram delta: `+0.06%`.

### Unit Tests (`tests/test_loom_f112.py`)
All 7 F112 unit tests (plus 11 F111 regression tests = 18/18 locally on CPU, and 7/7 on the 8x H200 pod) passed:
- Gate-off identity (`use_loom=False`, 0 extra params, bit-identical outputs)
- `strict_causal_attention` exact match against explicit `s < t` reference loop (including `t=0` zeros and `T=1`)
- End-to-end strict causality (`loom_m_shift` True and False) + `o_m` local bypass check at `t = 0`
- Appendix D.1 near-zero gate initialization invariants
- Finite non-zero gradients and Aurora vs AdamW parameter group routing
- Exact `F112A.json` vs `F111A.json` non-Engram parameter budget match (`< 0.5%`)

---

## 2. GPU Results (8x NVIDIA H200)

Environment: `torch 2.8.0+cu129`, `triton 3.4.0`, 8x NVIDIA H200, FineWeb10B (`104` shards).

### Smoke Test (`F112A_smoke`, 20 steps)
- `step:20/20 val_loss: 9.2493`
- Steady step time at `T = 384`: `~519 ms/step`
- Peak GPU memory: `28,536 MiB`
- Zero NaNs, zero OOMs.

### Full 5100-Step Run (`F112A`, seed 1337)
Config: LoomFormer (`use_loom=True`) on BlaGPT's Aurora + Engram x5 + Primer MLP + GatedNorm + HybridNorm base, with `use_attn_res=False`, `use_value_residual=False`, `use_unet_skips=False`, `lattice_enabled=False`, `rove=False`, `hisa=False`, `use_composable_gated_attn=False`, `use_goat_sink_prior=False`.

| Run | Architecture | Params (Non-Engram) | Seed | Val Loss | Delta vs F111A | Step Avg (ms) | Peak Mem (MiB) | Status |
|---|---|---|---|---:|---:|---:|---:|---|
| **F111A** | 3-way AttnRes + Lattice + XSA/GOAT/Gated + RoVE + HiSA + U-Net + VRes | 404.65M (211.48M) | 1337 | **3.1076** | — | 722.12 (703.30 triton) | 64,660 (70,560 triton) | Keep (best) |
| **F111A_s2** | F111A seed 2 | 404.65M (211.48M) | 2 | **3.1114** | +0.0038 | 719.22 | 64,660 | Keep (mean 3.1095) |
| **F110A** | F107b_e5 + HiSA (pre-F111 best) | 404.64M (211.47M) | 1337 | 3.1124 | +0.0048 | 680.03 | 60,235 | Prior best |
| **F112A** | **LoomFormer (clean, no Lattice/AttnRes/XSA/RoVE/HiSA/U-Net/VRes)** | **404.79M (211.62M)** | **1337** | **3.1124** | **+0.0048** | **690.10** | **60,314** | **Discard (`❌`)** |

### Validation Loss Trajectory (`F112A`)
- Step 0: `10.8623`
- Step 125: `5.1039` (`500.03 ms/step`, `T=384`)
- Step 500: `4.3407` (`537.58 ms/step`)
- Step 1000: `4.1889` (`541.94 ms/step`)
- Step 1250: `3.6352` (`549.39 ms/step`, clean transition to `T=1024`, zero recompile stall)
- Step 2000: `3.4446` (`619.35 ms/step`)
- Step 3000: `3.3836` (`658.04 ms/step`)
- Step 4000: `3.2567` (`677.48 ms/step`)
- Step 5000: `3.1187` (`688.77 ms/step`)
- Step 5100: **`3.1124`** (`690.10 ms/step`, total `3,512.59 s` = 58.5 min, peak `60,314 MiB`)

---

## 3. Diagnostic Notes & Next-Step Observations

1. **Healthy run, not a bug:** Training was completely stable (0 NaNs, smooth loss curve, no recompile stall at the `T=384 -> T=1024` switch, faster than F111A at `690.10 ms/step`, lower memory at `60,314 MiB`).
2. **Strong standalone showing despite dropping 7 BlaGPT mechanisms:** `F112A` matched `F110A` (`3.1124`) and landed within `0.0048` of `F111A` (`3.1076`) while simultaneously removing **seven** winning BlaGPT components: Latent Lattice (`-0.033` to `-0.015` historically), 3-way AttnRes (`-0.0048`), Value Residual (`-0.0117`), U-Net skips (`-0.0032`), Composable Gated Attention (`-0.0096`), GOAT sink prior, RoVE, and HiSA (`-0.0029`).
3. **Promising follow-up (`F112B`, not run here):** Because standalone LoomFormer is already at `3.1124` and faster/leaner than F111A, combining LoomFormer with Latent Lattice (`lattice_enabled=True`) and/or tuning `loom_lr` (e.g., `2e-3` or `3e-3` under BlaGPT's 5100-step budget) and applying RoPE/HiSA on the RoPE slice could plausibly beat `F111A` (`3.1076`). A fused Triton `MRead`/`MWrite` kernel is not the bottleneck right now (`690 ms/step` is already faster than F111A's `703 ms/step` Triton run).
