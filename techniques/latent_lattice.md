# Latent Lattice

*Adds a small recurrent state to each middle layer. The state runs along time and modulates that layer's attention and MLP.*

---

## Overview

A normal Transformer has one active latent per token: the residual stream. It flows up through the layers, and the KV cache only stores past projections of it.

```text
h[t, l] = block_l(h[t, l-1])
```

The Latent Lattice adds a second axis. Each "Lattice layer" `l` keeps its own horizontal latent `r[t, l]` that carries over from token to token. The vertical stream goes up through depth, the horizontal latents go forward through time, and together they form a lattice.

```text
r[t, l] = U(r[t-1, l], h[t, l-1], h[t, 0])
h[t, l] = block_l(h[t, l-1], conditioned on r[t, l])
```

The horizontal latent is not a sequence mixer like an SSM memory. It is a companion state that changes how the layer reads memory (queries), writes memory (keys and values) and runs its MLP.

The paper reports, on 20B FineWeb-Edu tokens, a 406M Lattice model at validation perplexity 14.40 against 15.41 for a parameter-matched Transformer.

## Method

### Horizontal update

The update is a Mamba-style gated affine recurrence. Its coefficients depend only on the current layer input, so training can use a parallel scan over the sequence instead of a token loop.

```text
n      = RMSNorm(h[t, l-1])
Delta  = softplus(W_delta n + b_delta)
g      = exp(-exp(a) * Delta)          # gate in (0, 1)
c      = SiLU(W_r n + b_r)
r[t]   = g * r[t-1] + (1 - g) * c      # r[-1] = learned r_0
r_hat  = RMSNorm(r[t])
```

`a` is a learned per-channel log-decay rate, and `r_0` is a learned initial state.

**Input injection** also feeds the normed token embedding `h[t, 0]` into `Delta` and `c` through zero-initialized maps. The recurrence stays affine, so the same scan works.

### How the latent acts

FiLM on the query, key and value projections:

```text
q = (W_q x) * (1 + tanh(W_gamma_q r_hat)) + W_beta_q r_hat
k = (W_k x) * (1 + tanh(W_gamma_k r_hat)) + W_beta_k r_hat
v = (W_v x) * (1 + tanh(W_gamma_v r_hat)) + W_beta_v r_hat
```

A shift of the MLP pre-activation:

```text
pre = W_fc x + W_beta_mlp r_hat
```

All `W_gamma` and `W_beta` maps are zero-initialized, so at step 0 the model is exactly the base model.

### Layer split

The layers split into an early block (no lattice), a Lattice block, and a late block (no lattice). The paper uses E4 / L16 / D4 for 24 layers.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/lattice.py`, with small hooks in `bla_gpt/bla_gpt.py`, `bla_gpt/attentions.py` and `bla_gpt/mlps.py`:

- `lattice_scan`: fp32 Hillis-Steele log-step scan. It multiplies only gates in `[0, 1]`, so it cannot overflow (no log-cumsum). Max abs error against a sequential loop is `9.5e-7` at `T=1024`.
- `LatticeLayer`: the update, the readout and the FiLM / MLP heads for one layer.
- `GPT._init_lattice`: builds one `LatticeLayer` per Lattice layer and attaches it to that block's attention and MLP.

Configuration fields:

```python
lattice_enabled = True
lattice_layers = [3, 4, 5, 6, 7]  # empty = paper ratio, skip round(n_layer/6) layers at each end
lattice_dr = 256                  # width of the horizontal latent
lattice_inj = True                # input injection
```

BlaGPT-specific details:

1. FiLM is applied after the q/k/v projection and before q/k norm and RoPE.
2. E7b uses `Primer_MLP` (ReLU²), not SwiGLU. The MLP shift is added to the `c_fc` output before the activation.
3. With Hyper-Connections, the update reads the `A_m`-mixed branch input. One latent per layer is shared by all streams.
4. FiLM `gamma` is per channel. The paper shares one `gamma` across groups of channels.
5. Init follows Mamba: `A = exp(a)` in `[1, 16]`, `dt = softplus(b_delta)` log-uniform in `[1e-3, 1e-1]`.
6. Optimizer: 2D maps go to Aurora; `a`, `r_0`, biases and norm weights go to the AdamW group.
7. The vstream and the Jacobi refiner from the paper are not implemented (single pass, J0).

## Result

Run: `F107`, on top of E7b (RoPE + RoVE, best `3.1106`), 8x H200, 5100 steps, seed 1337.

| | E7b | Latent Lattice |
|---|---|---|
| Params | 959.35M | 979.02M (+2.1%) |
| Final val loss | 3.1113 (best 3.1106) | **3.1139** |
| Peak memory | 85258 MiB | 92059 MiB |
| Step average | 1003.14ms | 1194.26ms |
| Total train time | 5106s | 6079s |

Validation loss at the same step:

| Step | E7b | Latent Lattice | Diff |
|---|---|---|---|
| 500 | 4.1440 | 4.0410 | -0.1030 |
| 1000 | 4.0615 | 3.8726 | -0.1889 |
| 1125 | 4.0650 | 3.8412 | -0.2238 |
| 1250 | 3.6495 | 3.6514 | +0.0019 |
| 2000 | 3.4486 | 3.4559 | +0.0073 |
| 4000 | 3.2585 | 3.2607 | +0.0022 |
| 5100 | 3.1113 | 3.1139 | +0.0026 |

At the same wall-clock time (5106s, when E7b ends at `3.1113`), the Lattice run is at about `3.218`.

## Takeaway

The Lattice learns much faster early: it is about `0.22` better at step 1125. But that gain lives only in the short-sequence phase of the sequence-length curriculum. After the switch to full length at step 1250, it is slightly worse at every check and ends `+0.0033` above the best. It also adds about 15% step time. So it stays as an implemented experiment, not part of the best config.

Open follow-ups: vstream, the Jacobi refiner (J3), a wider latent (`lattice_dr` 512 or `n_embd`) with more Lattice layers, groupwise `gamma`, and a fused scan kernel.

---

**Paper**: [Latent Lattice Language Models](https://arxiv.org/abs/2611.21608)  
**Implementation**: `bla_gpt/lattice.py::LatticeLayer`, `bla_gpt/lattice.py::lattice_scan`, `bla_gpt/bla_gpt.py::GPT._init_lattice`
