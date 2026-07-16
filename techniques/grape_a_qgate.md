# GRAPE-A Query-Gated: Replacing RoPE with Learned Additive Position Biases

*Position-dependent attention bias parameterized by the query, unifying ALiBi-style decay with a per-token gating signal.*

---

## Overview

GRAPE (Group Representational Position Encoding, arXiv:2512.07805) provides a unified framework for position encodings via group actions. The Additive branch subsumes ALiBi and FoX as special cases. The query-gated variant adds a data-dependent gate: instead of a fixed decay slope per head, the penalty for attending to distant keys is modulated by the current query.

BlaGPT tested the query-gated Additive variant as a full replacement for RoPE. The hypothesis: a learned, query-dependent distance penalty could outperform fixed rotary rotations, especially when composed with the existing XSA + GOAT sink + composable gated attention stack.

## Method

For head $h$, query position $i$, key position $j$:

$$b_h(i,j) = -(i-j) \cdot \text{softplus}(\omega_h) \cdot \text{slope}_h \cdot \text{softplus}\left(\frac{v_h^\top q_i}{\sqrt{D}}\right)$$

This bias is added to the attention scores before softmax:

```text
scores[i, j] = q_i k_j / sqrt(d) + b_h(i, j)
attn = softmax(scores)
```

The components:

- **Distance** $(i-j)$: causal, clamped to $\geq 0$. Same triangular structure as ALiBi.
- **Decay rate** $\omega_h$: one scalar per head, parameterized through `softplus` so it stays non-negative. Controls how aggressively distance penalizes attention.
- **ALiBi slopes** $\text{slope}_h$: the standard ALiBi geometric slope assignment per head. Provides the base multi-scale decay pattern.
- **Query gate** $v_h^\top q_i / \sqrt{D}$: a per-head projection vector $v_h$ (shape `[D]`) dotted with the query, then passed through `softplus`. This makes the distance penalty content-dependent — some queries penalize distant keys more, others less. The gate is L2-normalized at initialization and runtime.

When `pos_encoding="grape_a_qgate"`, RoPE is not applied at all. The positional signal comes entirely from the additive bias.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/attentions.py`:

- `GrapeQueryGatedBias` is an `nn.Module` that precomputes distance and slope buffers at `block_size` (non-persistent, compile-safe).
- Parameters: `omega` (`[n_head]`) and `v` (`[n_head, head_dim]`). Total overhead: $H + H \times D$ parameters.
- The bias is injected into three attention paths:
  - Base `Attention._flash_attention` — builds a combined causal + GRAPE float mask, disables SDPA's built-in `is_causal`.
  - Base `Attention._manual_attention` — adds the bias to the score matrix directly.
  - `GOATSinkAttention` overrides — GRAPE composes with the key-0 sink prior in both flash and manual paths.
- `ComposableGatedAttention` inherits from `GOATSinkAttention`, so the full stack (XSA + GOAT sink + gated output + GRAPE) works automatically.

Configuration:

```python
pos_encoding = "grape_a_qgate"
attention = "xsa"
use_goat_sink_prior = True
use_composable_gated_attn = True

# optional hyperparameters (defaults shown):
grape_omega_init = 1.0       # initial softplus-inverse target for omega
grape_v_init_std = None       # None => 1/sqrt(head_dim)
grape_v_l2_norm = True        # L2-normalize v at runtime
```

## Result

Full 5100-step autoresearch runs:

| Run | Change | Validation loss | Delta | Step avg | Peak memory |
|-----|--------|-----------------|-------|----------|-------------|
| F90r2 | lr ×1.4 confirmed (RoPE) | `3.1965` | — | `491ms` | `58632 MiB` |
| F92 | (lost with pod; not in GitHub ledger) | `3.1925` | `-0.0040` | — | — |
| **F94** | **GRAPE-A query-gated replaces RoPE** | **`3.1897`** | **`-0.0068`** | **`580ms`** | **`59643 MiB`** |

F94 is a clean keep and the new trusted best. The +18% step-time overhead (580ms vs 491ms) comes from replacing the flash causal fast-path with an explicit `(B, H, T, T)` float attention mask — the same cost as GOAT sink's masked path, just applied at every layer instead of only where the sink is active.

## Takeaway

RoPE is not the only way to encode position. A learned additive bias that adapts to the query — "should this token care about distant context?" — outperformed fixed rotary rotations at the same training budget. The mechanism is cheap in parameters (a few hundred) and composes cleanly with the existing attention stack. The wall-clock cost is real but bounded: it is an attention-mask problem, not a parameter-count problem.

---

**Paper**: [GRAPE: Generalizing Position Encoding via Group Representations](https://arxiv.org/abs/2512.07805)  
**Reference implementation**: [model-architectures/GRAPE](https://github.com/model-architectures/GRAPE)  
**Implementation**: `bla_gpt/attentions.py::GrapeQueryGatedBias`
