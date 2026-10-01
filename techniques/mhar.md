# Multi-Head Attention Residuals (MHAR)

*Split the depth-routing query of Attention Residuals into H heads, so each feature subspace picks its own mix of earlier layers.*

---

## Overview

The best stack already uses Attention Residuals (Kimi/MoonshotAI): instead of a plain additive residual stream, each block reads a softmax-weighted mix of all earlier layer outputs. That read uses **one** routing query across the full width. Every feature subspace has to accept the same depth mix, even when different subspaces want different layers.

MHAR (arXiv:2607.27230) reshapes the routing query into $H$ heads, each with its own softmax over depth. The reshape adds zero parameters and negligible compute, and $H=1$ is exactly the original Attention Residuals. The paper reports a U-shaped loss in $H$ with a flat optimum at $H=4$ or $H=8$.

## Method

Let $v_0, \dots, v_{N-1}$ be the earlier layer outputs (width $D$) and $w$ the learned routing query of the current block. Original Attention Residuals:

$$\alpha_n = \text{softmax}_n\left(w^\top \text{RMSNorm}(v_n)\right), \qquad x = \sum_n \alpha_n v_n$$

MHAR splits $w$ and each $v_n$ into $H$ slices of width $D/H$ and routes each slice on its own:

$$\alpha_n^{(h)} = \text{softmax}_n\left(w^{(h)\top} \text{RMSNorm}(v_n^{(h)})\right), \qquad x^{(h)} = \sum_n \alpha_n^{(h)} v_n^{(h)}$$

$$x = \text{concat}\left(x^{(1)}, \dots, x^{(H)}\right)$$

```text
w_h  = w.view(H, D/H)                    # reshape only, no new params
V_h  = stack(srcs).view(N, B, T, H, D/H)
s    = einsum("he,nbthe->nbth", w_h, rms_norm(V_h))
a    = softmax(s, dim=depth)             # H independent softmaxes
x    = einsum("nbth,nbthe->bthe", a, V_h).reshape(B, T, D)
```

The read becomes block-diagonal: head $h$ can pull mostly from layer 2 while head $h'$ pulls from layer 7.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/bla_gpt.py`:

- `GPT._attn_res_mix(srcs, w)` handles both paths. `attn_res_heads == 1` keeps the original code byte-for-byte; `H > 1` uses the reshaped einsum path.
- RMSNorm is applied within each head's slice, matching the per-subspace routing.
- `_attn_res_route` does the depth softmax and still supports the OASIS null-branch variant.
- `n_embd` must be divisible by `attn_res_heads`.

Configuration:

```python
use_attn_res = True
attn_res_heads = 8   # 1 = original Attention Residuals
```

## Result

Full 5100-step runs on the Aurora + Engram ×20 stack (B0):

| Run | Change | Seed 1337 | Seed 2 | Step avg (s1337) | Peak memory |
|-----|--------|-----------|--------|------------------|-------------|
| B0 | H=1 (baseline) | `3.1265` | `3.1241` | `1066ms` | `85625 MiB` |
| E6a | H=4 | `3.1241` | — | `1143ms` | `93215 MiB` |
| E6b | H=8 | `3.1238` | `3.1203` | `1149ms` | `93225 MiB` (seed 2) |
| E7b | rotary + RoVE (H=1) | `3.1106` | — | `1018ms` | — |
| Combined | E7b + H=8 | `3.1113` | — | — | `92855 MiB` |

- On B0, H=8 helped on both seeds (`-0.0027`, `-0.0038`), but both gaps sit close to the ~0.0024 seed noise.
- It costs about 8% step time and about 7,600 MiB more peak memory, against the paper's "negligible compute". The einsum path materializes a `(N, B, T, H, D/H)` stack at every block.
- On top of rotary + RoVE (Combined), H=8 gave `+0.0007` vs E7b. The gain did not carry over to the new best stack.

Not in the best config.

## Takeaway

Per-subspace depth routing is a clean idea and it showed a small, consistent gain on the GRAPE-A stack. It did not stack with rotary + RoVE, and the real wall-clock and memory cost in this implementation is not small. A fused kernel could change the cost side; the loss side would need a bigger model, where the paper reports the gain grows with width.

---

**Paper**: [Multi-Head Attention Residuals](https://arxiv.org/abs/2607.27230)  
**Implementation**: `bla_gpt/bla_gpt.py::GPT._attn_res_mix`
