# RoVE: Rotary Value Embeddings

*Rotate the values with the same RoPE rotation as the keys, so the message a token sends depends on how far it is from the query.*

---

## Overview

RoPE makes attention **scores** position-relative: $q_i^\top k_j$ only sees the offset $j - i$. The **values** stay position-blind. Once the softmax picks a token, the vector it contributes is the same whether it is 1 token or 900 tokens away.

RoVE (arXiv:2606.11275) fixes this with zero new parameters. It rotates each value by its own position before aggregation, then rotates the attention output back by the query position. The paper shows this turns RoPE attention into an attentive convolution and reports gains on 124M and 354M GPT-2 models, with the clearest gains on long-range aggregation tasks.

BlaGPT tested it in two settings: on top of plain RoPE (E7b), and as a hybrid on top of the GRAPE-A additive bias that the best stack used at the time (E7c).

## Method

Let $R_t$ be the standard RoPE rotation for position $t$ (block-diagonal 2D rotations, the same one applied to $q$ and $k$). Standard RoPE attention for query position $i$:

$$y_i = \sum_j a_{ij} \, v_j, \qquad a_{ij} = \text{softmax}_j\left(\frac{(R_i q_i)^\top (R_j k_j)}{\sqrt{d}}\right)$$

RoVE rotates the values by their source position and un-rotates the output by the query position:

$$y_i = R_i^{-1} \sum_j a_{ij} \, R_j v_j = \sum_j a_{ij} \, R_{j-i} \, v_j$$

The second form is the point. Because $R$ is a rotation, $R_i^{-1} R_j = R_{j-i}$, so each value is rotated by its **relative** offset from the query. The value pathway becomes position-relative in the same way the score pathway already is.

```text
v = rope_rotate(v, pos=key_positions)        # before attention
y = attention(q, k, v)                       # unchanged SDPA / flash path
y = rope_rotate(y, pos=query_positions, -1)  # inverse = transpose = (cos, -sin)
y = c_proj(y)
```

Properties:

- **Zero parameters.** It reuses the RoPE frequency table.
- **Flash-compatible.** Both rotations happen outside the attention kernel, so the causal fast path is untouched.
- **Needs an invertible rotation.** Only `rope_variant="standard"` (2D rotations) works; the "simplified" concatenation variant is rejected.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/attentions.py`:

- `Attention.__init__` builds a separate `self.rove_rotary = Rotary(head_dim, base=rope_theta)` when `rove=True`. A separate module lets RoVE also run with `pos_encoding="grape_a_qgate"`, where no RoPE is applied to $q$/$k$.
- `_apply_rove_value(v, T)` rotates `v` after `_prepare_qkv` and HybridNorm's V-norm, right before attention.
- `_apply_rove_output_inverse(y, T_q)` applies `apply_rotary_emb(y, cos, -sin)` to the attention output, before the output projection.
- Config validation: `rove=True` requires `pos_encoding="rotary"` with `rope_variant="standard"`, or `pos_encoding="grape_a_qgate"`.

Configuration:

```python
pos_encoding = "rotary"
rope_variant = "standard"
rope_theta = 1_000_000
rove = True
```

Note: with `zero_init_proj_layers=True` the output projection starts at zero, so RoVE on and RoVE off give identical loss at step 0. The difference appears after 2-3 steps. This is expected, not a bug.

## Result

Full 5100-step runs on the Aurora + Engram ×20 stack (B0):

| Run | Change | Seed 1337 | Seed 2 | Step avg (s1337) |
|-----|--------|-----------|--------|------------------|
| B0 | GRAPE-A (baseline) | `3.1265` | `3.1241` | `1066ms` |
| E7a | RoPE replaces GRAPE-A | `3.1134` | `3.1250` | `993ms` |
| **E7b** | **RoPE + RoVE** | **`3.1106`** | **`3.1240`** | **`1018ms`** |
| E7c | GRAPE-A + RoVE (hybrid) | `3.1262` | — | `1090ms` |
| Combined | E7b + MHAR H=8 | `3.1113` | — | — |

- RoVE vs plain RoPE: `-0.0028` (seed 1337) and `-0.0010` (seed 2). Same direction on both seeds, but both gaps are below the ~0.0024 seed noise of B0.
- Peak memory (seed-2 runs): `85258 MiB` with RoVE vs `85160 MiB` without. About 2.5% more step time in the seed-1337 runs.
- The GRAPE-A hybrid (E7c) showed no gain. RoVE needs the RoPE score pathway to pair with.

E7b is the new best and the active best config.

## Takeaway

RoVE is a nearly free change on top of RoPE: no parameters, two extra elementwise rotations, and the flash path stays intact. The gain at this scale is small but consistent across two seeds. The bigger win in this round came from going back to RoPE itself: dropping GRAPE-A's explicit float attention mask made the model about 7% faster in the seed-1337 runs (about 3% in the seed-2 runs, which used a different machine) at the same or better loss, and RoVE only works on top of RoPE.

---

**Paper**: [RoVE: Rotary Value Embeddings Attention for Relative Position-dependent Value Pathways](https://arxiv.org/abs/2606.11275)  
**Implementation**: `bla_gpt/attentions.py::Attention._apply_rove_value`, `_apply_rove_output_inverse`
