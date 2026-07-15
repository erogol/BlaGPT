# HybridNorm: Normalize Attention Values, Post-Norm the FFN

*QKV normalization inside attention plus a post-normalized residual path in the feed-forward branch.*

---

## Overview

HybridNorm comes from *HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization* by Zhuo et al. (2026). The paper's basic point is that attention and FFN branches do not need the same normalization pattern.

BlaGPT already normalized Q and K. HybridNorm adds V normalization and changes the FFN residual path.

## Method

### Attention side

```text
q = RMSNorm(q)
k = RMSNorm(k)
v = RMSNorm(v)
y = Attention(q, k, v)
```

Normalizing V controls the vectors that are averaged by attention, not just the logits.

### FFN side

The FFN branch uses the normalized input as the residual path:

```text
branch = FFN(Norm(x))
out = Norm(x) + branch
```

instead of:

```text
out = x + FFN(Norm(x))
```

## Implementation in BlaGPT

Configuration:

```python
use_hybrid_norm = True
```

Code locations:

- `bla_gpt/attentions.py`: adds `v_norm = RMSNorm(head_dim)` and applies it after QKV reshaping
- `bla_gpt/bla_gpt.py`: `Block._process_branch()` uses the normalized residual path when HybridNorm is active

It composes with XSA, GOAT sink prior, GatedNorm, and the later gated-attention stack.

## Result

Full 5100-step autoresearch run:

| Run | Change | Validation loss | Delta |
|-----|--------|-----------------|-------|
| F77 | GOAT + GatedNorm best | `3.2230` | - |
| F82 | + HybridNorm | `3.2224` | `-0.0006` |

The standalone gain was tiny and below confirmation threshold. It stayed in the path because it later combined well with Composable Gated Attention.

## Takeaway

HybridNorm was not exciting alone. It was still directionally positive and made the next attention change work better. Some wins are glue wins.

---

**Paper**: [HybridNorm](https://arxiv.org/abs/2503.04598)  
**Implementation**: `bla_gpt/attentions.py::Attention`, `bla_gpt/bla_gpt.py::Block`
