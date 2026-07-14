# GatedNorm: Low-Rank Gates After RMSNorm

*A lightweight feature gate applied after RMSNorm.*

---

## Overview

GatedNorm comes from *A Unified View of Attention and Residual Sinks* by Qiu et al. (2026). The paper connects attention sinks, residual sinks, and activation outliers. The useful mechanism for BlaGPT was simple: normalize first, then let a small gate suppress bad channels.

PreAffineRMSNorm from the same paper was tested and discarded. GatedNorm was the keeper.

## Method

Plain RMSNorm returns:

```text
y = RMSNorm(x)
```

GatedNorm returns:

```text
y  = RMSNorm(x)
g  = sigmoid(W_up(swish(W_down(y))))
y' = g * y
```

The gate is low-rank:

```text
W_down: d_model -> r
W_up:   r -> d_model
```

The kept run used `r = 16`.

## Initialization

`W_up` is initialized to zero, so the initial gate is `sigmoid(0) = 0.5`. That is not exact identity, but it avoids sigmoid saturation and gives the up projection immediate gradients.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/norms.py` and is selected by `get_norm()` in `bla_gpt/bla_gpt.py`.

Configuration:

```python
use_gated_norm = True
gated_norm_rank = 16
```

When enabled, model-level RMSNorm sites become `GatedNorm`: attention pre-norms, MLP pre-norms, final norm, and optional post-norms. Q/K/V per-head norms are separate and are not affected by this switch.

## Result

Full 5100-step autoresearch run:

| Run | Change | Validation loss | Delta |
|-----|--------|-----------------|-------|
| F74c | GOAT sink prior best | `3.2298` | - |
| F77 | + GatedNorm rank 16 | `3.2230` | `-0.0068` |

This cleared the `0.003` clean-keep threshold.

## Takeaway

Small parameter cost, easy composition, real full-run gain. GatedNorm gives every norm site a cheap way to mute outlier channels before the next branch has to deal with them.

---

**Paper**: [A Unified View of Attention and Residual Sinks](https://arxiv.org/abs/2601.22966)  
**Implementation**: `bla_gpt/norms.py::GatedNorm`
