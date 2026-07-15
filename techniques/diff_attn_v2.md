# Differential Attention v2: Subtracting Attention Noise

*A two-map attention variant that subtracts one attention pattern from another to cancel common-mode noise.*

---

## Overview

Differential Attention v2 is BlaGPT's second differential-attention implementation. The idea is to compute two attention maps, subtract a learned fraction of the second from the first, then apply the result to values.

The motivation is simple: if both maps attend to broad background tokens, subtraction can cancel some of that shared noise and leave sharper token selection.

## Method

For each head pair:

```text
A1 = softmax(Q1 K1^T / sqrt(d))
A2 = softmax(Q2 K2^T / sqrt(d))
lambda = sigmoid(lambda_param)
Y = (A1 - lambda * A2) V
```

BlaGPT's v2 path uses a sigmoid lambda rather than the older heavier variant. That keeps the subtraction bounded and avoids the instability of an unconstrained scale.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/attentions.py`:

- `MultiheadDiffAttnv2`
- `diff_attn_v2_func()` compiled helper
- dispatch through `attention = "DiffAttnv2"`

Configuration:

```python
attention = "DiffAttnv2"
```

## Result

Earlier README benchmark:

```text
best_model_loss: 3.2296 -> new_best_model_loss: 3.2274
peak memory: 52829 MiB
step_avg: 535.16ms
```

It improved that older stack, but it is not part of the latest autoresearch best stack.

## Takeaway

Differential Attention v2 worked better than the first differential-attention attempt, but it is heavier than the later small composable keeps. Useful experiment, not the current winner.

---

**Implementation**: `bla_gpt/attentions.py::MultiheadDiffAttnv2`
