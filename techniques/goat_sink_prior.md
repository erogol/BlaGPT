# GOAT Sink Prior: A Tiny Attention Prior That Helped

*A per-head log-prior on key position 0, added before the attention softmax.*

---

## Overview

GOAT sink prior comes from *You Need Better Attention Priors* by Litman and Guo (2026). The full paper studies optimal-transport attention priors. The useful slice for BlaGPT was much smaller: let every head learn a bias for the first key position, the usual attention-sink slot.

This is cheap and composable. It adds one scalar per head and keeps the existing Exclusive Self Attention path.

## Method

For each query position and head:

```text
scores[i, j] = q_i k_j / sqrt(d)
scores[i, 0] = scores[i, 0] + u_head
attn = softmax(scores)
```

`u_head` is initialized to zero, so training starts identical to the old XSA model. During training, each head can decide whether key 0 should be a stronger or weaker sink.

This is not the full GOAT package. The relative spectral prior was tested separately and did not beat the current best.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/attentions.py`:

- `GOATSinkAttention` subclasses `ExclusiveSelfAttention`
- `sink_prior` is an `nn.Parameter` with shape `[n_head]`
- FlashAttention and manual attention paths both add the key-0 prior before softmax

Configuration:

```python
attention = "xsa"
use_goat_sink_prior = True
```

`get_attention()` dispatches `attention="xsa"` + `use_goat_sink_prior=True` to `GOATSinkAttention`.

## Result

Full 5100-step autoresearch runs:

| Run | Change | Validation loss | Status |
|-----|--------|-----------------|--------|
| baseline | combined keeps | `3.2354` | baseline |
| F74 | GOAT sink prior | `3.2352` | pending confirmation |
| F74c | independent confirmation | `3.2298` | confirmed keep |

The first run barely moved, but the confirmation run was clearly better. It became part of the canonical best stack.

## Takeaway

Tiny prior, real gain. The model gets a cheap knob for sink behavior instead of forcing content logits to learn it from scratch.

---

**Paper**: [You Need Better Attention Priors](https://arxiv.org/abs/2601.15380)  
**Implementation**: `bla_gpt/attentions.py::GOATSinkAttention`
