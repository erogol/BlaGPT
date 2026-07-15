# Composable Gated Attention: Gating the Existing Attention Stack

*Gated Attention applied on top of XSA + GOAT instead of replacing the attention class.*

---

## Overview

BlaGPT already had standalone Gated Attention from the Qwen paper *Gated Attention for Large Language Models*. That implementation works as its own attention class.

The autoresearch stack needed a narrower version: keep the working XSA + GOAT path, then add the gated-attention output gate on top. That is `ComposableGatedAttention`.

## Method

Compute attention normally:

```text
y = Attention(q, k, v)
```

Compute a gate from the input hidden state:

```text
g = sigmoid(x W_gate)
```

Reshape it per head and multiply:

```text
g: [batch, time, d_model] -> [batch, n_head, time, head_dim]
y' = y * g
```

## Why Composable?

The class subclasses `GOATSinkAttention`, so it keeps:

- Exclusive Self Attention
- GOAT key-0 sink prior
- the normal XSA dispatch path

and adds only the post-attention gate.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/attentions.py`:

```python
class ComposableGatedAttention(GOATSinkAttention):
    ...
```

Configuration:

```python
attention = "xsa"
use_goat_sink_prior = True
use_composable_gated_attn = True
```

`get_attention()` checks `use_composable_gated_attn` before plain `use_goat_sink_prior`, because the composable class already includes the sink prior.

## Result

The combined full run was F82F84: HybridNorm plus Composable Gated Attention.

| Run | Change | Validation loss | Delta |
|-----|--------|-----------------|-------|
| F82 | HybridNorm path | `3.2224` | - |
| F82F84 | + Composable Gated Attention | `3.2128` | `-0.0096` |

That was a clean keep and the largest jump in this part of the chain.

## Takeaway

The useful move was not just "use Gated Attention". It was "add the gate without throwing away the attention stack that already works." Boring engineering, good result.

---

**Paper**: [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)  
**Implementation**: `bla_gpt/attentions.py::ComposableGatedAttention`
