# Key-Dimension Attention: Kimi Delta Attention in BlaGPT

*A linear-attention path with gated delta-state updates, modeled after Kimi Delta Attention.*

---

## Overview

Key-Dimension Attention (KDA) is BlaGPT's implementation path for Kimi Delta Attention style linear attention. It replaces quadratic softmax attention with a recurrent/chunked state update over keys and values.

The goal is lower long-context cost, not a small short-context loss trick.

## Method

KDA projects queries, keys, values, and gating terms, then updates a delta-style memory state over chunks. Instead of materializing a full `[time, time]` attention matrix, it uses the FLA chunk kernel:

```text
state_t = decay_t * state_{t-1} + key_t/value_t update
out_t   = query_t read from state_t
```

BlaGPT also exposes short-convolution and decay-rank knobs for the KDA path.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/attentions.py`:

- `KDAAttention`
- optional KDA/full-attention interleaving in `GPTConfig`
- dispatch through `attention = "kda"` or `use_kda_interleaving = True`

Configuration fields:

```python
attention = "kda"
kda_chunk_size = 64
kda_use_short_conv = True
kda_decay_rank = None

use_kda_interleaving = True
kda_interleave_ratio = 4
kda_interleave_with = "regular"
```

## Dependency Note

KDA requires the `fla` / Flash Linear Attention package for `chunk_kda`. If `fla` is not installed, `attention="kda"` will fail at runtime. Do not launch it on pods without that dependency.

## Result

Earlier README benchmark for the interleaved KDA setup:

```text
best_model_loss: 3.2411 -> loss: 3.2532
peak_memory: 47391 MiB
step_time: 568.1ms
```

It did not beat the transformer best in that run.

## Takeaway

KDA is mainly a long-context efficiency experiment here. It is implemented, but it is not part of the current best BlaGPT config.

---

**Implementation**: `bla_gpt/attentions.py::KDAAttention`
