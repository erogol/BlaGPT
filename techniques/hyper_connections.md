# Hyper-Connections

*A residual-connection replacement that keeps multiple hidden-state streams and learns how each block reads from and writes back to them.*

---

## Overview

Normal Transformer blocks have one residual stream:

```text
x = x + block(norm(x))
```

Hyper-Connections expand this into `n` parallel residual streams. Each attention or MLP branch first mixes the streams into one branch input, runs the normal branch, then injects the branch output back into all streams. The goal is to give the network a richer residual communication pattern without changing the attention or MLP internals.

In BlaGPT this is implemented as Dynamic Hyper-Connections with 4 streams, `DHCx4`.

## Method

For each branch, Hyper-Connections use three learned mixing objects:

- `A_m`: mixes the residual streams into the branch input
- `A_r`: carries and mixes the old streams forward
- `B`: decides how strongly the branch output is injected into each stream

Static form:

```text
branch_in = sum_s A_m[s] * stream[s]
branch_out = branch(norm(branch_in))
next_stream[u] = sum_s A_r[s, u] * stream[s] + B[u] * branch_out
```

With dynamic Hyper-Connections, `A_m`, `A_r`, and `B` also get token-dependent deltas predicted from the current streams:

```text
A_m = A_m_static + delta_m(streams)
A_r = A_r_static + delta_r(streams)
B   = B_static   + delta_b(streams)
```

The deltas are zero-initialized and scaled by small learned factors, so training starts close to the static schedule instead of immediately scrambling the residual paths.

## Implementation in BlaGPT

The implementation lives in `bla_gpt/bla_gpt.py`:

- `HyperConnection`: stream mixing and dynamic projections
- `HyperBlock`: wraps the normal attention and MLP branches with Hyper-Connections
- `GPT.forward`: expands the hidden state to `[batch, seq, streams, dim]` before the blocks and sums streams back to `[batch, seq, dim]` before the final norm

Configuration fields:

```python
use_hyper_connections = True
hyper_num_streams = 4
hyper_dynamic = True
hyper_tanh = True
```

The registered experiment model is in `bla_gpt/model_registers.py`:

```python
config.use_hyper_connections = True
config.hyper_num_streams = 4
config.hyper_dynamic = True
```

Two implementation details mattered:

1. The dynamic projection weights are initialized to zero, with small `1e-2` scales for the deltas.
2. Residual projection init is scaled by `sqrt(hyper_num_streams)` when Hyper-Connections are enabled, because the model later sums multiple streams.

## Result

Run: `best_hyper_connections_scale_fix`

Final result on the best-model stack:

- step: `5100/5100`
- validation loss: `3.2741`
- train loss: `3.2388`
- peak memory: `68729 MiB`
- step average: `928.26ms`

This did not beat the old best around `3.2327`. It also made training much heavier. The run saturated 8 H100 80GB GPUs and used roughly 69GB peak memory per GPU.

## Takeaway

Hyper-Connections are interesting mechanically, but for this BlaGPT setup they are not a free win. The extra streams and dynamic mixing add a lot of memory and compute, while the final validation loss was worse than the previous best. For now it stays as an implemented experiment, not part of the best config.

---

**Paper**: [Hyper-Connections](https://arxiv.org/abs/2409.19606)  
**Implementation**: `bla_gpt/bla_gpt.py::HyperConnection`, `bla_gpt/bla_gpt.py::HyperBlock`
