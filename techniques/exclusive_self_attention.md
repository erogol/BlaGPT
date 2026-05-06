# Exclusive Self Attention

*An attention modification that removes the value-direction component from the attention output.*

---

## What is Exclusive Self Attention?

Exclusive Self Attention (XSA) comes from Shuangfei Zhai's Apple paper, [Exclusive Self Attention](https://arxiv.org/html/2603.09078v1). The core idea is simple: after normal attention computes a per-token output, remove the part of that output that points in the same direction as the token's value vector.

So the attention output keeps information gathered from context, but is discouraged from copying the token's own value direction back into the residual stream.

## How does it work?

Standard attention computes:

```text
y = softmax(QK^T / sqrt(d)) V
```

XSA then normalizes the value vector and subtracts the projection of `y` onto that direction:

```python
v_norm = normalize(v)
y = y - dot(y, v_norm) * v_norm
```

In BlaGPT this lives in `ExclusiveSelfAttention(Attention)` and reuses the normal attention path. The only extra step is the exclusion projection after attention and before the output projection.

## Why might it help?

The paper argues that normal self-attention can spend capacity reinforcing the current token's own value representation. XSA removes that self-aligned component, forcing the attention output to carry more contextual information.

Intuitively: if attention is supposed to mix information across tokens, XSA stops part of the layer from being a fancy identity path.

## BlaGPT implementation

Enable it with:

```python
config = GPTConfig(attention="xsa")
```

or in the best model config:

```python
attention: str = "xsa"
```

Implementation notes:

- Subclass: `ExclusiveSelfAttention(Attention)` in `bla_gpt/attentions.py`
- Config key: `attention="xsa"`
- Works with the existing attention implementation and GQA path
- Adds no learned parameters
- Adds one normalize, dot product, and vector subtraction per attention output

## Experiment result

Best-model comparison, with the original pre-HyperConnections config kept the same except for the attention type:

```text
Exclusive Self Attention - best_model_loss: `3.2327` -> new_best_model_loss: `3.2303` - train_loss: `3.1987` - peak memory: `49859 MiB` - step_avg: `424.54ms`
```

This is a small win, but a real one in this run. It beat the previous comparable best by `0.0024` validation loss while using less memory than the heavier HyperConnections experiment.

## When to use it

Use XSA when you want a cheap attention variant that changes behavior without adding parameters. It is especially attractive for architecture search because it is easy to isolate: swap only the attention class and keep the rest of the model fixed.

Do not expect magic. The BlaGPT result was modest. The useful signal is that the change improved validation loss without extra parameters or a large speed penalty.
