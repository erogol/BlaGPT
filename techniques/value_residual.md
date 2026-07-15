# Composable Value Residual: ResFormer as a Config Gate

*Mix the first layer's value vectors into deeper attention layers without replacing the whole model class.*

---

## Overview

Value Residual Learning comes from ResFormer. The idea is that deeper attention layers can lose direct token-level information, so later layers receive a residual connection from the first layer's value vectors.

BlaGPT already had a standalone `resformer.py`. F85 moved the mechanism into the normal BlaGPT attention path so it could compose with XSA, GOAT, GatedNorm, HybridNorm, and Composable Gated Attention.

## Method

For layer `l > 1`, mix first-layer values into the local values:

```text
V'_l = lambda1_l * V_1 + lambda2_l * V_l
```

BlaGPT uses a learnable-plus variant:

```text
lambda1 = softmax(v_res_logits) * v_res_scale
lambda2 = learned per-layer scalar, init 0.5
```

The first layer stores `V_1`. Later layers read it from a shared holder during the same forward pass.

## Implementation in BlaGPT

Configuration:

```python
use_value_residual = True
```

Code locations:

- `bla_gpt/bla_gpt.py::_init_value_residual()` creates the lambda parameters
- `bla_gpt/bla_gpt.py::GPT.forward()` refreshes the per-forward holder
- `bla_gpt/attentions.py::Attention._apply_value_residual()` mixes `V_1` into deeper layers

The hook is in the base `Attention._project_kv()` path, so it composes with the XSA -> GOAT -> Composable Gated Attention inheritance chain.

## Result

Full 5100-step autoresearch run:

| Run | Change | Validation loss | Delta |
|-----|--------|-----------------|-------|
| F82F84 | HybridNorm + Composable Gated Attention best | `3.2128` | - |
| F85 | + Value Residual | `3.2011` | `-0.0117` |

This was a clean keep and one of the strongest gains in the chain.

## Takeaway

The standalone ResFormer model did not fit the greedy stack. The mechanism did. Moving value residual learning into the normal attention path made it composable, and that version worked.

---

**Paper**: [Value Residual Learning](https://arxiv.org/abs/2410.17897)  
**Implementation**: `bla_gpt/bla_gpt.py::_init_value_residual`, `bla_gpt/attentions.py::Attention._apply_value_residual`
