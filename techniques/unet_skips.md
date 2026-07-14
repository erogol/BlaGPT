# U-net Long Skips: Encoder-to-Decoder Layer Shortcuts

*Learned scalar skip connections from early transformer layers into mirrored later layers.*

---

## Overview

U-net long skips are a simple architecture trick from the modded-nanogpt lineage: save outputs from early layers and add them into later layers at mirrored depths. The shape is U-net-like, but the model stays a decoder-only transformer.

The short-horizon search had a promising U-net-skip candidate. F87 retested it in the full 5100-step regime on top of the latest best stack.

## Method

For the first half of layers:

```text
skip[k] = output_of_layer_k
```

For the second half:

```text
x = x + w_i * skip[n_layer - 1 - i]
```

Each skip has one learnable scalar. The kept run used:

```python
unet_skip_init = 0.25
```

So the mechanism adds only `n_layer // 2` parameters.

## Placement in BlaGPT

The implementation lives in `bla_gpt/bla_gpt.py::GPT.forward()`.

Configuration:

```python
use_unet_skips = True
unet_skip_init = 0.25
```

With Attention Residuals active, placement matters:

1. AttnResidual mixes previous layer deltas into the current stream.
2. U-net skip is added after that mix and before the block.
3. The block processes the skip-augmented stream.
4. Early layer outputs are stored after their block completes.

## Result

Full 5100-step autoresearch runs:

| Run | Change | Validation loss | Delta |
|-----|--------|-----------------|-------|
| F85 | Value Residual best | `3.2011` | - |
| F87 | + U-net long skips | `3.1979` | `-0.0032` |
| F87c | aggregate confirmation | `3.1961` | `-0.0018` vs F87 |

F87 crossed the clean-keep threshold. F87c showed the fully merged stack moved in the right direction again, but the extra gain over F87 is small and stays pending confirmation by the autoresearch rules.

## Takeaway

Late layers can reuse early representations directly instead of reconstructing them through the residual stream. It is not fancy. It just worked, which is rude but useful.

---

**Source lineage**: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)  
**Implementation**: `bla_gpt/bla_gpt.py::GPT.forward`, `GPTConfig.use_unet_skips`
