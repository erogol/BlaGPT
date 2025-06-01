# BlaGPT

Experimental playground for benchmarking language model (LM) architectures, layers, and tricks on smaller datasets. Designed for flexible experimentation and exploration.

## BlaGPT Model
BlaGPT is a flexible Transformer implementation that you can turn on/off following things in the config.

Multi-token prediction - [link](https://arxiv.org/pdf/2404.19737)

Weight tying - [link](https://arxiv.org/abs/1608.05859v3)

Grouped query attention - [link](https://arxiv.org/pdf/2305.13245)

Capping logits - [link](https://arxiv.org/pdf/2408.00118)

QKV bias - [link](https://arxiv.org/abs/2407.10671)

Zero-init projection layer - [link](https://arxiv.org/abs/2407.10671)

Post and pre-RMSNorm - [link](https://arxiv.org/pdf/2408.00118)

Setting base theta to 1_000_000 - [llama3](https://github.com/meta-llama/llama3/blob/main/llama/model.py#L49) - increased the final validation loss - best `3.3324`

Z-loss regularization - [link](https://arxiv.org/pdf/2309.14322) - increased the final validation loss by 0.02 - loss: `3.3527`

KV-Shifting attention - [link](https://arxiv.org/abs/2411.19574) - seems to improve performance - loss: `3.3310` -> `3.3138` - peak memory consumption: `42858 MiB`

Dilated Attention (LongNet) - [link](https://arxiv.org/pdf/2307.02486)

Multi-Head Latent Attention - [link](https://arxiv.org/abs/2502.07864) - loss: `3.3479` - peak memory consumption: `42192 MiB`

Per token output bias - [link]() - loss: `3.3257` - peak memory consumption: `42120 MiB`

DyT Norm - [link](https://arxiv.org/html/2503.10622v1) - didn't really work. Loss stuck too high

Forgetting Transformer (Vanilla and Pro vers) - [link](https://openreview.net/pdf?id=q2Lnyegkr8) - vanilla loss: `3.3243`, pro loss: `OOM`

Multi-Token Attention - [link](https://arxiv.org/pdf/2504.00927) - loss: `3.3357` - peak memory: `42136 MiB`

Differential Attention - [link](https://arxiv.org/abs/2410.05258) - loss: `3.3352` - peak memory: `41521 MiB`

Softpick - [link](https://arxiv.org/abs/2504.20966) - loss: `3.3446` - peak memory: `59417 MiB`

Canon Layer - [link](https://physics.allen-zhu.com/part-4-architecture-design/part-4-1) - loss: `3.3217` - peak memory: `43199 MiB`

Parallel Transformer Block - [link](https://arxiv.org/abs/2204.02311) - loss: `3.3473` - peak memory: `40302 MiB`

Per Layer Token Embedding - [link](https://blog.google/technology/developers/gemma-3/) - loss: `3.2411` - peak memory: `40916 MiB`

## Other Models
MegaByte - [link](https://arxiv.org/abs/2305.07185) - loss: `3.810`

FTP (heavily modified) - [link](https://arxiv.org/pdf/2410.18160) - loss: `3.901`

Rene - [link](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch) - loss: `3.340`

Rwkv7 - [link](https://github.com/BlinkDL/RWKV-LM) - loss: `4.450`

Zamba2 - [link](https://huggingface.co/Zyphra/Zamba2-2.7B) - Zamba2 > Rene > Rwkv7

Hourglass Transformer (modified) - [link](https://arxiv.org/abs/2110.13711) - Hourglass > MegaByte > FTP - loss: `3.710`

Hymba - [link](https://arxiv.org/html/2411.13676v1) - train step time is significantly slower than the transformers. Best validation loss so far: `4.7505`

Tokenformer (in BlaGPT model) - [link](https://github.com/Haiyang-W/TokenFormer) - loss: `3.390`

LLaDa (dLLM) - [link](https://arxiv.org/abs/2502.09992) - val-loss: `8.6930`, xentropy-loss: `4.2891` (comparable to other models and estimated by `llada_validation_cross_entropy.py`),

## Optimizers
PaLMForeachSOAP - [link](https://github.com/ClashLuke/HeavyBall) - almost 2 times slower than Adam but the best results

Ademamix - [link](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch/blob/main/AdEMAMix.py) - Unstable even after trying different learning rates.

Adopt - [link](https://github.com/iShohei220/adopt) - straight up Nan

CAdamW - [link](https://github.com/kyleliang919/C-Optim/blob/main/c_adamw.py) - loss: `3.3517`

AdamW with independent weight decay - [link](https://arxiv.org/pdf/2309.14322) - loss: `3.320`

Adam - loss: `3.3224`

AdamW - loss: `3.3310`, peak VRAM: `42053 MiB`, step_time: `533ms`

DeMo - [link](https://arxiv.org/abs/2411.19870) - Saves 7 GB per GPU, loss is higher than baseline, step time is slower than Adam -  loss: `3.4676`, peak VRAM: `41534 MiB`, step_time: `820ms`

Adam-Mini - [link]() - loss is higher than Adam and AdamW and also slower ??, saved a bit of VRAM  - loss: `3.3324`, peak VRAM: `41534 MiB`, step_time: `610ms`

MARS - [link](https://github.com/AGI-Arena/MARS) - loss: `3.3459`, peak VRAM: 40953 MiB, step_time: `628ms`

Muon - [link](https://kellerjordan.github.io/posts/muon/) - loss: `3.2923`, peak VRAM: `40332MB`, step_time: `620.24ms`


## Adding a New Model

- Implement the model
- Return the loss in the forward function
- Add model to `model_registry.py`
- And start training

See one of the implementations for details.


## Training

- Get the data by running `data/fineweb10B_cached.py`

- Start training with:

```bash
torchrun --standalone --nproc_per_node=8 train.py --run_name pre_post_norm --model_name blagpt
```

- (Optional) Run the learning rate finder before the training

```bash
torchrun --standalone --nproc_per_node=8 find_lr.py --model_name blagpt

# Output
Results:
Steepest gradient learning rate: 3.31e-06
Elbow point learning rate: 1.20e-01
Plot saved to: logs/lr_finder_blagpt/lr_finder_plot.png
Results saved to: logs/lr_finder_blagpt/lr_finder_results.pt
```

## Best Model So Far

- Check `best_model_config.py` for the best model configuration so far.

- You can run the training with the best model config by running:

```bash
torchrun --standalone --nproc_per_node=8 train.py --run_name best_model --model_name best
```

## Acknowledgements

The initial code is based on

Nano GPT - [link](https://github.com/karpathy/nanoGPT)

Modded NanoGPT - [link](https://github.com/KellerJordan/modded-nanogpt)

Thanks to @xumingyu2021 for memory friendly implementation of the Differential Attention
