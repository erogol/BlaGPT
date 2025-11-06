# BlaGPT

Experimental playground for benchmarking language model (LM) architectures, layers, and tricks on smaller datasets. Designed for flexible experimentation and exploration.

## 📚 Technique Documentation

See the [**techniques/**](./techniques/) directory for explanations of various techniques implemented in this repository.

**Latest**: [KDA from Kimi-Linear](./techniques/kda.md) - Linear attention alternative

## BlaGPT Model
BlaGPT is a flexible Transformer implementation that you can turn on/off following things in the config.

Results below are the numbers after an epoch of training with fineweb10B, mostly using the default parameters. My goal is to see how things work without fiddling with the model and hyperparameters a lot.

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

Differential Attention - [link](https://arxiv.org/abs/2410.05258) - best_model_loss: `3.2411` -> loss: `3.2460` - peak memory: `41521 MiB`

Softpick - [link](https://arxiv.org/abs/2504.20966) - loss: `3.3446` - peak memory: `59417 MiB`

Canon Layer - [link](https://physics.allen-zhu.com/part-4-architecture-design/part-4-1) - loss: `3.3217` - peak memory: `43199 MiB`

Parallel Transformer Block - [link](https://arxiv.org/abs/2204.02311) - loss: `3.3473` - peak memory: `40302 MiB`

Per Layer Token Embedding - [link](https://blog.google/technology/developers/gemma-3/) - loss: `3.2411` - peak memory: `40916 MiB`

PolyNorm - [link](https://arxiv.org/html/2411.03884v1) - best_model_loss: `3.2411` -> loss: `3.3017` - peak memory: `40895 MiB`

PolyReLU - [link](https://arxiv.org/html/2411.03884v1) - best_model_loss: `3.2411` -> loss: `3.2642` - peak memory: `40890 MiB`

TOP loss - [link](https://erogol.notion.site/Predicting-the-Order-of-Upcoming-Tokens-Improves-Language-Modeling-25c7621486338183a12ec3621ee8a6b5?source=copy_link) - best_model_loss: `3.2411` -> loss: `3.2636` - peak memory: `47816 MiB`


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

Avey - [link](https://arxiv.org/pdf/2506.11305v1) - loss: `3.323`, peak memory: `51962 MiB` (batch size 8), step_time: `2871ms` (very slow to train and uses >3x more memory than other models)

LFM2 - [link](https://huggingface.co/LiquidAI/LFM2-1.2B) - TBD

Kimi Delta Attention (1:3 interleaved Full Attention) - [link](https://arxiv.org/abs/2510.26692) - best_model_loss: `3.2411` -> loss: `3.2532`, peak_memory:`47391`,  step_time: `568.1ms`


## Byte-Level Models
Hourglass Transformer (modified) - [link](https://arxiv.org/abs/2110.13711) - `val_loss:1.0048 train_time:2671049ms step_avg:524.76ms`

AUNet - [link](https://arxiv.org/abs/2506.14761) - `val_loss:1.1502 train_time:7246104ms step_avg:1423.60ms`

SpaceByte - [link](https://arxiv.org/abs/2404.14408) - `val_loss:1.6755 train_time:2154923ms step_avg:423.36ms peak memory consumption: 27781 MiB`

HNet - [link](https://arxiv.org/pdf/2507.07955) - `val_loss:1.4554 train_time:2207809ms step_avg:433.75ms peak memory consumption: 23948 MiB`


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

AdaMuon - [link](https://arxiv.org/abs/2507.11005) - Adaptive Muon with second-moment estimation (default optimizer) - **See [detailed explanation](./techniques/adamuon.md)**

BiClip - [link](https://arxiv.org/pdf/2502.04164) - (not working well) loss: `7.2292`, peak VRAM: `39751 MiB`, step_time: `510ms`

NorMuon - [link](https://arxiv.org/html/2510.05491v1) - best_model_loss: `3.2411` -> loss: `3.4630`, peak VRAM: `44154 MiB`, step_time: `387.46 ms`  - **See [detailed explanation](./techniques/normuon.md)**


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
