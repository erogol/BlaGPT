# BlaGPT

A collection of LM architectures, layers, and tricks to easily benchmark on a relatively small dataset. It is created fully for my experiments in my free time with no serious intentions.

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

Z-loss regularization - [link](https://arxiv.org/pdf/2309.14322) - increased the final validation loss by 0.02 - best: `3.3527`

KV-Shifting attention - [link](https://arxiv.org/abs/2411.19574) - best: `3.331` - peak memory consumption: `42858 MiB`

## Other Models
MegaByte - [link](https://arxiv.org/abs/2305.07185) - best: `3.810`

FTP (heavily modified) - [link](https://arxiv.org/pdf/2410.18160) - best: `3.901`

Rene - [link](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch) - best: `3.340`

Rwkv7 - [link](https://github.com/BlinkDL/RWKV-LM) - best: `4.450`

Zamba2 - [link](https://huggingface.co/Zyphra/Zamba2-2.7B) - Zamba2 > Rene > Rwkv7

Hourglass Transformer (modified) - [link](https://arxiv.org/abs/2110.13711) - Hourglass > MegaByte > FTP - best: `3.710`

Hymba - [link](https://arxiv.org/html/2411.13676v1) - train step time is significantly slower than the transformers. Best validation loss so far: `4.7505`

Tokenformer (in BlaGPT model) - [link](https://github.com/Haiyang-W/TokenFormer) - best: `3.390`

## Optimizers
PaLMForeachSOAP - [link](https://github.com/ClashLuke/HeavyBall) - almost 2 times slower than Adam but the best results

Ademamix - [link](https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch/blob/main/AdEMAMix.py) - Unstable even after trying different learning rates.

Adopt - [link](https://github.com/iShohei220/adopt) - straight up Nan

CAdamW - [link](https://github.com/kyleliang919/C-Optim/blob/main/c_adamw.py) - best: `3.3517`

AdamW with independent weight decay - [link](https://arxiv.org/pdf/2309.14322) - best: `3.320`

Adam - best: `3.3224`

AdamW - best: `3.3310`

## Best Model So Far
BlaGPT with the following configurations:

```json
{
    "params": {
      "norm_layer": "rmsnorm",
      "attention": "GQA",
      "activation": "swiglu",
      "tie_embed_weights": true,
      "zero_init_proj_layers": true,
      "use_rotary_emb": true,
      "rmsnorm_before_qk": true
    },
    "config": {
      "block_size": 1024,
      "vocab_size": 50304,
      "n_layer": 12,
      "n_head": 12,
      "n_embd": 768,
      "dropout": 0.0,
      "bias": true,
      "norm_layer": "rmsnorm",
      "attention": "GQA",
      "activation": "swiglu",
      "use_soft_logit_capping": false,
      "n_kv_head": 4,
      "tie_embed_weights": true,
      "zero_init_proj_layers": true,
      "rmsnorm_before_qk": true,
      "use_rotary_emb": true
    },
    "val_loss": 3.2993,
    "memory_usage": 49403,
  },
```

## Adding a New Model

- Implement the model
- Return the loss in the forward function
- Register the model
- And start training

See one of the implementations for details.


## Training

- Get the data by running `data/fineweb10B_cached.py`

- Start training with:

```bash
torchrun --standalone --nproc_per_node=8 train.py --run_name pre_post_norm --model_name blagpt
```

## Acknowledgements

The initial code is based on

Nano GPT - [link](https://github.com/karpathy/nanoGPT)

Modded NanoGPT - [link](https://github.com/KellerJordan/modded-nanogpt)
