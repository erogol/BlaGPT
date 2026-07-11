# Combined autoresearch winner — full training

Confirmed keeps consolidated from experiments 4, 9b, 21b, 28b/71b, 31d,
32b, 35b, 61, 62c, 69b and 71b:

- full MHA (`n_kv_head=12`)
- RoPE theta 1e6
- no per-layer token embeddings
- 10 layers
- Primer/ReLU² MLP with expansion 10
- Kimi Attention Residuals
- sequence curriculum: first 1200 steps at length 384, then 1024
- LR warmup 100 steps (short-horizon/budget-artifact keep)

The full run uses normal `train.py`: 5100 iterations, global batch 512,
per-GPU batch 32 on 8 GPUs, sequence length 1024, validation every 125 steps,
and checkpointing at step 5000 plus final/best checkpoints.
