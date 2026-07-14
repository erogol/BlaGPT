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

## Latest archived checkpoint

- Run: `F87c` aggregate confirmation
- Validation loss: `3.1961` at step `5100`
- Local checkpoint: `bla_gpt/logs/ar_full_F87c_0/state_step005100.pt`
- S3 checkpoint: `s3://tts-team-dev/checkpoints/blagpt/autoresearch/jul8/F87c/state_step005100.pt`
- Size: `3,039,250,572` bytes
- Uploaded: `2026-07-14 08:12:27 UTC`
