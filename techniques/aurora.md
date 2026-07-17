# Aurora Optimizer

**Paper:** [Aurora: A Leverage-Aware Spectral Optimizer](https://arxiv.org/abs/2606.27715) (Tilde Research)
**Reference implementation:** [tilde-research/aurora-release](https://github.com/tilde-research/aurora-release)
**Code:** `bla_gpt/optimizers/aurora.py` (polar/aurora vendored byte-for-byte from upstream)

## What it is

Aurora is a Muon-family spectral optimizer. Like Muon, it replaces the raw
momentum-SGD update for 2D matrix parameters with an orthogonalized (polar)
update; unlike Muon, it makes the polar factor **leverage-uniform** via a
diagonal row preconditioner so that rows of the update carry equal energy for
non-square matrices.

## How it works

1. Nesterov momentum on the gradient (`mu=0.95`).
2. For square matrices: standard polar factor via a 12-step simple-quintic
   Newton-Schulz iteration (`polar()`), same family as Muon's NS5.
3. For rectangular matrices: iteratively refine a diagonal preconditioner `D`
   (`pp_iterations=2`, `pp_beta=0.5`) so `polar(D * G)` has uniform row
   leverage, then use that as the update.
4. Muon-convention aspect-ratio scaling `sqrt(max(1, m/n))`.
5. Decoupled weight decay (`weight_decay=0.025`), then `W -= eta * update`.

Non-matrix parameters (1D gains/biases) and the embedding / lm_head matrices
fall back to an internal AdamW, mirroring this repo's Muon integration.

## Usage

```json
"optimizer_name": "Aurora",
"learning_rate": 0.03,
"optimizer_args": {
  "weight_decay": 0.025,
  "mu": 0.95,
  "nesterov": true,
  "pp_iterations": 2,
  "pp_beta": 0.5
}
```

Registered in `get_optimizer()` (`bla_gpt/optimizers/__init__.py`); param split
is identical to Muon (ndim >= 2 and not embed/head -> Aurora, rest -> AdamW).

## Results (F99)

On the full best stack (XSA + GRAPE-A + U-net skips + value residual +
composable gated attention + hybrid norm + gated norm + GOAT sink prior),
swapping Muon (lr=0.0014) for Aurora (lr=0.03):

- val_loss: `3.1897` -> `3.1603` (-0.0294) — **new best**
- peak memory: `59643 MiB`; step_avg: `721ms` (vs ~580ms with Muon — the
  leverage iterations cost ~24% step time)

Note the much larger stable learning rate (0.03 vs 0.0014): the leverage-uniform
polar update tolerates aggressive eta, which is where most of the gain comes from.
LR sweep history (lost-pod runs, unverified): lr=0.05 -> 3.1572, lr=0.03 -> 3.1460,
lr=0.02 -> 3.1464 (plateau).
