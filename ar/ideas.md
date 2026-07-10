# ideas backlog (Phase 1 — config-space over implemented techniques)

Priority queue (README deltas were FIXED-STEP; time-budget re-ranks everything):

1. baseline ×3 — best config as-is → sigma, epsilon
2. engram_ngram=3 (best config uses 2)
3. attention=kvshift (KV-shifting was +win in README) on top of best
4. attention=gated (Gated Attention win) — vs xsa
5. use_parallel_blocks=True (PaLM parallel blocks — cheaper per step, time budget may flip it to win)
6. n_kv_head=12 (full MHA vs GQA4 — GQA saves VRAM, MHA may win on quality)
7. use_canon_layers=True
8. per_layer_token_emb_dim=128 vs 256 vs 512
9. engram_layers=[1..12] (all layers) vs [1..6]
10. rope_variant sweep (simplified rope was a README win)
11. use_soft_logit_capping=True (was a loss in fixed-step; cheap, retry under time)
12. learning_rate ×0.7 / ×1.4 around keeps
13. sequence_length 2048 @ device_batch 16 (same tokens/step, longer ctx)
14. activation=geglu vs swiglu
15. combos of any two keeps above

Rules: 70% exploit best, 20% explore untried, 10% wild. 5 straight discards in a family → ban family tonight.

## tried
(none yet)

## v2.1 invent-slot candidates (2026-07-10, pre-exp-14)
1. [valueembed] modded-nanogpt value embeddings: learned per-token embedding added into V at a subset of layers (U-net skip from tokens to values). Proven winner at GPT-2 scale speedruns; not in repo.
2. [averaging] EMA of model weights maintained during training, final val run on EMA weights. Strong short-horizon gains under time budget; training-loop change only, frozen invariants untouched.
3. [sink] learned attention-sink KV pair prepended per layer (streaming-LLM style). Cheap, stabilizes attention mass; not in repo.
Pick: valueembed — highest EV x novelty (directly proven at this exact scale/token budget in nanoGPT speedrun lineage).

## chain 14 postmortem (value embeddings) — 2026-07-10
- VE (modded-nanogpt style) DISCARDED: 3.4580/3.4567 (with PLTE), 3.4544/3.4544 (replacing PLTE); lambda 0.1 vs 0.5 inert in both regimes.
- Useful side-finding: PLTE-off frees 159M params and buys ~13% more steps (304 vs 332 ms/step) and VE-without-PLTE landed only +0.0117 off best → test "best minus PLTE, no VE" in an exploit slot: if PLTE contributes < its time cost, removing it alone could win.

## v2.1 invent-slot candidates (2026-07-10, pre-exp-17)
1. [averaging] EMA of model weights maintained during training (update every k steps, cost inside the clock — legal), final val on EMA weights. Proven short-horizon gains; pure train_ar.py change.
2. [unet] U-net long skips: add residual from layer i to layer n-1-i (modded-nanogpt encoder-decoder skips). Proven at this scale in speedrun lineage; new mechanism for this repo (distinct from hyper-connections).
3. [sink] learned attention-sink KV prepended per layer (carried over from pre-exp-14 set).
Pick: averaging — best EV x cost: no arch risk, no extra params, known to shave val loss on short budgets; clock-integrity clean (EMA update runs while clock ticks; only the final-val weight swap happens after).

## chain 17 postmortem (EMA eval) — 2026-07-10
- EMA DISCARDED after 3/4 runs: 3.4600 (d0.98/e5) -> 3.4575 (d0.99/e1) -> 3.4597 (d0.995/e1). Trajectory non-monotone.
- Root cause: warmdown LR collapses in final ~40% of budget. EMA with any decay < 1.0 will mix in stale high-LR weights; as decay -> 1.0 (more stable), horizon grows but still captures the rapid descent unevenly.
- Future idea: test snapshot averaging only over last K steps (uniform weight, not exponential) — exclude the descent phase entirely.

## v2.1 invent-slot candidates (2026-07-10, pre-exp-20)
1. [unet] U-net long skips: decoder layer i adds w_i * output of encoder layer (n-1-i), learnable scalar per pair. Proven in modded-nanogpt speedrun lineage.
2. [sink] learned attention-sink KV prepended per layer (carried).
3. [snapshot-avg] uniform average of last-K weight snapshots (from EMA postmortem — excludes descent phase, unlike exponential EMA).
Pick: unet — proven mechanism at this scale, cheap (scalars only), attacks gradient flow rather than token identity (orthogonal to failed VE/EMA lines).

## chain 20 postmortem (unet skips) — 2026-07-10
- DISCARDED at 3/4 but STRONGEST invent line: init curve 1.0->3.4570, 0.25->3.4446, 0.1->3.4454. Plateau = statistical tie with best (+0.002), never beats it.
- Early-training boost is real (best smoke of the day) but converts to zero net gain by 600s.
- COMBO candidate: unet(0.25) might stack with a mechanism that helps late training; also candidate if budget ever grows (helps early = helps more when budget shrinks?  test unet at 300s sometime).
- Meta-note: 3 invent chains today, knob moves within a mechanism ~0.001-0.012; mechanism CHOICE dominates. Spend run 4 only when trajectory slope > eps.

## v2.1 invent-slot #4 candidates (2026-07-10, pre-exp-25)
1. [sink] learned attention-sink: prepend 1 learnable KV pair per layer; all queries can attend to it, letting the model park "nothing useful here" attention mass. Distinct from VE (separate KV not V only) and from unet (no long-range skip).
2. [snapshot-avg] uniform weight average of last-K iterate snapshots (K=50 steps). Snapshot checkpoints saved after warmdown begins (~step 1100); final-val swaps in the average. No exponential decay bias from early high-LR weights. Different from EMA which was discarded.
3. [moe-ffn] Mixture-of-Experts on the FFN sublayer: 4 experts, top-2 routing, auxiliary load-balance loss. Well-proven quality/step tradeoff; no extra params at inference if fused. Distinct from all tried mechanisms.
Pick: moe-ffn — highest EV (proven quality gains at this scale), strong mechanism novelty, and the no-PLTE base has 14% more steps to amortize the routing overhead.
