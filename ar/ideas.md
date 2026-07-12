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

## Exp 25 (MoE-FFN) postmortem
- DISCARD: 3.8149 (+0.375), 993 steps (vs 2074 dense). 2× step-cost killed it.
- Root cause: sequential top-k dispatch in Python loop over 4 experts; routing overhead dominates at 600s budget.
- LESSON: MoE needs batched/fused dispatch (e.g. torch.gather scatter-add or triton kernel) to win under this budget. Not worth retrying without a fused kernel.
- Meta: BIG-FLOP additions hurt under the 600s constraint. Mechanisms must be param-neutral or compute-neutral to compete.

## Invent-slot #5 candidates (post-exp-25)
1. [learned-lr] Per-parameter adaptive learning-rate scaling: a small MLP that reads parameter gradient stats and outputs per-layer LR multipliers. Lightweight (tiny MLP, no backward through it), affects optimizer dynamics not model compute.
2. [alibi-pos] ALiBi positional bias instead of RoPE: no positional embedding params, learned only via attention bias slopes. Tests whether RoPE is optimal for this seq length.
3. [attention-sink-token] Prepend 1 learnable "sink" token to KV cache per layer. Sink absorbs "nothing useful" attention mass. Different from VE (no value-emb, just KV sink), no extra compute per non-sink token.
Best pick for EXPLORE slot: MTP (n_predict=2) from queue — quota says explore next, and this is a direct config change on the no-PLTE best.

## Exp 26 (MTP n_predict=2) postmortem
- DISCARD 4.3863: harness final_val_loss averages ALL heads; the t+2 head is ~1.9 nats harder -> MTP cannot win under this metric BY CONSTRUCTION (not a quality signal about MTP itself). Also -37% steps, +10GB VRAM.
- Queue is now fully drained (10 z_loss, 11 softpick-deprioritized, 12 MTP all resolved).

## Exp 27 (attention sink) postmortem
- DISCARD 3.5198 (+0.08): the custom bool mask (sink col + causal) drops SDPA off the causal fast-path -> -23% steps. Sink mechanism itself may be neutral; the kernel cost is what kills it. Same lesson family as MoE: under 600s wall-clock, anything that touches the attention kernel path must keep the flash fast-path.
- Next exploit hypothesis from the winning pattern (trade capacity for steps): n_layer 12->11.

## Invent-slot #9 candidates (post-exp-43, 2026-07-10 end-of-day)
1. [gradient-clipping] Per-layer gradient-norm clipping: current training uses no grad clip or global clip; per-layer norms can stabilize early high-LR phases and transfer to real horizons. TRANSFERABLE.
2. [tied-kv] Share K and V projections (K=V projection): halves KV param count, forces the model to represent query-relevant structure in a single projection. Architecture choice that would transfer. TRANSFERABLE.
3. [local-window-attn] First N layers use local-window attention (window=256), last 11-N use full MHA: local layers are cheaper, global layers get more steps budget, AND the stack forms a natural easy→hard curriculum in attention. Same flash-safe path if window = power of 2 and is_causal stays True. TRANSFERABLE.
Pick: tied-kv — pure config change if kv_proj already exists in the codebase (n_kv_head → 1 with repeat_interleave would functionally share), zero new code, architecture significance, transfers.

## chain 54 postmortem (gradient clipping) — 2026-07-11
- grad_clip=1.0: 3.4036 / 3.4084 — run1 beats best by 0.0011, run2 misses by 0.0037. High variance, statistical tie. DISCARD.
- Signal: grad_clip direction may be real (run1 is the strongest single result in a long time) but 1.0 is too coarse — Muon outputs have different gradient scale than AdamW.
- Candidates for chain continuation:
  1. [grad_clip=0.5] Tighter clip — if Muon grads are typically < 1.0 norm, clipping at 0.5 may be more useful (hits more often, controls more updates)
  2. [grad_clip=0.3] Even tighter — typical LLM production default with Muon
  3. [rope_theta] Sweep rope_theta (currently 1e6): try 1e4 (standard) or 5e5 — RoPE theta affects how fast positional frequencies decay; may be miscalibrated for seq_len=1024
- Pick: grad_clip=0.5 — exploit chain, tighten clip value. One more run before deciding if clipping signal is real.

## chain 55 postmortem (grad_clip sweep) — 2026-07-11
- clip=1.0: 3.4036/3.4084; clip=0.5: 3.4060. All three hover 3.403-3.408.
- Variance ~0.004 nats. Clip value is inert. Signal not strong enough to land both runs below best.
- CLOSED: gradient clipping does not reliably beat the noise floor at this budget.
- Next axes: rope_theta sweep (currently 1e6, perhaps miscalibrated for seq=1024), or combo: grad_clip + another mechanism on top.
- OR: accept that 3.4047 is a near-plateau for this config and try fundamentally different structural change (e.g. deeper smaller: n_layer=13, n_embd=704; or shallower wider: n_layer=9, n_embd=832).

## DIRECTIVE from Eren (2026-07-11): ARCH vs SCHED classification
- Pure LR/schedule tweaks (warmup length, anneal shape) are NOT real findings — horizon-bound to 600s budget.
- Every keep must be classified: ARCH (transferable mechanism/architecture) or SCHED (horizon-bound schedule tuning).
- Retroactive: warmup100 + exact curriculum durations = SCHED; full-MHA, no_plte, n_layer11, curriculum-as-mechanism, len384 = ARCH.
- No further experiment slots on pure schedule knobs. Invent/explore slots target MECHANISMS only.
- Headline metric for reports = ARCH keeps progress.
- Note: device_batch/throughput knobs (exp 58) are BUDGET-ARTIFACT class, not ARCH — finish current run, classify accordingly, deprioritize similar.

## Notion BlaGPT paper audit (2026-07-11, per Eren directive)
IMPLEMENTED already (skip):
- [x] Hyper-Connections (2409.19606) — bla_gpt.py hyper_num_streams
- [x] Value Residual Learning (2410.17897) — resformer.py
- [x] PolyCom activations (2411.03884) — polycom_order in config
- [x] Cautious Weight Decay (2510.12402) — use_cautious_weight_decay
- [x] AdaMuon (2507.11005) — optimizers/adamuon.py
- [x] Exclusive Self Attention (2603.09078) — xsa, CURRENT BEST attention
NOT implemented (invent-slot candidates, minimal config-gated impl):
- [ ] Attention Residuals (Kimi/MoonshotAI PDF) — HIGH PRIORITY: lightweight, attention-output residual
- [ ] Polar Coordinate PE / PoPE (2509.10534) — pos_encoding registry candidate
- [ ] NAG Norm-AGnostic Residual (Zyphra, X post) — residual rescaling scheme
- [ ] Aurora optimizer (Tilde blog) — leverage-aware for rectangular matrices, optimizer registry
- [x] Tapered Language Models (2606.23670) — IMPLEMENTED (F72, config-gated `use_tapered_mlp`, default off); see F72 audit below
- [ ] Better Attention Priors (2601.15380) — read first
- [ ] Unified Attention/Residual Sinks (2601.22966) — outlier-driven rescaling
- [ ] Lipschitz-enforced training (2507.13338) — constraint method
Out of scope for 600s harness: CARD diffusion LM, ConceptMoE, CALM continuous AR, byte-level U-Net LMs (tokenizer swap), superpowers (not a paper)
Infra unblocks pending: fla + flash_attn pip install running (/tmp/pip_install.log); forgetting_attn mask=None bug fix-and-retest

## chain 62 postmortem (Kimi Attention Residuals) — 2026-07-11
- KEEP, NEW BEST 3.3915 (3.3915/3.3920 confirmed). First Notion-queue paper pays off.
- Impl lesson: naive torch.stack over layer outputs cost -477 steps +10GB; loop-based accumulation (stack scores only) recovered it. Autograd-retained big stacks are the killer at this budget.
- Mechanism: per-layer learnable query w_l, softmax over RMSNorm-ed layer deltas (keys=values), aggregate as layer input. v0=embedding.
- Notion audit: mark Attention Residuals [x] IMPLEMENTED+KEPT.
- Remaining steps deficit vs non-AttnRes (2555 vs 2841): Block AttnRes (paper) could recover more — future exploit candidate.


## chain 71 postmortem (depth/width rebalance) — 2026-07-11
- KEEP, CONFIRMED NEW BEST 3.3750 (3.3775/3.3750), versus prior best 3.3836.
- Mechanism: reduce from 11 to 10 transformer layers while retaining MLP expand=10; saves 14M params and buys ~145 extra steps under the fixed wall-clock budget.
- Classification: ARCH — depth/width allocation is transferable, not a schedule knob.
- Next: return to invention queue; no more pure shape sweeps until a new mechanism is tested.


## F72 audit — Tapered Language Models (arXiv:2606.23670), paper read 2026-07-12
- Paper: "Tapered Language Models", Reza Bayat, Ali Behrouz, Aaron Courville (2026). Read the arXiv abstract + HTML body (not just the title).
- Mechanism (paper Eq. 5): d_ff(l) = d_end + (d_start - d_end)/2 * (1 + cos(pi*l/(L-1))). Default endpoints d_start/d_end = 1.5/0.5 x baseline width, which reduces to d_ff(l) = base*(1 + 0.5*cos(pi*l/(L-1))). Earlier layers wider, later layers narrower. Tapering applies to the MLP intermediate dim ONLY (model width d, head count, KV dim unchanged).
- Budget constraint (paper Eq. 7): (1/L) * sum_l d_ff(l) = base -> total MLP params/FLOPs identical to the uniform baseline ("free lever, no extra params/compute"). Paper rounds each width to the nearest multiple of 16, pins the first/last layers to d_start/d_end, and nudges interior widths in 16-unit steps to satisfy the average exactly while staying monotonically decreasing.
- Repo adaptation (smallest native diff): `tapered_mlp_dims()` in bla_gpt/mlps.py; gated by GPTConfig.use_tapered_mlp (default False). Alignment uses the REPO's own width granularity (multiple of 64, as used for vocab/n_embd) rather than the paper's 16; endpoints pinned to the aligned 1.5x/0.5x base, interior nudged in 64-unit steps so the aggregate == n_layer*base EXACTLY and widths stay monotone non-increasing. Only the Primer MLP path (the F72 baseline activation) is tapered; enabling taper with a non-primer activation raises. Default-off => byte-identical model construction when the gate is off.
- F72 config = combined_keeps full baseline + use_tapered_mlp=true (single added key, no other change). base_d_ff = mlp_expand*n_embd = 10*768 = 7680, L=10 -> per-layer widths [11520, 11264, 10624, 9600, 8320, 7040, 5760, 4736, 4096, 3840], sum = 76800 = 10*7680 (budget preserved).
- Class: ARCH (depth-aware capacity allocation is a transferable architecture change, not a schedule knob).
- Full result: F72 completed all 5100 steps from random init via normal train.py; final val_loss 3.2372 vs baseline 3.2354 (+0.0018). Decision: DISCARD. No confirmation required because it did not improve. Checkpoint: bla_gpt/logs/ar_full_F72_0/state_step005100.pt.

## F73 (GOAT sink prior) — "You Need Better Attention Priors", Litman & Guo (2026, arXiv:2601.15380)
- Mechanism: per-head key-only log-prior u added to XSA attention logits before softmax: softmax(qk^T/sqrt(d) + u(j)); u(j) is non-zero only at j=0, one scalar per head, initialized to zero. No Fourier component.
- Adaptation: GOATSinkAttention(ExclusiveSelfAttention) in bla_gpt/attentions.py; gate GPTConfig.use_goat_sink_prior, default false. F73 equals combined-keeps baseline plus use_goat_sink_prior=true.
- Smoke: PASS — 60s train_ar.py, 191 steps, finite loss/gradients; smoke metric excluded from full ledger.
- Full result: completed all 5100 steps from random init via normal train.py; final val_loss 3.3286 vs baseline 3.2354 (+0.0932). Decision: DISCARD. No confirmation required. Checkpoint: bla_gpt/logs/ar_full_F73_0/state_step005100.pt.
