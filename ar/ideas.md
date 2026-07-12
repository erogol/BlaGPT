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

## DIRECTIVE from owner (2026-07-11): ARCH vs SCHED classification
- Pure LR/schedule tweaks (warmup length, anneal shape) are NOT real findings — horizon-bound to 600s budget.
- Every keep must be classified: ARCH (transferable mechanism/architecture) or SCHED (horizon-bound schedule tuning).
- Retroactive: warmup100 + exact curriculum durations = SCHED; full-MHA, no_plte, n_layer11, curriculum-as-mechanism, len384 = ARCH.
- No further experiment slots on pure schedule knobs. Invent/explore slots target MECHANISMS only.
- Headline metric for reports = ARCH keeps progress.
- Note: device_batch/throughput knobs (exp 58) are BUDGET-ARTIFACT class, not ARCH — finish current run, classify accordingly, deprioritize similar.

## Notion BlaGPT paper audit (2026-07-11, per owner directive)
IMPLEMENTED already (skip):
- [x] Hyper-Connections (2409.19606) — bla_gpt.py hyper_num_streams
- [x] Value Residual Learning (2410.17897) — resformer.py
- [x] PolyCom activations (2411.03884) — polycom_order in config
- [x] Cautious Weight Decay (2510.12402) — use_cautious_weight_decay
- [x] AdaMuon (2507.11005) — optimizers/adamuon.py
- [x] Exclusive Self Attention (2603.09078) — xsa, CURRENT BEST attention
NOT implemented (invent-slot candidates, minimal config-gated impl):
- [ ] Attention Residuals (Kimi/MoonshotAI PDF) — HIGH PRIORITY: lightweight, attention-output residual
- [x] Polar Coordinate PE / PoPE (2509.10534) — PREPARED (F78, config-gated `pos_encoding=pope`, default off); see F78 audit below
- [x] NAG Norm-AGnostic Residual (2606.16112) — PREPARED (F80, config-gated `use_nag_residual`, default off); see F80 (NAG) audit below
- [x] Aurora optimizer (arXiv:2606.27715, Tilde Research) — PREPARED (F73, config-gated `optimizer_name=Aurora`, default off); see F73 audit below
- [x] Tapered Language Models (2606.23670) — IMPLEMENTED (F72, config-gated `use_tapered_mlp`, default off); see F72 audit below
- [ ] Better Attention Priors (2601.15380) — read first
- [ ] Unified Attention/Residual Sinks (2601.22966) — outlier-driven rescaling
- [ ] Lipschitz-enforced training (2507.13338) — constraint method — DEPRIORITIZED: robustness-focused, no val-loss transfer story (paper's 145M Lipschitz model reaches 21% acc vs 39.4% NanoGPT baseline; matching the baseline needs Lipschitz bound ~10^264). Not competitive for this val-loss harness.
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


## F74 (GOAT sink prior) — "You Need Better Attention Priors", Litman & Guo (2026, arXiv:2601.15380)
- Mechanism: per-head key-only log-prior u added to XSA attention logits before softmax: softmax(qk^T/sqrt(d) + u(j)); u(j) is non-zero only at j=0 (sink position), one scalar per head, initialized to zero. No Fourier component.
- Adaptation: GOATSinkAttention(ExclusiveSelfAttention) in bla_gpt/attentions.py; gate: GPTConfig.use_goat_sink_prior (default False); F74 = canonical combined-keeps baseline plus use_goat_sink_prior=true only.
- Full result: F74 completed all 5100 steps from random init via normal train.py; final val_loss 3.2352 vs baseline 3.2354 (-0.0002). Status: CONFIRMATION PENDING because improvement is <0.003. Checkpoint: bla_gpt/logs/ar_full_F74_0/state_step005100.pt. F75 will rerun the identical config from random init.


## F73 (Aurora optimizer) — "Aurora: A Leverage-Aware Spectral Optimizer", Dewulf, Pai, Yang, Zhang, Keigwin (2026, arXiv:2606.27715; official code github.com/tilde-research/aurora-release)
- NOTE ON LABEL: this Aurora F73 supersedes the abandoned uncommitted GOATSink "F73" draft above for the F73 full-run slot (`ar/full_runs/F73/`). The GOATSink working-tree changes are left untouched, not part of this change.
- Read the actual paper abstract + the official reference source (src/aurora.py, src/polar.py, README) — NOT just the blog title.
- Problem it fixes: Muon's update is polar(G)=U V^T (all singular values -> 1), but for TALL/rectangular matrices (MLP up/down projections) polar(G) has highly non-uniform left-singular ROW norms. Some rows (neurons) receive persistently tiny updates and stop contributing -> a self-reinforcing dead-neuron loop. Naive row-normalization fixes uniformity but pushes the update off the momentum matrix's polar factor (harmful). Aurora enforces row-uniformity WHILE staying on the polar geometry; reported gains over Muon GROW with the MLP expansion factor (this repo runs mlp_expand=10, so Aurora is well-matched).
- Mechanism / exact update (vendored byte-for-byte):
  1. Nesterov SGD-momentum: m <- lerp(m, G, 1-mu) = mu*m + (1-mu)*G; update = lerp(G, m, mu) = (1-mu)*G + mu*m (nesterov; else m.clone()).
  2. Leverage-uniform polar: if square (m==n) -> update = polar(update) (reduces to Muon). Else transpose wide->tall, set target_row_sq = n/m, D = 1/rownorm(G); for k in range(pp_iterations): U = polar(D*G); if not last: row_sq = sum_j U^2; D <- D * (target_row_sq/row_sq)^pp_beta. Diagonal preconditioner D drives every output-row norm of the polar factor to the same value (projection onto the intersection of the row-oblique and Stiefel manifolds) without leaving the polar factor.
  3. Spectral aspect-ratio scaling (Muon convention): update *= max(1, m/n)^0.5.
  4. Decoupled weight decay then apply: W *= (1 - eta*wd); W -= eta*update.
  polar(): 12-step simple-quintic Newton-Schulz p(s)=2s-1.5s^3+0.5s^5 (fixed points {0,1,sqrt2}, s=1 super-attracting), bf16; matches modded-nanogpt track-3 baseline byte-for-byte.
  Paper/reference defaults: eta=0.05, weight_decay=0.025, mu=0.95, nesterov=True, pp_iterations=2, pp_beta=0.5, eps=1e-7.
- Repo adaptation (smallest faithful diff): `bla_gpt/optimizers/aurora.py` vendors `polar()` and `aurora()` verbatim from aurora-release, plus `Aurora(torch.optim.Optimizer)` that MIRRORS `optimizers/muon.py`: 2D matrix params (ndim>=2, excluding embed/lm_head) -> `aurora()` with a per-param momentum buffer; 1D params + embedding + lm_head -> an internal AdamW backup identical to muon.py's. Uses a 2D view so any >2D matrix reduces to (rows, -1) like Muon. Registry: added `elif optimizer_name.lower()=="aurora"` in `optimizers/__init__.py`, mirroring the muon branch's exact param split, returning `Aurora(lr, aurora_params, adamw_params, **optimizer_params)`. Default-off => Aurora is constructed ONLY when optimizer_name==aurora; every existing branch and the Muon baseline path are byte-unchanged.
- LR CALIBRATION FLAG (launch-time): Muon in this repo multiplies lr by `adjust_lr_for_muon` = 0.2*sqrt(max(A,B)) (large effective LR), and the baseline `learning_rate` default (0.001) is tuned to that. Aurora applies `eta = learning_rate` DIRECTLY (only the aspect-ratio sqrt scale, no per-matrix sqrt(dim) blow-up), so eta=0.001 is far below the paper's eta=0.05. Per the "only required Aurora fields / no default changes" scope, `learning_rate` was NOT changed here — this is a KNOWN pre-launch calibration item (raise eta toward ~0.05, or add a lr override) before F73 is actually run.
- F73 config = `ar/best_config.json` (canonical combined-keeps baseline) with ONLY two changes: optimizer_name "Muon"->"Aurora" and optimizer_args -> {weight_decay:0.025, mu:0.95, nesterov:true, pp_iterations:2, pp_beta:0.5} (paper defaults; eps uses class default 1e-7). All model/arch keys identical to baseline; learning_rate untouched.
- Class: OPT (leverage-aware spectral optimizer for rectangular matrices — a transferable optimizer mechanism, not a schedule knob).
- Tests: `tests/test_aurora_optimizer.py` — default-off registry unchanged (Muon still built), Aurora registry construction + param split, finite one/multi-step updates on rectangular (tall+wide) and vector params, state_dict roundtrip, tiny forward/backward through get_optimizer. No debug metric used.
- Full result: F73 completed all 5100 steps from random init via normal train.py; final val_loss 3.3286 vs baseline 3.2354 (+0.0932). Decision: DISCARD. No confirmation required because it did not improve. Checkpoint: bla_gpt/logs/ar_full_F73_0/state_step005100.pt.



## F76 (PreAffineRMSNorm) -- "A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for Transformer Training", Qiu et al. (2026, arXiv:2601.22966)
- Paper: Qiu et al. show that training instability (attention/residual sinks, outlier activations) shares a single root cause: unbounded pre-norm activations that grow into a small number of extreme-magnitude channels. Outlier-driven rescaling -- applying a trainable per-feature affine before each norm -- allows the model to adaptively suppress or amplify channels before normalisation, keeping norms well-conditioned throughout training.
- Mechanism (paper Sec.3.3, exact equation): PreAffineRMSNorm(x) = RMSNorm(lambda1 odot x), where lambda1 in R^d is a trainable elementwise vector initialised to ones. Applied before each RMSNorm site so the model can rescale input channels prior to normalisation. At init (lambda1 = 1) output is identical to plain RMSNorm.
- Adaptation: PreAffineRMSNorm class added to bla_gpt/norms.py; wrapped via the existing get_norm(config) factory in bla_gpt/bla_gpt.py -- no other call sites changed. Gate: GPTConfig.use_pre_affine_norm (bool, default False). When off, get_norm returns plain RMSNorm unchanged. When on, every norm site built through get_norm (ln_1, ln_2, ln_f in Block; ln_3/ln_4 if use_pre_post_norm) becomes PreAffineRMSNorm. QK norms in attentions.py (different dimension, per-head) are not in scope; they are not constructed through get_norm. Scope: F76 = F74c config + use_pre_affine_norm: true only.
- Class: ARCH (per-feature affine before each norm -- transferable parameter-efficient mechanism; lambda1 adds n_embd params per norm site, ~3 norm sites per layer).
- Tests: tests/test_pre_affine_norm.py -- gate-off returns RMSNorm; gate-on returns PreAffineRMSNorm; lambda1 init=ones; ones-lambda1 matches plain RMSNorm under copied weights; non-uniform lambda1 changes output; gradients reach lambda1; shape/dtype preserved; state_dict roundtrip; full GPT gate-off norms are RMSNorm / gate-on norms are PreAffineRMSNorm (9 tests, all CPU, all pass).
- Config: ar/full_train/configs/F76_pre_affine_norm.json -- byte-for-value copy of ar/full_runs/F74c/config.json with only use_pre_affine_norm: true added.
- Status: PENDING SMOKE RUN (not yet trained).
- Full result: F76 completed 5100 steps from random init; val_loss 3.2405 vs confirmed best 3.2298 (+0.0107). Decision: DISCARD. No confirmation required. Checkpoint: bla_gpt/logs/ar_full_F76_0/state_step005100.pt.


## F77 (GatedNorm) -- "A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for Transformer Training", Qiu et al. (2026, arXiv:2601.22966)
- Paper: Qiu et al. 2026 (same paper as F76/PreAffineRMSNorm). Sec.3.4 proposes a complementary post-norm gating mechanism to adaptively suppress outlier activations in the normalized output stream.
- Mechanism (paper Sec.3.4, exact equations):
    y  = RMSNorm(x)
    yg = sigmoid(W_up(swish(W_down(y))))    -- W_down in R^{d x r}, W_up in R^{r x d}, rank r=16
    y' = yg odot y
  The low-rank bottleneck (rank r << d) bounds the parameter cost to 2*d*r per norm site (~6K params at d=768, r=16) while allowing per-feature output gating. Not combined with PreAffineRMSNorm (Sec.3.3).
- Initialization (paper-faithful near-identity with stable gradient flow): W_up initialized to zeros => gate=sigmoid(0)=0.5 at step 0 (half-identity, not exact 1.0 -- exact identity would require pre-sigmoid >> 0 which kills gradients). W_down uses PyTorch default Kaiming-normal, so swish(W_down(y)) is non-zero for typical y => gradients reach W_up from step 1. W_down gradients are zero at step 0 (W_up=0) but non-zero once W_up updates; mirrors standard LoRA-style bottleneck init (B=0, A=Kaiming). This is the cleanest paper-faithful near-identity init that preserves gradient flow at every parameter.
- Adaptation: GatedNorm class added to bla_gpt/norms.py; wrapped via get_norm(config) factory in bla_gpt/bla_gpt.py. Gates: GPTConfig.use_gated_norm (bool, default False), GPTConfig.gated_norm_rank (int, default 16). When off, get_norm is byte-identical to before. Applied to all sites built through get_norm (ln_1, ln_2, ln_f; ln_3/ln_4 if use_pre_post_norm). Not combined with use_pre_affine_norm. Parent config: F74c (GOAT retained).
- Class: ARCH (post-norm output gating -- transferable mechanism; 2*d*r params per norm site per layer).
- Tests: tests/test_gated_norm.py -- gate-off returns RMSNorm; gate-on returns GatedNorm; W_up init=zeros; gate=0.5 at init; output changes after W_up perturbed; gradients reach W_down (with non-zero W_up); gradients reach W_up (Kaiming W_down ensures non-zero swish); shape/dtype preserved; state_dict roundtrip; full GPT gate-off all norms are RMSNorm; full GPT gate-on all norms are GatedNorm (11 tests, all CPU, all pass).
- Config: ar/full_train/configs/F77_gated_norm.json -- byte-for-value copy of ar/full_runs/F74c/config.json with only use_gated_norm: true and gated_norm_rank: 16 added.
- Status: COMPLETE — KEEP. Full 5100-step run from random init reached val_loss 3.2230, improving on confirmed best F74c=3.2298 by 0.0068. No confirmation rerun required because improvement is >=0.003. Checkpoint: bla_gpt/logs/ar_full_F77_0/state_step005100.pt.


## F78 (PoPE — Polar Coordinate Positional Embedding) — "Decoupling the 'What' and 'Where' With Polar Coordinate Positional Embeddings", Gopalakrishnan, Csordas, Schmidhuber & Mozer (2026, arXiv:2509.10534v3)
- Paper read: arXiv HTML v3 body, Eqs. 3-10 (not just the abstract). PoPE removes RoPE's "what/where" confound: RoPE ties a feature's contribution to BOTH its content (dot-product magnitude) and relative position (rotation) so they interfere. PoPE decouples them — magnitude carries content, phase carries position.
- Mechanism (exact, paper Eqs. 3-8):
  - Eq.3 magnitudes: mu = softplus(x), softplus(x)=ln(1+e^x), applied to each real q/k feature -> nonnegative "content" magnitude.
  - Eq.4 frequencies: d frequencies (RoPE uses d/2), geometric ladder theta_c = rope_theta^(-(c-1)/d), c=1..d. Phase = position*frequency (phi_q = t*theta_c on the query at position t, phi_k = s*theta_c on the key at position s); no content term enters the phase.
  - Eqs.5-8 score identity: a_PoPE(t,s) = Re[q~_t^H k~_s] = sum_c mu_q mu_k cos((s-t) theta_c + delta_c). Each feature in Cartesian form x=mu*cos(phi), y=mu*sin(phi); the real part of the Hermitian product is the plain dot product of the doubled vectors [mu*cos, mu*sin], so standard attention (flash/SDPA or manual) over 2d-wide q/k reproduces the identity exactly. v and the output dimension are unchanged (only q/k are doubled).
  - Phase bias delta_c: per-head learnable KEY-phase bias, HARD-CLAMPED to [-2*pi, 0] (paper uses a min/max clamp, NOT a sigmoid), zero-initialized (the paper's length-generalization choice; Uniform(-2pi,0) is the slightly-better-in-distribution alternative, not used). Query phase has no delta; cos is even so cos((t-s)theta - delta) == cos((s-t)theta + delta).
- Repo adaptation (smallest native diff, config-gated default-off): implemented as a new pos_encoding option in the BASE Attention class (bla_gpt/attentions.py) — a "pope" branch in __init__ (registers pope_freqs buffer + pope_delta = nn.Parameter(zeros(n_kv_head, head_dim))), a dispatch branch in the base forward (elif use_pope -> _apply_pope), and _apply_pope() returning the doubled Cartesian q/k. Because it lives in the base class it composes automatically with the F77 attention stack: attention="xsa" + use_goat_sink_prior=true resolves to GOATSinkAttention (an Attention subclass that does NOT override forward), so it inherits PoPE for free while keeping XSA exclusion + GOAT sink prior + GatedNorm. pos_encoding="pope" is the SOLE config gate; wpe (absolute position embedding) is disabled for "pope" exactly as for rotary. The prior exp-68-prep standalone PoPEAttention class + attention="pope" dispatch (which could NOT compose with XSA/GOAT and had an unclamped delta) were removed as superseded.
- Numerical faithfulness: q2.k2 over the 2d dim verified equal to the Eq.6 sum_c softplus(q_c)softplus(k_c)cos((s-t)theta_c+delta_c) to atol 1e-4 (float64 reference), with mixed-sign/out-of-range delta exercising the clamp.
- Class: ARCH (positional-encoding mechanism; transferable — the what/where decoupling and d-frequency phase encoding is horizon-relevant, and the paper reports strong zero-shot length extrapolation).
- Tests: tests/test_pope_f78.py (12 CPU tests, all pass): default rotary path unchanged + reproducible; PoPE construction attributes; d-frequency count (not d/2) + exact ladder; delta zero-init; delta clamped to [-2pi,0] (upper & lower bound); numerical attention-logit equivalence to Eq.6; output shape (v/out dim preserved) + gradient (incl. pope_delta); unknown pos_encoding still rejected; PoPE composes with plain XSA and with GOAT-sink XSA; full GPT (XSA+GOAT+PoPE) forward/backward with wpe disabled. Additionally validated end-to-end that the F78 config's full stack (XSA+GOAT+GatedNorm+PoPE) constructs and runs forward/backward on CPU (activation swapped to gelu to sidestep the pre-existing Primer-MLP cuda device pin, orthogonal to PoPE; the real run is on GPU).
- Config: ar/full_train/configs/F78_pope.json — byte-for-byte copy of ar/best_config.json (F77 canonical best: XSA + GOAT sink + GatedNorm rank16, Muon, rope_theta 1e6) with ONLY pos_encoding "rotary" -> "pope" changed (verified single-line diff). rope_theta (1e6) is reused as the PoPE frequency base.
- Status: COMPLETE — DISCARD. Full 5100-step run from random init reached val_loss 3.2299 vs best F77=3.2230 (+0.0069). No confirmation required because not lower. Checkpoint: bla_gpt/logs/ar_full_F78_0/state_step005100.pt.
- Notion: mark Polar Coordinate PE / PoPE [x] IMPLEMENTED/PREPARED (F78).

## F75 (aborted)
- F75 was the first attempt at the F74 confirmation rerun (identical config, random init). It received SIGHUP ~60s after launch on 2026-07-12 03:02 UTC and produced no final_val_loss. Superseded by F74c, which completed the confirmation (3.2298). Artifacts kept in ar/full_runs/F75/ for audit; nothing enters the ledger.


## F79 (GOAT relative spectral prior) -- "You Need Better Attention Priors", Litman & Guo (2026, arXiv:2601.15380)
- Paper: same paper as the kept F74/F74c key-only sink prior. Sec. 6 proposes the second GOAT component: a learnable, continuous, translation-equivariant relative log-prior added to the attention logits, K_rel(i,j) = sum_r [alpha_{h,r} cos(w_r (i-j)) + beta_{h,r} sin(w_r (i-j))], with R fixed geometric frequencies and per-head learnable spectral coefficients. Paper motivation: EOT view -- standard attention implicitly assumes a uniform transport prior; a learned continuous prior absorbs spatial structure into attention while keeping the length generalization of fixed encodings (extrapolatable prior; combines learned-PE flexibility with fixed-PE extrapolation).
- Adaptation: GOATRelPriorAttention(GOATSinkAttention) in bla_gpt/attentions.py. Realized via the additive attn_mask (the exact same SDPA mask path the kept F74 sink prior uses, proven at ~356-466 ms/step in F77/F78 runs) -- mathematically identical to the paper's composite-vector FlashAttention embedding (Eqs. 20-24), which just realizes K_ij as an additive logit term. Frequencies: geometric ladder w_r = rope_theta^(-r/(R-1)), R=8, base rope_theta=1e6 (repo convention). alpha/beta zero-init => byte-identical to GOATSinkAttention at step 0 (repo near-identity init convention). Gates: GPTConfig.use_goat_rel_prior (default False), GPTConfig.goat_rel_num_freqs (default 8). Gate-off dispatch is unchanged (xsa -> GOATSinkAttention when use_goat_sink_prior).
- Class: ARCH (attention prior; 2*H*R params total, ~192 at H=12, R=8 -- negligible).
- Transfer story: translation-equivariant prior is horizon-relevant (paper reports length extrapolation); complements the kept sink prior from the same EOT decomposition (Fig. 1: K = K_sink + K_rel).
- Tests: tests/test_goat_rel_prior_f79.py (12 CPU tests, all pass): gate-off dispatch unchanged; gate-on dispatch; frequency-ladder values; zero rel-prior at init; zero-init output identity with GOATSinkAttention; translation equivariance of K_rel; alpha and beta perturbations change output; gradients reach alpha/beta/sink; state_dict roundtrip (non-persistent freq buffer excluded); shape/dtype; full GPT forward/backward on the F79 stack (XSA+GOAT sink+rel+GatedNorm). frozen_check: OK.
- Config: ar/full_train/configs/F79_goat_rel_prior.json -- byte-for-value copy of ar/best_config.json (F77 canonical best) with ONLY use_goat_rel_prior: true and goat_rel_num_freqs: 8 added (verified two-key diff).
- Status: COMPLETE -- DISCARD. Full 5100-step run from random init (launch commit ff3e595) reached val_loss 3.2257 vs confirmed best F77=3.2230 (+0.0027). Very close but not lower, so discarded per v3 protocol (keep iff strictly lower). Checkpoint: bla_gpt/logs/ar_full_F79_0/state_step005100.pt. Note: both GOAT components now tested -- sink prior kept (F74/F74c), relative spectral prior discarded at this scale/horizon.
- Notion: mark Better Attention Priors [x] IMPLEMENTED (F74 sink component + F79 relative component).


## F79 (NAG — Norm-Agnostic Residual) — "Scaling Adaptive Depth with Norm-Agnostic Residual Networks", Figliolia & Millidge (Zyphra, 2026, arXiv:2606.16112)
- Paper read: extracted from the arXiv PDF (pdftotext), Sec. III-A "The Norm-Agnostic Residual Stream Network", Eqs. 3-24 (not the abstract). Core claim: because the residual stream is updated additively, its norm grows with depth (up to 120x vs a norm-agnostic model in the paper's ablations); later layers must emit ever-larger updates just to keep the same relative effect, so their contribution is systematically suppressed (an "inhibitory" norm-dependent decay). NAG makes each layer's contribution independent of the current residual norm by separating magnitude from direction. We implement ONLY the core residual formulation; the Mixture-of-Depths part is out of scope (fixed-depth 10-layer harness).
- Mechanism (exact, paper Eqs. 3-15, implemented as the drop-in Eq. 14):
  - Eq.3-5: scale-only input normalization. R_bar_l = N_in(R_l) = sqrt(d) R_l / ||R_l||, and the scalar norm rho_l = ||R_l||_2 / sqrt(d), so R_l = rho_l R_bar_l (||R_bar_l|| = sqrt(d)).
  - Centering: f_l(R_bar_l) is the layer (attn/MLP) output with its feature-wise mean subtracted (paper: centering is the last op inside f_l; improves numerical stability).
  - Eq.10-11: orthogonalize the layer output against the residual direction, f_perp = f - (<f, R_bar>/||R_bar||^2) R_bar (||R_bar||^2 = d). Parallel writes only inflate the norm without rotating direction, so they are removed — each layer acts primarily as a controlled rotation.
  - Eq.12: output normalization N_out(x) = sqrt(d) x / ||x||, applied to f_perp so the orthogonal update keeps norm sqrt(d) even when it is initially tiny (early training f is highly correlated with R_bar).
  - Eq.13: norm modulator m_l = ( sum_{i=0}^{C-1} p_li * sigma(R_bar_l . w_li + b_li) )^{beta_l} in [0,1]. C learned preferred directions w_li (with biases b_li), p_li = softmax(learnable logits) (convex combination => in [0,1]), sigma = sigmoid, beta_l>0 sharpness. Acts as a lightweight input-dependent gate (MoE-router analogue).
  - Eq.14 (the full norm-agnostic update, what we implement): R_{l+1} = rho_l R_bar_l + rho_l alpha_l m_l(R_bar_l) N_out(f_perp). Since rho_l R_bar_l == R_l == x exactly, this is the drop-in residual add x + rho_l * alpha_l * m_l * N_out(f_perp). The added term is orthogonal to the residual direction with norm ||x|| * alpha_l * m_l, so the per-layer norm gain is sqrt(1 + alpha_l^2 m_l^2) (Eq.15) — independent of the absolute residual norm (the norm-agnostic property). Eqs.16-17 (log-space norm lane s_l = log rho_l) and Eq.18 (decoding-temperature head) are an equivalent numerical reparametrisation of Eq.14 / an unembedding change; not needed for the core fixed-depth residual and omitted.
  - Eq.23-24: depth-scaled init alpha_l = 1/l^p (l = 1-based NAG-layer index), default p=0.5 => alpha_l = 1/sqrt(l), matching the additive orthogonal-growth per-layer gain. p is exposed (nag_init_p) per Eq.24.
- Repo adaptation (smallest native diff, config-gated default-off): new NAGResidual(nn.Module) in bla_gpt/bla_gpt.py (immediately before Block), implementing Eq.14. Block.__init__ builds one NAGResidual per sub-layer when config.use_nag_residual (attn = NAG-layer 2*depth+1, mlp = 2*depth+2, 1-based; matches "each transformer layer = attention + MLP", both treated as NAG layers per the paper's "we apply the norm-agnostic formulation to all layer types"). Block._process_branch gains an optional nag= arg: when present it returns nag(x, branch_out) instead of the additive x + branch_out (or the res_weight variant). When use_nag_residual is False, nag_attn/nag_mlp are None, no NAG params are created, and the additive path is byte-identical to before. The Block subclasses that override _process_branch (ParallelBlock, CanonBlock) keep their own signatures and are unaffected; NAG is wired into the base Block path, which is the canonical stack.
- Composition with use_attn_res (Kimi attention-residuals, active in best config): no interaction in code. use_attn_res operates at the GPT.forward level — it reads each block's delta (block(x) - x) and softmax-mixes those deltas across layers; NAG only changes HOW that delta is produced inside the block. NAG makes each branch's contribution norm-agnostic w.r.t. the current (recombined) stream; use_attn_res decides how block contributions are weighted. They compose cleanly and are verified together in the full-stack test. res_weight (use_res_weights) is bypassed when NAG is on (NAG defines its own update); best config has use_res_weights=false, so no live conflict.
- Adaptation notes / deviations from paper (documented): beta is a fixed hyperparameter (nag_beta=1.0) rather than a trainable per-layer beta_l — the paper's per-layer beta matters mainly for MoD, which we omit. Embedding-table centering and the Gencode/Gembedding gain lanes (Eq.18 decode temperature) are not implemented (unembedding-side, out of core-residual scope). N_in is scale-only (no R_bar centering) exactly as Eqs.3-5.
- Class: ARCH (residual-stream mechanism; transferable, ~C*d params/sublayer for the modulator + 1 alpha, negligible vs d^2; the paper reports the largest gains in the deepest models and kernel-fusible scalar-per-token ops).
- Tests: tests/test_nag_residual_f80.py (17 CPU tests, all pass): default config nag off; gate-off Block has no NAG modules and full GPT has no nag params (byte-identical-off); NAG construction + param shapes/init (b,p_logits zero, w~N(0,0.02)); Eq.14 recomputed independently in float64 and matched (atol 1e-6); paper invariants — update orthogonal to residual (Eq.11), modulator in [0,1] (Eq.13), norm-gain sqrt(1+alpha^2 m^2) (Eq.15), relative update = alpha*m (norm-agnostic); depth-scaled alpha init 1/sqrt(l) and 1/l^p (Eq.23/24); near-identity controlled rotation at init; output shape/dtype (fp32+fp64); gradients reach every NAG param (alpha,w,b,p_logits); full canonical-best stack (XSA + GOAT sink + GatedNorm rank16 + AttnRes + NAG) forward/backward on CPU (activation swapped primer->gelu to dodge the Primer-MLP cuda pin, as test_pope_f78); NAG composes with AttnRes delta-capture; config validation (nag_num_directions>=1, nag_beta>0, layer_index>=1 raise ValueError). Regression: tests/test_gated_norm.py 11/11 pass. Total 28/28.
- Config: ar/full_train/configs/F80_nag.json — byte-for-byte copy of ar/best_config.json (F77 canonical best: XSA + GOAT sink + GatedNorm rank16, AttnRes, Muon, rope_theta 1e6) with ONLY the NAG gate fields added (use_nag_residual: true, nag_num_directions: 32, nag_beta: 1.0, nag_init_p: 0.5); verified 5-line diff, nothing else changed.
- Status: READY FOR FULL RUN (not yet trained). No smoke run performed.
- Notion: mark NAG Norm-AGnostic Residual [x] PREPARED (F80, config-gated `use_nag_residual`, default off).
