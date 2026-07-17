# Spec: Scale-robust updates for Muon & Aurora optimizers

Status: DRAFT (not implemented). Target: bla_gpt/optimizers/{muon.py,aurora.py,__init__.py}
Context: Kimi "Muon is Scalable for LLM Training" (arXiv:2502.16982) update rule:
W_t = W_{t-1} - eta * (0.2*sqrt(max(A,B)) * O_t + lambda*W_{t-1})

## Corrected baseline assessment (from code, 2026-07-17)
- muon.py IS already Kimi-style: adjust_lr_for_muon() applies 0.2*sqrt(max(A,B)) to the
  NS5 output; decoupled weight decay exists (p.mul_(1 - lr*wd), plain lr — matches Kimi,
  which couples decay to eta, not adjusted eta).
- aurora.py is NOT RMS-matched: uses Muon-convention aspect scaling sqrt(max(1, m/n)),
  leaving update RMS ~ 1/sqrt(min(A,B)) — shape-dependent across layers.
- BUG (muon.py + __init__.py): config optimizer_args uses key "weight_decay", but
  Muon.__init__ only accepts "wd" (default 0.1) + **kwargs. The config value is silently
  swallowed by **kwargs. Consequence: ALL Muon runs (incl. best-stack F90r2/F94 lineage)
  trained with wd=0.1 despite config saying weight_decay=0.0 — on BOTH the Muon path and
  the AdamW backup (group["wd"]).

## Change 1 — muon.py: kwarg alias + drop-guard  [bugfix, P0]
- Accept weight_decay as alias: `wd = kwargs.pop("weight_decay", wd)` (explicit param
  preferred); raise ValueError on any remaining unknown kwargs instead of silently
  ignoring (**kwargs currently eats typos).
- Log/expose effective wd in repr so run.log shows the truth.
- NOTE: fixing this CHANGES BEHAVIOR of existing configs (weight_decay=0.0 will now
  actually mean 0.0, while historical runs used 0.1). Ledger comparability: next full
  run after this fix must state effective wd in its note; treat wd=0.1 as the de-facto
  baseline setting (it produced 3.1897/3.1965 lineage).

## Change 2 — muon.py: decay/update consistency cleanup  [hygiene, P1]
- Keep decay on plain lr (matches Kimi: eta*(scaled_update + lambda*W)); delete the
  "not sure if this line should use adjusted_lr" comment and document the Kimi form.
- Cautious-WD branch currently uses adjusted_lr for decay while standard branch uses lr
  — unify both on lr for consistency (flag-gated behavior change; CWD was tested only
  as discard, low risk).

## Change 3 — aurora.py: RMS matching (config-gated)  [feature, P1]
- Add param `rms_match: bool = False` to aurora() and Aurora.
- When True: replace `update *= max(1, G.size(-2)/G.size(-1)) ** 0.5`
  with `update *= 0.2 * math.sqrt(max(m, n))` (use ORIGINAL param dims A,B of the 2D
  view, matching muon.py's adjust_lr_for_muon semantics).
- When False: current behavior, bit-identical to F99 (reproducibility guarantee).
- Weight decay stays `W.mul_(1 - eta*weight_decay)` on plain eta (Kimi form; already
  consistent).
- Plumb through __init__.py aurora branch (optimizer_args passthrough already works).

## Change 4 — optional lambda schedule coupling  [experiment-only, P2]
- Inkling reportedly ties weight decay to lr^2 over the schedule. Both our optimizers
  already get first-power coupling for free (decay term eta*lambda*W scales with the
  LambdaLR multiplier). lambda ∝ eta adds second-power coupling.
- If wanted: `wd_schedule: "const" | "lr_scaled"` group option; lr_scaled multiplies
  wd by (current_lr / base_lr). Do NOT enable by default; irrelevant at 5100 steps,
  aimed at long runs.

## Validation gates (lesson 9ccbe8b1)
1. Unit: for shapes (768,768), (768,3072), (3072,768), (768,50304) assert
   update RMS in [0.15, 0.25] with rms_match=True; assert bit-identical update with
   rms_match=False vs current code.
2. Kwarg guard test: Muon(weight_decay=0.0) sets wd=0.0; Muon(bogus_kwarg=1) raises.
3. Tiny overfit: 200 steps on one shard, loss must drop monotonically-ish for both
   optimizers with new flags on.
4. Grad-flow/finite: forward/backward finite grads, one step, no NaN/Inf.
5. Short real smoke: 300-step run on the best stack before any full 5100-step run.

## Experiment plan
- F100: Aurora + rms_match=True, LR sweep. Effective magnitude change vs F99 for a
  768x3072 matrix: 0.2*sqrt(3072) ≈ 11.1 vs sqrt(3072/768)=2.0 → ~5.5x larger updates
  → start sweep at lr ∈ {3e-3, 5e-3, 1e-2} (0.03/5.5 ≈ 5.4e-3 midpoint).
- F101 (optional): Muon wd audit — best stack with true wd=0.0 vs 0.1 after the alias
  fix, to learn what the silent 0.1 has been contributing.
- Expectation management: parity at 380M/5100 steps is a PASS (these are scale fixes);
  the goal is a validated recipe for larger runs.

## Out of scope
- Distributed/sharded Muon (single-node DDP only), Newton-Schulz step-count changes,
  AdaMuon/NorMuon variants.
