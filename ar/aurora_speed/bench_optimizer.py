#!/usr/bin/env python3
"""S-series microbench: time optimizer.step() in isolation on real model shapes.

Usage (run ONLY when no train.py is active, single GPU is enough):
  CUDA_VISIBLE_DEVICES=0 python ar/aurora_speed/bench_optimizer.py --optimizer aurora
  CUDA_VISIBLE_DEVICES=0 python ar/aurora_speed/bench_optimizer.py --optimizer muon

Times per-step optimizer cost with CUDA events over N steps with synthetic
gradients on the REAL parameter set of the current best config. Compare the
result against recorded step_avg (~720ms for F100-era runs) to get the
optimizer's share of the training step.

NOTE: import paths follow the repo layout at commit ea8784f; if get_optimizer
or config loading moved, adapt the two marked blocks — keep the timing core.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

ROOT = Path("/nvme/BlaGPT")
sys.path.insert(0, str(ROOT / "bla_gpt"))

def build(optimizer_name: str):
    # --- ADAPT BLOCK 1: model construction ---
    cfg_path = ROOT / "ar" / "best_config.json"
    raw = json.loads(cfg_path.read_text())
    from bla_gpt import GPT, GPTConfig  # noqa
    conf = GPTConfig(**{k: v for k, v in raw.items() if hasattr(GPTConfig(), k)}) \
        if not hasattr(GPTConfig, "from_dict") else GPTConfig.from_dict(raw)
    model = GPT(conf).cuda().bfloat16()
    # --- ADAPT BLOCK 2: optimizer construction (mirror train.py) ---
    from optimizers import get_optimizer  # noqa
    opt_args = raw.get("optimizer_args", {})
    optimizer = get_optimizer(optimizer_name, model,
                              learning_rate=raw.get("learning_rate", 0.03),
                              **opt_args) if callable(get_optimizer) else None
    return model, optimizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="aurora")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    model, optimizer = build(a.optimizer)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"optimizer={a.optimizer} params={n_params/1e6:.1f}M tensors={len(params)}")

    opts = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]

    def synth_grads():
        for p in params:
            p.grad = torch.randn_like(p, dtype=p.dtype) * 1e-3

    # warmup
    for _ in range(a.warmup):
        synth_grads()
        for o in opts:
            o.step()
        for o in opts:
            o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(a.steps):
        synth_grads()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for o in opts:
            o.step()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
        for o in opts:
            o.zero_grad(set_to_none=True)

    med = statistics.median(times)
    print(f"RESULT optimizer={a.optimizer} median_ms={med:.2f} "
          f"p10={sorted(times)[len(times)//10]:.2f} p90={sorted(times)[-len(times)//10]:.2f} "
          f"share_of_720ms_step={100*med/720:.1f}%")

if __name__ == "__main__":
    main()