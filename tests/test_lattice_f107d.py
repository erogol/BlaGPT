"""Tests for F107d: lattice_film_impl "bf16" and "bf16cat" (default "fp32").

Each variant must keep the step-0 identity (zero FiLM heads = base model) and
match the fp32 FiLM path within a bf16 tolerance after the heads get values.
"""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model_d", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig

import lattice  # noqa: E402

CFG_DIR = ROOT / "ar" / "full_train" / "configs"
LAT = {"lattice_enabled": True, "lattice_layers": [1, 2, 3, 4], "lattice_dr": 32,
       "lattice_inj": True, "lattice_film_after_qknorm": True,
       "lattice_adamw": True, "lattice_vstream": True}


def _mk_cfg(**kw):
    cfg = GPTConfig()
    for k, v in json.load(open(CFG_DIR / "base_x5.json")).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv_head = 6, 128, 4, 4
    cfg.mlp_expand, cfg.block_size, cfg.vocab_size, cfg.engram_vocab_mult = 2, 64, 512, 2
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def _build(cfg, device, seed=0):
    torch.manual_seed(seed)
    return GPT(cfg).to(device)


def _logits(m, x):
    m.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(x)
    return (out[0] if isinstance(out, tuple) else out).float()


def _randomize(m, std=0.02):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("c_proj") or n.endswith("c_proj.weight") or (
                "lattice" in n and any(h in n for h in ("film_", "mlp_beta", "w_e_", "vs_up"))
            ):
                p.normal_(std=std)


def test_default_film_impl_is_fp32():
    assert GPTConfig().lattice_film_impl == "fp32"
    assert lattice.LatticeLayer(64, 16, 64, 64, 128).film_impl == "fp32"


def test_unknown_film_impl_raises():
    with pytest.raises(ValueError):
        lattice.LatticeLayer(64, 16, 64, 64, 128, film_impl="nope")


@pytest.mark.parametrize("impl", ["bf16", "bf16cat"])
def test_zero_init_matches_base(device, impl):
    """Step 0: zero FiLM/MLP heads give the logits of the model without Lattice."""
    m_off = _build(_mk_cfg(), device)
    m_on = _build(_mk_cfg(**LAT, lattice_film_impl=impl), device)
    m_on.load_state_dict(m_off.state_dict(), strict=False)
    x = torch.randint(0, 512, (2, 32), device=device)
    assert (_logits(m_off, x) - _logits(m_on, x)).abs().max().item() == 0.0


@pytest.mark.parametrize("impl", ["bf16", "bf16cat"])
def test_matches_fp32_film(device, impl):
    """With non-zero heads the variant stays within bf16 noise of fp32 FiLM."""
    ref = _build(_mk_cfg(**LAT), device)
    _randomize(ref, std=0.05)
    var = _build(_mk_cfg(**LAT, lattice_film_impl=impl), device)
    var.load_state_dict(ref.state_dict())
    x = torch.randint(0, 512, (2, 32), device=device)
    a, b = _logits(ref, x), _logits(var, x)
    scale = a.abs().max().item()
    assert scale > 0
    assert (a - b).abs().max().item() < 0.03 * scale


@pytest.mark.parametrize("impl", ["bf16", "bf16cat"])
def test_grads_reach_all_heads(device, impl):
    m = _build(_mk_cfg(**LAT, lattice_film_impl=impl), device)
    _randomize(m, std=0.05)
    m.train()
    x = torch.randint(0, 512, (2, 32), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = m(x, x)
    loss = loss["total"] if isinstance(loss, dict) else loss
    loss.backward()
    for mod in m.lattice.values():
        for lin in (mod.film_q, mod.film_k, mod.film_v, mod.mlp_beta):
            assert lin.weight.grad is not None
            assert lin.weight.grad.abs().sum().item() > 0
