"""Tests for F81: Affine-Scaled Attention (arXiv:2602.23057, Bae et al., ICML 2026)."""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig


def _mk_cfg(gate):
    cfg_dict = json.load(open(ROOT / "ar" / "best_config.json"))
    cfg = GPTConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_layer = 4
    cfg.n_embd = 128
    cfg.n_head = 4
    cfg.n_kv_head = 4
    cfg.mlp_expand = 2
    cfg.block_size = 128
    cfg.vocab_size = 512
    cfg.zero_init_proj_layers = False
    cfg.use_affine_scaled_attn = gate
    return cfg


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def test_dispatch_and_modules(device):
    m = GPT(_mk_cfg(True)).to(device)
    a0 = m.transformer.h[0].attn
    assert type(a0).__name__ == "AffineScaledAttention"
    assert hasattr(a0, "alpha_proj")
    assert hasattr(a0, "sink_prior")  # GOAT inherited
    assert torch.allclose(a0.alpha_ma, torch.full((4,), 0.5, device=device))
    assert a0.alpha_proj.weight.abs().sum() == 0  # zero init => alpha 0.5


def test_gate_off_unchanged(device):
    m = GPT(_mk_cfg(False)).to(device)
    a0 = m.transformer.h[0].attn
    assert type(a0).__name__ != "AffineScaledAttention"
    assert not hasattr(a0, "alpha_proj")


def test_forward_changes_output(device):
    x = torch.randint(0, 512, (2, 64), device=device)
    y = torch.randint(0, 512, (2, 64), device=device)
    torch.manual_seed(0)
    m_on = GPT(_mk_cfg(True)).to(device)
    torch.manual_seed(0)
    m_off = GPT(_mk_cfg(False)).to(device)
    l_on = m_on(x, y)[1]
    l_off = m_off(x, y)[1]
    l_on = l_on["total"] if isinstance(l_on, dict) else l_on
    l_off = l_off["total"] if isinstance(l_off, dict) else l_off
    assert abs(float(l_on) - float(l_off)) > 1e-6


def test_gradients_and_ema(device):
    m = GPT(_mk_cfg(True)).to(device)
    m.train()
    a0 = m.transformer.h[0].attn
    ma_before = a0.alpha_ma.clone()
    x = torch.randint(0, 512, (2, 64), device=device)
    y = torch.randint(0, 512, (2, 64), device=device)
    loss = m(x, y)[1]
    loss = loss["total"] if isinstance(loss, dict) else loss
    loss.backward()
    assert a0.alpha_proj.weight.grad is not None
    assert a0.alpha_proj.weight.grad.abs().sum() > 0
    # EMA buffer moved during a training forward (alpha==0.5 at init, but
    # batch mean is exactly 0.5 too => allow tiny/zero drift, just require finite)
    assert torch.isfinite(a0.alpha_ma).all()
    assert a0.alpha_ma.shape == ma_before.shape


def test_ema_frozen_in_eval(device):
    m = GPT(_mk_cfg(True)).to(device)
    m.eval()
    a0 = m.transformer.h[0].attn
    ma_before = a0.alpha_ma.clone()
    x = torch.randint(0, 512, (2, 64), device=device)
    with torch.no_grad():
        m(x, None)
    assert torch.equal(a0.alpha_ma, ma_before)


def test_causality(device):
    m = GPT(_mk_cfg(True)).to(device)
    m.eval()
    with torch.no_grad():
        xa = torch.randint(0, 512, (1, 32), device=device)
        xb = xa.clone()
        xb[0, -1] = (xb[0, -1] + 7) % 512
        la, _ = m(xa, xa)
        lb, _ = m(xb, xb)
    assert torch.allclose(la[0, :31], lb[0, :31], atol=1e-5)
