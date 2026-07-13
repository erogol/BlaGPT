"""Tests for F87: U-net long skip connections (modded-nanogpt lineage, chain-20 retest at full horizon)."""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model_f87", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model_f87", blagpt_model)
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
    cfg.block_size = 64
    cfg.vocab_size = 512
    cfg.zero_init_proj_layers = False
    cfg.use_unet_skips = gate
    return cfg


if not torch.cuda.is_available():
    pytest.skip("CUDA required (Primer MLP init hardcodes cuda)", allow_module_level=True)
DEV = "cuda"


def _mk(gate, seed=7):
    torch.manual_seed(seed)
    return GPT(_mk_cfg(gate)).to(DEV)


def _fwd(m, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(0, 512, (2, 64), device=DEV)
    y = torch.randint(0, 512, (2, 64), device=DEV)
    out = m(x, y)
    loss = out[1]
    return loss["total"] if isinstance(loss, dict) else loss


def test_gate_off_no_param_and_matches_baseline():
    m_off = _mk(False)
    assert not hasattr(m_off, "skip_weights")
    m_base = _mk(False)
    assert torch.allclose(_fwd(m_off), _fwd(m_base))


def test_gate_on_param_shape_and_init():
    m = _mk(True)
    assert m.skip_weights.shape == (2,)
    assert torch.allclose(m.skip_weights, torch.full((2,), 0.25, device=DEV))


def test_gate_on_changes_output():
    m_off = _mk(False)
    m_on = _mk(True)
    assert not torch.allclose(_fwd(m_off), _fwd(m_on))


def test_grads_finite_and_skip_weights_learn():
    m = _mk(True)
    loss = _fwd(m)
    loss.backward()
    assert m.skip_weights.grad is not None
    assert torch.isfinite(m.skip_weights.grad).all()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_zero_weights_match_gate_off():
    m_on = _mk(True)
    with torch.no_grad():
        m_on.skip_weights.zero_()
    m_off = _mk(False)
    assert torch.allclose(_fwd(m_on), _fwd(m_off), atol=1e-6)
