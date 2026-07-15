"""Tests for F88: OASIS depth-Softmax1 null routing for AttnResidual."""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model_f88", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model_f88", blagpt_model)
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
    cfg.use_attn_res = True
    cfg.use_unet_skips = False
    cfg.use_oasis_depth_softmax1 = gate
    return cfg


if not torch.cuda.is_available():
    pytest.skip("CUDA required (Primer MLP init hardcodes cuda)", allow_module_level=True)
DEV = "cuda"


def _mk(gate, seed=11):
    torch.manual_seed(seed)
    return GPT(_mk_cfg(gate)).to(DEV)


def _fwd(m, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(0, 512, (2, 64), device=DEV)
    y = torch.randint(0, 512, (2, 64), device=DEV)
    out = m(x, y)
    loss = out[1]
    return loss["total"] if isinstance(loss, dict) else loss


def test_gate_off_no_new_params_and_matches_baseline():
    m_off = _mk(False)
    m_base = _mk(False)
    assert sum(p.numel() for p in m_off.parameters()) == sum(p.numel() for p in m_base.parameters())
    assert torch.allclose(_fwd(m_off), _fwd(m_base))


def test_softmax1_null_branch_reduces_real_mass():
    m = _mk(True)
    scores = torch.zeros(3, 2, 5, device=DEV)
    alpha = m._attn_res_route(scores)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.25))
    assert torch.allclose(alpha.sum(dim=0), torch.full((2, 5), 0.75, device=DEV))


def test_gate_on_changes_output():
    m_off = _mk(False)
    m_on = _mk(True)
    assert not torch.allclose(_fwd(m_off), _fwd(m_on))


def test_grads_finite_with_oasis_depth_route():
    m = _mk(True)
    loss = _fwd(m)
    loss.backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_gate_off_route_is_plain_softmax():
    cfg = SimpleNamespace(use_oasis_depth_softmax1=False)
    m = GPT.__new__(GPT)
    m.config = cfg
    scores = torch.randn(5, 2, 3, device=DEV)
    assert torch.allclose(m._attn_res_route(scores), scores.softmax(dim=0))
