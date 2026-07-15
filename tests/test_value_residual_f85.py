"""Tests for F85: composable Value Residual Learning (arXiv:2410.17897, learnable-plus)."""
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


def _mk_cfg(gate, zero_init=False):
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
    cfg.zero_init_proj_layers = zero_init
    cfg.use_value_residual = gate
    return cfg


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def test_params_created_when_enabled(device):
    m = GPT(_mk_cfg(True))
    assert hasattr(m, "v_res_logits")
    assert m.v_res_logits.shape == (3,)  # n_layer - 1
    assert m.v_res_scale.item() == pytest.approx(4.0)  # init = n_layer
    assert m.v_res_lambda2.shape == (3,)
    assert torch.allclose(m.v_res_lambda2, torch.full((3,), 0.5))
    for i, block in enumerate(m.transformer.h):
        assert block.attn.v_res_depth == i
        assert block.attn.v_res_holder is m._v_res_holder


def test_no_params_when_disabled(device):
    m = GPT(_mk_cfg(False))
    assert not hasattr(m, "v_res_logits")
    assert getattr(m.transformer.h[0].attn, "v_res_holder", None) is None


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


def test_gradients_flow(device):
    m = GPT(_mk_cfg(True)).to(device)
    x = torch.randint(0, 512, (2, 64), device=device)
    y = torch.randint(0, 512, (2, 64), device=device)
    loss = m(x, y)[1]
    loss = loss["total"] if isinstance(loss, dict) else loss
    loss.backward()
    assert m.v_res_logits.grad is not None and m.v_res_logits.grad.abs().sum() > 0
    assert m.v_res_scale.grad is not None and m.v_res_scale.grad.abs().sum() > 0
    assert m.v_res_lambda2.grad is not None and m.v_res_lambda2.grad.abs().sum() > 0
    # V1 stays in autograd graph (paper-faithful, unlike standalone ResFormer)
    assert m.transformer.h[0].attn.kv_proj.weight.grad.abs().sum() > 0


def test_repeated_forwards_consistent(device):
    m = GPT(_mk_cfg(True)).to(device)
    m.eval()
    x = torch.randint(0, 512, (2, 64), device=device)
    with torch.no_grad():
        l1 = m(x, None)[0]
        l2 = m(x, None)[0]
    assert torch.allclose(l1, l2)


def test_variable_seq_len(device):
    m = GPT(_mk_cfg(True)).to(device)
    m.eval()
    with torch.no_grad():
        for t in (16, 48, 128):
            x = torch.randint(0, 512, (2, t), device=device)
            logits, _ = m(x, None)
            assert logits.shape[1] in (t, 1)  # inference may return last-pos only
