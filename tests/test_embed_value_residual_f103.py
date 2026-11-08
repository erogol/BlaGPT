"""Tests for F103: embedding value residual (Ripple-65B "valres")."""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig


def _mk_cfg(gate, value_residual=True):
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
    cfg.use_value_residual = value_residual
    cfg.use_embed_value_residual = gate
    return cfg


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def _loss(m, x, y):
    loss = m(x, y)[1]
    return loss["total"] if isinstance(loss, dict) else loss


def test_off_has_no_params(device):
    m = GPT(_mk_cfg(False))
    assert not any("ev_res" in n for n, _ in m.named_parameters())
    assert getattr(m.transformer.h[0].attn, "ev_res_holder", None) is None


def test_on_one_zero_scalar_per_layer(device):
    m = GPT(_mk_cfg(True))
    names = [n for n, _ in m.named_parameters() if "ev_res_lambda" in n]
    assert len(names) == 4
    for block in m.transformer.h:
        assert block.attn.ev_res_lambda.shape == (1,)
        assert block.attn.ev_res_lambda.item() == 0.0
        assert block.attn.ev_res_holder is m._ev_res_holder


def test_blend_matches_reference(device):
    m = GPT(_mk_cfg(True, value_residual=False)).to(device)
    attn = m.transformer.h[1].attn
    with torch.no_grad():
        attn.ev_res_lambda.fill_(0.7)
    B, T, C = 2, 16, 128
    h = torch.randn(B, T, C, device=device)
    e = torch.randn(B, T, C, device=device)
    m._ev_res_holder["e"] = e
    _, v = attn._project_kv(h, B, T)
    s = torch.sigmoid(torch.tensor(0.7))
    mixed = torch.lerp(e, h, s.to(device))
    ref = attn.kv_proj(mixed).view(B, T, 2, attn.n_kv_head, attn.head_dim)[:, :, 1]
    assert torch.allclose(v, ref, atol=1e-5)


def test_holder_gets_normed_embedding(device):
    m = GPT(_mk_cfg(True)).to(device)
    x = torch.randint(0, 512, (2, 32), device=device)
    m(x, x)
    e = m._ev_res_holder["e"]
    ref = F.rms_norm(m.transformer.wte(x), (128,))
    assert torch.allclose(e, ref, atol=1e-5)


def test_forward_changes_and_grads_flow(device):
    x = torch.randint(0, 512, (2, 64), device=device)
    torch.manual_seed(0)
    m_on = GPT(_mk_cfg(True)).to(device)
    torch.manual_seed(0)
    m_off = GPT(_mk_cfg(False)).to(device)
    assert abs(float(_loss(m_on, x, x)) - float(_loss(m_off, x, x))) > 1e-6
    _loss(m_on, x, x).backward()
    for block in m_on.transformer.h:
        g = block.attn.ev_res_lambda.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert m_on.transformer.wte.weight.grad.abs().sum() > 0


def test_replace_mode_without_resformer(device):
    m = GPT(_mk_cfg(True, value_residual=False)).to(device)
    x = torch.randint(0, 512, (2, 64), device=device)
    assert torch.isfinite(_loss(m, x, x))
