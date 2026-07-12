"""CPU tests for PreAffineRMSNorm (F76, arXiv:2601.22966 Sec.3.3)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bla_gpt"))

from norms import PreAffineRMSNorm, RMSNorm
from bla_gpt import GPT, GPTConfig, get_norm


def _cfg(pre_affine=False):
    cfg = GPTConfig()
    cfg.block_size = 8
    cfg.vocab_size = 64
    cfg.n_layer = 2
    cfg.n_head = 4
    cfg.n_kv_head = 4
    cfg.n_embd = 16
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = "none"
    cfg.activation = "gelu"
    cfg.norm_layer = "rmsnorm"
    cfg.attention = "xsa"
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.use_engram = False
    cfg.use_pre_affine_norm = pre_affine
    return cfg


def test_gate_off_builds_rmsnorm():
    norm = get_norm(_cfg(pre_affine=False))
    assert type(norm) is RMSNorm, "gate-off must return plain RMSNorm"
    assert not isinstance(norm, PreAffineRMSNorm)


def test_gate_on_builds_pre_affine_rmsnorm():
    norm = get_norm(_cfg(pre_affine=True))
    assert isinstance(norm, PreAffineRMSNorm)


def test_lambda1_initialized_to_ones():
    norm = PreAffineRMSNorm(ndim=16)
    assert norm.lambda1.shape == (16,)
    assert norm.lambda1.eq(1.0).all(), "lambda1 must be ones at init"


def test_gate_on_matches_baseline_under_copied_weights():
    torch.manual_seed(0)
    ndim = 16
    rms = RMSNorm(ndim)
    pre = PreAffineRMSNorm(ndim)
    # copy weight; lambda1 stays ones
    pre.norm.weight.data.copy_(rms.weight.data)
    x = torch.randn(3, 8, ndim)
    with torch.no_grad():
        y_rms = rms(x)
        y_pre = pre(x)
    assert torch.allclose(y_rms, y_pre, atol=1e-6), (
        "ones-lambda1 PreAffineRMSNorm must match plain RMSNorm with same weights"
    )


def test_changing_lambda1_changes_output():
    torch.manual_seed(1)
    ndim = 16
    pre = PreAffineRMSNorm(ndim)
    x = torch.randn(2, 4, ndim)
    with torch.no_grad():
        y_ones = pre(x).clone()
    with torch.no_grad():
        # non-uniform lambda1 changes relative magnitudes across features,
        # which is NOT absorbed by the per-token RMS normalisation
        pre.lambda1.data = torch.linspace(0.5, 2.0, ndim)
        y_nonuniform = pre(x)
    assert not torch.allclose(y_ones, y_nonuniform, atol=1e-6), (
        "non-uniform lambda1 must change output"
    )


def test_gradient_reaches_lambda1():
    torch.manual_seed(2)
    pre = PreAffineRMSNorm(ndim=16)
    x = torch.randn(2, 4, 16)
    out = pre(x)
    out.sum().backward()
    assert pre.lambda1.grad is not None, "gradient must reach lambda1"
    assert pre.lambda1.grad.abs().sum() > 0, "lambda1 gradient must be non-zero"


def test_shape_and_dtype_preserved():
    pre = PreAffineRMSNorm(ndim=32)
    x = torch.randn(4, 16, 32)
    y = pre(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_state_dict_roundtrip():
    torch.manual_seed(3)
    ndim = 16
    pre1 = PreAffineRMSNorm(ndim)
    with torch.no_grad():
        pre1.lambda1.data.copy_(torch.rand(ndim))
        pre1.norm.weight.data.copy_(torch.rand(ndim))
    state = pre1.state_dict()
    assert set(state.keys()) == {"lambda1", "norm.weight"}, f"unexpected keys: {set(state.keys())}"
    pre2 = PreAffineRMSNorm(ndim)
    pre2.load_state_dict(state)
    x = torch.randn(2, 4, ndim)
    with torch.no_grad():
        assert torch.allclose(pre1(x), pre2(x), atol=1e-6), "state_dict roundtrip must reproduce output"


def test_gate_off_full_model_unchanged():
    torch.manual_seed(4)
    cfg_off = _cfg(pre_affine=False)
    cfg_on = _cfg(pre_affine=True)
    model_off = GPT(cfg_off)
    model_on = GPT(cfg_on)
    # verify gate-off norms are RMSNorm, gate-on norms are PreAffineRMSNorm
    for block in model_off.transformer.h:
        assert type(block.ln_1) is RMSNorm
        assert type(block.ln_2) is RMSNorm
    for block in model_on.transformer.h:
        assert isinstance(block.ln_1, PreAffineRMSNorm)
        assert isinstance(block.ln_2, PreAffineRMSNorm)
