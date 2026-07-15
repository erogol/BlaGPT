"""CPU tests for GatedNorm (F77, arXiv:2601.22966 Sec.3.4)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bla_gpt"))

from norms import GatedNorm, RMSNorm
from bla_gpt import GPT, GPTConfig, get_norm


def _cfg(gated=False, rank=16):
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
    cfg.use_gated_norm = gated
    cfg.gated_norm_rank = rank
    return cfg


def test_gate_off_returns_rmsnorm():
    norm = get_norm(_cfg(gated=False))
    assert type(norm) is RMSNorm, "gate-off must return plain RMSNorm"
    assert not isinstance(norm, GatedNorm)


def test_gate_on_returns_gated_norm():
    norm = get_norm(_cfg(gated=True, rank=16))
    assert isinstance(norm, GatedNorm)


def test_wup_initialized_to_zeros():
    """W_up=zeros => gate=sigmoid(0)=0.5 at init (half-identity init, paper Sec.3.4)."""
    norm = GatedNorm(ndim=16, rank=16)
    assert norm.W_up.weight.shape == (16, 16)
    assert norm.W_up.weight.eq(0.0).all(), "W_up must be zeros at init"


def test_gate_at_init_is_half():
    """With W_up=zeros, gate output is 0.5 for any input."""
    torch.manual_seed(0)
    norm = GatedNorm(ndim=16, rank=8)
    x = torch.randn(3, 8, 16)
    with torch.no_grad():
        y_rms = norm.norm(x)
        gate_val = torch.sigmoid(norm.W_up(torch.nn.functional.silu(norm.W_down(y_rms))))
    assert torch.allclose(gate_val, torch.full_like(gate_val, 0.5), atol=1e-6), (
        "gate must be 0.5 at init when W_up=zeros"
    )


def test_output_changes_after_wup_perturbed():
    """Perturbing W_up changes output (gate is not ignored)."""
    torch.manual_seed(1)
    norm = GatedNorm(ndim=16, rank=8)
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        y_zero = norm(x).clone()
        # Set W_up to non-zero so gate deviates from 0.5
        norm.W_up.weight.data.fill_(0.1)
        y_nonzero = norm(x)
    assert not torch.allclose(y_zero, y_nonzero, atol=1e-6), (
        "non-zero W_up must change output"
    )


def test_gradient_reaches_wdown():
    torch.manual_seed(2)
    norm = GatedNorm(ndim=16, rank=8)
    # Give W_up a non-zero init so gradient can propagate back through W_down
    with torch.no_grad():
        norm.W_up.weight.data.normal_(0, 0.01)
    x = torch.randn(2, 4, 16)
    out = norm(x)
    out.sum().backward()
    assert norm.W_down.weight.grad is not None
    assert norm.W_down.weight.grad.abs().sum() > 0, "gradient must reach W_down"


def test_gradient_reaches_wup():
    torch.manual_seed(3)
    norm = GatedNorm(ndim=16, rank=8)
    x = torch.randn(2, 4, 16)
    out = norm(x)
    out.sum().backward()
    assert norm.W_up.weight.grad is not None
    assert norm.W_up.weight.grad.abs().sum() > 0, (
        "gradient must reach W_up (W_down Kaiming ensures non-zero swish output)"
    )


def test_shape_and_dtype_preserved():
    norm = GatedNorm(ndim=32, rank=16)
    x = torch.randn(4, 16, 32)
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_state_dict_roundtrip():
    torch.manual_seed(4)
    ndim, rank = 16, 8
    n1 = GatedNorm(ndim, rank)
    with torch.no_grad():
        n1.W_down.weight.data.normal_()
        n1.W_up.weight.data.normal_()
        n1.norm.weight.data.uniform_()
    state = n1.state_dict()
    expected_keys = {"norm.weight", "W_down.weight", "W_up.weight"}
    assert set(state.keys()) == expected_keys, f"unexpected keys: {set(state.keys())}"
    n2 = GatedNorm(ndim, rank)
    n2.load_state_dict(state)
    x = torch.randn(2, 4, ndim)
    with torch.no_grad():
        assert torch.allclose(n1(x), n2(x), atol=1e-6), "state_dict roundtrip must reproduce output"


def test_gate_off_full_model_uses_rmsnorm():
    torch.manual_seed(5)
    model = GPT(_cfg(gated=False))
    for block in model.transformer.h:
        assert type(block.ln_1) is RMSNorm, f"gate-off: ln_1 must be RMSNorm, got {type(block.ln_1)}"
        assert type(block.ln_2) is RMSNorm, f"gate-off: ln_2 must be RMSNorm, got {type(block.ln_2)}"


def test_gate_on_full_model_uses_gated_norm():
    torch.manual_seed(6)
    model = GPT(_cfg(gated=True))
    for block in model.transformer.h:
        assert isinstance(block.ln_1, GatedNorm), f"gate-on: ln_1 must be GatedNorm, got {type(block.ln_1)}"
        assert isinstance(block.ln_2, GatedNorm), f"gate-on: ln_2 must be GatedNorm, got {type(block.ln_2)}"
    # Final norm also through get_norm
    assert isinstance(model.transformer.ln_f, GatedNorm)
