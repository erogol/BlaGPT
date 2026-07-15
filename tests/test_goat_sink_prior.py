"""Tests for GOAT key-only sink prior (arXiv:2601.15380)."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bla_gpt"))

from bla_gpt import GPT, GPTConfig, get_attention
from attentions import ExclusiveSelfAttention, GOATSinkAttention


def _tiny(goat=False):
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
    cfg.norm_layer = "layernorm"
    cfg.attention = "xsa"
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.use_engram = False
    cfg.use_goat_sink_prior = goat
    return cfg


def test_gate_off_returns_xsa():
    attn = get_attention(_tiny(goat=False))
    assert isinstance(attn, ExclusiveSelfAttention)
    assert not isinstance(attn, GOATSinkAttention)


def test_gate_on_returns_goat():
    attn = get_attention(_tiny(goat=True))
    assert isinstance(attn, GOATSinkAttention)


def test_gate_off_same_output():
    torch.manual_seed(0)
    cfg_off = _tiny(goat=False)
    cfg_on = _tiny(goat=True)
    attn_off = get_attention(cfg_off)
    attn_on = get_attention(cfg_on)
    # copy weights so the only difference is the class
    attn_on.load_state_dict(
        {k: v for k, v in attn_off.state_dict().items()},
        strict=False,
    )
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        y_off = attn_off(x)
        y_on = attn_on(x)
    assert torch.allclose(y_off, y_on, atol=1e-6), "gate-off vs gate-on zero-init must match"


def test_zero_init_gate_on_same_output():
    torch.manual_seed(7)
    cfg_xsa = _tiny(goat=False)
    cfg_goat = _tiny(goat=True)
    attn_xsa = get_attention(cfg_xsa)
    attn_goat = get_attention(cfg_goat)
    # share all common weights
    attn_goat.load_state_dict(
        {k: v for k, v in attn_xsa.state_dict().items()},
        strict=False,
    )
    # sink_prior is zero-initialised — output must match XSA exactly
    assert attn_goat.sink_prior.eq(0).all(), "sink_prior must be zero at init"
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        y_xsa = attn_xsa(x)
        y_goat = attn_goat(x)
    assert torch.allclose(y_xsa, y_goat, atol=1e-6), "zero-init goat must equal xsa"


def test_nonzero_key0_bias_changes_output():
    torch.manual_seed(3)
    cfg = _tiny(goat=True)
    attn = get_attention(cfg)
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        y_zero = attn(x).clone()
    # perturb the sink prior
    with torch.no_grad():
        attn.sink_prior.fill_(2.0)
    with torch.no_grad():
        y_bias = attn(x)
    assert not torch.allclose(y_zero, y_bias, atol=1e-6), "nonzero sink_prior must change output"


def test_causal_mask_preserved():
    torch.manual_seed(5)
    cfg = _tiny(goat=True)
    attn = get_attention(cfg)
    T = 6
    x = torch.randn(1, T, 16)
    with torch.no_grad():
        y = attn(x)
    # perturb a future token and verify earlier positions are unaffected
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(16)
    with torch.no_grad():
        y2 = attn(x2)
    # all query positions except the last must be unaffected
    assert torch.allclose(y[:, :-1, :], y2[:, :-1, :], atol=1e-6), (
        "causal mask broken: earlier positions depend on future token"
    )


def test_gradient_reaches_sink_prior():
    torch.manual_seed(9)
    cfg = _tiny(goat=True)
    attn = get_attention(cfg)
    x = torch.randn(2, 5, 16)
    out = attn(x)
    out.sum().backward()
    assert attn.sink_prior.grad is not None, "gradient must reach sink_prior"
    assert attn.sink_prior.grad.abs().sum() > 0, "sink_prior gradient must be non-zero"
