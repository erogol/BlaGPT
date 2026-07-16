"""Tests for GRAPE-A query-gated additive position bias (arXiv:2512.07805).

Query-gated Additive GRAPE special case:
    b_h(i, j) = -(i - j) * softplus(omega_h) * slope_h * softplus(v_h^T q_i / sqrt(D))
Enabled via pos_encoding == "grape_a_qgate"; replaces RoPE entirely.
"""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bla_gpt"))

from bla_gpt import GPT, GPTConfig, get_attention
from attentions import Attention, GOATSinkAttention, GrapeQueryGatedBias


def _tiny(pos_encoding="grape_a_qgate", attention="regular", goat=False):
    cfg = GPTConfig()
    cfg.block_size = 8
    cfg.vocab_size = 64
    cfg.n_layer = 2
    cfg.n_head = 4
    cfg.n_kv_head = 4
    cfg.n_embd = 16
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = pos_encoding
    cfg.activation = "gelu"
    cfg.norm_layer = "layernorm"
    cfg.attention = attention
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.use_engram = False
    cfg.use_goat_sink_prior = goat
    return cfg


def test_gate_off_no_grape_module():
    attn = get_attention(_tiny(pos_encoding="rotary"))
    assert not hasattr(attn, "grape_bias")
    assert hasattr(attn, "rotary")


def test_gate_on_creates_grape_no_rotary():
    attn = get_attention(_tiny())
    assert hasattr(attn, "grape_bias")
    assert isinstance(attn.grape_bias, GrapeQueryGatedBias)
    assert not hasattr(attn, "rotary")
    assert not hasattr(attn, "rel_pos_emb")


def test_bias_shape_causal_and_sign():
    torch.manual_seed(0)
    mod = GrapeQueryGatedBias(4, 8, 16)
    q = torch.randn(2, 4, 10, 8)
    bias = mod(q)
    assert bias.shape == (2, 4, 10, 10)
    # diagonal is exactly zero (dist == 0)
    assert torch.allclose(torch.diagonal(bias, dim1=-2, dim2=-1), torch.zeros(2, 4, 10))
    # lower triangle strictly non-positive (monotonic distance penalty)
    tril = torch.tril(torch.ones(10, 10, dtype=torch.bool), diagonal=-1)
    assert (bias[..., tril] <= 0).all()


def test_bias_relative_law_rows_scale_with_gate():
    """b(i, j) must factor as -(i-j) * rate_h(q_i): each row is the row gate
    times the plain distance ramp (exact relative law of the unipotent action)."""
    torch.manual_seed(1)
    mod = GrapeQueryGatedBias(2, 4, 8)
    q = torch.randn(1, 2, 8, 4)
    bias = mod(q)
    for h in range(2):
        for i in range(1, 8):
            row = bias[0, h, i, :i]  # strictly-past keys
            dist = torch.arange(i, 0, -1, dtype=torch.float32)
            rate = -row / dist
            # constant per-row rate => query-gated, offset-only dependence
            assert torch.allclose(rate, rate[0].expand_as(rate), atol=1e-5)


def test_matches_reference_formula():
    torch.manual_seed(2)
    H, D, T = 4, 8, 6
    mod = GrapeQueryGatedBias(H, D, T)
    q = torch.randn(1, H, T, D)
    bias = mod(q)
    omega = F.softplus(mod.omega).view(H, 1, 1) * mod.slopes.view(H, 1, 1)
    v = mod.v / mod.v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    gate = F.softplus((q[0] * v.view(H, 1, D)).sum(-1) / math.sqrt(D))  # (H, T)
    ar = torch.arange(T)
    dist = (ar.view(-1, 1) - ar.view(1, -1)).clamp_min(0).float()
    ref = -dist.view(1, T, T) * omega * gate.unsqueeze(-1)
    assert torch.allclose(bias[0], ref, atol=1e-6)


def test_flash_manual_agree_regular():
    torch.manual_seed(3)
    attn = get_attention(_tiny()).eval()
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y_flash = attn(x)
        attn.flash = False
        attn.mask = None
        y_manual = attn(x)
    assert torch.allclose(y_flash, y_manual, atol=1e-5)


def test_flash_manual_agree_goat_sink():
    torch.manual_seed(4)
    attn = get_attention(_tiny(attention="xsa", goat=True)).eval()
    assert isinstance(attn, GOATSinkAttention)
    assert hasattr(attn, "grape_bias")
    with torch.no_grad():
        attn.sink_prior.uniform_(-0.5, 0.5)
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y_flash = attn(x)
        attn.flash = False
        attn.mask = None
        y_manual = attn(x)
    assert torch.allclose(y_flash, y_manual, atol=1e-5)


def test_flash_manual_agree_composable_gated():
    """GRAPE must compose with the full best stack: xsa + goat sink + composable gate."""
    torch.manual_seed(8)
    cfg = _tiny(attention="xsa", goat=True)
    cfg.use_composable_gated_attn = True
    attn = get_attention(cfg).eval()
    from attentions import ComposableGatedAttention
    assert isinstance(attn, ComposableGatedAttention)
    assert hasattr(attn, "grape_bias")
    with torch.no_grad():
        attn.sink_prior.uniform_(-0.5, 0.5)
        attn.gate_proj.weight.normal_(std=0.02)
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y_flash = attn(x)
        attn.flash = False
        attn.mask = None
        y_manual = attn(x)
    assert torch.allclose(y_flash, y_manual, atol=1e-5)

def test_causality():
    """Changing a future token must not change past outputs."""
    torch.manual_seed(5)
    attn = get_attention(_tiny(attention="xsa", goat=True)).eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[0, -1] += 1.0
    with torch.no_grad():
        y1 = attn(x)
        y2 = attn(x2)
    assert torch.allclose(y1[0, :-1], y2[0, :-1], atol=1e-6)


def test_full_model_forward_backward_finite():
    torch.manual_seed(6)
    cfg = _tiny(attention="xsa", goat=True)
    model = GPT(cfg)
    idx = torch.randint(0, 64, (2, 8))
    tgt = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    grape_params = [p for n, p in model.named_parameters() if "grape_bias" in n]
    assert grape_params, "grape parameters missing from model"
    for p in grape_params:
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_tiny_overfit():
    """A tiny model with GRAPE must overfit a fixed batch (loss drops sharply)."""
    torch.manual_seed(7)
    cfg = _tiny(attention="xsa", goat=True)
    model = GPT(cfg)
    idx = torch.randint(0, 64, (4, 8))
    tgt = torch.randint(0, 64, (4, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    _, loss0 = model(idx, tgt)
    for _ in range(150):
        opt.zero_grad()
        _, loss = model(idx, tgt)
        loss.backward()
        opt.step()
    assert loss.item() < loss0.item() * 0.5, f"no overfit: {loss0.item()} -> {loss.item()}"
