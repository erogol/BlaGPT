"""Tests for F111: 3-way AttnRes (split Q/K/V depth routing)."""
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bla_gpt") not in sys.path:
    sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model_f111", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules.setdefault("blagpt_model_f111", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig


def _tiny_cfg(
    *,
    attn_res_split_qkv: bool = False,
    attention: str = "xsa",
    composable_gated: bool = True,
    goat_sink: bool = True,
    unet_skips: bool = True,
    engram: bool = True,
    value_residual: bool = True,
    rove: bool = True,
    hisa: bool = True,
    n_layer: int = 4,
    n_embd: int = 32,
    n_head: int = 4,
):
    cfg = GPTConfig()
    cfg.block_size = 16
    cfg.vocab_size = 64
    cfg.n_layer = n_layer
    cfg.n_head = n_head
    cfg.n_kv_head = n_head
    cfg.n_embd = n_embd
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = "rotary"
    cfg.rope_variant = "standard"
    cfg.rope_theta = 1000000.0
    cfg.activation = "gelu"
    cfg.norm_layer = "rmsnorm"
    cfg.use_gated_norm = True
    cfg.gated_norm_rank = 4
    cfg.use_hybrid_norm = True
    cfg.rmsnorm_before_qk = True
    cfg.attention = attention
    cfg.use_goat_sink_prior = goat_sink
    cfg.use_composable_gated_attn = composable_gated
    cfg.tie_embed_weights = True
    cfg.zero_init_proj_layers = False
    cfg.use_engram = engram
    cfg.engram_variant = "minimal"
    cfg.engram_ngram = 3
    cfg.engram_vocab_mult = 2
    cfg.use_attn_res = True
    cfg.attn_res_heads = 1
    cfg.attn_res_block_size = 0
    cfg.attn_res_split_qkv = attn_res_split_qkv
    cfg.use_hyper_connections = False
    cfg.use_unet_skips = unet_skips
    cfg.unet_skip_init = 0.25
    cfg.use_oasis_depth_softmax1 = False
    cfg.use_value_residual = value_residual
    cfg.rove = rove
    cfg.hisa = hisa
    return cfg


def _make_model(seed: int = 17, **kw):
    torch.manual_seed(seed)
    return GPT(_tiny_cfg(**kw))


def _batch(seed: int = 0, vocab: int = 64, b: int = 2, t: int = 16):
    torch.manual_seed(seed)
    x = torch.randint(0, vocab, (b, t))
    y = torch.randint(0, vocab, (b, t))
    return x, y


def test_gate_off_default_and_zero_extra_params():
    cfg_default = GPTConfig()
    assert cfg_default.attn_res_split_qkv is False

    m_off = _make_model(attn_res_split_qkv=False)
    assert not hasattr(m_off, "attn_res_w_k")
    assert not hasattr(m_off, "attn_res_w_v")

    m_off2 = _make_model(attn_res_split_qkv=False)
    x, y = _batch()
    l1, loss1 = m_off(x, y)
    l2, loss2 = m_off2(x, y)
    assert torch.equal(l1, l2)
    assert torch.equal(loss1, loss2)


@pytest.mark.parametrize("attention", ["regular", "xsa"])
def test_exact_param_delta(attention):
    n_layer, n_embd = 4, 32
    m_off = _make_model(attn_res_split_qkv=False, attention=attention, n_layer=n_layer, n_embd=n_embd)
    m_on = _make_model(attn_res_split_qkv=True, attention=attention, n_layer=n_layer, n_embd=n_embd)
    n_off = sum(p.numel() for p in m_off.parameters())
    n_on = sum(p.numel() for p in m_on.parameters())
    assert n_on - n_off == 2 * n_layer * n_embd
    assert m_on.attn_res_w_k.shape == (n_layer, n_embd)
    assert m_on.attn_res_w_v.shape == (n_layer, n_embd)
    assert torch.count_nonzero(m_on.attn_res_w_k) == 0
    assert torch.count_nonzero(m_on.attn_res_w_v) == 0


@pytest.mark.parametrize("attention", ["regular", "xsa"])
def test_zero_init_matches_gate_off(attention):
    m_off = _make_model(attn_res_split_qkv=False, attention=attention)
    m_on = _make_model(attn_res_split_qkv=True, attention=attention)
    # Copy shared parameters so both models have identical weights (init RNG consumed extra zeros? no, zeros doesn't use RNG, but load_state_dict guarantees exact equality).
    missing, unexpected = m_on.load_state_dict(m_off.state_dict(), strict=False)
    assert set(missing) == {"attn_res_w_k", "attn_res_w_v"}
    assert unexpected == []

    m_off.eval()
    m_on.eval()
    x, y = _batch()
    with torch.no_grad():
        logits_off, loss_off = m_off(x, y)
        logits_on, loss_on = m_on(x, y)
    assert torch.allclose(logits_off, logits_on, atol=1e-5, rtol=1e-5)
    assert torch.allclose(loss_off, loss_on, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("which", ["k", "v", "both"])
def test_nonzero_split_weights_change_output(which):
    m_zero = _make_model(attn_res_split_qkv=True)
    m_pert = _make_model(attn_res_split_qkv=True)
    m_pert.load_state_dict(m_zero.state_dict())

    with torch.no_grad():
        if which in ("k", "both"):
            m_pert.attn_res_w_k.fill_(0.5)
        if which in ("v", "both"):
            m_pert.attn_res_w_v.fill_(-0.5)

    m_zero.eval()
    m_pert.eval()
    x, y = _batch()
    with torch.no_grad():
        logits_zero, _ = m_zero(x, y)
        logits_pert, _ = m_pert(x, y)
    assert not torch.allclose(logits_zero, logits_pert, atol=1e-4)


@pytest.mark.parametrize("attention", ["regular", "xsa"])
def test_finite_gradients_on_all_attn_res_weights(attention):
    m = _make_model(attn_res_split_qkv=True, attention=attention)
    x, y = _batch()
    _, loss = m(x, y)
    loss.backward()

    for name in ("attn_res_w", "attn_res_w_k", "attn_res_w_v"):
        param = getattr(m, name)
        assert param.grad is not None, f"{name} has no grad"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"
        # Layer 0 has only 1 source (embedding), so softmax over 1 element has zero grad;
        # deeper layers (>= 1) have multiple sources and must receive non-zero grad.
        assert param.grad[1:].abs().sum().item() > 0.0, f"{name}[1:] grad is all zero"

    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{n} has non-finite grad"


def test_triton_flag_default_off_and_cpu_fallback_matches_pytorch():
    assert GPTConfig().attn_res_split_qkv_triton is False

    m_pt = _make_model(attn_res_split_qkv=True)
    cfg_tri = _tiny_cfg(attn_res_split_qkv=True)
    cfg_tri.attn_res_split_qkv_triton = True
    m_tri = GPT(cfg_tri)
    m_tri.load_state_dict(m_pt.state_dict())

    with torch.no_grad():
        m_pt.attn_res_w_k.fill_(0.25)
        m_pt.attn_res_w_v.fill_(-0.25)
        m_tri.attn_res_w_k.fill_(0.25)
        m_tri.attn_res_w_v.fill_(-0.25)

    m_pt.eval()
    m_tri.eval()
    x, y = _batch()
    with torch.no_grad():
        l_pt, loss_pt = m_pt(x, y)
        l_tri, loss_tri = m_tri(x, y)
    assert torch.equal(l_pt, l_tri)
    assert torch.equal(loss_pt, loss_tri)

