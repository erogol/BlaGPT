"""Tests for E6: Multi-Head Attention Residuals (arXiv:2607.27230).

CPU-only, tiny model. Covers: H=1 exact equivalence to the original
single-head attention-residual routing, H=4/H=8 finiteness, causality,
parameter-count invariance across H, and checkpoint round trip.
"""
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

_SPEC = spec_from_file_location("blagpt_model_e6", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig

DEV = "cpu"


def tiny_config(attn_res_heads=1, n_layer=3, n_embd=32, n_head=4):
    cfg = GPTConfig()
    cfg.block_size = 16
    cfg.vocab_size = 64
    cfg.n_layer = n_layer
    cfg.n_head = n_head
    cfg.n_kv_head = n_head
    cfg.n_embd = n_embd
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = "none"
    cfg.activation = "gelu"
    cfg.norm_layer = "layernorm"
    cfg.attention = "xsa"
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.use_engram = False
    cfg.use_attn_res = True
    cfg.attn_res_heads = attn_res_heads
    cfg.attn_res_block_size = 0
    cfg.use_hyper_connections = False
    cfg.use_unet_skips = False
    cfg.use_oasis_depth_softmax1 = False
    cfg.use_value_residual = False
    cfg.use_composable_gated_attn = False
    cfg.use_goat_sink_prior = False
    cfg.use_affine_scaled_attn = False
    return cfg


def _mk(attn_res_heads, seed=17, **kw):
    torch.manual_seed(seed)
    return GPT(tiny_config(attn_res_heads=attn_res_heads, **kw)).to(DEV)


def _batch(seed=0, vocab=64, b=2, t=16):
    torch.manual_seed(seed)
    x = torch.randint(0, vocab, (b, t))
    y = torch.randint(0, vocab, (b, t))
    return x, y


def _fwd_logits_loss(m, x, y):
    logits, loss = m(x, y)
    total = loss["total"] if isinstance(loss, dict) else loss
    return logits, total


def _reference_single_head_mix(model, _srcs, _w):
    """Original (pre-MHAR) single-head attention-residual routing, Eq.2 of
    arXiv:2607.27230 / the prior bla_gpt.py code. Reimplemented here,
    independent of GPT._attn_res_mix, to check H=1 equivalence honestly."""
    _scores = torch.stack(
        [(F.rms_norm(v, (v.size(-1),)) * _w).sum(-1) for v in _srcs], dim=0
    )
    _alpha = model._attn_res_route(_scores)
    x = _srcs[0] * _alpha[0].unsqueeze(-1)
    for _i in range(1, len(_srcs)):
        x = x + _srcs[_i] * _alpha[_i].unsqueeze(-1)
    return x


def test_h1_matches_reference_single_head_forward_and_grad_bitexact():
    m_new = _mk(1, seed=42)
    m_ref = _mk(1, seed=42)
    # m_ref uses an independent reimplementation of the pre-MHAR formula,
    # bypassing GPT._attn_res_mix entirely.
    m_ref._attn_res_mix = types.MethodType(
        lambda self, srcs, w: _reference_single_head_mix(self, srcs, w), m_ref
    )

    x, y = _batch(seed=7)
    logits_new, loss_new = _fwd_logits_loss(m_new, x, y)
    logits_ref, loss_ref = _fwd_logits_loss(m_ref, x, y)

    assert torch.equal(logits_new, logits_ref)
    assert torch.equal(loss_new, loss_ref)

    loss_new.backward()
    loss_ref.backward()
    for (n1, p1), (n2, p2) in zip(m_new.named_parameters(), m_ref.named_parameters()):
        assert n1 == n2
        if p1.grad is None and p2.grad is None:
            continue
        assert p1.grad is not None and p2.grad is not None, n1
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6, rtol=0), n1


def test_h1_default_equals_explicit_h1():
    # attn_res_heads defaults to 1: the default config must give H=1.
    assert GPTConfig().attn_res_heads == 1


def test_h4_h8_forward_and_grad_finite():
    for H in (4, 8):
        m = _mk(H, seed=3)
        x, y = _batch(seed=11)
        logits, loss = _fwd_logits_loss(m, x, y)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(loss).all()
        loss.backward()
        assert m.attn_res_w.grad is not None
        assert torch.isfinite(m.attn_res_w.grad).all()
        for n, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (H, n)


def test_param_count_invariant_across_heads():
    counts = {}
    for H in (1, 4, 8):
        m = _mk(H, seed=99)
        counts[H] = sum(p.numel() for p in m.parameters())
        assert m.attn_res_w.numel() == (m.config.n_layer + 1) * m.config.n_embd
    assert counts[1] == counts[4] == counts[8]


def test_invalid_head_count_raises():
    m = _mk(3, seed=5)  # n_embd=32 not divisible by 3
    x, y = _batch(seed=1)
    try:
        m(x, y)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError for n_embd % attn_res_heads != 0")


def test_causality_suffix_change_does_not_affect_earlier_logits():
    m = _mk(4, seed=21)
    torch.manual_seed(2)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits_a, _ = m(x, y)

    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 1) % 64  # change only a later suffix
    logits_b, _ = m(x2, y)

    assert torch.equal(logits_a[:, :10, :], logits_b[:, :10, :])
    assert not torch.equal(logits_a[:, 10:, :], logits_b[:, 10:, :])


def test_checkpoint_save_load_round_trip(tmp_path):
    m = _mk(4, seed=55)
    x, y = _batch(seed=6)
    logits_before, _ = m(x, y)

    ckpt_path = tmp_path / "e6_ckpt.pt"
    torch.save(m.state_dict(), ckpt_path)

    m2 = GPT(tiny_config(attn_res_heads=4))
    m2.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    logits_after, _ = m2(x, y)

    assert torch.equal(logits_before, logits_after)
