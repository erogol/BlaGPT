"""Tests for F112: LoomFormer (arXiv:2611.30634)."""
import json
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bla_gpt") not in sys.path:
    sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model_f112", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules.setdefault("blagpt_model_f112", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig

from best_model_config import BestConfig
from loom import strict_causal_attention
from optimizers import get_optimizer


def _tiny_cfg(*, use_loom: bool = True, loom_m_shift: bool = True, engram: bool = True, n_layer: int = 4):
    cfg = GPTConfig()
    cfg.block_size = 16
    cfg.vocab_size = 64
    cfg.n_layer = n_layer
    cfg.n_head = 4
    cfg.n_kv_head = 4
    cfg.n_embd = 32
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = "rotary"
    cfg.rope_variant = "standard"
    cfg.rope_theta = 10000.0
    cfg.activation = "gelu"
    cfg.norm_layer = "rmsnorm"
    cfg.use_gated_norm = True
    cfg.gated_norm_rank = 4
    cfg.use_hybrid_norm = True
    cfg.attention = "regular"
    cfg.tie_embed_weights = True
    cfg.zero_init_proj_layers = True
    cfg.use_engram = engram
    cfg.engram_variant = "minimal"
    cfg.engram_ngram = 3
    cfg.engram_vocab_mult = 2
    cfg.use_attn_res = False
    cfg.use_unet_skips = False
    cfg.use_value_residual = False
    cfg.use_loom = use_loom
    cfg.loom_d_k = 8
    cfg.loom_d_r = 16
    cfg.loom_rope_dim = 2
    cfg.loom_m_shift = loom_m_shift
    cfg.loom_ffn_x_dim = 64
    cfg.loom_ffn_m_dim = 32
    cfg.loom_last_ffn_x_dim = 96
    cfg.loom_adamw = True
    cfg.loom_lr = 1e-3
    cfg.loom_wd = 0.1
    return cfg


def _make_model(seed: int = 42, **kw):
    torch.manual_seed(seed)
    return GPT(_tiny_cfg(**kw))


def _batch(seed: int = 0, vocab: int = 64, b: int = 2, t: int = 12):
    torch.manual_seed(seed)
    x = torch.randint(0, vocab, (b, t))
    y = torch.randint(0, vocab, (b, t))
    return x, y


def test_gate_off_identity():
    cfg_default = GPTConfig()
    assert cfg_default.use_loom is False

    m1 = _make_model(seed=7, use_loom=False)
    m2 = _make_model(seed=7, use_loom=False)
    assert m1.loom_embed_write is None
    assert not any("loom" in n for n, _ in m1.named_parameters())

    x, y = _batch()
    l1, loss1 = m1(x, y)
    l2, loss2 = m2(x, y)
    assert torch.equal(l1, l2)
    assert torch.equal(loss1, loss2)


def test_strict_causal_attention_matches_reference():
    torch.manual_seed(123)
    B, T, n, d_v, r_dim = 2, 11, 4, 16, 4
    d_qk = d_v + r_dim
    q = torch.randn(B, T, n, d_qk)
    k = torch.randn(B, T, n, d_qk)
    v = torch.randn(B, T, n, d_v)
    scale = 1.0 / math.sqrt(d_v)

    got = strict_causal_attention(q, k, v, scale=scale)
    assert got.shape == (B, T, n, d_v)
    # Row t = 0 has no allowed keys (s < 0 is empty), so it must be exactly zero.
    assert torch.equal(got[:, 0], torch.zeros(B, n, d_v))

    # Explicit reference with strict s < t mask.
    logits = torch.einsum("btnd,bsnd->bnts", q, k) * scale
    s = torch.arange(T)[None, :]
    t = torch.arange(T)[:, None]
    allowed = s < t  # (T, T)
    ref_list = [torch.zeros(B, n, d_v)]
    for ti in range(1, T):
        w = torch.softmax(logits[:, :, ti, :ti], dim=-1)  # (B, n, ti)
        out_t = torch.einsum("bns,bsnd->bnd", w, v[:, :ti])
        ref_list.append(out_t)
    ref = torch.stack(ref_list, dim=1)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)

    # Single-token edge case T=1 -> exact zeros.
    got1 = strict_causal_attention(q[:, :1], k[:, :1], v[:, :1], scale=scale)
    assert torch.equal(got1, torch.zeros(B, 1, n, d_v))


@pytest.mark.parametrize("loom_m_shift", [True, False])
def test_model_strict_causality_and_om_bypass(loom_m_shift):
    m = _make_model(seed=11, use_loom=True, loom_m_shift=loom_m_shift, engram=False)
    m.eval()
    x, y = _batch(seed=3, b=2, t=10)
    with torch.no_grad():
        logits_a, _ = m(x, y)
        x_mod = x.clone()
        x_mod[:, 5:] = (x_mod[:, 5:] + 7) % 64
        logits_b, _ = m(x_mod, y)
    # Tokens 0..4 must be unaffected by changes at positions >= 5.
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-5, rtol=1e-5)
    # Position 5 must change (via o_m bypass and x stream).
    assert not torch.allclose(logits_a[:, 5], logits_b[:, 5])

    # Verify o_m bypass directly at t = 0 (where horizontal attention output is 0):
    # perturbing q_bar_o changes layer 0 LoomMHA output at t = 0.
    x0 = torch.randn(1, 4, 32)
    M0 = m.loom_embed_write(x0)
    with torch.no_grad():
        dx_before, _ = m.transformer.h[0].attn(x0, M0)
        m.transformer.h[0].attn.q_bar_o.add_(0.5)
        dx_after, _ = m.transformer.h[0].attn(x0, M0)
    assert not torch.allclose(dx_before[:, 0], dx_after[:, 0])


def test_appendix_d1_near_zero_gate_init_and_stability():
    m = _make_model(seed=19, use_loom=True, loom_m_shift=True)
    # EmbedWrite gate: weight=0, sigmoid(bias)=0.1
    assert torch.count_nonzero(m.loom_embed_write.W_wg.weight) == 0
    assert torch.allclose(torch.sigmoid(m.loom_embed_write.W_wg.bias), torch.full((4,), 0.1), atol=1e-5)

    for i, blk in enumerate(m.transformer.h):
        attn = blk.attn
        # MShift: weight=0, sigmoid(bias)=0.5
        assert torch.count_nonzero(attn.W_sg.weight) == 0
        assert torch.allclose(torch.sigmoid(attn.W_sg.bias), torch.full((8,), 0.5), atol=1e-5)
        # Dynamic/static read gates
        for arm in ("q", "k", "v", "o"):
            w_rg = getattr(attn, f"W_rg_{arm}")
            w_rg_bar = getattr(attn, f"W_rg_bar_{arm}")
            q_bar = getattr(attn, f"q_bar_{arm}")
            assert torch.count_nonzero(w_rg.weight) == 0
            assert torch.allclose(torch.sigmoid(w_rg.bias), torch.full((4,), 0.05), atol=1e-5)
            assert torch.count_nonzero(w_rg_bar.weight) == 0
            assert torch.allclose(torch.sigmoid(w_rg_bar.bias), torch.full((4,), 0.9), atol=1e-5)
            if arm == "v":
                assert torch.count_nonzero(q_bar) > 0
            else:
                assert torch.count_nonzero(q_bar) == 0
        # VO clone init
        assert torch.equal(attn.W_r2_v.weight, attn.W_r2_o.weight)
        # Write gates on layers 0..L-2, omitted on last layer
        if i < len(m.transformer.h) - 1:
            assert blk.write_M and attn.write_M
            assert torch.count_nonzero(attn.W_wg.weight) == 0
            assert torch.allclose(torch.sigmoid(attn.W_wg.bias), torch.full((4,), 0.1), atol=1e-5)
            assert torch.count_nonzero(blk.ffn_W_wg.weight) == 0
            assert torch.allclose(torch.sigmoid(blk.ffn_W_wg.bias), torch.full((4,), 0.1), atol=1e-5)
        else:
            assert not blk.write_M and not attn.write_M and blk.ffn_m is None


def test_finite_nonzero_gradients_and_aurora_routing():
    m = _make_model(seed=23, use_loom=True, loom_m_shift=True, engram=True)
    x, y = _batch(seed=5, b=2, t=12)
    logits, loss = m(x, y)
    assert torch.isfinite(loss)
    loss.backward()

    zero_grad_params = []
    for name, p in m.named_parameters():
        assert p.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"
        # W_down in GatedNorm and c_fc in zero-init FFN_x have zero grad on step 0 by design
        # (W_up=0 in GatedNorm, c_proj=0 in FFN_x); also W_r2_{q,k,o} have tiny/non-zero grads.
        if p.grad.abs().max().item() == 0.0:
            zero_grad_params.append(name)

    # Every Loom port/gate/embed/write parameter must have non-zero grad on step 0
    # except W_r1/W_r2_q/W_r2_k/W_r2_o only if upstream is zero; check key Loom params explicitly.
    for key in (
        "loom_embed_write.W_e.weight",
        "loom_embed_write.W_wg.bias",
        "transformer.h.0.attn.W_o.weight",
        "transformer.h.0.attn.W_sg.bias",
        "transformer.h.0.attn.W_rg_v.bias",
        "transformer.h.0.attn.q_bar_v",
        "transformer.h.0.attn.W_wg.bias",
        "transformer.h.0.ffn_m.c_proj.weight",
        "transformer.h.0.ffn_W_wg.bias",
    ):
        assert key not in zero_grad_params, f"Expected non-zero grad for {key}"

    # Verify Aurora optimizer routing: Loom port/gate params -> AdamW (loom_lr),
    # main matrix weights (W_o, W_e, W_q, W_k, ffn_m) -> Aurora.
    opt = get_optimizer(
        "Aurora",
        {"weight_decay": 0.025, "mu": 0.95, "nesterov": True, "pp_iterations": 2, "pp_beta": 0.5},
        lr=0.03,
        model=m,
    )
    param_by_name = dict(m.named_parameters())
    assert opt.state[param_by_name["transformer.h.0.attn.W_o.weight"]]["use_aurora"] is True
    assert opt.state[param_by_name["loom_embed_write.W_e.weight"]]["use_aurora"] is True
    assert opt.state[param_by_name["transformer.h.0.attn.W_q.weight"]]["use_aurora"] is True
    assert opt.state[param_by_name["transformer.h.0.attn.W_rg_q.weight"]]["use_aurora"] is False
    assert opt.state[param_by_name["transformer.h.0.attn.W_sg.weight"]]["use_aurora"] is False
    assert opt.state[param_by_name["transformer.h.0.attn.W_r1.weight"]]["use_aurora"] is False
    assert opt.state[param_by_name["transformer.h.0.ffn_W_w1.weight"]]["use_aurora"] is False
    # Check one optimizer step stays finite.
    opt.step()
    for name, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"Non-finite param after step: {name}"


def test_f112a_param_count_matches_f111a():
    import mlps

    orig_init = mlps.semi_orthogonal_init
    mlps.semi_orthogonal_init = lambda din, dout, steps=5: torch.zeros(dout, din)
    try:
        def _count(cfg_path: Path):
            cfg = BestConfig()
            with open(cfg_path) as f:
                for k, v in json.load(f).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            m = GPT(cfg)
            total = sum(p.numel() for p in m.parameters())
            engram = sum(p.numel() for p in m.engram.parameters()) if m.engram is not None else 0
            return total, total - engram

        f111_tot, f111_ne = _count(ROOT / "ar" / "full_train" / "configs" / "F111A.json")
        f112_tot, f112_ne = _count(ROOT / "ar" / "full_train" / "configs" / "F112A.json")
    finally:
        mlps.semi_orthogonal_init = orig_init

    rel_diff = abs(f112_ne - f111_ne) / f111_ne
    assert rel_diff < 0.005, f"Non-Engram param mismatch: F112A={f112_ne:,} vs F111A={f111_ne:,} ({rel_diff:.4%})"
    assert abs(f112_tot - f111_tot) / f111_tot < 0.005
