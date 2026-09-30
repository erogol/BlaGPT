"""Tests for RoVE (rotary value embeddings, arXiv:2606.11275).

RoVE rotates each value by its own source-position rotation before
aggregation, then applies the inverse query-position rotation to the
attention output before the output projection. Config gate: `rove`
(default False = unchanged behavior).
"""
import subprocess
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

_BLA_SPEC = spec_from_file_location(
    "blagpt_model_rove", REPO_ROOT / "bla_gpt" / "bla_gpt.py"
)
_BLA = module_from_spec(_BLA_SPEC)
assert _BLA_SPEC.loader is not None
_BLA_SPEC.loader.exec_module(_BLA)

_ATT_SPEC = spec_from_file_location(
    "blagpt_attentions_rove", REPO_ROOT / "bla_gpt" / "attentions.py"
)
_ATT = module_from_spec(_ATT_SPEC)
assert _ATT_SPEC.loader is not None
_ATT_SPEC.loader.exec_module(_ATT)

GPTConfig = _BLA.GPTConfig
get_attention = _BLA.get_attention
Attention = _ATT.Attention
apply_rotary_emb = _ATT.apply_rotary_emb
Rotary = _ATT.Rotary


def _tiny_cfg(pos_encoding="rotary", rove=False, n_head=4, n_kv_head=4, n_embd=16):
    cfg = GPTConfig()
    cfg.block_size = 16
    cfg.vocab_size = 64
    cfg.n_layer = 2
    cfg.n_head = n_head
    cfg.n_kv_head = n_kv_head
    cfg.n_embd = n_embd
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = pos_encoding
    cfg.rope_theta = 10000.0
    cfg.rope_variant = "standard"
    cfg.activation = "gelu"
    cfg.norm_layer = "layernorm"
    cfg.attention = "regular"
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.use_engram = False
    cfg.rove = rove
    return cfg


def _load_baseline_attentions_module():
    """Load bla_gpt/attentions.py as it stood at the repo HEAD commit (before
    this RoVE change), so rove=False can be checked against truly unchanged
    code, not just the current file re-read."""
    src = subprocess.run(
        ["git", "show", "HEAD:bla_gpt/attentions.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    tmp_dir = tempfile.mkdtemp()
    path = Path(tmp_dir) / "attentions_baseline_rove.py"
    path.write_text(src)
    spec = spec_from_file_location("attentions_baseline_rove", path)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. rove=False: output and gradients match the unchanged (pre-RoVE) code.
# ---------------------------------------------------------------------------

def test_rove_false_matches_baseline_forward_and_grad():
    baseline = _load_baseline_attentions_module()
    cfg = _tiny_cfg(pos_encoding="rotary", rove=False)

    torch.manual_seed(42)
    attn_old = baseline.Attention(cfg)
    torch.manual_seed(42)
    attn_new = Attention(cfg)

    old_params = dict(attn_old.named_parameters())
    new_params = dict(attn_new.named_parameters())
    assert set(old_params) == set(new_params)
    for name in old_params:
        assert torch.allclose(old_params[name], new_params[name]), name

    x1 = torch.randn(2, 6, cfg.n_embd, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y_old = attn_old(x1)
    y_new = attn_new(x2)
    assert torch.allclose(y_old, y_new, atol=1e-6)

    y_old.sum().backward()
    y_new.sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6)
    for name in old_params:
        g_old, g_new = old_params[name].grad, new_params[name].grad
        if g_old is None:
            assert g_new is None, name
        else:
            assert torch.allclose(g_old, g_new, atol=1e-6), name


def test_rove_attr_absent_when_off():
    attn = Attention(_tiny_cfg(pos_encoding="rotary", rove=False))
    assert attn.rove is False
    assert not hasattr(attn, "rove_rotary")


# ---------------------------------------------------------------------------
# 2. rove=True: finite forward and gradients (plain RoPE, and GRAPE hybrid).
# ---------------------------------------------------------------------------

def test_rove_true_finite_forward_and_grad_rotary():
    attn = Attention(_tiny_cfg(pos_encoding="rotary", rove=True))
    x = torch.randn(2, 7, attn.n_embd, requires_grad=True)
    y = attn(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in attn.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_rove_true_finite_forward_and_grad_grape_hybrid():
    attn = Attention(_tiny_cfg(pos_encoding="grape_a_qgate", rove=True))
    x = torch.randn(2, 7, attn.n_embd, requires_grad=True)
    y = attn(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_rove_true_changes_output_vs_off():
    torch.manual_seed(0)
    attn_off = Attention(_tiny_cfg(pos_encoding="rotary", rove=False))
    torch.manual_seed(0)
    attn_on = Attention(_tiny_cfg(pos_encoding="rotary", rove=True))
    x = torch.randn(2, 7, attn_off.n_embd)
    y_off = attn_off(x)
    y_on = attn_on(x)
    assert not torch.allclose(y_off, y_on)


def test_rove_simplified_variant_rejected():
    import pytest
    cfg = _tiny_cfg(pos_encoding="rotary", rove=True)
    cfg.rope_variant = "simplified"
    with pytest.raises(ValueError):
        Attention(cfg)


def test_rove_unsupported_pos_encoding_rejected():
    import pytest
    cfg = _tiny_cfg(pos_encoding="none", rove=True)
    with pytest.raises(ValueError):
        Attention(cfg)


# ---------------------------------------------------------------------------
# 3. Relative-position property: shifting every position by a constant k
#    leaves the RoVE-aggregated value unchanged (core math, no q/k/v proj).
# ---------------------------------------------------------------------------

def test_rove_relative_position_invariance():
    """y_i = R_i^-1 sum_j A_ij R_j v_j depends on positions only through the
    relative offsets (j - i), so shifting all absolute positions i, j by the
    same constant k must leave y_i unchanged for fixed A and v."""
    torch.manual_seed(0)
    head_dim = 8
    T = 5
    k_shift = 37
    rotary = Rotary(head_dim, base=10000.0, seq_len=64)

    v = torch.randn(1, 1, T, head_dim)  # (B, H, T, D), fixed values per position
    A = torch.softmax(torch.randn(1, 1, T, T), dim=-1)  # fixed attention weights

    def rove_output(pos_start):
        idx = torch.arange(pos_start, pos_start + T)
        cos = rotary.cos[idx][None, None, :, :]
        sin = rotary.sin[idx][None, None, :, :]
        v_rot = apply_rotary_emb(v, cos, sin)          # R_j v_j
        agg = A @ v_rot                                  # sum_j A_ij R_j v_j
        y = apply_rotary_emb(agg, cos, -sin)              # R_i^{-1} (...)
        return y

    y_at_0 = rove_output(0)
    y_at_k = rove_output(k_shift)
    assert torch.allclose(y_at_0, y_at_k, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Causality: changing a later token must not change earlier outputs.
# ---------------------------------------------------------------------------

def test_rove_causality_rotary():
    torch.manual_seed(0)
    attn = Attention(_tiny_cfg(pos_encoding="rotary", rove=True))
    attn.eval()
    T = 6
    x1 = torch.randn(1, T, attn.n_embd)
    x2 = x1.clone()
    x2[:, T // 2 :, :] = torch.randn_like(x2[:, T // 2 :, :])  # change suffix only

    with torch.no_grad():
        y1 = attn(x1)
        y2 = attn(x2)

    assert torch.allclose(y1[:, : T // 2], y2[:, : T // 2], atol=1e-6)
    assert not torch.allclose(y1[:, T // 2 :], y2[:, T // 2 :])


def test_rove_causality_grape_hybrid():
    torch.manual_seed(0)
    attn = Attention(_tiny_cfg(pos_encoding="grape_a_qgate", rove=True))
    attn.eval()
    T = 6
    x1 = torch.randn(1, T, attn.n_embd)
    x2 = x1.clone()
    x2[:, T // 2 :, :] = torch.randn_like(x2[:, T // 2 :, :])

    with torch.no_grad():
        y1 = attn(x1)
        y2 = attn(x2)

    assert torch.allclose(y1[:, : T // 2], y2[:, : T // 2], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Save / load round trip.
# ---------------------------------------------------------------------------

def test_rove_save_load_round_trip():
    torch.manual_seed(0)
    attn = Attention(_tiny_cfg(pos_encoding="rotary", rove=True))
    x = torch.randn(2, 5, attn.n_embd)
    with torch.no_grad():
        y_before = attn(x)

    state = attn.state_dict()
    attn2 = Attention(_tiny_cfg(pos_encoding="rotary", rove=True))
    attn2.load_state_dict(state)
    attn2.eval()
    attn.eval()
    with torch.no_grad():
        y_after = attn2(x)

    assert torch.allclose(y_before, y_after, atol=1e-6)


def test_rove_grape_hybrid_save_load_round_trip():
    torch.manual_seed(0)
    attn = Attention(_tiny_cfg(pos_encoding="grape_a_qgate", rove=True))
    x = torch.randn(2, 5, attn.n_embd)
    attn.eval()
    with torch.no_grad():
        y_before = attn(x)

    state = attn.state_dict()
    attn2 = Attention(_tiny_cfg(pos_encoding="grape_a_qgate", rove=True))
    attn2.load_state_dict(state)
    attn2.eval()
    with torch.no_grad():
        y_after = attn2(x)

    assert torch.allclose(y_before, y_after, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. get_attention() wiring: xsa stack (ComposableGatedAttention /
#    GOATSinkAttention / ExclusiveSelfAttention) also honors rove.
# ---------------------------------------------------------------------------

def test_rove_through_get_attention_xsa_stack():
    cfg = _tiny_cfg(pos_encoding="rotary", rove=True)
    cfg.attention = "xsa"
    cfg.use_goat_sink_prior = True
    cfg.use_composable_gated_attn = True
    attn = get_attention(cfg)
    assert attn.rove is True
    x = torch.randn(2, 5, cfg.n_embd, requires_grad=True)
    y = attn(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
