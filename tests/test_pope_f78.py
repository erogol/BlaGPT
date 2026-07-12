"""F78 — paper-faithful PoPE (Polar Coordinate Positional Embedding).

Covers: default rotary path unchanged, PoPE construction, d-frequency count,
zero/bounded phase bias, numerical equivalence of the doubled-vector dot product
to PoPE Eq. 6, output shape/gradient, and config validation + XSA/GOAT
compatibility. All CPU, no training metrics involved.

Paper: Gopalakrishnan, Csordas, Schmidhuber & Mozer, arXiv:2509.10534.
"""

import math

import torch
import torch.nn.functional as F
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SPEC = spec_from_file_location(
    "blagpt_model", Path(__file__).resolve().parents[1] / "bla_gpt" / "bla_gpt.py"
)
_BLA = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BLA)

GPT = _BLA.GPT
GPTConfig = _BLA.GPTConfig
get_attention = _BLA.get_attention


def tiny_config(pos_encoding="pope", attention="regular", n_head=2, n_kv_head=2):
    cfg = GPTConfig()
    cfg.block_size = 8
    cfg.vocab_size = 64
    cfg.n_layer = 2
    cfg.n_head = n_head
    cfg.n_kv_head = n_kv_head
    cfg.n_embd = 16
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = pos_encoding
    cfg.rope_theta = 10000.0
    cfg.activation = "gelu"
    cfg.norm_layer = "layernorm"
    cfg.attention = attention
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.rmsnorm_before_qk = False
    cfg.use_engram = False
    return cfg


# --- 1. Default rotary path is untouched by the PoPE addition -----------------

def test_default_rotary_path_unchanged():
    attn = get_attention(tiny_config(pos_encoding="rotary"))
    assert hasattr(attn, "rotary")
    assert not getattr(attn, "use_pope", False)
    assert not hasattr(attn, "pope_freqs")
    assert not hasattr(attn, "pope_delta")

    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    y = attn(x)
    assert y.shape == (2, 5, 16)


def test_rotary_forward_is_bitwise_stable_after_change():
    # A fixed-seed rotary forward must be reproducible (no accidental coupling
    # to the new PoPE branch).
    torch.manual_seed(123)
    attn = get_attention(tiny_config(pos_encoding="rotary"))
    x = torch.randn(1, 6, 16)
    y1 = attn(x)
    y2 = attn(x)
    assert torch.equal(y1, y2)


# --- 2. PoPE construction -----------------------------------------------------

def test_pope_construction_attributes():
    attn = get_attention(tiny_config(pos_encoding="pope"))
    assert getattr(attn, "use_pope", False) is True
    assert hasattr(attn, "pope_freqs")
    assert isinstance(attn.pope_delta, torch.nn.Parameter)
    # PoPE must NOT create the rotary buffers.
    assert not hasattr(attn, "rotary")
    assert not hasattr(attn, "rel_pos_emb")


# --- 3. Frequency count is d (not d/2) ---------------------------------------

def test_pope_uses_d_frequencies():
    cfg = tiny_config(pos_encoding="pope")
    attn = get_attention(cfg)
    d = cfg.n_embd // cfg.n_head  # head_dim
    assert attn.pope_freqs.shape == (d,)  # d frequencies, RoPE would be d/2
    expected = cfg.rope_theta ** (-torch.arange(0, d, dtype=torch.float32) / d)
    assert torch.allclose(attn.pope_freqs, expected, atol=1e-6)


# --- 4. Phase bias delta: zero-init + bounded to [-2*pi, 0] -------------------

def test_pope_delta_zero_initialized():
    cfg = tiny_config(pos_encoding="pope")
    attn = get_attention(cfg)
    d = cfg.n_embd // cfg.n_head
    assert attn.pope_delta.shape == (cfg.n_kv_head, d)
    assert torch.equal(attn.pope_delta, torch.zeros(cfg.n_kv_head, d))


def test_pope_delta_clamped_to_bounds():
    torch.manual_seed(1)
    cfg = tiny_config(pos_encoding="pope")
    attn = get_attention(cfg)
    q = torch.randn(1, 4, cfg.n_head, cfg.n_embd // cfg.n_head)
    k = torch.randn(1, 4, cfg.n_kv_head, cfg.n_embd // cfg.n_head)

    # Upper bound: any delta > 0 must behave exactly like delta == 0.
    with torch.no_grad():
        attn.pope_delta.fill_(5.0)
    _, k_pos = attn._apply_pope(q, k, 4, 4)
    with torch.no_grad():
        attn.pope_delta.zero_()
    _, k_zero = attn._apply_pope(q, k, 4, 4)
    assert torch.allclose(k_pos, k_zero, atol=1e-6)

    # Lower bound: delta < -2*pi must behave exactly like delta == -2*pi.
    with torch.no_grad():
        attn.pope_delta.fill_(-10.0)  # < -2*pi
    _, k_low = attn._apply_pope(q, k, 4, 4)
    with torch.no_grad():
        attn.pope_delta.fill_(-2.0 * math.pi)
    _, k_bound = attn._apply_pope(q, k, 4, 4)
    assert torch.allclose(k_low, k_bound, atol=1e-6)


# --- 5. Numerical equivalence of doubled dot product to PoPE Eq. 6 -----------

def test_pope_logits_match_eq6():
    torch.manual_seed(2)
    cfg = tiny_config(pos_encoding="pope")
    attn = get_attention(cfg)
    d = cfg.n_embd // cfg.n_head
    B, T, H = 2, 6, cfg.n_head

    # Non-trivial delta, including out-of-range values to exercise the clamp.
    with torch.no_grad():
        attn.pope_delta.uniform_(-2.0 * math.pi, 0.0)
        attn.pope_delta[0, 0] = 3.0    # clamps to 0
        attn.pope_delta[1, 1] = -12.0  # clamps to -2*pi

    q = torch.randn(B, T, H, d)
    k = torch.randn(B, T, H, d)
    q2, k2 = attn._apply_pope(q, k, T, T)
    assert q2.shape == (B, T, H, 2 * d)
    assert k2.shape == (B, T, H, 2 * d)

    # Doubled dot product: score[b,h,t,s] = <q2[b,t,h], k2[b,s,h]>.
    doubled = torch.einsum("bthc,bshc->bhts", q2, k2)

    # Reference: Eq. 6 directly, score = sum_c softplus(q) softplus(k)
    #            cos((s - t) theta_c + delta_c), delta clamped to [-2*pi, 0].
    theta = attn.pope_freqs.double()                      # (d,)
    delta = attn.pope_delta.clamp(-2.0 * math.pi, 0.0).double()  # (H, d)
    mu_q = F.softplus(q.double())                         # (B,T,H,d)
    mu_k = F.softplus(k.double())
    pos = torch.arange(T, dtype=torch.float64)
    # (s - t) for every (t, s) pair
    st = pos[None, :] - pos[:, None]                      # (T_t, T_s)
    # phase[h,t,s,c] = (s - t) theta_c + delta[h,c]
    phase = st[None, :, :, None] * theta[None, None, None, :] + delta[:, None, None, :]
    cos = torch.cos(phase)                                # (H,T,T,d)
    ref = torch.einsum("bthc,bshc,htsc->bhts", mu_q, mu_k, cos)

    assert torch.allclose(doubled.double(), ref, atol=1e-4, rtol=1e-4)


# --- 6. Output shape + gradient flow -----------------------------------------

def test_pope_output_shape_and_gradient():
    torch.manual_seed(3)
    cfg = tiny_config(pos_encoding="pope")
    attn = get_attention(cfg)
    x = torch.randn(2, 5, cfg.n_embd, requires_grad=True)
    y = attn(x)
    assert y.shape == (2, 5, cfg.n_embd)  # v/output dim preserved
    loss = y.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # Phase bias receives gradient (it is zero-init but in the interior of the
    # clamp at 0, so gradient flows).
    assert attn.pope_delta.grad is not None
    assert torch.isfinite(attn.pope_delta.grad).all()


# --- 7. Config validation + XSA / GOAT compatibility -------------------------

def test_unknown_pos_encoding_still_rejected():
    cfg = tiny_config(pos_encoding="bogus")
    try:
        get_attention(cfg)
    except ValueError:
        return
    raise AssertionError("unknown pos_encoding must raise ValueError")


def test_pope_composes_with_plain_xsa():
    attn = get_attention(tiny_config(pos_encoding="pope", attention="xsa"))
    assert attn.__class__.__name__ == "ExclusiveSelfAttention"
    assert getattr(attn, "use_pope", False) is True
    torch.manual_seed(4)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = attn(x)
    assert y.shape == (2, 5, 16)
    y.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_pope_composes_with_goat_sink_xsa():
    cfg = tiny_config(pos_encoding="pope", attention="xsa")
    cfg.use_goat_sink_prior = True
    attn = get_attention(cfg)
    assert attn.__class__.__name__ == "GOATSinkAttention"
    assert getattr(attn, "use_pope", False) is True
    torch.manual_seed(5)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = attn(x)
    assert y.shape == (2, 5, 16)
    y.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert attn.pope_delta.grad is not None


def test_full_gpt_pope_xsa_goat_forward_backward():
    # End-to-end: F78-shaped stack (XSA + GOAT sink + PoPE) trains one step.
    cfg = tiny_config(pos_encoding="pope", attention="xsa")
    cfg.use_goat_sink_prior = True
    torch.manual_seed(6)
    model = GPT(cfg)
    # PoPE replaces absolute position embeddings (wpe), like rotary.
    assert model.transformer.wpe is None
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
