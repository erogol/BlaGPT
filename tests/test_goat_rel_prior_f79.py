"""F79 — GOAT relative spectral prior (arXiv:2601.15380 Sec. 6, Litman & Guo 2026).

Covers: gate-off dispatch unchanged, gate-on dispatch, zero-init identity with
GOATSinkAttention, translation equivariance of the relative log-prior,
frequency-ladder construction, gradient flow to alpha/beta, output change on
perturbation, state_dict roundtrip, and full-GPT forward/backward with the
F79 stack. All CPU, no training metrics involved.
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
GOATSinkAttention = _BLA.GOATSinkAttention
GOATRelPriorAttention = _BLA.GOATRelPriorAttention


def tiny_config(rel=True, sink=True, n_head=2):
    cfg = GPTConfig()
    cfg.block_size = 8
    cfg.vocab_size = 64
    cfg.n_layer = 2
    cfg.n_head = n_head
    cfg.n_kv_head = n_head
    cfg.n_embd = 16
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = "rotary"
    cfg.rope_theta = 10000.0
    cfg.activation = "gelu"
    cfg.norm_layer = "layernorm"
    cfg.attention = "xsa"
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.rmsnorm_before_qk = False
    cfg.use_engram = False
    cfg.use_goat_sink_prior = sink
    cfg.use_goat_rel_prior = rel
    cfg.goat_rel_num_freqs = 4
    return cfg


# --- 1. Dispatch ---------------------------------------------------------------

def test_gate_off_returns_sink_attention():
    attn = get_attention(tiny_config(rel=False, sink=True))
    assert type(attn) is GOATSinkAttention


def test_gate_on_returns_rel_prior_attention():
    attn = get_attention(tiny_config(rel=True, sink=True))
    assert type(attn) is GOATRelPriorAttention


# --- 2. Construction -----------------------------------------------------------

def test_frequency_ladder():
    attn = get_attention(tiny_config())
    R = 4
    assert attn.goat_rel_freqs.shape == (R,)
    expected = torch.tensor([10000.0 ** (-r / (R - 1)) for r in range(R)])
    assert torch.allclose(attn.goat_rel_freqs, expected)
    assert attn.rel_alpha.shape == (2, R)
    assert attn.rel_beta.shape == (2, R)
    assert torch.all(attn.rel_alpha == 0) and torch.all(attn.rel_beta == 0)


def test_rel_log_prior_zero_at_init():
    attn = get_attention(tiny_config())
    K = attn._rel_log_prior(8, 8, torch.device("cpu"))
    assert K.shape == (2, 8, 8)
    assert torch.all(K == 0)


# --- 3. Zero-init identity with GOATSinkAttention -------------------------------

def test_zero_init_matches_sink_attention():
    torch.manual_seed(0)
    cfg = tiny_config(rel=True)
    rel_attn = get_attention(cfg)
    sink_attn = get_attention(tiny_config(rel=False))
    sink_sd = {k: v for k, v in rel_attn.state_dict().items()
               if k in sink_attn.state_dict()}
    sink_attn.load_state_dict(sink_sd)
    x = torch.randn(2, 8, 16)
    rel_attn.eval(); sink_attn.eval()
    with torch.no_grad():
        assert torch.allclose(rel_attn(x), sink_attn(x), atol=1e-6)


# --- 4. Translation equivariance -------------------------------------------------

def test_rel_prior_translation_equivariant():
    attn = get_attention(tiny_config())
    with torch.no_grad():
        attn.rel_alpha.normal_(); attn.rel_beta.normal_()
    K = attn._rel_log_prior(8, 8, torch.device("cpu"))
    for h in range(K.shape[0]):
        for d in range(-3, 4):
            idxs = [i for i in range(8) if 0 <= i - d < 8]
            vals = torch.tensor([K[h, i, i - d] for i in idxs])
            assert torch.allclose(vals, vals[0].expand_as(vals), atol=1e-5)


# --- 5. Perturbation changes output ----------------------------------------------

def test_alpha_perturbation_changes_output():
    torch.manual_seed(0)
    attn = get_attention(tiny_config())
    attn.eval()
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y0 = attn(x)
        attn.rel_alpha.fill_(1.0)
        y1 = attn(x)
    assert not torch.allclose(y0, y1)


def test_beta_perturbation_changes_output():
    torch.manual_seed(0)
    attn = get_attention(tiny_config())
    attn.eval()
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        y0 = attn(x)
        attn.rel_beta.fill_(1.0)
        y1 = attn(x)
    assert not torch.allclose(y0, y1)


# --- 6. Gradient flow -------------------------------------------------------------

def test_gradients_reach_alpha_beta_and_sink():
    torch.manual_seed(0)
    attn = get_attention(tiny_config())
    x = torch.randn(2, 8, 16)
    attn(x).sum().backward()
    assert attn.rel_alpha.grad is not None and attn.rel_alpha.grad.abs().sum() > 0
    assert attn.rel_beta.grad is not None and attn.rel_beta.grad.abs().sum() > 0
    assert attn.sink_prior.grad is not None


# --- 7. state_dict roundtrip -------------------------------------------------------

def test_state_dict_roundtrip():
    attn = get_attention(tiny_config())
    with torch.no_grad():
        attn.rel_alpha.normal_(); attn.rel_beta.normal_()
    sd = attn.state_dict()
    assert "rel_alpha" in sd and "rel_beta" in sd and "sink_prior" in sd
    assert "goat_rel_freqs" not in sd  # non-persistent buffer
    attn2 = get_attention(tiny_config())
    attn2.load_state_dict(sd)
    assert torch.allclose(attn2.rel_alpha, attn.rel_alpha)
    assert torch.allclose(attn2.rel_beta, attn.rel_beta)


# --- 8. Shape/dtype ---------------------------------------------------------------

def test_output_shape_dtype():
    attn = get_attention(tiny_config())
    x = torch.randn(3, 8, 16)
    y = attn(x)
    assert y.shape == x.shape and y.dtype == x.dtype


# --- 9. Full GPT forward/backward with the F79 stack -------------------------------

def test_full_gpt_forward_backward():
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.use_gated_norm = True
    cfg.gated_norm_rank = 4
    model = GPT(cfg)
    idx = torch.randint(0, 64, (2, 8))
    out = model(idx, targets=idx)
    loss = out[1] if isinstance(out, tuple) else out.loss
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "rel_alpha" in n]
    assert grads and all(g is not None for g in grads)
    assert torch.isfinite(loss)
