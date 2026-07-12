"""F79 — paper-faithful NAG (Norm-AGnostic residual network).

Covers: gate-off default is unchanged (no NAG modules/params, base additive
residual path), NAG construction, the eq.14 update recomputed independently in
float64, the paper's geometric invariants (orthogonality eq.11, N_out norm eq.12,
modulator range eq.13, norm-gain eq.15), depth-scaled alpha init (eq.23/24),
output shape/dtype, gradient flow to every NAG parameter, the full canonical-best
stack (XSA + GOAT sink + GatedNorm + AttnRes + NAG) forward/backward on CPU, and
config validation. All CPU, no training metrics involved.

Paper: Figliolia & Millidge, "Scaling Adaptive Depth with Norm-Agnostic Residual
Networks", Zyphra 2026, arXiv:2606.16112.
"""

import math

import pytest
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
Block = _BLA.Block
NAGResidual = _BLA.NAGResidual


def tiny_config(use_nag=True):
    cfg = GPTConfig()
    cfg.block_size = 8
    cfg.vocab_size = 64
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_kv_head = 2
    cfg.n_embd = 16
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.activation = "gelu"          # dodge the Primer-MLP cuda device pin (as test_pope_f78)
    cfg.norm_layer = "rmsnorm"
    cfg.attention = "regular"
    cfg.pos_encoding = "rotary"
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.rmsnorm_before_qk = False
    cfg.use_engram = False
    cfg.use_nag_residual = use_nag
    cfg.nag_num_directions = 8
    cfg.nag_beta = 1.0
    cfg.nag_init_p = 0.5
    return cfg


def canonical_config():
    """Canonical best stack: XSA + GOAT sink + GatedNorm rank16 + AttnRes + NAG."""
    cfg = tiny_config(use_nag=True)
    cfg.attention = "xsa"
    cfg.use_goat_sink_prior = True
    cfg.use_gated_norm = True
    cfg.gated_norm_rank = 16
    cfg.use_attn_res = True
    return cfg


# --- 1. Gate-off default is byte-identical structure (no NAG modules/params) ---

def test_default_config_has_nag_off():
    assert GPTConfig().use_nag_residual is False


def test_gate_off_block_has_no_nag():
    block = Block(tiny_config(use_nag=False), depth=0)
    assert block.nag_attn is None
    assert block.nag_mlp is None
    assert not any(isinstance(m, NAGResidual) for m in block.modules())
    # additive residual path is preserved: x + branch_out
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    y = block(x, token_ids=torch.randint(0, 64, (2, 4)))
    assert y.shape == x.shape


def test_gate_off_full_gpt_has_no_nag_params():
    model = GPT(tiny_config(use_nag=False))
    assert not any(isinstance(m, NAGResidual) for m in model.modules())
    assert not any("nag" in name for name, _ in model.named_parameters())


# --- 2. NAG construction ------------------------------------------------------

def test_nag_construction():
    block = Block(tiny_config(use_nag=True), depth=0)
    assert isinstance(block.nag_attn, NAGResidual)
    assert isinstance(block.nag_mlp, NAGResidual)
    # trainable params: alpha (scalar), w (C x d), b (C), p_logits (C)
    d, C = 16, 8
    assert block.nag_attn.alpha.shape == torch.Size([])
    assert block.nag_attn.w.shape == (C, d)
    assert block.nag_attn.b.shape == (C,)
    assert block.nag_attn.p_logits.shape == (C,)
    # modulator init: b, p_logits zeroed; w ~ N(0, 0.02)
    assert torch.equal(block.nag_attn.b, torch.zeros(C))
    assert torch.equal(block.nag_attn.p_logits, torch.zeros(C))
    assert block.nag_attn.w.abs().mean() > 0


# --- 3. Numerical check: eq.14 recomputed independently in float64 ------------

def test_nag_matches_eq14():
    torch.manual_seed(1)
    cfg = tiny_config()
    nag = NAGResidual(cfg, layer_index=3)
    with torch.no_grad():
        nag.w.normal_(0, 0.5)
        nag.b.uniform_(-1, 1)
        nag.p_logits.uniform_(-1, 1)
        nag.alpha.fill_(0.7)
    B, T, d = 2, 5, cfg.n_embd
    x = torch.randn(B, T, d)
    bo = torch.randn(B, T, d)
    out = nag(x, bo)

    # Independent float64 reference straight from eq.3-14.
    sqrt_d = math.sqrt(d)
    x64, bo64 = x.double(), bo.double()
    xn = x64.norm(dim=-1, keepdim=True)
    r_bar = sqrt_d * x64 / xn
    rho = xn / sqrt_d
    f = bo64 - bo64.mean(dim=-1, keepdim=True)
    proj = (f * r_bar).sum(dim=-1, keepdim=True) / d
    f_perp = f - proj * r_bar
    u = sqrt_d * f_perp / f_perp.norm(dim=-1, keepdim=True)
    gates = torch.sigmoid(r_bar @ nag.w.double().t() + nag.b.double())
    p = torch.softmax(nag.p_logits.double(), dim=0)
    m = (gates * p).sum(dim=-1, keepdim=True) ** nag.beta
    ref = x64 + rho * nag.alpha.double() * m * u

    assert torch.allclose(out.double(), ref, atol=1e-6, rtol=1e-6)


def test_nag_paper_invariants():
    """Orthogonality (eq.11), N_out norm (eq.12), modulator range (eq.13),
    and norm-gain sqrt(1 + alpha^2 m^2) (eq.15)."""
    torch.manual_seed(2)
    cfg = tiny_config()
    nag = NAGResidual(cfg, layer_index=2)
    with torch.no_grad():
        nag.w.normal_(0, 0.5)
        nag.b.uniform_(-1, 1)
        nag.p_logits.uniform_(-1, 1)
        nag.alpha.fill_(0.9)
    d = cfg.n_embd
    x = torch.randn(3, 4, d).double()
    bo = torch.randn(3, 4, d).double()
    nag.double()
    out = nag(x, bo)
    update = out - x

    # (eq.11) the update is orthogonal to the residual direction.
    cos = (update * x).sum(-1) / (update.norm(dim=-1) * x.norm(dim=-1))
    assert cos.abs().max() < 1e-9

    # recompute the modulator m to obtain the expected norm gain.
    sqrt_d = math.sqrt(d)
    r_bar = sqrt_d * x / x.norm(dim=-1, keepdim=True)
    gates = torch.sigmoid(r_bar @ nag.w.t() + nag.b)
    p = torch.softmax(nag.p_logits, dim=0)
    m = (gates * p).sum(dim=-1) ** nag.beta            # (B, T)
    assert (m >= 0).all() and (m <= 1).all()           # (eq.13) modulator ∈ [0,1]

    # (eq.15) norm gain = sqrt(1 + alpha^2 m^2), independent of ||x||.
    gain = out.norm(dim=-1) / x.norm(dim=-1)
    expected_gain = torch.sqrt(1 + nag.alpha ** 2 * m ** 2)
    assert torch.allclose(gain, expected_gain, atol=1e-9)
    # relative update magnitude = alpha * m (norm-agnostic: no dependence on ||x||).
    assert torch.allclose(update.norm(dim=-1) / x.norm(dim=-1), nag.alpha.abs() * m, atol=1e-9)


# --- 4. Depth-scaled alpha init (eq.23/24) + near-identity --------------------

def test_alpha_depth_scaled_init():
    cfg = tiny_config()          # nag_init_p = 0.5 => alpha_l = 1/sqrt(l)
    for depth in range(cfg.n_layer):
        block = Block(cfg, depth)
        # attn = NAG-layer 2*depth+1, mlp = NAG-layer 2*depth+2
        assert math.isclose(block.nag_attn.alpha.item(), 1.0 / math.sqrt(2 * depth + 1), rel_tol=1e-6)
        assert math.isclose(block.nag_mlp.alpha.item(), 1.0 / math.sqrt(2 * depth + 2), rel_tol=1e-6)


def test_alpha_init_p_exponent():
    cfg = tiny_config()
    cfg.nag_init_p = 0.25
    block = Block(cfg, depth=4)   # NAG-layer 9 for attn
    assert math.isclose(block.nag_attn.alpha.item(), 1.0 / (9 ** 0.25), rel_tol=1e-6)


def test_near_identity_at_init():
    """At init the per-layer relative update alpha*m is a small controlled rotation."""
    torch.manual_seed(3)
    cfg = tiny_config()
    nag = NAGResidual(cfg, layer_index=5)   # alpha = 1/sqrt(5) ~ 0.447
    x = torch.randn(2, 4, cfg.n_embd)
    bo = torch.randn(2, 4, cfg.n_embd)
    out = nag(x, bo)
    rel = (out - x).norm(dim=-1) / x.norm(dim=-1)
    # alpha*m with m~0.5 at init => rel ~ 0.22 < 0.5, bounded and non-zero.
    assert (rel > 0).all()
    assert rel.max() < 0.5


# --- 5. Output shape / dtype --------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_shape_dtype(dtype):
    cfg = tiny_config()
    nag = NAGResidual(cfg, layer_index=1).to(dtype)
    x = torch.randn(2, 5, cfg.n_embd, dtype=dtype)
    bo = torch.randn(2, 5, cfg.n_embd, dtype=dtype)
    out = nag(x, bo)
    assert out.shape == x.shape
    assert out.dtype == dtype


# --- 6. Gradient flow to every NAG parameter ---------------------------------

def test_gradients_reach_all_nag_params():
    torch.manual_seed(4)
    cfg = tiny_config()
    nag = NAGResidual(cfg, layer_index=2)
    x = torch.randn(2, 4, cfg.n_embd, requires_grad=True)
    bo = torch.randn(2, 4, cfg.n_embd, requires_grad=True)
    out = nag(x, bo)
    out.square().mean().backward()
    for name in ("alpha", "w", "b", "p_logits"):
        g = getattr(nag, name).grad
        assert g is not None, f"{name} received no gradient"
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0, f"{name} gradient is all-zero"
    assert x.grad is not None and x.grad.abs().sum() > 0


# --- 7. Full canonical-best stack (XSA + GOAT + GatedNorm + AttnRes + NAG) ----

def test_full_gpt_canonical_stack_forward_backward():
    cfg = canonical_config()
    torch.manual_seed(5)
    model = GPT(cfg)
    # every base Block carries NAG sub-modules
    n_nag = sum(isinstance(m, NAGResidual) for m in model.modules())
    assert n_nag == 2 * cfg.n_layer
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    # gradients reach NAG params inside the full model
    nag_params = [p for n, p in model.named_parameters() if "nag" in n]
    assert nag_params, "no NAG parameters in the full model"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in nag_params)


def test_nag_composes_with_attn_res_delta_capture():
    """use_attn_res reads block(x) - x; NAG changes that delta but not the contract."""
    cfg = canonical_config()
    torch.manual_seed(6)
    model = GPT(cfg)
    assert getattr(model.config, "use_attn_res") is True
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    targets = torch.randint(0, cfg.vocab_size, (1, 6))
    logits, loss = model(idx, targets)
    assert logits.shape == (1, 6, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is not None and torch.isfinite(loss)


# --- 8. Config validation -----------------------------------------------------

def test_validation_num_directions():
    cfg = tiny_config()
    cfg.nag_num_directions = 0
    with pytest.raises(ValueError):
        NAGResidual(cfg, layer_index=1)


def test_validation_beta_positive():
    cfg = tiny_config()
    cfg.nag_beta = 0.0
    with pytest.raises(ValueError):
        NAGResidual(cfg, layer_index=1)


def test_validation_layer_index():
    cfg = tiny_config()
    with pytest.raises(ValueError):
        NAGResidual(cfg, layer_index=0)
