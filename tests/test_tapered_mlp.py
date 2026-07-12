"""Tests for Tapered Language Models (arXiv:2606.23670) — experiment F72.

Cosine-tapered per-layer Primer MLP width, config-gated via `use_tapered_mlp`
(default off), budget-preserving (aggregate width == n_layer * base_d_ff).
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

# Load bla_gpt.py by path: the sibling `bla_gpt/` package dir would otherwise
# shadow the module under pytest (same trick as test_exclusive_self_attention).
_SPEC = spec_from_file_location(
    "blagpt_model", Path(__file__).resolve().parents[1] / "bla_gpt" / "bla_gpt.py"
)
_BLA = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BLA)

GPT = _BLA.GPT
GPTConfig = _BLA.GPTConfig
get_mlp = _BLA.get_mlp

from mlps import Primer_MLP, tapered_mlp_dims  # noqa: E402  (path set by conftest)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA, reason="Primer_MLP initialises weights on CUDA (semi_orthogonal_init)"
)


def _primer_cfg(**over):
    cfg = GPTConfig()
    cfg.block_size = 32
    cfg.vocab_size = 256
    cfg.n_layer = 4
    cfg.n_head = 4
    cfg.n_kv_head = 4
    cfg.n_embd = 128
    cfg.activation = "primer"
    cfg.attention = "regular"
    cfg.use_engram = False
    cfg.use_attn_res = False
    cfg.n_predict = 1
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


# ---- (2) pure width schedule: on -> monotone, aligned, exact aggregate ----

def test_tapered_dims_full_config():
    # canonical full baseline: n_layer=10, n_embd=768, mlp_expand=10 -> base 7680
    base, L = 10 * 768, 10
    dims = tapered_mlp_dims(base, L)
    assert dims == [11520, 11264, 10624, 9600, 8320, 7040, 5760, 4736, 4096, 3840]
    assert dims[0] == int(round(1.5 * base / 64)) * 64   # d_start endpoint pinned
    assert dims[-1] == int(round(0.5 * base / 64)) * 64  # d_end endpoint pinned
    assert sum(dims) == L * base                         # exact budget (paper Eq. 7)
    assert all(dims[i] >= dims[i + 1] for i in range(L - 1))  # monotone non-increasing
    assert all(d % 64 == 0 for d in dims)                # repo width alignment


@pytest.mark.parametrize("L", list(range(2, 25)))
def test_tapered_dims_invariants(L):
    base = 7680
    dims = tapered_mlp_dims(base, L)
    assert len(dims) == L
    assert all(d % 64 == 0 for d in dims)                     # aligned to multiple
    assert all(dims[i] >= dims[i + 1] for i in range(L - 1))  # monotone non-increasing
    assert sum(dims) == L * base                              # exact aggregate
    assert dims[0] >= dims[-1]                                # early >= late (taper)


def test_tapered_dims_reject_unaligned_budget():
    with pytest.raises(ValueError):
        tapered_mlp_dims(100, 3)  # 300 is not a multiple of 64


# ---- (1) feature OFF reproduces uniform widths / default construction ----

def test_off_is_the_default():
    assert GPTConfig().use_tapered_mlp is False


@requires_cuda
def test_off_reproduces_uniform_widths():
    cfg = _primer_cfg(use_tapered_mlp=False)
    base = cfg.mlp_expand * cfg.n_embd
    widths = [get_mlp(cfg, l).c_fc.shape[0] for l in range(cfg.n_layer)]
    assert widths == [base] * cfg.n_layer                    # uniform, unchanged
    assert Primer_MLP(cfg).c_fc.shape[0] == base             # identical default build


# ---- (2b) feature ON produces tapered per-layer widths in the built model ----

@requires_cuda
def test_on_model_widths_monotone_and_budget():
    cfg = _primer_cfg(use_tapered_mlp=True)
    base = cfg.mlp_expand * cfg.n_embd
    widths = [get_mlp(cfg, l).c_fc.shape[0] for l in range(cfg.n_layer)]
    assert widths == tapered_mlp_dims(base, cfg.n_layer)
    assert all(widths[i] >= widths[i + 1] for i in range(cfg.n_layer - 1))
    assert sum(widths) == cfg.n_layer * base                 # budget preserved
    assert widths != [base] * cfg.n_layer                    # actually tapered


@requires_cuda
def test_on_guard_rejects_non_primer():
    cfg = _primer_cfg(use_tapered_mlp=True, activation="swiglu")
    with pytest.raises(ValueError):
        get_mlp(cfg, 0)


# ---- (3) forward / backward finite on a tiny CUDA synthetic input ----

@requires_cuda
def test_forward_backward_finite():
    torch.manual_seed(0)
    cfg = _primer_cfg(use_tapered_mlp=True)
    model = GPT(cfg).cuda()
    B, T = 2, cfg.block_size
    x = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    logits, loss = model(x, targets=y)
    assert torch.isfinite(loss).all()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)
