"""Tests for F104: SAMuon-lite head-anchored allocation in Aurora.

SAMuon-lite (arXiv 2608.25990, Algorithm 1, Variant #2) keeps the leading
singular direction u1 v1^T of the momentum update at the base (Muon / Aurora)
scale and scales every other whitened direction by a cosine-warmed tail boost
gamma_t. It is config-gated through ``optimizer_args["samuon_lite"]``; the
default (off) must be bit-identical to the pre-F104 Aurora.
"""

import copy
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# `bla_gpt/` is placed on sys.path by tests/conftest.py.
from optimizers import get_optimizer  # noqa: E402
from optimizers.aurora import (  # noqa: E402
    Aurora,
    aurora,
    polar,
    samuon_gamma_t,
    samuon_lite_reallocate,
    top_singular_pair,
)

ROOT = Path(__file__).resolve().parents[1]
E7B_ARGS = json.load(open(ROOT / "ar" / "best_config.json"))["optimizer_args"]
F104_FLAGS = {
    "samuon_lite": True,
    "samuon_gamma": 10.0,
    "samuon_warmup_steps": 1530,
    "samuon_power_iters": 10,
}


def _gapped(m, n, top=5.0):
    """Random m x n matrix with sigma_1 = top and the rest in [0.1, 2.5]."""
    U, _ = torch.linalg.qr(torch.randn(m, m))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    r = min(m, n)
    S = torch.linspace(2.5, 0.1, r)
    S[0] = top
    return U[:, :r] @ torch.diag(S) @ V[:, :r].mT


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


class TinyModel(nn.Module):
    """GPT-shaped toy model with tied head, a wide, a tall and a square matrix."""

    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(vocab, dim)))
        self.norm = nn.Parameter(torch.ones(dim))
        self.up = nn.Linear(dim, 3 * dim, bias=True)  # tall 48x16
        self.down = nn.Linear(3 * dim, dim, bias=False)  # wide 16x48
        self.sq = nn.Linear(dim, dim, bias=False)  # square
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

    def forward(self, idx):
        h = self.transformer["wte"](idx) * self.norm
        h = h + self.down(torch.relu(self.up(h)))
        h = self.sq(h)
        return self.lm_head(h)


def _train(model, args, steps=6, device="cpu", seed=0):
    model = model.to(device)
    opt = get_optimizer("Aurora", dict(args), lr=0.03, model=model)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, 32, (4, 8), generator=g).to(device)
        tgt = torch.randint(0, 32, (4, 8), generator=g).to(device)
        loss = nn.functional.cross_entropy(model(idx).flatten(0, 1), tgt.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, opt


@pytest.mark.parametrize("device", _devices())
def test_flag_off_bit_identical_to_current_aurora(device):
    torch.manual_seed(1)
    base = TinyModel()
    explicit_off = copy.deepcopy(base)
    m_a, _ = _train(base, E7B_ARGS, device=device)
    m_b, _ = _train(explicit_off, dict(E7B_ARGS, samuon_lite=False), device=device)
    for (na, pa), (nb, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), na


def test_flag_off_aurora_function_identical():
    torch.manual_seed(2)
    W = torch.randn(24, 40)
    G = torch.randn(24, 40)
    M = torch.randn(24, 40)
    W1, G1, M1 = W.clone(), G.clone(), M.clone()
    W2, G2, M2 = W.clone(), G.clone(), M.clone()
    aurora(W1, G1, M1, eta=0.03)
    aurora(W2, G2, M2, eta=0.03, samuon_lite=False, samuon_gamma=10.0, step=900)
    assert torch.equal(W1, W2) and torch.equal(M1, M2)


def test_power_iteration_matches_svd():
    torch.manual_seed(3)
    U, _ = torch.linalg.qr(torch.randn(40, 40))
    V, _ = torch.linalg.qr(torch.randn(24, 24))
    S = torch.linspace(2.5, 0.1, 24)
    S[0] = 5.0  # clear gap sigma_1 / sigma_2 = 2
    A = U[:, :24] @ torch.diag(S) @ V.mT
    u, v = top_singular_pair(A, iters=10)
    assert abs(abs(float(u @ U[:, 0])) - 1.0) < 1e-5
    assert abs(abs(float(v @ V[:, 0])) - 1.0) < 1e-5
    assert float(u @ A @ v) > 0  # sign-consistent pair: u^T A v = +sigma_1


def test_reallocation_keeps_head_scales_bulk_vs_explicit_svd():
    torch.manual_seed(4)
    A = _gapped(32, 20).double()
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    O = U @ Vh  # exact whitening
    gamma = 7.0
    u, v = top_singular_pair(A.float(), iters=50)
    out = samuon_lite_reallocate(O.float(), u, v, gamma).double()
    # Expected: paper Eq. 8 with k=1 -> head coefficient 1, all others gamma.
    expected = gamma * O - (gamma - 1.0) * torch.outer(U[:, 0], Vh[0])
    assert torch.allclose(out, expected, atol=1e-4)
    # Spectral check in the momentum basis.
    coeffs = torch.diagonal(U.mT @ out @ Vh.mT)
    assert abs(float(coeffs[0]) - 1.0) < 1e-4
    assert torch.allclose(coeffs[1:], torch.full_like(coeffs[1:], gamma), atol=1e-4)


def test_gamma_one_is_identity_and_warmup_schedule():
    torch.manual_seed(5)
    O = torch.randn(16, 12)
    u, v = top_singular_pair(torch.randn(16, 12))
    assert torch.allclose(samuon_lite_reallocate(O, u, v, 1.0), O)
    assert samuon_gamma_t(10.0, 0, 1530) == 1.0
    assert samuon_gamma_t(10.0, 1530, 1530) == 10.0
    assert samuon_gamma_t(10.0, 5000, 1530) == 10.0
    assert math.isclose(samuon_gamma_t(10.0, 765, 1530), 5.5, rel_tol=1e-9)


def test_aurora_flag_on_scales_bulk_after_warmup():
    """In aurora(), the post-polar update keeps its head component and scales the rest."""
    torch.manual_seed(6)
    W = torch.randn(30, 18)
    G = _gapped(30, 18)
    mom = torch.zeros(30, 18)
    W_off, W_on = W.clone(), W.clone()
    aurora(W_off, G.clone(), mom.clone(), eta=1.0, weight_decay=0.0)
    aurora(
        W_on, G.clone(), mom.clone(), eta=1.0, weight_decay=0.0,
        samuon_lite=True, samuon_gamma=4.0, samuon_warmup_steps=10, step=10,
    )
    d_off = (W - W_off).float()
    d_on = (W - W_on).float()
    # Nesterov update with zero momentum: (1-mu)*mu*G + (1-mu)*G is a multiple of G.
    u, v = top_singular_pair(G, iters=30)
    head_off = u @ d_off @ v
    head_on = u @ d_on @ v
    assert torch.isclose(head_on, head_off, rtol=2e-2)
    bulk_off = d_off - head_off * torch.outer(u, v)
    bulk_on = d_on - head_on * torch.outer(u, v)
    ratio = bulk_on.norm() / bulk_off.norm()
    assert abs(float(ratio) - 4.0) < 0.1


@pytest.mark.parametrize("device", _devices())
def test_flag_on_finite_and_differs_with_e7b_groups(device):
    torch.manual_seed(7)
    base = TinyModel()
    on = copy.deepcopy(base)
    args_on = dict(E7B_ARGS, **dict(F104_FLAGS, samuon_warmup_steps=2))
    m_off, _ = _train(base, E7B_ARGS, device=device)
    m_on, opt_on = _train(on, args_on, device=device)
    assert isinstance(opt_on, Aurora)
    assert opt_on.param_groups[0]["samuon_lite"] is True
    for p in m_on.parameters():
        assert torch.isfinite(p).all()
    assert not torch.equal(m_off.sq.weight, m_on.sq.weight)
    # Aurora-path params count steps; AdamW-path params keep their own step.
    for name, p in m_on.named_parameters():
        st = opt_on.state[p]
        if st.get("use_aurora"):
            assert st["step"] == 6, name


def test_flag_on_state_dict_roundtrip():
    torch.manual_seed(8)
    m, opt = _train(TinyModel(), dict(E7B_ARGS, **F104_FLAGS), steps=3)
    sd = opt.state_dict()
    m2 = copy.deepcopy(m)
    opt2 = get_optimizer("Aurora", dict(E7B_ARGS, **F104_FLAGS), lr=0.03, model=m2)
    opt2.load_state_dict(sd)
    assert opt2.param_groups[0]["samuon_gamma"] == 10.0


def test_polar_unchanged():
    torch.manual_seed(9)
    X = polar(torch.randn(20, 12))
    s = torch.linalg.svdvals(X.float())
    assert torch.all(s > 0.6) and torch.all(s < 1.4)
