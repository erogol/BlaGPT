"""Tests for F105: NorMuon row normalization and cautious weight decay in Aurora.

NorMuon (arXiv 2510.05491, Algorithm 1) divides each row (output neuron) of
the orthogonalized update by the square root of an EMA of its mean squared
entry; the reference code keeps the update's Frobenius norm. Cautious weight
decay (arXiv 2510.12402, Algorithm 1) decays only entries where
``update * W >= 0``. Both are config-gated through ``optimizer_args`` and the
defaults (off) must be bit-identical to the pre-F105 Aurora.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# `bla_gpt/` is placed on sys.path by tests/conftest.py.
from optimizers import get_optimizer  # noqa: E402
from optimizers.aurora import (  # noqa: E402
    Aurora,
    aurora,
    cautious_decay_,
    normuon_row_normalize,
)

ROOT = Path(__file__).resolve().parents[1]
E7B_ARGS = json.load(open(ROOT / "ar" / "best_config.json"))["optimizer_args"]
F105C_FLAGS = {"cautious_wd": True}
F105N_FLAGS = {"cautious_wd": True, "normuon": True, "normuon_beta2": 0.95}


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
def test_flags_off_bit_identical_to_current_aurora(device):
    torch.manual_seed(1)
    base = TinyModel()
    explicit_off = copy.deepcopy(base)
    m_a, _ = _train(base, E7B_ARGS, device=device)
    m_b, opt_b = _train(
        explicit_off, dict(E7B_ARGS, cautious_wd=False, normuon=False), device=device
    )
    for (na, pa), (nb, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), na
    assert all("normuon_v" not in st for st in opt_b.state.values())


def test_flags_off_aurora_function_identical():
    torch.manual_seed(2)
    for shape in [(24, 40), (40, 24), (32, 32)]:
        W, G, M = torch.randn(shape), torch.randn(shape), torch.randn(shape)
        W1, G1, M1 = W.clone(), G.clone(), M.clone()
        W2, G2, M2 = W.clone(), G.clone(), M.clone()
        aurora(W1, G1, M1, eta=0.03, weight_decay=0.025)
        aurora(W2, G2, M2, eta=0.03, weight_decay=0.025, cautious_wd=False, normuon=False)
        assert torch.equal(W1, W2) and torch.equal(M1, M2)


def test_cautious_decay_hand_built_mask():
    W = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, 0.0]])
    U = torch.tensor([[0.5, 0.5, -1.0], [-1.0, 0.0, 2.0]])
    # update*W >= 0 -> [[T, F, F], [T, T(0), T(0)]]
    out = cautious_decay_(W.clone(), U, decay=0.1)
    expected = torch.tensor([[0.9, -2.0, 3.0], [-3.6, 4.5, 0.0]])
    assert torch.allclose(out, expected)


def test_cautious_wd_in_aurora_matches_manual():
    """aurora(cautious_wd=True) = plain update + decay only where update*W >= 0."""
    torch.manual_seed(3)
    W = torch.randn(20, 36)
    G = torch.randn(20, 36)
    M = torch.randn(20, 36)
    eta, lam = 0.03, 0.5
    W_nowd = W.clone()
    aurora(W_nowd, G.clone(), M.clone(), eta=eta, weight_decay=0.0)
    update = (W - W_nowd) / eta  # the step that gets subtracted
    W_cwd = W.clone()
    aurora(W_cwd, G.clone(), M.clone(), eta=eta, weight_decay=lam, cautious_wd=True)
    mask = (update * W >= 0).float()
    expected = W - eta * lam * mask * W - eta * update
    assert torch.allclose(W_cwd, expected, atol=1e-6)
    frac = float(mask.mean())
    assert 0.2 < frac < 0.8  # the mask actually selects a subset


def test_normuon_matches_reference_computation():
    torch.manual_seed(4)
    O = torch.randn(12, 20)
    v = torch.zeros(12, 1)
    beta2, eps = 0.95, 1e-10
    # Two steps so the EMA carries state.
    out1 = normuon_row_normalize(O, v, beta2=beta2, eps=eps)
    O2 = torch.randn(12, 20) * torch.linspace(0.2, 3.0, 12)[:, None]
    out2 = normuon_row_normalize(O2, v, beta2=beta2, eps=eps)
    # Direct reference (Algorithm 1 lines 7-9 + reference-code norm keep).
    v_ref = torch.zeros(12, 1)
    refs = []
    for X in (O, O2):
        v_ref = beta2 * v_ref + (1 - beta2) * (X * X).mean(dim=1, keepdim=True)
        Y = X / (v_ref.sqrt() + eps)
        refs.append(Y * X.norm() / Y.norm())
    assert torch.allclose(out1, refs[0], atol=1e-5)
    assert torch.allclose(out2, refs[1], atol=1e-5)
    assert torch.allclose(v, v_ref, atol=1e-7)
    # Norm kept; row norms more uniform than input.
    assert torch.isclose(out2.norm(), O2.norm(), rtol=1e-5)
    cv = lambda X: float(X.norm(dim=1).std() / X.norm(dim=1).mean())  # noqa: E731
    assert cv(out2) < cv(O2)


def test_normuon_in_aurora_inserted_after_polar():
    """normuon=True changes only the post-polar update, per row."""
    torch.manual_seed(5)
    W = torch.randn(30, 18)
    G = torch.randn(30, 18)
    M = torch.zeros(30, 18)
    eta = 1.0
    W_off = W.clone()
    aurora(W_off, G.clone(), M.clone(), eta=eta, weight_decay=0.0)
    d_off = (W - W_off) / eta
    aspect = max(1, 30 / 18) ** 0.5
    v = torch.zeros(30, 1)
    W_on = W.clone()
    aurora(W_on, G.clone(), M.clone(), eta=eta, weight_decay=0.0, normuon=True, normuon_v=v)
    d_on = (W - W_on) / eta
    v_ref = torch.zeros(30, 1)
    expected = normuon_row_normalize(d_off / aspect, v_ref) * aspect
    # polar() returns bfloat16; the flag-off path also multiplies the aspect
    # factor in bfloat16, so d_off carries bf16 rounding (rel. ~4e-3).
    rel = float((d_on - expected).norm() / expected.norm())
    assert rel < 1e-2, rel
    assert torch.allclose(v, v_ref, rtol=1e-2)
    # And it really changes the update (rows equalized).
    assert float((d_on - d_off).norm() / d_off.norm()) > 0.02


def test_normuon_requires_buffer():
    with pytest.raises(ValueError):
        aurora(torch.randn(8, 4), torch.randn(8, 4), torch.zeros(8, 4), eta=0.1, normuon=True)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("flags", [F105C_FLAGS, F105N_FLAGS], ids=["F105c", "F105n"])
def test_flags_on_finite_and_differ_with_e7b_groups(device, flags):
    torch.manual_seed(7)
    base = TinyModel()
    on = copy.deepcopy(base)
    m_off, _ = _train(base, E7B_ARGS, device=device)
    m_on, opt_on = _train(on, dict(E7B_ARGS, **flags), device=device)
    assert isinstance(opt_on, Aurora)
    assert opt_on.param_groups[0]["cautious_wd"] is True
    assert opt_on.param_groups[0]["normuon"] is flags.get("normuon", False)
    for p in m_on.parameters():
        assert torch.isfinite(p).all()
    assert not torch.equal(m_off.sq.weight, m_on.sq.weight)
    for name, p in m_on.named_parameters():
        st = opt_on.state[p]
        if not st.get("use_aurora"):
            assert "normuon_v" not in st, name
        elif flags.get("normuon"):
            assert st["normuon_v"].shape == (p.size(0), 1), name
            assert torch.isfinite(st["normuon_v"]).all()


def test_flag_on_state_dict_roundtrip():
    torch.manual_seed(8)
    m, opt = _train(TinyModel(), dict(E7B_ARGS, **F105N_FLAGS), steps=3)
    sd = opt.state_dict()
    m2 = copy.deepcopy(m)
    opt2 = get_optimizer("Aurora", dict(E7B_ARGS, **F105N_FLAGS), lr=0.03, model=m2)
    opt2.load_state_dict(sd)
    assert opt2.param_groups[0]["normuon"] is True
    assert any("normuon_v" in st for st in opt2.state.values())
