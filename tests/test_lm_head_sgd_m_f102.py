"""Tests for the F102 LM-head SGD-M routing (arXiv 2610.38277).

Experiment F102 routes the tied embedding / LM head tensor to plain SGD with
momentum (paper Figure 4: momentum 0.99) while the trunk keeps Aurora. The
routing is config-gated through ``optimizer_args["lm_head_sgd_m"]`` in
``get_optimizer("Aurora", ...)``. Default off must be bit-identical.
"""

import io
import sys
from contextlib import redirect_stdout

import torch
import torch.nn as nn

# `bla_gpt/` is placed on sys.path by tests/conftest.py.
from optimizers import get_optimizer  # noqa: E402
from optimizers.aurora import Aurora  # noqa: E402

AURORA_ARGS = {
    "weight_decay": 0.025,
    "mu": 0.95,
    "nesterov": True,
    "pp_iterations": 2,
    "pp_beta": 0.5,
}


class TiedTinyModel(nn.Module):
    """GPT-style tiny model with weight tying, like bla_gpt.py.

    ``transformer.wte.weight`` is assigned ``lm_head.weight`` (same object),
    so named_parameters() yields the tied tensor once under the embedding's
    name. That name contains neither "embed_tokens" nor "lm_head", so the
    name filter routes the tied tensor into the Aurora path by default.
    """

    def __init__(self, vocab=16, dim=12):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(vocab, dim)))
        self.fc = nn.Linear(dim, dim, bias=True)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.transformer["wte"].weight = self.lm_head.weight

    def forward(self, idx):
        h = self.transformer["wte"](idx)
        return self.lm_head(self.fc(h))


def _make_pair(seed):
    """Two identically initialized copies of TiedTinyModel."""
    torch.manual_seed(seed)
    a = TiedTinyModel()
    torch.manual_seed(seed)
    b = TiedTinyModel()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)
    return a, b


# ---------------------------------------------------------------------------
# 1. Flag off: bit-identical to the pre-F102 Aurora construction.
# ---------------------------------------------------------------------------
def test_flag_off_bit_identical_to_direct_aurora():
    torch.manual_seed(0)
    m_reg, m_dir = _make_pair(1)
    opt_reg = get_optimizer("Aurora", dict(AURORA_ARGS), lr=0.05, model=m_reg)
    aurora_params = [
        p
        for name, p in m_dir.named_parameters()
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
    ]
    adamw_params = [
        p
        for name, p in m_dir.named_parameters()
        if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name
    ]
    opt_dir = Aurora(lr=0.05, aurora_params=aurora_params, adamw_params=adamw_params, **AURORA_ARGS)
    assert len(opt_reg.param_groups) == 1 and len(opt_dir.param_groups) == 1
    for _ in range(4):
        grads = [torch.randn_like(p) for p in m_reg.parameters()]
        gs = iter(grads)
        for p in m_reg.parameters():
            p.grad = next(gs).clone()
        gs = iter(grads)
        for p in m_dir.parameters():
            p.grad = next(gs).clone()
        opt_reg.step()
        opt_dir.step()
    for (n, p_reg), (_, p_dir) in zip(m_reg.named_parameters(), m_dir.named_parameters()):
        assert torch.equal(p_reg, p_dir), f"{n} diverged with flag off"


# ---------------------------------------------------------------------------
# 2. Flag on: the tied tensor lands in a separate SGD-M group.
# ---------------------------------------------------------------------------
def test_flag_on_routes_tied_tensor_to_sgd_m_group():
    torch.manual_seed(2)
    model = TiedTinyModel()
    args = dict(AURORA_ARGS)
    args["lm_head_sgd_m"] = True
    args["lm_head_sgd_lr"] = 0.01
    opt = get_optimizer("Aurora", args, lr=0.05, model=model)
    assert isinstance(opt, Aurora)
    assert len(opt.param_groups) == 2
    head = model.lm_head.weight
    g0, g1 = opt.param_groups
    assert all(p is not head for p in g0["params"]), "tied tensor must leave the Aurora/AdamW group"
    assert g1.get("use_sgd_m") is True
    assert [p is head for p in g1["params"]] == [True]
    assert g1["lr"] == 0.01
    assert g1["momentum"] == 0.99
    # Aurora group still holds the trunk matrices and keeps its hyperparameters.
    assert g0["weight_decay"] == 0.025 and g0["mu"] == 0.95


# ---------------------------------------------------------------------------
# 3. SGD-M group numerics match torch.optim.SGD(momentum=0.99).
# ---------------------------------------------------------------------------
def test_sgd_m_matches_torch_sgd():
    torch.manual_seed(3)
    grads = [torch.randn(16, 12) for _ in range(5)]
    m_a, m_b = _make_pair(4)
    args = dict(AURORA_ARGS)
    args["lm_head_sgd_m"] = True
    args["lm_head_sgd_lr"] = 0.02
    opt = get_optimizer("Aurora", args, lr=0.05, model=m_a)
    sgd = torch.optim.SGD([m_b.lm_head.weight], lr=0.02, momentum=0.99)
    for g in grads:
        m_a.lm_head.weight.grad = g.clone()
        opt.step()
        m_b.lm_head.weight.grad = g.clone()
        sgd.step()
    assert torch.equal(m_a.lm_head.weight, m_b.lm_head.weight)


# ---------------------------------------------------------------------------
# 4. Flag on: finite loss, finite updates through a real forward/backward.
# ---------------------------------------------------------------------------
def test_flag_on_finite_forward_backward():
    torch.manual_seed(5)
    model = TiedTinyModel()
    args = dict(AURORA_ARGS)
    args["lm_head_sgd_m"] = True
    args["lm_head_sgd_lr"] = 0.01
    opt = get_optimizer("Aurora", args, lr=0.05, model=model)
    for _ in range(3):
        idx = torch.randint(0, 16, (2, 6))
        target = torch.randint(0, 16, (2, 6))
        logits = model(idx)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 16), target.reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        for p in model.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all()
        opt.step()
        model.zero_grad(set_to_none=True)
    for n, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{n} became non-finite"


# ---------------------------------------------------------------------------
# 5. Routing probe: always-on stdout line naming the tied tensor and rule.
# ---------------------------------------------------------------------------
def _probe_output(model, args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        get_optimizer("Aurora", args, lr=0.05, model=model)
    return buf.getvalue()


def test_probe_reports_aurora_when_flag_off():
    torch.manual_seed(6)
    model = TiedTinyModel()
    out = _probe_output(model, dict(AURORA_ARGS))
    assert "[lm_head_probe]" in out
    assert "param_name=transformer.wte.weight" in out
    assert "update_rule=aurora" in out


def test_probe_reports_sgd_m_when_flag_on():
    torch.manual_seed(7)
    model = TiedTinyModel()
    args = dict(AURORA_ARGS)
    args["lm_head_sgd_m"] = True
    args["lm_head_sgd_lr"] = 0.01
    out = _probe_output(model, args)
    assert "[lm_head_probe]" in out
    assert "param_name=transformer.wte.weight" in out
    assert "update_rule=sgd_m(lr=0.01, momentum=0.99)" in out


# ---------------------------------------------------------------------------
# 5b. Step-1 runtime probe: reads the update path from optimizer state.
# ---------------------------------------------------------------------------
def _one_step(model, args):
    opt = get_optimizer("Aurora", args, lr=0.05, model=model)
    idx = torch.randint(0, 16, (2, 6))
    target = torch.randint(0, 16, (2, 6))
    loss = nn.functional.cross_entropy(model(idx).reshape(-1, 16), target.reshape(-1))
    loss.backward()
    opt.step()
    return opt

def test_update_path_probe_flag_off_reports_aurora():
    from optimizers import describe_update_paths

    torch.manual_seed(9)
    model = TiedTinyModel()
    opt = _one_step(model, dict(AURORA_ARGS))
    lines = describe_update_paths(model, opt)
    assert len(lines) == 1, lines
    assert "param=transformer.wte.weight" in lines[0]
    assert "tied_lm_head=True" in lines[0]
    assert "path=aurora" in lines[0]

def test_update_path_probe_flag_on_reports_sgd_m():
    from optimizers import describe_update_paths

    torch.manual_seed(10)
    model = TiedTinyModel()
    args = dict(AURORA_ARGS)
    args["lm_head_sgd_m"] = True
    args["lm_head_sgd_lr"] = 0.01
    opt = _one_step(model, args)
    lines = describe_update_paths(model, opt)
    assert len(lines) == 1, lines
    assert "tied_lm_head=True" in lines[0]
    assert "path=sgd_m" in lines[0]

# ---------------------------------------------------------------------------
# 6. Flag on without an lm_head must fail loudly, not silently.
# ---------------------------------------------------------------------------
def test_flag_on_without_lm_head_raises():
    torch.manual_seed(8)
    model = nn.Linear(8, 8)
    args = dict(AURORA_ARGS)
    args["lm_head_sgd_m"] = True
    try:
        get_optimizer("Aurora", args, lr=0.05, model=model)
    except ValueError:
        return
    raise AssertionError("lm_head_sgd_m without model.lm_head must raise ValueError")