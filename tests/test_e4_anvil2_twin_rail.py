"""Tests for E4a: optional twin-rail slow momentum in the Aurora optimizer.

E4a adds a config-gated, default-off second momentum rail to Aurora
(``aurora_twin_rail``), inspired by ANVIL2's twin-rail velocity. When the
flag is off, behavior must stay byte-identical to the vendored Aurora
implementation (see tests/test_aurora_optimizer.py).
"""

import torch
import torch.nn as nn

from optimizers import get_optimizer  # noqa: E402
from optimizers.aurora import Aurora  # noqa: E402


AURORA_ARGS = {
    "weight_decay": 0.025,
    "mu": 0.95,
    "nesterov": True,
    "pp_iterations": 2,
    "pp_beta": 0.5,
}


class TinyModel(nn.Module):
    """Same tiny GPT-shaped model as test_aurora_optimizer.py."""

    def __init__(self, vocab=16, dim=8, hidden=12):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.norm = nn.Parameter(torch.ones(dim))
        self.fc = nn.Linear(dim, hidden, bias=True)
        self.lm_head = nn.Linear(hidden, vocab, bias=True)

    def forward(self, idx):
        h = self.embed_tokens(idx) * self.norm
        h = self.fc(h)
        return self.lm_head(h)


def _step_n(opt, params, grads_per_step):
    for grads in grads_per_step:
        for p, g in zip(params, grads):
            p.grad = g.clone()
        opt.step()


# ---------------------------------------------------------------------------
# a. Default-off is byte-identical to the baseline Aurora implementation.
# ---------------------------------------------------------------------------
def test_default_off_identical_to_baseline():
    torch.manual_seed(0)
    tall_a = nn.Parameter(torch.randn(8, 4))
    tall_b = nn.Parameter(tall_a.detach().clone())
    grads = [torch.randn(8, 4) for _ in range(5)]

    opt_baseline = Aurora(lr=0.05, aurora_params=[tall_a], adamw_params=[], **AURORA_ARGS)
    opt_flagged_off = Aurora(
        lr=0.05, aurora_params=[tall_b], adamw_params=[], aurora_twin_rail=False, **AURORA_ARGS
    )
    _step_n(opt_baseline, [tall_a], [[g] for g in grads])
    _step_n(opt_flagged_off, [tall_b], [[g] for g in grads])

    assert torch.equal(tall_a, tall_b)
    # No twin-rail state leaked into the param state when off.
    st = opt_flagged_off.state[tall_b]
    assert "momentum_buffer_slow" not in st
    assert "step" not in st


# ---------------------------------------------------------------------------
# b. Twin-rail flag allocates extra state only for aurora-routed params.
# ---------------------------------------------------------------------------
def test_twin_rail_flag_constructs_and_state():
    torch.manual_seed(1)
    model = TinyModel()
    args = dict(AURORA_ARGS)
    args["aurora_twin_rail"] = True
    opt = get_optimizer("Aurora", args, lr=0.05, model=model)
    assert isinstance(opt, Aurora)
    assert opt.param_groups[0]["aurora_twin_rail"] is True

    idx = torch.randint(0, 16, (2, 5))
    target = torch.randint(0, 16, (2, 5))
    logits = model(idx)
    loss = nn.functional.cross_entropy(logits.reshape(-1, 16), target.reshape(-1))
    loss.backward()
    opt.step()

    name_by_param = {p: name for name, p in model.named_parameters()}
    for p in opt.param_groups[0]["params"]:
        st = opt.state[p]
        if st.get("use_aurora", False):
            assert "momentum_buffer_slow" in st, name_by_param[p]
            assert st["step"] == 1, name_by_param[p]
        else:
            assert "momentum_buffer_slow" not in st, name_by_param[p]


# ---------------------------------------------------------------------------
# c. Before rail_engage_step, twin-rail matches the single-rail (flag off) run.
# ---------------------------------------------------------------------------
def test_twin_rail_before_engage_step_matches_fast_only():
    torch.manual_seed(2)
    engage = 3
    tall_single = nn.Parameter(torch.randn(8, 4))
    tall_twin = nn.Parameter(tall_single.detach().clone())
    # step counter reaches "engage" only on the engage-th step, and blending
    # starts once step >= rail_engage_step, so run engage-1 steps to stay
    # strictly in the fast-only regime.
    grads = [torch.randn(8, 4) for _ in range(engage - 1)]

    opt_single = Aurora(lr=0.05, aurora_params=[tall_single], adamw_params=[], **AURORA_ARGS)
    opt_twin = Aurora(
        lr=0.05,
        aurora_params=[tall_twin],
        adamw_params=[],
        aurora_twin_rail=True,
        rail_engage_step=engage,
        **AURORA_ARGS,
    )
    _step_n(opt_single, [tall_single], [[g] for g in grads])
    _step_n(opt_twin, [tall_twin], [[g] for g in grads])

    assert torch.allclose(tall_single, tall_twin, atol=1e-6)


# ---------------------------------------------------------------------------
# d. After rail_engage_step, twin-rail diverges from the single-rail run.
# ---------------------------------------------------------------------------
def test_twin_rail_after_engage_step_diverges_from_single_rail():
    torch.manual_seed(3)
    engage = 3
    n_steps = engage + 5
    tall_single = nn.Parameter(torch.randn(8, 4))
    tall_twin = nn.Parameter(tall_single.detach().clone())
    grads = [torch.randn(8, 4) for _ in range(n_steps)]

    opt_single = Aurora(lr=0.05, aurora_params=[tall_single], adamw_params=[], **AURORA_ARGS)
    opt_twin = Aurora(
        lr=0.05,
        aurora_params=[tall_twin],
        adamw_params=[],
        aurora_twin_rail=True,
        rail_engage_step=engage,
        **AURORA_ARGS,
    )
    _step_n(opt_single, [tall_single], [[g] for g in grads])
    _step_n(opt_twin, [tall_twin], [[g] for g in grads])

    assert not torch.allclose(tall_single, tall_twin, atol=1e-6)
    assert torch.isfinite(tall_single).all() and torch.isfinite(tall_twin).all()


# ---------------------------------------------------------------------------
# e. Finite updates with twin-rail on, rectangular and vector params.
# ---------------------------------------------------------------------------
def test_finite_updates_with_twin_rail_on_rectangular_and_vector():
    torch.manual_seed(4)
    tall = nn.Parameter(torch.randn(8, 4))
    wide = nn.Parameter(torch.randn(4, 8))
    square = nn.Parameter(torch.randn(5, 5))
    vec = nn.Parameter(torch.zeros(8))
    opt = Aurora(
        lr=0.05,
        aurora_params=[tall, wide, square],
        adamw_params=[vec],
        aurora_twin_rail=True,
        rail_engage_step=2,
        **AURORA_ARGS,
    )
    before = {id(p): p.detach().clone() for p in (tall, wide, square, vec)}
    for _ in range(5):
        for p in (tall, wide, square, vec):
            p.grad = torch.randn_like(p)
        opt.step()
    for p in (tall, wide, square, vec):
        assert torch.isfinite(p).all(), "update produced non-finite values"
        moved = (p.detach() - before[id(p)]).abs().max().item()
        assert moved > 0.0, "parameter did not move"


# ---------------------------------------------------------------------------
# f. state_dict roundtrip preserves momentum_buffer_slow byte-exactly.
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip_twin_rail():
    torch.manual_seed(5)
    tall = nn.Parameter(torch.randn(8, 4))
    vec = nn.Parameter(torch.zeros(8))
    args = dict(AURORA_ARGS)
    opt = Aurora(
        lr=0.05,
        aurora_params=[tall],
        adamw_params=[vec],
        aurora_twin_rail=True,
        rail_engage_step=2,
        **args,
    )
    for _ in range(3):
        tall.grad = torch.randn_like(tall)
        vec.grad = torch.randn_like(vec)
        opt.step()
    sd = opt.state_dict()

    tall2 = nn.Parameter(tall.detach().clone())
    vec2 = nn.Parameter(vec.detach().clone())
    opt2 = Aurora(
        lr=0.05,
        aurora_params=[tall2],
        adamw_params=[vec2],
        aurora_twin_rail=True,
        rail_engage_step=2,
        **args,
    )
    opt2.load_state_dict(sd)

    st_new = opt2.state_dict()["state"]
    st_old = sd["state"]
    buf_old = next(v["momentum_buffer_slow"] for v in st_old.values() if "momentum_buffer_slow" in v)
    buf_new = next(v["momentum_buffer_slow"] for v in st_new.values() if "momentum_buffer_slow" in v)
    assert torch.equal(buf_old, buf_new)

    tall2.grad = torch.randn_like(tall2)
    vec2.grad = torch.randn_like(vec2)
    opt2.step()
    assert torch.isfinite(tall2).all() and torch.isfinite(vec2).all()
