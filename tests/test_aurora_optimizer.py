"""Tests for the Aurora optimizer integration — experiment F73.

Aurora (arXiv:2606.27715, github.com/tilde-research/aurora-release) is a
leverage-aware spectral optimizer: 2D matrix params are updated by the vendored
`aurora()` rule; 1D params and the embedding / head matrices fall back to an
internal AdamW, mirroring this repo's Muon integration. The optimizer is
config-gated (`optimizer_name="Aurora"`) and default-off.
"""

import torch
import torch.nn as nn

# `bla_gpt/` is placed on sys.path by tests/conftest.py.
from optimizers import get_optimizer  # noqa: E402
from optimizers.aurora import Aurora, aurora, polar  # noqa: E402


# Baseline (default) optimizer args from ar/best_config.json — Muon, not Aurora.
BASELINE_MUON_ARGS = {
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "use_cautious_weight_decay": False,
}

# Aurora required fields from ar/full_runs/F73/config.json.
AURORA_ARGS = {
    "weight_decay": 0.025,
    "mu": 0.95,
    "nesterov": True,
    "pp_iterations": 2,
    "pp_beta": 0.5,
}


class TinyModel(nn.Module):
    """Tiny GPT-shaped model exercising both optimizer paths.

    `fc.weight` (12x8) is a rectangular matrix -> Aurora path. Everything else
    (fc.bias, norm, embedding, lm_head) -> AdamW backup path.
    """

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


def _split_names(model):
    """Replicate the registry split so tests can assert routing by name/ndim."""
    aurora_names = {
        name
        for name, p in model.named_parameters()
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
    }
    adamw_names = {
        name
        for name, p in model.named_parameters()
        if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name
    }
    return aurora_names, adamw_names


# ---------------------------------------------------------------------------
# 1. Default-off: the baseline (Muon) path is unchanged by adding Aurora.
# ---------------------------------------------------------------------------
def test_default_off_builds_muon_not_aurora():
    model = TinyModel()
    opt = get_optimizer("Muon", dict(BASELINE_MUON_ARGS), lr=0.001, model=model)
    assert type(opt).__name__ == "Muon"
    assert not isinstance(opt, Aurora)


# ---------------------------------------------------------------------------
# 2. Registry construction + param split.
# ---------------------------------------------------------------------------
def test_registry_builds_aurora():
    model = TinyModel()
    opt = get_optimizer("Aurora", dict(AURORA_ARGS), lr=0.05, model=model)
    assert isinstance(opt, Aurora)
    # Aurora hyperparameters propagate into the param group.
    g = opt.param_groups[0]
    assert g["weight_decay"] == 0.025
    assert g["mu"] == 0.95
    assert g["nesterov"] is True
    assert g["pp_iterations"] == 2
    assert g["pp_beta"] == 0.5


def test_registry_param_split_routes_matrices_to_aurora():
    model = TinyModel()
    opt = get_optimizer("Aurora", dict(AURORA_ARGS), lr=0.05, model=model)
    aurora_names, adamw_names = _split_names(model)
    # Only fc.weight is a non-embed/head matrix.
    assert aurora_names == {"fc.weight"}
    assert {"embed_tokens.weight", "lm_head.weight", "norm", "fc.bias"} <= adamw_names
    # The optimizer state flags match the intended routing.
    name_by_param = {p: name for name, p in model.named_parameters()}
    for p in opt.param_groups[0]["params"]:
        want_aurora = name_by_param[p] in aurora_names
        assert opt.state[p]["use_aurora"] is want_aurora


# ---------------------------------------------------------------------------
# 3. Finite update steps on rectangular and vector params.
# ---------------------------------------------------------------------------
def test_finite_updates_rectangular_and_vector():
    torch.manual_seed(0)
    tall = nn.Parameter(torch.randn(8, 4))   # rectangular (m > n) -> Aurora
    wide = nn.Parameter(torch.randn(4, 8))   # rectangular (m < n) -> Aurora
    square = nn.Parameter(torch.randn(5, 5))  # square -> Aurora (reduces to Muon)
    vec = nn.Parameter(torch.zeros(8))       # vector -> AdamW backup
    opt = Aurora(
        lr=0.05,
        aurora_params=[tall, wide, square],
        adamw_params=[vec],
        **AURORA_ARGS,
    )
    before = {id(p): p.detach().clone() for p in (tall, wide, square, vec)}
    for _ in range(3):  # multiple steps
        for p in (tall, wide, square, vec):
            p.grad = torch.randn_like(p)
        opt.step()
    for p in (tall, wide, square, vec):
        assert torch.isfinite(p).all(), "update produced non-finite values"
        moved = (p.detach() - before[id(p)]).abs().max().item()
        assert moved > 0.0, "parameter did not move"


def test_finite_update_non_nesterov():
    """Cover the nesterov=False branch (momentum.clone path)."""
    torch.manual_seed(1)
    tall = nn.Parameter(torch.randn(6, 3))
    args = dict(AURORA_ARGS)
    args["nesterov"] = False
    opt = Aurora(lr=0.05, aurora_params=[tall], adamw_params=[], **args)
    tall.grad = torch.randn_like(tall)
    opt.step()
    assert torch.isfinite(tall).all()


# ---------------------------------------------------------------------------
# 4. state_dict roundtrip.
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip():
    torch.manual_seed(2)
    tall = nn.Parameter(torch.randn(8, 4))
    vec = nn.Parameter(torch.zeros(8))
    opt = Aurora(lr=0.05, aurora_params=[tall], adamw_params=[vec], **AURORA_ARGS)
    for _ in range(2):
        tall.grad = torch.randn_like(tall)
        vec.grad = torch.randn_like(vec)
        opt.step()
    sd = opt.state_dict()

    tall2 = nn.Parameter(tall.detach().clone())
    vec2 = nn.Parameter(vec.detach().clone())
    opt2 = Aurora(lr=0.05, aurora_params=[tall2], adamw_params=[vec2], **AURORA_ARGS)
    opt2.load_state_dict(sd)

    st_new = opt2.state_dict()["state"]
    st_old = sd["state"]
    assert set(st_new.keys()) == set(st_old.keys())
    # Momentum buffer of the Aurora param survives the roundtrip byte-exactly.
    buf_old = next(v["momentum_buffer"] for v in st_old.values() if "momentum_buffer" in v)
    buf_new = next(v["momentum_buffer"] for v in st_new.values() if "momentum_buffer" in v)
    assert torch.equal(buf_old, buf_new)
    # A further step on the restored optimizer stays finite.
    tall2.grad = torch.randn_like(tall2)
    vec2.grad = torch.randn_like(vec2)
    opt2.step()
    assert torch.isfinite(tall2).all() and torch.isfinite(vec2).all()


# ---------------------------------------------------------------------------
# 5. Tiny forward / backward through the registry-built optimizer.
# ---------------------------------------------------------------------------
def test_tiny_forward_backward_step():
    torch.manual_seed(3)
    model = TinyModel()
    opt = get_optimizer("Aurora", dict(AURORA_ARGS), lr=0.05, model=model)
    idx = torch.randint(0, 16, (2, 5))
    target = torch.randint(0, 16, (2, 5))

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    logits = model(idx)
    loss = nn.functional.cross_entropy(logits.reshape(-1, 16), target.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()

    for n, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{n} became non-finite"
    # The Aurora-updated matrix and an AdamW-updated param both moved.
    assert not torch.equal(model.fc.weight.detach(), before["fc.weight"])
    assert not torch.equal(model.embed_tokens.weight.detach(), before["embed_tokens.weight"])


# ---------------------------------------------------------------------------
# Vendored primitives: polar is idempotent-ish; square Aurora == polar geometry.
# ---------------------------------------------------------------------------
def test_polar_maps_singular_values_to_one():
    torch.manual_seed(4)
    G = torch.randn(6, 4)
    U = polar(G).float()
    # Columns are near-orthonormal (tall polar factor): U^T U ~ I.
    gram = U.t() @ U
    assert torch.allclose(gram, torch.eye(4), atol=5e-2)


def test_aurora_function_requires_2d():
    W = torch.randn(3)
    G = torch.randn(3)
    m = torch.zeros(3)
    try:
        aurora(W, G, m)
    except ValueError:
        return
    raise AssertionError("aurora() must reject non-2D weight tensors")
