"""Tests for F109: latent feedback (arXiv:2611.20404; LoRM lorm/model/feedback.py) on F107b_e5."""
import json
import subprocess
import sys
import tarfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig

BASE_SHA = "614f81c6"  # BlaGPT main before F109
CFG_JSON = ROOT / "ar" / "full_train" / "configs" / "F107b_e5.json"


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def _mk_cfg(**kw):
    cfg = GPTConfig()
    for k, v in json.load(open(CFG_JSON)).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_embd, cfg.n_head, cfg.n_kv_head = 128, 4, 4
    cfg.mlp_expand, cfg.block_size, cfg.vocab_size, cfg.engram_vocab_mult = 2, 64, 512, 2
    cfg.lattice_dr = 32
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _model(device, **kw):
    torch.manual_seed(0)
    m = GPT(_mk_cfg(**kw)).to(device)
    # F107b_e5 zero-inits output projections; give them values so blocks are not the identity.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("c_proj") or n.endswith("c_proj.weight"):
                p.normal_(std=0.02)
    return m


class _Wrap:
    """Stand-in for the DDP wrapper: callable, with .module."""

    def __init__(self, m):
        self.module = m

    def __call__(self, *a, **k):
        return self.module(*a, **k)


def _loss(o):
    return o["total"] if isinstance(o, dict) else o


def test_flag_off_bit_identical_to_base(device, tmp_path):
    """latent_feedback=False: same params, logits and loss as the code at 614f81c6."""
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_SHA, "bla_gpt", "-o", str(tar)])
    assert r.returncode == 0, f"git archive {BASE_SHA} failed"
    base = tmp_path / "base"
    with tarfile.open(tar) as t:
        t.extractall(base)
    helper = ROOT / "tests" / "_mbt_dump.py"
    outs = []
    for tree in (base, ROOT):
        out = tmp_path / f"{tree.name}.pt"
        subprocess.run([sys.executable, str(helper), str(tree), str(out), str(CFG_JSON)], check=True)
        outs.append(torch.load(out, weights_only=False))
    a, b = outs
    assert a["sd"].keys() == b["sd"].keys()
    for k in a["sd"]:
        assert torch.equal(a["sd"][k], b["sd"][k]), k
    assert torch.equal(a["logits"], b["logits"])
    assert torch.equal(_loss(a["loss"]), _loss(b["loss"]))


def test_fusion_shape_and_init(device):
    m = _model(device, latent_feedback=True)
    W = m.lf_fusion.weight
    assert W.shape == (128, 128) and m.lf_fusion.bias is None
    assert abs(W.std().item() - 0.02) < 0.002
    assert not hasattr(_model(device), "lf_fusion")


def test_feedback_shift(device):
    """Position 0 keeps its embedding; the top state of the last position feeds nothing."""
    m = _model(device, latent_feedback=True).eval()
    x = torch.randint(0, 512, (2, 32), device=device)
    with torch.no_grad():
        top = m(x, return_top=True)
        assert top.shape == (2, 32, 128)
        plain, _ = m(x, x)
        fed, _ = m(x, x, feedback=top)
        assert torch.allclose(plain[:, 0], fed[:, 0])
        assert not torch.allclose(plain[:, 1:], fed[:, 1:])
        top2 = top.clone()
        top2[:, -1] += 5.0
        fed2, _ = m(x, x, feedback=top2)
        assert torch.equal(fed, fed2)


def test_lf_passes_deterministic_and_p():
    tr = pytest.importorskip("train")
    a = [tr.lf_passes(1337, s, 0.15) for s in range(4000)]
    assert a == [tr.lf_passes(1337, s, 0.15) for s in range(4000)]
    assert set(a) == {1, 2}
    frac1 = a.count(1) / len(a)
    assert abs(frac1 - 0.15) < 0.03
    assert a[:50] != [tr.lf_passes(2, s, 0.15) for s in range(50)]
    rng = np.random.default_rng([1337, 7])  # LoRM SampledPasses draw
    assert tr.lf_passes(1337, 7, 0.15) == int(rng.choice(2, p=[0.15, 0.85])) + 1


def test_lf_forward_grads(device):
    """2 passes: loss of the fed pass only, fusion gets a gradient; 1 pass: no fusion gradient."""
    tr = pytest.importorskip("train")
    m = _model(device, latent_feedback=True).train()
    w = _Wrap(m)
    x = torch.randint(0, 512, (2, 32), device=device)
    _, l1 = tr.lf_forward(w, x, x, 1)
    _loss(l1).backward()
    assert m.lf_fusion.weight.grad is None
    m.zero_grad(set_to_none=True)
    _, l2 = tr.lf_forward(w, x, x, 2)
    _loss(l2).backward()
    g = m.lf_fusion.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert torch.isfinite(_loss(l2))
