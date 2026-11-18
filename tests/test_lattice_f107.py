"""Tests for F107: Lattice of Latents (arXiv:2611.21608)."""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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

from lattice import lattice_scan, lattice_scan_ref  # noqa: E402


def _mk_cfg(lattice, hyper=False, **kw):
    cfg = GPTConfig()
    base = json.load(open(ROOT / "ar" / "full_train" / "configs" / "E7b_h200_control.json"))
    for k, v in base.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_layer = 6
    cfg.n_embd = 128
    cfg.n_head = 4
    cfg.n_kv_head = 4
    cfg.mlp_expand = 2
    cfg.block_size = 64
    cfg.vocab_size = 512
    cfg.engram_vocab_mult = 2
    cfg.use_hyper_connections = hyper
    if hyper:
        cfg.use_attn_res = False
    cfg.lattice_enabled = lattice
    cfg.lattice_dr = 32
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def _build(cfg, device, seed=0):
    torch.manual_seed(seed)
    return GPT(cfg).to(device)


def _logits(m, x):
    m.eval()
    with torch.no_grad():
        out = m(x)
    return out[0] if isinstance(out, tuple) else out


def test_scan_matches_loop():
    torch.manual_seed(0)
    for T in (1, 7, 64, 1024):
        g = torch.rand(2, T, 48)
        b = torch.randn(2, T, 48)
        r0 = torch.randn(48)
        err = (lattice_scan(g, b, r0) - lattice_scan_ref(g, b, r0)).abs().max().item()
        assert err < 1e-4, (T, err)


def test_scan_stable_with_tiny_gates():
    g = torch.full((1, 1024, 8), 1e-30)
    b = torch.randn(1, 1024, 8)
    out = lattice_scan(g, b, torch.zeros(8))
    assert torch.isfinite(out).all()
    assert (out - lattice_scan_ref(g, b, torch.zeros(8))).abs().max() < 1e-5


BASE_SHA = "d07a4640"  # BlaGPT main before F107


def test_flag_off_has_no_lattice(device):
    m = _build(_mk_cfg(False), device)
    assert m.lattice is None
    assert not any("lattice" in n for n, _ in m.named_parameters())


@pytest.mark.parametrize("hyper", [False, True])
def test_flag_off_bit_identical_to_base(device, hyper, tmp_path):
    """Flag off: same params and bit-identical logits/loss as the pre-F107 code."""
    import subprocess
    import tarfile

    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_SHA, "bla_gpt", "-o", str(tar)])
    if r.returncode != 0:
        pytest.skip(f"git archive {BASE_SHA} not available")
    base = tmp_path / "base"
    with tarfile.open(tar) as t:
        t.extractall(base)
    assert (base / "bla_gpt" / "bla_gpt.py").exists()
    cfg_json = ROOT / "ar" / "full_train" / "configs" / "E7b_h200_control.json"
    helper = ROOT / "tests" / "_lattice_dump.py"
    outs = []
    for tree in (base, ROOT):
        out = tmp_path / f"{tree.name}.pt"
        subprocess.run([sys.executable, str(helper), str(tree), str(out),
                        "1" if hyper else "0", str(cfg_json)], check=True)
        outs.append(torch.load(out, weights_only=False))
    a, b = outs
    assert a["sd"].keys() == b["sd"].keys()
    for k in a["sd"]:
        assert torch.equal(a["sd"][k], b["sd"][k]), k
    assert torch.equal(a["logits"], b["logits"])
    la = a["loss"]["total"] if isinstance(a["loss"], dict) else a["loss"]
    lb = b["loss"]["total"] if isinstance(b["loss"], dict) else b["loss"]
    assert torch.equal(la, lb)


@pytest.mark.parametrize("hyper", [False, True])
def test_zero_init_matches_no_lattice(device, hyper):
    """Step 0: zero-init FiLM/MLP/inj maps -> same output as the base model."""
    m_off = _build(_mk_cfg(False, hyper), device)
    m_on = _build(_mk_cfg(True, hyper), device)
    missing = m_on.load_state_dict(m_off.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all("lattice" in k for k in missing.missing_keys)
    x = torch.randint(0, 512, (2, 32), device=device)
    a, b = _logits(m_off, x), _logits(m_on, x)
    assert (a - b).abs().max().item() < 1e-5


@pytest.mark.parametrize("hyper", [False, True])
def test_shapes_and_grads(device, hyper):
    m = _build(_mk_cfg(True, hyper), device)
    assert sorted(m.lattice.keys(), key=int) == ["1", "2", "3", "4"]
    # E7b zero-inits the attention and Primer output projections, so at
    # step 0 no gradient reaches q/k/v or the MLP pre-activation. Give
    # them small values so the FiLM/MLP heads must get a gradient.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("c_proj") or n.endswith("c_proj.weight"):
                p.normal_(std=0.02)
    m.train()
    x = torch.randint(0, 512, (2, 32), device=device)
    out = m(x, x)
    loss = out[1]["total"] if isinstance(out[1], dict) else out[1]
    loss.backward()
    for k, mod in m.lattice.items():
        r = m._lattice_holder[k]
        assert r.shape == (2, 32, 32)
        # zero-init heads get a gradient at step 0
        assert mod.film_q.weight.grad is not None
        assert mod.film_q.weight.grad.abs().sum() > 0
        assert mod.mlp_beta.weight.grad.abs().sum() > 0


def test_explicit_layers(device):
    m = _build(_mk_cfg(True, lattice_layers=[2, 3]), device)
    assert sorted(m.lattice.keys()) == ["2", "3"]
