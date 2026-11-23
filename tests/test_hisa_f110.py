"""Tests for F110: HiSA (arXiv 2611.26363), exact per-pair rule, on F107b_e5."""
import json
import math
import subprocess
import sys
import tarfile
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

import attentions as A  # noqa: E402

BASE_SHA = "187cb7de"  # BlaGPT main before F110
CFG_JSON = ROOT / "ar" / "full_train" / "configs" / "F107b_e5.json"


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda; Triton kernel)")
    return "cuda"


def _mk_cfg(**kw):
    cfg = GPTConfig()
    for k, v in json.load(open(CFG_JSON)).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_embd, cfg.n_head, cfg.n_kv_head = 256, 4, 4  # head_dim 64 (kernel size)
    cfg.mlp_expand, cfg.block_size, cfg.vocab_size, cfg.engram_vocab_mult = 2, 128, 512, 2
    cfg.lattice_dr = 32
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _model(device, **kw):
    torch.manual_seed(0)
    m = GPT(_mk_cfg(**kw)).to(device)
    with torch.no_grad():  # F107b_e5 zero-inits output projections and the sink prior
        for n, p in m.named_parameters():
            if n.endswith("c_proj") or n.endswith("c_proj.weight"):
                p.normal_(std=0.02)
            if n.endswith("sink_prior"):
                p.normal_(std=0.5)
    return m


def _naive(q, k, v, base, angle_deg, sink=None):
    """Eq. 3 by a loop over (i, j, l): rotation angle test in float64, half-split pairs."""
    B, H, T, d = q.shape
    P = d // 2
    phi = math.radians(angle_deg)
    s = torch.full((B, H, T, T), float("-inf"), dtype=torch.float64)
    qd, kd = q.double().cpu(), k.double().cpu()
    for i in range(T):
        for j in range(i + 1):
            acc, any_kept = 0.0, False
            for l in range(P):
                if (i - j) * base ** (-2.0 * l / d) < phi:
                    acc = acc + qd[:, :, i, l] * kd[:, :, j, l] + qd[:, :, i, l + P] * kd[:, :, j, l + P]
                    any_kept = True
            if any_kept:
                s[:, :, i, j] = acc / math.sqrt(d)
    if sink is not None:
        s[:, :, :, 0] = s[:, :, :, 0] + sink.double().cpu()[None, :, None]
    return (torch.softmax(s, -1) @ v.double().cpu()).float()


def test_flag_off_bit_identical_to_base(device, tmp_path):
    """hisa=False: same params, logits and loss as the code at 187cb7de."""
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_SHA, "bla_gpt", "-o", str(tar)])
    assert r.returncode == 0, f"git archive {BASE_SHA} failed"
    base = tmp_path / "base"
    with tarfile.open(tar) as t:
        t.extractall(base)
    assert len(list((base / "bla_gpt").glob("*.py"))) > 10
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
    assert torch.equal(a["loss"], b["loss"])


@pytest.mark.parametrize("base", [1e4, 1e6])
def test_rho_matches_eq5(base):
    for T in (384, 1024):
        brute = 0.0
        for l in range(32):
            w = math.radians(90.0) / base ** (-2.0 * l / 64)
            brute += sum(T - D for D in range(T) if D < w)
        brute /= 32 * T * (T + 1) / 2
        assert abs(A.hisa_rho(T, base) - brute) < 1e-12


def test_thresholds_exact():
    for base in (1e4, 1e6):
        L = A.hisa_thresholds(base, 64, 90.0)
        for l in range(32):
            th = base ** (-2.0 * l / 64)
            D = int(L[l])
            assert (D - 1) * th < math.pi / 2 <= D * th
    assert (A.hisa_thresholds(1e4, 64, float("inf")) == 1 << 30).all()


@pytest.mark.parametrize("base,sink", [(1e4, False), (1e4, True), (1e2, True), (1e6, True)])
def test_reference_and_kernel_vs_naive_loop(device, base, sink):
    torch.manual_seed(1)
    B, H, T, d = 1, 2, 40, 64
    q, k, v = (torch.randn(B, H, T, d, device=device) for _ in range(3))
    u = torch.randn(H, device=device) if sink else None
    L = A.hisa_thresholds(base, d, 90.0)
    want = _naive(q, k, v, base, 90.0, u)
    ref = A.hisa_reference(q, k, v, L.to(device), u).cpu()
    assert torch.allclose(ref, want, atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        got = A.hisa_attention(q, k, v, L.to(device), int(L.max()), sink=u).cpu()
    # kernel runs fp32 inputs with ieee dots
    assert torch.allclose(got, want, atol=2e-4, rtol=2e-4), (got - want).abs().max()


def test_kernel_grads_vs_reference(device):
    torch.manual_seed(2)
    B, H, T, d = 2, 3, 257, 64
    L = A.hisa_thresholds(1e4, d, 90.0).to(device)
    q0, k0, v0 = (torch.randn(B, H, T, d, device=device) for _ in range(3))
    u0 = torch.randn(H, device=device)
    go = torch.randn(B, H, T, d, device=device)
    outs = []
    for fn in ("ref", "kernel"):
        q, k, v, u = (t.clone().requires_grad_() for t in (q0, k0, v0, u0))
        if fn == "ref":
            y = A.hisa_reference(q, k, v, L, u)
        else:
            y = A.hisa_attention(q, k, v, L, int(L.max()), sink=u)
        (y * go).sum().backward()
        outs.append((y.detach(), q.grad, k.grad, v.grad, u.grad))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), (a - b).abs().max()


def test_phi_inf_equals_dense(device):
    """hisa=True with Phi = inf gives the dense model (same params; logits equal up to kernel rounding)."""
    x = torch.randint(0, 512, (2, 128), device=device)
    mods = [_model(device).eval(), _model(device, hisa=True, hisa_angle_deg=float("inf")).eval()]
    sd = mods[0].state_dict()
    assert sd.keys() == mods[1].state_dict().keys()
    for k_ in sd:
        assert torch.equal(sd[k_], mods[1].state_dict()[k_]), k_
    with torch.no_grad():
        outs = [m(x, x) for m in mods]
    assert torch.allclose(outs[0][0], outs[1][0], atol=2e-3, rtol=2e-3), (outs[0][0] - outs[1][0]).abs().max()
    assert abs(float(outs[0][1]) - float(outs[1][1])) < 1e-3
    # finite Phi changes the model
    m90 = _model(device, hisa=True).eval()
    with torch.no_grad():
        l90 = m90(x, x)[0]
    assert not torch.allclose(outs[0][0], l90, atol=1e-3)


def test_hisa_model_compiled_bf16_train_step(device):
    """Full-run path: torch.compile + bf16 autocast, forward + backward, finite grads, matches eager."""
    m = _model(device, hisa=True, rope_theta=1e4).train()
    x = torch.randint(0, 512, (2, 128), device=device)
    attn = m.transformer.h[0].attn if hasattr(m, "transformer") else None
    if attn is not None:
        assert attn.hisa and attn.head_dim == 64
    cm = torch.compile(m)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss_c = cm(x, x)
        _, loss_e = m(x, x)
    loss_c.backward()
    assert torch.isfinite(loss_c)
    assert abs(float(loss_c) - float(loss_e)) < 2e-2
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
