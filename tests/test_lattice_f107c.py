"""Tests for F107c: Triton Lattice scan (lattice_scan_impl="triton").

The Triton scan must match lattice_scan_ref (fwd and grads, fp32) and must
not add a guard on T, so a dynamic graph compiled at a short length also
serves T=1024 (no full recompile at the seq-curriculum switch).
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

import lattice  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _inputs(B, T, D, seed=0):
    torch.manual_seed(seed)
    delta = torch.nn.functional.softplus(torch.randn(B, T, D, device="cuda") - 3)
    g = torch.exp(-torch.empty(D, device="cuda").uniform_(1, 16) * delta)
    b = (1 - g) * torch.nn.functional.silu(torch.randn(B, T, D, device="cuda"))
    r0 = torch.randn(D, device="cuda") * 0.1
    return g, b, r0


def _fwd_bwd(fn, g, b, r0, go):
    g, b, r0 = (t.clone().requires_grad_() for t in (g, b, r0))
    out = fn(g, b, r0)
    out.backward(go)
    return out.detach(), g.grad, b.grad, r0.grad


def test_unknown_scan_impl_raises():
    with pytest.raises(ValueError):
        lattice.LatticeLayer(64, 16, 64, 64, 128, scan_impl="nope")


@cuda
@pytest.mark.parametrize("T", [1, 7, 384, 1024])
def test_triton_scan_matches_ref(T):
    from lattice_triton import lattice_scan_triton
    g, b, r0 = _inputs(4, T, 256)
    go = torch.randn_like(g)
    ref = _fwd_bwd(lattice.lattice_scan_ref, g, b, r0, go)
    got = _fwd_bwd(lattice_scan_triton, g, b, r0, go)
    for a, r in zip(got, ref):
        assert (a - r).abs().max().item() < 1e-4


@cuda
def test_triton_layer_matches_hs_layer():
    torch.manual_seed(0)
    hs = lattice.LatticeLayer(128, 32, 128, 128, 256, scan_impl="hs").cuda()
    tr = lattice.LatticeLayer(128, 32, 128, 128, 256, scan_impl="triton").cuda()
    tr.load_state_dict(hs.state_dict())
    x = torch.randn(2, 96, 128, device="cuda")
    e = torch.randn(2, 96, 128, device="cuda")
    assert (hs.update(x, e) - tr.update(x, e)).abs().max().item() < 1e-4


@cuda
@pytest.mark.parametrize("impl,max_frames", [("triton", 2), ("hs", None)])
def test_switch_recompile(impl, max_frames):
    """384 -> 500 makes the graph dynamic; 1024 must reuse it for triton.

    hs bounds T by its pass count (258..512), so 1024 adds a third frame.
    """
    from torch._dynamo.testing import CompileCounter
    from lattice_triton import lattice_scan_triton
    fn = lattice_scan_triton if impl == "triton" else lattice.lattice_scan
    torch._dynamo.reset()
    cnt = CompileCounter()
    cfn = torch.compile(fn, backend=cnt)
    for T in (384, 500, 1024):
        g, b, r0 = _inputs(2, T, 64)
        cfn(g, b, r0)
    if max_frames is None:
        assert cnt.frame_count > 2
    else:
        assert cnt.frame_count <= max_frames
