"""F107c: sequential Triton scan for the Lattice recurrence (CUDA only).

r_t = g_t * r_{t-1} + b_t, r_{-1} = r0, fp32. One program per (batch, 32
channels) walks T in a device loop, so Dynamo sees one opaque custom op and
adds no guard on T (the Hillis-Steele Python loop bounds T and forces a full
recompile at the seq-curriculum switch). Backward is a reverse device loop;
it saves only g and r (no per-pass autograd tensors).
"""
import torch
import triton
import triton.language as tl

_BD = 32


@triton.jit
def _scan_fwd_kernel(g_ptr, b_ptr, r0_ptr, out_ptr, T, D, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m = offs < D
    r = tl.load(r0_ptr + offs, mask=m, other=0.0)
    base = pid_b * T * D
    for t in range(T):
        p = base + t * D + offs
        gt = tl.load(g_ptr + p, mask=m, other=0.0)
        bt = tl.load(b_ptr + p, mask=m, other=0.0)
        r = gt * r + bt
        tl.store(out_ptr + p, r, mask=m)


@triton.jit
def _scan_bwd_kernel(g_ptr, r_ptr, r0_ptr, go_ptr, dg_ptr, db_ptr, dr0_ptr,
                     T, D, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m = offs < D
    r0v = tl.load(r0_ptr + offs, mask=m, other=0.0)
    base = pid_b * T * D
    h = tl.zeros([BLOCK_D], dtype=tl.float32)
    gnext = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(T):
        t = T - 1 - i
        p = base + t * D + offs
        h = tl.load(go_ptr + p, mask=m, other=0.0) + gnext * h
        tl.store(db_ptr + p, h, mask=m)
        gt = tl.load(g_ptr + p, mask=m, other=0.0)
        rp = tl.load(r_ptr + p - D, mask=m & (t > 0), other=0.0)
        rprev = tl.where(t > 0, rp, r0v)
        tl.store(dg_ptr + p, h * rprev, mask=m)
        gnext = gt
    tl.store(dr0_ptr + pid_b * D + offs, gnext * h, mask=m)


@torch.library.custom_op("blagpt::lattice_scan_fwd", mutates_args=())
def scan_fwd(g: torch.Tensor, b: torch.Tensor, r0: torch.Tensor) -> torch.Tensor:
    B, T, D = g.shape
    out = torch.empty_like(g)
    _scan_fwd_kernel[(B, triton.cdiv(D, _BD))](g, b, r0, out, T, D, BLOCK_D=_BD)
    return out


@scan_fwd.register_fake
def _(g, b, r0):
    return torch.empty_like(g)


@torch.library.custom_op("blagpt::lattice_scan_bwd", mutates_args=())
def scan_bwd(g: torch.Tensor, r: torch.Tensor, r0: torch.Tensor,
             go: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, D = g.shape
    dg = torch.empty_like(g)
    db = torch.empty_like(g)
    dr0 = torch.empty((B, D), device=g.device, dtype=torch.float32)
    _scan_bwd_kernel[(B, triton.cdiv(D, _BD))](g, r, r0, go.contiguous(), dg, db, dr0,
                                              T, D, BLOCK_D=_BD)
    return dg, db, dr0.sum(0)


@scan_bwd.register_fake
def _(g, r, r0, go):
    return torch.empty_like(g), torch.empty_like(g), r0.new_empty(r0.shape)


def _setup(ctx, inputs, output):
    g, _, r0 = inputs
    ctx.save_for_backward(g, output, r0)


def _backward(ctx, go):
    g, r, r0 = ctx.saved_tensors
    return scan_bwd(g, r, r0, go)


scan_fwd.register_autograd(_backward, setup_context=_setup)


def lattice_scan_triton(g, b, r0):
    """Drop-in for lattice.lattice_scan: same inputs, fp32 output."""
    return scan_fwd(g.float().contiguous(), b.float().contiguous(),
                    r0.float().contiguous())
