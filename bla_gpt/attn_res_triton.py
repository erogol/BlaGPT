"""F111 speed study: fused Triton 3-way AttnRes depth mix (CUDA).

Exact mathematical equivalent of running GPT._attn_res_mix three times for
(w_q, w_k, w_v) over the same source stack (N, B, T, D):
  score[n, m, k] = sum_d(RMSNorm(srcs[n, m, d]) * W[k, d])
  alpha[:, m, k] = softmax_n(score[:, m, k])
  out_k[m, d]    = sum_n(alpha[n, m, k] * srcs[n, m, d])

Saves only (srcs, W, alpha[N, M, 3], rstd[N, M]) for backward instead of 3
copies of normalized sources. Uses register-vector softmax over depth N (no
global scratch buffer, no read-after-write hazards). Wrapped with
torch.library.custom_op + register_fake + register_autograd so torch.compile
compiles with zero graph breaks.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_res_3way_fwd_kernel(
    srcs_ptr,
    w_ptr,
    xq_ptr,
    xk_ptr,
    xv_ptr,
    alpha_ptr,
    rstd_ptr,
    N,
    M,
    D,
    eps,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    z64 = pid_m * 0
    M_i64 = z64 + M
    D_i64 = z64 + D
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_n = tl.arange(0, BLOCK_N)
    valid_n = offs_n < N

    w0 = tl.load(w_ptr + 0 * D_i64 + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + 1 * D_i64 + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + 2 * D_i64 + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    s0_vec = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    s1_vec = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    s2_vec = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    inv_D = 1.0 / D

    for n in range(N):
        nm = (z64 + n) * M_i64 + pid_m
        v_off = nm * D_i64 + offs_d
        v = tl.load(srcs_ptr + v_off, mask=mask_d, other=0.0).to(tl.float32)
        mean_sq = tl.sum(v * v, axis=0) * inv_D
        r = tl.rsqrt(mean_sq + eps)
        tl.store(rstd_ptr + nm, r)
        vn = v * r
        s0 = tl.sum(vn * w0, axis=0)
        s1 = tl.sum(vn * w1, axis=0)
        s2 = tl.sum(vn * w2, axis=0)
        mask_n = offs_n == n
        s0_vec = tl.where(mask_n, s0, s0_vec)
        s1_vec = tl.where(mask_n, s1, s1_vec)
        s2_vec = tl.where(mask_n, s2, s2_vec)

    m0 = tl.max(s0_vec, axis=0)
    m1 = tl.max(s1_vec, axis=0)
    m2 = tl.max(s2_vec, axis=0)
    e0 = tl.where(valid_n, tl.exp(s0_vec - m0), 0.0)
    e1 = tl.where(valid_n, tl.exp(s1_vec - m1), 0.0)
    e2 = tl.where(valid_n, tl.exp(s2_vec - m2), 0.0)
    a0_vec = e0 / tl.sum(e0, axis=0)
    a1_vec = e1 / tl.sum(e1, axis=0)
    a2_vec = e2 / tl.sum(e2, axis=0)

    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n in range(N):
        nm = (z64 + n) * M_i64 + pid_m
        mask_n = offs_n == n
        a0 = tl.sum(tl.where(mask_n, a0_vec, 0.0), axis=0)
        a1 = tl.sum(tl.where(mask_n, a1_vec, 0.0), axis=0)
        a2 = tl.sum(tl.where(mask_n, a2_vec, 0.0), axis=0)
        a_base = nm * 3
        tl.store(alpha_ptr + a_base + 0, a0)
        tl.store(alpha_ptr + a_base + 1, a1)
        tl.store(alpha_ptr + a_base + 2, a2)
        v_off = nm * D_i64 + offs_d
        v = tl.load(srcs_ptr + v_off, mask=mask_d, other=0.0).to(tl.float32)
        acc0 += a0 * v
        acc1 += a1 * v
        acc2 += a2 * v

    out_off = pid_m * D_i64 + offs_d
    tl.store(xq_ptr + out_off, acc0, mask=mask_d)
    tl.store(xk_ptr + out_off, acc1, mask=mask_d)
    tl.store(xv_ptr + out_off, acc2, mask=mask_d)


@triton.jit
def _attn_res_3way_bwd_kernel(
    srcs_ptr,
    w_ptr,
    alpha_ptr,
    rstd_ptr,
    gxq_ptr,
    gxk_ptr,
    gxv_ptr,
    dsrcs_ptr,
    dw_partial_ptr,
    N,
    M,
    D,
    num_chunks,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    z64 = pid * 0
    M_i64 = z64 + M
    D_i64 = z64 + D
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_n = tl.arange(0, BLOCK_N)
    inv_D = 1.0 / D

    w0 = tl.load(w_ptr + 0 * D_i64 + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + 1 * D_i64 + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + 2 * D_i64 + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    dw0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    dw1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    dw2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    chunk_size = (M_i64 + num_chunks - 1) // num_chunks
    m_start = pid * chunk_size
    m_end = tl.minimum(m_start + chunk_size, M_i64)

    for m in range(m_start, m_end):
        g_off = m * D_i64 + offs_d
        gq = tl.load(gxq_ptr + g_off, mask=mask_d, other=0.0).to(tl.float32)
        gk = tl.load(gxk_ptr + g_off, mask=mask_d, other=0.0).to(tl.float32)
        gv = tl.load(gxv_ptr + g_off, mask=mask_d, other=0.0).to(tl.float32)

        a0_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
        a1_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
        a2_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
        da0_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
        da1_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
        da2_vec = tl.zeros([BLOCK_N], dtype=tl.float32)

        for n in range(N):
            nm = (z64 + n) * M_i64 + m
            v_off = nm * D_i64 + offs_d
            v = tl.load(srcs_ptr + v_off, mask=mask_d, other=0.0).to(tl.float32)
            a_base = nm * 3
            a0 = tl.load(alpha_ptr + a_base + 0)
            a1 = tl.load(alpha_ptr + a_base + 1)
            a2 = tl.load(alpha_ptr + a_base + 2)
            da0 = tl.sum(gq * v, axis=0)
            da1 = tl.sum(gk * v, axis=0)
            da2 = tl.sum(gv * v, axis=0)
            mask_n = offs_n == n
            a0_vec = tl.where(mask_n, a0, a0_vec)
            a1_vec = tl.where(mask_n, a1, a1_vec)
            a2_vec = tl.where(mask_n, a2, a2_vec)
            da0_vec = tl.where(mask_n, da0, da0_vec)
            da1_vec = tl.where(mask_n, da1, da1_vec)
            da2_vec = tl.where(mask_n, da2, da2_vec)

        dot0 = tl.sum(a0_vec * da0_vec, axis=0)
        dot1 = tl.sum(a1_vec * da1_vec, axis=0)
        dot2 = tl.sum(a2_vec * da2_vec, axis=0)
        ds0_vec = a0_vec * (da0_vec - dot0)
        ds1_vec = a1_vec * (da1_vec - dot1)
        ds2_vec = a2_vec * (da2_vec - dot2)

        for n in range(N):
            nm = (z64 + n) * M_i64 + m
            v_off = nm * D_i64 + offs_d
            v = tl.load(srcs_ptr + v_off, mask=mask_d, other=0.0).to(tl.float32)
            r = tl.load(rstd_ptr + nm)
            mask_n = offs_n == n
            a0 = tl.sum(tl.where(mask_n, a0_vec, 0.0), axis=0)
            a1 = tl.sum(tl.where(mask_n, a1_vec, 0.0), axis=0)
            a2 = tl.sum(tl.where(mask_n, a2_vec, 0.0), axis=0)
            ds0 = tl.sum(tl.where(mask_n, ds0_vec, 0.0), axis=0)
            ds1 = tl.sum(tl.where(mask_n, ds1_vec, 0.0), axis=0)
            ds2 = tl.sum(tl.where(mask_n, ds2_vec, 0.0), axis=0)

            vn = v * r
            dw0 += ds0 * vn
            dw1 += ds1 * vn
            dw2 += ds2 * vn

            d_vnorm = ds0 * w0 + ds1 * w1 + ds2 * w2
            c = tl.sum(d_vnorm * vn, axis=0) * inv_D
            dv = (a0 * gq + a1 * gk + a2 * gv) + r * (d_vnorm - c * vn)
            tl.store(dsrcs_ptr + v_off, dv, mask=mask_d)

    dw_base = pid * 3 * D_i64
    tl.store(dw_partial_ptr + dw_base + 0 * D_i64 + offs_d, dw0, mask=mask_d)
    tl.store(dw_partial_ptr + dw_base + 1 * D_i64 + offs_d, dw1, mask=mask_d)
    tl.store(dw_partial_ptr + dw_base + 2 * D_i64 + offs_d, dw2, mask=mask_d)


def _out_dtype(srcs: torch.Tensor, W: torch.Tensor) -> torch.dtype:
    return torch.promote_types(srcs.dtype, W.dtype)


@torch.library.custom_op("blagpt::attn_res_3way_fwd", mutates_args=())
def attn_res_3way_fwd(
    srcs: torch.Tensor, W: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    srcs_c = srcs.contiguous()
    W_c = W.contiguous()
    N, M, D = srcs_c.shape
    out_dt = _out_dtype(srcs_c, W_c)
    xq = torch.empty((M, D), device=srcs_c.device, dtype=out_dt)
    xk = torch.empty((M, D), device=srcs_c.device, dtype=out_dt)
    xv = torch.empty((M, D), device=srcs_c.device, dtype=out_dt)
    alpha = torch.empty((N, M, 3), device=srcs_c.device, dtype=torch.float32)
    rstd = torch.empty((N, M), device=srcs_c.device, dtype=torch.float32)
    eps = float(torch.finfo(srcs_c.dtype).eps)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = triton.next_power_of_2(max(N, 1))
    num_warps = 4 if BLOCK_D <= 1024 else 8
    _attn_res_3way_fwd_kernel[(M,)](
        srcs_c,
        W_c,
        xq,
        xk,
        xv,
        alpha,
        rstd,
        N,
        M,
        D,
        eps,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return xq, xk, xv, alpha, rstd


@attn_res_3way_fwd.register_fake
def _(srcs: torch.Tensor, W: torch.Tensor):
    N, M, D = srcs.shape
    out_dt = _out_dtype(srcs, W)
    return (
        srcs.new_empty((M, D), dtype=out_dt),
        srcs.new_empty((M, D), dtype=out_dt),
        srcs.new_empty((M, D), dtype=out_dt),
        srcs.new_empty((N, M, 3), dtype=torch.float32),
        srcs.new_empty((N, M), dtype=torch.float32),
    )


@torch.library.custom_op("blagpt::attn_res_3way_bwd", mutates_args=())
def attn_res_3way_bwd(
    srcs: torch.Tensor,
    W: torch.Tensor,
    alpha: torch.Tensor,
    rstd: torch.Tensor,
    gxq: torch.Tensor,
    gxk: torch.Tensor,
    gxv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    srcs_c = srcs.contiguous()
    W_c = W.contiguous()
    alpha_c = alpha.contiguous()
    rstd_c = rstd.contiguous()
    gxq_c = gxq.contiguous()
    gxk_c = gxk.contiguous()
    gxv_c = gxv.contiguous()
    N, M, D = srcs_c.shape
    d_srcs = torch.empty_like(srcs_c)
    num_chunks = min(M, 528) if M > 0 else 1
    dw_partial = torch.empty((num_chunks, 3, D), device=srcs_c.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = triton.next_power_of_2(max(N, 1))
    num_warps = 4 if BLOCK_D <= 1024 else 8
    _attn_res_3way_bwd_kernel[(num_chunks,)](
        srcs_c,
        W_c,
        alpha_c,
        rstd_c,
        gxq_c,
        gxk_c,
        gxv_c,
        d_srcs,
        dw_partial,
        N,
        M,
        D,
        num_chunks,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    dW = dw_partial.sum(dim=0).to(W_c.dtype)
    return d_srcs, dW


@attn_res_3way_bwd.register_fake
def _(srcs, W, alpha, rstd, gxq, gxk, gxv):
    return torch.empty_like(srcs), torch.empty_like(W)


def _setup_context(ctx, inputs, output):
    srcs, W = inputs
    _xq, _xk, _xv, alpha, rstd = output
    ctx.save_for_backward(srcs, W, alpha, rstd)


def _backward(ctx, gxq, gxk, gxv, _galpha, _grstd):
    srcs, W, alpha, rstd = ctx.saved_tensors
    _, M, D = srcs.shape
    out_dt = _out_dtype(srcs, W)
    if gxq is None:
        gxq = torch.zeros((M, D), device=srcs.device, dtype=out_dt)
    if gxk is None:
        gxk = torch.zeros((M, D), device=srcs.device, dtype=out_dt)
    if gxv is None:
        gxv = torch.zeros((M, D), device=srcs.device, dtype=out_dt)
    return attn_res_3way_bwd(srcs, W, alpha, rstd, gxq, gxk, gxv)


attn_res_3way_fwd.register_autograd(_backward, setup_context=_setup_context)


def attn_res_mix_3way_triton(
    srcs_list: list[torch.Tensor],
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in 3-way AttnRes depth mix for (w_q, w_k, w_v) on CUDA."""
    B, T, D = srcs_list[0].shape
    N = len(srcs_list)
    srcs = torch.stack(srcs_list, dim=0).reshape(N, B * T, D)
    W = torch.stack((w_q, w_k, w_v), dim=0)
    xq, xk, xv, _, _ = attn_res_3way_fwd(srcs, W)
    return xq.view(B, T, D), xk.view(B, T, D), xv.view(B, T, D)
