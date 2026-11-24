"""F110 HiSA (arXiv 2611.26363), exact rule: Triton kernels (forward + backward).

Derived from hisa/kernels/triton_exact.py of github.com/ambroise-b/hisa (commit 0fae2ce),
Copyright 2026 Ambroise Borbely, Mathieu Salzmann, Olivier Verscheure, Apache License 2.0.
Modifications for BlaGPT: thresholds and chunk permutation defined here (no `hisa` package);
custom op namespace `blagpt_hisa`; the backward also takes the gradient of the returned base-2
log-sum-exp (used by the GOAT sink-prior correction); `hisa_triton_exact` returns (o, lse2).

Original module docstring follows.

HiSA, exact rule: Triton kernels (forward + backward), FlashAttention-2 style.

Computes (see `attentions.hisa_reference`, the readable definition)

    s_ij = 1/sqrt(d) * sum_p 1(0 <= i - j < w_p) <q_i[p], k_j[p]>,   w_p = Phi / theta_p,
    s_ij = -inf if no pair is kept,

softmax over j, then P V. Inputs are the rotated q [B, H, T, 64] and k, v [B, H_kv, T, 64] (GQA: query head h
uses kv head h // (H / H_kv)), Hugging Face pair layout (pair p = features p and p + 32).

Exactness: the rule is passed as integer thresholds L_p = ceil(w_p) computed on the host in float64
(`thresholds`), so the integer test D < L_p is exactly the test D < w_p.

Layout: q and k are permuted once so that chunk c (16 contiguous features) holds pairs 8c .. 8c+7; a suffix of
pairs is then a suffix of chunks. On a query x key tile with distances in [dmin, dmax]:
    p <  a_lo          dropped on the whole tile (skipped),
    a_lo <= p < a_hi   "mixed": kept only where D < L_p,
    p >= a_hi          kept on the whole tile.
Keys of a query block are processed in 4 contiguous segments (far to near); in segment C0 the chunks c >= C0 are
computed by tensor-core dots over 16-feature slices (only those slices of k are loaded), with pairs below
a_eff = max(a_hi, 8 C0) masked; every pair in [a_lo, a_eff) gets an exact 16-wide masked dot under D < L_p.
Segment boundaries only affect speed, never the result. Loop bodies have no data-dependent loads, so Triton can
pipeline them; masks are applied only on tiles that need them. Backward uses the same decomposition with
per-chunk dq / dk accumulators. Softmax uses exp2 (log2(e) folded into the scale); the saved row log-sum-exp is
in base 2.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


BIG = 1 << 30  # integer threshold for "always kept" (infinite budget = dense attention)


def thresholds(theta: torch.Tensor, max_angle_deg: float, device=None) -> torch.Tensor:
    """Integer windows L_p = ceil(Phi / theta_p) (int32 [P], capped at BIG), computed in float64:
    for an integer distance D, D < L_p  <=>  D * theta_p < Phi. Phi = inf gives BIG (dense)."""
    theta = theta.double().cpu()
    if not math.isfinite(max_angle_deg):
        return torch.full(theta.shape, BIG, dtype=torch.int32, device=device)
    w = math.radians(max_angle_deg) / theta
    return torch.ceil(w).clamp(max=BIG).to(torch.int32).to(device)

LOG2E = 1.4426950408889634
HEAD_DIM = 64
CHUNK = 16     # features per chunk (tensor-core K atom); 8 pairs


def chunk_permutation(d: int = HEAD_DIM) -> torch.Tensor:
    """perm with x_chunked = x[..., perm]: chunk c = features [8c, 8c+8) then [P+8c, P+8c+8)."""
    P, half = d // 2, CHUNK // 2
    return torch.tensor([(c * half + j) if j < half else (P + c * half + j - half)
                         for c in range(d // CHUNK) for j in range(CHUNK)])


# --------------------------------------------------------------------------- kernel helpers


@triton.jit
def _count_le(Lv, x):
    """#{p : L_p <= x} (Lv: the [P] thresholds, loaded once per program)."""
    return tl.sum((Lv <= x).to(tl.int32), axis=0)


@triton.jit
def _ld(X, base, rows, rmask, c, D: tl.constexpr):
    """Rows x chunk c ([R, 16]) of a [.., T, D] tensor in chunked layout."""
    return tl.load(X + base + rows[:, None] * D + c * 16 + tl.arange(0, 16)[None, :], mask=rmask[:, None],
                   other=0.0)


@triton.jit
def _keep(x, c, a_eff):
    """Zero the columns of chunk c whose pair is below a_eff."""
    return tl.where((c * 8 + tl.arange(0, 16) % 8)[None, :] >= a_eff, x, 0.0).to(x.dtype)


@triton.jit
def _only(x, p):
    """Keep only the two columns of pair p (x is the chunk containing p)."""
    return tl.where(((p // 8) * 8 + tl.arange(0, 16) % 8)[None, :] == p, x, 0.0).to(x.dtype)


@triton.jit
def _pick(c, x0, x1, x2, x3):
    return tl.where(c == 0, x0, tl.where(c == 1, x1, tl.where(c == 2, x2, x3)))


@triton.jit
def _seg_bounds_keys(Lptr, i_hi, j_start, j_end, BN: tl.constexpr):
    """Key-tile starts where the segment changes (far -> near): tiles before b3 have a_hi >= 24, ..."""
    b3 = tl.maximum(i_hi - tl.load(Lptr + 23) + 1, j_start)
    b2 = tl.maximum(i_hi - tl.load(Lptr + 15) + 1, j_start)
    b1 = tl.maximum(i_hi - tl.load(Lptr + 7) + 1, j_start)
    b3 = tl.minimum((b3 + BN - 1) // BN * BN, j_end)
    b2 = tl.minimum((b2 + BN - 1) // BN * BN, j_end)
    b1 = tl.minimum((b1 + BN - 1) // BN * BN, j_end)
    return b3, b2, b1


@triton.jit
def _scores_qres(q0, q1, q2, q3, K, kv_base, offs_n, kmask, Lptr, a_lo, a_eff, delta,
                 C0: tl.constexpr, D: tl.constexpr, PREC: tl.constexpr):
    """Unscaled scores with resident query chunks and key chunks loaded per tile."""
    s = tl.zeros([q0.shape[0], offs_n.shape[0]], tl.float32)
    if C0 <= 0:
        s = tl.dot(_keep(q0, 0, a_eff), tl.trans(_ld(K, kv_base, offs_n, kmask, 0, D)), s, input_precision=PREC)
    if C0 <= 1:
        s = tl.dot(_keep(q1, 1, a_eff), tl.trans(_ld(K, kv_base, offs_n, kmask, 1, D)), s, input_precision=PREC)
    if C0 <= 2:
        s = tl.dot(_keep(q2, 2, a_eff), tl.trans(_ld(K, kv_base, offs_n, kmask, 2, D)), s, input_precision=PREC)
    s = tl.dot(_keep(q3, 3, a_eff), tl.trans(_ld(K, kv_base, offs_n, kmask, 3, D)), s, input_precision=PREC)
    for p in range(a_lo, a_eff):
        c = p // 8
        kc = _ld(K, kv_base, offs_n, kmask, c, D)
        sp = tl.dot(_only(_pick(c, q0, q1, q2, q3), p), tl.trans(kc), input_precision=PREC)
        s += tl.where(delta < tl.load(Lptr + p), sp, 0.0)
    return s


# --------------------------------------------------------------------------- forward


@triton.jit
def _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lptr, Lv, Lmax, qk_scale,
              C0: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    offs_n = j0 + tl.arange(0, BN)
    kmask = offs_n < T
    i_hi = i_lo + offs_m.shape[0] - 1
    dmax = i_hi - j0
    a_lo = _count_le(Lv, tl.maximum(i_lo - (j0 + BN - 1), 0))
    a_eff = tl.maximum(_count_le(Lv, dmax), 8 * C0)
    delta = offs_m[:, None] - offs_n[None, :]
    s = _scores_qres(q0, q1, q2, q3, K, kv_base, offs_n, kmask, Lptr, a_lo, a_eff, delta, C0, D, PREC) * qk_scale
    if (j0 + BN - 1 > i_lo) | (j0 + BN > T) | (dmax >= Lmax):
        s = tl.where((delta >= 0) & (delta < Lmax) & kmask[None, :], s, float("-inf"))
    v = tl.load(V + kv_base + offs_n[:, None] * D + tl.arange(0, D)[None, :], mask=kmask[:, None], other=0.0)
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp2(s - m_safe[:, None])
    alpha = tl.exp2(m_i - m_safe)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
    return m_new, l_i, acc


def _configs(kind: str):
    if os.environ.get("TRITON_INTERPRET") == "1":
        return [triton.Config({"BM": 32, "BN": 32}, num_warps=4, num_stages=1)]
    shapes = [(64, 64, 4, 3), (128, 64, 8, 3), (128, 64, 4, 3), (64, 32, 4, 3), (128, 32, 8, 3), (64, 64, 4, 2)]
    if kind == "dkdv":
        shapes = [(64, 64, 4, 3), (64, 128, 8, 3), (32, 64, 4, 3), (64, 128, 4, 3), (32, 128, 8, 3), (64, 64, 4, 2)]
    return [triton.Config({"BM": m, "BN": n}, num_warps=w, num_stages=st) for m, n, w, st in shapes]


@triton.autotune(configs=_configs("fwd"), key=["T", "H", "G"])
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE, Lptr, qk_scale, T, H, G,
                D: tl.constexpr, P: tl.constexpr, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    hk = (bh % H) // G
    q_base = bh.to(tl.int64) * T * D
    kv_base = (b * (H // G) + hk).to(tl.int64) * T * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    rmask = offs_m < T
    q0 = _ld(Q, q_base, offs_m, rmask, 0, D)
    q1 = _ld(Q, q_base, offs_m, rmask, 1, D)
    q2 = _ld(Q, q_base, offs_m, rmask, 2, D)
    q3 = _ld(Q, q_base, offs_m, rmask, 3, D)
    Lv = tl.load(Lptr + tl.arange(0, P))
    Lmax = tl.load(Lptr + P - 1)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    i_lo = pid_m * BM
    i_hi = i_lo + BM - 1
    j_start = (tl.maximum(i_lo - Lmax + 1, 0) // BN) * BN
    j_end = tl.minimum(i_hi, T - 1) + 1
    b3, b2, b1 = _seg_bounds_keys(Lptr, i_hi, j_start, j_end, BN)
    for j0 in range(j_start, b3, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lptr, Lv,
                                  Lmax, qk_scale, 3, D, BN, PREC)
    for j0 in range(b3, b2, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lptr, Lv,
                                  Lmax, qk_scale, 2, D, BN, PREC)
    for j0 in range(b2, b1, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lptr, Lv,
                                  Lmax, qk_scale, 1, D, BN, PREC)
    for j0 in range(b1, j_end, BN):
        m_i, l_i, acc = _fwd_tile(m_i, l_i, acc, q0, q1, q2, q3, K, V, kv_base, j0, offs_m, i_lo, T, Lptr, Lv,
                                  Lmax, qk_scale, 0, D, BN, PREC)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    offs_d = tl.arange(0, D)
    tl.store(O + q_base + offs_m[:, None] * D + offs_d[None, :], (acc / l_safe[:, None]).to(O.dtype.element_ty),
             mask=rmask[:, None])
    tl.store(LSE + bh.to(tl.int64) * T + offs_m, m_i + tl.log2(l_safe), mask=rmask)  # base-2 log-sum-exp


# --------------------------------------------------------------------------- backward: dq


@triton.jit
def _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m, rmask, i_lo, T,
             Lptr, Lv, Lmax, qk_scale, C0: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, PREC: tl.constexpr):
    offs_n = j0 + tl.arange(0, BN)
    kmask = offs_n < T
    BM: tl.constexpr = offs_m.shape[0]
    dmax = i_lo + BM - 1 - j0
    a_lo = _count_le(Lv, tl.maximum(i_lo - (j0 + BN - 1), 0))
    a_eff = tl.maximum(_count_le(Lv, dmax), 8 * C0)
    delta = offs_m[:, None] - offs_n[None, :]
    s = _scores_qres(q0, q1, q2, q3, K, kv_base, offs_n, kmask, Lptr, a_lo, a_eff, delta, C0, D, PREC)
    p = tl.exp2(s * qk_scale - lse[:, None])
    if (j0 + BN - 1 > i_lo) | (j0 + BN > T) | (dmax >= Lmax) | (i_lo + BM > T):
        p = tl.where((delta >= 0) & (delta < Lmax) & kmask[None, :] & rmask[:, None], p, 0.0)
    v = tl.load(V + kv_base + offs_n[:, None] * D + tl.arange(0, D)[None, :], mask=kmask[:, None], other=0.0)
    dp = tl.dot(do, tl.trans(v), input_precision=PREC)
    ds = p * (dp - Di[:, None])
    dsb = ds.to(do.dtype)
    # d s_ij / d q_i[f] = scale * active(pair(f), D_ij) * k_j[f]
    if C0 <= 0:
        dq0 = tl.dot(dsb, _keep(_ld(K, kv_base, offs_n, kmask, 0, D), 0, a_eff), dq0, input_precision=PREC)
    if C0 <= 1:
        dq1 = tl.dot(dsb, _keep(_ld(K, kv_base, offs_n, kmask, 1, D), 1, a_eff), dq1, input_precision=PREC)
    if C0 <= 2:
        dq2 = tl.dot(dsb, _keep(_ld(K, kv_base, offs_n, kmask, 2, D), 2, a_eff), dq2, input_precision=PREC)
    dq3 = tl.dot(dsb, _keep(_ld(K, kv_base, offs_n, kmask, 3, D), 3, a_eff), dq3, input_precision=PREC)
    for pp in range(a_lo, a_eff):
        c = pp // 8
        dsp = tl.where(delta < tl.load(Lptr + pp), ds, 0.0).to(do.dtype)
        g = tl.dot(dsp, _only(_ld(K, kv_base, offs_n, kmask, c, D), pp), input_precision=PREC)
        dq0 += tl.where(c == 0, g, 0.0)
        dq1 += tl.where(c == 1, g, 0.0)
        dq2 += tl.where(c == 2, g, 0.0)
        dq3 += tl.where(c == 3, g, 0.0)
    return dq0, dq1, dq2, dq3


@triton.autotune(configs=_configs("dq"), key=["T", "H", "G"])
@triton.jit
def _bwd_dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, Lptr, sm_scale, qk_scale, T, H, G,
                   D: tl.constexpr, P: tl.constexpr, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    hk = (bh % H) // G
    q_base = bh.to(tl.int64) * T * D
    kv_base = (b * (H // G) + hk).to(tl.int64) * T * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    rmask = offs_m < T
    offs_d = tl.arange(0, D)
    q0 = _ld(Q, q_base, offs_m, rmask, 0, D)
    q1 = _ld(Q, q_base, offs_m, rmask, 1, D)
    q2 = _ld(Q, q_base, offs_m, rmask, 2, D)
    q3 = _ld(Q, q_base, offs_m, rmask, 3, D)
    do = tl.load(DO + q_base + offs_m[:, None] * D + offs_d[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    Di = tl.load(DELTA + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    Lv = tl.load(Lptr + tl.arange(0, P))
    Lmax = tl.load(Lptr + P - 1)
    dq0 = tl.zeros([BM, 16], tl.float32)
    dq1 = tl.zeros([BM, 16], tl.float32)
    dq2 = tl.zeros([BM, 16], tl.float32)
    dq3 = tl.zeros([BM, 16], tl.float32)

    i_lo = pid_m * BM
    i_hi = i_lo + BM - 1
    j_start = (tl.maximum(i_lo - Lmax + 1, 0) // BN) * BN
    j_end = tl.minimum(i_hi, T - 1) + 1
    b3, b2, b1 = _seg_bounds_keys(Lptr, i_hi, j_start, j_end, BN)
    for j0 in range(j_start, b3, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lptr, Lv, Lmax, qk_scale, 3, D, BN, PREC)
    for j0 in range(b3, b2, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lptr, Lv, Lmax, qk_scale, 2, D, BN, PREC)
    for j0 in range(b2, b1, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lptr, Lv, Lmax, qk_scale, 1, D, BN, PREC)
    for j0 in range(b1, j_end, BN):
        dq0, dq1, dq2, dq3 = _dq_tile(dq0, dq1, dq2, dq3, q0, q1, q2, q3, do, lse, Di, K, V, kv_base, j0, offs_m,
                                      rmask, i_lo, T, Lptr, Lv, Lmax, qk_scale, 0, D, BN, PREC)
    cc = tl.arange(0, 16)[None, :]
    rows = DQ + q_base + offs_m[:, None] * D
    tl.store(rows + cc, (dq0 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(rows + 16 + cc, (dq1 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(rows + 32 + cc, (dq2 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(rows + 48 + cc, (dq3 * sm_scale).to(DQ.dtype.element_ty), mask=rmask[:, None])


# --------------------------------------------------------------------------- backward: dk, dv


@triton.jit
def _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, q_base, bh, i0, offs_n, kmask,
               j_lo, j_hi, T, Lptr, Lv, Lmax, qk_scale,
               C0: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr):
    offs_m = i0 + tl.arange(0, BM)
    rmask = offs_m < T
    do = tl.load(DO + q_base + offs_m[:, None] * D + tl.arange(0, D)[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    Di = tl.load(DELTA + bh.to(tl.int64) * T + offs_m, mask=rmask, other=0.0)
    dmax = i0 + BM - 1 - j_lo
    a_lo = _count_le(Lv, tl.maximum(i0 - j_hi, 0))
    a_eff = tl.maximum(_count_le(Lv, dmax), 8 * C0)
    delta = offs_m[:, None] - offs_n[None, :]
    # scores: query chunks loaded per tile, key chunks resident
    s = tl.zeros([BM, offs_n.shape[0]], tl.float32)
    if C0 <= 0:
        s = tl.dot(_keep(_ld(Q, q_base, offs_m, rmask, 0, D), 0, a_eff), tl.trans(k0), s, input_precision=PREC)
    if C0 <= 1:
        s = tl.dot(_keep(_ld(Q, q_base, offs_m, rmask, 1, D), 1, a_eff), tl.trans(k1), s, input_precision=PREC)
    if C0 <= 2:
        s = tl.dot(_keep(_ld(Q, q_base, offs_m, rmask, 2, D), 2, a_eff), tl.trans(k2), s, input_precision=PREC)
    s = tl.dot(_keep(_ld(Q, q_base, offs_m, rmask, 3, D), 3, a_eff), tl.trans(k3), s, input_precision=PREC)
    for pp in range(a_lo, a_eff):
        c = pp // 8
        qc = _only(_ld(Q, q_base, offs_m, rmask, c, D), pp)
        sp = tl.dot(qc, tl.trans(_pick(c, k0, k1, k2, k3)), input_precision=PREC)
        s += tl.where(delta < tl.load(Lptr + pp), sp, 0.0)
    p = tl.exp2(s * qk_scale - lse[:, None])
    if (j_hi > i0) | (j_lo + offs_n.shape[0] > T) | (dmax >= Lmax) | (i0 + BM > T):
        p = tl.where((delta >= 0) & (delta < Lmax) & kmask[None, :] & rmask[:, None], p, 0.0)
    dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv, input_precision=PREC)
    dp = tl.dot(do, tl.trans(v), input_precision=PREC)
    ds = p * (dp - Di[:, None])
    dst = tl.trans(ds.to(do.dtype))
    if C0 <= 0:
        dk0 = tl.dot(dst, _keep(_ld(Q, q_base, offs_m, rmask, 0, D), 0, a_eff), dk0, input_precision=PREC)
    if C0 <= 1:
        dk1 = tl.dot(dst, _keep(_ld(Q, q_base, offs_m, rmask, 1, D), 1, a_eff), dk1, input_precision=PREC)
    if C0 <= 2:
        dk2 = tl.dot(dst, _keep(_ld(Q, q_base, offs_m, rmask, 2, D), 2, a_eff), dk2, input_precision=PREC)
    dk3 = tl.dot(dst, _keep(_ld(Q, q_base, offs_m, rmask, 3, D), 3, a_eff), dk3, input_precision=PREC)
    for pp in range(a_lo, a_eff):
        c = pp // 8
        dspt = tl.trans(tl.where(delta < tl.load(Lptr + pp), ds, 0.0).to(do.dtype))
        g = tl.dot(dspt, _only(_ld(Q, q_base, offs_m, rmask, c, D), pp), input_precision=PREC)
        dk0 += tl.where(c == 0, g, 0.0)
        dk1 += tl.where(c == 1, g, 0.0)
        dk2 += tl.where(c == 2, g, 0.0)
        dk3 += tl.where(c == 3, g, 0.0)
    return dk0, dk1, dk2, dk3, dv


@triton.jit
def _seg_bounds_queries(Lptr, j_lo, i_start, i_end, BM: tl.constexpr):
    """Query-tile starts where the segment changes (near -> far) for a fixed key tile."""
    s1 = tl.maximum(j_lo + tl.load(Lptr + 7) - BM + 1, i_start)
    s2 = tl.maximum(j_lo + tl.load(Lptr + 15) - BM + 1, i_start)
    s3 = tl.maximum(j_lo + tl.load(Lptr + 23) - BM + 1, i_start)
    s1 = tl.minimum((s1 + BM - 1) // BM * BM, i_end)
    s2 = tl.minimum((s2 + BM - 1) // BM * BM, i_end)
    s3 = tl.minimum((s3 + BM - 1) // BM * BM, i_end)
    return s1, s2, s3


@triton.autotune(configs=_configs("dkdv"), key=["T", "H", "G"])
@triton.jit
def _bwd_dkdv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, Lptr, sm_scale, qk_scale, T, H, G,
                     D: tl.constexpr, P: tl.constexpr, PREC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)
    bhk = tl.program_id(1)          # b * H_kv + kv head
    Hkv = H // G
    b = bhk // Hkv
    hk = bhk % Hkv
    kv_base = bhk.to(tl.int64) * T * D
    offs_n = pid_n * BN + tl.arange(0, BN)
    kmask = offs_n < T
    offs_d = tl.arange(0, D)
    k0 = _ld(K, kv_base, offs_n, kmask, 0, D)
    k1 = _ld(K, kv_base, offs_n, kmask, 1, D)
    k2 = _ld(K, kv_base, offs_n, kmask, 2, D)
    k3 = _ld(K, kv_base, offs_n, kmask, 3, D)
    v = tl.load(V + kv_base + offs_n[:, None] * D + offs_d[None, :], mask=kmask[:, None], other=0.0)
    Lv = tl.load(Lptr + tl.arange(0, P))
    Lmax = tl.load(Lptr + P - 1)
    dk0 = tl.zeros([BN, 16], tl.float32)
    dk1 = tl.zeros([BN, 16], tl.float32)
    dk2 = tl.zeros([BN, 16], tl.float32)
    dk3 = tl.zeros([BN, 16], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)

    j_lo = pid_n * BN
    j_hi = j_lo + BN - 1
    i_start = (j_lo // BM) * BM
    i_end = tl.minimum(T, j_hi + Lmax)  # queries i with i - j < Lmax for some key j of the tile
    s1, s2, s3 = _seg_bounds_queries(Lptr, j_lo, i_start, i_end, BM)
    for g in range(0, G):
        bh = b * H + hk * G + g
        q_base = bh.to(tl.int64) * T * D
        for i0 in range(i_start, s1, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, q_base,
                                                bh, i0, offs_n, kmask, j_lo, j_hi, T, Lptr, Lv, Lmax, qk_scale,
                                                0, D, BM, PREC)
        for i0 in range(s1, s2, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, q_base,
                                                bh, i0, offs_n, kmask, j_lo, j_hi, T, Lptr, Lv, Lmax, qk_scale,
                                                1, D, BM, PREC)
        for i0 in range(s2, s3, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, q_base,
                                                bh, i0, offs_n, kmask, j_lo, j_hi, T, Lptr, Lv, Lmax, qk_scale,
                                                2, D, BM, PREC)
        for i0 in range(s3, i_end, BM):
            dk0, dk1, dk2, dk3, dv = _dkdv_tile(dk0, dk1, dk2, dk3, dv, k0, k1, k2, k3, v, Q, DO, LSE, DELTA, q_base,
                                                bh, i0, offs_n, kmask, j_lo, j_hi, T, Lptr, Lv, Lmax, qk_scale,
                                                3, D, BM, PREC)
    cc = tl.arange(0, 16)[None, :]
    rows = DK + kv_base + offs_n[:, None] * D
    tl.store(rows + cc, (dk0 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(rows + 16 + cc, (dk1 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(rows + 32 + cc, (dk2 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(rows + 48 + cc, (dk3 * sm_scale).to(DK.dtype.element_ty), mask=kmask[:, None])
    tl.store(DV + kv_base + offs_n[:, None] * D + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=kmask[:, None])


# --------------------------------------------------------------------------- launchers


def _prec(x: torch.Tensor) -> str:
    return "ieee" if x.dtype in (torch.float32, torch.float64) else "tf32"


def _check(q, k, v, L):
    B, H, T, D = q.shape
    if not (k.shape == v.shape and k.shape[0] == B and k.shape[2] == T and k.shape[3] == D):
        raise ValueError(f"k, v must be [B, H_kv, T, d] matching q {tuple(q.shape)}: "
                         f"got {tuple(k.shape)}, {tuple(v.shape)}")
    if H % k.shape[1]:
        raise ValueError("H must be a multiple of H_kv")
    if D != HEAD_DIM:
        raise ValueError(f"these kernels are written for head_dim {HEAD_DIM} (4 chunks of 16 features)")
    if not (L.dtype == torch.int32 and L.numel() == D // 2):
        raise ValueError("L must be the int32 thresholds of the d / 2 pairs (thresholds)")
    return B, H, T, D, H // k.shape[1]


def _contiguous_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)  # the kernels return contiguous tensors


_PERM: dict = {}


def _perm(device) -> tuple[torch.Tensor, torch.Tensor]:
    if device not in _PERM:
        perm = chunk_permutation().to(device)
        _PERM[device] = (perm, torch.argsort(perm))
    return _PERM[device]


@torch.library.custom_op("blagpt_hisa::triton_attn_fwd", mutates_args=())
def triton_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (o, lse2, q_chunked, k_chunked); the chunked copies are kept for the backward pass."""
    B, H, T, D, G = _check(q, k, v, L)
    perm, _ = _perm(q.device)
    qc, kc, v = q.index_select(-1, perm).contiguous(), k.index_select(-1, perm).contiguous(), v.contiguous()
    o = _contiguous_like(q)
    lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(T, meta["BM"]), B * H)
    _fwd_kernel[grid](qc, kc, v, o, lse, L, D ** -0.5 * LOG2E, T, H, G, D=D, P=D // 2, PREC=_prec(q))
    return o, lse, qc, kc


@triton_attn_fwd.register_fake
def _(q, k, v, L):
    B, H, T, D = q.shape
    return (_contiguous_like(q), torch.empty(B, H, T, dtype=torch.float32, device=q.device),
            _contiguous_like(q), _contiguous_like(k))


@torch.library.custom_op("blagpt_hisa::triton_attn_bwd", mutates_args=())
def triton_attn_bwd(do: torch.Tensor, qc: torch.Tensor, kc: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                    lse: torch.Tensor, dlse: torch.Tensor, L: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients w.r.t. the original (unpermuted) q, k and v."""
    v, o, do = v.contiguous(), o.contiguous(), do.contiguous()
    B, H, T, D, G = _check(qc, kc, v, L)
    _, inv = _perm(qc.device)
    # [B, H, T] = rowsum(dO * O) - dlse2 / ln2: d lse2 / d z_ij = p_ij / ln2 (z = natural-log logits)
    delta = (do.float() * o.float()).sum(-1) - dlse.float() / math.log(2.0)
    dqc, dkc, dv = _contiguous_like(qc), _contiguous_like(kc), _contiguous_like(v)
    scales = (D ** -0.5, D ** -0.5 * LOG2E)
    consts = dict(D=D, P=D // 2, PREC=_prec(qc))
    _bwd_dq_kernel[lambda meta: (triton.cdiv(T, meta["BM"]), B * H)](
        qc, kc, v, do, lse, delta, dqc, L, *scales, T, H, G, **consts)
    _bwd_dkdv_kernel[lambda meta: (triton.cdiv(T, meta["BN"]), B * (H // G))](
        qc, kc, v, do, lse, delta, dkc, dv, L, *scales, T, H, G, **consts)
    return dqc.index_select(-1, inv), dkc.index_select(-1, inv), dv


@triton_attn_bwd.register_fake
def _(do, qc, kc, v, o, lse, dlse, L):
    return _contiguous_like(qc), _contiguous_like(kc), _contiguous_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, L = inputs
    o, lse, qc, kc = output
    ctx.save_for_backward(qc, kc, v, o, lse, L)


def _backward(ctx, do, dlse, dqc_unused, dkc_unused):
    qc, kc, v, o, lse, L = ctx.saved_tensors
    if do is None:
        do = torch.zeros_like(o)
    if dlse is None:
        dlse = torch.zeros_like(lse)
    dq, dk, dv = triton_attn_bwd(do, qc, kc, v, o, lse, dlse, L)
    return dq, dk, dv, None


triton_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


def hisa_triton_exact(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """q [B, H, T, 64], k, v [B, H_kv, T, 64] (rotated) -> (o [B, H, T, 64], lse2 [B, H, T] fp32);
    L = thresholds(theta, max_angle_deg) (int32 [32], on q's device). Tile sizes are autotuned.

    Under autocast, q/k/v are cast to the autocast dtype, as autocast does for SDPA and for the matmuls of
    the PyTorch implementations (HF's RoPE returns fp32 q/k from the fp32 residual stream, v is bf16)."""
    dev = q.device.type
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        q, k, v = q.to(dt), k.to(dt), v.to(dt)
    elif not (q.dtype == k.dtype == v.dtype):
        raise TypeError(f"q/k/v dtypes differ: {q.dtype}, {k.dtype}, {v.dtype}")
    o, lse, _, _ = triton_attn_fwd(q, k, v, L)
    return o, lse
