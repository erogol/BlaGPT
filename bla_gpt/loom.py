"""F112: LoomFormer — Rewirable Circuits in Transformers with Loom Attention.

Xiao et al. (2026), arXiv:2611.30634, github.com/Caiyun-AI/LoomFormer.
Dual residual stream:
  - vector stream x: (B, T, D)
  - per-token matrix memory M: (B, T, d_k, d) with d = D // n_head
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def rms_norm_unit(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Parameter-free RMSNorm over the last axis with FP32 statistics."""
    f = x.float()
    return (f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


def m_norm(M: torch.Tensor, eps: float) -> torch.Tensor:
    """Whole-matrix RMSNorm over (-2, -1) = (d_k, d) with FP32 statistics."""
    f = M.float()
    return (f * torch.rsqrt(f.pow(2).mean(dim=(-2, -1), keepdim=True) + eps)).to(M.dtype)


def _gelu_approx(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    return x.unflatten(-1, (n_head, -1))


def _mark_loom_adamw(module: nn.Module, no_wd_names: Tuple[str, ...] = ()) -> None:
    """Tag all parameters in `module` for the Loom AdamW optimizer group."""
    for name, p in module.named_parameters():
        p._loom_adamw = True
        if name.endswith("bias") or name in no_wd_names or p.ndim < 2:
            p._no_weight_decay = True


class LoomRotaryEmbedding(nn.Module):
    """Half-split RoPE over `dim` coordinates (matches official LoomFormer)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"loom_rope_dim must be even, got {dim}"
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or self._cached_len < seq_len
            or self._cos_cached.device != device
        ):
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            angles = torch.outer(pos, self.inv_freq.to(device))  # (T, dim//2)
            self._cos_cached = angles.cos()[None, :, None, :]  # (1, T, 1, dim//2)
            self._sin_cached = angles.sin()[None, :, None, :]
            self._cached_len = seq_len
        return (
            self._cos_cached[:, :seq_len].to(dtype),
            self._sin_cached[:, :seq_len].to(dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n, dim)
        cos, sin = self._get_cos_sin(x.size(1), x.device, x.dtype)
        a, b = x.chunk(2, dim=-1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def strict_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Strictly causal (s < t) SDPA where position t = 0 outputs zero.

    Inputs:
        q, k: (B, T, n, d_qk)
        v:    (B, T, n, d_v) with d_v <= d_qk
    Returns:
        out:  (B, T, n, d_v)

    For t >= 1, attending to s < t is identical to standard inclusive causal
    attention (is_causal=True) on (q[:, 1:], k[:, :-1], v[:, :-1]). Padding v
    from d_v to d_qk along the last axis lets FlashAttention / cuDNN run at
    full speed without allocating an O(T^2) mask or hitting NaN on row t = 0.
    """
    b, t, n, d_v = v.shape
    if t <= 1:
        return torch.zeros_like(v)

    d_qk = q.size(-1)
    q_s = q[:, 1:].transpose(1, 2)   # (B, n, T-1, d_qk)
    k_s = k[:, :-1].transpose(1, 2)  # (B, n, T-1, d_qk)
    v_s = v[:, :-1].transpose(1, 2)  # (B, n, T-1, d_v)
    if d_qk > d_v:
        v_s = F.pad(v_s, (0, d_qk - d_v))

    y = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True, scale=scale)
    if d_qk > d_v:
        y = y[..., :d_v]
    y = y.transpose(1, 2)  # (B, T-1, n, d_v)
    return F.pad(y, (0, 0, 0, 0, 1, 0))


class LoomEmbedWrite(nn.Module):
    """Initializes M_0 (B, T, d_k, d) from token embeddings x_0 (Paper Eq. 10)."""

    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        n = config.n_head
        d = D // n
        d_k = config.loom_d_k
        d_r = config.loom_d_r
        self.n = n
        self.d = d
        self.write_eps = config.loom_write_eps
        self.init_std = config.loom_init_std
        self.write_gate_init = config.loom_write_gate_init

        self.W_e = nn.Linear(D, n * d, bias=False)
        self.W_w1 = nn.Linear(D, d_r, bias=False)
        self.W_w2 = nn.Linear(d_r, n * d_k, bias=True)
        self.W_wg = nn.Linear(D, n, bias=True)

        _mark_loom_adamw(self.W_w1)
        _mark_loom_adamw(self.W_w2)
        _mark_loom_adamw(self.W_wg)
        self.reset_loom_parameters()

    def reset_loom_parameters(self) -> None:
        nn.init.normal_(self.W_e.weight, 0.0, self.init_std)
        nn.init.normal_(self.W_w1.weight, 0.0, self.init_std)
        nn.init.normal_(self.W_w2.weight, 0.0, self.init_std)
        nn.init.zeros_(self.W_w2.bias)
        nn.init.zeros_(self.W_wg.weight)
        nn.init.constant_(self.W_wg.bias, _logit(self.write_gate_init))

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        o = rms_norm_unit(_split_heads(self.W_e(x_0), self.n), self.write_eps)
        km = rms_norm_unit(_split_heads(self.W_w2(_gelu_approx(self.W_w1(x_0))), self.n), self.write_eps)
        gate = torch.sigmoid(self.W_wg(x_0))[..., None]
        return torch.einsum("btnk,btnd->btkd", km * gate, o)


class LoomMHA(nn.Module):
    """Loom Multi-Head Attention sub-layer (Paper Section 3.2, Eqs. 5-8)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        D = config.n_embd
        n = config.n_head
        d = D // n
        d_k = config.loom_d_k
        d_r = config.loom_d_r
        r_dim = config.loom_rope_dim

        self.layer_idx = layer_idx
        self.n = n
        self.d = d
        self.d_k = d_k
        self.r_dim = r_dim
        self.use_m_shift = config.loom_m_shift
        self.rms_eps = config.loom_rms_eps
        self.key_eps = config.loom_read_key_eps
        self.key_scale = config.loom_read_key_scale
        self.write_eps = config.loom_write_eps
        self.static_init = config.loom_static_read_gate_init
        self.read_gate_init = config.loom_read_gate_init
        self.write_gate_init = config.loom_write_gate_init
        self.m_shift_init = config.loom_m_shift_init
        self.init_std = config.loom_init_std
        self.clone_vo_init = config.loom_clone_vo_init
        self.attn_scale = 1.0 / math.sqrt(d)
        self.write_M = layer_idx < config.n_layer - 1

        if self.use_m_shift:
            self.W_sg = nn.Linear(D, d_k, bias=True)
            _mark_loom_adamw(self.W_sg)
        else:
            self.W_sg = None

        self.W_q = nn.Linear(D, n * r_dim, bias=False)
        self.W_k = nn.Linear(D, n * r_dim, bias=False)
        self.rotary = LoomRotaryEmbedding(r_dim, theta=getattr(config, "rope_theta", 10000.0))

        self.W_r1 = nn.Linear(D, d_r, bias=False)
        _mark_loom_adamw(self.W_r1)
        for arm in ("q", "k", "v", "o"):
            w_r2 = nn.Linear(d_r, n * d_k, bias=True)
            w_rg = nn.Linear(D, n, bias=True)
            q_bar = nn.Parameter(torch.zeros(n, d_k))
            w_rg_bar = nn.Linear(D, n, bias=True)
            _mark_loom_adamw(w_r2)
            _mark_loom_adamw(w_rg)
            _mark_loom_adamw(w_rg_bar)
            q_bar._loom_adamw = True
            q_bar._no_weight_decay = True
            setattr(self, f"W_r2_{arm}", w_r2)
            setattr(self, f"W_rg_{arm}", w_rg)
            setattr(self, f"q_bar_{arm}", q_bar)
            setattr(self, f"W_rg_bar_{arm}", w_rg_bar)

        self.c_qk = nn.Parameter(torch.ones(1))
        self.c_vo = nn.Parameter(torch.ones(1))
        self.c_qk._loom_adamw = True
        self.c_qk._no_weight_decay = True
        self.c_vo._loom_adamw = True
        self.c_vo._no_weight_decay = True

        self.W_o = nn.Linear(n * d, D, bias=False)

        if self.write_M:
            self.W_w1 = nn.Linear(D, d_r, bias=False)
            self.W_w2 = nn.Linear(d_r, n * d_k, bias=True)
            self.W_wg = nn.Linear(D, n, bias=True)
            _mark_loom_adamw(self.W_w1)
            _mark_loom_adamw(self.W_w2)
            _mark_loom_adamw(self.W_wg)

        self.reset_loom_parameters()

    def reset_loom_parameters(self) -> None:
        if self.W_sg is not None:
            nn.init.zeros_(self.W_sg.weight)
            nn.init.constant_(self.W_sg.bias, _logit(self.m_shift_init))

        nn.init.normal_(self.W_q.weight, 0.0, self.init_std)
        nn.init.normal_(self.W_k.weight, 0.0, self.init_std)
        nn.init.normal_(self.W_r1.weight, 0.0, self.init_std)
        nn.init.normal_(self.W_o.weight, 0.0, self.init_std)

        read_bias = _logit(self.read_gate_init)
        stat_bias = math.log(self.static_init / (1.1 - self.static_init))
        for arm in ("q", "k", "o", "v"):
            w_r2 = getattr(self, f"W_r2_{arm}")
            w_rg = getattr(self, f"W_rg_{arm}")
            q_bar = getattr(self, f"q_bar_{arm}")
            w_rg_bar = getattr(self, f"W_rg_bar_{arm}")

            nn.init.normal_(w_r2.weight, 0.0, self.init_std)
            nn.init.zeros_(w_r2.bias)
            nn.init.zeros_(w_rg.weight)
            nn.init.constant_(w_rg.bias, read_bias)
            nn.init.zeros_(w_rg_bar.weight)
            nn.init.constant_(w_rg_bar.bias, stat_bias)
            if arm == "v":
                nn.init.normal_(q_bar, 0.0, self.d_k ** -0.5)
            else:
                nn.init.zeros_(q_bar)

        if self.clone_vo_init:
            with torch.no_grad():
                self.W_r2_v.weight.copy_(self.W_r2_o.weight)

        nn.init.ones_(self.c_qk)
        nn.init.ones_(self.c_vo)

        if self.write_M:
            nn.init.normal_(self.W_w1.weight, 0.0, self.init_std)
            nn.init.normal_(self.W_w2.weight, 0.0, self.init_std)
            nn.init.zeros_(self.W_w2.bias)
            nn.init.zeros_(self.W_wg.weight)
            nn.init.constant_(self.W_wg.bias, _logit(self.write_gate_init))

    def _m_shift_and_norm(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        if self.W_sg is not None:
            z = torch.sigmoid(self.W_sg(x).float()).to(M.dtype)[..., None]  # (B, T, d_k, 1)
            M_prev = torch.cat((M[:, :1], M[:, :-1]), dim=1)
            mixed = (1.0 - z) * M + z * M_prev
            # Position 0 has no previous token; keep M[:, :1] untouched so W_sg gets no t=0 grad.
            M_s = torch.cat((M[:, :1], mixed[:, 1:]), dim=1)
        else:
            M_s = M
        return m_norm(M_s, self.rms_eps)

    def _m_read(
        self,
        M_tilde: torch.Tensor,
        arm: str,
        x: torch.Tensor,
        xr: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        r = rms_norm_unit(_split_heads(getattr(self, f"W_r2_{arm}")(xr), self.n), self.key_eps)
        g = (self.key_scale * torch.sigmoid(getattr(self, f"W_rg_{arm}")(x)))[..., None]
        g_bar = ((1.1 / self.static_init) * torch.sigmoid(getattr(self, f"W_rg_bar_{arm}")(x)))[..., None]
        q_bar = getattr(self, f"q_bar_{arm}").to(M_tilde.dtype)
        qm = g * r + g_bar * q_bar
        u = torch.einsum("btnk,btkd->btnd", qm, M_tilde)
        return (u.float() * scale.float()).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        M: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        M_tilde = self._m_shift_and_norm(x, M)
        xr = _gelu_approx(self.W_r1(x))

        q_tilde = self._m_read(M_tilde, "q", x, xr, self.c_qk)
        k_tilde = self._m_read(M_tilde, "k", x, xr, self.c_qk)
        v = self._m_read(M_tilde, "v", x, xr, self.c_vo)
        o_m = self._m_read(M_tilde, "o", x, xr, self.c_vo)

        qr = self.rotary(_split_heads(self.W_q(x), self.n))
        kr = self.rotary(_split_heads(self.W_k(x), self.n))
        q = torch.cat((q_tilde, qr), dim=-1)
        k = torch.cat((k_tilde, kr), dim=-1)

        o = strict_causal_attention(q, k, v, scale=self.attn_scale) + o_m
        dx = self.W_o(o.flatten(-2))

        if not self.write_M:
            return dx, None
        content = torch.sigmoid(self.W_wg(x))[..., None] * rms_norm_unit(o, self.write_eps)
        km = rms_norm_unit(_split_heads(self.W_w2(_gelu_approx(self.W_w1(x))), self.n), self.write_eps)
        return dx, (km, content)


class LoomBlock(nn.Module):
    """Single LoomFormer block: (x, M) <- (x, M) + LoomMHA + (FFN_x, FFN_M)."""

    def __init__(self, config, layer_idx: int, ln_1: nn.Module, ln_2: nn.Module, ffn_x: nn.Module, ffn_m: Optional[nn.Module]):
        super().__init__()
        D = config.n_embd
        n = config.n_head
        d_k = config.loom_d_k
        d_r = config.loom_d_r
        self.layer_idx = layer_idx
        self.n = n
        self.write_eps = config.loom_write_eps
        self.init_std = config.loom_init_std
        self.write_gate_init = config.loom_write_gate_init
        self.write_M = layer_idx < config.n_layer - 1

        self.ln_1 = ln_1
        self.attn = LoomMHA(config, layer_idx)
        self.ln_2 = ln_2
        self.mlp = ffn_x

        if self.write_M:
            assert ffn_m is not None
            self.ffn_m = ffn_m
            self.ffn_W_w1 = nn.Linear(D, d_r, bias=False)
            self.ffn_W_w2 = nn.Linear(d_r, n * d_k, bias=True)
            self.ffn_W_wg = nn.Linear(D, n, bias=True)
            _mark_loom_adamw(self.ffn_W_w1)
            _mark_loom_adamw(self.ffn_W_w2)
            _mark_loom_adamw(self.ffn_W_wg)
        else:
            self.ffn_m = None

        self.reset_loom_parameters()

    def reset_loom_parameters(self) -> None:
        self.attn.reset_loom_parameters()
        if self.write_M:
            nn.init.normal_(self.ffn_W_w1.weight, 0.0, self.init_std)
            nn.init.normal_(self.ffn_W_w2.weight, 0.0, self.init_std)
            nn.init.zeros_(self.ffn_W_w2.bias)
            nn.init.zeros_(self.ffn_W_wg.weight)
            nn.init.constant_(self.ffn_W_wg.bias, _logit(self.write_gate_init))
            # Ensure FFN_M output projection is non-zero at step 0 so FFN_M
            # and its write ports receive non-zero gradients immediately.
            if hasattr(self.ffn_m, "c_proj"):
                proj = self.ffn_m.c_proj
                if isinstance(proj, nn.Parameter):
                    nn.init.normal_(proj, 0.0, self.init_std)
                elif isinstance(proj, nn.Linear):
                    nn.init.normal_(proj.weight, 0.0, self.init_std)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor, M: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        dx, attn_write = self.attn(self.ln_1(x), M)
        x = x + dx
        h = self.ln_2(x)
        df_x = self.mlp(h, **kwargs)
        if attn_write is not None:
            km_a, content_a = attn_write
            dM = torch.einsum("btnk,btnd->btkd", km_a, content_a)
            mem = _split_heads(self.ffn_m(h), self.n)
            content_f = torch.sigmoid(self.ffn_W_wg(h))[..., None] * mem
            km_f = rms_norm_unit(_split_heads(self.ffn_W_w2(_gelu_approx(self.ffn_W_w1(h))), self.n), self.write_eps)
            dM = dM + torch.einsum("btnk,btnd->btkd", km_f, content_f)
            M = M + dM
        return x + df_x, M


def resolve_loom_ffn_dims(config) -> Tuple[int, int, int]:
    """Return (ffn_x_dim, ffn_m_dim, last_ffn_x_dim) for LoomFormer blocks.

    When explicit dims are non-zero in config, use them directly.
    Otherwise split total FFN width according to `loom_ffn_m_ratio` (default 1/3,
    giving a 2:1 FFN_x : FFN_M ratio) snapped to multiples of 64.
    """
    if config.loom_ffn_x_dim > 0 and config.loom_ffn_m_dim > 0:
        last_dim = config.loom_last_ffn_x_dim if config.loom_last_ffn_x_dim > 0 else (
            config.loom_ffn_x_dim + config.loom_ffn_m_dim
        )
        return int(config.loom_ffn_x_dim), int(config.loom_ffn_m_dim), int(last_dim)

    base_ff = getattr(config, "mlp_expand", 4) * config.n_embd
    ratio = float(getattr(config, "loom_ffn_m_ratio", 1.0 / 3.0))
    snap = 64 if base_ff >= 256 else 8
    ffn_m_dim = max(snap, int(round(base_ff * ratio / snap)) * snap)
    ffn_x_dim = max(snap, base_ff - ffn_m_dim)
    last_ffn_x_dim = int(config.loom_last_ffn_x_dim) if config.loom_last_ffn_x_dim > 0 else base_ff
    return ffn_x_dim, ffn_m_dim, last_ffn_x_dim
