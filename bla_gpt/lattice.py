"""F107: Lattice of Latents (Duggal et al. 2026, arXiv:2611.21608).

One horizontal latent r_t in R^{d_r} per Lattice layer, updated along time by
an input-dependent affine recurrence (Sec. 3.2.2) and read out as
r_hat = Norm(r_t). r_hat conditions the layer through zero-initialized FiLM on
q, k, v (Sec. 3.2.1) and a zero-initialized shift of the MLP pre-activation.
Input injection (App. A.2.2) feeds the normed token embedding into Delta and c.
At init every injection map is zero, so the model equals the base model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(x, weight=None, eps=1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def lattice_scan(g, b, r0):
    """Inclusive affine scan r_t = g_t * r_{t-1} + b_t with r_{-1} = r0.

    g, b: (B, T, D); r0: (D,). Hillis-Steele log-step scan in fp32. It only
    multiplies gates in [0, 1], so it cannot overflow (no log-cumsum).
    """
    g = g.float()
    b = b.float()
    T = g.size(1)
    k = 1
    while k < T:
        g_prev = F.pad(g[:, :-k], (0, 0, k, 0), value=1.0)
        b_prev = F.pad(b[:, :-k], (0, 0, k, 0), value=0.0)
        b = g * b_prev + b
        g = g * g_prev
        k *= 2
    return g * r0.float() + b


def lattice_scan_ref(g, b, r0):
    """Sequential reference loop for lattice_scan (tests only)."""
    r = r0.float().expand(g.size(0), -1)
    out = []
    for t in range(g.size(1)):
        r = g[:, t].float() * r + b[:, t].float()
        out.append(r)
    return torch.stack(out, dim=1)


class LatticeLayer(nn.Module):
    """Horizontal latent update, readout and modulation heads for one layer."""

    def __init__(self, n_embd, d_r, q_dim, kv_dim, mlp_dim, inj=True,
                 film_after_qknorm=False, vstream=False, scan_impl="hs"):
        super().__init__()
        self.d_r = d_r
        # F107c: "hs" = Hillis-Steele (default); "triton" = sequential Triton
        # custom op (CUDA only, no guard on T, no switch recompile).
        if scan_impl not in ("hs", "triton"):
            raise ValueError(f"unknown lattice_scan_impl {scan_impl!r}")
        self.scan_impl = scan_impl
        self.inj = inj
        # F107b fix 1: FiLM q/k after the per-head QK RMSNorm, before RoPE.
        self.film_after_qknorm = film_after_qknorm
        # F107b fix 3: depth-wise readout stream (Sec. 3.2.2), hidden width d_r.
        self.vstream = vstream
        if vstream:
            self.vs_norm_r_w = nn.Parameter(torch.ones(d_r))
            self.vs_norm_s_w = nn.Parameter(torch.ones(d_r))
            self.vs_norm_out_w = nn.Parameter(torch.ones(d_r))
            self.vs_down = nn.Linear(2 * d_r, d_r, bias=False)
            self.vs_up = nn.Linear(d_r, d_r, bias=False)
        self.norm_in_w = nn.Parameter(torch.ones(n_embd))
        self.norm_out_w = nn.Parameter(torch.ones(d_r))
        self.w_delta = nn.Linear(n_embd, d_r, bias=True)
        self.w_r = nn.Linear(n_embd, d_r, bias=True)
        self.a_log = nn.Parameter(torch.zeros(d_r))
        self.r0 = nn.Parameter(torch.zeros(d_r))
        if inj:
            self.w_e_delta = nn.Linear(n_embd, d_r, bias=False)
            self.w_e_c = nn.Linear(n_embd, d_r, bias=False)
        # FiLM heads: one linear per target, output = [gamma | beta].
        self.film_q = nn.Linear(d_r, 2 * q_dim, bias=False)
        self.film_k = nn.Linear(d_r, 2 * kv_dim, bias=False)
        self.film_v = nn.Linear(d_r, 2 * kv_dim, bias=False)
        self.mlp_beta = nn.Linear(d_r, mlp_dim, bias=False)
        self.reset_lattice_parameters()

    @torch.no_grad()
    def reset_lattice_parameters(self):
        """Paper init. GPT calls this again after its global _init_weights."""
        nn.init.normal_(self.w_delta.weight, std=0.02)
        nn.init.normal_(self.w_r.weight, std=0.02)
        nn.init.zeros_(self.w_r.bias)
        # Mamba-style: A in [1, 16], dt = softplus(b_delta) log-uniform [1e-3, 1e-1].
        self.a_log.copy_(torch.log(torch.empty(self.d_r).uniform_(1.0, 16.0)))
        dt = torch.exp(torch.empty(self.d_r).uniform_(math.log(1e-3), math.log(1e-1)))
        self.w_delta.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.r0.zero_()
        self.norm_in_w.fill_(1.0)
        self.norm_out_w.fill_(1.0)
        zero = [self.film_q, self.film_k, self.film_v, self.mlp_beta]
        if self.inj:
            zero += [self.w_e_delta, self.w_e_c]
        if self.vstream:
            nn.init.normal_(self.vs_down.weight, std=0.02)
            for w in (self.vs_norm_r_w, self.vs_norm_s_w, self.vs_norm_out_w):
                w.fill_(1.0)
            zero.append(self.vs_up)
        for lin in zero:
            nn.init.zeros_(lin.weight)
        self.a_log._no_weight_decay = True
        self.w_delta.bias._no_weight_decay = True

    def update(self, x, e=None):
        """x: (B, T, C) block input; e: normed token embedding or None.

        Returns the fp32 readout r_hat (B, T, d_r).
        """
        n = _rms(x, self.norm_in_w)
        pre_d = self.w_delta(n)
        pre_c = self.w_r(n)
        if self.inj and e is not None:
            pre_d = pre_d + self.w_e_delta(e)
            pre_c = pre_c + self.w_e_c(e)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.softplus(pre_d.float())
            g = torch.exp(-torch.exp(self.a_log.float()) * delta)
            c = F.silu(pre_c.float())
            if self.scan_impl == "triton":
                from lattice_triton import lattice_scan_triton
                r = lattice_scan_triton(g, (1.0 - g) * c, self.r0)
            else:
                r = lattice_scan(g, (1.0 - g) * c, self.r0)
            return _rms(r, self.norm_out_w.float())

    def step(self, holder, key, x):
        """Run the update for this layer and store its readout in holder[key].

        With vstream, holder["s"] carries the depth-wise state s^{l-1}
        (None = zero seed at the first Lattice layer) and the stored readout
        is the enriched r_tilde = r_hat + Norm(s^l). The scan state stays raw r.
        """
        r_hat = self.update(x, holder.get("e"))
        if self.vstream:
            r_hat, holder["s"] = self.vstream_step(r_hat, holder.get("s"))
        holder[key] = r_hat

    def vstream_step(self, r_hat, s_prev):
        """s^l = s^{l-1} + W_up SiLU(W_down [Norm(r_hat), Norm(s^{l-1})]), fp32."""
        with torch.autocast(device_type=r_hat.device.type, enabled=False):
            r_hat = r_hat.float()
            if s_prev is None:
                s_prev = torch.zeros_like(r_hat)
            z = torch.cat([_rms(r_hat, self.vs_norm_r_w.float()),
                           _rms(s_prev, self.vs_norm_s_w.float())], dim=-1)
            h = F.silu(F.linear(z, self.vs_down.weight.float()))
            s = s_prev + F.linear(h, self.vs_up.weight.float())
            return r_hat + _rms(s, self.vs_norm_out_w.float()), s

    def film(self, which, x, r_hat):
        """(W x) * (1 + tanh(gamma(r_hat))) + beta(r_hat), in fp32.

        x: (B, T, H, hd) projected q, k or v.
        """
        lin = getattr(self, "film_" + which)
        B, T = x.shape[:2]
        with torch.autocast(device_type=x.device.type, enabled=False):
            gamma, beta = F.linear(r_hat, lin.weight.float()).chunk(2, dim=-1)
            y = x.float().reshape(B, T, -1) * (1.0 + torch.tanh(gamma)) + beta
        return y.view(x.shape).to(x.dtype)

    def mlp_shift(self, h, r_hat):
        """Add beta(r_hat) to the MLP pre-activation h (B, T, ff)."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            beta = F.linear(r_hat, self.mlp_beta.weight.float())
        return h + beta.to(h.dtype)


def attach_lattice(attn, mlp, mod, holder, key):
    """Give one block's attention and MLP access to its LatticeLayer.

    object.__setattr__ keeps the module out of the block's own registry, so
    each parameter has one name (under GPT.lattice) in state_dict and DDP.
    """
    for m in (attn, mlp):
        object.__setattr__(m, "_lattice_mod", mod)
        object.__setattr__(m, "_lattice_holder", holder)
        object.__setattr__(m, "_lattice_key", key)
