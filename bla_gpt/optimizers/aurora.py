"""Aurora: A Leverage-Aware Spectral Optimizer (arXiv:2606.27715).

Faithful integration of the official reference implementation
github.com/tilde-research/aurora-release (src/polar.py, src/aurora.py).

`polar()` and `aurora()` are vendored byte-for-byte from that source so the
update rule reproduces upstream exactly. `Aurora` wraps them in a
`torch.optim.Optimizer` that mirrors this repo's Muon integration: 2D matrix
parameters are updated by Aurora, while 1D parameters and the embedding / head
matrices fall back to an internal AdamW (identical to `optimizers/muon.py`).
"""

import torch


@torch.no_grad()
def polar(G: torch.Tensor) -> torch.Tensor:
    """Polar factor via 12-step simple-quintic Newton-Schulz.

    For G with SVD G = U S V^T, returns U V^T (all non-zero singular values
    mapped to 1). Vendored from aurora-release/src/polar.py.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm <= 1 so the iteration converges to polar.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Simple-quintic coefficients: p(s) = a*s + b*s^3 + c*s^5, s=1 super-attracting.
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.no_grad()
def aurora(
    W,
    G,
    momentum,
    eta=0.05,
    weight_decay=0.025,
    mu=0.95,
    nesterov=True,
    pp_iterations=2,
    pp_beta=0.5,
    eps=1e-7,
    rms_match=False,
):
    """Aurora update rule. Vendored from aurora-release/src/aurora.py.

    Mutates ``W`` (weight) in place; ``G`` (gradient) and ``momentum`` are the
    caller-managed grad and momentum buffers.
    """
    if W.ndim != 2:
        raise ValueError(f"aurora expects 2D weight tensors, got shape {tuple(W.shape)}")
    if G.shape != W.shape:
        raise ValueError(f"G shape {tuple(G.shape)} must match W shape {tuple(W.shape)}")
    if momentum.shape != W.shape:
        raise ValueError(f"momentum shape {tuple(momentum.shape)} must match W shape {tuple(W.shape)}")
    if not (0.0 < mu < 1.0):
        raise ValueError(f"mu must be in (0, 1), got {mu}")
    if eta <= 0.0:
        raise ValueError(f"eta must be positive, got {eta}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")

    # SGD-momentum (Nesterov by default).
    momentum.lerp_(G, 1 - mu)
    # Clone when not using Nesterov to avoid scaling the momentum buffer in-place below.
    update = G.lerp_(momentum, mu) if nesterov else momentum.clone()
    # Aurora's leverage-uniform polar via diagonal preconditioning.
    m, n = update.size(-2), update.size(-1)
    if m == n:
        # Square: standard polar (no leverage freedom to exploit).
        update = polar(update)
    else:
        # For wide G, transpose to tall, apply, transpose back.
        # polar(G * D) = polar(D * G^T)^T
        transposed = m < n
        if transposed:
            update = update.mT
            m, n = n, m
        G32 = update.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32)
            if k < pp_iterations - 1:
                row_sq = U.to(torch.float32).pow(2).sum(dim=-1, keepdim=True).clamp_(min=eps * eps)
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U.mT if transposed else U
    if rms_match:
        # Kimi-style RMS matching (arXiv:2502.16982): scale the polar update so its
        # entry-wise RMS is ~0.2 for every matrix shape (matches AdamW's typical
        # update RMS, makes one global lr work across layers of different widths).
        update *= 0.2 * (max(G.size(-2), G.size(-1)) ** 0.5)
    else:
        # Spectral aspect-ratio scaling (Muon convention). Bit-identical to F99.
        update *= max(1, G.size(-2) / G.size(-1)) ** 0.5
    if not update.isfinite().all():
        raise RuntimeError(
            f"aurora produced non-finite update for parameter of shape {tuple(W.shape)}. "
            "Check for NaN/Inf in gradients or an ill-conditioned weight matrix."
        )
    # Decoupled weight decay then apply.
    W.mul_(1 - eta * weight_decay)
    W.add_(update, alpha=-eta)
    return W


class Aurora(torch.optim.Optimizer):
    """Leverage-aware spectral optimizer (arXiv:2606.27715).

    2D matrix parameters passed as ``aurora_params`` are updated by ``aurora()``;
    every parameter in ``adamw_params`` (1D params, embedding, head) uses an
    internal AdamW backup identical to this repo's Muon optimizer.

    Args:
        lr: learning rate (Aurora ``eta`` and the AdamW backup lr).
        aurora_params: >= 2D matrix params optimized by Aurora.
        adamw_params: params optimized by the AdamW backup.
        weight_decay: decoupled weight decay for the Aurora path (paper default 0.025).
        mu: SGD-momentum coefficient (paper default 0.95).
        nesterov: use Nesterov momentum (paper default True).
        pp_iterations: leverage-uniform refinement iterations (paper default 2).
        pp_beta: row-normalization damping exponent (paper default 0.5).
        eps: numerical floor for the row-norm preconditioner.
        betas: AdamW-backup betas.
        adamw_eps: AdamW-backup epsilon.
        adamw_wd: AdamW-backup decoupled weight decay.
    """

    def __init__(
        self,
        lr=0.05,
        aurora_params=None,
        adamw_params=None,
        weight_decay=0.025,
        mu=0.95,
        nesterov=True,
        pp_iterations=2,
        pp_beta=0.5,
        eps=1e-7,
        rms_match=False,
        betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.0,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            nesterov=nesterov,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
            eps=eps,
            rms_match=rms_match,
            adamw_betas=betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        aurora_params = list(aurora_params) if aurora_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        super().__init__(aurora_params + adamw_params, defaults)
        for p in aurora_params:
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_aurora"] = True
        for p in adamw_params:
            self.state[p]["use_aurora"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            ############################
            #          Aurora          #
            ############################
            for p in group["params"]:
                if not self.state[p].get("use_aurora", False):
                    continue
                g = p.grad
                if g is None:
                    continue
                # Operate on a 2D view so >2D matrices reduce to (rows, -1),
                # matching Muon; the view shares storage with p.data.
                w2d = p.data.view(p.size(0), -1)
                g2d = g.view(p.size(0), -1)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(w2d)
                aurora(
                    w2d,
                    g2d,
                    state["momentum_buffer"],
                    eta=lr,
                    weight_decay=group["weight_decay"],
                    mu=group["mu"],
                    nesterov=group["nesterov"],
                    pp_iterations=group["pp_iterations"],
                    pp_beta=group["pp_beta"],
                    eps=group["eps"],
                    rms_match=group["rms_match"],
                )

            ############################
            #       AdamW backup       #
            ############################
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]
            for p in group["params"]:
                if self.state[p].get("use_aurora", False):
                    continue
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
