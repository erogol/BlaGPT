"""Aurora: A Leverage-Aware Spectral Optimizer (arXiv:2606.27715).

Faithful integration of the official reference implementation
github.com/tilde-research/aurora-release (src/polar.py, src/aurora.py).

`polar()` and `aurora()` are vendored byte-for-byte from that source so the
update rule reproduces upstream exactly. `Aurora` wraps them in a
`torch.optim.Optimizer` that mirrors this repo's Muon integration: 2D matrix
parameters are updated by Aurora, while 1D parameters and the embedding / head
matrices fall back to an internal AdamW (identical to `optimizers/muon.py`).

Optional experimental extension (E4a, off by default): a second "slow"
momentum rail (`aurora_twin_rail`) inspired by ANVIL2's twin-rail velocity.
When enabled, a slow EMA of the gradient is blended into the fast-rail
momentum after `rail_engage_step` steps. When disabled, behavior is
byte-identical to the vendored implementation above.
"""

import math

import torch


@torch.no_grad()
def top_singular_pair(M: torch.Tensor, iters: int = 10, eps: float = 1e-12):
    """Leading singular pair (u1, v1) of a 2D matrix by rank-one power iteration.

    F104 SAMuon-lite (arXiv 2608.25990, Algorithm 1 line 12). Stateless and
    deterministic: starts from the largest-norm row of ``M`` (that row has a
    non-zero overlap sigma_1 * u1[i] with v1). Runs in float32.
    """
    A = M.float()
    v = A[A.norm(dim=1).argmax()].clone()
    v = v / v.norm().clamp_min(eps)
    for _ in range(iters):
        u = A @ v
        u = u / u.norm().clamp_min(eps)
        v = A.mT @ u
        v = v / v.norm().clamp_min(eps)
    u = A @ v
    u = u / u.norm().clamp_min(eps)
    return u, v


def samuon_gamma_t(gamma: float, step: int, warmup_steps: int) -> float:
    """Cosine-warmed tail boost gamma_t = 1 + (gamma - 1) * w_t (paper G.3)."""
    w = 0.5 * (1.0 - math.cos(math.pi * min(step / max(warmup_steps, 1), 1.0)))
    return 1.0 + (gamma - 1.0) * w


@torch.no_grad()
def samuon_lite_reallocate(O: torch.Tensor, u: torch.Tensor, v: torch.Tensor, gamma_t: float):
    """Head-anchored two-level allocation on a whitened update ``O``.

    Returns gamma_t * O - (gamma_t - 1) * c * u v^T with c = u^T O v: the
    component of ``O`` on the momentum head direction u v^T keeps its base
    value, and everything orthogonal to it is scaled by gamma_t. Under exact
    whitening (O = U V^T) c = 1, so this equals paper Eq. 8 with k = 1.
    """
    O32 = O.float()
    c = u @ O32 @ v
    out = gamma_t * O32 - (gamma_t - 1.0) * c * torch.outer(u, v)
    return out.to(O.dtype)


@torch.no_grad()
def normuon_row_normalize(
    update: torch.Tensor, second_moment: torch.Tensor, beta2: float = 0.95, eps: float = 1e-10
) -> torch.Tensor:
    """F105 NorMuon neuron-wise normalization (arXiv 2510.05491, Algorithm 1 lines 7-9).

    Applied after orthogonalization. ``second_moment`` (shape ``(rows, 1)``,
    float32, caller-managed) is an EMA over steps of each row's mean squared
    entry; each row (output neuron) is divided by ``sqrt(v) + eps``. As in the
    reference code (github.com/zichongli5/NorMuon, normuon.py) the result is
    rescaled so its Frobenius norm equals the pre-normalization norm, which
    keeps the base optimizer's step size. Returns float32 (both reference
    implementations cast the polar output to float32 before this step).
    """
    O = update.float()
    norm_before = O.norm()
    second_moment.lerp_(O.pow(2).mean(dim=-1, keepdim=True), 1 - beta2)
    O = O / (second_moment.sqrt() + eps)
    return O * (norm_before / (O.norm() + eps))


@torch.no_grad()
def cautious_decay_(W: torch.Tensor, update: torch.Tensor, decay: float) -> torch.Tensor:
    """F105 cautious weight decay (arXiv 2510.12402, Algorithm 1), in place.

    ``W <- W - decay * I(update * W >= 0) * W`` entrywise, where ``update`` is
    the step that gets subtracted (``W <- W - eta * update``) and
    ``decay = eta * lambda``. Entries where the update would grow ``|W|``
    are not decayed.
    """
    mask = (update.float() * W.float()) >= 0
    W.addcmul_(W, mask.to(W.dtype), value=-decay)
    return W


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
    twin_rail=False,
    momentum_slow=None,
    slow_beta=0.98,
    step=0,
    rail_engage_step=514,
    rail_fast_weight=0.4385,
    samuon_lite=False,
    samuon_gamma=10.0,
    samuon_warmup_steps=1530,
    samuon_power_iters=10,
    cautious_wd=False,
    normuon=False,
    normuon_v=None,
    normuon_beta2=0.95,
    normuon_eps=1e-10,
):
    """Aurora update rule. Vendored from aurora-release/src/aurora.py.

    Mutates ``W`` (weight) in place; ``G`` (gradient) and ``momentum`` are the
    caller-managed grad and momentum buffers.

    Optional twin-rail mode (E4a, off by default when ``twin_rail=False``):
    when enabled, ``momentum_slow`` is an additional caller-managed EMA buffer
    (beta=``slow_beta``) that gets blended into the fast rail with weight
    ``rail_fast_weight`` once ``step >= rail_engage_step``.

    Optional SAMuon-lite mode (F104, arXiv 2608.25990, off by default): the
    leading singular pair (u1, v1) of the pre-polar update matrix is found by
    power iteration; after the leverage-aware polar, the update component on
    u1 v1^T keeps its base scale and the rest is scaled by the cosine-warmed
    tail boost gamma_t (``samuon_gamma`` after ``samuon_warmup_steps``).

    Optional F105 modes (off by default): ``normuon`` applies NorMuon's
    per-row second-moment normalization (``normuon_v`` is the caller-managed
    ``(rows, 1)`` EMA buffer) after the leverage-aware polar and before the
    aspect-ratio scaling; ``cautious_wd`` replaces the decoupled decay by
    cautious weight decay masked with the final update.
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

    # SGD-momentum (Nesterov by default). Fast rail, always updated.
    momentum.lerp_(G, 1 - mu)
    if twin_rail:
        # Slow rail: a second, slower-moving EMA of the gradient (ANVIL2-style
        # twin-rail velocity). Blended in only after the engage step.
        if momentum_slow is None:
            raise ValueError("twin_rail=True requires a momentum_slow buffer")
        momentum_slow.lerp_(G, 1 - slow_beta)
        if step < rail_engage_step:
            blended = momentum
        else:
            blended = rail_fast_weight * momentum + (1 - rail_fast_weight) * momentum_slow
        # Clone when not using Nesterov to avoid scaling the momentum buffer in-place below.
        update = G.lerp_(blended, mu) if nesterov else blended.clone()
    else:
        # Clone when not using Nesterov to avoid scaling the momentum buffer in-place below.
        update = G.lerp_(momentum, mu) if nesterov else momentum.clone()
    if samuon_lite:
        # F104: head of the matrix that gets whitened (paper: PowerIter(M)).
        head_u, head_v = top_singular_pair(update, iters=samuon_power_iters)
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
    if samuon_lite:
        gamma_t = samuon_gamma_t(samuon_gamma, step, samuon_warmup_steps)
        update = samuon_lite_reallocate(update, head_u, head_v, gamma_t)
    if normuon:
        if normuon_v is None:
            raise ValueError("normuon=True requires a normuon_v buffer")
        update = normuon_row_normalize(update, normuon_v, beta2=normuon_beta2, eps=normuon_eps)
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
    if cautious_wd:
        cautious_decay_(W, update, eta * weight_decay)
    else:
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
        sgd_m_params: params optimized by plain SGD with momentum (F102, arXiv 2610.38277).
        sgd_m_lr: base lr for the SGD-M group (scaled by the train scheduler).
        sgd_m_momentum: momentum coefficient for the SGD-M group (paper Figure 4: 0.99).
        weight_decay: decoupled weight decay for the Aurora path (paper default 0.025).
        mu: SGD-momentum coefficient (paper default 0.95).
        nesterov: use Nesterov momentum (paper default True).
        pp_iterations: leverage-uniform refinement iterations (paper default 2).
        pp_beta: row-normalization damping exponent (paper default 0.5).
        eps: numerical floor for the row-norm preconditioner.
        rms_match: use Kimi-style RMS matching instead of aspect-ratio scaling.
        betas: AdamW-backup betas.
        adamw_eps: AdamW-backup epsilon.
        adamw_wd: AdamW-backup decoupled weight decay.
        aurora_twin_rail: enable the E4a slow-momentum-rail extension (default off).
        rail_slow_beta: slow-rail EMA coefficient, used only if aurora_twin_rail.
        rail_engage_step: step at which the slow rail starts blending in.
        rail_fast_weight: fast-rail weight in the blend after rail_engage_step.
        samuon_lite: enable F104 SAMuon-lite head-anchored allocation (default off).
        samuon_gamma: tail boost gamma (paper Table 3, bs 1024 at 124M: 10).
        samuon_warmup_steps: cosine warmup of gamma_t (paper: 30% of training).
        samuon_power_iters: rank-one power iterations (paper G.6, 124M: 10).
    """

    def __init__(
        self,
        lr=0.05,
        aurora_params=None,
        adamw_params=None,
        sgd_m_params=None,
        sgd_m_lr=0.01,
        sgd_m_momentum=0.99,
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
        aurora_twin_rail=False,
        rail_slow_beta=0.98,
        rail_engage_step=514,
        rail_fast_weight=0.4385,
        samuon_lite=False,
        samuon_gamma=10.0,
        samuon_warmup_steps=1530,
        samuon_power_iters=10,
        cautious_wd=False,
        normuon=False,
        normuon_beta2=0.95,
        normuon_eps=1e-10,
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
            aurora_twin_rail=aurora_twin_rail,
            rail_slow_beta=rail_slow_beta,
            rail_engage_step=rail_engage_step,
            rail_fast_weight=rail_fast_weight,
            samuon_lite=samuon_lite,
            samuon_gamma=samuon_gamma,
            samuon_warmup_steps=samuon_warmup_steps,
            samuon_power_iters=samuon_power_iters,
            cautious_wd=cautious_wd,
            normuon=normuon,
            normuon_beta2=normuon_beta2,
            normuon_eps=normuon_eps,
        )
        aurora_params = list(aurora_params) if aurora_params is not None else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        sgd_m_params = list(sgd_m_params) if sgd_m_params is not None else []
        if sgd_m_params:
            param_groups = [
                dict(params=aurora_params + adamw_params, **defaults),
                dict(
                    params=sgd_m_params,
                    lr=sgd_m_lr,
                    momentum=sgd_m_momentum,
                    use_sgd_m=True,
                ),
            ]
            super().__init__(param_groups, defaults)
        else:
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
            #   SGD-M group (F102)    #
            ############################
            if group.get("use_sgd_m", False):
                momentum = group["momentum"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "sgd_momentum_buffer" not in state:
                        state["sgd_momentum_buffer"] = torch.zeros_like(g)
                    buf = state["sgd_momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    p.data.add_(buf, alpha=-lr)
                continue

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
                if group["aurora_twin_rail"]:
                    if "momentum_buffer_slow" not in state:
                        state["momentum_buffer_slow"] = torch.zeros_like(w2d)
                if group.get("normuon", False) and "normuon_v" not in state:
                    state["normuon_v"] = torch.zeros(
                        w2d.size(0), 1, dtype=torch.float32, device=w2d.device
                    )
                if group["aurora_twin_rail"] or group["samuon_lite"]:
                    state["step"] = state.get("step", 0) + 1
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
                    twin_rail=group["aurora_twin_rail"],
                    momentum_slow=state.get("momentum_buffer_slow"),
                    slow_beta=group["rail_slow_beta"],
                    step=state.get("step", 0),
                    rail_engage_step=group["rail_engage_step"],
                    rail_fast_weight=group["rail_fast_weight"],
                    samuon_lite=group["samuon_lite"],
                    samuon_gamma=group["samuon_gamma"],
                    samuon_warmup_steps=group["samuon_warmup_steps"],
                    samuon_power_iters=group["samuon_power_iters"],
                    cautious_wd=group.get("cautious_wd", False),
                    normuon=group.get("normuon", False),
                    normuon_v=state.get("normuon_v"),
                    normuon_beta2=group.get("normuon_beta2", 0.95),
                    normuon_eps=group.get("normuon_eps", 1e-10),
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
