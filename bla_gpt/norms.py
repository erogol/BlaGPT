import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    def __init__(self, ndim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return x * self.weight


class DyTNorm(nn.Module):
    # https://arxiv.org/abs/2503.10622

    def __init__(self, ndim, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(ndim))
        self.beta = nn.Parameter(torch.zeros(ndim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


class PreAffineRMSNorm(nn.Module):
    """Qiu et al. 2026 (arXiv:2601.22966) Sec.3.3: RMSNorm(lambda1 * x) with trainable lambda1 init=ones."""

    def __init__(self, ndim, eps=1e-8):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(ndim))
        self.norm = RMSNorm(ndim, eps=eps)

    def forward(self, x):
        return self.norm(self.lambda1 * x)


class GatedNorm(nn.Module):
    """Qiu et al. 2026 (arXiv:2601.22966) Sec.3.4: low-rank sigmoid gate applied after RMSNorm.

    y  = RMSNorm(x)
    yg = sigmoid(W_up(swish(W_down(y))))   # W_down: d->r, W_up: r->d
    y' = yg * y

    Init: W_up=zeros so gate=sigmoid(0)=0.5 at step 0 (half-identity).
    W_down is Kaiming-normal (default), so swish(W_down(y)) is non-zero for typical y,
    ensuring gradients reach W_up immediately. As W_up trains toward positive weights the
    gate converges toward 1. W_down gradients are zero at step 0 (W_up=0) but non-zero
    once W_up is updated. This mirrors the standard LoRA-style bottleneck init (B=0).
    """

    def __init__(self, ndim, rank=16, eps=1e-8):
        super().__init__()
        self.norm = RMSNorm(ndim, eps=eps)
        self.W_down = nn.Linear(ndim, rank, bias=False)
        self.W_up = nn.Linear(rank, ndim, bias=False)
        nn.init.zeros_(self.W_up.weight)

    def forward(self, x):
        y = self.norm(x)
        gate = torch.sigmoid(self.W_up(F.silu(self.W_down(y))))
        return gate * y
