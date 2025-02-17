import torch
import torch.nn as nn
from torch.nn import functional as F


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def semi_orthogonal_init(dim_in, dim_out, steps=5):
    # default pytorch linear layer init
    weight = (1 / dim_in) ** 0.5 * (
        2 * (torch.rand(dim_out, dim_in, device="cuda")) - 1.0
    )

    w_std = weight.std(dim=0).mean()
    weight = zeropower_via_newtonschulz5(weight, steps=steps)
    weight *= w_std / weight.std(dim=0).mean().add_(1e-8)
    return weight.float()


class Primer_MLP(nn.Module):
    # from 🎩 https://gist.github.com/tysam-code/b3519fd58ce5c94d1016c8903e50736d
    def __init__(self, config):
        super().__init__()
        expand = 4
        expand_dim = expand * config.n_embd
        self.c_fc_scale = nn.Parameter(torch.ones(config.n_embd))
        self.c_fc = nn.Parameter(semi_orthogonal_init(config.n_embd, expand_dim))
        self.c_proj = nn.Parameter(torch.zeros(config.n_embd, expand_dim))
        self.c_proj_scale = nn.Parameter(torch.ones(config.n_embd))

    def forward(self, x):
        x = F.linear(
            x, (self.c_fc * self.c_fc_scale.unsqueeze(0)).type_as(x)
        )  # fuse weight & weight_scale mults
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, (self.c_proj_scale.unsqueeze(1) * self.c_proj).type_as(x))
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GeGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = self.c_gate(x)
        x = self.c_fc(x)
        x = F.gelu(gate) * x
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = self.c_gate(x)
        x = self.c_fc(x)
        x = F.silu(gate) * x
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Negout_MLP(nn.Module):
    def __init__(self, config):
        """Negout MLP
        I developed Negout when I was a child
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_fc_neg = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        pos = self.c_fc(x)
        neg = self.c_fc_neg(x)
        x = torch.where(neg > pos, -neg, pos)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Maxout_MLP(nn.Module):
    def __init__(self, config):
        """Maxout MLP implementation
        Uses maxout activation function that takes the maximum value
        across multiple linear transformations
        """
        super().__init__()
        self.num_pieces = 2  # number of pieces for maxout

        # Create multiple linear layers for maxout
        self.c_fc = nn.ModuleList(
            [
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
                for _ in range(self.num_pieces)
            ]
        )

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Apply each linear transformation
        pieces = [fc(x) for fc in self.c_fc]

        # Stack the pieces and take maximum along the pieces dimension
        x = torch.stack(pieces, dim=-1)
        x = torch.max(x, dim=-1)[0]

        # Project back to original dimension and apply dropout
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
