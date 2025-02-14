import math

import torch
import torch.nn.functional as F
from modules.pattention import Pattention
from torch import nn
from torch.nn import functional as F

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from norms import RMSNorm


def soft_cap(x, cap):
    return x.div_(cap).tanh_().mul_(cap)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    with torch.autocast(device_type=x.device.type, enabled=False):
        d = x.shape[3] // 2
        x1 = x[..., :d].float()
        x2 = x[..., d:].float()
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000, seq_len=1024):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.cos = nn.Buffer(freqs.cos().float(), persistent=False)
        self.sin = nn.Buffer(freqs.sin().float(), persistent=False)

    def forward(self, x):
        return self.cos[None, : x.size(-3), None, :], self.sin[
            None, : x.size(-3), None, :
        ]


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head  # New parameter for number of key-value heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.soft_cap = 50.0 if config.use_soft_logit_capping else 0.0

        # RMSNorm before q and k projections
        if config.rmsnorm_before_qk:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Rotary embeddings
        if config.pos_encoding == "rotary":
            self.rotary = Rotary(self.head_dim, base=config.rope_theta)
        elif config.pos_encoding == "relative":
            self.rel_pos_emb = nn.Parameter(
                torch.zeros(2 * config.block_size - 1, self.head_dim)
            )
            nn.init.normal_(self.rel_pos_emb, std=0.02)
        elif config.pos_encoding == "none" or config.pos_encoding is None:
            pass
        else:
            raise ValueError(f"Unknown positional encoding: {config.pos_encoding}")

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Flash attention support
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.set_layers(config)

    def set_layers(self, config):
        # Projections for query, key, and value
        self.q_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias or config.use_qkv_bias
        )
        self.kv_proj = nn.Linear(
            config.n_embd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias or config.use_qkv_bias,
        )
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, q=None, mask=None):
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Update mask if provided
        if mask is not None:
            self.mask = mask

        # Project inputs
        q = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Apply normalization and rotary embeddings if configured
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)

        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T_q, T)
        elif hasattr(self, "rel_pos_emb"):
            q, k = self._apply_relative_pos(q, k, T_q, T)

        # Prepare attention inputs
        q, k, v = self._prepare_qkv(q, k, v)

        # Compute attention
        if self.flash and self.soft_cap == 0:
            y = self._flash_attention(q, k, v)
        else:
            y = self._manual_attention(q, k, v, T)

        # Project output
        return self._project_output(y, B, T_q, C)

    def _project_query(self, x, B, T):
        return self.q_proj(x).view(B, T, self.n_head, self.head_dim)

    def _project_kv(self, x, B, T):
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        return kv.unbind(dim=2)

    def _apply_norm(self, q, k):
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k

    def _apply_rotary(self, q, k, T_q, T):
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        cos, sin = self.rotary(k) if T_q != T else (cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        return q, k

    def _apply_relative_pos(self, q, k, T_q, T):
        # Get relative position embeddings
        pos_emb = self._get_rel_pos_emb(T_q, T)

        # Apply relative position embeddings
        q = q + pos_emb[:T_q].unsqueeze(0).unsqueeze(0)
        k = k + pos_emb[:T].unsqueeze(0).unsqueeze(0)

        return q, k

    def _get_rel_pos_emb(self, T_q, T):
        # Get relative position embeddings centered around each position
        seq_length = max(T_q, T)
        positions = torch.arange(seq_length, device=self.rel_pos_emb.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions = (
            relative_positions + seq_length - 1
        )  # shift to all positive
        return self.rel_pos_emb[relative_positions]

    def _prepare_qkv(self, q, k, v):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Repeat k,v for multi-query attention
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        return q, k, v

    def _flash_attention(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )

    def _manual_attention(self, q, k, v, T):
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.soft_cap > 0:
            att = soft_cap(att, self.soft_cap)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    def _project_output(self, y, B, T_q, C):
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        return self.resid_dropout(self.c_proj(y))


class MultiHeadLatentAttention(Attention):
    def __init__(self, config):
        assert config.n_lantentd > 0, "Must provide number of latent dimensions"
        self.n_latentd = config.n_latentd
        super().__init__(config)

    def set_layers(self, config):
        self.q_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias or config.use_qkv_bias
        )
        self.kv_latent = nn.Linear(
            config.n_embd,
            self.n_latentd,
            bias=config.bias or config.use_qkv_bias,
        )
        self.kv_proj = nn.Linear(
            self.n_latentd,
            2 * config.n_embd // (config.n_head // config.n_kv_head),
            bias=config.bias or config.use_qkv_bias,
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def _project_kv(self, x, B, T):
        kv = self.kv_latent(x)
        kv = self.kv_proj(kv).view(B, T, 2, self.n_kv_head, self.head_dim)
        return kv.unbind(dim=2)


class PattentionSelfAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

    def set_layers(self, config):
        self.q_proj = Pattention(config)
        self.k_proj = Pattention(config)
        self.v_proj = Pattention(config)
        self.c_proj = Pattention(config)

    def _project_kv(self, x, B, T):
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        return k, v


class KVShiftingAttention(Attention):
    def __init__(self, config):
        super().__init__(config)

        # Initialize KV shifting parameters
        # Following paper's initialization: randomly initialize from U(0,1)
        # and make them sum to 1
        self.alpha1 = nn.Parameter(torch.rand(self.n_kv_head))
        self.alpha2 = nn.Parameter(torch.ones(self.n_kv_head) - self.alpha1)
        self.beta1 = nn.Parameter(torch.rand(self.n_kv_head))
        self.beta2 = nn.Parameter(torch.ones(self.n_kv_head) - self.beta1)

    def _shift_kv(self, x):
        """Perform shifting operation on key/value tensors.
        Shifts the sequence by padding a zero at the beginning and dropping last element.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_kv_head, head_dim)

        Returns:
            Shifted tensor of same shape
        """
        # Get shifted version by padding front and removing last element
        # Keep same dimensions by dropping last element after padding
        x_shifted = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        return x_shifted

    def _project_kv(self, x, B, T):
        """Override parent's _project_kv to add KV shifting.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            B: Batch size
            T: Sequence length

        Returns:
            Tuple of processed key and value tensors
        """
        # Get initial K,V projections using parent method
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv.unbind(dim=2)

        # Get shifted versions
        k_shifted = self._shift_kv(k)
        v_shifted = self._shift_kv(v)

        # Combine original and shifted versions with learned parameters
        k = (
            self.alpha1.view(1, 1, -1, 1) * k
            + self.alpha2.view(1, 1, -1, 1) * k_shifted
        )
        v = self.beta1.view(1, 1, -1, 1) * v + self.beta2.view(1, 1, -1, 1) * v_shifted

        return k, v


class DilatedAttention(Attention):  # TOOO SLOW !!
    def __init__(
        self,
        config,
    ):
        """
        Implements the dilated attention mechanism from the LongNet paper.

        Args:
            config: Configuration object with attention parameters
            segment_sizes: List of window sizes for each dilated attention
            dilation_rates: List of dilation rates corresponding to each segment size
        """
        super().__init__(config)
        assert len(config.segment_sizes) == len(
            config.dilation_rates
        ), "Must provide same number of segment sizes and dilation rates"

        self.segment_sizes = config.segment_sizes
        self.dilation_rates = config.dilation_rates

    def _get_dilated_indices(self, seq_len, segment_size, dilation_rate, num_heads):
        """
        Generate dilated indices for all heads at once.
        Ensures output size is consistent with the input sequence length.
        """
        # Calculate how many tokens per segment after dilation
        tokens_per_segment = math.ceil(segment_size / dilation_rate)

        # Calculate number of segments
        num_segments = math.ceil(seq_len / segment_size)

        # Initialize indices for all heads
        all_indices = []
        for head in range(num_heads):
            head_indices = []
            offset = head % dilation_rate

            # Generate indices for each segment
            for seg in range(num_segments):
                start_idx = seg * segment_size
                # Generate base indices for this segment
                seg_indices = torch.arange(
                    start_idx + offset,
                    min(start_idx + segment_size, seq_len),
                    dilation_rate,
                )
                head_indices.append(seg_indices)

            # Concatenate all segment indices
            head_indices = torch.cat(head_indices)

            # Pad or truncate to match sequence length
            if len(head_indices) < seq_len:
                padding = torch.full(
                    (seq_len - len(head_indices),),
                    seq_len - 1,
                    dtype=head_indices.dtype,
                )
                head_indices = torch.cat([head_indices, padding])
            else:
                head_indices = head_indices[:seq_len]

            all_indices.append(head_indices)

        # Stack indices for all heads
        return torch.stack(all_indices)  # (num_heads, seq_len)

    def _dilate_qkv(self, qkv, segment_size, dilation_rate):
        """
        Apply dilation to query, key, or value tensor while maintaining sequence length.
        """
        batch_size, num_heads, seq_len, head_dim = qkv.shape
        indices = self._get_dilated_indices(
            seq_len, segment_size, dilation_rate, num_heads
        )
        indices = indices.to(qkv.device)

        # Expand indices for batch and head dimensions
        expanded_indices = (
            indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, head_dim)
        )

        # Gather along sequence length dimension
        return torch.gather(qkv, 2, expanded_indices)

    def _prepare_qkv(self, q, k, v):
        """
        Prepare query, key, and value tensors for dilated attention computation.
        """
        # First apply the standard preparation from parent class
        q, k, v = super()._prepare_qkv(q, k, v)

        dilated_outputs = []

        # Process each segment size and dilation rate
        for segment_size, dilation_rate in zip(self.segment_sizes, self.dilation_rates):
            # Apply dilation to q, k, v while maintaining sequence length
            q_dilated = self._dilate_qkv(q, segment_size, dilation_rate)
            k_dilated = self._dilate_qkv(k, segment_size, dilation_rate)
            v_dilated = self._dilate_qkv(v, segment_size, dilation_rate)

            dilated_outputs.append((q_dilated, k_dilated, v_dilated))

        return dilated_outputs

    def _flash_attention(self, dilated_qkv):
        """
        Compute attention using flash attention for each dilated version.
        """
        outputs = []
        attention_weights = []

        for q, k, v in dilated_qkv:
            # Compute attention with flash attention
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
            outputs.append(out)

            # Compute attention weights for dynamic weighting
            with torch.no_grad():
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * (
                    1.0 / math.sqrt(k.size(-1))
                )
                attn_weights = attn_weights.softmax(dim=-1).mean(dim=(0, 1))
                attention_weights.append(attn_weights.max().item())

        # Compute dynamic weights based on attention scores
        weights = torch.softmax(
            torch.tensor(attention_weights, device=outputs[0].device), dim=0
        )

        # Combine outputs using dynamic weights
        combined_output = sum(w * out for w, out in zip(weights, outputs))
        return combined_output

    def _manual_attention(self, dilated_qkv, T):
        """
        Compute attention manually for each dilated version.
        """
        outputs = []
        attention_weights = []

        for q, k, v in dilated_qkv:
            # Compute attention scores
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if self.soft_cap > 0:
                attn = soft_cap(attn, self.soft_cap)

            # Create causal mask for dilated attention
            causal_mask = torch.ones_like(attn, dtype=torch.bool).triu_()
            attn = attn.masked_fill(~causal_mask, float("-inf"))

            # Compute attention weights
            attn_probs = F.softmax(attn, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)

            # Compute output
            out = attn_probs @ v
            outputs.append(out)

            # Store attention weights for dynamic weighting
            with torch.no_grad():
                attention_weights.append(attn_probs.mean(dim=(0, 1)).max().item())

        # Compute dynamic weights based on attention scores
        weights = torch.softmax(
            torch.tensor(attention_weights, device=outputs[0].device), dim=0
        )

        # Combine outputs using dynamic weights
        combined_output = sum(w * out for w, out in zip(weights, outputs))
        return combined_output

    def forward(self, x, q=None, mask=None):
        """
        Forward pass for dilated attention.
        """
        B, T, C = x.size()
        T_q = q.size(1) if q is not None else T

        # Project inputs
        q = self._project_query(q if q is not None else x, B, T_q)
        k, v = self._project_kv(x, B, T)

        # Apply normalization and positional encodings if configured
        if hasattr(self, "q_norm"):
            q, k = self._apply_norm(q, k)

        if hasattr(self, "rotary"):
            q, k = self._apply_rotary(q, k, T_q, T)
        elif hasattr(self, "rel_pos_emb"):
            q, k = self._apply_relative_pos(q, k, T_q, T)

        # Prepare dilated attention inputs
        dilated_qkv = self._prepare_qkv(q, k, v)

        # Compute attention
        if self.flash and self.soft_cap == 0:
            y = self._flash_attention(dilated_qkv)
        else:
            y = self._manual_attention(dilated_qkv, T)

        # Project output
        return self._project_output(y, B, T_q, C)


# Example usage:
# config = DilatedConfig(
#     n_embd=512,
#     n_head=8,
#     n_kv_head=8,
#     block_size=1024,
#     dilation_rate=2,
#     segment_size=64,
#     use_xpos=True,
#     use_rel_pos_bias=True,
#     qk_norm=True
# )
# dilated_attention = DilatedAttention(config)

"""
Differential Attention - WIP (Getting OOM)
"""


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        config,
        depth,
    ):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        # num_heads set to half of Transformer's #heads
        self.num_kv_heads = (
            config.n_kv_head if config.n_kv_head is not None else self.num_heads
        )
        self.n_rep = self.num_heads // self.num_kv_heads

        self.head_dim = self.embed_dim // self.num_heads // 2
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(
            self.embed_dim, self.embed_dim // self.n_rep, bias=False
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.embed_dim // self.n_rep, bias=False
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        self.subln = RMSNorm(2 * self.head_dim)
        self.rotary = Rotary(self.head_dim)

    def forward(
        self,
        x,
        attn_mask=None,
    ):
        bsz, tgt_len, _ = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        cos, sin = self.rotary(q)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )

        attn = self.out_proj(attn)
        return attn
