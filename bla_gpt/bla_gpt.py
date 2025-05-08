"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from attentions import (Attention, DilatedAttention, ForgettingAttention,
                        KVShiftingAttention, MultiheadDiffAttn,
                        MultiHeadLatentAttention, MultiTokenAttention,
                        PattentionSelfAttention, soft_cap)
from coqpit import Coqpit
from mlps import MLP, GeGLU_MLP, Maxout_MLP, Negout_MLP, Primer_MLP, SwiGLU_MLP
from modules.pattention import Pattention
from modules.canon_layer import CanonLayer
from norms import DyTNorm, LayerNorm, RMSNorm
from torch.nn import functional as F
from utils import register_model


@dataclass
class GPTConfig(Coqpit):
    """Default values for the best performed model so far"""

    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_latentd: int = 0  # 192  # only for MultiHeadLatentAttention
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # New multi-token prediction parameters
    n_predict: int = 1  # Number of future tokens to predict (1 = standard GPT)
    share_prediction_heads: bool = (
        False  # Whether to share parameters between prediction heads
    )

    # Transformer parameters
    norm_layer: str = "rmsnorm" # type of normalization layer to use
    attention: str = "regular" # attention type in `get_attention()`
    activation: str = "swiglu" # activation type in `get_mlp()`
    use_soft_logit_capping: bool = False
    n_kv_head: int = 4  # Number of heads for the key and value (Grouped Query Attention), if n_kv_head == n_head, it is full attention
    tie_embed_weights: bool = True
    zero_init_proj_layers: bool = True
    rmsnorm_before_qk: bool = True
    pos_encoding: bool = "rotary"
    use_res_weights: bool = False
    use_qkv_bias: bool = False  # from Qwen, for better length generalization. Not an issue with block_size=1024
    use_pre_post_norm: bool = False  # from Qwen, for better training stability
    rope_theta: float = 10000  # 1000000.0 in llama3 models
    use_per_token_output_bias: bool = (
        False  # use an embedding layer to add a bias to each token prediction
    )
    use_softpick: bool = False  # use softpick instead of softmax in attention block - https://arxiv.org/html/2504.20966v1
                                # when True model defaults to vanilla attention instead of flash attention

    # Multi-token attention parameters
    use_key_query_conv=True,
    query_kernel_size=6,
    key_kernel_size=11,
    pre_softmax_key_query=True,
    use_head_conv=True,
    head_kernel_size=2,
    pre_softmax_head=False,
    use_group_norm=True,
    apply_key_query_every_n_layers=4

    # Canon layer parameters
    use_canon_layers: bool = False  # Whether to use Canon layers before MLP and Attention blocks (Configs A and C in the paper)

    # Parallel Transformer (like PaLM) block parameters
    use_parallel_blocks: bool = False  # Whether to apply attention and mlp blocks in parallel instead of sequentially

    # Dilated attention parameters
    segment_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    dilation_rates: list[int] = field(default_factory=lambda: [1, 2, 4, 6, 12])

    """About z-loss: instability occurs
    when the logits diverge and become very negative, as
    illustrated in Figure 4 for a 2.4M parameter model at
    learning rate 0.1. In contrast to the attention logit
    growth instability, this divergence occurs towards the
    end of training. The mitigation proposed by Chowdhery et al.
    [6] is to encourage log Z to remain close to
    zero."""

    z_loss_weight: float = 0.0  # 1e-4 from deepmind paper

    # optimizer - overriding Hyperparameters
    optimizer_name: str = (
        "Muon"  # check get_optimizer() in bla_gpt/optimizers/__init__.py
    )
    optimizer_args: dict = field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
        }
    )


@dataclass
class TokenformerConfig(GPTConfig):
    """Default values for the best performed model so far"""

    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # New multi-token prediction parameters
    n_predict: int = 1  # Number of future tokens to predict (1 = standard GPT)
    share_prediction_heads: bool = (
        False  # Whether to share parameters between prediction heads
    )

    norm_layer: str = "rmsnorm"
    attention: str = "pattention"
    activation: str = "pattention"
    use_soft_logit_capping: bool = False
    n_kv_head: int = 12  # Number of heads for the key and value (Grouped Query Attention), if n_kv_head == n_head, it is full attention
    tie_embed_weights: bool = True
    zero_init_proj_layers: bool = False
    rmsnorm_before_qk: bool = True
    pos_encoding: bool = "rotary"
    use_res_weights: bool = False
    use_pre_post_norm: bool = True

    # pattention parameters
    param_token_num: int = 1024


def compute_z_loss(logits, dim=-1, eps=1e-20):
    """
    Compute z-loss regularization to prevent model from being too confident.

    Args:
        logits: Raw logits from model of shape [batch, seq_len, vocab_size]
        dim: Dimension along which to compute z-loss (usually vocab dimension)
        eps: Small constant for numerical stability

    Returns:
        z_loss: Scalar z-loss term to add to training loss
    """
    # Get log of the partition function (logsumexp)
    log_z = torch.logsumexp(logits, dim=dim, keepdim=True)

    # Compute mean of log_z squared
    z_loss = torch.square(torch.max(log_z, torch.zeros_like(log_z)))
    z_loss = torch.mean(z_loss)

    return z_loss


def get_attention(config, depth=None):
    if config.attention == "regular":
        return Attention(config)
    if config.attention == "latent":
        return MultiHeadLatentAttention(config)
    elif config.attention == "DiffAttn":
        return MultiheadDiffAttn(config, depth)
    elif config.attention == "pattention":
        return PattentionSelfAttention(config)
    elif config.attention == "kvshifting":
        return KVShiftingAttention(config)
    elif config.attention == "dilated":
        return DilatedAttention(config)
    elif config.attention == "forgetting":
        return ForgettingAttention(config)
    elif config.attention == "multi-token":
        return MultiTokenAttention(config)
    raise ValueError(f"Unrecognized attention type {config.attention}")


def get_norm(config):
    if config.norm_layer == "rmsnorm":
        return RMSNorm(config.n_embd)
    elif config.norm_layer == "layernorm":
        return LayerNorm(config.n_embd, config.bias)
    elif config.norm_layer == "dytnorm":
        return DyTNorm(config.n_embd)
    raise ValueError(f"Unrecognized norm type {config.norm_layer}")


def get_mlp(config):
    if config.activation == "gelu":
        return MLP(config)
    elif config.activation == "geglu":
        return GeGLU_MLP(config)
    elif config.activation == "swiglu":
        return SwiGLU_MLP(config)
    elif config.activation == "primer":
        return Primer_MLP(config)
    elif config.activation == "negout":
        return Negout_MLP(config)
    elif config.activation == "maxout":
        return Maxout_MLP(config)
    elif config.activation == "pattention":
        return Pattention(config)
    raise ValueError(f"Unrecognized activation type {config.activation}")


class MultiTokenPredictionHead(nn.Module):
    """Head for predicting multiple future tokens"""

    def __init__(self, config, head_index):
        super().__init__()
        self.head_index = head_index
        self.norm = get_norm(config)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)


class Block(nn.Module):
    """
    A single Transformer block with attention and MLP layers.
    """
    def __init__(self, config, depth):
        super().__init__()
        self.ln_1 = get_norm(config)
        self.attn = get_attention(config, depth)
        self.ln_2 = get_norm(config)
        self.mlp = get_mlp(config)

        # Optional norm layers as per Gemma2
        self.ln_3 = get_norm(config) if config.use_pre_post_norm else None
        self.ln_4 = get_norm(config) if config.use_pre_post_norm else None

        # Initialize closer to zero for better training stability
        self.res_w1 = nn.Parameter(torch.ones(1)) if config.use_res_weights else None
        self.res_w2 = nn.Parameter(torch.ones(1)) if config.use_res_weights else None

    def _process_branch(self, x, ln_pre, branch_fn, ln_post=None, res_weight=None):
        # Process input through branch (attention or MLP)
        branch_out = branch_fn(ln_pre(x))

        # Apply optional post-normalization
        if ln_post is not None:
            branch_out = ln_post(branch_out)

        # Apply optional residual weight
        if res_weight is not None:
            return res_weight * x + branch_out
        return x + branch_out

    def forward(self, x):
        # Attention branch
        x = self._process_branch(x, self.ln_1, self.attn, self.ln_3, self.res_w1)

        # MLP branch
        x = self._process_branch(x, self.ln_2, self.mlp, self.ln_4, self.res_w2)

        return x


class ParallelBlock(Block):
    """
    A Transformer block that runs Attention and MLP branches in parallel
    with causal constraints.
    """

    def _process_branch(self, x, ln_pre, branch_fn, ln_post=None):
        # Process input through branch (attention or MLP)
        branch_out = branch_fn(ln_pre(x))

        # Apply optional post-normalization
        if ln_post is not None:
            branch_out = ln_post(branch_out)
        return branch_out

    def forward(self, x):
        # Attention branch - already has causal masking
        x1 = self._process_branch(x, self.ln_1, self.attn, self.ln_3)

        # Apply MLP to each position separately then mask to maintain causality
        x2 = self._process_branch(x, self.ln_2, self.mlp, self.ln_4)

        # Zero out future influences - each position only influenced by current and past tokens
        return x + x1 + x2


class CanonBlock(Block):
    """
    A Transformer block that applies a CanonLayer before both attention and MLP operations.
    Inherits from the standard Block class and adds causal 1D convolution layers.

    """
    # NOTE: For some reason, there is future leakage in the CanonBlock, getting very low training loss very fast.

    def __init__(self, config, depth, canon_kernel_size=4):
        super().__init__(config, depth)

        # Add CanonLayers before attention and MLP
        self.canon_attn = CanonLayer(hidden_dim=config.n_embd, kernel_size=canon_kernel_size)
        self.canon_mlp = CanonLayer(hidden_dim=config.n_embd, kernel_size=canon_kernel_size)

    def _process_branch(self, x, ln_pre, branch_fn, canon_fn, ln_post=None, res_weight=None):
        # Pre-norm
        branch_out = ln_pre(x)

        # Apply Canon layer
        branch_out = branch_out + canon_fn(branch_out)

        # Apply branch function (attention or MLP)
        branch_out = branch_fn(branch_out)

        # Apply optional post-normalization
        if ln_post is not None:
            branch_out = ln_post(branch_out)

        # Apply optional residual weight
        if res_weight is not None:
            return res_weight * x + branch_out

        # Residual connection
        return x + branch_out

    def forward(self, x):
        # Attention branch
        x = self._process_branch(x, self.ln_1, self.attn, self.canon_attn, self.ln_3, self.res_w1)

        # MLP branch
        x = self._process_branch(x, self.ln_2, self.mlp, self.canon_mlp, self.ln_4, self.res_w2)

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.soft_cap = 30.0 if config.use_soft_logit_capping else 0.0
        self.zero_init_proj_layers = config.zero_init_proj_layers


        _Block = Block
        if "use_parallel_blocks" in config and config.use_parallel_blocks:
            _Block = ParallelBlock
        elif "use_canon_layer" in config and config.use_canon_layers:
            _Block = CanonBlock

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd)
                if not config.pos_encoding not in [None, "none"]
                else None,
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([_Block(config, d) for d in range(config.n_layer)]),
                ln_f=get_norm(config),
            )
        )

        if config.n_predict == 1:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        else:
            # Independent parameters for each head
            self.prediction_heads = nn.ModuleList(
                [MultiTokenPredictionHead(config, i) for i in range(config.n_predict)]
            )

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        if config.tie_embed_weights and config.n_predict == 1:
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

        # optional per token bias
        if config.use_per_token_output_bias:
            self.output_bias_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                if self.zero_init_proj_layers:
                    torch.nn.init.zeros_(p)
                else:
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                    )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.transformer.wpe is not None:
            pos_emb = self.transformer.wpe(
                pos
            )  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # TODO: simplify this
        if targets is not None:
            # single token prediction head
            # if we are given some desired targets also calculate the loss
            if hasattr(self, "lm_head"):
                if self.config.use_per_token_output_bias:
                    x = x + self.output_bias_emb(idx)
                logits = self.lm_head(x)
                logits = logits.float()
                if self.soft_cap > 0.0:
                    logits = soft_cap(logits, self.soft_cap)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )

                if self.config.z_loss_weight > 0.0:
                    z_loss = compute_z_loss(logits)
                    loss += self.config.z_loss_weight * z_loss

                    # we need to see lm_loss and z_loss separately for logging
                    loss = {
                        "total": loss,
                        "z_loss": z_loss.detach().item(),
                        "lm_loss": loss.detach().item(),
                    }
            else:
                # multi-token prediction heads
                total_loss = 0
                loss_dict = {}
                logits = []
                for i, head in enumerate(self.prediction_heads):
                    if self.config.use_per_token_output_bias:
                        x = x + self.output_bias_emb(idx)
                    head_logits = head(x)  # (b, t, vocab_size)
                    logits.append(head_logits)

                    if targets is not None:
                        # Shift targets for each prediction head
                        shifted_targets = (
                            targets[:, i:] if i < targets.size(1) else None
                        )
                        if shifted_targets is not None:
                            head_logits = head_logits[
                                :, : shifted_targets.size(1), :
                            ].float()
                            if self.soft_cap > 0.0:
                                head_logits = soft_cap(head_logits, self.soft_cap)
                            loss = F.cross_entropy(
                                head_logits.reshape(-1, head_logits.size(-1)),
                                shifted_targets.reshape(-1),
                                ignore_index=-1,
                            )

                            if self.config.z_loss_weight > 0.0:
                                z_loss = compute_z_loss(head_logits)
                                loss += self.config.z_loss_weight * z_loss

                            total_loss += loss
                            loss_dict[f"head{i+1}"] = loss.detach().item()

                # Average the losses across heads
                loss = total_loss / len(self.prediction_heads)
                loss = {"total": loss}
                loss.update(loss_dict)

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
