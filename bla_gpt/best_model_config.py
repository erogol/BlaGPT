from dataclasses import dataclass, field

from coqpit import Coqpit

from bla_gpt import GPTConfig


@dataclass
class BestConfig(GPTConfig):
    """Best model configuration for BLA-GPT.
    """

    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 10
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
    norm_layer: str = "rmsnorm"  # type of normalization layer to use
    attention: str = "xsa"  # attention type in `get_attention()`
    activation: str = "primer"  # activation type in `get_mlp()`
    use_soft_logit_capping: bool = False
    n_kv_head: int = 12  # Number of heads for the key and value (Grouped Query Attention), if n_kv_head == n_head, it is full attention
    tie_embed_weights: bool = True
    zero_init_proj_layers: bool = True
    rmsnorm_before_qk: bool = True
    pos_encoding: str = "grape_a_qgate"
    use_res_weights: bool = False
    use_qkv_bias: bool = False  # from Qwen, for better length generalization. Not an issue with block_size=1024
    use_pre_post_norm: bool = False  # from Qwen, for better training stability
    rope_theta: float = 1000000  # 1000000.0 in llama3 models
    rope_variant: str = "standard"
    use_per_token_output_bias: bool = (
        False  # use an embedding layer to add a bias to each token prediction
    )
    use_softpick: bool = False  # use softpick instead of softmax in attention block - https://arxiv.org/html/2504.20966v1
    # when True model defaults to vanilla attention instead of flash attention

    # Multi-token attention parameters
    use_key_query_conv = (True,)
    query_kernel_size = (6,)
    key_kernel_size = (11,)
    pre_softmax_key_query = (True,)
    use_head_conv = (True,)
    head_kernel_size = (2,)
    pre_softmax_head = (False,)
    use_group_norm = (True,)
    apply_key_query_every_n_layers = 4

    # Canon layer parameters
    use_canon_layers: bool = False  # Whether to use Canon layers before MLP and Attention blocks (Configs A and C in the paper)

    # Parallel Transformer (like PaLM) block parameters
    use_parallel_blocks: bool = False  # Whether to apply attention and mlp blocks in parallel instead of sequentially

    use_per_layer_token_emb: bool = False  # Whether to add token embedding to the block input
    per_layer_token_emb_dim: int = 256  # Dimension of the per-layer token embedding, if use_per_layer_token_emb is True

    # Using TOP loss
    use_top: bool = False
    top_window_size: int = 1024  # Window size for TOP target construction (should be <= block_size)
    top_loss_weight: float = 1.0

    # # engram config
    use_engram: bool = True
    engram_layers=[1,2,3,4,5,6]
    engram_ngram=3
    engram_vocab_mult: int = 5
    engram_share_embedding: bool = True
    engram_variant: str = "minimal"

    # Dilated attention parameters
    segment_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    dilation_rates: list[int] = field(default_factory=lambda: [1, 2, 4, 6, 12])

    # GRAPE-A query-gated positional encoding
    grape_omega_init: float = 1.0
    grape_v_l2_norm: bool = True

    # Architecture: depth/width
    mlp_expand: int = 10

    # GatedNorm
    use_gated_norm: bool = True
    gated_norm_rank: int = 16

    # HybridNorm
    use_hybrid_norm: bool = True

    # GOAT sink prior + composable gated attention
    use_goat_sink_prior: bool = True
    use_composable_gated_attn: bool = True

    # Value residual + attention residual + U-net skips
    use_value_residual: bool = True
    use_attn_res: bool = True
    use_unet_skips: bool = True
    unet_skip_init: float = 0.25

    # PolyNorm
    polycom_order: int = 3

    # Hyper connections
    use_hyper_connections: bool = False
    hyper_dynamic: bool = True
    hyper_num_streams: int = 4
    hyper_tanh: bool = True

    # KDA (off but configured)
    use_kda_interleaving: bool = False
    kda_chunk_size: int = 64
    kda_interleave_ratio: int = 4
    kda_interleave_with: str = "regular"
    kda_use_short_conv: bool = True
    kda_decay_rank: int | None = None

    # Sequence curriculum
    seq_curriculum_len: int = 384
    seq_curriculum_steps: int = 1200
    warmup_iters: int = 100

    # STEM (off)
    use_stem: bool = False
    stem_layers: list = field(default_factory=list)
    stem_ratio: float | None = None

    # Weight sharing (off)
    enable_weight_sharing: bool = False
    weight_sharing_groups: int = 1
    weight_sharing_pattern: list = field(default_factory=list)

    # TOP (off but configured)
    top_force_optimized: bool = True

    # Learning rate override (F90r2: 1.4x default)
    learning_rate: float = 0.0014

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
            "use_cautious_weight_decay": False,
        }
    )
