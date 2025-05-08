from dataclasses import field, dataclass
from coqpit import Coqpit


@dataclass
class BestConfig(Coqpit):
    """Best model configuration for BLA-GPT.

    Results:
    step:5100/5100 val_loss:3.2923 train_time:3159041ms step_avg:620.64ms
    peak memory consumption: 40332 MiB
    """

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