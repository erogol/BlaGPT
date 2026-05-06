import torch
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SPEC = spec_from_file_location("blagpt_model", Path(__file__).resolve().parents[1] / "bla_gpt" / "bla_gpt.py")
_BLA = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BLA)

_ATT_SPEC = spec_from_file_location("blagpt_attentions", Path(__file__).resolve().parents[1] / "bla_gpt" / "attentions.py")
_ATT = module_from_spec(_ATT_SPEC)
assert _ATT_SPEC.loader is not None
_ATT_SPEC.loader.exec_module(_ATT)

GPT = _BLA.GPT
GPTConfig = _BLA.GPTConfig
get_attention = _BLA.get_attention
ExclusiveSelfAttention = _ATT.ExclusiveSelfAttention


def tiny_config(n_head=2, n_kv_head=2):
    cfg = GPTConfig()
    cfg.block_size = 8
    cfg.vocab_size = 64
    cfg.n_layer = 2
    cfg.n_head = n_head
    cfg.n_kv_head = n_kv_head
    cfg.n_embd = 16
    cfg.dropout = 0.0
    cfg.bias = True
    cfg.pos_encoding = "none"
    cfg.activation = "gelu"
    cfg.norm_layer = "layernorm"
    cfg.attention = "xsa"
    cfg.tie_embed_weights = False
    cfg.zero_init_proj_layers = False
    cfg.use_engram = False
    return cfg


def test_xsa_registry_returns_dedicated_subclass():
    attn = get_attention(tiny_config())
    assert attn.__class__.__name__ == "ExclusiveSelfAttention"


def test_xsa_forward_backward_shapes():
    torch.manual_seed(0)
    attn = get_attention(tiny_config())
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = attn(x)
    assert y.shape == (2, 5, 16)
    loss = y.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_xsa_output_is_orthogonal_to_value_direction():
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = get_attention(cfg)
    x = torch.randn(2, 5, cfg.n_embd)
    B, T, _ = x.shape

    q = attn._project_query(x, B, T)
    k, v = attn._project_kv(x, B, T)
    q, k, v = attn._prepare_qkv(q, k, v)
    y = attn._manual_attention(q, k, v, T, T)

    v_norm = torch.nn.functional.normalize(v, p=2, dim=-1, eps=1e-6)
    dot = (y * v_norm).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-5)


def test_xsa_supports_gqa_value_repetition():
    torch.manual_seed(0)
    cfg = tiny_config(n_head=4, n_kv_head=2)
    attn = get_attention(cfg)
    x = torch.randn(2, 5, cfg.n_embd, requires_grad=True)
    y = attn(x)
    assert y.shape == (2, 5, cfg.n_embd)
    y.square().mean().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_gpt_with_xsa_forward_backward():
    torch.manual_seed(0)
    model = GPT(tiny_config())
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, 64)
    assert loss is not None
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
