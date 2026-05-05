import torch
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SPEC = spec_from_file_location("blagpt_model", Path(__file__).resolve().parents[1] / "bla_gpt" / "bla_gpt.py")
_BLA = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BLA)
GPT = _BLA.GPT
GPTConfig = _BLA.GPTConfig

def tiny_hyper_config():
    config = GPTConfig()
    config.block_size = 8
    config.vocab_size = 64
    config.n_layer = 2
    config.n_head = 2
    config.n_kv_head = 2
    config.n_embd = 16
    config.dropout = 0.0
    config.bias = True
    config.pos_encoding = "none"
    config.activation = "gelu"
    config.norm_layer = "layernorm"
    config.tie_embed_weights = False
    config.zero_init_proj_layers = False
    config.use_engram = False
    config.use_hyper_connections = True
    config.hyper_num_streams = 4
    config.hyper_dynamic = True
    return config

def test_hyper_connections_forward_backward_shapes():
    torch.manual_seed(0)
    model = GPT(tiny_hyper_config())
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 8, 64)
    assert loss is not None
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

def test_static_hyper_parameters_skip_weight_decay():
    model = GPT(tiny_hyper_config())
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    decay_ids = {id(p) for p in optimizer.param_groups[0]["params"]}
    nodecay_ids = {id(p) for p in optimizer.param_groups[1]["params"]}

    static_params = [
        p for n, p in model.named_parameters()
        if n.endswith((".B", ".A_m", ".A_r"))
    ]
    assert static_params
    assert all(id(p) in nodecay_ids for p in static_params)
    assert all(id(p) not in decay_ids for p in static_params)

def test_dynamic_hyper_starts_equivalent_to_static():
    torch.manual_seed(0)
    cfg_dyn = tiny_hyper_config()
    cfg_dyn.hyper_dynamic = True
    cfg_static = tiny_hyper_config()
    cfg_static.hyper_dynamic = False

    dyn = _BLA.HyperConnection(cfg_dyn, depth=1)
    static = _BLA.HyperConnection(cfg_static, depth=1)
    with torch.no_grad():
        static.B.copy_(dyn.B)
        static.A_m.copy_(dyn.A_m)
        static.A_r.copy_(dyn.A_r)

    streams = torch.randn(2, 3, cfg_dyn.hyper_num_streams, cfg_dyn.n_embd)
    ln = torch.nn.Identity()
    branch = torch.nn.Identity()

    out_dyn = dyn(streams, branch, ln)
    out_static = static(streams, branch, ln)
    assert torch.allclose(out_dyn, out_static, atol=1e-6)
