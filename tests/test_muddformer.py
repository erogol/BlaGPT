import torch
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SPEC = spec_from_file_location("blagpt_model", Path(__file__).resolve().parents[1] / "bla_gpt" / "bla_gpt.py")
_BLA = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BLA)
GPT = _BLA.GPT
GPTConfig = _BLA.GPTConfig


def tiny_config():
    config = GPTConfig()
    config.block_size = 8
    config.vocab_size = 64
    config.n_layer = 4
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
    return config


def test_muddformer_off_matches_baseline():
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.muddformer_mix = False
    model = GPT(cfg)
    idx = torch.randint(0, 64, (2, 8))
    logits_off, _ = model(idx, idx)

    torch.manual_seed(0)
    cfg2 = tiny_config()
    cfg2.muddformer_mix = True
    cfg2.muddformer_ref_layer = 0
    model2 = GPT(cfg2)
    logits_on_init, _ = model2(idx, idx)

    # Mix coefficients start at 0, so the two forward passes must match.
    assert torch.allclose(logits_off, logits_on_init, atol=1e-6)


def test_muddformer_nonzero_coef_changes_output():
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.muddformer_mix = False
    model_off = GPT(cfg)
    idx = torch.randint(0, 64, (2, 8))
    logits_off, _ = model_off(idx, idx)

    torch.manual_seed(0)
    cfg2 = tiny_config()
    cfg2.muddformer_mix = True
    cfg2.muddformer_ref_layer = 0
    model_on = GPT(cfg2)
    with torch.no_grad():
        model_on.muddformer_mix_coef.fill_(0.1)
    logits_on, _ = model_on(idx, idx)

    assert not torch.allclose(logits_off, logits_on, atol=1e-6)


def test_muddformer_nonzero_ref_layer_runs():
    torch.manual_seed(0)
    cfg = tiny_config()
    cfg.muddformer_mix = True
    cfg.muddformer_ref_layer = 2
    model = GPT(cfg)
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))

    logits, loss = model(idx, targets)

    assert logits.shape == (2, 8, 64)
    assert loss is not None
    loss.backward()
    assert model.muddformer_mix_coef.grad is not None
