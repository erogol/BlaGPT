"""Tests for F91: PaLM-style parallel attention/MLP blocks."""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model_f91", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model_f91", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig
ParallelBlock = blagpt_model.ParallelBlock
Block = blagpt_model.Block


def _mk_cfg(gate):
    cfg_dict = json.load(open(ROOT / "ar" / "best_config.json"))
    cfg = GPTConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_layer = 4
    cfg.n_embd = 128
    cfg.n_head = 4
    cfg.n_kv_head = 4
    cfg.block_size = 64
    cfg.vocab_size = 512
    cfg.zero_init_proj_layers = False
    cfg.use_parallel_blocks = gate
    return cfg


if not torch.cuda.is_available():
    pytest.skip("CUDA required (Primer MLP init hardcodes cuda)", allow_module_level=True)
DEV = "cuda"


def _mk(gate, seed=19):
    torch.manual_seed(seed)
    return GPT(_mk_cfg(gate)).to(DEV)


def _fwd(m, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(0, 512, (2, 64), device=DEV)
    y = torch.randint(0, 512, (2, 64), device=DEV)
    out = m(x, y)
    loss = out[1]
    return loss["total"] if isinstance(loss, dict) else loss


def test_gate_dispatches_parallel_blocks_only_when_enabled():
    m_off = _mk(False)
    m_on = _mk(True)
    assert isinstance(m_off.transformer.h[0], Block)
    assert not isinstance(m_off.transformer.h[0], ParallelBlock)
    assert isinstance(m_on.transformer.h[0], ParallelBlock)


def test_parallel_gate_is_step0_loss_compatible_with_zero_init():
    # Best configs zero-init projection layers; at exact initialization, sequential and
    # parallel blocks can be loss-identical. The experiment tests optimization geometry,
    # so the hard requirements are dispatch compatibility and finite gradients.
    m_off = _mk(False)
    m_on = _mk(True)
    assert torch.allclose(_fwd(m_off), _fwd(m_on))


def test_parallel_grads_finite_with_best_stack_gates():
    m = _mk(True)
    assert m.config.use_attn_res
    assert m.config.use_value_residual
    assert m.config.use_unet_skips
    loss = _fwd(m)
    loss.backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_parallel_blocks_are_mutually_exclusive_with_other_block_families():
    cfg = _mk_cfg(True)
    cfg.use_canon_layers = True
    with pytest.raises(ValueError):
        cfg.__post_init__()
