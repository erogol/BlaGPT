"""Tests for F89: Block AttnRes routing over block-level residual sums."""
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model_f89", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model_f89", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig


def _mk_cfg(block_size):
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
    cfg.use_attn_res = True
    cfg.attn_res_block_size = block_size
    cfg.use_unet_skips = False
    cfg.use_oasis_depth_softmax1 = False
    return cfg


if not torch.cuda.is_available():
    pytest.skip("CUDA required (Primer MLP init hardcodes cuda)", allow_module_level=True)
DEV = "cuda"


def _mk(block_size, seed=17):
    torch.manual_seed(seed)
    return GPT(_mk_cfg(block_size)).to(DEV)


def _fwd(m, seed=0):
    torch.manual_seed(seed)
    x = torch.randint(0, 512, (2, 64), device=DEV)
    y = torch.randint(0, 512, (2, 64), device=DEV)
    out = m(x, y)
    loss = out[1]
    return loss["total"] if isinstance(loss, dict) else loss


def test_block_attn_res_uses_no_new_parameters():
    full = _mk(0)
    block = _mk(2)
    assert sum(p.numel() for p in block.parameters()) == sum(p.numel() for p in full.parameters())
    assert block.config.attn_res_block_size == 2


def test_block_attn_res_changes_output_vs_full_attn_res():
    full = _mk(0)
    block = _mk(2)
    assert not torch.allclose(_fwd(full), _fwd(block))


def test_block_attn_res_source_count_is_block_level():
    m = _mk(2)
    seen = []
    orig = m._attn_res_route

    def spy(scores):
        seen.append(scores.size(0))
        return orig(scores)

    m._attn_res_route = spy
    _ = _fwd(m)
    # Four layers with block size 2: first block routes over current partial only;
    # second block routes over one completed block plus current partial; final route
    # sees the completed block-level summaries (plus current partial implementation detail).
    assert seen[:4] == [1, 1, 2, 2]
    assert max(seen) <= 3


def test_block_attn_res_grads_finite():
    m = _mk(2)
    loss = _fwd(m)
    loss.backward()
    assert m.attn_res_w.grad is not None
    assert torch.isfinite(m.attn_res_w.grad).all()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


def test_invalid_block_size_zero_is_full_attn_res_path():
    m = _mk(0)
    seen = []
    orig = m._attn_res_route

    def spy(scores):
        seen.append(scores.size(0))
        return orig(scores)

    m._attn_res_route = spy
    _ = _fwd(m)
    assert seen == [1, 2, 3, 4, 5]
