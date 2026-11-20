"""Tests for F108: Multi-Branch Transformer (arXiv:2611.19800) inside E7b."""
import json
import math
import subprocess
import sys
import tarfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bla_gpt"))

_SPEC = spec_from_file_location("blagpt_model", ROOT / "bla_gpt" / "bla_gpt.py")
blagpt_model = module_from_spec(_SPEC)
sys.modules.setdefault("blagpt_model", blagpt_model)
_SPEC.loader.exec_module(blagpt_model)
GPT = blagpt_model.GPT
GPTConfig = blagpt_model.GPTConfig

BASE_SHA = "894bc45d"  # BlaGPT main before F108
CFG_JSON = ROOT / "ar" / "full_train" / "configs" / "base_x5.json"


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def _mk_cfg(**kw):
    """Small E7b config, n_layer 10 as in the full run."""
    cfg = GPTConfig()
    for k, v in json.load(open(CFG_JSON)).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_embd, cfg.n_head, cfg.n_kv_head = 128, 4, 4
    cfg.mlp_expand, cfg.block_size, cfg.vocab_size, cfg.engram_vocab_mult = 2, 64, 512, 2
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_flag_off_bit_identical_to_base(device, tmp_path):
    """mbt_branches=0: same params, logits and loss as the code at 894bc45d."""
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_SHA, "bla_gpt", "-o", str(tar)])
    assert r.returncode == 0, f"git archive {BASE_SHA} failed"
    base = tmp_path / "base"
    with tarfile.open(tar) as t:
        t.extractall(base)
    helper = ROOT / "tests" / "_mbt_dump.py"
    outs = []
    for tree in (base, ROOT):
        out = tmp_path / f"{tree.name}.pt"
        subprocess.run([sys.executable, str(helper), str(tree), str(out), str(CFG_JSON)], check=True)
        outs.append(torch.load(out, weights_only=False))
    a, b = outs
    assert a["sd"].keys() == b["sd"].keys()
    for k in a["sd"]:
        assert torch.equal(a["sd"][k], b["sd"][k]), k
    assert torch.equal(a["logits"], b["logits"])
    la = a["loss"]["total"] if isinstance(a["loss"], dict) else a["loss"]
    lb = b["loss"]["total"] if isinstance(b["loss"], dict) else b["loss"]
    assert torch.equal(la, lb)


def test_mbt_forward_backward(device):
    """3/(2+2)/3 layout: shapes, finite loss, nonzero W_F gradient, dropped paths."""
    torch.manual_seed(0)
    m = GPT(_mk_cfg(mbt_branches=2, mbt_prelude=3, mbt_branch_depth=2)).to(device)
    W = m.mbt_fuse_proj.weight
    assert W.shape == (128, 256) and m.mbt_fuse_proj.bias is None
    assert len(m.mbt_fuse_norms) == 2
    std = 0.02 / math.sqrt(2 * 10)
    assert abs(W.std().item() - std) < 0.1 * std  # A.8 init survives the global init
    # E7b zero-inits the output projections, so every block is the identity
    # at step 0. Give them small values so the two branches differ.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("c_proj") or n.endswith("c_proj.weight"):
                p.normal_(std=0.02)
    m.train()
    x = torch.randint(0, 512, (2, 32), device=device)
    logits, loss = m(x, x)
    loss = loss["total"] if isinstance(loss, dict) else loss
    assert logits.shape == (2, 32, 512)
    assert torch.isfinite(loss)
    loss.backward()
    assert W.grad is not None and W.grad.abs().sum() > 0
    for g in m.mbt_fuse_norms:
        assert g.weight.grad.abs().sum() > 0
    for i in range(3, 7):  # branch A = h[3..4], branch B = h[5..6]
        g = m.transformer.h[i].mlp.c_proj.grad
        assert g is not None and g.abs().sum() > 0, i
    # U-net pairs 4->5 and 3->6 are dropped; 2->7, 1->8, 0->9 stay.
    sw = m.skip_weights.grad
    assert torch.all(sw[:2] == 0) and torch.all(sw[2:] != 0)
    # AttnRes rows 4-6 are unused; row 3 (branch input) and 7-10 are used.
    ar = m.attn_res_w.grad
    assert torch.all(ar[4:7] == 0)
    assert ar[3].abs().sum() > 0 and all(ar[i].abs().sum() > 0 for i in range(7, 11))
