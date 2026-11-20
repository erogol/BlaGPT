"""Tests for F107b: the 3 Latent Lattice fixes (arXiv:2611.21608).

Fix 1: FiLM on q/k after the per-head QK RMSNorm (before RoPE).
Fix 2: all Lattice params in separate AdamW groups, not Aurora.
Fix 3: depth-wise readout stream (vstream, Sec. 3.2.2).
"""
import json
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

from optimizers import get_optimizer  # noqa: E402

CFG_DIR = ROOT / "ar" / "full_train" / "configs"
BASE_SHA = "8341109c"  # origin/main with the F107 code (pod clones from GitHub)
AURORA_ARGS = {"weight_decay": 0.025, "mu": 0.95, "nesterov": True,
               "pp_iterations": 2, "pp_beta": 0.5}
LAT = {"lattice_enabled": True, "lattice_layers": [1, 2, 3, 4], "lattice_dr": 32,
       "lattice_inj": True}
FIXES = {"lattice_film_after_qknorm": True, "lattice_adamw": True,
         "lattice_vstream": True}


def _mk_cfg(hyper=False, **kw):
    cfg = GPTConfig()
    for k, v in json.load(open(CFG_DIR / "base_x5.json")).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv_head = 6, 128, 4, 4
    cfg.mlp_expand, cfg.block_size, cfg.vocab_size, cfg.engram_vocab_mult = 2, 64, 512, 2
    cfg.use_hyper_connections = hyper
    if hyper:
        cfg.use_attn_res = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required (Primer MLP init hardcodes cuda)")
    return "cuda"


def _build(cfg, device, seed=0):
    torch.manual_seed(seed)
    return GPT(cfg).to(device)


def _logits(m, x):
    m.eval()
    with torch.no_grad():
        out = m(x)
    return out[0] if isinstance(out, tuple) else out


def _randomize(m, std=0.02):
    """Give zero-init output projections and FiLM/MLP heads small values."""
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("c_proj") or n.endswith("c_proj.weight") or (
                "lattice" in n and any(h in n for h in ("film_", "mlp_beta", "w_e_"))
            ):
                p.normal_(std=std)


@pytest.mark.parametrize("hyper", [False, True])
@pytest.mark.parametrize("lattice", [False, True])
def test_fixes_off_bit_identical_to_f107(device, hyper, lattice, tmp_path):
    """Fix flags off: params, logits and loss equal the F107 code (BASE_SHA),
    for the plain model and for the F107 lattice config."""
    tar = tmp_path / "base.tar"
    r = subprocess.run(["git", "-C", str(ROOT), "archive", BASE_SHA, "bla_gpt", "-o", str(tar)])
    if r.returncode != 0:
        pytest.skip(f"git archive {BASE_SHA} not available")
    base = tmp_path / "base"
    with tarfile.open(tar) as t:
        t.extractall(base)
    assert (base / "bla_gpt" / "lattice.py").exists()
    cfg = json.load(open(CFG_DIR / "base_x5.json"))
    if lattice:
        cfg.update(LAT)
    cfg_json = tmp_path / "cfg.json"
    cfg_json.write_text(json.dumps(cfg))
    helper = ROOT / "tests" / "_lattice_dump.py"
    outs = []
    for tree in (base, ROOT):
        out = tmp_path / f"{tree.name}.pt"
        subprocess.run([sys.executable, str(helper), str(tree), str(out),
                        "1" if hyper else "0", str(cfg_json)], check=True)
        outs.append(torch.load(out, weights_only=False))
    a, b = outs
    assert a["sd"].keys() == b["sd"].keys()
    assert any("lattice" in k for k in a["sd"]) == lattice
    for k in a["sd"]:
        assert torch.equal(a["sd"][k], b["sd"][k]), k
    assert torch.equal(a["logits"], b["logits"])
    la = a["loss"]["total"] if isinstance(a["loss"], dict) else a["loss"]
    lb = b["loss"]["total"] if isinstance(b["loss"], dict) else b["loss"]
    assert torch.equal(la, lb)


def test_f107b_configs():
    base = json.load(open(CFG_DIR / "base_x5.json"))
    for name, seed in (("F107b_e5.json", 1337), ("F107b_e5_s2.json", 2)):
        c = json.load(open(CFG_DIR / name))
        assert c["lattice_layers"] == [2, 3, 4, 5, 6, 7]
        assert c["lattice_dr"] == 256 and c["lattice_inj"] and c["lattice_enabled"]
        assert all(c[k] for k in FIXES) and c["lattice_lr"] == 1e-3
        assert c["seed"] == seed
        diff = {k for k in c if base.get(k) != c[k]}
        assert diff <= set(LAT) | set(FIXES) | {"lattice_lr", "lattice_wd", "seed"}


@pytest.mark.parametrize("hyper", [False, True])
def test_all_fixes_zero_init_matches_base(device, hyper):
    """Step 0 with all fixes on: same logits as the model without Lattice."""
    m_off = _build(_mk_cfg(hyper), device)
    m_on = _build(_mk_cfg(hyper, **LAT, **FIXES), device)
    missing = m_on.load_state_dict(m_off.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all("lattice" in k for k in missing.missing_keys)
    x = torch.randint(0, 512, (2, 32), device=device)
    assert (_logits(m_off, x) - _logits(m_on, x)).abs().max().item() < 1e-5


@pytest.mark.parametrize("hyper", [False, True])
def test_vstream_init_identity(device, hyper):
    """vstream at init (W_up = 0): same readouts and logits as without it,
    also when the FiLM/MLP heads are non-zero."""
    m_a = _build(_mk_cfg(hyper, **LAT), device)
    _randomize(m_a)
    m_b = _build(_mk_cfg(hyper, **LAT, lattice_vstream=True), device)
    missing = m_b.load_state_dict(m_a.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert missing.missing_keys and all(".vs_" in k for k in missing.missing_keys)
    x = torch.randint(0, 512, (2, 32), device=device)
    la, lb = _logits(m_a, x), _logits(m_b, x)
    for k in m_a.lattice:
        assert torch.allclose(m_a._lattice_holder[k], m_b._lattice_holder[k], atol=1e-6)
    assert m_b._lattice_holder["s"].abs().max().item() == 0.0
    assert (la - lb).abs().max().item() < 1e-5
    # vs_up must learn: it gets a gradient at step 0.
    m_b.train()
    out = m_b(x, x)
    (out[1]["total"] if isinstance(out[1], dict) else out[1]).backward()
    assert all(m.vs_up.weight.grad.abs().sum() > 0 for m in m_b.lattice.values())


def test_film_after_qknorm_placement(device):
    """Fix 1: q/k pass the projection unchanged and get FiLM after q_norm/k_norm;
    v keeps its FiLM at the projection."""
    m = _build(_mk_cfg(**LAT, lattice_film_after_qknorm=True), device)
    _randomize(m)
    x = torch.randint(0, 512, (2, 32), device=device)
    _logits(m, x)  # fills the readout holder
    attn = m.transformer.h[1].attn
    mod = m.lattice["1"]
    r_hat = m._lattice_holder["1"]
    h = torch.randn(2, 32, 128, device=device)
    with torch.no_grad():
        q = attn._project_query(h, 2, 32)
        assert torch.equal(q, attn.q_proj(h).view(2, 32, 4, 32))
        k, v = attn._project_kv(h, 2, 32)
        kv = attn.kv_proj(h).view(2, 32, 2, 4, 32)
        assert torch.equal(k, kv[:, :, 0])
        assert not torch.equal(v, kv[:, :, 1])  # v FiLM still at the projection
        qn, kn = attn._apply_norm(q, k)
        assert torch.equal(qn, mod.film("q", attn.q_norm(q), r_hat))
        assert torch.equal(kn, mod.film("k", attn.k_norm(k), r_hat))
        assert not torch.equal(qn, attn.q_norm(q))


def test_film_before_qknorm_when_flag_off(device):
    m = _build(_mk_cfg(**LAT), device)
    _randomize(m)
    x = torch.randint(0, 512, (2, 32), device=device)
    _logits(m, x)
    attn = m.transformer.h[1].attn
    h = torch.randn(2, 32, 128, device=device)
    with torch.no_grad():
        q = attn._project_query(h, 2, 32)
        assert not torch.equal(q, attn.q_proj(h).view(2, 32, 4, 32))
        k, _ = attn._project_kv(h, 2, 32)
        qn, kn = attn._apply_norm(q, k)
        assert torch.equal(qn, attn.q_norm(q)) and torch.equal(kn, attn.k_norm(k))


def _group_of(opt, p):
    return next(g for g in opt.param_groups if any(q is p for q in g["params"]))


def test_lattice_params_all_in_adamw(device):
    """Fix 2: every Lattice param is in a Lattice AdamW group, none in Aurora."""
    m = _build(_mk_cfg(**LAT, **FIXES), device)
    opt = get_optimizer("Aurora", AURORA_ARGS, 0.03, m)
    lat = dict(m.lattice.named_parameters())
    assert len(lat) > 0
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen)) == len(list(m.parameters()))
    for name, p in lat.items():
        g = _group_of(opt, p)
        assert g.get("lattice") is True, name
        assert opt.state[p]["use_aurora"] is False, name
        assert g["lr"] == 1e-3
        nodecay = name.endswith(".bias") or name.endswith("a_log")
        assert g["adamw_wd"] == (0.0 if nodecay else 0.1), name
    for n in ("r0", "norm_in_w", "norm_out_w", "vs_norm_r_w", "vs_up.weight", "film_q.weight"):
        assert _group_of(opt, lat["1." + n])["adamw_wd"] == 0.1
    lat_ids = {id(p) for p in lat.values()}
    for p in m.parameters():
        if id(p) not in lat_ids:
            assert _group_of(opt, p).get("lattice") is None
    # One step: lattice params get AdamW state, no Aurora momentum.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5)
    x = torch.randint(0, 512, (2, 32), device=device)
    out = m(x, x)
    (out[1]["total"] if isinstance(out[1], dict) else out[1]).backward()
    opt.step()
    sched.step()
    for name, p in lat.items():
        if p.grad is not None:
            assert "moment1" in opt.state[p] and "momentum_buffer" not in opt.state[p], name
    assert _group_of(opt, lat["1.a_log"])["lr"] == pytest.approx(0.5e-3)


def test_lattice_adamw_off_keeps_f107_routing(device):
    m = _build(_mk_cfg(**LAT), device)
    opt = get_optimizer("Aurora", AURORA_ARGS, 0.03, m)
    assert len(opt.param_groups) == 1
    assert opt.state[m.lattice["1"].film_q.weight]["use_aurora"] is True
    assert opt.state[m.lattice["1"].a_log]["use_aurora"] is False
