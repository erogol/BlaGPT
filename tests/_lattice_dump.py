"""Helper for test_lattice_f107: build a small E7b model from one source tree
with a fixed seed and save its state_dict and logits. Run as a subprocess so
two trees never share sys.modules."""
import json
import sys
from pathlib import Path

import torch

tree, out, hyper = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
sys.path.insert(0, str(tree / "bla_gpt"))
from bla_gpt import GPT, GPTConfig  # noqa: E402

cfg = GPTConfig()
base = json.load(open(Path(sys.argv[4])))
for k, v in base.items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)
cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv_head = 6, 128, 4, 4
cfg.mlp_expand, cfg.block_size, cfg.vocab_size, cfg.engram_vocab_mult = 2, 64, 512, 2
cfg.use_hyper_connections = hyper
if hyper:
    cfg.use_attn_res = False
torch.manual_seed(0)
m = GPT(cfg).cuda().eval()
torch.manual_seed(1)
x = torch.randint(0, 512, (2, 32), device="cuda")
with torch.no_grad():
    o = m(x, x)
torch.save({"sd": {k: v.cpu() for k, v in m.state_dict().items()},
            "logits": o[0].cpu(), "loss": o[1]}, out)
