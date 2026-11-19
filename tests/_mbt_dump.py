"""Helper for test_mbt_f108: build a small flag-off E7b model (n_layer 10) from
one source tree with a fixed seed and save its state_dict, logits and loss.
Run as a subprocess so two trees never share sys.modules."""
import json
import sys
from pathlib import Path

import torch

tree, out, cfg_json = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(tree / "bla_gpt"))
from bla_gpt import GPT, GPTConfig  # noqa: E402

cfg = GPTConfig()
for k, v in json.load(open(cfg_json)).items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)
cfg.n_embd, cfg.n_head, cfg.n_kv_head = 128, 4, 4
cfg.mlp_expand, cfg.block_size, cfg.vocab_size, cfg.engram_vocab_mult = 2, 64, 512, 2
torch.manual_seed(0)
m = GPT(cfg).cuda().eval()
torch.manual_seed(1)
x = torch.randint(0, 512, (2, 32), device="cuda")
with torch.no_grad():
    o = m(x, x)
torch.save({"sd": {k: v.cpu() for k, v in m.state_dict().items()},
            "logits": o[0].cpu(), "loss": o[1]}, out)
