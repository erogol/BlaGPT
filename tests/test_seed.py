"""Tests for the RNG seed flag in train.py (CPU only)."""

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bla_gpt"))

import torch

_SPEC = spec_from_file_location(
    "blagpt_model", Path(__file__).resolve().parents[1] / "bla_gpt" / "bla_gpt.py"
)
_BLA = module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BLA)
GPT = _BLA.GPT
GPTConfig = _BLA.GPTConfig


def _make_tiny_model():
    """Build a small GPT model for fast CPU weight-init comparisons."""
    config = GPTConfig()
    config.block_size = 32
    config.vocab_size = 64
    config.n_layer = 2
    config.n_head = 2
    config.n_kv_head = 2
    config.n_embd = 32
    return GPT(config)


class TestHyperparametersSeedField:
    def test_seed_default_is_none(self):
        from train import Hyperparameters

        args = Hyperparameters()
        assert args.seed is None

    def test_seed_json_override(self, tmp_path):
        import json

        from train import Hyperparameters, _apply_json_overrides

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"seed": 1234}))

        args = Hyperparameters()
        _apply_json_overrides(args, str(config))

        assert args.seed == 1234


class TestSetSeed:
    def test_seed_set_twice_gives_identical_initial_weights(self):
        from train import set_seed

        set_seed(42)
        model_a = _make_tiny_model()

        set_seed(42)
        model_b = _make_tiny_model()

        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        assert state_a.keys() == state_b.keys()
        for key in state_a:
            assert torch.equal(state_a[key], state_b[key]), f"mismatch in {key}"

    def test_different_seeds_give_different_initial_weights(self):
        from train import set_seed

        set_seed(1)
        model_a = _make_tiny_model()

        set_seed(2)
        model_b = _make_tiny_model()

        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        differs = any(
            not torch.equal(state_a[key], state_b[key]) for key in state_a
        )
        assert differs

    def test_seed_none_is_a_no_op(self):
        """set_seed(None) must not touch any RNG state (unchanged behavior)."""
        from train import set_seed

        py_state_before = __import__("random").getstate()
        torch_state_before = torch.get_rng_state()

        set_seed(None)

        assert __import__("random").getstate() == py_state_before
        assert torch.equal(torch.get_rng_state(), torch_state_before)
