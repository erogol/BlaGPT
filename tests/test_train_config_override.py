"""Tests for JSON config → Hyperparameters override in train.py."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bla_gpt"))

import pytest


class TestConfigOverride:
    def test_learning_rate_default(self):
        from train import Hyperparameters
        args = Hyperparameters()
        assert args.learning_rate == 0.001

    def test_learning_rate_json_override(self, tmp_path):
        from train import Hyperparameters, _apply_json_overrides
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"learning_rate": 0.0014}))

        args = Hyperparameters()
        _apply_json_overrides(args, str(config))

        assert args.learning_rate == 0.0014

    def test_other_hyperparams_unchanged_by_unrelated_key(self, tmp_path):
        from train import Hyperparameters, _apply_json_overrides
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"learning_rate": 0.002, "not_a_field": 999}))

        args = Hyperparameters()
        _apply_json_overrides(args, str(config))

        assert args.learning_rate == 0.002
        assert args.warmup_iters == 250  # untouched default

    def test_multiple_hyperparams_overridden(self, tmp_path):
        from train import Hyperparameters, _apply_json_overrides
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"learning_rate": 0.0014, "warmup_iters": 500}))

        args = Hyperparameters()
        _apply_json_overrides(args, str(config))

        assert args.learning_rate == 0.0014
        assert args.warmup_iters == 500
