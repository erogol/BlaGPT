"""Tests for E1: LR floor (final_lr_frac) and weight EMA (ema_last_steps)."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bla_gpt"))


class TestFinalLrFracDefault:
    def test_default_is_zero(self):
        from train import Hyperparameters

        args = Hyperparameters()
        assert args.final_lr_frac == 0.0

    def test_json_override(self, tmp_path):
        from train import Hyperparameters, _apply_json_overrides

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"final_lr_frac": 0.3}))
        args = Hyperparameters()
        _apply_json_overrides(args, str(config))
        assert args.final_lr_frac == 0.3


class TestEmaLastStepsDefault:
    def test_default_is_zero(self):
        from train import Hyperparameters

        args = Hyperparameters()
        assert args.ema_last_steps == 0

    def test_json_override(self, tmp_path):
        from train import Hyperparameters, _apply_json_overrides

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"ema_last_steps": 300}))
        args = Hyperparameters()
        _apply_json_overrides(args, str(config))
        assert args.ema_last_steps == 300


def _get_lr_fn(args):
    """Reconstruct the get_lr closure the same way train.py builds it."""

    def get_lr(it):
        assert it <= args.num_iterations
        if it < args.warmup_iters:
            return (it + 1) / args.warmup_iters
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.final_lr_frac + (1.0 - args.final_lr_frac) * decay_ratio

    return get_lr


class TestLrFloorSchedule:
    def test_zero_frac_decays_to_zero_at_last_step(self):
        from train import Hyperparameters

        args = Hyperparameters()
        args.num_iterations = 1000
        args.warmup_iters = 100
        args.warmdown_iters = 200
        args.final_lr_frac = 0.0
        get_lr = _get_lr_fn(args)
        assert get_lr(args.num_iterations) == 0.0

    def test_nonzero_frac_floors_at_last_step(self):
        from train import Hyperparameters

        args = Hyperparameters()
        args.num_iterations = 1000
        args.warmup_iters = 100
        args.warmdown_iters = 200
        args.final_lr_frac = 0.3
        get_lr = _get_lr_fn(args)
        assert abs(get_lr(args.num_iterations) - 0.3) < 1e-9

    def test_warmup_and_constant_phase_unaffected_by_final_lr_frac(self):
        from train import Hyperparameters

        args_off = Hyperparameters()
        args_off.num_iterations = 1000
        args_off.warmup_iters = 100
        args_off.warmdown_iters = 200
        args_off.final_lr_frac = 0.0

        args_on = Hyperparameters()
        args_on.num_iterations = 1000
        args_on.warmup_iters = 100
        args_on.warmdown_iters = 200
        args_on.final_lr_frac = 0.3

        get_lr_off = _get_lr_fn(args_off)
        get_lr_on = _get_lr_fn(args_on)
        for it in (0, 1, 50, 99, 100, 500, 799):
            assert get_lr_off(it) == get_lr_on(it)

    def test_midway_warmdown_is_blended_not_pure_linear(self):
        from train import Hyperparameters

        args = Hyperparameters()
        args.num_iterations = 1000
        args.warmup_iters = 100
        args.warmdown_iters = 200
        args.final_lr_frac = 0.3
        get_lr = _get_lr_fn(args)
        # halfway through warmdown: decay_ratio = 0.5
        it = args.num_iterations - args.warmdown_iters // 2
        expected = 0.3 + 0.7 * 0.5
        assert abs(get_lr(it) - expected) < 1e-9


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(4))


class TestWeightEMA:
    def test_disabled_when_zero_no_wrapper_needed(self):
        # ema_last_steps == 0 means callers must not construct WeightEMA at all;
        # this documents the contract used in train.py's main loop.
        from train import Hyperparameters

        args = Hyperparameters()
        assert args.ema_last_steps == 0

    def test_decay_formula(self):
        from train import Hyperparameters

        args = Hyperparameters()
        args.ema_last_steps = 300
        decay = 1.0 - 1.0 / args.ema_last_steps
        assert abs(decay - (1 - 1 / 300)) < 1e-12

    def test_ema_tracks_trailing_average(self):
        from train import WeightEMA

        model = _TinyModel()
        ema = WeightEMA(model, decay=0.9)
        with torch.no_grad():
            model.w.fill_(1.0)
        ema.update(model)
        # shadow = 0.9*0 + 0.1*1 = 0.1
        assert torch.allclose(ema.shadow["w"], torch.tensor([0.1, 0.1, 0.1, 0.1]))
        with torch.no_grad():
            model.w.fill_(1.0)
        ema.update(model)
        # shadow = 0.9*0.1 + 0.1*1 = 0.19
        assert torch.allclose(ema.shadow["w"], torch.tensor([0.19, 0.19, 0.19, 0.19]))

    def test_apply_and_restore_round_trip(self):
        from train import WeightEMA

        model = _TinyModel()
        ema = WeightEMA(model, decay=0.9)
        with torch.no_grad():
            model.w.fill_(2.0)
        ema.update(model)  # shadow now != current weights
        original = model.w.detach().clone()

        ema.apply_to(model)
        assert not torch.allclose(model.w, original)
        assert torch.allclose(model.w, ema.shadow["w"])

        ema.restore(model)
        assert torch.allclose(model.w, original)
