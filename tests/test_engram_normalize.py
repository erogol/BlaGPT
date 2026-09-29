"""Targeted tests for E3: token normalization before n-gram hash (PR #375 style)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bla_gpt"))

from engram import MinimalEngram, build_token_norm_map  # noqa: E402

GPT2_VOCAB_SIZE = 50304  # padded GPT-2 vocab used by the current best config


def test_norm_map_single_bytes_are_unique():
    norm_map = build_token_norm_map(GPT2_VOCAB_SIZE)
    ids = torch.arange(256)
    assert torch.equal(norm_map[ids], ids), "ids 0-255 must map to themselves"


def test_norm_map_groups_shared_prefix_and_splits_different_prefix():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    norm_map = build_token_norm_map(GPT2_VOCAB_SIZE)

    # Find one pair (>=256) sharing a 2-byte prefix, and one pair that doesn't.
    prefix_to_ids: dict[bytes, list[int]] = {}
    for token_id in range(256, 2000):
        try:
            b = enc.decode_single_token_bytes(token_id)
        except Exception:
            continue
        if len(b) < 2:
            continue
        prefix_to_ids.setdefault(b[:2], []).append(token_id)

    same_prefix_pair = next(
        (ids for ids in prefix_to_ids.values() if len(ids) >= 2), None
    )
    assert same_prefix_pair is not None, "expected at least one shared-prefix pair in this range"
    a, b = same_prefix_pair[0], same_prefix_pair[1]
    assert norm_map[a].item() == norm_map[b].item(), "same 2-byte prefix must share a class"

    prefixes = list(prefix_to_ids.keys())
    different_prefix_pair = None
    for p1 in prefixes:
        for p2 in prefixes:
            if p1 != p2:
                different_prefix_pair = (prefix_to_ids[p1][0], prefix_to_ids[p2][0])
                break
        if different_prefix_pair:
            break
    assert different_prefix_pair is not None
    c, d = different_prefix_pair
    assert norm_map[c].item() != norm_map[d].item(), "different 2-byte prefixes must not share a class"


def test_minimal_engram_default_off_unchanged_behavior():
    # Exactly how existing callers construct it today: no vocab_size arg at all.
    torch.manual_seed(0)
    model = MinimalEngram(hidden_size=32, table_size=1000, ngram=3)
    assert model.normalize_tokens is False
    assert not hasattr(model, "norm_map")

    hidden = torch.randn(2, 16, 32)
    input_ids = torch.randint(0, 500, (2, 16))
    out = model(hidden, input_ids)
    assert out.shape == (2, 16, 32)


def test_minimal_engram_normalize_requires_vocab_size():
    with pytest.raises(ValueError):
        MinimalEngram(hidden_size=32, table_size=1000, ngram=3, normalize_tokens=True)


def test_minimal_engram_normalize_forward_runs():
    torch.manual_seed(0)
    model = MinimalEngram(
        hidden_size=32,
        table_size=1000,
        ngram=3,
        vocab_size=GPT2_VOCAB_SIZE,
        normalize_tokens=True,
    )
    assert model.normalize_tokens is True
    assert model.norm_map.shape == (GPT2_VOCAB_SIZE,)

    hidden = torch.randn(2, 16, 32)
    input_ids = torch.randint(0, GPT2_VOCAB_SIZE, (2, 16))
    out = model(hidden, input_ids)
    assert out.shape == (2, 16, 32)


def test_same_class_tokens_hash_identically():
    """Two different token ids in the same normalization class must hash to
    the same table row when they occupy the same relative n-gram position."""
    norm_map = build_token_norm_map(GPT2_VOCAB_SIZE)

    # Find two distinct ids sharing a class (skip the identity-mapped 0-255 range).
    class_to_ids: dict[int, list[int]] = {}
    for token_id in range(256, 3000):
        cls = norm_map[token_id].item()
        class_to_ids.setdefault(cls, []).append(token_id)
    pair = next((ids for ids in class_to_ids.values() if len(ids) >= 2), None)
    assert pair is not None
    tok_a, tok_b = pair[0], pair[1]

    model = MinimalEngram(
        hidden_size=32,
        table_size=1000,
        ngram=3,
        vocab_size=GPT2_VOCAB_SIZE,
        normalize_tokens=True,
    )

    seq_a = torch.tensor([[10, 20, tok_a]])
    seq_b = torch.tensor([[10, 20, tok_b]])

    normed_a = model.norm_map[seq_a]
    normed_b = model.norm_map[seq_b]

    def compute_hash(ids: torch.Tensor) -> torch.Tensor:
        h = torch.zeros_like(ids)
        for k in range(model.ngram):
            shifted = torch.nn.functional.pad(ids, (k, 0), value=model.pad_id)[:, : ids.shape[1]]
            h = h ^ (shifted * model.multipliers[k])
        return h % model.table_size

    hash_a = compute_hash(normed_a)
    hash_b = compute_hash(normed_b)
    assert torch.equal(hash_a, hash_b), "same-class tokens at the same position must hash identically"
