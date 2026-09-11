"""Tests for the Phase 3 mechanistic-probe pieces (low-compute plan,
2026-09-11): MockScoredBackend.extract_hidden_state() and the probe
script's decisive/not-decisive labeling logic. No GPU needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from early_stop.backend import MockScoredBackend
from train_decisiveness_probe import _decisive


def test_mock_hidden_state_is_deterministic():
    backend = MockScoredBackend()
    a = backend.extract_hidden_state("some prefix text")
    b = backend.extract_hidden_state("some prefix text")
    assert a == b


def test_mock_hidden_state_differs_by_prefix():
    backend = MockScoredBackend()
    a = backend.extract_hidden_state("prefix one")
    b = backend.extract_hidden_state("prefix two")
    assert a != b


def test_mock_hidden_state_is_fixed_length_floats():
    backend = MockScoredBackend()
    vec = backend.extract_hidden_state("anything")
    assert len(vec) == 16
    assert all(isinstance(x, float) for x in vec)
    assert all(0.0 <= x <= 1.0 for x in vec)


def test_decisive_majority_real_answers():
    assert _decisive({"42": 30, "": 10}, threshold=0.5) == 1


def test_decisive_majority_empty():
    assert _decisive({"42": 10, "": 30}, threshold=0.5) == 0


def test_decisive_respects_custom_threshold():
    # 25% empty: decisive under the default 0.5 threshold, not decisive
    # under a stricter 0.2 threshold.
    counts = {"42": 30, "": 10}
    assert _decisive(counts, threshold=0.5) == 1
    assert _decisive(counts, threshold=0.2) == 0


def test_decisive_handles_empty_counts():
    assert _decisive({}, threshold=0.5) is None
