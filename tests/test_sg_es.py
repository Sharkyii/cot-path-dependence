"""Tests for SG-ES (Phase 4, low-compute plan) -- pure logic, no GPU.
Builds synthetic scored-prefix sequences to check the stopping rule
directly, the same way test_path_dependence.py tests matching/PDI logic
without a model.
"""
from early_stop.path_dependence import ObservableState, Prefix
from early_stop.sg_es import (
    forced_extraction_baseline,
    full_generation_answer,
    sg_es_decision,
)


def _prefix(step: int, total: int, answer: str, confidence: float, entropy: float = 1.0) -> Prefix:
    return Prefix(
        problem_id="p0", trajectory_id="t0", step_index=step, total_steps=total,
        state=ObservableState(answer=answer, confidence=confidence, entropy=entropy),
    )


def test_full_generation_uses_last_prefix():
    prefixes = [_prefix(i, 10, str(i), 0.5) for i in range(1, 11)]
    result = full_generation_answer(prefixes)
    assert result.answer == "10"
    assert result.stopped_at_step == 10
    assert result.triggered


def test_full_generation_empty_input():
    result = full_generation_answer([])
    assert result.answer is None
    assert not result.triggered


def test_forced_extraction_picks_nearest_to_fraction():
    prefixes = [_prefix(i, 10, str(i), 0.5) for i in range(1, 11)]
    result = forced_extraction_baseline(prefixes, fraction=0.5)
    assert result.stopped_at_step == 5
    assert result.answer == "5"


def test_sg_es_triggers_on_stable_confident_answer():
    # answer flips early, then settles on "42" with high confidence for 2 steps
    prefixes = [
        _prefix(1, 10, "1", 0.9),
        _prefix(2, 10, "7", 0.9),
        _prefix(3, 10, "42", 0.9),
        _prefix(4, 10, "42", 0.95),  # stable + confident here -> should stop
        _prefix(5, 10, "42", 0.95),
        _prefix(6, 10, "13", 0.9),  # would flip later if it ran further -- SG-ES shouldn't see this
    ]
    result = sg_es_decision(prefixes, confidence_threshold=0.7, stability_window=2)
    assert result.triggered
    assert result.answer == "42"
    assert result.stopped_at_step == 4


def test_sg_es_falls_through_when_never_confident():
    prefixes = [_prefix(i, 5, "same", 0.3) for i in range(1, 6)]  # stable but never confident
    result = sg_es_decision(prefixes, confidence_threshold=0.7, stability_window=2)
    assert not result.triggered
    assert result.answer == "same"  # falls through to full trace's answer
    assert result.stopped_at_step == 5


def test_sg_es_falls_through_when_never_stable():
    prefixes = [_prefix(i, 5, str(i), 0.9) for i in range(1, 6)]  # confident but always changing
    result = sg_es_decision(prefixes, confidence_threshold=0.7, stability_window=2)
    assert not result.triggered
    assert result.stopped_at_step == 5


def test_sg_es_never_worse_than_full_generation_answer():
    # By construction: whatever SG-ES falls through to must equal the full
    # trace's answer, so SG-ES's accuracy floor equals full generation's.
    prefixes = [_prefix(i, 5, str(i), 0.2) for i in range(1, 6)]
    sg = sg_es_decision(prefixes, confidence_threshold=0.9, stability_window=2)
    full = full_generation_answer(prefixes)
    assert sg.answer == full.answer


def test_sg_es_empty_input():
    result = sg_es_decision([])
    assert result.answer is None
    assert not result.triggered


def test_fraction_of_trace_used():
    prefixes = [_prefix(i, 10, "x", 0.9) for i in range(1, 11)]
    result = forced_extraction_baseline(prefixes, fraction=0.3)
    assert 0.0 < result.fraction_of_trace_used <= 1.0
