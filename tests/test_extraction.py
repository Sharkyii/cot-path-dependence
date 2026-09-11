from early_stop.cost import CostTracker
from early_stop.extraction import FORCED_ANSWER_SUFFIX, MockBackend, forced_extract


def test_forced_extract_appends_suffix_to_prompt():
    backend = MockBackend(default=" 42.")
    prefix = "Some partial reasoning trace here"
    result = forced_extract(backend, prefix, step_index=3)
    assert result.prompt == prefix + FORCED_ANSWER_SUFFIX
    assert result.completion == " 42."
    assert result.full_text == prefix + FORCED_ANSWER_SUFFIX + " 42."
    assert result.step_index == 3


def test_forced_extract_records_call_on_backend():
    backend = MockBackend()
    forced_extract(backend, "prefix text", step_index=0, max_new_tokens=16)
    assert len(backend.calls) == 1
    prompt, max_new_tokens = backend.calls[0]
    assert prompt.endswith(FORCED_ANSWER_SUFFIX)
    assert max_new_tokens == 16


def test_mock_backend_canned_completion_matches_by_substring():
    backend = MockBackend(canned_completions={"quadratic": r" \boxed{2, 3}"})
    result = forced_extract(backend, "This is a quadratic equation problem", step_index=1)
    assert result.completion == r" \boxed{2, 3}"


def test_mock_backend_falls_back_to_default_when_no_match():
    backend = MockBackend(canned_completions={"unrelated_key": " x"}, default=" fallback")
    result = forced_extract(backend, "totally different prefix", step_index=0)
    assert result.completion == " fallback"


def test_forced_extraction_cost_is_recorded_not_free():
    # Regression guard for the "for free" honesty requirement (project plan
    # §3.1, §10): every forced_extract call must be attributable to a cost
    # category by the caller. This test just verifies CostTracker's
    # bookkeeping API works the way the pilot/calibration loop will use it.
    tracker = CostTracker()
    backend = MockBackend(default=" 7.")
    prefix = "reasoning so far"
    result = forced_extract(backend, prefix, step_index=2, max_new_tokens=32)
    tracker.record(
        "forced_extraction",
        prompt_tokens=len(result.prompt.split()),
        completion_tokens=len(result.completion.split()),
    )
    assert tracker.calls["forced_extraction"] == 1
    assert tracker.total_tokens("forced_extraction") > 0
    assert tracker.total_tokens("main_generation") == 0
