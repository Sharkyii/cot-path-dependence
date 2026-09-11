from early_stop.datasets import Problem
from early_stop.extraction import MockBackend
from early_stop.grading import accuracy, extraction_failure_rate, forced_extract_and_grade, grade


def test_grade_math_correct_boxed():
    problem = Problem(problem_id="math500:0", domain="math500", question="q", gold_answer="12")
    result = grade(problem, r"Working through it... the answer is \boxed{12}.")
    assert result.correct is True
    assert result.extraction_method == "boxed"
    assert result.extraction_failed is False


def test_grade_math_incorrect_boxed():
    problem = Problem(problem_id="math500:0", domain="math500", question="q", gold_answer="12")
    result = grade(problem, r"the answer is \boxed{13}.")
    assert result.correct is False


def test_grade_math_extraction_failure_counts_as_incorrect_not_error():
    problem = Problem(problem_id="math500:0", domain="math500", question="q", gold_answer="12")
    result = grade(problem, "I really don't know how to solve this.")
    assert result.correct is False
    assert result.extraction_failed is True


def test_grade_gsm8k_fraction_equivalence():
    problem = Problem(problem_id="gsm8k:0", domain="gsm8k", question="q", gold_answer="1/2")
    result = grade(problem, r"So the final fraction is \boxed{\frac{1}{2}}.")
    assert result.correct is True


def test_grade_commonsenseqa():
    problem = Problem(
        problem_id="commonsenseqa:0",
        domain="commonsenseqa",
        question="q",
        gold_answer="C",
        choices={"A": "x", "B": "y", "C": "z", "D": "w", "E": "v"},
    )
    result = grade(problem, "Weighing the options, I'll go with (C).")
    assert result.correct is True


def test_grade_strategyqa():
    problem = Problem(problem_id="strategyqa:0", domain="strategyqa", question="q", gold_answer="no")
    result = grade(problem, "After considering both sides, the answer is no.")
    assert result.correct is True


def test_grade_unknown_domain_raises():
    import pytest

    problem = Problem(problem_id="x:0", domain="not_a_real_domain", question="q", gold_answer="1")
    with pytest.raises(ValueError):
        grade(problem, "whatever")


def test_grading_symmetric_between_full_and_truncated_trace():
    # This is the bias-control property from project plan §8: the SAME
    # grading path must fire whether the text came from a full CoT or a
    # forced-extraction continuation after early stopping.
    problem = Problem(problem_id="math500:0", domain="math500", question="q", gold_answer="7")
    full_trace_text = r"Lots of reasoning here... eventually \boxed{7}."
    forced_extraction_text = r"\n\nTherefore, the final answer is: \boxed{7}"
    assert grade(problem, full_trace_text).correct is True
    assert grade(problem, forced_extraction_text).correct is True


def test_accuracy_and_extraction_failure_rate_aggregate():
    problem = Problem(problem_id="math500:0", domain="math500", question="q", gold_answer="1")
    results = [
        grade(problem, r"\boxed{1}"),  # correct
        grade(problem, r"\boxed{2}"),  # wrong but extracted
        grade(problem, "no idea"),  # extraction failure
    ]
    assert accuracy(results) == 1 / 3
    assert extraction_failure_rate(results) == 1 / 3


def test_accuracy_empty_list_is_zero_not_error():
    assert accuracy([]) == 0.0
    assert extraction_failure_rate([]) == 0.0


def test_forced_extract_and_grade_uses_domain_suffix_and_primes_boxed():
    # Math suffix primes an open \boxed{ -- the model only needs to supply
    # the content and closing brace.
    problem = Problem(problem_id="math500:0", domain="math500", question="q", gold_answer="7")
    backend = MockBackend(default="7}.")
    fx, result = forced_extract_and_grade(backend, problem, "some reasoning trace so far", step_index=2)
    assert fx.prompt.endswith("\\boxed{")
    assert result.correct is True


def test_forced_extract_and_grade_boxed_priming_requires_full_text():
    # Regression guard for a real bug found via the smoke-test script: the
    # math suffix primes an OPEN \boxed{ that lives in the PROMPT, not the
    # completion. A parser that only sees the completion has no \boxed{ to
    # brace-match against, so it falls back to a last-number regex -- which
    # silently mis-grades non-numeric answers (grabs "2" out of "(3\pi/2)",
    # or fails outright on text answers). Confirm grading on completion
    # alone reproduces the bug, and forced_extract_and_grade (full_text)
    # avoids it.
    problem = Problem(problem_id="math500:0", domain="math500", question="q", gold_answer="(3\\pi/2)")
    backend = MockBackend(default="(3\\pi/2)}.")
    fx, result = forced_extract_and_grade(backend, problem, "reasoning trace", step_index=0)

    completion_only_result = grade(problem, fx.completion)
    assert completion_only_result.correct is False  # demonstrates the bug that full_text grading avoids
    assert result.correct is True


def test_forced_extract_and_grade_commonsenseqa_suffix():
    problem = Problem(
        problem_id="commonsenseqa:0",
        domain="commonsenseqa",
        question="q",
        gold_answer="B",
        choices={"A": "x", "B": "y"},
    )
    backend = MockBackend(default="B) because it fits.")
    fx, result = forced_extract_and_grade(backend, problem, "reasoning trace", step_index=0)
    assert fx.prompt.endswith("(")
    assert result.correct is True


def test_forced_extract_and_grade_strategyqa_suffix():
    problem = Problem(problem_id="strategyqa:0", domain="strategyqa", question="q", gold_answer="no")
    backend = MockBackend(default=" no.")
    fx, result = forced_extract_and_grade(backend, problem, "reasoning trace", step_index=0)
    assert result.correct is True
