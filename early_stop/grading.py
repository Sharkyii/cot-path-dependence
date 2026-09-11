"""Uniform grading, applied identically to full-CoT and early-stopped traces.

The whole point of this module existing separately is that there is exactly
ONE grading code path per domain, used for every condition (full, truncated,
forced-extraction). If early-stopped traces were graded by different logic
than full traces, any accuracy gap could be an artifact of grading, not a
real effect of stopping early -- see project plan §8.
"""
from __future__ import annotations

from dataclasses import dataclass

from early_stop.datasets import Problem
from early_stop.extraction import ForcedExtractionResult, ModelBackend, forced_extract, suffix_for_domain
from early_stop.parsers import (
    ParseResult,
    math_answers_equal,
    parse_commonsenseqa_answer,
    parse_math_answer,
    parse_strategyqa_answer,
)

MATH_DOMAINS = {"math500", "math500_all", "gsm8k"}
MC_DOMAINS = {"commonsenseqa"}
YES_NO_DOMAINS = {"strategyqa"}


@dataclass
class GradeResult:
    problem_id: str
    domain: str
    correct: bool
    parsed_answer: str | None
    gold_answer: str
    extraction_method: str  # from ParseResult.method -- lets us report
    # extraction-failure rate per method/domain, per project plan §8
    extraction_failed: bool


def _parse_for_domain(problem: Problem, text: str) -> ParseResult:
    if problem.domain in MATH_DOMAINS:
        return parse_math_answer(text)
    if problem.domain in MC_DOMAINS:
        num_choices = len(problem.choices) if problem.choices else 5
        return parse_commonsenseqa_answer(text, num_choices=num_choices)
    if problem.domain in YES_NO_DOMAINS:
        return parse_strategyqa_answer(text)
    raise ValueError(f"No grading rule registered for domain {problem.domain!r}")


def _answers_equal(problem: Problem, parsed: str | None, gold: str) -> bool:
    if parsed is None:
        return False
    if problem.domain in MATH_DOMAINS:
        return math_answers_equal(parsed, gold)
    # MC / yes-no: exact normalized string match
    return parsed.strip().upper() == gold.strip().upper()


def grade(problem: Problem, model_output_text: str) -> GradeResult:
    """Grade one model output against its problem's gold answer.

    `model_output_text` should be exactly the text the forced-extraction
    suffix produced (or the full natural completion for the full-CoT
    baseline) -- callers must not pre-process or truncate it differently
    per condition.
    """
    parse_result = _parse_for_domain(problem, model_output_text)
    failed = parse_result.answer is None
    correct = (not failed) and _answers_equal(problem, parse_result.answer, problem.gold_answer)
    return GradeResult(
        problem_id=problem.problem_id,
        domain=problem.domain,
        correct=correct,
        parsed_answer=parse_result.answer,
        gold_answer=problem.gold_answer,
        extraction_method=parse_result.method,
        extraction_failed=failed,
    )


def forced_extract_and_grade(
    backend: ModelBackend,
    problem: Problem,
    prefix: str,
    step_index: int,
    max_new_tokens: int = 32,
) -> tuple[ForcedExtractionResult, GradeResult]:
    """Do a forced-extraction at this stop boundary and grade it, using the
    domain-correct suffix and grading on the FULL accumulated text (prefix +
    suffix + completion) rather than the completion alone.

    Grading on the completion alone is a trap: the math suffix primes an
    open \\boxed{ in the prompt so the model only completes the content and
    closing brace, so the boxed marker itself lives in the prompt, not the
    completion. Grading on completion-only silently breaks non-numeric
    answers (see extraction.suffix_for_domain docstring). Always go through
    this function for forced-extraction grading rather than calling
    forced_extract() and grade() separately, so this can't regress.
    """
    suffix = suffix_for_domain(problem.domain)
    fx = forced_extract(backend, prefix, step_index, max_new_tokens=max_new_tokens, suffix=suffix)
    result = grade(problem, fx.full_text)
    return fx, result


def extraction_failure_rate(results: list[GradeResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.extraction_failed) / len(results)


def accuracy(results: list[GradeResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)
