"""Forced-answer extraction primitive (project plan §3.1).

At any candidate stop boundary, this appends a fixed suffix to the
trace-so-far and asks the model to produce a short continuation. This is a
REAL, COSTED operation -- callers must record it via CostTracker (see
cost.py), not treat it as free introspection. Thresholds are calibrated
against the accuracy of THIS forced answer, never a hypothetical natural one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Generic suffix, kept for domains/tests that don't need a domain-specific
# form. Prefer suffix_for_domain() in real pipeline code: math needs the
# \boxed{ "priming" trick below to be reliably parseable (see docstring on
# suffix_for_domain), and MC/yes-no domains read more naturally with their
# own phrasing.
FORCED_ANSWER_SUFFIX = "\n\nTherefore, the final answer is:"

# Math: "prime" an open \boxed{ so the model only has to complete the
# content and closing brace. Grading MUST run on the full accumulated text
# (prefix + suffix + completion), not the completion alone -- the opening
# brace lives in the prompt, and early testing showed that grading on the
# completion in isolation makes any non-numeric answer (LaTeX expressions,
# text answers) silently fail, because the completion alone often has no
# \boxed{} marker for parse_math_answer's brace-matcher to find, and its
# last-number fallback then grabs the wrong token (e.g. the "2" out of
# "3\pi/2"). See forced_extract_and_grade() in grading.py.
_MATH_SUFFIX = "\n\nTherefore, the final answer is \\boxed{"
_COMMONSENSEQA_SUFFIX = "\n\nTherefore, the final answer is ("
_STRATEGYQA_SUFFIX = "\n\nTherefore, the final answer (yes or no) is:"

FORCED_ANSWER_SUFFIXES = {
    "math500": _MATH_SUFFIX,
    "math500_all": _MATH_SUFFIX,
    "gsm8k": _MATH_SUFFIX,
    "commonsenseqa": _COMMONSENSEQA_SUFFIX,
    "strategyqa": _STRATEGYQA_SUFFIX,
}


def suffix_for_domain(domain: str) -> str:
    if domain not in FORCED_ANSWER_SUFFIXES:
        raise ValueError(f"No forced-answer suffix registered for domain {domain!r}")
    return FORCED_ANSWER_SUFFIXES[domain]


class ModelBackend(Protocol):
    """Minimal interface forced-extraction needs from a model backend.

    Implementations: HFBackend (transformers/vLLM, GPU) for real runs,
    MockBackend (below) for offline pipeline testing on the laptop.
    """

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


@dataclass
class ForcedExtractionResult:
    step_index: int  # which candidate stop boundary this was taken at
    prompt: str  # prefix + suffix, exactly what was sent to the model
    completion: str  # raw model output after the suffix
    full_text: str  # prompt + completion, ready for grading.grade()


def forced_extract(
    backend: ModelBackend,
    prefix: str,
    step_index: int,
    max_new_tokens: int = 32,
    suffix: str = FORCED_ANSWER_SUFFIX,
) -> ForcedExtractionResult:
    prompt = prefix + suffix
    completion = backend.generate(prompt, max_new_tokens=max_new_tokens)
    return ForcedExtractionResult(
        step_index=step_index,
        prompt=prompt,
        completion=completion,
        full_text=prompt + completion,
    )


class MockBackend:
    """Deterministic stand-in for a real model.

    For offline testing of the forced-extraction/parsing pipeline against
    hand-copied real R1-Distill traces. NOT a substitute for the real pilot
    on the actual model (project plan §5) -- only for shredding code bugs
    before spending GPU time.
    """

    def __init__(self, canned_completions: dict[str, str] | None = None, default: str = " 42."):
        self.canned_completions = canned_completions or {}
        self.default = default
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.calls.append((prompt, max_new_tokens))
        for key, completion in self.canned_completions.items():
            if key in prompt:
                return completion
        return self.default
