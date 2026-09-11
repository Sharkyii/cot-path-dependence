"""Candidate B orchestration: ties early_stop.backend (GPU-facing) and
early_stop.path_dependence (pure logic) together into the actual pipeline
described in DESIGN_CANDIDATE_B.md §1.5:

  1. generate_diverse_trajectories -- k base traces per problem (different
     histories, via stochastic decoding)
  2. extract_candidate_prefixes -- candidate stop points within each trace
  3. compute_observable_state -- phi(p_t) at each prefix (forced
     extraction + confidence/entropy, reusing the existing extraction
     convention from early_stop.extraction/grading)
  4. find_matched_pairs -- already in path_dependence.py, used as-is
  5. branch_naturally -- N unforced continuations from a matched prefix
     (explicitly NOT forced-extraction -- see DESIGN_CANDIDATE_B.md §1.7:
     this proposal only observes natural branching, never forces a
     truncated answer, unlike the rest of this project's pipeline)

Every function here takes a `backend` object satisfying a minimal
protocol (`.generate`, `.continue_generate`, `.forced_extract_with_scores`)
so the whole pipeline is testable with
early_stop.backend.MockScoredBackend before touching a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from early_stop.backend import ScoredCompletion
from early_stop.parsers import parse_math_answer
from early_stop.path_dependence import (
    BranchOutcome,
    BranchSet,
    ObservableState,
    Prefix,
)
from early_stop.segmentation import split_into_steps, step_prefix

# Candidate B's phase-2 forced suffix primes an open \boxed{ (see
# early_stop.extraction.suffix_for_domain's math variant) so the answer is
# both parseable by the existing parser AND scoreable via
# forced_extract_with_scores's first-generated-token confidence/entropy.
CANDIDATE_B_SUFFIX = "\n\nTherefore, the final answer is \\boxed{"


class ScoredBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> str: ...
    def continue_generate(self, prefix_text: str, max_new_tokens: int) -> str: ...
    def forced_extract_with_scores(
        self, prefix_text: str, suffix: str, max_new_tokens: int
    ) -> ScoredCompletion: ...


def generate_diverse_trajectories(
    backend: ScoredBackend, prompt: str, k: int, max_new_tokens: int = 4000,
) -> list[str]:
    """k independently-sampled full traces for the SAME prompt. Diversity
    comes entirely from temperature sampling already baked into the
    backend (DESIGN_CANDIDATE_B.md §1.5 step 1) -- no special engineering
    needed beyond calling generate() k times.
    """
    return [backend.generate(prompt, max_new_tokens=max_new_tokens) for _ in range(k)]


def extract_candidate_prefixes_text(trace_text: str) -> list[str]:
    """Reconstruct the trace-text prefix ending at each step boundary,
    reusing the same blank-line segmentation as the rest of this project
    (early_stop.segmentation) so step counting stays consistent with the
    pilot's own step-based conventions."""
    steps = split_into_steps(trace_text)
    return [step_prefix(steps, k) for k in range(1, len(steps) + 1)]


def compute_observable_state(
    backend: ScoredBackend, base_prompt: str, prefix_text: str,
    max_new_tokens: int = 48,
) -> tuple[ObservableState | None, ScoredCompletion]:
    """phi(p_t): force-extract an answer at this prefix and score its
    confidence/entropy, using the SAME suffix-priming convention already
    used elsewhere in this project's pipeline (see
    early_stop.extraction.suffix_for_domain and
    early_stop.grading.forced_extract_and_grade's docstring for why
    grading must run on the FULL accumulated text, not the completion
    alone -- the same reasoning applies to answer parsing here).

    Requires parsed.method == "boxed" (a genuinely closed, primed box),
    NOT the "last_number" fallback. This is STRICTER than the rest of
    this project's grading convention, deliberately: for grading raw
    accuracy, an occasional fallback-scraped number is tolerable noise.
    For Candidate B, phi(p_t)'s answer is used to MATCH prefixes against
    each other -- a number incidentally scraped from the PROBLEM TEXT
    itself (e.g. "x + 5 = 12" containing the digits "12") when the primed
    box fails to close can silently produce a spurious matched pair (two
    prefixes "agreeing" on a number neither actually attempted as an
    answer), corrupting the core comparison rather than just adding
    grading noise. Verified concretely during Step 2 testing: a garbled
    completion with no digits still produced answer="12", scraped from
    unrelated reasoning text earlier in the prefix, via the last-number
    fallback -- caught by a test assertion that expected None and got "12"
    instead.

    Returns (None, scored) if the answer fails to parse AS A GENUINE
    ATTEMPT (i.e. via the boxed path) -- the caller decides whether an
    unparseable prefix should be dropped or logged, this function doesn't
    silently swallow the failure.
    """
    full_prefix = base_prompt + "\n\n" + prefix_text
    scored = backend.forced_extract_with_scores(
        full_prefix, CANDIDATE_B_SUFFIX, max_new_tokens=max_new_tokens
    )
    parsed = parse_math_answer(scored.full_text)
    if parsed.answer is None or parsed.method != "boxed":
        return None, scored
    state = ObservableState(
        answer=parsed.answer,
        confidence=scored.top_token_confidence,
        entropy=scored.entropy_bits,
    )
    return state, scored


@dataclass
class TrajectoryPrefixes:
    """All candidate prefixes extracted from one trajectory, with their
    observable states -- the per-trajectory unit find_matched_pairs()
    consumes across trajectories of the same problem."""

    problem_id: str
    trajectory_id: str
    prefixes: list[Prefix]  # only successfully-scored prefixes (state is not None)
    n_unparseable: int  # prefixes dropped because forced extraction failed to parse


def build_trajectory_prefixes(
    backend: ScoredBackend, problem_id: str, trajectory_id: str,
    base_prompt: str, trace_text: str, max_new_tokens: int = 48,
) -> TrajectoryPrefixes:
    """Runs compute_observable_state at every step boundary in one
    trajectory, producing the Prefix objects find_matched_pairs() needs.
    """
    prefix_texts = extract_candidate_prefixes_text(trace_text)
    total_steps = len(prefix_texts)
    prefixes: list[Prefix] = []
    n_unparseable = 0
    for step_index, prefix_text in enumerate(prefix_texts, start=1):
        state, _scored = compute_observable_state(
            backend, base_prompt, prefix_text, max_new_tokens=max_new_tokens
        )
        if state is None:
            n_unparseable += 1
            continue
        prefixes.append(
            Prefix(
                problem_id=problem_id,
                trajectory_id=trajectory_id,
                step_index=step_index,
                total_steps=total_steps,
                state=state,
            )
        )
    return TrajectoryPrefixes(problem_id, trajectory_id, prefixes, n_unparseable)


def branch_naturally(
    backend: ScoredBackend, base_prompt: str, prefix_step_text: str,
    n: int, max_new_tokens: int = 2000,
) -> BranchSet:
    """N independent NATURAL continuations from a matched prefix -- no
    forced-suffix intervention (DESIGN_CANDIDATE_B.md §1.7: this proposal
    only observes natural branching, unlike the forced-extraction
    convention used to compute phi itself). Each continuation's final
    answer is parsed the same way a full natural trace would be (via
    parse_math_answer's boxed-then-last-number fallback chain), since a
    natural continuation may or may not end in \\boxed{}.
    """
    full_prefix = base_prompt + "\n\n" + prefix_step_text
    outcomes: list[BranchOutcome] = []
    for _ in range(n):
        completion = backend.continue_generate(full_prefix, max_new_tokens=max_new_tokens)
        parsed = parse_math_answer(full_prefix + completion)
        outcomes.append(BranchOutcome(parsed.answer))
    return BranchSet(outcomes)
