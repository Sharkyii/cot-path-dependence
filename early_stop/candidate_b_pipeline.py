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

from early_stop.backend import BranchTelemetry, ScoredCompletion
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


class TelemetryBackend(Protocol):
    """Superset of ScoredBackend -- adds continue_generate_with_telemetry(),
    the one extra call branch_naturally_with_telemetry() needs. Kept as a
    separate protocol (not folded into ScoredBackend) so every existing
    caller of ScoredBackend -- and every existing test's minimal fake
    backend -- keeps working unchanged; only the two Tier 1 audit scripts
    (REVISION_PLAN.md) need this wider one."""

    def continue_generate_with_telemetry(
        self, prefix_text: str, max_new_tokens: int
    ) -> BranchTelemetry: ...


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


def branch_naturally_with_telemetry(
    backend: TelemetryBackend, base_prompt: str, prefix_step_text: str,
    n: int, max_new_tokens: int = 3000,
) -> tuple[BranchSet, list[dict]]:
    """Same natural-branching procedure as branch_naturally(), but via
    continue_generate_with_telemetry() instead of continue_generate() --
    returns both the usual BranchSet (so pair_permutation_test() works
    unchanged) AND a parallel list of per-branch telemetry dicts
    (finish_reason, n_generated_tokens, boxed_close_token_index, raw
    completion text, and the exact prefix_text branched from).

    Exists only for REVISION_PLAN.md Tier 1's two audit scripts
    (modal_audit_l1_l2_contrast.py, modal_audit_censoring.py) -- the main
    pilot pipeline keeps using branch_naturally(), which is cheaper (one
    fewer decode per token-probe-step) and already has 66 pairs' worth of
    results built on its exact output shape.
    """
    full_prefix = base_prompt + "\n\n" + prefix_step_text
    outcomes: list[BranchOutcome] = []
    telemetry: list[dict] = []
    for _ in range(n):
        t = backend.continue_generate_with_telemetry(full_prefix, max_new_tokens=max_new_tokens)
        parsed = parse_math_answer(t.full_text)
        outcomes.append(BranchOutcome(parsed.answer))
        telemetry.append({
            "answer": parsed.answer,
            "finish_reason": t.finish_reason,
            "n_generated_tokens": t.n_generated_tokens,
            "boxed_close_token_index": t.boxed_close_token_index,
            "raw_completion_text": t.completion_text,
        })
    return BranchSet(outcomes), telemetry


# ---------------------------------------------------------------------------
# Tier 1 audit helpers (REVISION_PLAN.md): split GPU work into small,
# independently-runnable phases instead of one long script, so a failure or
# a budget check partway through doesn't waste money already spent on an
# earlier phase, and so the SAME functions used on a real Modal GPU can be
# exercised for free against MockScoredBackend in a smoke test first.
#
# Phase 1 (build_prefix_pool) is GPU-bound but cheap and has no branching in
# it. Phase 2 (matching) is pure CPU logic already in path_dependence.py.
# Phase 3 (sample_and_branch_with_telemetry) is the expensive GPU part and
# is kept separate so it can be pointed at a prefix pool that Phase 1
# already paid for and saved, without re-running Phase 1.
# ---------------------------------------------------------------------------

def build_prefix_pool(
    backend: ScoredBackend, problem_id: str, base_prompt: str,
    n_trajectories: int, max_new_tokens_trace: int = 3000, max_new_tokens_score: int = 64,
    on_trajectory_done=None,
) -> dict:
    """Phase 1 of the audit scripts: generate n_trajectories independent
    traces for ONE problem and score every step boundary, returning a
    JSON-safe dict (see load_prefix_pool for the inverse) instead of
    Prefix objects directly -- this is what gets written to the Modal
    volume so a separate Phase 2/3 script can load it without re-running
    generation.

    on_trajectory_done(trajectory_id, n_scored, n_unparseable, elapsed_s), if
    given, is called after each trajectory -- used by the Modal scripts to
    print progress and commit the volume incrementally, and by the smoke
    test to assert it was called the right number of times.
    """
    import time

    trajectories: dict[str, list[str]] = {}
    prefixes: list[dict] = []
    for ti in range(n_trajectories):
        traj_id = f"traj_{ti}"
        try:
            t0 = time.time()
            trace_text = generate_diverse_trajectories(backend, base_prompt, k=1, max_new_tokens=max_new_tokens_trace)[0]
            tp = build_trajectory_prefixes(
                backend, problem_id, traj_id, base_prompt, trace_text, max_new_tokens=max_new_tokens_score,
            )
            trajectories[traj_id] = split_into_steps(trace_text)
            for p in tp.prefixes:
                prefixes.append({
                    "problem_id": p.problem_id, "trajectory_id": p.trajectory_id,
                    "step_index": p.step_index, "total_steps": p.total_steps,
                    "state": {"answer": p.state.answer, "confidence": p.state.confidence, "entropy": p.state.entropy},
                })
            elapsed = time.time() - t0
            if on_trajectory_done is not None:
                on_trajectory_done(traj_id, len(tp.prefixes), tp.n_unparseable, elapsed)
        except Exception as e:
            # Same reasoning as sample_and_branch_with_telemetry's per-pair
            # try/except: an 8h+ unattended run must survive one bad
            # trajectory (OOM surviving the retry, a decode edge case)
            # rather than losing every trajectory generated before it.
            print(f"[build_prefix_pool] {traj_id} FAILED, skipping: {type(e).__name__}: {e}")

        # 2026-09-12, found by the audit1a diagnostic run: HFBackend.free_vram()
        # exists (docstring: "call between problems if fragmentation becomes
        # an issue") but was never actually called anywhere in this project,
        # including the original full-pilot script. It went unnoticed there
        # because no prior run did this many forced_extract_with_scores
        # calls (each retains an output_scores tensor) back-to-back on one
        # problem. d3_prob5 alone did -- traj_0 used ~56 scoring calls, and
        # by partway through traj_1 the allocator reported 59MB free out of
        # 22GB, collapsing that trajectory's scoring success rate from
        # 14/56 to 4/94. free_vram() is a no-op if the backend doesn't
        # define it (e.g. MockScoredBackend in the smoke test), so this is
        # safe for both real and mock backends.
        free_vram = getattr(backend, "free_vram", None)
        if free_vram is not None:
            free_vram()

    return {
        "problem_id": problem_id,
        "base_prompt": base_prompt,
        "n_trajectories": n_trajectories,
        "trajectories": trajectories,
        "prefixes": prefixes,
    }


def load_prefix_pool(data: dict) -> tuple[str, str, dict[str, list[str]], list[Prefix]]:
    """Inverse of build_prefix_pool(). Returns (problem_id, base_prompt,
    trajectories, prefixes) with prefixes reconstructed as real Prefix
    objects, ready for early_stop.path_dependence.find_matched_pairs()."""
    prefixes = [
        Prefix(
            problem_id=p["problem_id"], trajectory_id=p["trajectory_id"],
            step_index=p["step_index"], total_steps=p["total_steps"],
            state=ObservableState(**p["state"]),
        )
        for p in data["prefixes"]
    ]
    return data["problem_id"], data["base_prompt"], data["trajectories"], prefixes


def sample_and_branch_with_telemetry(
    backend: TelemetryBackend, base_prompt: str, trajectories: dict[str, list[str]],
    pairs: list[tuple[Prefix, Prefix]], n_branches: int, max_new_tokens: int,
    on_pair_done=None,
) -> list[dict]:
    """Phase 3: branch each of the given (already sampled) pairs with
    telemetry, returning one result dict per pair. Pure orchestration --
    takes an already-matched-and-sampled pair list rather than doing its
    own matching, so Phase 2 (find_matched_pairs + stratified_sample_pairs,
    both pure CPU functions already in path_dependence.py) stays a
    separate, free step callers can inspect before paying for this one.

    on_pair_done(entry), if given, is called after each pair with the
    result dict -- used by the Modal scripts for incremental volume
    commits, and by the smoke test to assert shape without a GPU.
    """
    import time

    from early_stop.path_dependence import state_distance

    results = []
    for a, b in pairs:
        try:
            steps_a = trajectories[a.trajectory_id]
            steps_b = trajectories[b.trajectory_id]
            prefix_text_a = step_prefix(steps_a, a.step_index)
            prefix_text_b = step_prefix(steps_b, b.step_index)

            t0 = time.time()
            branch_set_a, telemetry_a = branch_naturally_with_telemetry(
                backend, base_prompt, prefix_text_a, n_branches, max_new_tokens,
            )
            branch_set_b, telemetry_b = branch_naturally_with_telemetry(
                backend, base_prompt, prefix_text_b, n_branches, max_new_tokens,
            )
            elapsed = time.time() - t0

            entry = {
                "problem_id": a.problem_id,
                "side_a_state": {
                    "answer": a.state.answer, "confidence": a.state.confidence,
                    "entropy": a.state.entropy, "normalized_position": a.normalized_position,
                },
                "side_b_state": {
                    "answer": b.state.answer, "confidence": b.state.confidence,
                    "entropy": b.state.entropy, "normalized_position": b.normalized_position,
                },
                "state_distance": state_distance(a.state, b.state),
                "side_a_raw_answers": branch_set_a.parsed_answers(),
                "side_b_raw_answers": branch_set_b.parsed_answers(),
                "side_a_telemetry": telemetry_a,
                "side_b_telemetry": telemetry_b,
                "prefix_text_a": prefix_text_a,
                "prefix_text_b": prefix_text_b,
                "elapsed_s": elapsed,
            }
            results.append(entry)
            if on_pair_done is not None:
                on_pair_done(entry)
        except Exception as e:
            # A single pair's failure (OOM surviving the retry in
            # _generate_retrying_oom, a stray parse error, anything) must
            # not kill the whole run -- this function is used by 8h+
            # unattended Modal jobs where nobody is watching to restart
            # one. Log and move on to the next pair; whatever pairs DID
            # succeed are still saved by on_pair_done's incremental commit.
            print(f"[sample_and_branch_with_telemetry] pair ({a.problem_id}) FAILED, "
                  f"skipping: {type(e).__name__}: {e}")

        # Same fragmentation guard as build_prefix_pool -- see that
        # function's comment. Each pair here is 2*n_branches generate()
        # calls; cheap insurance against the same failure mode. Runs even
        # after a caught failure, so the next pair starts clean.
        free_vram = getattr(backend, "free_vram", None)
        if free_vram is not None:
            free_vram()

    return results


def empty_rate_at_cutoff(telemetry: list[dict], cutoff: int) -> float | None:
    """From ONE set of branches generated at a HIGH token budget, compute
    what the empty-answer rate WOULD have been at an earlier, LOWER
    cutoff -- "empty" iff \\boxed{} never closed by `cutoff` tokens. This
    is what lets Tier 1.3's censoring audit test multiple candidate token
    budgets (1500, 3000, ...) from a single round of generation instead of
    one full (expensive) run per budget. See
    HFBackend.continue_generate_with_telemetry's docstring for how
    boxed_close_token_index is computed."""
    if not telemetry:
        return None
    n_empty = sum(
        1 for t in telemetry
        if t["boxed_close_token_index"] is None or t["boxed_close_token_index"] > cutoff
    )
    return n_empty / len(telemetry)
