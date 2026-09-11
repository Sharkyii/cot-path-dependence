from early_stop.backend import MockScoredBackend
from early_stop.candidate_b_pipeline import (
    CANDIDATE_B_SUFFIX,
    build_trajectory_prefixes,
    compute_observable_state,
    extract_candidate_prefixes_text,
    generate_diverse_trajectories,
    branch_naturally,
    branch_naturally_with_telemetry,
)
from early_stop.path_dependence import find_matched_pairs, pair_permutation_test

TRACE = """Let me set up the equation.

We have x + 5 = 12.

Subtracting 5 from both sides gives x = 7.

So the final answer is \\boxed{7}."""


def test_generate_diverse_trajectories_calls_backend_k_times():
    backend = MockScoredBackend()
    backend.continue_default = TRACE
    trajs = generate_diverse_trajectories(backend, "solve for x", k=4, max_new_tokens=999)
    assert len(trajs) == 4
    assert all(t == TRACE for t in trajs)
    assert len(backend.calls) == 4
    assert backend.calls[0].startswith("[chat]")


def test_extract_candidate_prefixes_text_matches_step_count():
    prefixes = extract_candidate_prefixes_text(TRACE)
    # TRACE has 4 blank-line-delimited steps
    assert len(prefixes) == 4
    assert prefixes[0] == "Let me set up the equation."
    assert prefixes[-1] == TRACE.strip()  # full trace == prefix at the last step


def test_compute_observable_state_parses_answer_and_scores():
    backend = MockScoredBackend(canned={
        "Subtracting": ("7}.", 0.92, 0.3),
    })
    state, scored = compute_observable_state(backend, "solve for x", "...Subtracting 5...", max_new_tokens=48)
    assert state is not None
    assert state.answer == "7"
    assert state.confidence == 0.92
    assert state.entropy == 0.3
    assert scored.full_text.endswith("7}.")


def test_compute_observable_state_returns_none_on_unparseable_answer():
    backend = MockScoredBackend(default=(" I am not sure how to proceed", 0.4, 2.5))
    state, scored = compute_observable_state(backend, "solve for x", "garbled prefix", max_new_tokens=48)
    assert state is None
    assert scored is not None  # the scored completion is still returned for logging/debugging


def test_compute_observable_state_primes_correct_suffix():
    backend = MockScoredBackend()
    compute_observable_state(backend, "solve for x", "some reasoning", max_new_tokens=48)
    assert backend.calls[0].endswith(CANDIDATE_B_SUFFIX)


def test_build_trajectory_prefixes_end_to_end():
    # Every candidate prefix in TRACE, when force-extracted, should
    # resolve to the SAME final answer "7" (this is a coherent trace, not
    # one with self-correction) -- verifies the whole
    # trace -> prefixes -> phi(p_t) -> Prefix objects pipeline works.
    backend = MockScoredBackend(default=(" 7}.", 0.85, 0.4))
    result = build_trajectory_prefixes(
        backend, problem_id="prob1", trajectory_id="traj_a",
        base_prompt="solve for x", trace_text=TRACE, max_new_tokens=48,
    )
    assert result.n_unparseable == 0
    assert len(result.prefixes) == 4
    for i, p in enumerate(result.prefixes, start=1):
        assert p.step_index == i
        assert p.total_steps == 4
        assert p.state.answer == "7"
        assert p.problem_id == "prob1"
        assert p.trajectory_id == "traj_a"


def test_build_trajectory_prefixes_counts_unparseable_separately():
    # Only prefixes containing "Subtracting" (steps 3-4) get a genuine
    # parseable completion; earlier prefixes (steps 1-2) fall through to
    # the default, an unterminated, no-digit garbage completion --
    # unparseable via the shared parser's strict extract_boxed (requires a
    # real closing brace) combined with compute_observable_state's
    # method=="boxed" requirement, which rejects the last-number fallback
    # entirely (see test_regression_last_number_fallback_rejected below
    # for why that fallback specifically must not be trusted here).
    #
    # NOTE: the canned key must not be a substring of CANDIDATE_B_SUFFIX
    # itself ("Therefore, the final answer is \boxed{") or it will match
    # every call regardless of prefix content -- this bit a first version
    # of this test, caught by n_unparseable coming back 0 when it should
    # not have been.
    backend = MockScoredBackend(
        canned={"Subtracting": ("7}.", 0.85, 0.4)},
        default=(" gibberish, no digits here", 0.3, 3.0),
    )
    result = build_trajectory_prefixes(
        backend, "prob1", "traj_a", "solve for x", TRACE, max_new_tokens=48,
    )
    assert result.n_unparseable + len(result.prefixes) == 4
    assert result.n_unparseable == 2  # steps 1-2
    assert len(result.prefixes) == 2  # steps 3-4
    for p in result.prefixes:
        assert p.state.answer == "7"


def test_regression_last_number_fallback_rejected_not_scraped_as_phi():
    # The exact bug found during Step 2 testing: prefix_text legitimately
    # contains a number ("x + 5 = 12") as part of the problem's own
    # reasoning, unrelated to any attempted answer. When the primed box
    # fails to close (garbled/no-digit completion), parse_math_answer's
    # last-number fallback would scrape "12" from that unrelated context
    # and report it as if it were a real answer -- which, for Candidate
    # B's matching use case specifically, could manufacture a spurious
    # matched pair between two prefixes that never actually agreed on
    # anything. compute_observable_state must reject this (method !=
    # "boxed"), not just reject a hard parse failure.
    backend = MockScoredBackend(default=(" no digits in this completion", 0.3, 3.0))
    prefix_with_incidental_number = "We have x + 5 = 12."
    state, scored = compute_observable_state(
        backend, "solve for x", prefix_with_incidental_number, max_new_tokens=48
    )
    assert state is None, (
        "phi(p_t) must not be built from a last-number-fallback scrape of "
        "incidental problem-text digits -- got a state instead of None"
    )


def test_branch_naturally_produces_n_outcomes():
    backend = MockScoredBackend()
    backend.continue_default = " continuing... the answer is 7."
    branches = branch_naturally(backend, "solve for x", "step 1 reasoning", n=15, max_new_tokens=500)
    assert branches.n_total == 15
    assert branches.n_parsed == 15  # "the answer is 7" parses via last-number fallback
    assert branches.distribution() == {"7": 1.0}


def test_branch_naturally_does_not_use_forced_suffix():
    # Regression guard for DESIGN_CANDIDATE_B.md SS1.7: branching must be
    # UNFORCED. continue_generate (not forced_extract_with_scores) is the
    # only backend method branch_naturally may call.
    class NoScoringBackend:
        """A backend that only implements continue_generate -- if
        branch_naturally tried to call forced_extract_with_scores, this
        would raise AttributeError and fail the test."""
        def continue_generate(self, prefix_text, max_new_tokens):
            return " the answer is 3."

    branches = branch_naturally(NoScoringBackend(), "prompt", "prefix", n=5, max_new_tokens=100)
    assert branches.n_total == 5
    assert branches.distribution() == {"3": 1.0}


def test_branch_naturally_with_telemetry_matches_branch_naturally_outcomes():
    # Same BranchSet-shaped output as branch_naturally() -- so
    # pair_permutation_test() works unchanged on the returned branches.
    backend = MockScoredBackend()
    backend.continue_default = " the answer is \\boxed{7}."
    branches, telemetry = branch_naturally_with_telemetry(
        backend, "solve for x", "step 1 reasoning", n=6, max_new_tokens=500,
    )
    assert branches.n_total == 6
    assert branches.distribution() == {"7": 1.0}
    assert len(telemetry) == 6
    for t in telemetry:
        assert t["answer"] == "7"
        assert t["finish_reason"] in ("eos", "length")
        assert t["boxed_close_token_index"] is not None  # closes -- box is present


def test_branch_naturally_with_telemetry_flags_unclosed_box_as_no_close_index():
    backend = MockScoredBackend()
    backend.continue_default = " let me think about this further and \\boxed{unclosed"
    _branches, telemetry = branch_naturally_with_telemetry(
        backend, "prompt", "prefix", n=3, max_new_tokens=50,
    )
    for t in telemetry:
        assert t["boxed_close_token_index"] is None
        assert t["answer"] is None  # unclosed box never parses as a genuine answer


def test_full_mechanical_pipeline_two_trajectories_find_a_match_and_branch():
    """The end-to-end check the micro-pilot report needs: two DIFFERENT
    trajectories on the same problem that happen to reach the same
    observable state at some step -- find_matched_pairs must find them,
    and branching + PDI must run without error on the result.
    """
    backend = MockScoredBackend(default=(" 7}.", 0.85, 0.4))

    traj_a = build_trajectory_prefixes(
        backend, "probX", "traj_a", "solve for x",
        "Step one.\n\nStep two.\n\nSo the final answer is \\boxed{7}.",
        max_new_tokens=48,
    )
    traj_b = build_trajectory_prefixes(
        backend, "probX", "traj_b", "solve for x",
        "Different approach here.\n\nStill working.\n\nAlso reaches \\boxed{7}.",
        max_new_tokens=48,
    )
    all_prefixes = traj_a.prefixes + traj_b.prefixes
    pairs = find_matched_pairs(all_prefixes, "L1", position_tol=None)
    assert len(pairs) > 0  # every prefix here scores confidence=0.85, answer=7 -> should match broadly

    # Branch from one matched pair and confirm PDI/stats run cleanly
    a, b = pairs[0]
    branches_a = branch_naturally(backend, "solve for x", "prefix a", n=12, max_new_tokens=200)
    branches_b = branch_naturally(backend, "solve for x", "prefix b", n=12, max_new_tokens=200)
    result = pair_permutation_test(branches_a, branches_b, reps=200, seed=0)
    assert 0.0 <= result.p_value <= 1.0
