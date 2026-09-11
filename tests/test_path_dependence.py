import numpy as np
import pytest

from early_stop.path_dependence import (
    BranchOutcome,
    BranchSet,
    ObservableState,
    Prefix,
    answer_distribution,
    combine_pair_results,
    find_matched_pairs,
    matches,
    noise_floor_distribution,
    pair_permutation_test,
    pdi,
    state_distance,
    stratified_sample_pairs,
)


def S(answer="12", confidence=0.8, entropy=0.5):
    return ObservableState(answer=answer, confidence=confidence, entropy=entropy)


def P(problem_id="prob1", trajectory_id="traj_a", step_index=10, total_steps=20, state=None):
    return Prefix(problem_id, trajectory_id, step_index, total_steps, state or S())


def BS(answers):
    """Build a BranchSet from a list of answers (None = extraction failure)."""
    return BranchSet([BranchOutcome(a) for a in answers])


# ---------------------------------------------------------------------------
# Prefix -- normalized position, validation
# ---------------------------------------------------------------------------

def test_prefix_rejects_nonpositive_total_steps():
    with pytest.raises(ValueError):
        Prefix("p", "t", 5, 0, S())


def test_prefix_rejects_step_index_out_of_range():
    with pytest.raises(ValueError):
        Prefix("p", "t", 25, 20, S())


def test_prefix_normalized_position():
    p = Prefix("p", "t", 15, 30, S())
    assert p.normalized_position == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# state_distance -- continuous alternative to the L1/L2/L3 buckets (fix #1)
# ---------------------------------------------------------------------------

def test_state_distance_none_when_answers_differ():
    assert state_distance(S(answer="12"), S(answer="13")) is None


def test_state_distance_zero_for_identical_states():
    a = S(confidence=0.7, entropy=0.4)
    assert state_distance(a, a) == pytest.approx(0.0)


def test_state_distance_increases_with_confidence_or_entropy_gap():
    ref = S(confidence=0.7, entropy=0.4)
    close = S(confidence=0.72, entropy=0.41)
    far = S(confidence=0.2, entropy=0.9)
    assert state_distance(ref, close) < state_distance(ref, far)


# ---------------------------------------------------------------------------
# matches() -- level nesting is a MATCH-CRITERION property, not a PDI claim
# ---------------------------------------------------------------------------

def test_matches_l1_answer_only():
    a, b = S(confidence=0.9, entropy=0.9), S(confidence=0.1, entropy=0.1)
    assert matches(a, b, "L1") is True


def test_matches_different_answer_fails_every_level():
    a, b = S(answer="12"), S(answer="13")
    for level in ("L1", "L2", "L3"):
        assert matches(a, b, level) is False


def test_matches_l2_respects_confidence_tolerance():
    a = S(confidence=0.80)
    close = S(confidence=0.85)
    far = S(confidence=0.50)
    assert matches(a, close, "L2", confidence_tol=0.10) is True
    assert matches(a, far, "L2", confidence_tol=0.10) is False


def test_matches_l3_respects_entropy_tolerance_given_l2_passes():
    a = S(confidence=0.80, entropy=0.50)
    close = S(confidence=0.82, entropy=0.55)
    far_entropy = S(confidence=0.82, entropy=0.90)
    assert matches(a, close, "L3", confidence_tol=0.10, entropy_tol=0.15) is True
    assert matches(a, far_entropy, "L3", confidence_tol=0.10, entropy_tol=0.15) is False


def test_l3_match_implies_l1_and_l2_match_nesting_property():
    # This is ONLY a claim about the match CRITERION for a single pair
    # (L3 => L2 => L1). It is explicitly NOT a claim that mean(PDI) is
    # monotonic across the L1/L2/L3 SUBSETS -- see the module docstring
    # correction. state_distance() is the right tool for testing H2.
    a = S(confidence=0.80, entropy=0.50)
    b = S(confidence=0.82, entropy=0.55)
    if matches(a, b, "L3"):
        assert matches(a, b, "L2")
        assert matches(a, b, "L1")


# ---------------------------------------------------------------------------
# find_matched_pairs -- including the depth/position confound guard (fix #5)
# ---------------------------------------------------------------------------

def test_find_matched_pairs_excludes_same_trajectory():
    p1 = P(trajectory_id="traj_a", step_index=3, total_steps=20)
    p2 = P(trajectory_id="traj_a", step_index=7, total_steps=20)
    assert find_matched_pairs([p1, p2], "L1") == []


def test_find_matched_pairs_excludes_different_problems():
    p1 = P(problem_id="prob1", trajectory_id="traj_a")
    p2 = P(problem_id="prob2", trajectory_id="traj_b")
    assert find_matched_pairs([p1, p2], "L1") == []


def test_find_matched_pairs_finds_real_cross_history_match():
    p1 = P(trajectory_id="traj_a", step_index=10, total_steps=20, state=S(answer="12"))
    p2 = P(trajectory_id="traj_b", step_index=9, total_steps=18, state=S(answer="12"))
    p3 = P(trajectory_id="traj_c", step_index=10, total_steps=20, state=S(answer="99"))
    pairs = find_matched_pairs([p1, p2, p3], "L1")
    assert len(pairs) == 1
    assert {pairs[0][0].trajectory_id, pairs[0][1].trajectory_id} == {"traj_a", "traj_b"}


def test_find_matched_pairs_position_tol_excludes_mismatched_depth():
    # Same problem, same answer, but wildly different normalized depth:
    # step 2/20 (10% through) vs step 81/90 (90% through). Without a depth
    # guard this counts as "different history, same state" -- but it may
    # really just be "different amount of reasoning done" (the confound
    # methodology fix #5 addresses).
    shallow = P(trajectory_id="traj_a", step_index=2, total_steps=20, state=S(answer="12"))
    deep = P(trajectory_id="traj_b", step_index=81, total_steps=90, state=S(answer="12"))
    assert find_matched_pairs([shallow, deep], "L1", position_tol=0.15) == []
    # disabling the guard explicitly (a deliberate robustness experiment) restores it
    assert len(find_matched_pairs([shallow, deep], "L1", position_tol=None)) == 1


def test_find_matched_pairs_position_tol_allows_close_depth():
    a = P(trajectory_id="traj_a", step_index=10, total_steps=20, state=S(answer="12"))  # 50%
    b = P(trajectory_id="traj_b", step_index=12, total_steps=20, state=S(answer="12"))  # 60%
    assert len(find_matched_pairs([a, b], "L1", position_tol=0.15)) == 1


# ---------------------------------------------------------------------------
# stratified_sample_pairs -- fix for the micro-pilot's observed failure
# mode: naive first-N selection drew almost entirely from the converged/
# high-confidence regime, giving zero pairs in the medium-confidence range
# where path dependence is most plausible.
# ---------------------------------------------------------------------------

def _pair_at_confidence(conf: float, tag: str) -> tuple[Prefix, Prefix]:
    a = P(trajectory_id=f"traj_a_{tag}", state=S(answer="7", confidence=conf))
    b = P(trajectory_id=f"traj_b_{tag}", state=S(answer="7", confidence=conf))
    return (a, b)


def test_stratified_sample_regression_naive_selection_starves_medium_confidence():
    # Reproduce the exact micro-pilot failure mode: 90 high-confidence
    # pairs enumerated first, only 5 medium-confidence pairs after them.
    # Naive pairs[:20] gets ZERO medium-confidence coverage.
    high = [_pair_at_confidence(0.95, f"h{i}") for i in range(90)]
    medium = [_pair_at_confidence(0.65, f"m{i}") for i in range(5)]
    pairs = high + medium

    naive = pairs[:20]
    naive_medium_count = sum(
        1 for a, b in naive if (a.state.confidence + b.state.confidence) / 2 < 0.85
    )
    assert naive_medium_count == 0, "naive slicing should reproduce the starvation bug"

    stratified = stratified_sample_pairs(pairs, budget=20, seed=0)
    strat_medium_count = sum(
        1 for a, b in stratified if (a.state.confidence + b.state.confidence) / 2 < 0.85
    )
    assert strat_medium_count > 0, "stratified sampling must give the medium bucket SOME budget"
    assert len(stratified) == 20


def _pair_at(conf: float, problem_id: str, tag: str) -> tuple[Prefix, Prefix]:
    a = P(problem_id=problem_id, trajectory_id=f"traj_a_{tag}", state=S(answer="7", confidence=conf))
    b = P(problem_id=problem_id, trajectory_id=f"traj_b_{tag}", state=S(answer="7", confidence=conf))
    return (a, b)


def test_stratified_sample_regression_one_problem_does_not_starve_others_within_a_bucket():
    # Reproduce the micro-pilot run #2 finding (SS4.6/4.8 in
    # DESIGN_CANDIDATE_B.md): confidence stratification alone still let a
    # single problem dominate a bucket's budget when that problem simply
    # had more matched pairs than its neighbors -- notebook 1/3's whole
    # 8-pair budget landed on only 2 of its 3 problems even with
    # confidence strata working as designed. All pairs here sit in the
    # SAME confidence stratum, split unevenly by problem.
    dominant = [_pair_at(0.6, "probA", f"a{i}") for i in range(50)]
    sparse_b = [_pair_at(0.6, "probB", f"b{i}") for i in range(3)]
    sparse_c = [_pair_at(0.6, "probC", f"c{i}") for i in range(3)]
    pairs = dominant + sparse_b + sparse_c

    result = stratified_sample_pairs(pairs, budget=9, seed=0)
    problems_hit = {a.problem_id for a, b in result}
    assert problems_hit == {"probA", "probB", "probC"}, (
        "budget must spread across all problems present in a bucket, not just the largest"
    )
    assert len(result) == 9


def test_stratified_sample_respects_budget():
    pairs = [_pair_at_confidence(0.3, f"a{i}") for i in range(50)]
    result = stratified_sample_pairs(pairs, budget=10, seed=0)
    assert len(result) == 10


def test_stratified_sample_thin_stratum_does_not_shrink_total_budget():
    # Only 2 pairs available in the low bucket, plenty in medium/high --
    # the shortfall must be redistributed, not silently leave budget unmet.
    low = [_pair_at_confidence(0.1, f"l{i}") for i in range(2)]
    medium = [_pair_at_confidence(0.6, f"m{i}") for i in range(30)]
    high = [_pair_at_confidence(0.9, f"h{i}") for i in range(30)]
    result = stratified_sample_pairs(low + medium + high, budget=15, seed=0)
    assert len(result) == 15


def test_stratified_sample_empty_input():
    assert stratified_sample_pairs([], budget=10) == []


def test_stratified_sample_zero_budget():
    pairs = [_pair_at_confidence(0.5, "x")]
    assert stratified_sample_pairs(pairs, budget=0) == []


def test_stratified_sample_budget_exceeds_available_returns_all():
    pairs = [_pair_at_confidence(0.5, f"x{i}") for i in range(5)]
    result = stratified_sample_pairs(pairs, budget=100, seed=0)
    assert len(result) == 5


def test_stratified_sample_is_deterministic_given_seed():
    # NOTE: compare by trajectory_id, not tuple id() -- (a, b) is a freshly
    # repacked tuple on every call even when a/b are the same underlying
    # Prefix objects, so identity comparison would fail even for correct,
    # fully-deterministic output. Content is what actually needs to match.
    pairs = [_pair_at_confidence(0.5, f"x{i}") for i in range(40)]
    r1 = stratified_sample_pairs(pairs, budget=10, seed=42)
    r2 = stratified_sample_pairs(pairs, budget=10, seed=42)
    ids1 = [(a.trajectory_id, b.trajectory_id) for a, b in r1]
    ids2 = [(a.trajectory_id, b.trajectory_id) for a, b in r2]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# answer_distribution
# ---------------------------------------------------------------------------

def test_answer_distribution_basic():
    d = answer_distribution(["12", "12", "13", "12"])
    assert d["12"] == pytest.approx(0.75)
    assert d["13"] == pytest.approx(0.25)


def test_answer_distribution_all_none_raises():
    with pytest.raises(ValueError):
        answer_distribution([None, None])


# ---------------------------------------------------------------------------
# BranchSet -- parse-rate accounting (fix #3)
# ---------------------------------------------------------------------------

def test_branch_set_parse_rate():
    bs = BS(["12", "12", None, "13", None])
    assert bs.n_total == 5
    assert bs.n_parsed == 3
    assert bs.parse_rate == pytest.approx(0.6)


def test_branch_set_passes_parse_gate():
    high = BS(["12"] * 9 + [None])  # 90%
    low = BS(["12"] * 5 + [None] * 5)  # 50%
    assert high.passes_parse_gate(min_parse_rate=0.80) is True
    assert low.passes_parse_gate(min_parse_rate=0.80) is False


def test_branch_set_distribution_drops_failures_option_a():
    bs = BS(["12", "12", None, "13"])
    d = bs.distribution()
    assert d == {"12": pytest.approx(2 / 3), "13": pytest.approx(1 / 3)}


def test_branch_set_distribution_with_failures_option_b():
    bs = BS(["12", "12", None, "13"])
    d = bs.distribution_with_failures()
    assert d["UNPARSEABLE"] == pytest.approx(0.25)
    assert d["12"] == pytest.approx(0.5)


def test_regression_none_extraction_failure_does_not_silently_bias_pdi():
    # The exact scenario from methodology review point #3: prefix A looks
    # "unanimous" only because its failures were silently dropped, while
    # prefix B has none. Comparing via distribution() alone on a
    # low-parse-rate set would misattribute an extraction artifact to path
    # dependence -- the parse gate must catch this before pdi() runs.
    prefix_a = BS(["12", "12", "12", None, None, None, None])  # 43% parse rate
    prefix_b = BS(["12", "13", "13", "13", "12", "13", "12"])  # 100% parse rate
    assert prefix_a.passes_parse_gate(min_parse_rate=0.80) is False
    with pytest.raises(ValueError):
        pair_permutation_test(prefix_a, prefix_b, min_parse_rate=0.80)


# ---------------------------------------------------------------------------
# pdi -- the core metric (unchanged behavior)
# ---------------------------------------------------------------------------

def test_pdi_identical_distributions_is_zero():
    d = {"12": 0.7, "13": 0.3}
    assert pdi(d, d) == pytest.approx(0.0, abs=1e-9)


def test_pdi_disjoint_distributions_is_maximal():
    assert pdi({"12": 1.0}, {"99": 1.0}) == pytest.approx(1.0, abs=1e-6)


def test_pdi_is_symmetric():
    a = {"12": 0.6, "13": 0.4}
    b = {"12": 0.2, "13": 0.8}
    assert pdi(a, b) == pytest.approx(pdi(b, a))


def test_pdi_monotonic_in_distributional_distance():
    ref = {"12": 0.9, "13": 0.1}
    close = {"12": 0.8, "13": 0.2}
    far = {"12": 0.3, "13": 0.7}
    assert pdi(ref, close) < pdi(ref, far)


# ---------------------------------------------------------------------------
# noise_floor_* -- now explicitly diagnostic-only (fix #2), still correct
# as a descriptive statistic of pure branching noise
# ---------------------------------------------------------------------------

def test_noise_floor_requires_minimum_branches():
    with pytest.raises(ValueError):
        noise_floor_distribution(["12", "12", "13"], reps=5)


def test_noise_floor_of_unanimous_branches_is_always_zero():
    floor = noise_floor_distribution(["12"] * 20, reps=200, seed=1)
    assert np.allclose(floor, 0.0)


def test_noise_floor_of_highly_variable_branches_is_nonzero_but_bounded():
    rng = np.random.default_rng(2)
    answers = [rng.choice(["12", "13", "14"]) for _ in range(40)]
    floor = noise_floor_distribution(answers, reps=500, seed=3)
    assert floor.mean() > 0.0
    assert floor.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# pair_permutation_test -- the FIXED per-pair unit of analysis (fix #2)
# ---------------------------------------------------------------------------

def test_pair_permutation_test_no_real_effect_gives_large_p_value():
    rng = np.random.default_rng(5)
    # Both branch sets drawn from the SAME true distribution -- no real
    # path dependence, just finite-sample noise.
    shared_probs = {"12": 0.6, "13": 0.4}
    a = BS(list(rng.choice(list(shared_probs), size=20, p=list(shared_probs.values()))))
    b = BS(list(rng.choice(list(shared_probs), size=20, p=list(shared_probs.values()))))
    result = pair_permutation_test(a, b, reps=3000, seed=6)
    assert result.p_value > 0.05


def test_pair_permutation_test_real_effect_gives_small_p_value():
    rng = np.random.default_rng(7)
    # Two CLEARLY different true distributions -- real path dependence.
    a = BS(list(rng.choice(["12", "13"], size=25, p=[0.95, 0.05])))
    b = BS(list(rng.choice(["12", "13"], size=25, p=[0.10, 0.90])))
    result = pair_permutation_test(a, b, reps=3000, seed=8)
    assert result.p_value < 0.01
    assert result.effect_size > 0.1


def test_pair_permutation_test_uses_only_this_pairs_own_data():
    # Regression guard for the pseudoreplication bug: unit sizes (n_a, n_b)
    # must reflect just these two branch sets, never some larger pooled N.
    a = BS(["12"] * 10)
    b = BS(["13"] * 8)
    result = pair_permutation_test(a, b, reps=500, seed=9)
    assert result.n_a == 10
    assert result.n_b == 8


def test_pair_permutation_test_rejects_low_parse_rate_branch_set():
    a = BS(["12"] * 3 + [None] * 7)  # 30% parse rate
    b = BS(["12"] * 10)
    with pytest.raises(ValueError):
        pair_permutation_test(a, b, min_parse_rate=0.80)


# ---------------------------------------------------------------------------
# combine_pair_results -- Fisher's method over K INDEPENDENT pair tests
# ---------------------------------------------------------------------------

def test_combine_pair_results_requires_at_least_one():
    with pytest.raises(ValueError):
        combine_pair_results([])


def test_combine_pair_results_no_effect_across_many_pairs_stays_nonsignificant():
    rng = np.random.default_rng(10)
    shared = {"12": 0.5, "13": 0.5}
    results = []
    for i in range(8):
        a = BS(list(rng.choice(list(shared), size=15, p=list(shared.values()))))
        b = BS(list(rng.choice(list(shared), size=15, p=list(shared.values()))))
        results.append(pair_permutation_test(a, b, reps=1000, seed=100 + i))
    combined = combine_pair_results(results)
    assert combined.n_pairs == 8
    assert combined.combined_p_value > 0.05


def test_combine_pair_results_real_effect_across_many_pairs_is_significant():
    rng = np.random.default_rng(11)
    results = []
    for i in range(8):
        a = BS(list(rng.choice(["12", "13"], size=15, p=[0.9, 0.1])))
        b = BS(list(rng.choice(["12", "13"], size=15, p=[0.15, 0.85])))
        results.append(pair_permutation_test(a, b, reps=1000, seed=200 + i))
    combined = combine_pair_results(results)
    assert combined.combined_p_value < 0.01
    assert combined.mean_effect_size > 0.1
    assert combined.frac_individually_significant > 0.5


def test_regression_pseudoreplication_bug_no_longer_reproducible():
    # The exact bug this whole redesign fixes: 8 modest, real matched pairs
    # should NOT collapse to an absurd p-value like the old pooled-global
    # test gave (p=0.0002, verified before the fix). With the corrected
    # per-pair design at N=8 independent pairs, the combined p-value must
    # be far less extreme even when the effect is real.
    rng = np.random.default_rng(12)
    results = []
    for i in range(8):
        # modest, real shift -- analogous to the H1 "small but real effect"
        # scenario, not an enormous one
        a = BS(list(rng.choice(["12", "13"], size=12, p=[0.65, 0.35])))
        b = BS(list(rng.choice(["12", "13"], size=12, p=[0.45, 0.55])))
        results.append(pair_permutation_test(a, b, reps=1000, seed=300 + i))
    combined = combine_pair_results(results)
    # A p-value like 0.0002 from 8 modest pairs would indicate the bug is
    # back; the fixed design should not produce that level of false
    # confidence from this little, this-modest-an-effect data.
    assert combined.combined_p_value > 0.0002
