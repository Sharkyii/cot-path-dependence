"""Path Dependence Index (PDI): is observable reasoning state a sufficient
statistic for future outcomes?

Core question (see PROPOSAL_CANDIDATE_B.md): two reasoning prefixes on the
SAME problem, reached via DIFFERENT histories, that land on the SAME
observable state (phi) -- do they produce the same distribution of future
answers when continued? If yes, phi is sufficient and every early-stopping
signal built only on phi has no more information to gain from history. If
no, there's a real, measurable ceiling on any current-state-only signal.

This module is pure logic (JS divergence, matching, permutation tests) --
no GPU, no model calls. Trajectory pools and branched continuations are
generated elsewhere (backend.py) and fed in as plain data here.

Methodology note: the statistical design below was revised after an
external review round found a real pseudoreplication bug in the first
version (v1's permutation test pooled ~1000 correlated same-prefix noise
resamples as if they were independent observations, alongside a handful of
genuinely independent matched pairs -- this produced absurdly overconfident
p-values, e.g. p=0.0002 from only 8 real pairs, verified by reproduction
before fixing). See DESIGN_CANDIDATE_B.md for the full review and the
fix rationale for each of the four issues raised. Do not revert to a
single pooled global permutation test without re-reading that document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import combine_pvalues, linregress

MatchLevel = Literal["L1", "L2", "L3"]


@dataclass(frozen=True)
class ObservableState:
    """phi(p_t): the features early-stopping methods actually use in
    practice. Deliberately excludes hidden states / embeddings (L4/L5) --
    see PROPOSAL_CANDIDATE_B.md §7, kept out of scope to avoid the RAD
    collision (RAD operates on MoE routing internals, not this).

    `confidence` is LOCAL TOKEN CONFIDENCE at the answer token (probability
    mass on the top token), NOT verified answer-correctness confidence --
    these are not the same thing (a model can be highly confident in the
    token "42" while the surrounding reasoning that produced it is shaky).
    Never call this "answer confidence" in analysis or writeup; the
    distinction matters for what the paper is actually allowed to claim.
    """

    answer: str  # normalized, comparable via early_stop.parsers
    confidence: float  # LOCAL TOKEN confidence, probability mass on top answer token, in [0, 1]
    entropy: float  # entropy of the answer-token distribution, nats or bits (consistent within a run)


@dataclass(frozen=True)
class Prefix:
    """One candidate stop point within one generated trajectory.

    total_steps is required (not optional) so normalized position
    (step_index / total_steps) is always computable -- matching across
    trajectories of very different lengths without controlling for depth
    confounds "different history" with "different amount of reasoning
    done," which is not the claim this study makes. See
    normalized_position and the position_tol argument to matches().
    """

    problem_id: str
    trajectory_id: str  # which of the k diverse trajectories for this problem
    step_index: int
    total_steps: int
    state: ObservableState

    def __post_init__(self):
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {self.total_steps}")
        if not (0 <= self.step_index <= self.total_steps):
            raise ValueError(
                f"step_index {self.step_index} out of range for total_steps {self.total_steps}"
            )

    @property
    def normalized_position(self) -> float:
        return self.step_index / self.total_steps


def state_distance(a: ObservableState, b: ObservableState,
                    confidence_scale: float = 1.0, entropy_scale: float = 1.0) -> float | None:
    """Continuous distance between two observable states, for treating H2
    as an empirical correlation (PDI vs distance) rather than relying on
    the discrete L1/L2/L3 buckets alone -- see methodology fix #1.

    Returns None if answers differ: distance is only meaningful for pairs
    that already agree on the answer (L1's own criterion), since a
    same-problem pair with different answers isn't really "close" in any
    sense this study cares about, and folding that into a single
    continuous scale would conflate "different answer" with "same answer,
    slightly different confidence/entropy."
    """
    if a.answer != b.answer:
        return None
    return float(
        np.hypot((a.confidence - b.confidence) / confidence_scale,
                  (a.entropy - b.entropy) / entropy_scale)
    )


def matches(a: ObservableState, b: ObservableState, level: MatchLevel,
            confidence_tol: float = 0.10, entropy_tol: float = 0.15) -> bool:
    """Whether two observable states match at the given strictness level.

    L1: answer only. L2: + confidence within tolerance. L3: + entropy too.
    Levels are STRICTLY NESTED as MATCH CRITERIA (L3 match => L2 match =>
    L1 match for the SAME pair) -- but this does NOT imply
    mean(PDI over L1 pairs) > mean(PDI over L2 pairs) > mean(PDI over L3
    pairs), because L1/L2/L3 select DIFFERENT SUBSETS of pairs, not the
    same pairs measured three ways. H2 (PDI decreases as matching gets
    stricter) is an empirical hypothesis about those different subsets,
    not a mathematical consequence of this nesting -- test it with
    state_distance() vs PDI as a continuous relationship, which is the
    stronger and correct way to check it. (This corrects a wrong claim in
    an earlier version of this docstring, caught in external review.)
    """
    if a.answer != b.answer:
        return False
    if level == "L1":
        return True
    if abs(a.confidence - b.confidence) > confidence_tol:
        return False
    if level == "L2":
        return True
    if abs(a.entropy - b.entropy) > entropy_tol:
        return False
    return True  # L3


def find_matched_pairs(
    prefixes: list[Prefix], level: MatchLevel,
    confidence_tol: float = 0.10, entropy_tol: float = 0.15,
    position_tol: float | None = 0.15,
) -> list[tuple[Prefix, Prefix]]:
    """All pairs of prefixes from the SAME problem but DIFFERENT
    trajectories whose observable state matches at the given level.

    position_tol (default 0.15, i.e. within 15% of normalized trajectory
    length): without this, a match between step 15/20 and step 15/90 gets
    counted as "different history, same state" when it may really just be
    "different amount of reasoning done" -- a confound, not path
    dependence. Pass position_tol=None to disable this constraint
    entirely, e.g. for an explicit robustness experiment checking whether
    the depth constraint itself changes the result (recommended as a
    planned follow-up, not the default first pilot).

    O(n^2) within each problem's prefix set -- fine at pilot scale (a
    handful of problems, a few trajectories/depths each). Revisit if this
    becomes the bottleneck at full scale.
    """
    by_problem: dict[str, list[Prefix]] = {}
    for p in prefixes:
        by_problem.setdefault(p.problem_id, []).append(p)

    pairs = []
    for group in by_problem.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.trajectory_id == b.trajectory_id:
                    continue  # same history -- not a path-dependence test
                if position_tol is not None:
                    if abs(a.normalized_position - b.normalized_position) > position_tol:
                        continue
                if matches(a.state, b.state, level, confidence_tol, entropy_tol):
                    pairs.append((a, b))
    return pairs


def stratified_sample_pairs(
    pairs: list[tuple[Prefix, Prefix]],
    budget: int,
    confidence_bounds: tuple[float, ...] = (0.0, 0.5, 0.85, 1.01),
    seed: int = 0,
) -> list[tuple[Prefix, Prefix]]:
    """Sample up to `budget` matched pairs, spread across confidence
    strata rather than taking the first `budget` found.

    Motivation (found empirically during the Candidate B micro-pilot,
    2026-08-28): once a trajectory converges on an answer, essentially
    every subsequent step force-extracts to that SAME answer at HIGH
    confidence -- so two independently-converged trajectories trivially
    produce dozens of L1 matches (17,147 from 1,364 prefixes in the
    micro-pilot run). Naive `pairs[:budget]` selection, given typical
    enumeration order, ends up drawing almost entirely from this
    converged/high-confidence regime, and every one of the first 6
    branched pairs showed PDI=0.000 exactly (unanimous branches on both
    sides) -- not because state sufficiency is false, but because a
    fully-converged point IS close to a tautology to branch from.

    This is NOT a reason to exclude high-confidence pairs -- a match at
    high confidence/low entropy is a legitimate L2/L3 test point, and
    PDI=0 there is a real, informative result (state sufficiency holding
    in the converged regime). The actual problem is budget allocation: a
    small fixed budget spent entirely in one regime has zero power to
    detect divergence in the regime path dependence is most plausible in
    (medium confidence, still-uncertain points) -- H1 needs coverage
    across the confidence range to be tested at all, not just answered in
    the easy corner of it.

    Buckets by the PAIR's mean confidence (both prefixes already matched,
    so their confidences are close by construction at L2/L3; at L1 they
    may differ more, hence averaging rather than requiring both in the
    same bucket). Within each confidence bucket, pairs are further grouped
    by problem_id and sampled evenly across problems before being sampled
    evenly across confidence strata -- found necessary empirically (micro-
    pilot run #2, 2026-08-28): confidence stratification alone still let
    one problem's pairs dominate a bucket whenever that problem simply had
    more matched pairs in it than others, e.g. notebook 1/3's entire
    8-pair budget landed on only 2 of its 3 problems even with confidence
    strata working exactly as designed. Both dimensions are stratified so
    a small budget can't collapse onto one problem OR one confidence
    regime. Samples as evenly as possible across non-empty groups at each
    level; a group with fewer pairs than its even share just contributes
    all of what it has, and the remainder is redistributed -- never pads
    with duplicates or silently shrinks the requested budget without
    reason.
    """
    if budget <= 0 or not pairs:
        return []
    n_strata = len(confidence_bounds) - 1
    buckets: list[list[tuple[Prefix, Prefix]]] = [[] for _ in range(n_strata)]
    for a, b in pairs:
        mean_conf = (a.state.confidence + b.state.confidence) / 2.0
        for s in range(n_strata):
            lo, hi = confidence_bounds[s], confidence_bounds[s + 1]
            if lo <= mean_conf < hi:
                buckets[s].append((a, b))
                break

    rng = np.random.default_rng(seed)

    def _sample_evenly(items: list[tuple[Prefix, Prefix]], take_budget: int) -> list[tuple[Prefix, Prefix]]:
        """Group `items` by problem_id, shuffle each group, then draw as
        evenly as possible across non-empty groups up to `take_budget`."""
        if take_budget <= 0 or not items:
            return []
        by_problem: dict[str, list[tuple[Prefix, Prefix]]] = {}
        for pair in items:
            by_problem.setdefault(pair[0].problem_id, []).append(pair)
        for group in by_problem.values():
            rng.shuffle(group)
        groups = list(by_problem.values())

        out: list[tuple[Prefix, Prefix]] = []
        remaining = take_budget
        active_groups = [g for g in groups if g]
        while remaining > 0 and active_groups:
            share = max(1, remaining // len(active_groups))
            next_active = []
            for g in active_groups:
                take = min(share, len(g), remaining)
                out.extend(g[:take])
                del g[:take]
                remaining -= take
                if g and remaining > 0:
                    next_active.append(g)
                if remaining <= 0:
                    break
            active_groups = next_active
        return out

    for bucket in buckets:
        rng.shuffle(bucket)

    selected: list[tuple[Prefix, Prefix]] = []
    remaining_budget = budget
    active = [i for i in range(n_strata) if buckets[i]]
    while remaining_budget > 0 and active:
        share = max(1, remaining_budget // len(active))
        next_active = []
        for i in active:
            take = min(share, len(buckets[i]), remaining_budget)
            selected.extend(_sample_evenly(buckets[i], take))
            buckets[i] = buckets[i][take:]
            remaining_budget -= take
            if buckets[i] and remaining_budget > 0:
                next_active.append(i)
            if remaining_budget <= 0:
                break
        active = next_active
    return selected


@dataclass(frozen=True)
class BranchOutcome:
    """One branched continuation's outcome, keeping the raw
    parseable-or-not status visible instead of silently dropping failures
    -- methodology fix #3. `answer` is None iff extraction failed."""

    answer: str | None


@dataclass(frozen=True)
class BranchSet:
    """N branched continuations from one prefix, with parse-rate
    accounting surfaced explicitly rather than discovered by accident.

    Per the pre-registered rule (fix #3, option A as primary + B as
    sensitivity): the main PDI analysis should exclude any prefix whose
    branch set falls below `min_parse_rate`, so a difference in extraction
    FAILURE rate between two matched prefixes can't masquerade as a
    difference in their future ANSWER distributions. A separate sensitivity
    analysis (option B) can instead treat "UNPARSEABLE" as its own outcome
    category via distribution_with_failures() -- run both, report both.
    """

    outcomes: list[BranchOutcome]

    @property
    def n_total(self) -> int:
        return len(self.outcomes)

    @property
    def n_parsed(self) -> int:
        return sum(1 for o in self.outcomes if o.answer is not None)

    @property
    def parse_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_parsed / self.n_total

    def parsed_answers(self) -> list[str]:
        return [o.answer for o in self.outcomes if o.answer is not None]

    def passes_parse_gate(self, min_parse_rate: float = 0.80) -> bool:
        return self.parse_rate >= min_parse_rate

    def distribution(self) -> dict[str, float]:
        """Option A: distribution over PARSED answers only. Use only after
        passes_parse_gate() -- calling this on a low-parse-rate branch set
        without checking the gate first silently reintroduces the bias
        fix #3 exists to prevent."""
        return answer_distribution(self.parsed_answers())

    def distribution_with_failures(self) -> dict[str, float]:
        """Option B (sensitivity analysis): UNPARSEABLE as an explicit
        outcome category, so extraction-failure-rate differences show up
        as part of the compared distributions instead of being hidden."""
        outs = [o.answer if o.answer is not None else "UNPARSEABLE" for o in self.outcomes]
        return answer_distribution(outs)


def answer_distribution(answers: list[str]) -> dict[str, float]:
    """Empirical distribution over a list of answer strings (already
    resolved -- None-dropping or None->"UNPARSEABLE" handling happens in
    BranchSet, one level up, where the parse-rate context is visible).
    """
    counts: dict[str, int] = {}
    for a in answers:
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError("answer_distribution: no parseable answers in the branch set")
    return {k: v / total for k, v in counts.items()}


def _align(p: dict[str, float], q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Two distributions over possibly-different answer-string supports,
    aligned onto their union support (missing mass = 0)."""
    keys = sorted(set(p) | set(q))
    pv = np.array([p.get(k, 0.0) for k in keys])
    qv = np.array([q.get(k, 0.0) for k in keys])
    return pv, qv


def pdi(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two future-answer distributions.
    Bounded in [0, 1] when using log base 2 (scipy's default), which
    scipy.spatial.distance.jensenshannon returns as a DISTANCE (sqrt of
    the divergence) -- squared here to report the divergence itself,
    matching PROPOSAL_CANDIDATE_B.md's D_JS notation.
    """
    pv, qv = _align(dist_a, dist_b)
    d = jensenshannon(pv, qv, base=2)
    return float(d ** 2) if not np.isnan(d) else 0.0  # nan when both dists are identical incl. degenerate


def noise_floor_sample(answers: list[str], rng: np.random.Generator) -> float:
    """One draw from the intra-prefix noise-floor DIAGNOSTIC: split a
    SINGLE prefix's own N branches randomly in half, compute PDI between
    the halves. Pure sampling variance -- no history difference at all,
    since both halves came from literally the same prefix.

    STATUS (post-review): this remains useful as a DESCRIPTIVE diagnostic
    ("how much pure branching noise exists at this N?") but is NOT used as
    the hypothesis-test null anymore -- see pair_permutation_test() for
    that. Reason: pooling many correlated same-prefix resamples (e.g.
    1000 from one prefix) against a handful of independent matched pairs
    inflates the apparent sample size and produces overconfident p-values
    (verified: p=0.0002 from only 8 real pairs in the pre-fix version).
    """
    n = len(answers)
    if n < 4:
        raise ValueError(f"need >=4 branches to split in half meaningfully, got {n}")
    idx = rng.permutation(n)
    half = n // 2
    a_half = [answers[i] for i in idx[:half]]
    b_half = [answers[i] for i in idx[half:]]
    return pdi(answer_distribution(a_half), answer_distribution(b_half))


def noise_floor_distribution(
    answers: list[str], reps: int = 1000, seed: int = 0,
) -> np.ndarray:
    """DIAGNOSTIC ONLY (see noise_floor_sample docstring) -- the full
    within-prefix branching-noise distribution for one prefix's branch
    set. Report this per prefix for context; do not pool it across
    prefixes and feed it into a single global significance test."""
    rng = np.random.default_rng(seed)
    return np.array([noise_floor_sample(answers, rng) for _ in range(reps)])


@dataclass
class PairPermutationResult:
    """One matched pair's own, self-contained significance test.

    Null: prefix A's and prefix B's branches are exchangeable -- i.e. they
    really do come from the same underlying future-answer distribution,
    and the observed cross-history PDI is just what you'd see from
    splitting that combined pool into groups of these two sizes by chance.
    Built ONLY from this pair's own two branch sets, so it is a legitimate
    independent unit of evidence -- this is the fix for methodology issue
    #2 (pseudoreplication in the pooled global test).
    """

    observed_pdi: float
    null_mean: float
    effect_size: float  # observed - null mean
    p_value: float  # one-sided: P(null resplit PDI >= observed)
    n_a: int
    n_b: int
    reps: int


def pair_permutation_test(
    branch_set_a: BranchSet, branch_set_b: BranchSet,
    min_parse_rate: float = 0.80, reps: int = 2000, seed: int = 0,
) -> PairPermutationResult:
    """The correct per-pair unit of analysis (methodology fix #2): pool
    BOTH branch sets' PARSED answers, repeatedly resplit into groups of
    the two branch sets' original sizes, compute PDI on each resplit. The
    resulting null distribution is built entirely from this pair's own
    data -- no borrowing of another prefix's noise samples, no inflated N.

    Applies the parse-rate gate (methodology fix #3, option A) before
    testing: raises if either branch set is below min_parse_rate, since
    testing on an under-parsed branch set silently reintroduces the
    extraction-failure-rate confound fix #3 exists to prevent. Call
    BranchSet.distribution_with_failures()-based analysis separately for
    the option-B sensitivity check instead of lowering this gate.
    """
    for name, bs in (("a", branch_set_a), ("b", branch_set_b)):
        if not bs.passes_parse_gate(min_parse_rate):
            raise ValueError(
                f"branch_set_{name} parse rate {bs.parse_rate:.0%} is below "
                f"min_parse_rate={min_parse_rate:.0%} -- excluded from the main "
                f"analysis per methodology fix #3 (run the option-B sensitivity "
                f"analysis separately if you need this pair)"
            )

    a_answers = branch_set_a.parsed_answers()
    b_answers = branch_set_b.parsed_answers()
    observed = pdi(answer_distribution(a_answers), answer_distribution(b_answers))

    pooled = a_answers + b_answers
    n_a, n_b = len(a_answers), len(b_answers)
    rng = np.random.default_rng(seed)
    null = np.empty(reps)
    for i in range(reps):
        idx = rng.permutation(len(pooled))
        a_resplit = [pooled[k] for k in idx[:n_a]]
        b_resplit = [pooled[k] for k in idx[n_a:]]
        null[i] = pdi(answer_distribution(a_resplit), answer_distribution(b_resplit))

    p_value = float((null >= observed).sum() + 1) / (reps + 1)  # +1: avoid p=0

    return PairPermutationResult(
        observed_pdi=observed,
        null_mean=float(null.mean()),
        effect_size=observed - float(null.mean()),
        p_value=p_value,
        n_a=n_a,
        n_b=n_b,
        reps=reps,
    )


@dataclass
class CombinedResult:
    """K independent per-pair tests combined into one overall verdict via
    Fisher's method -- legitimate because each input p-value came from an
    independent permutation test on non-overlapping data (a different
    matched pair's own branches), unlike the old pooled-noise design."""

    n_pairs: int
    per_pair: list[PairPermutationResult] = field(default_factory=list)
    combined_statistic: float = 0.0
    combined_p_value: float = 1.0
    mean_effect_size: float = 0.0
    frac_individually_significant: float = 0.0  # at alpha=0.05, purely descriptive


def combine_pair_results(results: list[PairPermutationResult], alpha: float = 0.05) -> CombinedResult:
    if len(results) == 0:
        raise ValueError("no pair results to combine")
    p_values = [r.p_value for r in results]
    stat, combined_p = combine_pvalues(p_values, method="fisher")
    return CombinedResult(
        n_pairs=len(results),
        per_pair=results,
        combined_statistic=float(stat),
        combined_p_value=float(combined_p),
        mean_effect_size=float(np.mean([r.effect_size for r in results])),
        frac_individually_significant=float(np.mean([r.p_value < alpha for r in results])),
    )


@dataclass(frozen=True)
class ContinuousRegressionResult:
    """H2 tested the statistically stronger way (see `matches()`'s own
    docstring): PDI regressed against continuous state_distance across ALL
    matched pairs pooled, regardless of which discrete L1/L2/L3 bucket they
    fell in, instead of three separate small-sample discrete comparisons.

    Requires per-pair `state_distance` (see `state_distance()`) to have
    been persisted alongside each pair's PDI at collection time -- the
    discrete-only full pilot did not do this (see
    CANDIDATE_B_PREREGISTRATION.md 5.4); this function is for pairs
    collected after that gap was fixed.
    """

    n_pairs: int
    slope: float          # PDI per unit of state_distance -- the H2 effect
    intercept: float      # predicted PDI at state_distance == 0 (should be ~0 under H0)
    r_value: float
    p_value: float        # two-sided test of slope != 0
    std_err: float


def regress_pdi_vs_state_distance(
    pairs: list[tuple[float, float]],
) -> ContinuousRegressionResult:
    """`pairs` is a list of (state_distance, observed_pdi) tuples, pooled
    across matching levels and strata -- deliberately not bucketed, since
    bucketing is exactly the weaker approach this function replaces.

    A positive, significant slope is evidence FOR path dependence scaling
    with state distance (H1): prefixes matched more loosely (larger
    state_distance) diverge more in their future answers. A slope
    indistinguishable from zero, with an intercept near zero, is evidence
    for H0 across the whole continuous range, not just at the three
    discrete strictness levels the pre-registered design tested.
    """
    if len(pairs) < 3:
        raise ValueError(
            f"need at least 3 pairs to fit a regression, got {len(pairs)} -- "
            "this is an even lower bar than a single discrete bucket's pair "
            "count, but 3 is still too few to trust; treat n<10 results as "
            "a pilot check, not a real test"
        )
    distances = np.array([d for d, _ in pairs])
    pdis = np.array([p for _, p in pairs])
    fit = linregress(distances, pdis)
    return ContinuousRegressionResult(
        n_pairs=len(pairs),
        slope=float(fit.slope),
        intercept=float(fit.intercept),
        r_value=float(fit.rvalue),
        p_value=float(fit.pvalue),
        std_err=float(fit.stderr),
    )
