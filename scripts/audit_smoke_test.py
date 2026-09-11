"""Zero-cost, zero-GPU smoke test for the Tier 1 audit pipeline
(REVISION_PLAN.md). Exercises the EXACT SAME orchestration functions the
real Modal scripts call -- build_prefix_pool, load_prefix_pool, the
JSON round-trip between them, matching, sample_and_branch_with_telemetry,
empty_rate_at_cutoff -- against MockScoredBackend instead of a real
model, so a bug in the plumbing (a wrong dict key, a JSON round-trip
mismatch, an off-by-one in matching) is caught in well under a second,
for $0, before any Modal spend.

This is NOT a substitute for the cheap-diagnostic-run step each Modal
script's own docstring recommends (--n-trajectories 2, etc.) -- that
step checks the REAL model/GPU path (precision, chat template, actual
generation quality). This script only checks the Python plumbing around
it: the part that's free and fast to get right before paying to find out
the hard way.

Run:
    .venv/bin/python scripts/audit_smoke_test.py

Always run this FIRST, before touching Modal, and again after any edit
to early_stop/candidate_b_pipeline.py.
"""
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from early_stop.backend import MockScoredBackend
from early_stop.candidate_b_pipeline import (
    build_prefix_pool,
    empty_rate_at_cutoff,
    load_prefix_pool,
    sample_and_branch_with_telemetry,
)
from early_stop.path_dependence import find_matched_pairs, stratified_sample_pairs

PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(label: str, cond: bool):
    print(f"  [{PASS if cond else FAIL}] {label}")
    if not cond:
        _failures.append(label)


def smoke_test_phase1_and_json_roundtrip():
    print("Phase 1: build_prefix_pool + JSON round-trip (phase1 script -> volume -> phase2/3 script)")
    trace = (
        "Let me set up the equation.\n\n"
        "We have x + 5 = 12.\n\n"
        "So the final answer is \\boxed{7}."
    )
    # default=(completion, confidence, entropy): every forced-extraction
    # call returns the SAME canned state, on purpose -- this is what makes
    # matching trivially find pairs below without needing per-trajectory
    # mock plumbing. Real traces vary; the plumbing doesn't care.
    backend = MockScoredBackend(default=(" 7}.", 0.80, 0.50))
    backend.continue_default = trace  # generate() also reads continue_default -- see backend.py's MockScoredBackend.generate

    calls = []
    pool = build_prefix_pool(
        backend, "smoke_prob0", "solve for x", n_trajectories=4,
        max_new_tokens_trace=999, max_new_tokens_score=48,
        on_trajectory_done=lambda tid, n_scored, n_unparse, elapsed: calls.append((tid, n_scored, n_unparse)),
    )
    check("on_trajectory_done fires once per trajectory", len(calls) == 4)
    check("every trajectory scores all 3 step boundaries", all(n == 3 for _, n, _ in calls))
    check("zero unparseable (canned completion always closes the box)", all(u == 0 for _, _, u in calls))
    check("pool holds 4 trajectories", len(pool["trajectories"]) == 4)
    check("pool holds 12 scored prefixes (4 trajectories x 3 steps)", len(pool["prefixes"]) == 12)

    # This IS what happens between the phase1 Modal script (writes to the
    # volume as JSON) and the phase2/3 script (reads it back) -- if this
    # round-trip silently drops or mangles a field, the real run would too.
    roundtripped = json.loads(json.dumps(pool))
    problem_id, base_prompt, trajectories, prefixes = load_prefix_pool(roundtripped)
    check("problem_id survives the round-trip", problem_id == "smoke_prob0")
    check("base_prompt survives the round-trip", base_prompt == "solve for x")
    check("trajectories dict survives the round-trip", len(trajectories) == 4)
    check("prefixes reconstruct as real Prefix objects", len(prefixes) == 12)
    check("reconstructed state keeps the canned confidence", prefixes[0].state.confidence == 0.80)
    check("reconstructed state keeps the canned entropy", prefixes[0].state.entropy == 0.50)

    return base_prompt, trajectories, prefixes


def smoke_test_matching_and_branching(base_prompt, trajectories, prefixes):
    print("\nPhase 2+3: matching -> stratified sampling -> sample_and_branch_with_telemetry")
    l1_pairs = find_matched_pairs(prefixes, "L1", position_tol=None)
    l2_pairs = find_matched_pairs(prefixes, "L2", position_tol=None)
    check("L1 finds matched pairs (identical canned state across trajectories)", len(l1_pairs) > 0)
    check("L2 finds matched pairs too (confidence is identical everywhere in this scenario)", len(l2_pairs) > 0)

    sampled = stratified_sample_pairs(l1_pairs, budget=2, seed=0)
    check("stratified_sample_pairs respects the budget", 0 < len(sampled) <= 2)

    branch_backend = MockScoredBackend()
    branch_backend.continue_default = " so the final answer is \\boxed{7}."
    seen = []
    results = sample_and_branch_with_telemetry(
        branch_backend, base_prompt, trajectories, sampled, n_branches=5, max_new_tokens=100,
        on_pair_done=lambda entry: seen.append(entry),
    )
    check("on_pair_done fires once per sampled pair", len(seen) == len(sampled))
    check(
        "each result carries both sides' telemetry at length n_branches",
        all(len(r["side_a_telemetry"]) == 5 and len(r["side_b_telemetry"]) == 5 for r in results),
    )
    check(
        "telemetry entries carry a resolved boxed_close_token_index (box closes in this scenario)",
        all(r["side_a_telemetry"][0]["boxed_close_token_index"] is not None for r in results),
    )
    check(
        "result JSON-serializes cleanly (this is what actually gets written to the volume)",
        _json_serializable(results),
    )
    return results


def smoke_test_empty_rate_at_cutoff():
    print("\nempty_rate_at_cutoff (the core of the censoring audit's analysis)")
    telemetry = [
        {"boxed_close_token_index": 200},
        {"boxed_close_token_index": 1800},
        {"boxed_close_token_index": None},
        {"boxed_close_token_index": 2900},
    ]
    at_1500 = empty_rate_at_cutoff(telemetry, 1500)
    at_3000 = empty_rate_at_cutoff(telemetry, 3000)
    check("empty@1500 counts the 1800/None/2900 entries as empty (3/4 = 0.75)", at_1500 == 0.75)
    check("empty@3000 counts only the None entry as empty (1/4 = 0.25)", at_3000 == 0.25)
    check("empty_rate_at_cutoff([]) returns None, does not crash", empty_rate_at_cutoff([], 1500) is None)


def smoke_test_unclosed_box_telemetry():
    print("\nUnclosed-box branch: boxed_close_token_index must be None, not a crash")
    backend = MockScoredBackend()
    backend.continue_default = " let me think further and \\boxed{unclosed"
    from early_stop.candidate_b_pipeline import branch_naturally_with_telemetry
    _branches, telemetry = branch_naturally_with_telemetry(backend, "prompt", "prefix", n=3, max_new_tokens=50)
    check("all 3 branches flagged as never closing", all(t["boxed_close_token_index"] is None for t in telemetry))
    check("unclosed box does not parse as a genuine answer", all(t["answer"] is None for t in telemetry))


def _json_serializable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def main():
    print("=" * 78)
    print("AUDIT SMOKE TEST -- $0, no GPU, no Modal. Run this before spending anything.")
    print("=" * 78)
    base_prompt, trajectories, prefixes = smoke_test_phase1_and_json_roundtrip()
    smoke_test_matching_and_branching(base_prompt, trajectories, prefixes)
    smoke_test_empty_rate_at_cutoff()
    smoke_test_unclosed_box_telemetry()

    print("\n" + "=" * 78)
    if _failures:
        print(f"SMOKE TEST FAILED: {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        print("=" * 78)
        raise SystemExit(1)
    print("ALL SMOKE TESTS PASSED. Pipeline plumbing is sound.")
    print("Next: run each Modal script's own cheap diagnostic mode (see its docstring)")
    print("to check the REAL GPU/model path before committing to the full paid run.")
    print("=" * 78)


if __name__ == "__main__":
    main()
