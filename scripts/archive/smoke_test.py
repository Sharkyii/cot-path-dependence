#!/usr/bin/env python3
"""End-to-end wiring smoke test -- NOT the real pilot.

Runs the full dataset -> split -> segment -> truncate -> forced-extract ->
grade pipeline using MockBackend instead of a real model, on real MATH-500
and GSM8K problems pulled from HF. Purpose: catch integration bugs (wrong
argument order, mismatched types between modules, off-by-one in step
indices) for free, before spending any GPU time on the real pilot script
(scripts/pilot.py). A green run here says nothing about segmentation
quality or overthinking on the real model -- only that the code paths
connect correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from early_stop.cost import CostTracker
from early_stop.datasets import load_domain
from early_stop.extraction import MockBackend
from early_stop.grading import forced_extract_and_grade
from early_stop.segmentation import compute_segmentation_stats, split_into_steps, step_prefix
from early_stop.splits import SplitConfig, SplitManager, assert_disjoint

FAKE_TRACE_TEMPLATE = """Let me think about this problem.

First, I'll consider the setup.

Working through the details, I compute the intermediate result.

Double-checking my work here.

So the final answer is \\boxed{{{answer}}}."""


def main() -> None:
    print("=== Smoke test: dataset loading ===")
    math_problems = load_domain("math500")[:5]
    gsm8k_problems = load_domain("gsm8k")[:5]
    print(f"Loaded {len(math_problems)} math500 problems, {len(gsm8k_problems)} gsm8k problems")

    print("\n=== Smoke test: split manager + disjointness ===")
    cfg = SplitConfig(pilot_frac=0.1, difficulty_frac=0.2, dev_frac=0.15, test_frac=0.55, seed=0)
    mgr = SplitManager(config=cfg)
    all_ids = [p.problem_id for p in math_problems]
    dev_ids = mgr.filter_split("math500", all_ids, "dev") if any(
        mgr.split_for("math500", pid) == "dev" for pid in all_ids
    ) else []
    test_ids = mgr.filter_split("math500", all_ids, "test") if any(
        mgr.split_for("math500", pid) == "test" for pid in all_ids
    ) else []
    assert_disjoint(dev_ids, test_ids)
    print(f"Split check OK. dev={len(dev_ids)} test={len(test_ids)} (of {len(all_ids)} sampled)")

    print("\n=== Smoke test: segmentation + truncation + forced extraction + grading ===")
    # The math forced-answer suffix primes an OPEN \boxed{ in the prompt, so
    # what actually gets graded (fx.full_text) is the primed brace closed by
    # whatever the model's forced continuation says -- NOT whatever a prior,
    # independent \boxed{} the model wrote earlier in its own reasoning
    # said. That's intentional: forced extraction is "what would the model
    # say if forced to answer right now," and the completion is the thing
    # that answers that. So the mock backend below must actually return the
    # correct closing content when (and only when) the prefix already
    # contains enough of the trace to know the answer -- it returns a
    # deliberately wrong default otherwise, so a truncated prefix (frac=0.4,
    # missing the trace's final boxed line) is expected to grade incorrect.
    cost = CostTracker()
    backend = MockBackend(default="definitely_the_wrong_answer}.")
    traces = []
    results = []
    for problem in math_problems:
        trace = FAKE_TRACE_TEMPLATE.format(answer=problem.gold_answer)
        traces.append(trace)
        backend.canned_completions[f"\\boxed{{{problem.gold_answer}}}"] = f"{problem.gold_answer}}}."

        steps = split_into_steps(trace)
        n_steps = len(steps)
        assert n_steps == 5, f"expected 5 steps, got {n_steps}"

        for frac in (0.4, 1.0):
            k = max(1, round(n_steps * frac))
            prefix = step_prefix(steps, k)
            fx, g = forced_extract_and_grade(backend, problem, prefix, step_index=k)
            cost.record("smoke_forced_extraction", len(fx.prompt.split()), len(fx.completion.split()))
            results.append((problem.problem_id, frac, g.correct))

    seg_stats = compute_segmentation_stats("math500_smoke", traces)
    print(seg_stats.summary())
    assert seg_stats.passes_gate(), "smoke-test synthetic traces should trivially pass the gate"

    full_trace_results = [correct for (_, frac, correct) in results if frac == 1.0]
    print(f"Accuracy at fraction=1.0 (full trace): {sum(full_trace_results)}/{len(full_trace_results)}")
    assert all(full_trace_results), "full-trace forced extraction should recover the boxed gold answer every time"

    print("\n" + cost.summary())
    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
