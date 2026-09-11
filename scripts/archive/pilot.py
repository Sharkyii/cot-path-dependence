#!/usr/bin/env python3
"""Pilot gate (project plan §5) -- run this BEFORE writing any more pipeline
code. It is the highest-priority step: if segmentation or truncation-room
fails here, the step-level early-exit design breaks and nothing downstream
is worth building yet.

Usage (debug locally on the small model first, zero cost, catches code bugs):
    python scripts/pilot.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --domains math500 gsm8k commonsenseqa strategyqa \\
        --n-per-domain 8 --max-new-tokens 1024

Then, with no code changes, rerun for real on rented GPU:
    python scripts/pilot.py \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --domains math500 gsm8k commonsenseqa strategyqa \\
        --n-per-domain 40 --max-new-tokens 4096

For each domain this:
  1. Draws n_per_domain problems from a dedicated PILOT bucket (SplitConfig
     pilot_frac) so these problem ids never resurface in difficulty/dev/test
     later -- reusing them would leak pilot-time signal into the real study.
  2. Generates one full CoT trace per problem (fixed temperature).
  3. Reports segmentation stats (steps/problem, blank-line-split gate) via
     early_stop.segmentation.SegmentationStats.passes_gate().
  4. Reports accuracy-vs-fraction-of-trace-kept under naive step truncation
     + forced extraction, for a human to eyeball "is there room to save
     tokens" (project plan §5's non-trivial-curve requirement is judged by
     a person, not a hard threshold -- 30-50 problems is too few to trust
     an automated cutoff here).
  5. Writes per-domain JSON results + prints a go/no-go summary line.

Requires: pip install torch transformers (see requirements.txt Phase 1+).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from early_stop.backend import ForcedExtractionBackendAdapter, HFBackend
from early_stop.cost import CostTracker
from early_stop.datasets import Problem, load_domain
from early_stop.grading import forced_extract_and_grade
from early_stop.segmentation import compute_segmentation_stats, split_into_steps, step_prefix
from early_stop.splits import SplitConfig, SplitManager

TRUNCATION_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)

MATH_PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MC_PROMPT_SUFFIX = "\n\nPlease reason step by step, then give your final answer as a single letter."
YESNO_PROMPT_SUFFIX = "\n\nPlease reason step by step, then answer with a single word: yes or no."


def build_prompt(problem: Problem) -> str:
    if problem.domain in {"math500", "gsm8k"}:
        return problem.question + MATH_PROMPT_SUFFIX
    if problem.domain == "commonsenseqa":
        choices_text = "\n".join(f"({k}) {v}" for k, v in sorted(problem.choices.items()))
        return f"{problem.question}\n{choices_text}{MC_PROMPT_SUFFIX}"
    if problem.domain == "strategyqa":
        return problem.question + YESNO_PROMPT_SUFFIX
    raise ValueError(f"No prompt template for domain {problem.domain!r}")


def word_tokens_estimate(text: str) -> int:
    # Cheap proxy for logging when we don't want to re-tokenize just for a
    # progress print; real token accounting for the cost report should use
    # tokenizer output lengths, wired up alongside the calibration script.
    return len(text.split())


def run_pilot_for_domain(
    domain: str,
    backend: HFBackend,
    n_per_domain: int,
    max_new_tokens: int,
    split_mgr: SplitManager,
    cost: CostTracker,
    out_dir: Path,
) -> dict:
    print(f"\n=== Pilot: {domain} ===")
    all_problems = load_domain(domain)
    pilot_problems = [p for p in all_problems if split_mgr.split_for(domain, p.problem_id) == "pilot"]
    pilot_problems = pilot_problems[:n_per_domain]
    if len(pilot_problems) < n_per_domain:
        print(
            f"WARNING: only {len(pilot_problems)}/{n_per_domain} pilot problems "
            f"available for {domain} (pilot_frac too small for this dataset size?)"
        )

    extraction_backend = ForcedExtractionBackendAdapter(backend)
    traces: list[str] = []
    per_problem_records = []

    for i, problem in enumerate(pilot_problems):
        t0 = time.time()
        prompt = build_prompt(problem)
        trace = backend.generate(prompt, max_new_tokens=max_new_tokens)
        cost.record("pilot_main_generation", word_tokens_estimate(prompt), word_tokens_estimate(trace))
        traces.append(trace)

        steps = split_into_steps(trace)
        n_steps = len(steps)

        trunc_results = {}
        for frac in TRUNCATION_FRACTIONS:
            k = max(1, round(n_steps * frac)) if n_steps > 0 else 0
            prefix_text = prompt + "\n" + step_prefix(steps, k) if k > 0 else prompt
            fx, g = forced_extract_and_grade(extraction_backend, problem, prefix_text, step_index=k)
            cost.record(
                "pilot_forced_extraction",
                word_tokens_estimate(fx.prompt),
                word_tokens_estimate(fx.completion),
            )
            trunc_results[frac] = g.correct

        per_problem_records.append(
            {
                "problem_id": problem.problem_id,
                "n_steps": n_steps,
                "truncation_correct_by_fraction": trunc_results,
            }
        )
        print(
            f"  [{i+1}/{len(pilot_problems)}] {problem.problem_id}: "
            f"{n_steps} steps, {time.time()-t0:.1f}s"
        )

    seg_stats = compute_segmentation_stats(domain, traces)
    print(seg_stats.summary())

    trunc_curve = {}
    for frac in TRUNCATION_FRACTIONS:
        results_at_frac = [r["truncation_correct_by_fraction"][frac] for r in per_problem_records]
        trunc_curve[frac] = sum(results_at_frac) / len(results_at_frac) if results_at_frac else 0.0
    print("Truncation curve (fraction-of-steps-kept -> accuracy):")
    for frac in TRUNCATION_FRACTIONS:
        print(f"    {frac:.1f}: {trunc_curve[frac]:.2%}")

    domain_result = {
        "domain": domain,
        "n_problems": len(pilot_problems),
        "segmentation_stats": {
            "mean_steps_per_trace": seg_stats.mean_steps_per_trace,
            "frac_traces_lt_3_steps": seg_stats.frac_traces_lt_3_steps,
            "mean_step_word_length": seg_stats.mean_step_word_length,
            "segmentation_gate_pass": seg_stats.passes_gate(),
        },
        "truncation_curve": trunc_curve,
        "per_problem": per_problem_records,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pilot_{domain}.json"
    out_path.write_text(json.dumps(domain_result, indent=2))
    print(f"Wrote {out_path}")
    return domain_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model name, e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--domains", nargs="+", default=["math500", "gsm8k", "commonsenseqa", "strategyqa"])
    parser.add_argument("--n-per-domain", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6, help="Fixed sampling temperature (project plan §8) -- DeepSeek-R1 recommends 0.5-0.7, avoid 0 (greedy can degenerate into repetition loops on these models)")
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "8bit", "4bit"], help="Weight precision. 'auto' picks the best that fits VRAM with KV headroom. NOTE: 4bit/8bit perturb logits -- fine for the pilot gate, but a confound for the logit-based signals in the main study")
    parser.add_argument("--param-count-b", type=float, default=7.0, help="Model size in billions, used for the VRAM fit check (7.0 for the 7B, 1.5 for the 1.5B)")
    parser.add_argument("--pilot-frac", type=float, default=0.10, help="Fraction of each domain's pool reserved for pilot -- burned, never reused for difficulty/dev/test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "pilot")
    args = parser.parse_args()

    split_cfg = SplitConfig(pilot_frac=args.pilot_frac, difficulty_frac=0.20, dev_frac=0.15, test_frac=0.55, seed=args.seed)
    split_mgr = SplitManager(config=split_cfg, record_path=args.out_dir / "splits.json")

    print(f"Loading model {args.model} (temperature={args.temperature})...")
    backend = HFBackend(
        args.model,
        temperature=args.temperature,
        precision=args.precision,
        param_count_b=args.param_count_b,
    )

    cost = CostTracker()
    verdicts = {}
    for domain in args.domains:
        result = run_pilot_for_domain(
            domain, backend, args.n_per_domain, args.max_new_tokens, split_mgr, cost, args.out_dir
        )
        verdicts[domain] = result["segmentation_stats"]["segmentation_gate_pass"]

    split_mgr.save()

    print("\n=== PILOT GATE SUMMARY ===")
    for domain, passed in verdicts.items():
        print(f"  {domain}: {'PASS (segmentation)' if passed else 'FAIL (segmentation)'}")
    print(
        "\nSegmentation PASS/FAIL above is automated. The truncation-curve "
        "'is there room to save tokens' call is NOT automated -- inspect "
        "the per-domain JSON/console output and judge by eye (project plan §5)."
    )
    print("\nToken cost this pilot run:")
    print(cost.summary())


if __name__ == "__main__":
    main()
