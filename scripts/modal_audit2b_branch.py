"""
AUDIT 2, PHASE B (REVISION_PLAN.md Tier 1.3) -- late-stage matching +
3000-token branching with telemetry, for the censoring question.

Requires modal_audit2a_trajectories.py to have already run and saved
prefix_pool_d1_prob1.json and prefix_pool_d2_prob1.json to the volume.
Reads those (no trajectory generation here -- if THIS phase fails
partway, Phase A's cost isn't wasted; just re-run this one).

THE TEST: generate branches at MAX_NEW_TOKENS_BRANCH=3000 (double the
original 1500) and record, per branch, the token index at which
\\boxed{...} first closes (or None if it never does) -- see
HFBackend.continue_generate_with_telemetry's docstring. From that ONE
set of generations, empty_rate_at_cutoff() computes what the empty rate
WOULD have been at 1500 tokens vs the true rate at 3000, without paying
for two separate runs.

  - If most originally-empty branches turn out to close well past 1500
    but before 3000 (empty rate drops a lot from 1500->3000): CENSORING
    CONFIRMED. Retract the decisiveness-gap mechanism from the paper's
    main findings.
  - If empties persist even at 3000 with finish_reason="eos" (the model
    generated an end-of-sequence token without ever closing the box, a
    deliberate stop, not a cutoff): the effect is REAL, and this
    telemetry is the rebuttal to the censoring objection.

Only considers prefixes at normalized_position >= 0.7 (late-stage) --
an early prefix wouldn't be expected to close the box within budget
regardless of the censoring question, so including it would just add
noise to the comparison.

Cost (d1_prob1 ~7.5s/branch, d2_prob1 ~15.4s/branch at 1500 tokens --
these are FLOORS; 3000-token branches cost more only for the fraction
that runs past 1500, which is exactly the effect being measured):
  2 problems x up to 2 pairs x 2 sides x n_branches. At the defaults (40
  branches) that's up to 320 branches, well under $3 at A10G's ~$1.10/hr
  even at 2x the 1500-token floor. ALWAYS run the cheap diagnostic first.

Usage:
    # cheap diagnostic first:
    modal run scripts/modal_audit2b_branch.py --n-branches 4 --max-pairs-per-problem 1

    # full run:
    modal run scripts/modal_audit2b_branch.py

To pull results back down:
    modal volume get candidate-b-results /candidate_b_full_pilot/audit_censoring ./modal_results_audit2
"""

import pathlib
import sys

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TARGET_PROBLEM_IDS = ["d1_prob1", "d2_prob1"]
N_BRANCHES_PER_PAIR = 40  # matches the pre-registered / original round-1 branch count --
                          # cheap enough here that there's no reason to shrink it
MAX_PAIRS_PER_PROBLEM = 2  # matches the original dataset's informative-pair count
MIN_NORMALIZED_POSITION = 0.7
MAX_NEW_TOKENS_BRANCH = 3000  # DOUBLE the original 1500 -- this is the whole point
TEMPERATURE = 0.6
CONFIDENCE_TOL = 0.10
ENTROPY_TOL = 0.15
POSITION_TOL = 0.15

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("candidate-b-audit2b-branch")

volume = modal.Volume.from_name("candidate-b-results", create_if_missing=True)
VOL_PATH = pathlib.Path("/vol")
OUT_DIR_REL = "candidate_b_full_pilot/audit_censoring"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "accelerate", "bitsandbytes",
        "sentencepiece", "datasets", "scipy", "numpy",
    )
    .add_local_dir(str(REPO_ROOT / "early_stop"), remote_path="/root/early_stop")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=2 * 60 * 60,
    volumes={VOL_PATH: volume},
)
def run_phase_b(n_branches: int = N_BRANCHES_PER_PAIR, max_pairs_per_problem: int = MAX_PAIRS_PER_PROBLEM):
    import json
    import os

    os.environ["HF_HOME"] = str(VOL_PATH / "hf_cache")
    sys.path.insert(0, "/root")

    from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu
    from early_stop.candidate_b_pipeline import (
        empty_rate_at_cutoff, load_prefix_pool, sample_and_branch_with_telemetry,
    )
    from early_stop.path_dependence import find_matched_pairs, stratified_sample_pairs

    tag = "audit2b"
    out_dir = VOL_PATH / OUT_DIR_REL

    gpu = probe_gpu()
    print(f"[{tag}] {gpu.summary()}")
    backend = HFBackend(
        model_name=MODEL_REPO, temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, precision="fp16",
    )
    volume.commit()

    all_results = {}
    for problem_id in TARGET_PROBLEM_IDS:
        pool_path = out_dir / f"prefix_pool_{problem_id}.json"
        if not pool_path.exists():
            print(f"[{tag}]   SKIP {problem_id}: {pool_path} not found -- run "
                  f"modal_audit2a_trajectories.py first.")
            continue
        with open(pool_path) as f:
            pool = json.load(f)
        _pid, base_prompt, trajectories, prefixes = load_prefix_pool(pool)
        print(f"\n[{tag}] --- {problem_id}: {len(prefixes)} prefixes loaded ---")

        l1_pairs = find_matched_pairs(prefixes, "L1", confidence_tol=CONFIDENCE_TOL, entropy_tol=ENTROPY_TOL, position_tol=POSITION_TOL)
        late_pairs = [
            (a, b) for a, b in l1_pairs
            if a.normalized_position >= MIN_NORMALIZED_POSITION and b.normalized_position >= MIN_NORMALIZED_POSITION
        ]
        print(f"[{tag}] {problem_id}: {len(l1_pairs)} L1 pairs total, {len(late_pairs)} at "
              f"normalized_position >= {MIN_NORMALIZED_POSITION}")
        sampled = stratified_sample_pairs(late_pairs, max_pairs_per_problem, seed=7)

        def _on_pair_done(entry, problem_id=problem_id):
            entry["side_a_empty_rate_at_1500"] = empty_rate_at_cutoff(entry["side_a_telemetry"], 1500)
            entry["side_b_empty_rate_at_1500"] = empty_rate_at_cutoff(entry["side_b_telemetry"], 1500)
            entry["side_a_empty_rate_at_3000"] = empty_rate_at_cutoff(entry["side_a_telemetry"], 3000)
            entry["side_b_empty_rate_at_3000"] = empty_rate_at_cutoff(entry["side_b_telemetry"], 3000)
            print(f"[{tag}]   [{problem_id}] empty@1500: A={entry['side_a_empty_rate_at_1500']:.2f} "
                  f"B={entry['side_b_empty_rate_at_1500']:.2f}   empty@3000: "
                  f"A={entry['side_a_empty_rate_at_3000']:.2f} B={entry['side_b_empty_rate_at_3000']:.2f} "
                  f"[{entry['elapsed_s']:.1f}s -- USE THIS to project remaining cost]")
            with open(out_dir / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            volume.commit()

        problem_results = sample_and_branch_with_telemetry(
            backend, base_prompt, trajectories, sampled, n_branches, MAX_NEW_TOKENS_BRANCH,
            on_pair_done=_on_pair_done,
        )
        all_results[problem_id] = problem_results
        with open(out_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        volume.commit()

    print(f"\n[{tag}] DONE.")
    for pid, entries in all_results.items():
        if not entries:
            continue
        drops = [
            (round(e["side_a_empty_rate_at_1500"] - e["side_a_empty_rate_at_3000"], 3),
             round(e["side_b_empty_rate_at_1500"] - e["side_b_empty_rate_at_3000"], 3))
            for e in entries
        ]
        print(f"[{tag}] {pid}: empty-rate drop 1500->3000 tokens per pair (A, B): {drops}")
        print(f"[{tag}]   Large drops => censoring. Near-zero drops with finish_reason='eos' "
              f"in the telemetry => the effect is real. See REVISION_PLAN.md 1.6.")
    return {pid: len(v) for pid, v in all_results.items()}


@app.local_entrypoint()
def main(n_branches: int = N_BRANCHES_PER_PAIR, max_pairs_per_problem: int = MAX_PAIRS_PER_PROBLEM):
    print(f"Audit 2, Phase B: n_branches={n_branches}, max_pairs_per_problem={max_pairs_per_problem}")
    if n_branches >= N_BRANCHES_PER_PAIR:
        print("This is the FULL run. If you haven't run the cheap diagnostic yet "
              "(--n-branches 4 --max-pairs-per-problem 1), consider Ctrl+C and doing that first.\n")
    result = run_phase_b.remote(n_branches, max_pairs_per_problem)
    print(f"\nDone: {result}")
    print("Pull results: modal volume get candidate-b-results "
          "/candidate_b_full_pilot/audit_censoring ./modal_results_audit2")
