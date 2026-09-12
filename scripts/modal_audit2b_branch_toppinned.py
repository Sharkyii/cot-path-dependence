"""
AUDIT 2, PHASE B, RERUN WITH top_p PINNED (REVISION_PLAN.md Tier 1.3) --
late-stage matching + 3000-token branching with telemetry, for the
censoring question, this time at the configuration that actually
produced the effect.

Requires modal_audit2a_trajectories.py (the top_p=0.95 version, NOT the
original modal_audit2a_trajectories.py -- same file, rerun after the
top_p fix) to have already run and saved prefix_pool_d1_prob1.json and
prefix_pool_d2_prob1.json to
candidate_b_full_pilot/audit_censoring_toppinned/. Reads those.

WHY THIS RERUN, AND WHY IT RUNS BOTH PROBLEMS IN TRUE PARALLEL: the
first version of this audit (modal_audit2b_branch.py, results still on
the volume at audit_censoring/, not touched by this script) ruled out
censoring but under the wrong generation config -- HFBackend never
supported top_p at all, so it silently ran at the model's default
instead of the primary sample's top_p=0.95. early_stop/backend.py now
supports top_p, and modal_audit2a_trajectories.py now pins it, so this
script (and the trajectories it reads) test the configuration this audit
was always supposed to test. Split into one container per problem via
.starmap, same pattern as Phase A, so both problems branch at once
instead of sequentially -- roughly halves wall-clock time for no extra
cost (GPU-seconds billed are the same either way, this only buys speed).

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

Unlike the first version of this audit, THIS result is definitive either
way, because it is run at the configuration that actually produced the
decisiveness gap, not an accidentally-substituted one.

Only considers prefixes at normalized_position >= 0.7 (late-stage) --
an early prefix wouldn't be expected to close the box within budget
regardless of the censoring question, so including it would just add
noise to the comparison.

Cost: real timing from the first audit run (d1_prob1/d2_prob1 combined
Phase B) came in well under $2 total at 40 branches/pair, 2 pairs/problem,
3000-token cap -- top_p pinning does not change generation speed, only
the sampling distribution, so this rerun should cost about the same.
ALWAYS run the cheap diagnostic first if you haven't already validated
this exact code path.

Usage:
    # cheap diagnostic first:
    modal run scripts/modal_audit2b_branch_toppinned.py --n-branches 4 --max-pairs-per-problem 1

    # full run, both problems in parallel:
    modal run scripts/modal_audit2b_branch_toppinned.py

To pull results back down:
    modal volume get candidate-b-results /candidate_b_full_pilot/audit_censoring_toppinned ./modal_results_audit2_toppinned
"""

import pathlib
import sys

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TARGET_PROBLEM_IDS = ["d1_prob1", "d2_prob1"]
N_BRANCHES_PER_PAIR = 40
MAX_PAIRS_PER_PROBLEM = 2
MIN_NORMALIZED_POSITION = 0.7
MAX_NEW_TOKENS_BRANCH = 3000
TEMPERATURE = 0.6
TOP_P = 0.95  # THE fix -- matches the primary sample's real config, and modal_audit2a_trajectories.py's.
CONFIDENCE_TOL = 0.10
ENTROPY_TOL = 0.15
POSITION_TOL = 0.15

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("candidate-b-audit2b-branch-toppinned")

volume = modal.Volume.from_name("candidate-b-results", create_if_missing=True)
VOL_PATH = pathlib.Path("/vol")
OUT_DIR_REL = "candidate_b_full_pilot/audit_censoring_toppinned"

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
def run_phase_b_one_problem(problem_id: str, n_branches: int, max_pairs_per_problem: int):
    import json
    import os

    os.environ["HF_HOME"] = str(VOL_PATH / "hf_cache")
    sys.path.insert(0, "/root")

    from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu
    from early_stop.candidate_b_pipeline import (
        empty_rate_at_cutoff, load_prefix_pool, sample_and_branch_with_telemetry,
    )
    from early_stop.path_dependence import find_matched_pairs, stratified_sample_pairs

    tag = f"audit2b[{problem_id}]"
    out_dir = VOL_PATH / OUT_DIR_REL

    pool_path = out_dir / f"prefix_pool_{problem_id}.json"
    if not pool_path.exists():
        raise SystemExit(
            f"[{tag}] {pool_path} not found -- run modal_audit2a_trajectories.py "
            f"(the top_p=0.95 version) first."
        )

    gpu = probe_gpu()
    print(f"[{tag}] {gpu.summary()}")
    backend = HFBackend(
        model_name=MODEL_REPO, temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, precision="fp16",
        top_p=TOP_P,
    )
    volume.commit()

    with open(pool_path) as f:
        pool = json.load(f)
    _pid, base_prompt, trajectories, prefixes = load_prefix_pool(pool)
    print(f"[{tag}] {len(prefixes)} prefixes loaded")

    l1_pairs = find_matched_pairs(prefixes, "L1", confidence_tol=CONFIDENCE_TOL, entropy_tol=ENTROPY_TOL, position_tol=POSITION_TOL)
    late_pairs = [
        (a, b) for a, b in l1_pairs
        if a.normalized_position >= MIN_NORMALIZED_POSITION and b.normalized_position >= MIN_NORMALIZED_POSITION
    ]
    print(f"[{tag}] {len(l1_pairs)} L1 pairs total, {len(late_pairs)} at "
          f"normalized_position >= {MIN_NORMALIZED_POSITION}")
    sampled = stratified_sample_pairs(late_pairs, max_pairs_per_problem, seed=7)

    results = []

    def _on_pair_done(entry):
        entry["side_a_empty_rate_at_1500"] = empty_rate_at_cutoff(entry["side_a_telemetry"], 1500)
        entry["side_b_empty_rate_at_1500"] = empty_rate_at_cutoff(entry["side_b_telemetry"], 1500)
        entry["side_a_empty_rate_at_3000"] = empty_rate_at_cutoff(entry["side_a_telemetry"], 3000)
        entry["side_b_empty_rate_at_3000"] = empty_rate_at_cutoff(entry["side_b_telemetry"], 3000)
        print(f"[{tag}]   empty@1500: A={entry['side_a_empty_rate_at_1500']:.2f} "
              f"B={entry['side_b_empty_rate_at_1500']:.2f}   empty@3000: "
              f"A={entry['side_a_empty_rate_at_3000']:.2f} B={entry['side_b_empty_rate_at_3000']:.2f} "
              f"[{entry['elapsed_s']:.1f}s -- USE THIS to project remaining cost]")
        with open(out_dir / f"results_{problem_id}.json", "w") as f:
            json.dump(results, f, indent=2)
        volume.commit()

    results = sample_and_branch_with_telemetry(
        backend, base_prompt, trajectories, sampled, n_branches, MAX_NEW_TOKENS_BRANCH,
        on_pair_done=_on_pair_done,
    )
    with open(out_dir / f"results_{problem_id}.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    if results:
        drops = [
            (round(e["side_a_empty_rate_at_1500"] - e["side_a_empty_rate_at_3000"], 3),
             round(e["side_b_empty_rate_at_1500"] - e["side_b_empty_rate_at_3000"], 3))
            for e in results
        ]
        print(f"[{tag}] DONE. empty-rate drop 1500->3000 tokens per pair (A, B): {drops}")
        print(f"[{tag}]   Large drops => censoring. Near-zero drops with finish_reason='eos' "
              f"in the telemetry => the effect is real, AT THE CORRECT CONFIGURATION this time.")
    return {"problem_id": problem_id, "n_pairs": len(results)}


@app.local_entrypoint()
def main(n_branches: int = N_BRANCHES_PER_PAIR, max_pairs_per_problem: int = MAX_PAIRS_PER_PROBLEM,
         problem_ids: str = ""):
    """problem_ids: comma-separated subset of TARGET_PROBLEM_IDS to run (default: both).
    Added to let a dropped/incomplete problem be rerun alone without re-paying for
    a problem that already finished -- e.g. --problem-ids d2_prob1."""
    target_ids = [p.strip() for p in problem_ids.split(",") if p.strip()] or TARGET_PROBLEM_IDS
    print(f"Audit 2, Phase B (top_p=0.95 pinned): n_branches={n_branches}, "
          f"max_pairs_per_problem={max_pairs_per_problem}, problems={target_ids}, TRUE PARALLEL")
    if n_branches >= N_BRANCHES_PER_PAIR:
        print("This is the FULL run. If you haven't run the cheap diagnostic yet "
              "(--n-branches 4 --max-pairs-per-problem 1), consider Ctrl+C and doing that first.\n")
    args = [(pid, n_branches, max_pairs_per_problem) for pid in target_ids]
    results = list(run_phase_b_one_problem.starmap(args))
    print(f"\nDone: {results}")
    print("Pull results: modal volume get candidate-b-results "
          "/candidate_b_full_pilot/audit_censoring_toppinned ./modal_results_audit2_toppinned")
