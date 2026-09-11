"""
AUDIT 2, PHASE A (REVISION_PLAN.md Tier 1.3, SECONDARY -- run after audit 1
if budget remains) -- trajectory generation only, for d1_prob1 AND
d2_prob1, run in TRUE PARALLEL (two containers via .starmap) so wall-clock
time is roughly halved. Same split rationale as audit 1: this phase is
cheap, so if Phase B (the expensive branching phase) fails, you haven't
lost this phase's cost.

QUESTION THIS AUDIT ANSWERS: is d1_prob1 and d2_prob1's "decisiveness
gap" (one side of a matched pair trails off with an unclosed \\boxed{}
far more often than the other) real, or token-budget right-censoring?
The original branching used MAX_NEW_TOKENS_BRANCH=1500. d1_prob1 is a
100-term enumeration; d2_prob1 is the dataset's one text-answer problem
-- both plausible candidates for needing more than 1500 tokens to close
the box from a less-advanced prefix. See modal_audit2b_branch.py's
docstring for how Phase B tests this.

Does NOT reuse the original dataset's exact prefixes (never persisted,
see CANDIDATE_B_PREREGISTRATION.md 2026-09-11 amendment) -- regenerates
fresh trajectories for the same two problems.

Cost: trajectory generation is cheap and roughly fixed regardless of
branch budget; both problems together should be well under $1.

Usage:
    pip install modal
    python3 -m modal setup

    # cheap diagnostic first -- 2 trajectories/problem, a couple minutes:
    modal run scripts/modal_audit2a_trajectories.py --n-trajectories 2

    # full run (default n_trajectories=10, both problems in parallel):
    modal run scripts/modal_audit2a_trajectories.py

Then run:
    modal run scripts/modal_audit2b_branch.py
"""

import pathlib
import sys

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# (difficulty, problem_index, problem_id) -- exact same problem_ids as the
# original dataset's d1_prob1 and d2_prob1.
TARGET_PROBLEMS = [("1", 1, "d1_prob1"), ("2", 1, "d2_prob1")]
N_TRAJECTORIES = 10
MAX_NEW_TOKENS_TRACE = 3000
MAX_NEW_TOKENS_SCORE = 64
TEMPERATURE = 0.6

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("candidate-b-audit2a-trajectories")

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
    timeout=2 * 60 * 60,  # 2h -- REVISION_PLAN.md 1.0 budget safeguard
    volumes={VOL_PATH: volume},
)
def run_phase_a_one_problem(difficulty: str, problem_index: int, problem_id: str, n_trajectories: int):
    import json
    import os
    import time

    os.environ["HF_HOME"] = str(VOL_PATH / "hf_cache")
    sys.path.insert(0, "/root")

    from datasets import load_dataset

    from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu
    from early_stop.candidate_b_pipeline import build_prefix_pool

    tag = f"audit2a[{problem_id}]"
    out_dir = VOL_PATH / OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = probe_gpu()
    print(f"[{tag}] {gpu.summary()}")

    t0 = time.time()
    backend = HFBackend(
        model_name=MODEL_REPO, temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, precision="fp16",
    )
    print(f"[{tag}] Model loaded in {time.time() - t0:.1f}s")
    volume.commit()

    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    level_ds = math_ds.filter(lambda r: str(r["level"]) == difficulty)
    if len(level_ds) <= problem_index:
        raise SystemExit(f"[{tag}] Only {len(level_ds)} problems at level {difficulty}, need index {problem_index}")
    item = level_ds.select([problem_index])[0]
    base_prompt = item["problem"]
    print(f"[{tag}] {base_prompt[:200]}")

    def _on_done(traj_id, n_scored, n_unparse, elapsed):
        print(f"[{tag}]   {traj_id}: {n_scored} scored, {n_unparse} unparseable, "
              f"{elapsed:.1f}s [USE THIS to project remaining cost]")
        volume.commit()

    pool = build_prefix_pool(
        backend, problem_id, base_prompt, n_trajectories,
        max_new_tokens_trace=MAX_NEW_TOKENS_TRACE, max_new_tokens_score=MAX_NEW_TOKENS_SCORE,
        on_trajectory_done=_on_done,
    )

    with open(out_dir / f"prefix_pool_{problem_id}.json", "w") as f:
        json.dump(pool, f, indent=2)
    volume.commit()

    print(f"[{tag}] DONE. {len(pool['prefixes'])} scored prefixes across {n_trajectories} trajectories.")
    return {"problem_id": problem_id, "n_prefixes": len(pool["prefixes"])}


@app.local_entrypoint()
def main(n_trajectories: int = N_TRAJECTORIES):
    print(f"Audit 2, Phase A: generating {n_trajectories} trajectories each for "
          f"{[p[2] for p in TARGET_PROBLEMS]}, in TRUE PARALLEL")
    if n_trajectories >= N_TRAJECTORIES:
        print("This is the FULL run. If you haven't run the cheap diagnostic yet "
              "(--n-trajectories 2), consider Ctrl+C and doing that first.\n")
    args = [(diff, idx, pid, n_trajectories) for diff, idx, pid in TARGET_PROBLEMS]
    results = list(run_phase_a_one_problem.starmap(args))
    print(f"\nDone: {results}")
    print("Next: modal run scripts/modal_audit2b_branch.py")
