"""
AUDIT 1, PHASE A (REVISION_PLAN.md Tier 1.2) -- trajectory generation only.

Split out of a single big script per user request: each phase is now its
own small file, so a failure or a budget check partway through doesn't
waste money already spent on an earlier phase, and each phase can be
smoke-tested (scripts/audit_smoke_test.py, $0, no GPU) before it's ever
run for real.

This phase regenerates trajectories for d3_prob5 (difficulty 3, index 5
-- the exact same problem_id as the original 66-pair dataset) and scores
every step boundary, writing the result as JSON to the Modal volume.
Phase B (modal_audit1b_branch.py) reads that JSON and does the actual
matching + branching. Run THIS phase first; if it fails or the model
loads wrong, you haven't spent anything on the expensive part yet.

Why this problem, why this test: see modal_audit1b_branch.py's docstring
for the full experimental rationale (L1-vs-L2 contrast). This script only
builds the shared prefix pool both matching levels draw from.

Cost: trajectory generation is a small, roughly fixed cost independent of
the branching budget -- this phase alone should be well under $1. ALWAYS
run the cheap diagnostic first.

Usage:
    pip install modal
    python3 -m modal setup

    # cheap diagnostic first -- 2 trajectories, a couple minutes:
    modal run scripts/modal_audit1a_trajectories.py --n-trajectories 2

    # full run (default n_trajectories=10):
    modal run scripts/modal_audit1a_trajectories.py

Then run:
    modal run scripts/modal_audit1b_branch.py
"""

import pathlib
import sys

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DIFFICULTY = "3"
PROBLEM_INDEX = 5  # d3_prob5
N_TRAJECTORIES = 10
MAX_NEW_TOKENS_TRACE = 3000
MAX_NEW_TOKENS_SCORE = 64
TEMPERATURE = 0.6

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("candidate-b-audit1a-trajectories")

volume = modal.Volume.from_name("candidate-b-results", create_if_missing=True)
VOL_PATH = pathlib.Path("/vol")
OUT_DIR_REL = "candidate_b_full_pilot/audit_l1_l2_d3_prob5"

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
    timeout=2 * 60 * 60,  # 2h, not 20h -- REVISION_PLAN.md 1.0: caps a hung
                          # container at ~$2.20 instead of ~$22 on a $10 total budget.
    volumes={VOL_PATH: volume},
)
def run_phase_a(n_trajectories: int = N_TRAJECTORIES):
    import json
    import os
    import time

    os.environ["HF_HOME"] = str(VOL_PATH / "hf_cache")
    sys.path.insert(0, "/root")

    from datasets import load_dataset

    from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu
    from early_stop.candidate_b_pipeline import build_prefix_pool

    tag = "audit1a"
    out_dir = VOL_PATH / OUT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = probe_gpu()
    print(f"[{tag}] {gpu.summary()}")

    t0 = time.time()
    backend = HFBackend(
        model_name=MODEL_REPO, temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B,
        precision="fp16",  # matches every proven-working run in this project
    )
    print(f"[{tag}] Model loaded in {time.time() - t0:.1f}s")
    volume.commit()

    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    level_ds = math_ds.filter(lambda r: str(r["level"]) == DIFFICULTY)
    if len(level_ds) <= PROBLEM_INDEX:
        raise SystemExit(f"Only {len(level_ds)} MATH-500 problems at level {DIFFICULTY}, need index {PROBLEM_INDEX}")
    item = level_ds.select([PROBLEM_INDEX])[0]
    problem_id = f"d{DIFFICULTY}_prob{PROBLEM_INDEX}"
    base_prompt = item["problem"]
    print(f"[{tag}] {problem_id}: {base_prompt[:200]}")

    def _on_done(traj_id, n_scored, n_unparse, elapsed):
        print(f"[{tag}]   {traj_id}: {n_scored} scored, {n_unparse} unparseable, "
              f"{elapsed:.1f}s [USE THIS to project remaining cost]")
        volume.commit()

    pool = build_prefix_pool(
        backend, problem_id, base_prompt, n_trajectories,
        max_new_tokens_trace=MAX_NEW_TOKENS_TRACE, max_new_tokens_score=MAX_NEW_TOKENS_SCORE,
        on_trajectory_done=_on_done,
    )

    with open(out_dir / "prefix_pool.json", "w") as f:
        json.dump(pool, f, indent=2)
    volume.commit()

    print(f"\n[{tag}] DONE. {len(pool['prefixes'])} scored prefixes across {n_trajectories} trajectories.")
    print(f"[{tag}] Saved to {out_dir}/prefix_pool.json -- next run modal_audit1b_branch.py")
    return {"n_prefixes": len(pool["prefixes"]), "n_trajectories": n_trajectories}


@app.local_entrypoint()
def main(n_trajectories: int = N_TRAJECTORIES):
    print(f"Audit 1, Phase A: generating {n_trajectories} trajectories for d3_prob5")
    if n_trajectories >= N_TRAJECTORIES:
        print("This is the FULL run. If you haven't run the cheap diagnostic yet "
              "(--n-trajectories 2), consider Ctrl+C and doing that first.\n")
    result = run_phase_a.remote(n_trajectories)
    print(f"\nDone: {result}")
    print("Next: modal run scripts/modal_audit1b_branch.py")
