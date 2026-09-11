"""
AUDIT 1, PHASE B (REVISION_PLAN.md Tier 1.2) -- matching + branching.

Requires modal_audit1a_trajectories.py to have already run and saved
/vol/candidate_b_full_pilot/audit_l1_l2_d3_prob5/prefix_pool.json -- this
script reads that (no trajectory generation here, so if THIS phase fails
partway, Phase A's cost isn't wasted; just re-run this one).

THE EXPERIMENT: the reframed paper's central claim (post-2026-09-12
revision) is "answer-only matching is insufficient, and confidence
carries measurable signal that answer-only early-stopping rules
discard." d3_prob5 is the evidence -- in the existing 66-pair dataset,
its 4 pairs were matched on ANSWER ONLY (L1), and the lower-confidence
side of each pair is the one that later scattered into wrong answers.

This is suggestive, not yet a direct test: "lower confidence scatters"
and "confidence explains the divergence" aren't the same claim until you
show the divergence goes away once confidence is ALSO matched. This
script pulls BOTH L1-matched and L2-matched (answer AND confidence)
pairs from the ONE shared prefix pool Phase A built, and branches a
sample from each.

  - L1 pairs still diverge, L2 pairs don't (or much less): thesis
    CONFIRMED. Confidence closes the gap.
  - L2 pairs ALSO diverge at similar magnitude: thesis REFUTED for this
    problem -- something else is doing the work.
  - Too few L2 pairs found: itself informative, see REVISION_PLAN.md 1.6.

Improvement over the original (pre-split) L2/L3 followup: that followup
compared fresh L2/L3 data against L1 data collected in a DIFFERENT run
weeks earlier -- flagged in CANDIDATE_B_PREREGISTRATION.md's 2026-09-11
amendment as vulnerable to regression to the mean. Both arms here come
from ONE shared regeneration and ONE branching configuration, so that
confound doesn't apply.

Cost (measured d3_prob5 branch timing: ~34.6s/branch at 1500 tokens):
  up to 2 levels x up to 4 pairs x 2 sides x n_branches. At the defaults
  (20 branches) that's up to 320 branches x ~35s = ~3.1h -> ~$3.40 at
  A10G's ~$1.10/hr. ALWAYS run the cheap diagnostic first.

Usage:
    # cheap diagnostic first:
    modal run scripts/modal_audit1b_branch.py --n-branches 4 --max-pairs-per-level 1

    # full run:
    modal run scripts/modal_audit1b_branch.py

To pull results back down:
    modal volume get candidate-b-results /candidate_b_full_pilot/audit_l1_l2_d3_prob5 ./modal_results_audit1
"""

import pathlib
import sys

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_BRANCHES_PER_PAIR = 20  # half the pre-registered 40 -- adequate to tell "PDI ~0.15"
                          # (what L1 shows) from "PDI ~0" (see REVISION_PLAN.md 1.2)
MAX_PAIRS_PER_LEVEL = 4  # matches the original dataset's d3_prob5 pair count
MAX_NEW_TOKENS_BRANCH = 1500  # UNCHANGED from the original run, keeps PDI comparable.
                               # d3_prob5 had 0/40 empty answers either side originally,
                               # so the censoring question audit 2 asks doesn't apply here.
TEMPERATURE = 0.6
CONFIDENCE_TOL = 0.10
ENTROPY_TOL = 0.15
POSITION_TOL = 0.15

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("candidate-b-audit1b-branch")

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
    timeout=2 * 60 * 60,  # 2h -- see modal_audit1a_trajectories.py's matching comment
    volumes={VOL_PATH: volume},
)
def run_phase_b(n_branches: int = N_BRANCHES_PER_PAIR, max_pairs_per_level: int = MAX_PAIRS_PER_LEVEL):
    import json
    import os

    os.environ["HF_HOME"] = str(VOL_PATH / "hf_cache")
    sys.path.insert(0, "/root")

    from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu
    from early_stop.candidate_b_pipeline import load_prefix_pool, sample_and_branch_with_telemetry
    from early_stop.path_dependence import find_matched_pairs, stratified_sample_pairs

    tag = "audit1b"
    out_dir = VOL_PATH / OUT_DIR_REL
    pool_path = out_dir / "prefix_pool.json"
    if not pool_path.exists():
        raise SystemExit(
            f"[{tag}] {pool_path} not found -- run modal_audit1a_trajectories.py first."
        )

    with open(pool_path) as f:
        pool = json.load(f)
    problem_id, base_prompt, trajectories, prefixes = load_prefix_pool(pool)
    print(f"[{tag}] Loaded {len(prefixes)} prefixes for {problem_id} from Phase A's output")

    l1_pairs = find_matched_pairs(prefixes, "L1", confidence_tol=CONFIDENCE_TOL, entropy_tol=ENTROPY_TOL, position_tol=POSITION_TOL)
    l2_pairs = find_matched_pairs(prefixes, "L2", confidence_tol=CONFIDENCE_TOL, entropy_tol=ENTROPY_TOL, position_tol=POSITION_TOL)
    print(f"[{tag}] L1 matched pairs: {len(l1_pairs)}   L2 matched pairs: {len(l2_pairs)}")
    if len(l2_pairs) == 0:
        print(f"[{tag}]   NOTE: zero L2 pairs. Per REVISION_PLAN.md 1.6, this is itself "
              f"informative -- confidence-matched pairs may be rare precisely because "
              f"confidence varies a lot at this problem's checkpoints.")
    with open(out_dir / "phase_b_matched_pair_counts.json", "w") as f:
        json.dump({"L1": len(l1_pairs), "L2": len(l2_pairs)}, f, indent=2)
    volume.commit()

    gpu = probe_gpu()
    print(f"[{tag}] {gpu.summary()}")
    backend = HFBackend(
        model_name=MODEL_REPO, temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, precision="fp16",
    )
    volume.commit()

    results = {"L1": [], "L2": []}
    for level, pairs in (("L1", l1_pairs), ("L2", l2_pairs)):
        if not pairs:
            continue
        sampled = stratified_sample_pairs(pairs, max_pairs_per_level, seed={"L1": 1, "L2": 2}[level])

        def _on_pair_done(entry, level=level):
            print(f"[{tag}]   [{level}] confA={entry['side_a_state']['confidence']:.3f} "
                  f"confB={entry['side_b_state']['confidence']:.3f} "
                  f"d={entry['state_distance']:.3f} [{entry['elapsed_s']:.1f}s -- "
                  f"USE THIS to project remaining cost]")
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            volume.commit()

        level_results = sample_and_branch_with_telemetry(
            backend, base_prompt, trajectories, sampled, n_branches, MAX_NEW_TOKENS_BRANCH,
            on_pair_done=_on_pair_done,
        )
        results[level] = level_results

    # PDI per pair, computed here (not inside the shared helper, which is
    # branching-only) so this phase owns its own verdict.
    from early_stop.path_dependence import answer_distribution, pdi
    for level in ("L1", "L2"):
        for r in results[level]:
            r["observed_pdi"] = pdi(answer_distribution(r["side_a_raw_answers"]), answer_distribution(r["side_b_raw_answers"]))
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    def _mean_pdi(level):
        if not results[level]:
            return None
        return sum(r["observed_pdi"] for r in results[level]) / len(results[level])

    print(f"\n[{tag}] DONE. L1: {len(results['L1'])} pairs (mean PDI={_mean_pdi('L1')}), "
          f"L2: {len(results['L2'])} pairs (mean PDI={_mean_pdi('L2')})")
    print(f"[{tag}] See REVISION_PLAN.md 1.6 for how to read this comparison.")
    return {"L1": _mean_pdi("L1"), "L2": _mean_pdi("L2"), "n_l1": len(results["L1"]), "n_l2": len(results["L2"])}


@app.local_entrypoint()
def main(n_branches: int = N_BRANCHES_PER_PAIR, max_pairs_per_level: int = MAX_PAIRS_PER_LEVEL):
    print(f"Audit 1, Phase B: n_branches={n_branches}, max_pairs_per_level={max_pairs_per_level}")
    if n_branches >= N_BRANCHES_PER_PAIR:
        print("This is the FULL run (~$4). If you haven't run the cheap diagnostic yet "
              "(--n-branches 4 --max-pairs-per-level 1), consider Ctrl+C and doing that first.\n")
    result = run_phase_b.remote(n_branches, max_pairs_per_level)
    print(f"\nDone: {result}")
    print("Pull results: modal volume get candidate-b-results "
          "/candidate_b_full_pilot/audit_l1_l2_d3_prob5 ./modal_results_audit1")
