# ============================================================
# LOCAL TIMING PROBE -- measure real tok/s on this machine before
# committing to the full Candidate B pilot (18 problems, 24 pairs x 40
# branches, pre-registered in CANDIDATE_B_PREREGISTRATION.md).
#
# Run directly: python scripts/local_5090_timing_probe.py
# Needs: torch, transformers, datasets (pip install -r requirements.txt
# with the Phase 1+ lines uncommented, or just: pip install torch
# transformers accelerate datasets)
#
# What it measures, using the REAL project code path (early_stop.backend
# .HFBackend), not a synthetic benchmark:
#   1. Model load time + actual VRAM used
#   2. One full trace generation (2000 tokens, same as MAX_NEW_TOKENS_TRACE
#      in the pilot scripts) -> tok/s
#   3. Three forced-extraction calls (48 tokens, output_scores=True, same
#      as MAX_NEW_TOKENS_SCORE) -> tok/s
#   4. One branch continuation (1500 tokens, same as MAX_NEW_TOKENS_BRANCH)
#      -> tok/s
# Then projects wall-clock time for the full pre-registered pilot from
# these real numbers instead of the earlier T4-calibrated guess.
# ============================================================

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TEMPERATURE = 0.6

# Real MATH-500-style prompt, long enough to exercise a realistic
# context length, not a toy one-liner.
SAMPLE_PROBLEM = (
    "Find the sum of all positive integers $n$ such that $n^2 - 19n + 99$ "
    "is a perfect square. Show your work step by step, then give the "
    "final answer."
)


def tok_per_sec(backend: HFBackend, text: str, n_new_tokens: int, elapsed: float) -> float:
    n_tokens = len(backend.tokenizer(text, add_special_tokens=False).input_ids)
    return n_tokens / elapsed if elapsed > 0 else float("nan")


def main():
    gpu = probe_gpu()
    print(f"GPU: {gpu.summary()}")
    if not gpu.available:
        print("No CUDA GPU detected -- this probe needs the local GPU machine, not this shell.")
        sys.exit(1)

    print(f"\nLoading {MODEL_REPO} ...")
    t0 = time.time()
    backend = HFBackend(
        model_name=MODEL_REPO,
        temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    import torch
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM after load: {vram_used:.2f} GiB")

    results = {}

    # ---- 1. full trace generation (2000 tokens) ----
    print("\n[1/3] Full trace generation (2000 tokens)...")
    t0 = time.time()
    trace = backend.generate(SAMPLE_PROBLEM, max_new_tokens=2000)
    elapsed = time.time() - t0
    rate = tok_per_sec(backend, trace, 2000, elapsed)
    results["trace_2000tok"] = (elapsed, rate)
    print(f"  {elapsed:.1f}s, ~{rate:.1f} tok/s")

    # ---- 2. forced-extraction calls (48 tokens x 3) ----
    print("\n[2/3] Forced-extraction calls (48 tokens x 3)...")
    suffix = "\n\nTherefore, the final answer is \\boxed{"
    prefix_text = SAMPLE_PROBLEM + "\n\n" + trace[: len(trace) // 2]
    extraction_times = []
    for i in range(3):
        t0 = time.time()
        scored = backend.forced_extract_with_scores(prefix_text, suffix, max_new_tokens=48)
        elapsed = time.time() - t0
        extraction_times.append(elapsed)
        print(f"  call {i+1}: {elapsed:.2f}s (confidence={scored.top_token_confidence:.3f}, entropy={scored.entropy_bits:.2f} bits)")
    avg_extraction = sum(extraction_times) / len(extraction_times)
    results["forced_extract_48tok"] = (avg_extraction, 48 / avg_extraction if avg_extraction > 0 else float("nan"))

    # ---- 3. branch continuation (1500 tokens) ----
    print("\n[3/3] Branch continuation (1500 tokens)...")
    t0 = time.time()
    branch = backend.continue_generate(prefix_text, max_new_tokens=1500)
    elapsed = time.time() - t0
    rate = tok_per_sec(backend, branch, 1500, elapsed)
    results["branch_1500tok"] = (elapsed, rate)
    print(f"  {elapsed:.1f}s, ~{rate:.1f} tok/s")

    # ---- projection onto the pre-registered full pilot ----
    print("\n" + "=" * 60)
    print("PROJECTION onto the pre-registered full pilot")
    print("(CANDIDATE_B_PREREGISTRATION.md: 18 problems x 5 trajectories,")
    print(" 24 pairs x 40 branches/pair)")
    print("=" * 60)

    trace_time, _ = results["trace_2000tok"]
    extract_time, _ = results["forced_extract_48tok"]
    branch_time, _ = results["branch_1500tok"]

    n_trajectories = 18 * 5
    # rough: assume ~30 scored prefixes/trajectory (micro-pilot average
    # was ~1271/40 traj =~ 32) -- an approximation, not exact.
    avg_prefixes_per_traj = 32
    n_scoring_calls = n_trajectories * avg_prefixes_per_traj
    n_branch_calls = 24 * 40 * 2  # both sides of each pair

    phase1_hours = (n_trajectories * trace_time) / 3600
    phase2_hours = (n_scoring_calls * extract_time) / 3600
    phase3_hours = (n_branch_calls * branch_time) / 3600
    total_hours = phase1_hours + phase2_hours + phase3_hours

    print(f"Phase 1 (90 trajectories x {trace_time:.1f}s):      ~{phase1_hours:.1f}h")
    print(f"Phase 2 (~{n_scoring_calls} scoring calls x {extract_time:.2f}s): ~{phase2_hours:.1f}h")
    print(f"Phase 3 (1920 branches x {branch_time:.1f}s):       ~{phase3_hours:.1f}h")
    print(f"TOTAL projected wall-clock:                    ~{total_hours:.1f}h")
    print("\n(This is a projection from 5 real calls, not a measurement of the")
    print(" full run -- generation time has real variance across problems/")
    print(" trajectory length. Treat as a planning estimate.)")


if __name__ == "__main__":
    main()
