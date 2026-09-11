"""
CANDIDATE B PHASE 4 (low-compute plan, 2026-09-11): SG-ES validation.

For each problem, generate ONE full natural trace with dense step-wise
phi(p_t) scoring (same Phase-1 machinery as the main pilot script --
early_stop.candidate_b_pipeline.build_trajectory_prefixes). From that single
list of scored Prefix objects, compute all 3 conditions POST-HOC (no extra
generation needed, see early_stop/sg_es.py's module docstring):

  1. full_generation_answer   -- the trace's own final answer
  2. forced_extraction_baseline(fraction=0.5) -- naive fixed-point stop
  3. sg_es_decision           -- stop when answer stable + confident

Each condition's answer is graded against MATH-500's gold answer (same
normalize_math_answer() used everywhere else in this project). Reports,
per condition: accuracy and mean fraction_of_trace_used (token savings
proxy) across the problem sample.

Usage:
    modal run scripts/modal_candidate_b_phase4_sges.py --n-problems 2   # smoke test
    modal run scripts/modal_candidate_b_phase4_sges.py --n-problems 24  # real run,
        stratified across MATH-500 levels 1-4 (6 problems/level)

To pull results back down:
    modal volume get candidate-b-results candidate_b_phase4_sges ./phase4_results
"""
import pathlib
import sys

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_NEW_TOKENS_TRACE = 3000
MAX_NEW_TOKENS_SCORE = 64
TEMPERATURE = 0.6
CONFIDENCE_THRESHOLD = 0.70
STABILITY_WINDOW = 2
FORCED_FRACTION = 0.5

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("candidate-b-phase4-sges")
volume = modal.Volume.from_name("candidate-b-results", create_if_missing=True)
VOL_PATH = pathlib.Path("/vol")

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
    timeout=20 * 60 * 60,
    volumes={VOL_PATH: volume},
)
def run_phase4(n_problems: int, levels: tuple[str, ...] = ("1", "2", "3", "4")):
    import json
    import os
    import time

    os.environ["HF_HOME"] = str(VOL_PATH / "hf_cache")
    sys.path.insert(0, "/root")

    from datasets import load_dataset

    from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu
    from early_stop.candidate_b_pipeline import (
        generate_diverse_trajectories, build_trajectory_prefixes,
    )
    from early_stop.parsers import normalize_math_answer
    from early_stop.sg_es import (
        full_generation_answer, forced_extraction_baseline, sg_es_decision,
    )

    out_dir = VOL_PATH / "candidate_b_phase4_sges"
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = probe_gpu()
    print(f"[phase4] {gpu.summary()}")

    t0 = time.time()
    backend = HFBackend(
        model_name=MODEL_REPO, temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B,
        precision="fp16",  # see modal_candidate_b_full_pilot.py -- fp16 proven, bf16 regressed badly
    )
    print(f"[phase4] Model loaded in {time.time() - t0:.1f}s")
    volume.commit()

    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    per_level = max(1, n_problems // len(levels))
    problems = []
    for lvl in levels:
        level_ds = math_ds.filter(lambda r, lvl=lvl: str(r["level"]) == lvl)
        take = min(per_level, len(level_ds))
        for i in range(take):
            row = level_ds[i]
            problems.append({
                "problem_id": f"phase4_d{lvl}_prob{i}",
                "level": lvl,
                "question": row["problem"],
                "gold_answer": normalize_math_answer(row["answer"]),
            })
    problems = problems[:n_problems]
    print(f"[phase4] {len(problems)} problems stratified across levels {levels}")

    results = []
    for idx, item in enumerate(problems):
        pid = item["problem_id"]
        base_prompt = item["question"]
        gold = item["gold_answer"]
        t0 = time.time()

        traces = generate_diverse_trajectories(backend, base_prompt, k=1, max_new_tokens=MAX_NEW_TOKENS_TRACE)
        trace_text = traces[0]
        tp = build_trajectory_prefixes(
            backend, pid, "traj_0", base_prompt, trace_text,
            max_new_tokens=MAX_NEW_TOKENS_SCORE,
        )
        elapsed = time.time() - t0

        prefixes = tp.prefixes
        if not prefixes:
            print(f"[phase4] {pid}: 0 scored prefixes ({tp.n_unparseable} unparseable), skipping -- {elapsed:.1f}s")
            continue

        full = full_generation_answer(prefixes)
        forced = forced_extraction_baseline(prefixes, fraction=FORCED_FRACTION)
        sges = sg_es_decision(prefixes, confidence_threshold=CONFIDENCE_THRESHOLD, stability_window=STABILITY_WINDOW)

        def _correct(ans):
            return ans is not None and normalize_math_answer(ans) == gold

        entry = {
            "problem_id": pid,
            "level": item["level"],
            "n_scored_prefixes": len(prefixes),
            "n_unparseable": tp.n_unparseable,
            "elapsed_s": elapsed,
            "full_generation": {
                "answer": full.answer, "correct": _correct(full.answer),
                "fraction_of_trace_used": full.fraction_of_trace_used,
            },
            "forced_extraction_0.5": {
                "answer": forced.answer, "correct": _correct(forced.answer),
                "fraction_of_trace_used": forced.fraction_of_trace_used,
            },
            "sg_es": {
                "answer": sges.answer, "correct": _correct(sges.answer),
                "fraction_of_trace_used": sges.fraction_of_trace_used,
                "triggered": sges.triggered,
            },
        }
        results.append(entry)
        print(f"[phase4] {pid} (L{item['level']}): full={full.answer!r}({entry['full_generation']['correct']}) "
              f"forced={forced.answer!r}({entry['forced_extraction_0.5']['correct']}) "
              f"sges={sges.answer!r}({entry['sg_es']['correct']}, triggered={sges.triggered}, "
              f"used={sges.fraction_of_trace_used:.2f}) -- {elapsed:.1f}s [{idx+1}/{len(problems)}]")

        with open(out_dir / "phase4_results.json", "w") as f:
            json.dump(results, f, indent=2)
        volume.commit()

    # summary
    def _agg(key):
        n = len(results)
        acc = sum(1 for r in results if r[key]["correct"]) / n if n else 0.0
        frac = sum(r[key]["fraction_of_trace_used"] for r in results) / n if n else 0.0
        return acc, frac

    print("\n[phase4] ==== SUMMARY ====")
    for key, label in [
        ("full_generation", "Full generation"),
        ("forced_extraction_0.5", "Forced extraction @50%"),
        ("sg_es", "SG-ES"),
    ]:
        acc, frac = _agg(key)
        print(f"[phase4] {label}: accuracy={acc:.3f}, mean fraction_of_trace_used={frac:.3f}")
    n_triggered = sum(1 for r in results if r["sg_es"]["triggered"])
    print(f"[phase4] SG-ES triggered (vs. fell through to full trace) on {n_triggered}/{len(results)} problems")
    print(f"[phase4] DONE. Results saved to /vol/candidate_b_phase4_sges/phase4_results.json")


@app.local_entrypoint()
def main(n_problems: int = 2, levels: str = "1,2,3,4"):
    level_tuple = tuple(levels.split(","))
    run_phase4.remote(n_problems=n_problems, levels=level_tuple)
