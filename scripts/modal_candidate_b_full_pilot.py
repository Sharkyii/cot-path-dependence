"""
CANDIDATE B FULL PILOT -- Modal is ONE track of a two-track plan, not a
replacement for Kaggle. Current split (see chat/DESIGN_CANDIDATE_B.md):
  - Kaggle (free, 3 parallel tabs max): "1of3" chunks (rerun, all 3
    levels) + "2of3" chunks (fresh, all 3 levels) -- 18 pairs, $0.
  - Modal (this script, paid, ~$14 budget): "3of3" chunks, all 3 levels,
    reduced to 1 pair/chunk (full 40 branches/pair preserved) -- 3 pairs,
    run in TRUE parallel via `--slice 3of3` so it finishes alongside
    Kaggle's first wave rather than adding to the critical path.

COMPREHENSIVE SAVING (the fix for an earlier Kaggle gap: the first 3
"1of3" notebooks originally saved only PDI/p-value, not raw branch
answers -- since fixed in both the Kaggle scripts and here): every
trajectory's scored-prefix count and every branched pair's FULL raw
outcome -- side A/B answer counts AND the raw per-branch answer lists, not
just PDI/p-value -- are written to the Modal Volume and committed
incrementally (after every trajectory, after every pair). If any chunk's
container dies or times out partway, everything it completed up to that
point is still on the volume and inspectable.

WHY REDUCED SCOPE ON MODAL (1 pair/chunk, not the full 3):
Real per-branch generation time on an A10G was never fully measured before
budget became the binding constraint (see chat history -- a full 3-pair
chunk was conservatively estimated at ~$12, and 3 of those exceeds the
~$14 allotted to this track). Reducing to 1 pair/chunk keeps the branch
count per pair at the full pre-registered 40 (what actually matters for
statistical power, per the power analysis in DESIGN_CANDIDATE_B.md SS4.9)
while fitting the budget -- cutting pairs, not branches-per-pair.

Function timeout is set to 20h (Modal's max is 24h) as a safety margin --
irrelevant at this reduced scope (a 1-pair chunk finishes in a few hours
at most) but kept in case `--all` or a larger `--max-pairs` is used later.

Usage:
    pip install modal
    python3 -m modal setup                              # one-time auth

    # The current plan: all 3 levels' "3of3" chunks, true parallel, 1 pair
    # each (fits the ~$14 budget for this track):
    modal run scripts/modal_candidate_b_full_pilot.py --slice 3of3 --max-pairs 1

    # Single chunk, full pre-registered design (3 pairs, 40 branches) --
    # useful for a real cost/timing check before committing more budget:
    modal run scripts/modal_candidate_b_full_pilot.py --level 1 --sub-start 0 --sub-end 2

    # All 9 chunks, full design -- expensive (~$70+), only after checking
    # real per-chunk cost first:
    modal run scripts/modal_candidate_b_full_pilot.py --all

To pull results back down:
    modal volume get candidate-b-results / ./modal_results_full_pilot
"""

import pathlib
import sys

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
K_TRAJECTORIES_PER_PROBLEM = 5
N_BRANCHES_PER_PAIR = 40           # SS4.9 power-corrected value
MAX_PAIRS_PER_CHUNK = 3            # same proven-safe size as the Kaggle sub-notebooks (ceil(8*2/6))
MAX_NEW_TOKENS_TRACE = 3000        # bumped from 2000: level3_4-6's two problems (digit-search /
                                    # smallest-multiple style) truncated mid-reasoning at 2000,
                                    # never reaching \boxed{} naturally -- see level3_4-6 diagnostic
                                    # (trace cut off mid-sentence: "Wait, perhaps 2220"), which then
                                    # cascaded into 0/500 forced-extraction parses for that chunk.
MAX_NEW_TOKENS_SCORE = 64          # bumped from 48: a little extra room to close \boxed{} even
                                    # when the model hedges ("...} Wait, actually...") before committing.
MAX_NEW_TOKENS_BRANCH = 1500
TEMPERATURE = 0.6
CONFIDENCE_TOL = 0.10
ENTROPY_TOL = 0.15
POSITION_TOL = 0.15
MIN_PARSE_RATE = 0.80

# All 9 chunks: 3 difficulty levels x 3 problem-slices each (0-2, 2-4, 4-6)
# -- identical structure to the 9 Kaggle sub-notebooks, so total coverage
# and per-chunk pair budget exactly matches CANDIDATE_B_PREREGISTRATION.md.
ALL_CHUNKS = [
    (level, sub_start, sub_start + 2)
    for level in ("1", "2", "3")
    for sub_start in (0, 2, 4)
]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("candidate-b-full-pilot")

# Persistent volume: HF model cache (avoids re-downloading 15GB on every
# chunk -- a real Modal advantage over Kaggle here) AND all result output.
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
    timeout=20 * 60 * 60,  # 20h -- margin under Modal's 24h max; see module docstring
    volumes={VOL_PATH: volume},
)
def run_chunk(
    difficulty: str, sub_start: int, sub_end: int,
    n_branches: int = N_BRANCHES_PER_PAIR, max_pairs: int = MAX_PAIRS_PER_CHUNK,
    match_levels: tuple[str, ...] = ("L1", "L2", "L3"),
    capture_hidden_states: bool = False,
):
    """n_branches/max_pairs default to the real pre-registered design
    (40 branches/pair, 3 pairs/chunk) -- override with smaller values for
    a cheap timing/cost diagnostic run (e.g. n_branches=10, max_pairs=1)
    before committing budget to the full design.

    match_levels restricts which of L1/L2/L3 Phase 3 actually branches --
    e.g. ("L2", "L3") for a targeted follow-up on a pair that already has
    L1 data, without re-spending on L1. Each level in match_levels gets
    its own independent `max_pairs` budget (see the 2026-09-10 amendment
    in CANDIDATE_B_PREREGISTRATION.md: the original code shared one budget
    counter across all three levels, so L1 always exhausted it before L2/L3
    ever ran -- every one of the 21 pairs collected so far is L1-only).

    capture_hidden_states=True additionally runs one extra forward pass per
    side per pair (early_stop/backend.py's extract_hidden_state()) and
    saves the resulting activation vector into each pair's saved entry, for
    the Phase 3 mechanistic probe (low-compute plan, 2026-09-11): is
    "will this prefix go on to state a decisive answer or trail off empty"
    linearly decodable from hidden state? Cheap (no generation, ~1s/call)
    but must be captured now -- prefix text is not persisted after a run
    ends, so this cannot be added retroactively to already-collected pairs."""
    import json
    import os
    import time

    os.environ["HF_HOME"] = str(VOL_PATH / "hf_cache")
    sys.path.insert(0, "/root")

    from datasets import load_dataset

    from early_stop.backend import HFBackend, DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B, probe_gpu
    from early_stop.candidate_b_pipeline import (
        generate_diverse_trajectories, build_trajectory_prefixes, branch_naturally,
    )
    from early_stop.path_dependence import (
        find_matched_pairs, stratified_sample_pairs, pair_permutation_test, state_distance,
    )
    from early_stop.segmentation import split_into_steps, step_prefix

    chunk_tag = f"level{difficulty}_{sub_start}-{sub_end}"
    out_dir = VOL_PATH / "candidate_b_full_pilot" / chunk_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = probe_gpu()
    print(f"[{chunk_tag}] {gpu.summary()}")

    t0 = time.time()
    backend = HFBackend(
        model_name=MODEL_REPO, temperature=TEMPERATURE,
        param_count_b=DEEPSEEK_R1_DISTILL_QWEN_7B_PARAMS_B,
        # Forced fp16, NOT "auto" -- every Kaggle run (0-4% unparseable,
        # consistently) used fp16 because T4 doesn't support bf16. The
        # first Modal run auto-selected bf16 (A10G supports it) and saw
        # 65-82% unparseable, a severe, consistent regression -- DeepSeek
        # R1-Distill models are known to be precision-sensitive. Testing
        # the precision hypothesis directly by forcing the previously-
        # proven-working precision instead of the auto-selected one.
        precision="fp16",
    )
    print(f"[{chunk_tag}] Model loaded in {time.time() - t0:.1f}s")
    volume.commit()

    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    level_ds = math_ds.filter(lambda r: str(r["level"]) == difficulty)
    if len(level_ds) < 6:
        raise SystemExit(f"Only {len(level_ds)} MATH-500 problems at level {difficulty}, need 6")
    level_problems = level_ds.select(range(6))               # same 6, full stratum
    problems = level_problems.select(range(sub_start, sub_end))  # this chunk's slice

    print(f"[{chunk_tag}] {len(problems)} problems (global indices {sub_start}-{sub_end-1})")

    all_prefixes = []
    trace_steps_cache = {}
    problem_prompts = {}
    n_unparseable = 0

    # ---- PHASE 1 ----
    for _local_pi, item in enumerate(problems):
        pi = _local_pi + sub_start
        problem_id = f"d{difficulty}_prob{pi}"
        base_prompt = item["problem"]
        problem_prompts[problem_id] = base_prompt
        print(f"\n[{chunk_tag}] --- {problem_id} ---")
        for ti in range(K_TRAJECTORIES_PER_PROBLEM):
            traj_id = f"traj_{ti}"
            t0 = time.time()
            traces = generate_diverse_trajectories(backend, base_prompt, k=1, max_new_tokens=MAX_NEW_TOKENS_TRACE)
            trace_text = traces[0]
            if _local_pi == 0 and ti == 0:
                # Diagnostic only, first trajectory of the whole chunk --
                # so if the unparseable rate is still bad after forcing
                # fp16, we have actual raw text to look at instead of
                # guessing again. First/last 600 chars: the boxed answer
                # (if any) is usually near the end.
                print(f"[{chunk_tag}]   DIAGNOSTIC -- first trace, first 600 chars:")
                print(trace_text[:600])
                print(f"[{chunk_tag}]   DIAGNOSTIC -- first trace, last 600 chars:")
                print(trace_text[-600:])
            tp = build_trajectory_prefixes(
                backend, problem_id, traj_id, base_prompt, trace_text,
                max_new_tokens=MAX_NEW_TOKENS_SCORE,
            )
            all_prefixes.extend(tp.prefixes)
            n_unparseable += tp.n_unparseable
            trace_steps_cache[(problem_id, traj_id)] = split_into_steps(trace_text)
            elapsed = time.time() - t0
            print(f"[{chunk_tag}]   {traj_id}: {len(tp.prefixes)} scored, {tp.n_unparseable} unparseable, {elapsed:.1f}s")

            # incremental save -- every trajectory, not just at the end
            with open(out_dir / "phase1_progress.json", "w") as f:
                json.dump({
                    "chunk": chunk_tag,
                    "n_prefixes_so_far": len(all_prefixes),
                    "n_unparseable_so_far": n_unparseable,
                    "last_trajectory_elapsed_s": elapsed,
                }, f, indent=2)
            volume.commit()

    print(f"\n[{chunk_tag}] Phase 1 done: {len(all_prefixes)} scored prefixes, {n_unparseable} unparseable")

    # ---- PHASE 2 ----
    pairs_by_level = {}
    for match_level in ("L1", "L2", "L3"):
        pairs = find_matched_pairs(
            all_prefixes, match_level,
            confidence_tol=CONFIDENCE_TOL, entropy_tol=ENTROPY_TOL, position_tol=POSITION_TOL,
        )
        pairs_by_level[match_level] = pairs
        print(f"[{chunk_tag}]   {match_level}: {len(pairs)} matched pairs")

    with open(out_dir / "phase2_matched_pair_counts.json", "w") as f:
        json.dump({k: len(v) for k, v in pairs_by_level.items()}, f, indent=2)
    volume.commit()

    # ---- PHASE 3 ----
    chunk_pair_results = []
    level_seed = {"L1": 1, "L2": 2, "L3": 3}
    for match_level in match_levels:
        budget = max_pairs  # independent per level -- see match_levels docstring above
        if budget <= 0:
            break
        sampled = stratified_sample_pairs(pairs_by_level[match_level], budget, seed=level_seed[match_level])
        for a, b in sampled:
            try:
                steps_a = trace_steps_cache[(a.problem_id, a.trajectory_id)]
                steps_b = trace_steps_cache[(b.problem_id, b.trajectory_id)]
                prefix_text_a = step_prefix(steps_a, a.step_index)
                prefix_text_b = step_prefix(steps_b, b.step_index)
                base_prompt = problem_prompts[a.problem_id]

                t0 = time.time()
                branch_set_a = branch_naturally(backend, base_prompt, prefix_text_a, n_branches, MAX_NEW_TOKENS_BRANCH)
                branch_set_b = branch_naturally(backend, base_prompt, prefix_text_b, n_branches, MAX_NEW_TOKENS_BRANCH)
                result = pair_permutation_test(branch_set_a, branch_set_b, min_parse_rate=MIN_PARSE_RATE, reps=2000, seed=0)
                elapsed = time.time() - t0

                hidden_state_a = hidden_state_b = None
                if capture_hidden_states:
                    # Same exact input branch_naturally() used (base_prompt +
                    # "\n\n" + prefix_step_text), so the captured activation
                    # corresponds to the precise checkpoint that was branched
                    # -- see early_stop/backend.py's extract_hidden_state()
                    # docstring: this cannot be reconstructed after the fact,
                    # must be captured in the same run.
                    hidden_state_a = backend.extract_hidden_state(base_prompt + "\n\n" + prefix_text_a)
                    hidden_state_b = backend.extract_hidden_state(base_prompt + "\n\n" + prefix_text_b)

                def _dist(bs):
                    counts = {}
                    for ans in bs.parsed_answers():
                        counts[ans] = counts.get(ans, 0) + 1
                    return counts

                entry = {
                    "chunk": chunk_tag,
                    "level": match_level,
                    "difficulty": difficulty,
                    "problem_id": a.problem_id,
                    "observed_pdi": result.observed_pdi,
                    "null_mean": result.null_mean,
                    "effect_size": result.effect_size,
                    "p_value": result.p_value,
                    "n_a": result.n_a,
                    "n_b": result.n_b,
                    "side_a_answer_counts": _dist(branch_set_a),
                    "side_b_answer_counts": _dist(branch_set_b),
                    "side_a_raw_answers": branch_set_a.parsed_answers(),
                    "side_b_raw_answers": branch_set_b.parsed_answers(),
                    "elapsed_s": elapsed,
                    # Phase 1 fix (2026-09-11): persist the matched checkpoint's own
                    # observable state so state_distance()-vs-PDI can be regressed
                    # continuously later, instead of only via the discrete L1/L2/L3
                    # buckets -- see CANDIDATE_B_PREREGISTRATION.md 5.4. Without this
                    # the continuous test can't be run even retroactively.
                    "side_a_state": {
                        "answer": a.state.answer, "confidence": a.state.confidence,
                        "entropy": a.state.entropy, "normalized_position": a.normalized_position,
                    },
                    "side_b_state": {
                        "answer": b.state.answer, "confidence": b.state.confidence,
                        "entropy": b.state.entropy, "normalized_position": b.normalized_position,
                    },
                    "state_distance": state_distance(a.state, b.state),
                    "hidden_state_a": hidden_state_a,
                    "hidden_state_b": hidden_state_b,
                }
                chunk_pair_results.append(entry)
                print(f"[{chunk_tag}]   [{match_level}] {a.problem_id}: PDI={result.observed_pdi:.3f} "
                      f"null={result.null_mean:.3f} p={result.p_value:.4f} "
                      f"(n_a={result.n_a} n_b={result.n_b}) [{elapsed:.1f}s -- USE THIS to project remaining cost]")
                print(f"[{chunk_tag}]       side A: {entry['side_a_answer_counts']}")
                print(f"[{chunk_tag}]       side B: {entry['side_b_answer_counts']}")
                budget -= 1

                # incremental save -- every pair, full raw data
                with open(out_dir / "pair_results.json", "w") as f:
                    json.dump(chunk_pair_results, f, indent=2)
                volume.commit()
            except Exception as e:
                print(f"[{chunk_tag}]   [{match_level}] branching ERROR: {type(e).__name__}: {e}")
            if budget <= 0:
                break

    print(f"\n[{chunk_tag}] DONE. {len(chunk_pair_results)} pairs branched, saved to {out_dir}/pair_results.json")
    return chunk_pair_results


SLICE_RANGES = {"1of3": (0, 2), "2of3": (2, 4), "3of3": (4, 6)}


@app.local_entrypoint()
def main(
    level: str = "", sub_start: int = -1, sub_end: int = -1, all: bool = False,
    slice: str = "",
    n_branches: int = N_BRANCHES_PER_PAIR, max_pairs: int = MAX_PAIRS_PER_CHUNK,
    levels: str = "L1,L2,L3",
    capture_hidden_states: bool = False,
):
    match_levels = tuple(x.strip() for x in levels.split(",") if x.strip())
    if all:
        print(f"Launching all {len(ALL_CHUNKS)} chunks in parallel "
              f"(n_branches={n_branches}, max_pairs={max_pairs}, match_levels={match_levels}, "
              f"capture_hidden_states={capture_hidden_states})...")
        print("COST WARNING: real per-branch timing is not yet measured. If you haven't")
        print("run a single chunk first to check the real 'elapsed_s' numbers, consider")
        print("Ctrl+C now and running one chunk alone first. See this file's docstring.\n")
        args = [(lvl, s, e, n_branches, max_pairs, match_levels, capture_hidden_states) for lvl, s, e in ALL_CHUNKS]
        results = list(run_chunk.starmap(args))
        total_pairs = sum(len(r) for r in results)
        print(f"\nAll chunks complete. Total pairs branched: {total_pairs}")
    elif slice:
        if slice not in SLICE_RANGES:
            raise SystemExit(f"--slice must be one of {list(SLICE_RANGES)}, got {slice!r}")
        s, e = SLICE_RANGES[slice]
        print(f"Launching all 3 levels' '{slice}' chunks (problems {s}-{e-1}) in TRUE parallel "
              f"(n_branches={n_branches}, max_pairs={max_pairs}, match_levels={match_levels}, "
              f"capture_hidden_states={capture_hidden_states})...")
        args = [(lvl, s, e, n_branches, max_pairs, match_levels, capture_hidden_states) for lvl in ("1", "2", "3")]
        results = list(run_chunk.starmap(args))
        total_pairs = sum(len(r) for r in results)
        print(f"\nAll 3 '{slice}' chunks complete. Total pairs branched: {total_pairs}")
    elif level and sub_start >= 0 and sub_end >= 0:
        print(f"Running single chunk: level {level}, problems {sub_start}-{sub_end-1} "
              f"(n_branches={n_branches}, max_pairs={max_pairs}, match_levels={match_levels}, "
              f"capture_hidden_states={capture_hidden_states})")
        result = run_chunk.remote(level, sub_start, sub_end, n_branches, max_pairs, match_levels, capture_hidden_states)
        print(f"\nChunk complete: {len(result)} pairs branched.")
        print("Check the printed 'elapsed_s' values above to project real cost for --all.")
    else:
        print("Usage:")
        print("  # Cheap diagnostic (small n_branches, 1 pair) -- run this first:")
        print("  modal run scripts/modal_candidate_b_full_pilot.py --level 1 --sub-start 0 --sub-end 2 --n-branches 10 --max-pairs 1")
        print("  # Full single chunk (real pre-registered design):")
        print("  modal run scripts/modal_candidate_b_full_pilot.py --level 1 --sub-start 0 --sub-end 2")
        print("  # All 3 levels of one slice, TRUE parallel (e.g. the '3of3' track):")
        print("  modal run scripts/modal_candidate_b_full_pilot.py --slice 3of3 --max-pairs 1")
        print("  modal run scripts/modal_candidate_b_full_pilot.py --all")
