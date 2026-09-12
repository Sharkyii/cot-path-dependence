"""Diagnostics for the four checkable critiques from the 2026-09-13
external review round. Each one either survives or it doesn't, on data we
already have, at zero GPU cost.

1. INVERSE NORMALIZER ARTIFACT. Two divergences in this dataset turned out
   to be answer-normalizer artifacts ("2000"/"2000calories", "evelyn"/"e"),
   both caught by hand-inspecting flagged (nonzero-PDI) pairs. Nobody
   checked the 56 zero-PDI pairs for the OPPOSITE error: genuinely distinct
   answers collapsed onto one string by normalization, which would
   manufacture a false zero. Computes PDI on RAW answer strings and on
   NORMALIZED ones for every pair, and reports any pair where normalization
   moved the number. If normalization is systematically hiding divergence,
   the null majority is not trustworthy.

2. GLOBAL RIGHT-CENSORING. Traces run 2000-3000 tokens but branches are
   capped at 1500, so every branch set is censored relative to the
   generation it came from. The Section 6.3 audit examined this for two
   problems; the reviewer's point is that it applies to all 66 pairs and
   plausibly suppresses divergence globally. We cannot recover
   finish_reason retroactively (never logged for the original 66), but the
   empty-answer rate IS the observable footprint of censoring, so we report
   it per problem.

3. EFFECT SIZE IN CONTEXT. PDI is Jensen-Shannon divergence in [0, 1].
   Reports the observed aggregate against that scale, and against the
   within-pair sampling noise floor, so "p = 0.0005" can be read next to
   what it is actually measuring.

4. PREFIX REUSE BELOW THE PROBLEM LEVEL. K=5 trajectories per problem but
   up to 8 sampled pairs per problem, so pairs within a problem necessarily
   reuse trajectories. trajectory_id was never persisted, but each side's
   (confidence, entropy, normalized_position) triple acts as a fingerprint
   for the prefix it came from: repeated triples within a problem mean
   repeated prefixes. Problem-level clustering does not fix this, it only
   accounts for the outer level.

Run:
    .venv/bin/python scripts/review_diagnostics.py
"""
import glob
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from early_stop.parsers import normalize_math_answer
from early_stop.path_dependence import answer_distribution, pdi

RESULTS_DIR = REPO_ROOT / "results" / "candidate_b_fullpilot"


def load_pairs():
    pairs = []
    for fpath in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        fname = pathlib.Path(fpath).name
        if "followup" in fname:
            continue
        track = "round1" if "round2" not in fname else "round2"
        for e in json.load(open(fpath)):
            if "side_a_raw_answers" in e:
                a, b = e["side_a_raw_answers"], e["side_b_raw_answers"]
            elif "branches_a" in e:
                a, b = e["branches_a"], e["branches_b"]
            else:
                continue
            pairs.append({
                "problem_id": e["problem_id"], "raw_a": a, "raw_b": b,
                "observed_pdi": e["observed_pdi"], "null_mean": e["null_mean"],
                "n": e["n_a"], "track": track,
                "state_a": e.get("side_a_state"), "state_b": e.get("side_b_state"),
            })
    return pairs


def _dist(answers):
    """answer_distribution() raises on an all-unparseable set, and treats
    None as droppable. We need EMPTY to be its own category here, since
    whether an answer was stated at all is exactly what some of these
    pairs differ on."""
    cleaned = [x if x else "EMPTY" for x in answers]
    return answer_distribution(cleaned)


def _pdi_raw(p):
    return pdi(_dist(p["raw_a"]), _dist(p["raw_b"]))


def _pdi_normalized(p):
    na = [normalize_math_answer(x) if x else "EMPTY" for x in p["raw_a"]]
    nb = [normalize_math_answer(x) if x else "EMPTY" for x in p["raw_b"]]
    return pdi(answer_distribution(na), answer_distribution(nb))


def check_1_inverse_normalizer_artifact(pairs):
    print("=" * 78)
    print("1. INVERSE NORMALIZER ARTIFACT: does normalization hide divergence?")
    print("=" * 78)
    moved_down, moved_up = [], []
    for p in pairs:
        raw, norm = _pdi_raw(p), _pdi_normalized(p)
        if norm < raw - 1e-9:
            moved_down.append((p, raw, norm))
        elif norm > raw + 1e-9:
            moved_up.append((p, raw, norm))

    print(f"  pairs where normalization LOWERED PDI (hid divergence): {len(moved_down)}")
    for p, raw, norm in moved_down:
        collapsed_a = {x for x in p["raw_a"] if x} - {normalize_math_answer(x) for x in p["raw_a"] if x}
        collapsed_b = {x for x in p["raw_b"] if x} - {normalize_math_answer(x) for x in p["raw_b"] if x}
        print(f"    {p['problem_id']:10s} raw={raw:.4f} -> norm={norm:.4f}  "
              f"(track {p['track']}, n={p['n']})")
        print(f"       raw A distinct: {sorted({x for x in p['raw_a'] if x})}")
        print(f"       raw B distinct: {sorted({x for x in p['raw_b'] if x})}")
        if collapsed_a or collapsed_b:
            print(f"       strings changed by normalization: A={sorted(collapsed_a)} B={sorted(collapsed_b)}")
    print(f"  pairs where normalization RAISED PDI: {len(moved_up)}")

    zero_norm = [p for p in pairs if _pdi_normalized(p) == 0.0]
    zero_raw = [p for p in pairs if _pdi_raw(p) == 0.0]
    print(f"\n  zero-PDI pairs on RAW strings:        {len(zero_raw)} of {len(pairs)}")
    print(f"  zero-PDI pairs on NORMALIZED strings: {len(zero_norm)} of {len(pairs)}")
    manufactured = [p for p in pairs if _pdi_normalized(p) == 0.0 and _pdi_raw(p) > 0.0]
    print(f"  zeros MANUFACTURED by normalization:  {len(manufactured)}")
    if manufactured:
        for p in manufactured:
            print(f"    {p['problem_id']}: raw PDI={_pdi_raw(p):.4f} collapsed to 0")
    print("  => if this count is 0, the null majority is not a normalization artifact.\n")


def check_2_global_censoring(pairs):
    print("=" * 78)
    print("2. GLOBAL RIGHT-CENSORING: empty-answer rate across ALL pairs")
    print("=" * 78)
    print("  (branches capped at 1500 tokens; traces ran 2000-3000. finish_reason")
    print("   was never logged for these 66 pairs, so empty rate is the only")
    print("   observable footprint of censoring available retroactively.)\n")
    by_problem = defaultdict(lambda: [0, 0])
    for p in pairs:
        for side in ("raw_a", "raw_b"):
            for x in p[side]:
                by_problem[p["problem_id"]][1] += 1
                if not x:
                    by_problem[p["problem_id"]][0] += 1
    total_empty = sum(v[0] for v in by_problem.values())
    total_all = sum(v[1] for v in by_problem.values())
    for pid, (empty, tot) in sorted(by_problem.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        rate = empty / tot if tot else 0
        flag = "  <-- nonzero" if empty else ""
        print(f"    {pid:10s} {empty:4d}/{tot:4d} empty ({rate*100:5.1f}%){flag}")
    print(f"\n  OVERALL: {total_empty}/{total_all} branches empty ({100*total_empty/total_all:.1f}%)")
    print("  => censoring, if it acts globally, would show up as empties spread")
    print("     across many problems, not concentrated in a few.\n")


def check_3_effect_size_in_context(pairs):
    print("=" * 78)
    print("3. EFFECT SIZE IN CONTEXT: what is 0.006 on a [0, 1] scale?")
    print("=" * 78)
    excess = np.array([p["observed_pdi"] - p["null_mean"] for p in pairs])
    by_problem = defaultdict(list)
    for p in pairs:
        by_problem[p["problem_id"]].append(p["observed_pdi"] - p["null_mean"])
    clustered = float(np.mean([float(np.mean(v)) for v in by_problem.values()]))
    null_means = np.array([p["null_mean"] for p in pairs])
    print(f"  aggregate (problem-clustered mean excess PDI): {clustered:.6f}")
    print(f"  PDI is Jensen-Shannon divergence, bounded [0, 1]")
    print(f"  => the aggregate effect is {clustered*100:.3f}% of the metric's full range")
    print(f"  mean within-pair sampling noise floor (null_mean): {null_means.mean():.6f}")
    print(f"  => the effect is {clustered/null_means.mean():.2f}x the average noise floor")
    nonzero = [p for p in pairs if p["observed_pdi"] > 0]
    print(f"\n  the 3 problems carrying it, largest single-pair PDI: "
          f"{max(p['observed_pdi'] for p in nonzero):.4f}")
    print(f"  pairs at exactly zero: {len(pairs) - len(nonzero)} of {len(pairs)}")
    print("  => a significant p-value here is a statement about whether 0.006 is")
    print("     distinguishable from 0, NOT about whether 0.006 matters.\n")


def check_4_prefix_reuse(pairs):
    print("=" * 78)
    print("4. PREFIX REUSE BELOW THE PROBLEM LEVEL")
    print("=" * 78)
    print("  K=5 trajectories per problem, but up to 8 pairs sampled per problem.")
    print("  trajectory_id was never persisted; (confidence, entropy, position)")
    print("  fingerprints the prefix instead. Repeated fingerprints = reused prefixes.\n")
    by_problem = defaultdict(list)
    for p in pairs:
        if p["state_a"] is None:
            continue
        for s in (p["state_a"], p["state_b"]):
            by_problem[p["problem_id"]].append(
                (round(s["confidence"], 6), round(s["entropy"], 6), round(s["normalized_position"], 6))
            )
    if not by_problem:
        print("  (no state data on these pairs)\n")
        return
    print(f"  {'problem':12s} {'sides':>6s} {'distinct prefixes':>18s} {'reuse factor':>14s}")
    for pid, fps in sorted(by_problem.items(), key=lambda kv: -len(kv[1])):
        n_sides, n_distinct = len(fps), len(set(fps))
        print(f"  {pid:12s} {n_sides:6d} {n_distinct:18d} {n_sides/n_distinct:13.2f}x")
    all_fps = [fp for v in by_problem.values() for fp in v]
    print(f"\n  OVERALL: {len(all_fps)} pair-sides drawn from {len(set(all_fps))} distinct prefixes")
    print(f"  => each distinct prefix is used {len(all_fps)/len(set(all_fps)):.2f}x on average.")
    print("     Problem-level clustering accounts for the outer level only; pairs")
    print("     within a problem are overlapping draws, not independent units.\n")


def main():
    pairs = load_pairs()
    print(f"\nLoaded {len(pairs)} pairs across "
          f"{len({p['problem_id'] for p in pairs})} problems\n")
    check_1_inverse_normalizer_artifact(pairs)
    check_2_global_censoring(pairs)
    check_3_effect_size_in_context(pairs)
    check_4_prefix_reuse(pairs)


if __name__ == "__main__":
    main()
