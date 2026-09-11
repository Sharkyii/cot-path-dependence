"""Problem-clustered aggregate permutation test (2026-09-12 revision,
prompted by external review catching that Fisher's method on discrete
per-pair p-values is distorted by degenerate PDI=0 pairs, whose
permutation p-value is exactly 1 by construction, not weak null evidence
-- pooling them with informative pairs mechanically drags the combined
statistic toward 1, while naively restricting to PDI>0 pairs is itself a
post-hoc selection that inflates significance the other direction).

This test avoids both failure modes: it never discards or filters pairs
by their outcome, and it accounts for multiple pairs sharing a problem
(the independence assumption Fisher's method and Bonferroni both need,
which review also correctly flagged as violated -- e.g. d1_prob1 and
d2_prob1 each have two sampled pairs).

Statistic: mean-of-problem-means of PDI, i.e. average each problem's
pairs together first, then average across problems, so a problem
contributing more pairs doesn't get more weight than one problem, one
vote. Null: for EVERY replicate, resplit EVERY pair's own pooled branches
(same procedure as pair_permutation_test, done independently per pair),
recompute the same problem-clustered statistic. This is a single
global test, not 66 separate ones, so no multiple-comparisons correction
is needed and no pair is ever excluded regardless of its outcome.
"""
import glob
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from early_stop.path_dependence import answer_distribution, pdi

RESULTS_DIR = REPO_ROOT / "results" / "candidate_b_fullpilot"


def load_all_pairs():
    """Normalizes both the original (branches_a/branches_b) and round2
    (side_a_raw_answers/side_b_raw_answers) key naming onto one schema."""
    pairs = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        if "followup" in f:
            continue  # exploratory L2/L3 n=2 check, not part of the main pooled design
        for e in json.load(open(f)):
            if "side_a_raw_answers" in e:
                a, b = e["side_a_raw_answers"], e["side_b_raw_answers"]
            elif "branches_a" in e:
                a, b = e["branches_a"], e["branches_b"]
            else:
                continue
            pairs.append({"problem_id": e["problem_id"], "a": a, "b": b})
    return pairs


def clustered_statistic(pair_pdis: list[tuple[str, float]]) -> float:
    """Mean of per-problem means -- one problem, one vote."""
    by_problem: dict[str, list[float]] = {}
    for pid, val in pair_pdis:
        by_problem.setdefault(pid, []).append(val)
    problem_means = [float(np.mean(vals)) for vals in by_problem.values()]
    return float(np.mean(problem_means))


def main(reps: int = 5000, seed: int = 0):
    pairs = load_all_pairs()
    n_pairs = len(pairs)
    n_problems = len(set(p["problem_id"] for p in pairs))
    print(f"Loaded {n_pairs} pairs across {n_problems} distinct problems")

    observed_pdis = [(p["problem_id"], pdi(answer_distribution(p["a"]), answer_distribution(p["b"])))
                      for p in pairs]
    t_obs = clustered_statistic(observed_pdis)
    print(f"Observed problem-clustered statistic T = {t_obs:.5f}")

    rng = np.random.default_rng(seed)
    null_ts = np.empty(reps)
    pooled_cache = [(p["problem_id"], p["a"] + p["b"], len(p["a"]), len(p["b"])) for p in pairs]

    for r in range(reps):
        resplit_pdis = []
        for pid, pooled, n_a, n_b in pooled_cache:
            idx = rng.permutation(len(pooled))
            a_resplit = [pooled[k] for k in idx[:n_a]]
            b_resplit = [pooled[k] for k in idx[n_a:]]
            resplit_pdis.append((pid, pdi(answer_distribution(a_resplit), answer_distribution(b_resplit))))
        null_ts[r] = clustered_statistic(resplit_pdis)

    p_value = (float((null_ts >= t_obs).sum()) + 1) / (reps + 1)
    print(f"Null mean T = {null_ts.mean():.5f}, null SD = {null_ts.std():.5f}")
    print(f"One-sided p-value (T_obs vs null, {reps} reps) = {p_value:.5f}")


if __name__ == "__main__":
    main()
