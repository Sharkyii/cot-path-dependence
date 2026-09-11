"""Phase 3 (low-compute plan, mechanistic probe): is "will this prefix go
on to state a decisive answer, or trail off into an empty \\boxed{}"
linearly decodable from its hidden state at the exact matched checkpoint
used for branching?

Reads every results/candidate_b_fullpilot/*.json entry that has
hidden_state_a/hidden_state_b (only pairs collected with
--capture-hidden-states on modal_candidate_b_full_pilot.py have this --
see that script's docstring). Builds one labeled example per side
(hidden_state, decisive-or-not), trains a logistic regression probe with
leave-one-PAIR-out cross-validation (not leave-one-sample-out: the two
sides of a pair are correlated, so scoring a held-out side of a pair
whose OTHER side was in the training set would overstate accuracy).

Usage: python3 scripts/train_decisiveness_probe.py [--threshold 0.5]
"""
import argparse
import glob
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "candidate_b_fullpilot"


def _decisive(answer_counts: dict[str, int], threshold: float) -> int | None:
    total = sum(answer_counts.values())
    if total == 0:
        return None
    empty = answer_counts.get("", 0)
    return int((empty / total) < threshold)


def load_examples(threshold: float):
    X, y, groups, meta = [], [], [], []
    n_pairs_seen = 0
    n_pairs_usable = 0
    for f in sorted(glob.glob(str(RESULTS / "*.json"))):
        for e in json.load(open(f)):
            n_pairs_seen += 1
            if e.get("hidden_state_a") is None or e.get("hidden_state_b") is None:
                continue
            n_pairs_usable += 1
            pair_id = f"{f}:{e['problem_id']}:{e['level']}:{n_pairs_usable}"
            for side in ("a", "b"):
                counts = e[f"side_{side}_answer_counts"]
                label = _decisive(counts, threshold)
                if label is None:
                    continue
                X.append(e[f"hidden_state_{side}"])
                y.append(label)
                groups.append(pair_id)
                meta.append((e["problem_id"], side, label))
    return np.array(X), np.array(y), np.array(groups), meta, n_pairs_seen, n_pairs_usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="empty-rate below this = 'decisive' (label 1)")
    args = ap.parse_args()

    X, y, groups, meta, n_seen, n_usable = load_examples(args.threshold)

    print(f"Pairs seen across all result files: {n_seen}")
    print(f"Pairs WITH hidden states (usable): {n_usable}")
    print(f"Probe examples (2 sides/pair): {len(X)}")

    if n_usable == 0:
        print("\nNo pairs with hidden states yet -- this is expected until a run")
        print("with --capture-hidden-states has been collected. Nothing to train.")
        return

    n_pos = int(y.sum())
    print(f"Class balance: {n_pos} decisive / {len(y) - n_pos} not-decisive")

    n_groups = len(set(groups))
    if n_groups < 4:
        print(f"\nOnly {n_groups} distinct pairs -- too few for a trustworthy "
              f"leave-one-pair-out estimate. Reporting anyway, but treat this as a "
              f"pilot check, not a real result (same caveat as everywhere else in "
              f"this project when n is this small).")

    majority_baseline = max(n_pos, len(y) - n_pos) / len(y)

    clf = LogisticRegression(max_iter=2000)
    logo = LeaveOneGroupOut()
    preds = cross_val_predict(clf, X, y, groups=groups, cv=logo)
    accuracy = float((preds == y).mean())

    print(f"\nMajority-class baseline accuracy: {majority_baseline:.3f}")
    print(f"Leave-one-pair-out probe accuracy: {accuracy:.3f}")
    if accuracy > majority_baseline:
        print("Probe beats the majority baseline -- decisiveness is at least "
              "partially linearly decodable from hidden state at this checkpoint.")
    else:
        print("Probe does NOT beat the majority baseline at this n -- no evidence "
              "of linear decodability yet (could be too little data, or a real null).")


if __name__ == "__main__":
    main()
