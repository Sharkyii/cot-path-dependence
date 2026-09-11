"""Run the continuous state_distance-vs-PDI regression (Phase 1 item 3 of
the low-compute plan; see CANDIDATE_B_PREREGISTRATION.md 5.4) across every
results/candidate_b_fullpilot/*.json file that has the per-pair state_distance
field (added 2026-09-11 to modal_candidate_b_full_pilot.py -- pairs collected
before that date won't have it and are reported separately, not silently
dropped).

Usage: python3 scripts/analyze_continuous_regression.py
"""
import glob
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from early_stop.path_dependence import regress_pdi_vs_state_distance

RESULTS = REPO_ROOT / "results" / "candidate_b_fullpilot"


def main():
    with_distance = []
    without_distance = 0
    for f in sorted(glob.glob(str(RESULTS / "*.json"))):
        for e in json.load(open(f)):
            if "state_distance" in e:
                with_distance.append((e["state_distance"], e["observed_pdi"]))
            else:
                without_distance += 1

    print(f"Pairs WITH state_distance (usable): {len(with_distance)}")
    print(f"Pairs WITHOUT state_distance (pre-fix, not usable here): {without_distance}")

    if len(with_distance) < 3:
        print("\nNot enough pairs with state_distance yet to fit a regression.")
        print("This is expected until new data is collected with the 2026-09-11 fix.")
        return

    result = regress_pdi_vs_state_distance(with_distance)
    print(f"\nn = {result.n_pairs}")
    print(f"slope = {result.slope:.4f}  (PDI per unit state_distance)")
    print(f"intercept = {result.intercept:.4f}  (predicted PDI at distance=0)")
    print(f"r = {result.r_value:.4f}")
    print(f"p = {result.p_value:.4f}")
    print(f"std_err = {result.std_err:.4f}")


if __name__ == "__main__":
    main()
