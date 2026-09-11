# scripts/

Active scripts for the Candidate B full pilot (see `../CANDIDATE_B_PREREGISTRATION.md`
and `../DESIGN_CANDIDATE_B.md`). Data collection is split across two tracks:

## Kaggle track (free, 3 parallel tabs max)

Paste each `.py` as one Kaggle cell (GPU T4 x2, Internet on). 9 total, one
per (difficulty level x problem-slice):

| File | Difficulty | Problems | Status |
|---|---|---|---|
| `kaggle_candidate_b_fullpilot_level1_1of3.py` | 1 | 0-1 | rerun (backfilling raw answers) |
| `kaggle_candidate_b_fullpilot_level2_1of3.py` | 2 | 0-1 | rerun |
| `kaggle_candidate_b_fullpilot_level3_1of3.py` | 3 | 0-1 | rerun |
| `kaggle_candidate_b_fullpilot_level1_2of3.py` | 1 | 2-3 | fresh |
| `kaggle_candidate_b_fullpilot_level2_2of3.py` | 2 | 2-3 | fresh |
| `kaggle_candidate_b_fullpilot_level3_2of3.py` | 3 | 2-3 | fresh |
| `kaggle_candidate_b_fullpilot_level1_3of3.py` | 1 | 4-5 | superseded by Modal track (kept as reference / fallback) |
| `kaggle_candidate_b_fullpilot_level2_3of3.py` | 2 | 4-5 | superseded by Modal track |
| `kaggle_candidate_b_fullpilot_level3_3of3.py` | 3 | 4-5 | superseded by Modal track |

Results (raw JSON) accumulate in `../results/candidate_b_fullpilot/`.

## Modal track (paid, ~$14 budget)

`modal_candidate_b_full_pilot.py` -- covers the `3of3` slice (problems 4-5,
all 3 levels) in true parallel, reduced to 1 pair/chunk to fit budget while
keeping the full 40-branches-per-pair the power analysis calls for. See the
file's own docstring for exact commands.

## Local (your own GPU)

`candidate_b_full_pilot_local.ipynb` -- run-anywhere notebook (timing
probe -> go/no-go checkpoint -> full pilot), for a local GPU (e.g. an
RTX 5090) instead of Kaggle/Modal.

## archive/

Superseded scripts from earlier phases of this project (the original
early-stopping pilot, the Candidate B micro-pilot, undivided per-level
scripts too large for one Kaggle session). Kept for history, not part of
the current data collection. See `archive/README.md`.
