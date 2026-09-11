# archive/

Superseded scripts, kept for project history -- not part of current data
collection. See `../README.md` for what's active.

- `kaggle_pilot.py`, `kaggle_pilot_1_math500_rerun.py`,
  `kaggle_pilot_2_gsm8k_rerun.py`, `kaggle_pilot_3_commonsenseqa_escalation.py`,
  `pilot.py`, `smoke_test.py`, `diagnose_cell.py` -- the original
  training-free early-stopping pilot (pre-Candidate-B pivot). See
  `../../PILOT_RESULTS.md` and the `early_stop_literature_pivot` memory
  for why this direction was set aside.

- `kaggle_candidate_b_micropilot.py` and its `_1of3`/`_2of3`/`_3of3`/
  `_2a_of3`/`_2b_of3` variants -- Candidate B's Step 3 micro-pilot
  (mechanism check, N_BRANCHES_PER_PAIR=15, underpowered by design). See
  `DESIGN_CANDIDATE_B.md` SS4.5-4.9. Complete and superseded by the full
  pilot (`../kaggle_candidate_b_fullpilot_*.py`).

- `kaggle_candidate_b_fullpilot_level1.py` / `level2.py` / `level3.py` --
  the undivided per-level scripts (6 problems, 8 pairs each). Never run:
  timing analysis showed a full stratum in one Kaggle session risks
  exceeding the session cap. Superseded by the `_Nof3.py` 3-way split.

- `local_5090_timing_probe.py` -- standalone timing probe. Superseded by
  `../candidate_b_full_pilot_local.ipynb`, which includes the same probe
  as its first phase plus the full pilot after a go/no-go checkpoint.
