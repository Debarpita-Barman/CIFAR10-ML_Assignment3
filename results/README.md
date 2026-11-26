# results/

Filled by running the notebooks (see the top-level README). Expected contents:

- `q1_outputs/`, `q2_outputs/`, `q3_outputs/`: tables (`*.csv`), numbers (`results_summary.json`), figures (`fig_*.png`),
  generated interpretation (`interpretation.txt` / `conclusion.txt`), `environment_used.json`, `pip_freeze.txt`.
  `q1_outputs/split_indices.npz` is the saved 45k/5k split. The `runs/` subfolders (checkpoints) are not committed.
- `executed/`: the three notebooks after execution, with all outputs (the complete run logs).
- `logs/`: papermill console logs (written by `scripts/run_all.sh` for local runs).

`python scripts/check_results.py` lists anything that is missing.
