#!/usr/bin/env bash
# Reproduce every result: runs Q1, Q2 and Q3 in order on one GPU with papermill.
# Usage: bash scripts/run_all.sh <ROLL_NUMBER>
# Outputs: results/q*_outputs/ (tables, figures, JSON, saved split, run cache), results/executed/*.ipynb
# (executed notebooks = full logs) and results/logs/*.log (papermill console logs).
set -euo pipefail
ROLL="${1:?usage: bash scripts/run_all.sh <ROLL_NUMBER>}"
cd "$(dirname "$0")/.."
mkdir -p results/executed results/logs data
python -c "import torch; assert torch.cuda.is_available(), 'No GPU visible: the budget is sized for a GPU'; print('GPU:', torch.cuda.get_device_name(0))"
for pair in "q1:Q1_MLP_CIFAR10" "q2:Q2_CNN_vs_MLP_CIFAR10" "q3:Q3_Autoencoders_CIFAR10"; do
  q="${pair%%:*}"; nb="${pair##*:}"
  echo "=== ${nb} ==="
  papermill "notebooks/${nb}.ipynb" "results/executed/${nb}.ipynb" \
      -f "configs/${q}.yaml" -p ROLL_NUMBER "${ROLL}" --cwd . --log-output 2>&1 | tee "results/logs/${q}.log"
done
python scripts/check_results.py
