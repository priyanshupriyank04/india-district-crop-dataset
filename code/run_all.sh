#!/usr/bin/env bash
# Runs every experiment in order. Slow on one machine (India SVM/MLP/XGB take hours each);
# the per-fold commands are independent and can be run in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODELS="LR MLP RF XGB SVM"

for d in india kaggle benue ethiopia; do
  for m in $MODELS; do $PY code/run_fusion.py --dataset $d --model $m; done
done

for f in 0 1 2 3 4; do
  for m in $MODELS; do $PY code/feature_selection.py --model $m --fold $f; done
  $PY code/ensembles.py --fold $f
  for m in LR RF XGB; do $PY code/fidelity.py --model $m --fold $f; done
  for fam in RF ET HGB XGB; do $PY code/improve.py --family $fam --fold $f; done
  $PY code/improve.py --ensemble --fold $f
done
for m in $MODELS; do $PY code/leakage.py --model $m; done

$PY code/analyze.py
