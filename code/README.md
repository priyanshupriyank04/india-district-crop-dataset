# Experiments: SHAP-LIME fused feature selection

Code for the crop-recommendation experiments run on this dataset and on the three supplementary datasets.

## Method

For a classifier `A` with hyperparameters `θ`:

1. `S`, `L`: normalised SHAP and LIME feature importances, computed **on training data only**
2. `I(α) = α·S + (1 − α)·L`
3. Rank features by `I(α)` and keep the top `k`
4. Train `A(θ)` on those `k` features
5. Choose `(α, k, θ)` by validation macro-F1

Evaluation is nested cross-validation. The outer 5 folds give the reported score, and the inner 3 folds choose `(α, k, θ)`. On the India data both levels are grouped by district, so test districts are never seen in training.

Grid: `α ∈ {0, 0.25, 0.5, 0.75, 1}`, `k ∈ {3, 5, 8, 10, 15, 20, all}`, two `θ` per model. Models: Logistic Regression, MLP, Random Forest, XGBoost, SVM.

## Files

| Script | What it does | Output (`results/`) |
|---|---|---|
| `common.py` | Dataset loaders, folds, scoring, significance test | |
| `fusion.py` | SHAP/LIME importance, fusion, nested CV, model definitions | |
| `run_fusion.py` | Main experiment for one dataset and model | `<dataset>_<model>.json` |
| `feature_selection.py` | India: SHAP vs LIME vs fused vs permutation, random, MI, ANOVA, L1, RFE, PCA and feature groups at fixed `k` | `selection/` |
| `ensembles.py` | India: hard/soft/weighted voting and stacking of the five tuned models; top-3 accuracy, ROC-AUC, Brier, ECE | `ensembles/` |
| `leakage.py` | India: random vs district vs state vs later-years splits | `leakage/` |
| `fidelity.py` | India: deletion test and stability of local SHAP and LIME explanations | `fidelity/` |
| `improve.py` | India: tuned RF, ExtraTrees, HistGradientBoosting and XGBoost with class weights and engineered features; nested choice of the best soft-voting ensemble | `improve/` |
| `analyze.py` | Tables, figures and significance tests | `analysis/` |
| `run_all.sh` | Runs everything in order | |

`feature_selection.py`, `ensembles.py`, `leakage.py` and `fidelity.py` read the India results of `run_fusion.py`, so run that first for all five models.

## Run

```bash
pip install -r code/requirements.txt
bash supplementary_datasets/kaggle_ingle/download.sh     # Kaggle file is not stored in the repo
python code/run_fusion.py --dataset india --model RF
python code/analyze.py
```

`bash code/run_all.sh` runs everything. On one machine this takes about a day, mostly SVM, MLP and XGBoost on India. Commands that take `--fold` are independent, so they can run in parallel. Use `N_JOBS` to limit threads per process and `RESULTS_DIR` to change the output folder.
