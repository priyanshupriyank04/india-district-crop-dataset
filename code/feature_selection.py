"""Feature-selection comparison on India: which top-k features give the best held-out score?

Uses the outer folds and the SHAP/LIME vectors from run_fusion.py, so every ranking is built from
outer-train data only. Needs results/india_<model>.json for all five models.

  python code/feature_selection.py --model RF --fold 0
"""
import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS, N_JOBS, RESULTS, india_folds, load_run, save_json, score
from fusion import fuse, make_model, normalise, rank

KS = (3, 5, 8, 10, 15, 20)


def model_free_rankings(X, y, tr, fold, feats):
    """Rankings that do not depend on the evaluated model (same for all five)."""
    Xtr, ytr = X[tr], y[tr]
    sub = np.random.RandomState(0).choice(len(tr), min(8000, len(tr)), replace=False)
    Xs, ys = Xtr[sub], ytr[sub]
    l1 = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=300)).fit(Xs, ys)
    rfe = RFE(RandomForestClassifier(100, n_jobs=N_JOBS, random_state=0), n_features_to_select=1, step=2).fit(Xs, ys)
    runs = [load_run(m)["folds"][fold] for m in MODELS]
    shap_all = [normalise([r["S"][f] for f in feats]) for r in runs]
    lime_all = [normalise([r["L"][f] for f in feats]) for r in runs]
    return {
        "Mutual information": mutual_info_classif(Xs, ys, random_state=0),
        "ANOVA F": np.nan_to_num(f_classif(Xtr, ytr)[0]),
        "L1 logistic": np.abs(l1[-1].coef_).sum(0),
        "RFE (random forest)": (len(feats) + 1 - rfe.ranking_).astype(float),
        "Consensus SHAP (5 models)": np.mean(shap_all, 0),
        "Consensus SHAP+LIME (5 models)": np.mean(shap_all + lime_all, 0),
    }


def permutation_scores(model, theta, X, y, groups, tr):
    """Permutation importance on a district-held-out split of outer-train."""
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X[tr], y[tr], groups[tr]))
    rng = np.random.RandomState(0)
    if model == "SVM" and len(a) > 6000:
        a = rng.choice(a, 6000, replace=False)
    if len(b) > 4000:
        b = rng.choice(b, 4000, replace=False)
    h = make_model(model, theta, proba=False).fit(X[tr][a], y[tr][a])
    p = permutation_importance(h, X[tr][b], y[tr][b], n_repeats=3, random_state=0, scoring="f1_macro", n_jobs=1)
    return normalise(np.clip(p.importances_mean, 0, None))


def borda(*importances):
    position = np.zeros(len(importances[0]))
    for imp in importances:
        position[rank(imp)] += np.arange(len(imp))
    return -position


def correlation_pruned(X, order, threshold=0.85):
    corr = np.abs(np.corrcoef(X, rowvar=False))
    kept = []
    for j in order:
        if all(corr[j, i] <= threshold for i in kept):
            kept.append(j)
    return kept


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=MODELS)
ap.add_argument("--fold", type=int, required=True)
args = ap.parse_args()

X, y, groups, feats, _, _, folds = india_folds()
tr, te = folds[args.fold]
run = load_run(args.model)
fr = run["folds"][args.fold]
best = fr["by_metric"]["macro_f1"]
theta = best["theta"]
S = np.array([fr["S"][f] for f in feats])
L = np.array([fr["L"][f] for f in feats])
perm = permutation_scores(args.model, theta, X, y, groups, tr)
other = model_free_rankings(X, y, tr, args.fold, feats)
grid = np.array(fr["grid"])  # alpha, k, theta index, inner macro-F1, inner accuracy

orders = {"SHAP only (a=1)": rank(S), "LIME only (a=0)": rank(L), "Fused (a=0.5)": rank(fuse(S, L, 0.5)),
          "Fused (a*)": rank(fuse(S, L, best["alpha"])), "Permutation importance": rank(perm),
          "Borda (SHAP+LIME+permutation)": rank(borda(S, L, perm))}
orders.update({name: rank(v) for name, v in other.items()})
for s in range(3):
    orders[f"Random #{s}"] = np.random.RandomState(100 + s).permutation(len(feats))

cache, results = {}, []


def evaluate(method, F, th=theta):
    key = (tuple(sorted(F)), str(th))
    if key not in cache:
        cols = list(key[0])
        h = make_model(args.model, th, proba=False).fit(X[tr][:, cols], y[tr])
        cache[key] = score(y[te], h.predict(X[te][:, cols]))
    results.append({"method": method, "k": len(F), **cache[key], "features": [feats[i] for i in F]})


for k in KS:
    for name, order in orders.items():
        evaluate(name, order[:k])
    # alpha re-tuned for this budget k (inner-CV score from run_fusion)
    rows = grid[grid[:, 1] == k]
    a, _, t = rows[rows[:, 3].argmax(), :3]
    evaluate("Fused, alpha tuned for this k", rank(fuse(S, L, a))[:k], run["config"]["thetas"][int(t)])
    scaler = StandardScaler().fit(X[tr])
    pca = PCA(k, random_state=0).fit(scaler.transform(X[tr]))
    h = make_model(args.model, theta, proba=False).fit(pca.transform(scaler.transform(X[tr])), y[tr])
    results.append({"method": "PCA (k components)", "k": k, **score(y[te], h.predict(pca.transform(scaler.transform(X[te])))),
                    "features": []})

soil = [i for i, f in enumerate(feats) if f.startswith("soil_")]
season = [i for i, f in enumerate(feats) if f.startswith("season_")]
climate = [i for i in range(len(feats)) if i not in soil + season]
subsets = {"Soil only": soil, "Climate only": climate, "Season only": season, "Soil + season": soil + season,
           "Climate + season": climate + season, "Soil + climate (no season)": soil + climate,
           "Correlation-pruned (|r|<=0.85)": correlation_pruned(X[tr], rank(other["Consensus SHAP (5 models)"])),
           "All features": list(range(len(feats)))}
for name, F in subsets.items():
    evaluate(name, F)

save_json({"model": args.model, "fold": args.fold, "permutation": dict(zip(feats, perm.tolist())), "results": results},
          RESULTS / f"selection/{args.model}_fold{args.fold}.json")
print(f"{args.model} fold {args.fold}: {len(cache)} feature sets evaluated")
