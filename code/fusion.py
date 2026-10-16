"""SHAP-LIME fused feature selection with nested cross-validation.

  S, L        normalised SHAP and LIME importances, computed on training data only
  I(alpha)    = alpha * S + (1 - alpha) * L
  F(alpha, k) = top-k features of I(alpha)
  (alpha*, k*, theta*) = argmax of validation score of A(D_train[F]; theta)

Outer folds give the reported score; inner folds choose (alpha, k, theta).
"""
import itertools
import time
import warnings

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from common import N_JOBS, encode, score

warnings.filterwarnings("ignore")
METRICS = ("macro_f1", "accuracy")
TREES = ("RandomForestClassifier", "XGBClassifier")


def make_model(name, theta, seed=0, proba=True):
    if name == "RF":
        return RandomForestClassifier(n_estimators=theta.get("n_estimators", 200), max_depth=theta.get("max_depth"),
                                      min_samples_leaf=theta.get("min_samples_leaf", 1), n_jobs=N_JOBS, random_state=seed)
    if name == "XGB":
        return XGBClassifier(n_estimators=theta.get("n_estimators", 300), max_depth=theta.get("max_depth", 6),
                             learning_rate=theta.get("learning_rate", 0.1), subsample=0.9, colsample_bytree=0.9,
                             tree_method="hist", n_jobs=N_JOBS, random_state=seed, verbosity=0)
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=theta.get("C", 1.0), max_iter=1000, random_state=seed))
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVC(C=theta.get("C", 1.0), gamma=theta.get("gamma", "scale"),
                                                   probability=proba, cache_size=2000, random_state=seed))
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=theta.get("hidden", (128, 64)),
                                                             alpha=theta.get("alpha", 1e-4), max_iter=200,
                                                             early_stopping=True, random_state=seed))
    raise ValueError(name)


def normalise(v):
    v = np.abs(np.asarray(v, dtype=float))
    return v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v))


def shap_importance(model, X, n=500, n_background=100, seed=0):
    rng = np.random.RandomState(seed)
    rows = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    est, prep = (model[-1], model[:-1]) if hasattr(model, "steps") else (model, None)
    transform = prep.transform if prep is not None else (lambda a: a)
    rows = transform(rows)
    if type(est).__name__ in TREES:
        sv = shap.TreeExplainer(est).shap_values(rows, check_additivity=False)
    else:
        background = transform(X[rng.choice(len(X), min(n_background, len(X)), replace=False)])
        if type(est).__name__ == "LogisticRegression":
            sv = shap.LinearExplainer(est, background).shap_values(rows)
        else:  # model-agnostic and slow, so fewer rows
            background = shap.kmeans(background, min(20, len(background)))
            sv = shap.KernelExplainer(est.predict_proba, background).shap_values(rows[:60], nsamples=150, silent=True)
    if isinstance(sv, list):  # one array per class
        return normalise(np.mean([np.abs(s).mean(0) for s in sv], axis=0))
    sv = np.asarray(sv)
    return normalise(np.abs(sv).mean(axis=tuple(i for i in range(sv.ndim) if i != 1)))


def lime_importance(model, X, features, classes, n=200, num_samples=1000, seed=0):
    rng = np.random.RandomState(seed)
    explainer = LimeTabularExplainer(X, feature_names=list(features), class_names=list(classes),
                                     discretize_continuous=True, random_state=seed)
    rows = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    labels = list(model.classes_)
    total = np.zeros(X.shape[1])
    for row, pred in zip(rows, model.predict(rows)):
        c = labels.index(pred)
        exp = explainer.explain_instance(row, model.predict_proba, labels=(c,), num_features=X.shape[1], num_samples=num_samples)
        for j, w in exp.as_map()[c]:
            total[j] += abs(w)
    return normalise(total / len(rows))


def fuse(S, L, alpha):
    return alpha * S + (1 - alpha) * L


def rank(importance):
    return np.argsort(-np.asarray(importance), kind="stable")


def explain(model, theta, X, y, features, classes, shap_n, lime_n, seed):
    ref = make_model(model, theta, seed).fit(X, y)
    return (shap_importance(ref, X, n=shap_n, seed=seed),
            lime_importance(ref, X, features, classes, n=lime_n, seed=seed))


def nested_cv(X, y, groups, features, model, thetas, alphas, ks, outer_splits=5, inner_splits=3,
              shap_n=500, lime_n=200, max_rows=None, seed=0, only_folds=None, log=print):
    """max_rows subsamples the training rows used for the grid search and the explanations (slow models only);
    the final model of each outer fold is always trained on the full outer-train set."""
    y, classes = encode(y)

    def splitter(n):
        return GroupKFold(n) if groups is not None else StratifiedKFold(n, shuffle=True, random_state=seed)

    grid = list(itertools.product(alphas, ks, range(len(thetas))))
    rng = np.random.RandomState(seed)
    folds = []
    for fold, (tr, te) in enumerate(splitter(outer_splits).split(X, y, groups)):
        if only_folds is not None and fold not in only_folds:
            continue
        t0 = time.time()
        Xtr, ytr = X[tr], y[tr]
        J = {m: np.zeros(len(grid)) for m in METRICS}
        for itr, iva in splitter(inner_splits).split(Xtr, ytr, groups[tr] if groups is not None else None):
            if max_rows and len(itr) > max_rows:
                itr = rng.choice(itr, max_rows, replace=False)
            S, L = explain(model, thetas[0], Xtr[itr], ytr[itr], features, classes, shap_n, lime_n, seed)
            scores = {}  # different (alpha, k) often give the same feature set: fit it once
            for i, (alpha, k, t) in enumerate(grid):
                F = tuple(rank(fuse(S, L, alpha))[:k])
                if (F, t) not in scores:
                    h = make_model(model, thetas[t], seed, proba=False).fit(Xtr[itr][:, list(F)], ytr[itr])
                    scores[(F, t)] = score(ytr[iva], h.predict(Xtr[iva][:, list(F)]))
                for m in METRICS:
                    J[m][i] += scores[(F, t)][m] / inner_splits
        log(f"fold {fold}: inner search done ({time.time() - t0:.0f}s)")

        sub = np.arange(len(Xtr))
        if max_rows and len(Xtr) > max_rows:
            sub = rng.choice(len(Xtr), max_rows, replace=False)
        S, L = explain(model, thetas[0], Xtr[sub], ytr[sub], features, classes, shap_n, lime_n, seed)
        result = {"fold": fold, "S": dict(zip(features, S.tolist())), "L": dict(zip(features, L.tolist())),
                  "grid": [(a, k, t, J["macro_f1"][i], J["accuracy"][i]) for i, (a, k, t) in enumerate(grid)],
                  "by_metric": {}}
        for metric in METRICS:
            best = int(np.argmax(J[metric]))
            alpha, k, t = grid[best]
            I = fuse(S, L, alpha)
            F = rank(I)[:k]
            pred = make_model(model, thetas[t], seed, proba=False).fit(Xtr[:, F], ytr).predict(X[te][:, F])
            result["by_metric"][metric] = {
                "alpha": alpha, "k": k, "theta": thetas[t], "inner_J": float(J[metric][best]),
                "test_macro_f1": float(f1_score(y[te], pred, average="macro")),
                "test_weighted_f1": float(f1_score(y[te], pred, average="weighted")),
                "test_accuracy": float(np.mean(pred == y[te])),
                "selected": [features[j] for j in F], "I": dict(zip(features, I.tolist())),
                "y_true": y[te].tolist(), "y_pred": pred.tolist()}
            r = result["by_metric"][metric]
            log(f"fold {fold} [{metric}]: alpha={alpha} k={k} theta={thetas[t]} "
                f"macro-F1={r['test_macro_f1']:.3f} acc={r['test_accuracy']:.3f}")
        result["seconds"] = time.time() - t0
        folds.append(result)
    return {"model": model, "classes": classes, "folds": folds, "summary": summarise(folds)}


def summarise(folds):
    return {metric: {m: (float(np.mean([f["by_metric"][metric][m] for f in folds])),
                         float(np.std([f["by_metric"][metric][m] for f in folds])))
                     for m in ("test_macro_f1", "test_weighted_f1", "test_accuracy")}
            for metric in METRICS}
