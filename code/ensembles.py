"""Ensembles of the five tuned models on India (voting and stacking), with recommendation metrics.

Each model uses the features and hyperparameters chosen for it in run_fusion.py.

  python code/ensembles.py --fold 0
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import MODELS, RESULTS, class_proba, india_folds, load_run, save_json, score
from fusion import make_model, rank


def calibration_error(y, P, n_bins=15):
    conf, correct = P.max(1), P.argmax(1) == y
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def metrics(y, P):
    return {**score(y, P.argmax(1)),
            "top3_accuracy": float((np.argsort(-P, 1)[:, :3] == y[:, None]).any(1).mean()),
            "macro_roc_auc": float(np.mean([roc_auc_score(y == c, P[:, c]) for c in np.unique(y)])),
            "brier": float(((P - np.eye(P.shape[1])[y]) ** 2).sum(1).mean()),
            "ece": calibration_error(y, P)}


ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
args = ap.parse_args()

X, y, groups, feats, _, classes, folds = india_folds()
nc = len(classes)
tr, te = folds[args.fold]
inner = list(GroupKFold(3).split(X[tr], y[tr], groups[tr]))
col = {f: i for i, f in enumerate(feats)}

P, P15, oof, weight = {}, {}, {}, {}
for m in MODELS:
    fr = load_run(m)["folds"][args.fold]
    best = fr["by_metric"]["macro_f1"]
    F = [col[f] for f in best["selected"]]
    top15 = list(rank([fr["S"][f] for f in feats])[:15])
    new = lambda: make_model(m, best["theta"], proba=(m != "SVM"))
    P[m] = class_proba(new().fit(X[tr][:, F], y[tr]), X[te][:, F], nc)
    P15[m] = class_proba(new().fit(X[tr][:, top15], y[tr]), X[te][:, top15], nc)
    oof[m] = np.zeros((len(tr), nc))
    for itr, iva in inner:
        oof[m][iva] = class_proba(new().fit(X[tr][itr][:, F], y[tr][itr]), X[tr][iva][:, F], nc)
    weight[m] = best["inner_J"]
    print(f"fold {args.fold}: {m} done")


def votes(models, w=None):
    V = np.zeros((len(te), nc))
    for m in models:
        V[np.arange(len(te)), P[m].argmax(1)] += w[m] if w else 1
    return V.argmax(1)


soft = {"Soft voting (5)": sum(P[m] for m in MODELS) / 5,
        "Weighted soft voting (5)": sum(P[m] * weight[m] for m in MODELS) / sum(weight.values()),
        "Soft voting (RF+XGB+SVM)": (P["RF"] + P["XGB"] + P["SVM"]) / 3,
        "Soft voting (RF+XGB)": (P["RF"] + P["XGB"]) / 2,
        "Soft voting, SHAP top-15 (5)": sum(P15[m] for m in MODELS) / 5}
stack = LogisticRegression(max_iter=2000).fit(np.hstack([oof[m] for m in MODELS]), y[tr])
hard = {"Hard voting (5)": votes(MODELS), "Weighted hard voting (5)": votes(MODELS, weight),
        "Stacking (LR meta-learner)": stack.predict(np.hstack([P[m] for m in MODELS]))}

results = {m: metrics(y[te], P[m]) for m in MODELS}
results.update({f"{m} (SHAP top-15)": metrics(y[te], P15[m]) for m in MODELS})
results.update({name: metrics(y[te], p) for name, p in soft.items()})
results.update({name: score(y[te], p) for name, p in hard.items()})
preds = {**{m: P[m].argmax(1) for m in MODELS}, **{n: p.argmax(1) for n, p in soft.items()}, **hard}
save_json({"fold": args.fold, "results": results, "y_true": y[te].tolist(), "preds": {k: v.tolist() for k, v in preds.items()}},
          RESULTS / f"ensembles/fold{args.fold}.json")
