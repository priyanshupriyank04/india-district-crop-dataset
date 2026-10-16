"""Stronger tree models on India: tuned settings, class weighting and 7 engineered agronomic features.

For each outer fold the best candidate is chosen on inner district-grouped CV, then scored once on the
held-out districts. The ensemble step picks the best soft-voting combination of the four families the
same way (on inner out-of-fold predictions), so no choice uses the test fold.

  python code/improve.py --family RF --fold 0      # families: RF, ET, HGB, XGB
  python code/improve.py --ensemble --fold 0       # after all four families are done for this fold
"""
import argparse
import itertools
import json
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common import N_JOBS, RESULTS, india_folds, save_json, score

FAMILIES = ["RF", "ET", "HGB", "XGB"]
forest_grid = [{"n_estimators": 500, "max_features": mf, "min_samples_leaf": leaf, "class_weight": cw}
               for mf, leaf, cw in itertools.product(["sqrt", 0.5], [1, 3], [None, "balanced_subsample"])]
CANDIDATES = {
    "RF": forest_grid,
    "ET": forest_grid,
    "HGB": [{"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": n, "class_weight": cw}
            for n, cw in itertools.product([31, 63], [None, "balanced"])],
    "XGB": [{"n_estimators": 300, "max_depth": d, "learning_rate": 0.1, "balanced": b}
            for d, b in itertools.product([6, 8], [False, True])],
}


def engineer(X, feats):
    c = {f: X[:, i] for i, f in enumerate(feats)}
    extra = {
        "temp_range_c": c["temp_max_c"] - c["temp_min_c"],
        "rain_share_of_annual": c["rainfall_mm"] / (c["annual_rainfall_mm"] + 1),
        "aridity_annual_demartonne": c["annual_rainfall_mm"] / (c["annual_temp_mean_c"] + 10),
        "aridity_season_demartonne": c["rainfall_mm"] / (c["temp_mean_c"] + 10 + 1e-6),
        "clay_sand_ratio": c["soil_clay_pct_0_30cm"] / (c["soil_sand_pct_0_30cm"] + 1),
        "c_n_ratio": c["soil_soc_g_kg_0_30cm"] / (c["soil_total_n_g_kg_0_30cm"] + 1e-6),
        "heat_humidity_index": c["temp_mean_c"] * c["humidity_pct"] / 100,
    }
    return np.column_stack([X] + list(extra.values())), feats + list(extra)


def fit(family, params, X, y):
    params = dict(params)
    balanced = params.pop("balanced", False)
    if family == "RF":
        model = RandomForestClassifier(n_jobs=N_JOBS, random_state=0, **params)
    elif family == "ET":
        model = ExtraTreesClassifier(n_jobs=N_JOBS, random_state=0, **params)
    elif family == "HGB":
        model = HistGradientBoostingClassifier(random_state=0, early_stopping=True, validation_fraction=0.1, **params)
    else:
        model = XGBClassifier(subsample=0.9, colsample_bytree=0.8, tree_method="hist", n_jobs=N_JOBS, random_state=0,
                              verbosity=0, **params)
    weights = compute_sample_weight("balanced", y) if balanced else None
    return model.fit(X, y, sample_weight=weights)


def proba(model, X, n_classes):
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--family", choices=FAMILIES)
ap.add_argument("--ensemble", action="store_true")
ap.add_argument("--fold", type=int, required=True)
args = ap.parse_args()

X, y, groups, feats, _, classes, folds = india_folds()
XE, feats_e = engineer(X, feats)
feature_sets = {"base28": X, "engineered35": XE}
nc = len(classes)
tr, te = folds[args.fold]
inner = list(GroupKFold(3).split(tr, y[tr], groups[tr]))
out_dir = RESULTS / "improve"

if not args.ensemble:
    t0 = time.time()
    candidates = [(fs, p) for fs in feature_sets for p in CANDIDATES[args.family]]
    inner_scores = []
    for fs, p in candidates:
        Xf = feature_sets[fs]
        s = [f1_score(y[tr[iva]], fit(args.family, p, Xf[tr[itr]], y[tr[itr]]).predict(Xf[tr[iva]]), average="macro")
             for itr, iva in inner]
        inner_scores.append(float(np.mean(s)))
        print(args.family, args.fold, fs, p, f"inner macro-F1 {inner_scores[-1]:.4f}", flush=True)
    fs, p = candidates[int(np.argmax(inner_scores))]
    model = fit(args.family, p, feature_sets[fs][tr], y[tr])
    P = proba(model, feature_sets[fs][te], nc)
    res = {"family": args.family, "fold": args.fold, "chosen": {"feature_set": fs, "params": p},
           **score(y[te], P.argmax(1)), "top3_accuracy": float((np.argsort(-P, 1)[:, :3] == y[te][:, None]).any(1).mean()),
           "importances": dict(zip(feats_e if fs == "engineered35" else feats, model.feature_importances_.tolist()))
           if hasattr(model, "feature_importances_") else {},
           "y_true": y[te].tolist(), "proba": np.round(P, 5).tolist(), "minutes": (time.time() - t0) / 60}
    save_json(res, out_dir / f"{args.family}_fold{args.fold}.json")
    print(f"{args.family} fold {args.fold}: macro-F1 {res['macro_f1']:.4f}, accuracy {res['accuracy']:.4f}")
else:
    chosen = {f: json.load(open(out_dir / f"{f}_fold{args.fold}.json")) for f in FAMILIES}
    oof = {}
    for f in FAMILIES:
        Xf, p = feature_sets[chosen[f]["chosen"]["feature_set"]], chosen[f]["chosen"]["params"]
        oof[f] = np.zeros((len(tr), nc))
        for itr, iva in inner:
            oof[f][iva] = proba(fit(f, p, Xf[tr[itr]], y[tr[itr]]), Xf[tr[iva]], nc)
    combos = [c for r in range(1, 5) for c in itertools.combinations(FAMILIES, r)]
    test = {f: np.array(chosen[f]["proba"]) for f in FAMILIES}
    avg = lambda probs, combo: sum(probs[f] for f in combo) / len(combo)
    inner_scores = {"+".join(c): float(f1_score(y[tr], avg(oof, c).argmax(1), average="macro")) for c in combos}
    best = max(inner_scores, key=inner_scores.get)
    P = avg(test, best.split("+"))
    save_json({"fold": args.fold, "inner_scores": inner_scores, "chosen": best, "y_true": y[te].tolist(),
               "preds": {"+".join(c): avg(test, c).argmax(1).tolist() for c in combos},
               "top3_chosen": float((np.argsort(-P, 1)[:, :3] == y[te][:, None]).any(1).mean())},
              out_dir / f"ensemble_fold{args.fold}.json")
    print(f"fold {args.fold}: chose {best}")
