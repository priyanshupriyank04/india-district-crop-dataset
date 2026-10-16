"""Main experiment: nested-CV SHAP-LIME fusion for one dataset and model.

  python code/run_fusion.py --dataset india --model RF
  python code/run_fusion.py --dataset india --model XGB --fold 0     # one outer fold (run folds in parallel)
  python code/run_fusion.py --dataset india --model XGB --merge      # combine the 5 fold files
"""
import argparse
import json
import time

from common import LOADERS, RESULTS, save_json
from fusion import nested_cv, summarise

THETAS = {
    "RF": [{"n_estimators": 200, "max_depth": None}, {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2}],
    "XGB": [{"n_estimators": 100, "max_depth": 6, "learning_rate": 0.2}, {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.15}],
    "LR": [{"C": 1.0}, {"C": 10.0}],
    "SVM": [{"C": 1.0}, {"C": 10.0}],
    "MLP": [{"hidden": (128, 64)}, {"hidden": (256, 128), "alpha": 1e-3}],
}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_ROWS = {"SVM": 6000, "MLP": 12000, "XGB": 12000}  # India only, to keep the grid search tractable

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="india", choices=list(LOADERS))
ap.add_argument("--model", default="RF", choices=list(THETAS))
ap.add_argument("--fold", type=int)
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
tag = f"{args.dataset}_{args.model}"

if args.merge:
    parts = [json.load(open(RESULTS / f"{tag}_fold{i}.json")) for i in range(5)]
    res = dict(parts[0])
    res["folds"] = [p["folds"][0] for p in parts]
    res["summary"] = summarise(res["folds"])
    res["seconds"] = sum(p["seconds"] for p in parts)
    save_json(res, RESULTS / f"{tag}.json")
    raise SystemExit(f"merged {tag}")

X, y, groups, feats, _ = LOADERS[args.dataset]()
ks = sorted(k for k in {3, 5, 8, 10, 15, 20, len(feats)} if k <= len(feats))
print(f"{tag}: {len(X)} rows, {len(feats)} features, {len(set(y))} classes")
t0 = time.time()
res = nested_cv(X, y, groups, feats, args.model, THETAS[args.model], ALPHAS, ks,
                max_rows=MAX_ROWS.get(args.model) if args.dataset == "india" else None,
                only_folds=None if args.fold is None else [args.fold])
res.update(dataset=args.dataset, features=feats, config={"alphas": ALPHAS, "ks": ks, "thetas": THETAS[args.model]},
           seconds=time.time() - t0)
save_json(res, RESULTS / (f"{tag}.json" if args.fold is None else f"{tag}_fold{args.fold}.json"))
for metric, s in res["summary"].items():
    print(f"tuned on {metric}: macro-F1 {s['test_macro_f1'][0]:.3f} ± {s['test_macro_f1'][1]:.3f}, "
          f"accuracy {s['test_accuracy'][0]:.3f} ± {s['test_accuracy'][1]:.3f}")
