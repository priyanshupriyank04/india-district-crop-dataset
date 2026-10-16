"""How much does the way we split the data change the score? (India, all features)

  python code/leakage.py --model RF
"""
import argparse

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import MODELS, RESULTS, india_folds, load_run, save_json, score
from fusion import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=MODELS)
args = ap.parse_args()

X, y, groups, feats, meta, _, district_folds = india_folds()
theta = load_run(args.model)["folds"][0]["by_metric"]["macro_f1"]["theta"]
year = meta["year"]
splits = {
    "Random 5-fold": list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)),
    "District-grouped 5-fold": district_folds,
    "State-grouped 5-fold": list(GroupKFold(5).split(X, y, meta["state"])),
    "Later years held out (train <=2010, test >=2011)": [(np.where(year <= 2010)[0], np.where(year >= 2011)[0])],
}
results = {}
for name, folds in splits.items():
    s = [score(y[te], make_model(args.model, theta, proba=False).fit(X[tr], y[tr]).predict(X[te])) for tr, te in folds]
    results[name] = {m: (float(np.mean([x[m] for x in s])), float(np.std([x[m] for x in s]))) for m in ("macro_f1", "accuracy")}
    print(f"{args.model} {name}: accuracy {results[name]['accuracy'][0]:.3f}")
save_json({"model": args.model, "results": results}, RESULTS / f"leakage/{args.model}.json")
