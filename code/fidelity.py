"""Are the explanations faithful and stable? Local SHAP vs LIME on held-out India rows.

Deletion test: replace a row's top-j features (j = 1..10) with the training mean and record how much the
predicted-class probability drops. A faithful explanation ranks the features whose removal hurts most.

  python code/fidelity.py --model RF --fold 0      # models: LR, RF, XGB
"""
import argparse

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from common import RESULTS, class_proba, india_folds, load_run, save_json
from fusion import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["LR", "RF", "XGB"])
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--rows", type=int, default=120)
args = ap.parse_args()

X, y, groups, feats, _, classes, folds = india_folds()
tr, te = folds[args.fold]
m = len(feats)
theta = load_run(args.model)["folds"][args.fold]["by_metric"]["macro_f1"]["theta"]
model = make_model(args.model, theta).fit(X[tr], y[tr])  # all features, so attributions cover every feature
predict = lambda Z: class_proba(model, Z, len(classes))
rng = np.random.RandomState(args.fold)
rows = rng.choice(te, args.rows, replace=False)
mean = X[tr].mean(0)

if args.model == "LR":
    background = model[:-1].transform(X[tr][rng.choice(len(tr), 200, replace=False)])
    explainer = shap.LinearExplainer(model[-1], background)
    shap_values = lambda x: explainer.shap_values(model[:-1].transform(x))
else:
    explainer = shap.TreeExplainer(model)
    shap_values = lambda x: explainer.shap_values(x, check_additivity=False)
limes = [LimeTabularExplainer(X[tr], feature_names=feats, discretize_continuous=True, random_state=s) for s in range(3)]


def deletion_curve(x, p0, c, order):
    drops = []
    for j in range(1, 11):
        z = x.copy()
        z[order[:j]] = mean[order[:j]]
        drops.append(p0 - predict(z[None])[0, c])
    return drops


out = []
for r in rows:
    x = X[r]
    probs = predict(x[None])[0]
    c = int(probs.argmax())
    sv = shap_values(x[None])
    sv = np.asarray(sv[c][0]) if isinstance(sv, list) else np.asarray(sv)[0, :, c]
    lime_w, lime_r2 = [], []
    for lime in limes:  # three LIME runs with different seeds, to measure stability
        e = lime.explain_instance(x, model.predict_proba, labels=(c,), num_features=m, num_samples=1000)
        w = np.zeros(m)
        for j, v in e.as_map()[c]:
            w[j] = v
        lime_w.append(w)
        lime_r2.append(e.score)
    top5 = [set(np.argsort(-w)[:5]) for w in lime_w]
    jaccard = np.mean([len(a & b) / len(a | b) for i, a in enumerate(top5) for b in top5[i + 1:]])
    out.append({"row": int(r), "pred": c, "true": int(y[r]),
                "del_shap": deletion_curve(x, probs[c], c, np.argsort(-sv)),
                "del_lime": deletion_curve(x, probs[c], c, np.argsort(-lime_w[0])),
                "del_random": deletion_curve(x, probs[c], c, rng.permutation(m)),
                "lime_r2": float(np.mean(lime_r2)), "lime_top5_jaccard": float(jaccard),
                "shap_lime_rho": float(spearmanr(sv, lime_w[0]).correlation),
                "shap_top5": [feats[i] for i in np.argsort(-np.abs(sv))[:5]],
                "lime_top5": [feats[i] for i in np.argsort(-np.abs(lime_w[0]))[:5]]})
save_json({"model": args.model, "fold": args.fold, "instances": out}, RESULTS / f"fidelity/{args.model}_fold{args.fold}.json")
print(f"{args.model} fold {args.fold}: {len(out)} rows explained")
