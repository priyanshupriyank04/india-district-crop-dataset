"""Turn the result files into tables and figures (results/analysis/). Sections whose results are missing are skipped.

  python code/analyze.py
"""
import glob
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from common import MODELS, RESULTS, compare, india_folds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = RESULTS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
lines = []


def report(text=""):
    lines.append(str(text))
    print(text)


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(str(RESULTS / pattern)))]


def grid_table(run):
    return pd.concat([pd.DataFrame(f["grid"], columns=["alpha", "k", "theta", "J_f1", "J_acc"]).assign(fold=f["fold"])
                      for f in run["folds"]])


# 1. Main nested-CV results ---------------------------------------------------------------
runs = {}
for path in sorted(glob.glob(str(RESULTS / "*_*.json"))):
    if "_fold" not in path:
        r = json.load(open(path))
        runs[(r["dataset"], r["model"])] = r
rows = []
for (ds, m), r in sorted(runs.items()):
    s = r["summary"]["macro_f1"]
    chosen = [(f["by_metric"]["macro_f1"]["alpha"], f["by_metric"]["macro_f1"]["k"]) for f in r["folds"]]
    rows.append({"dataset": ds, "model": m,
                 "macro_f1": f"{s['test_macro_f1'][0]:.3f} ± {s['test_macro_f1'][1]:.3f}",
                 "weighted_f1": f"{s['test_weighted_f1'][0]:.3f} ± {s['test_weighted_f1'][1]:.3f}",
                 "accuracy": f"{s['test_accuracy'][0]:.3f} ± {s['test_accuracy'][1]:.3f}",
                 "alpha_per_fold": "/".join(str(a) for a, _ in chosen), "k_per_fold": "/".join(str(k) for _, k in chosen)})
if rows:
    main = pd.DataFrame(rows)
    main.to_csv(OUT / "main_results.csv", index=False)
    report("1. NESTED-CV TEST RESULTS (tuned on macro-F1, mean ± sd over 5 outer folds)")
    report(main.to_string(index=False))

# 2. Sensitivity to alpha (at fixed k) and to k (inner-CV macro-F1) -------------------------
for ds in sorted({d for d, _ in runs}):
    ds_runs = {m: r for (d, m), r in runs.items() if d == ds}
    n_feats = len(next(iter(ds_runs.values()))["features"])
    k_fixed = [k for k in (5, 10, 15) if k < n_feats]
    fig, axes = plt.subplots(1, len(k_fixed) + 1, figsize=(4 * (len(k_fixed) + 1), 3.4))
    for m, r in ds_runs.items():
        g = grid_table(r)
        for ax, k in zip(axes, k_fixed):
            curve = g[g.k == k].groupby(["fold", "alpha"]).J_f1.max().groupby("alpha").mean()
            ax.plot(curve.index, curve.values, marker="o", label=m)
            ax.set_title(f"k = {k}")
            ax.set_xlabel("alpha (0 = LIME, 1 = SHAP)")
        curve = g.groupby(["fold", "k"]).J_f1.max().groupby("k").mean()
        axes[-1].plot(curve.index, curve.values, marker="o", label=m)
    axes[-1].set_title("best alpha per k")
    axes[-1].set_xlabel("k features kept")
    axes[0].set_ylabel("inner-CV macro-F1")
    axes[0].legend(frameon=False)
    fig.suptitle(ds)
    plt.tight_layout()
    plt.savefig(OUT / f"sensitivity_{ds}.png", dpi=150)
    plt.close()

# 3. SHAP vs LIME agreement and stability across folds --------------------------------------
rows = []
for (ds, m), r in sorted(runs.items()):
    feats = r["features"]
    S = np.array([[f["S"][x] for x in feats] for f in r["folds"]])
    L = np.array([[f["L"][x] for x in feats] for f in r["folds"]])
    stability = lambda M: np.mean([spearmanr(M[i], M[j]).correlation for i in range(len(M)) for j in range(i + 1, len(M))])
    rows.append({"dataset": ds, "model": m, "shap_vs_lime": np.mean([spearmanr(s, l).correlation for s, l in zip(S, L)]),
                 "shap_stability": stability(S), "lime_stability": stability(L)})
if rows:
    agree = pd.DataFrame(rows).round(3)
    agree.to_csv(OUT / "explanation_agreement.csv", index=False)
    report("\n3. SHAP vs LIME RANK AGREEMENT AND FOLD-TO-FOLD STABILITY (Spearman rho)")
    report(agree.to_string(index=False))

# India: pooled held-out predictions, used for per-class results and significance tests
if any(d == "india" for d, _ in runs):
    X, y, groups, feats, _, classes, folds = india_folds()
    test_rows = np.concatenate([te for _, te in folds])
    y_test, g_test = y[test_rows], groups[test_rows]
    india = {m: r for (d, m), r in runs.items() if d == "india"}
    pred = {m: np.concatenate([f["by_metric"]["macro_f1"]["y_pred"] for f in r["folds"]]) for m, r in india.items()}
    best = max(india, key=lambda m: india[m]["summary"]["macro_f1"]["test_macro_f1"][0])
    per_class = pd.DataFrame(classification_report(y_test, pred[best], target_names=classes, output_dict=True, zero_division=0)).T
    per_class.round(3).to_csv(OUT / f"per_class_india_{best}.csv")
    cm = confusion_matrix(y_test, pred[best], normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes)), classes, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix, India, {best} (row-normalised)")
    plt.colorbar(im, fraction=0.03)
    plt.tight_layout()
    plt.savefig(OUT / f"confusion_india_{best}.png", dpi=150)
    plt.close()

# 4. Feature selection ---------------------------------------------------------------------
sel = load("selection/*.json")
if sel:
    df = pd.DataFrame([{"model": d["model"], "fold": d["fold"], **{k: r[k] for k in ("method", "k", "macro_f1", "accuracy")}}
                       for d in sel for r in d["results"]])
    df["method"] = df.method.str.replace(r"Random #\d", "Random (mean of 3)", regex=True)
    df = df.groupby(["model", "fold", "method", "k"], as_index=False)[["macro_f1", "accuracy"]].mean()
    table = df.groupby(["model", "method", "k"], as_index=False)[["macro_f1", "accuracy"]].mean()
    table.to_csv(OUT / "feature_selection_long.csv", index=False)
    models = [m for m in MODELS if m in set(table.model)]
    report("\n4. FEATURE SELECTION: held-out macro-F1 when keeping the top k features")
    for k in (5, 10, 15, 20):
        wide = table[table.k == k].pivot(index="method", columns="model", values="macro_f1")[models]
        wide = wide[~wide.index.str.contains("Soil|Climate|Season|All|pruned")]
        wide["mean"] = wide.mean(1)
        report(f"\n  k = {k}")
        report(wide.sort_values("mean", ascending=False).round(3).to_string())
    fixed = table[table.method.str.contains("Soil|Climate|Season|All|pruned")]
    wide = fixed.pivot_table(index=["method", "k"], columns="model", values="macro_f1")[models]
    wide["mean"] = wide.mean(1)
    report("\n  Fixed feature groups")
    report(wide.sort_values("mean", ascending=False).round(3).to_string())

    show = ["SHAP only (a=1)", "LIME only (a=0)", "Fused, alpha tuned for this k", "Permutation importance",
            "Random (mean of 3)", "PCA (k components)"]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        for method in show:
            s = table[(table.model == m) & (table.method == method)].sort_values("k")
            ax.plot(s.k, s.macro_f1, marker="o", ms=4, label=method)
        ax.axhline(table[(table.model == m) & (table.method == "All features")].macro_f1.iloc[0], color="k", ls=":", lw=1)
        ax.set_title(m)
        ax.set_xlabel("k features kept")
    np.atleast_1d(axes)[0].set_ylabel("held-out macro-F1 (dotted = all features)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "feature_selection.png", dpi=150)
    plt.close()

# 5. Leakage: effect of the split ---------------------------------------------------------
leak = {d["model"]: d["results"] for d in load("leakage/*.json")}
if leak:
    table = pd.DataFrame({m: {s: f"{v['accuracy'][0]:.3f} ({v['macro_f1'][0]:.3f})" for s, v in leak[m].items()}
                          for m in MODELS if m in leak})
    table.to_csv(OUT / "leakage.csv")
    report("\n5. SPLIT COMPARISON: accuracy (macro-F1), all features")
    report(table.to_string())

# 6. Ensembles ----------------------------------------------------------------------------
ens = load("ensembles/fold*.json")
if ens:
    names = list(ens[0]["results"])
    table = pd.DataFrame([{"method": n, **pd.DataFrame([e["results"][n] for e in ens]).mean().to_dict()} for n in names])
    table = table.sort_values("macro_f1", ascending=False).round(3)
    table.to_csv(OUT / "ensembles.csv", index=False)
    report(f"\n6. ENSEMBLES ({len(ens)} folds)")
    report(table.to_string(index=False))
    if len(ens) == 5:
        pooled = {k: np.concatenate([e["preds"][k] for e in ens]) for k in ens[0]["preds"]}
        report("\n  vs Random Forest (district bootstrap 95% CI, McNemar p)")
        for k in ["Soft voting (RF+XGB+SVM)", "Soft voting (5)", "Weighted soft voting (5)", "Stacking (LR meta-learner)"]:
            c = compare(y_test, pooled[k], pooled["RF"], g_test)
            report(f"  {k:28s} macro-F1 {c['f1_diff']:+.4f} [{c['f1_ci'][0]:+.4f}, {c['f1_ci'][1]:+.4f}]  "
                   f"acc {c['acc_diff']:+.4f} [{c['acc_ci'][0]:+.4f}, {c['acc_ci'][1]:+.4f}]  p={c['mcnemar_p']:.4f}")

# 7. Explanation fidelity -----------------------------------------------------------------
fid = load("fidelity/*.json")
if fid:
    rows = [{"model": d["model"], "aopc_shap": np.mean(i["del_shap"]), "aopc_lime": np.mean(i["del_lime"]),
             "aopc_random": np.mean(i["del_random"]), "lime_r2": i["lime_r2"], "lime_top5_jaccard": i["lime_top5_jaccard"],
             "shap_lime_rho": i["shap_lime_rho"], "top5_overlap": len(set(i["shap_top5"]) & set(i["lime_top5"])) / 5}
            for d in fid for i in d["instances"]]
    table = pd.DataFrame(rows).groupby("model").mean().round(3)
    table.to_csv(OUT / "fidelity.csv")
    report("\n7. LOCAL EXPLANATION FIDELITY (aopc = mean probability drop when removing top 1..10 features; higher = more faithful)")
    report(table.to_string())

# 8. Improved models ----------------------------------------------------------------------
imp = load("improve/[A-Z]*_fold*.json")
nested = load("improve/ensemble_fold*.json")
if imp:
    df = pd.DataFrame([{k: d[k] for k in ("family", "fold", "macro_f1", "accuracy", "top3_accuracy")} for d in imp])
    report("\n8. IMPROVED MODELS (engineered features, class weights, nested candidate choice)")
    report(df.groupby("family")[["macro_f1", "accuracy", "top3_accuracy"]].mean().round(4).to_string())
if len(nested) == 5:
    chosen = np.concatenate([d["preds"][d["chosen"]] for d in nested])
    report("  Soft-voting combination chosen per fold: " + ", ".join(d["chosen"] for d in nested))
    report(f"  Nested-selected ensemble: macro-F1 {f1_score(y_test, chosen, average='macro'):.4f}, "
           f"accuracy {(chosen == y_test).mean():.4f}, top-3 {np.mean([d['top3_chosen'] for d in nested]):.4f}")
    c = compare(y_test, chosen, pred["RF"], g_test)
    report(f"  vs Random Forest: macro-F1 {c['f1_diff']:+.4f} [{c['f1_ci'][0]:+.4f}, {c['f1_ci'][1]:+.4f}]  "
           f"acc {c['acc_diff']:+.4f} [{c['acc_ci'][0]:+.4f}, {c['acc_ci'][1]:+.4f}]  p={c['mcnemar_p']:.4f}")

(OUT / "summary.txt").write_text("\n".join(lines))
