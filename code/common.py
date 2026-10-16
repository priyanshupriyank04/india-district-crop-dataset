"""Data loading and helpers shared by all experiment scripts."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
N_JOBS = int(os.environ.get("N_JOBS", -1))
MODELS = ["LR", "MLP", "RF", "XGB", "SVM"]

INDIA = ROOT / "data/final/india_district_crop_dataset_1997_2015.csv"
KAGGLE = ROOT / "supplementary_datasets/kaggle_ingle/Crop_recommendation.csv"
BENUE = ROOT / "supplementary_datasets/benue_nigeria/benue_original_dataset.csv"
ETHIOPIA = ROOT / "supplementary_datasets/ethiopia/Crop Recommendation using Soil Properties and Weather Prediction.csv"

CLIMATE = ["temp_mean_c", "temp_max_c", "temp_min_c", "humidity_pct", "rainfall_mm", "rain_days_proxy_mm_per_day",
           "solar_mj_m2_day", "wind_m_s", "annual_rainfall_mm", "annual_temp_mean_c", "annual_humidity_pct", "elevation_m"]
NPK = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# Each loader returns X, y, groups (None = no grouping), feature names, meta.
def load_india(min_rows=100):
    df = pd.read_csv(INDIA, low_memory=False)
    counts = df.dominant_crop.value_counts()
    df = df[df.dominant_crop.isin(counts[counts >= min_rows].index)].reset_index(drop=True)
    soil = [c for c in df.columns if c.startswith("soil_") and c.endswith("_0_30cm")]
    season = pd.get_dummies(df.season, prefix="season").astype(float)
    X = pd.concat([df[soil + CLIMATE], season], axis=1)  # no lat/lon: the model must not memorise places
    meta = {"state": df.state.values, "year": df.crop_year.values}
    return X.values.astype(float), df.dominant_crop.values, df.district_key.values, list(X.columns), meta


def load_kaggle():
    df = pd.read_csv(KAGGLE)
    return df[NPK].values.astype(float), df.label.values, None, NPK, {}


def load_benue():
    df = pd.read_csv(BENUE)
    return df[NPK].values.astype(float), df.crop_label.str.strip().values, None, NPK, {}


def load_ethiopia():
    df = pd.read_csv(ETHIOPIA)
    feats = [c for c in df.columns if c not in ("Soilcolor", "label")]
    return df[feats].values.astype(float), df.label.values, None, feats, {}


LOADERS = {"india": load_india, "kaggle": load_kaggle, "benue": load_benue, "ethiopia": load_ethiopia}


def encode(y):
    codes, classes = pd.factorize(pd.Series(y), sort=True)
    return np.asarray(codes), [str(c) for c in classes]


def india_folds():
    """India data with integer labels and the 5 district-grouped outer folds used everywhere."""
    X, y, groups, feats, meta = load_india()
    y, classes = encode(y)
    folds = list(GroupKFold(5).split(X, y, groups))
    return X, y, groups, feats, meta, classes, folds


def score(y_true, y_pred):
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_true, y_pred))}


def class_proba(model, X, n_classes):
    """Class probabilities over all classes; softmax of decision scores if the model has no predict_proba."""
    try:
        p = model.predict_proba(X)
    except AttributeError:
        d = model.decision_function(X)
        p = np.exp(d - d.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = p
    return out


def load_run(model, dataset="india"):
    with open(RESULTS / f"{dataset}_{model}.json") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def compare(y, pred_a, pred_b, groups, n_boot=500, seed=0):
    """a minus b: macro-F1 and accuracy differences with district-bootstrap 95% CIs, and exact McNemar p."""
    districts = np.unique(groups)
    rows = {d: np.where(groups == d)[0] for d in districts}
    rng = np.random.RandomState(seed)
    boots = [np.concatenate([rows[d] for d in rng.choice(districts, len(districts))]) for _ in range(n_boot)]
    ok_a, ok_b = pred_a == y, pred_b == y
    f1 = lambda p, i: f1_score(y[i], p[i], average="macro")
    d_f1 = [f1(pred_a, i) - f1(pred_b, i) for i in boots]
    d_acc = [ok_a[i].mean() - ok_b[i].mean() for i in boots]
    n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    return {"f1_diff": f1_score(y, pred_a, average="macro") - f1_score(y, pred_b, average="macro"),
            "f1_ci": (np.percentile(d_f1, 2.5), np.percentile(d_f1, 97.5)),
            "acc_diff": ok_a.mean() - ok_b.mean(),
            "acc_ci": (np.percentile(d_acc, 2.5), np.percentile(d_acc, 97.5)),
            "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0}
