import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
DATA_PATH = REPO_ROOT / 'data/frozen/v1_final/model_a_train_ready.csv'
MODELS_DIR = REPO_ROOT / 'models/model_a/rf_pct'
REPORTS_DIR = REPO_ROOT / 'reports/model_a/metrics'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# % targets (key fix)
df["target_min_pct"] = df["target_rule_min"] / df["original_price"]
df["target_max_pct"] = df["target_rule_max"] / df["original_price"]

feature_cols = [
    "brand", "category", "material", "size", "condition", "tier_primary",
    "age_months", "original_price",
    "base_min_pct", "base_max_pct", "cond_mult", "age_mult", "cat_mult", "mat_mult"
]
cat_cols = ["brand", "category", "material", "size", "condition", "tier_primary"]
num_cols = [c for c in feature_cols if c not in cat_cols]

train_df = df[df["split_set"] == "train"].copy()
val_df = df[df["split_set"] == "val"].copy()
test_df = df[df["split_set"] == "test"].copy()

def bucket_round(v):
    if v < 200:
        return round(v / 10) * 10
    if v <= 1000:
        return round(v / 50) * 50
    return round(v / 100) * 100

def postprocess(pred_min_pct, pred_max_pct, original_price):
    out_min, out_max = [], []
    for a, b, op in zip(pred_min_pct, pred_max_pct, original_price):
        a = max(0.0, min(float(a), 0.20))
        b = max(0.0, min(float(b), 0.20))
        pmin, pmax = a * op, b * op
        if pmin > pmax:
            pmin, pmax = pmax, pmin
        pmin = bucket_round(pmin)
        pmax = bucket_round(pmax)
        if pmax < pmin:
            pmax = pmin
        out_min.append(pmin)
        out_max.append(pmax)
    return np.array(out_min), np.array(out_max)

def make_model():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    return Pipeline([("pre", pre), ("rf", rf)])

def eval_split(name, model_min, model_max, data):
    X = data[feature_cols]
    y_min = data["target_rule_min"].values
    y_max = data["target_rule_max"].values
    op = data["original_price"].values

    pred_min_pct = model_min.predict(X)
    pred_max_pct = model_max.predict(X)
    pred_min, pred_max = postprocess(pred_min_pct, pred_max_pct, op)

    return {
        f"{name}_mae_min": float(mean_absolute_error(y_min, pred_min)),
        f"{name}_rmse_min": float(np.sqrt(mean_squared_error(y_min, pred_min))),
        f"{name}_mae_max": float(mean_absolute_error(y_max, pred_max)),
        f"{name}_rmse_max": float(np.sqrt(mean_squared_error(y_max, pred_max))),
        f"{name}_range_violations_after_fix": int((pred_min > pred_max).sum())
    }

X_train = train_df[feature_cols]
y_min_train = train_df["target_min_pct"].values
y_max_train = train_df["target_max_pct"].values

model_min = make_model()
model_max = make_model()
model_min.fit(X_train, y_min_train)
model_max.fit(X_train, y_max_train)

metrics = {}
metrics.update(eval_split("val", model_min, model_max, val_df))
metrics.update(eval_split("test", model_min, model_max, test_df))

print("Model A RF (% target) Metrics")
for k, v in metrics.items():
    print(f"{k}: {v}")

joblib.dump(model_min, MODELS_DIR / "model_a_min_rf_pct.pkl")
joblib.dump(model_max, MODELS_DIR / "model_a_max_rf_pct.pkl")
with open(REPORTS_DIR / "model_a_rf_pct_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(MODELS_DIR / "model_a_min_rf_pct.pkl")
print(MODELS_DIR / "model_a_max_rf_pct.pkl")
print(REPORTS_DIR / "model_a_rf_pct_metrics.json")
