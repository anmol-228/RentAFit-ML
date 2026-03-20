import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
DATA_PATH = REPO_ROOT / 'data/frozen/v1_final/model_a_train_ready.csv'
MODELS_DIR = REPO_ROOT / 'models/model_a/baseline'
REPORTS_DIR = REPO_ROOT / 'reports/model_a/metrics'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 1) Keep features clean (no IDs, no targets, no split column)
feature_cols = [
    "brand", "category", "material", "size", "condition", "tier_primary",
    "age_months", "original_price",
    "base_min_pct", "base_max_pct", "cond_mult", "age_mult", "cat_mult", "mat_mult"
]
target_min = "target_rule_min"
target_max = "target_rule_max"

# 2) Fixed split (already prepared manually)
train_df = df[df["split_set"] == "train"].copy()
val_df = df[df["split_set"] == "val"].copy()
test_df = df[df["split_set"] == "test"].copy()

X_train, X_val, X_test = train_df[feature_cols], val_df[feature_cols], test_df[feature_cols]
y_min_train, y_min_val, y_min_test = train_df[target_min], val_df[target_min], test_df[target_min]
y_max_train, y_max_val, y_max_test = train_df[target_max], val_df[target_max], test_df[target_max]

cat_cols = ["brand", "category", "material", "size", "condition", "tier_primary"]
num_cols = [c for c in feature_cols if c not in cat_cols]

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

def build_model():
    return Pipeline([
        ("prep", make_preprocessor()),
        ("reg", LinearRegression())
    ])

def evaluate(name, model_min, model_max, X, y_min, y_max):
    pred_min = model_min.predict(X)
    pred_max = model_max.predict(X)

    # enforce range consistency
    pred_min_fixed = np.minimum(pred_min, pred_max)
    pred_max_fixed = np.maximum(pred_min, pred_max)

    out = {
        f"{name}_mae_min": float(mean_absolute_error(y_min, pred_min_fixed)),
        f"{name}_rmse_min": float(np.sqrt(mean_squared_error(y_min, pred_min_fixed))),
        f"{name}_mae_max": float(mean_absolute_error(y_max, pred_max_fixed)),
        f"{name}_rmse_max": float(np.sqrt(mean_squared_error(y_max, pred_max_fixed))),
        f"{name}_range_violations_before_fix": int((pred_min > pred_max).sum()),
    }
    return out

model_min = build_model()
model_max = build_model()

model_min.fit(X_train, y_min_train)
model_max.fit(X_train, y_max_train)

metrics = {}
metrics.update(evaluate("val", model_min, model_max, X_val, y_min_val, y_max_val))
metrics.update(evaluate("test", model_min, model_max, X_test, y_min_test, y_max_test))

print("Model A Baseline Metrics")
for k, v in metrics.items():
    print(f"{k}: {v}")

joblib.dump(model_min, MODELS_DIR / "model_a_min_linear.pkl")
joblib.dump(model_max, MODELS_DIR / "model_a_max_linear.pkl")
with open(REPORTS_DIR / "model_a_baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\\nSaved:")
print(MODELS_DIR / "model_a_min_linear.pkl")
print(MODELS_DIR / "model_a_max_linear.pkl")
print(REPORTS_DIR / "model_a_baseline_metrics.json")
