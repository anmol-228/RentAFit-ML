import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_a.utils.feature_builder import load_brand_master, postprocess_range_from_pct, get_brand_features

BASE_DIR = Path('/Users/mypc/RentAFit')
DATA_PATH = BASE_DIR / 'data/frozen/v1_final/model_a_train_ready.csv'
BRAND_MASTER_PATH = BASE_DIR / 'data/frozen/v1_final/brand_tier_master_project_final.csv'

MODEL_DIR = BASE_DIR / 'models/model_a/rf_pct_tier_split'
REPORT_METRICS_DIR = BASE_DIR / 'reports/model_a/metrics'
REPORT_ANALYSIS_DIR = BASE_DIR / 'reports/model_a/analysis'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

feature_cols = [
    'brand', 'category', 'material', 'size', 'condition', 'tier_primary',
    'age_months', 'original_price',
    'base_min_pct', 'base_max_pct', 'cond_mult', 'age_mult', 'cat_mult', 'mat_mult',
    'brand_avg_price_min', 'brand_avg_price_max', 'is_open_ended_brand_price', 'brand_known_in_master',
]
cat_cols = ['brand', 'category', 'material', 'size', 'condition', 'tier_primary']
num_cols = [c for c in feature_cols if c not in cat_cols]

residual_feature_cols = feature_cols + ['base_pred_max_pct']
residual_cat_cols = cat_cols
residual_num_cols = [c for c in residual_feature_cols if c not in residual_cat_cols]

RESIDUAL_SCALE = 0.35
OVERSAMPLE_MIN_TARGET_PER_BRAND = 14


def make_model(random_state=42):
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols),
    ])
    rf = RandomForestRegressor(
        n_estimators=700,
        max_depth=24,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([('pre', pre), ('rf', rf)])


def make_residual_model(random_state=52):
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), residual_cat_cols),
        ('num', 'passthrough', residual_num_cols),
    ])
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([('pre', pre), ('rf', rf)])


def attach_brand_master_features(df: pd.DataFrame, brand_master_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bm = brand_master_df[['brand_norm', 'avg_price_min_num', 'avg_price_max_num', 'is_open_ended_brand_price']].copy()
    out['brand_norm'] = out['brand'].astype(str).str.strip().str.lower()
    out = out.merge(bm, on='brand_norm', how='left')

    out['brand_avg_price_min'] = out['avg_price_min_num'].fillna(0.0)
    out['brand_avg_price_max'] = out['avg_price_max_num'].fillna(0.0)
    out['is_open_ended_brand_price'] = out['is_open_ended_brand_price'].fillna(0).astype(int)
    out['brand_known_in_master'] = (out['brand_avg_price_min'] > 0).astype(int)

    drop_cols = ['avg_price_min_num', 'avg_price_max_num']
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out


def oversample_tier5_sparse_brands(train_t5: pd.DataFrame, random_state: int = 42):
    if train_t5.empty:
        return train_t5.copy(), {'oversample_added_rows': 0, 'target_per_brand': 0}

    counts = train_t5['brand'].value_counts()
    target_per_brand = int(max(np.percentile(counts.values, 75), OVERSAMPLE_MIN_TARGET_PER_BRAND))

    pieces = [train_t5]
    added_rows = 0
    rng = np.random.RandomState(random_state)

    for brand, cnt in counts.items():
        need = max(0, target_per_brand - int(cnt))
        if need <= 0:
            continue
        part = train_t5[train_t5['brand'] == brand]
        sampled = part.sample(n=need, replace=True, random_state=int(rng.randint(0, 1_000_000)))
        pieces.append(sampled)
        added_rows += need

    out = pd.concat(pieces, ignore_index=True)
    info = {
        'oversample_added_rows': int(added_rows),
        'target_per_brand': int(target_per_brand),
        'brands_in_tier5_train': int(len(counts)),
    }
    return out, info


def predict_pct_with_models(split_df: pd.DataFrame, models: dict, residual_scale: float = RESIDUAL_SCALE):
    X = split_df[feature_cols]
    tier = split_df['tier_primary'].astype(str).values

    pred_min_pct = np.zeros(len(split_df), dtype=float)
    pred_max_pct = np.zeros(len(split_df), dtype=float)
    route_used = np.array(['tier_split_tier1to4'] * len(split_df), dtype=object)

    idx_t5 = np.where(tier == 'Tier 5')[0]
    idx_non_t5 = np.where(tier != 'Tier 5')[0]

    if len(idx_t5) > 0:
        x_t5 = X.iloc[idx_t5].copy()
        base_min_t5 = models['min_t5'].predict(x_t5)
        base_max_t5 = models['max_t5'].predict(x_t5)

        if models.get('max_t5_residual') is not None:
            x_t5_res = x_t5.copy()
            x_t5_res['base_pred_max_pct'] = base_max_t5
            res = models['max_t5_residual'].predict(x_t5_res)
            max_t5 = base_max_t5 + (float(residual_scale) * res)
        else:
            max_t5 = base_max_t5

        pred_min_pct[idx_t5] = base_min_t5
        pred_max_pct[idx_t5] = max_t5
        route_used[idx_t5] = 'tier_split_tier5'

    if len(idx_non_t5) > 0:
        x_non_t5 = X.iloc[idx_non_t5]
        pred_min_pct[idx_non_t5] = models['min_non_t5'].predict(x_non_t5)
        pred_max_pct[idx_non_t5] = models['max_non_t5'].predict(x_non_t5)

    return pred_min_pct, pred_max_pct, route_used


def evaluate_split(name, split_df, models, residual_scale: float = RESIDUAL_SCALE):
    y_min = split_df['target_rule_min'].values
    y_max = split_df['target_rule_max'].values
    op = split_df['original_price'].values

    pred_min_pct, pred_max_pct, route_used = predict_pct_with_models(split_df, models, residual_scale=residual_scale)

    pred_min, pred_max = [], []
    for a, b, price in zip(pred_min_pct, pred_max_pct, op):
        mn, mx = postprocess_range_from_pct(a, b, price)
        pred_min.append(mn)
        pred_max.append(mx)

    pred_min = np.array(pred_min)
    pred_max = np.array(pred_max)

    out_df = split_df.copy()
    out_df['pred_min_pct_raw'] = pred_min_pct
    out_df['pred_max_pct_raw'] = pred_max_pct
    out_df['pred_min'] = pred_min
    out_df['pred_max'] = pred_max
    out_df['abs_error_min'] = (out_df['pred_min'] - out_df['target_rule_min']).abs()
    out_df['abs_error_max'] = (out_df['pred_max'] - out_df['target_rule_max']).abs()
    out_df['pct_error_min'] = np.where(out_df['target_rule_min'] == 0, np.nan, out_df['abs_error_min'] / out_df['target_rule_min'])
    out_df['pct_error_max'] = np.where(out_df['target_rule_max'] == 0, np.nan, out_df['abs_error_max'] / out_df['target_rule_max'])
    out_df['range_violations_after_fix'] = (out_df['pred_min'] > out_df['pred_max']).astype(int)
    out_df['model_route'] = route_used

    metrics = {
        f'{name}_mae_min': float(mean_absolute_error(y_min, pred_min)),
        f'{name}_rmse_min': float(np.sqrt(mean_squared_error(y_min, pred_min))),
        f'{name}_mae_max': float(mean_absolute_error(y_max, pred_max)),
        f'{name}_rmse_max': float(np.sqrt(mean_squared_error(y_max, pred_max))),
        f'{name}_range_violations_after_fix': int((pred_min > pred_max).sum()),
    }

    tier_metrics = (
        out_df
        .groupby('tier_primary', as_index=False)
        .agg(
            rows=('listing_id', 'count'),
            mae_min=('abs_error_min', 'mean'),
            mae_max=('abs_error_max', 'mean'),
            rmse_min=('abs_error_min', lambda s: float(np.sqrt(np.mean(s ** 2)))),
            rmse_max=('abs_error_max', lambda s: float(np.sqrt(np.mean(s ** 2)))),
            avg_pct_error_min=('pct_error_min', 'mean'),
            avg_pct_error_max=('pct_error_max', 'mean'),
            avg_original_price=('original_price', 'mean'),
        )
        .sort_values('tier_primary')
    )

    route_metrics = (
        out_df
        .groupby('model_route', as_index=False)
        .agg(
            rows=('listing_id', 'count'),
            mae_max=('abs_error_max', 'mean'),
            p90_pct_error_max=('pct_error_max', lambda s: float(np.nanquantile(s, 0.90))),
            p95_pct_error_max=('pct_error_max', lambda s: float(np.nanquantile(s, 0.95))),
        )
    )

    return metrics, out_df, tier_metrics, route_metrics


def main():
    df = pd.read_csv(DATA_PATH)
    bm = load_brand_master(str(BRAND_MASTER_PATH))
    df = attach_brand_master_features(df, bm)

    df['target_min_pct'] = df['target_rule_min'] / df['original_price']
    df['target_max_pct'] = df['target_rule_max'] / df['original_price']

    train_df = df[df['split_set'] == 'train'].copy()
    val_df = df[df['split_set'] == 'val'].copy()
    test_df = df[df['split_set'] == 'test'].copy()

    train_t5 = train_df[train_df['tier_primary'] == 'Tier 5'].copy()
    train_non_t5 = train_df[train_df['tier_primary'] != 'Tier 5'].copy()

    if train_t5.empty or train_non_t5.empty:
        raise RuntimeError('Tier split training failed: need both Tier 5 and non-Tier5 rows in train split.')

    train_t5_os, os_info = oversample_tier5_sparse_brands(train_t5, random_state=42)

    models = {
        'min_non_t5': make_model(random_state=42),
        'max_non_t5': make_model(random_state=43),
        'min_t5': make_model(random_state=44),
        'max_t5': make_model(random_state=45),
        'max_t5_residual': make_residual_model(random_state=52),
    }

    models['min_non_t5'].fit(train_non_t5[feature_cols], train_non_t5['target_min_pct'])
    models['max_non_t5'].fit(train_non_t5[feature_cols], train_non_t5['target_max_pct'])
    models['min_t5'].fit(train_t5_os[feature_cols], train_t5_os['target_min_pct'])
    models['max_t5'].fit(train_t5_os[feature_cols], train_t5_os['target_max_pct'])

    # Residual correction for Tier 5 max_pct.
    base_pred_max_t5 = models['max_t5'].predict(train_t5_os[feature_cols])
    residual_target = train_t5_os['target_max_pct'].values - base_pred_max_t5
    train_t5_res = train_t5_os[feature_cols].copy()
    train_t5_res['base_pred_max_pct'] = base_pred_max_t5
    models['max_t5_residual'].fit(train_t5_res[residual_feature_cols], residual_target)

    all_metrics = {}

    train_metrics, train_pred, train_tier, train_route = evaluate_split('train', train_df, models, residual_scale=RESIDUAL_SCALE)
    val_metrics, val_pred, val_tier, val_route = evaluate_split('val', val_df, models, residual_scale=RESIDUAL_SCALE)
    test_metrics, test_pred, test_tier, test_route = evaluate_split('test', test_df, models, residual_scale=RESIDUAL_SCALE)

    all_metrics.update(train_metrics)
    all_metrics.update(val_metrics)
    all_metrics.update(test_metrics)

    # Confidence calibration from validation route metrics.
    confidence = {
        'confidence_threshold': 0.55,
        'route_p90_pct_error_max': {},
        'route_p95_pct_error_max': {},
    }
    for _, r in val_route.iterrows():
        route = str(r['model_route'])
        confidence['route_p90_pct_error_max'][route] = float(r['p90_pct_error_max'])
        confidence['route_p95_pct_error_max'][route] = float(r['p95_pct_error_max'])

    # Save models
    joblib.dump(models['min_non_t5'], MODEL_DIR / 'model_a_min_rf_pct_tier1to4.pkl')
    joblib.dump(models['max_non_t5'], MODEL_DIR / 'model_a_max_rf_pct_tier1to4.pkl')
    joblib.dump(models['min_t5'], MODEL_DIR / 'model_a_min_rf_pct_tier5.pkl')
    joblib.dump(models['max_t5'], MODEL_DIR / 'model_a_max_rf_pct_tier5.pkl')
    joblib.dump(models['max_t5_residual'], MODEL_DIR / 'model_a_max_rf_pct_tier5_residual.pkl')

    seen_brands_non_t5 = sorted(train_non_t5['brand'].astype(str).str.strip().str.lower().unique().tolist())
    seen_brands_t5 = sorted(train_t5['brand'].astype(str).str.strip().str.lower().unique().tolist())

    metadata = {
        'model_family': 'RandomForestRegressor',
        'target_type': 'percentage_targets',
        'tier_strategy': 'split_models_tier5_vs_non_tier5_with_tier5_residual',
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'residual_feature_cols': residual_feature_cols,
        'residual_num_cols': residual_num_cols,
        'data_path': str(DATA_PATH),
        'brand_master_path': str(BRAND_MASTER_PATH),
        'train_rows': int(len(train_df)),
        'train_rows_tier5': int(len(train_t5)),
        'train_rows_tier5_after_oversample': int(len(train_t5_os)),
        'train_rows_non_tier5': int(len(train_non_t5)),
        'val_rows': int(len(val_df)),
        'test_rows': int(len(test_df)),
        'oversample_info': os_info,
        'seen_brands_tier_split_tier1to4': seen_brands_non_t5,
        'seen_brands_tier_split_tier5': seen_brands_t5,
        'confidence': confidence,
        'residual_scale': float(RESIDUAL_SCALE),
    }

    with open(MODEL_DIR / 'model_a_rf_pct_tier_split_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    with open(REPORT_METRICS_DIR / 'model_a_rf_pct_tier_split_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)

    # Save analysis outputs
    train_pred.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_train_predictions.csv', index=False)
    val_pred.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_val_predictions.csv', index=False)
    test_pred.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_test_predictions.csv', index=False)

    train_tier.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_train_tier_summary.csv', index=False)
    val_tier.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_val_tier_summary.csv', index=False)
    test_tier.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_test_tier_summary.csv', index=False)

    train_route.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_train_route_summary.csv', index=False)
    val_route.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_val_route_summary.csv', index=False)
    test_route.to_csv(REPORT_ANALYSIS_DIR / 'model_a_rf_pct_tier_split_test_route_summary.csv', index=False)

    print('Model A Tier-Split RF (% target, v2 with Tier5 residual + oversampling) Metrics')
    for k, v in all_metrics.items():
        print(f'{k}: {v}')

    print('\nOversample info:', os_info)
    print('Saved models in:', MODEL_DIR)
    print('Saved metrics:', REPORT_METRICS_DIR / 'model_a_rf_pct_tier_split_metrics.json')
    print('Saved analysis files in:', REPORT_ANALYSIS_DIR)


if __name__ == '__main__':
    main()
