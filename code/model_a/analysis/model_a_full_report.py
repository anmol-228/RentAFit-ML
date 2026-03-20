import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_a.utils.feature_builder import load_brand_master, postprocess_range_from_pct

BASE = Path('/Users/mypc/RentAFit')
DATA_PATH = BASE / 'data/frozen/v1_final/model_a_train_ready.csv'
BRAND_MASTER_PATH = BASE / 'data/frozen/v1_final/brand_tier_master_project_final.csv'

MODEL_DIR = BASE / 'models/model_a/rf_pct_tier_split'
REPORT_METRICS_DIR = BASE / 'reports/model_a/metrics'
REPORT_ANALYSIS_DIR = BASE / 'reports/model_a/analysis'
CHART_DIR = BASE / 'reports/model_a/charts'

REPORT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'brand', 'category', 'material', 'size', 'condition', 'tier_primary',
    'age_months', 'original_price',
    'base_min_pct', 'base_max_pct', 'cond_mult', 'age_mult', 'cat_mult', 'mat_mult',
    'brand_avg_price_min', 'brand_avg_price_max', 'is_open_ended_brand_price', 'brand_known_in_master',
]
RESIDUAL_FEATURE_COLS = FEATURE_COLS + ['base_pred_max_pct']


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def attach_brand_master_features(df: pd.DataFrame, bm: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['brand_norm'] = out['brand'].astype(str).str.strip().str.lower()
    merge_cols = ['brand_norm', 'avg_price_min_num', 'avg_price_max_num', 'is_open_ended_brand_price']
    out = out.merge(bm[merge_cols], on='brand_norm', how='left')
    out['brand_avg_price_min'] = out['avg_price_min_num'].fillna(0.0)
    out['brand_avg_price_max'] = out['avg_price_max_num'].fillna(0.0)
    out['is_open_ended_brand_price'] = out['is_open_ended_brand_price'].fillna(0).astype(int)
    out['brand_known_in_master'] = (out['brand_avg_price_min'] > 0).astype(int)
    out = out.drop(columns=['avg_price_min_num', 'avg_price_max_num'])
    return out


def load_models():
    models = {
        'min_non_t5': joblib.load(MODEL_DIR / 'model_a_min_rf_pct_tier1to4.pkl'),
        'max_non_t5': joblib.load(MODEL_DIR / 'model_a_max_rf_pct_tier1to4.pkl'),
        'min_t5': joblib.load(MODEL_DIR / 'model_a_min_rf_pct_tier5.pkl'),
        'max_t5': joblib.load(MODEL_DIR / 'model_a_max_rf_pct_tier5.pkl'),
    }
    residual_path = MODEL_DIR / 'model_a_max_rf_pct_tier5_residual.pkl'
    if residual_path.exists():
        models['max_t5_residual'] = joblib.load(residual_path)
    metadata = {}
    meta_path = MODEL_DIR / 'model_a_rf_pct_tier_split_metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    models['metadata'] = metadata
    return models


def predict_split(split_df, models):
    X = split_df[FEATURE_COLS]
    tier = split_df['tier_primary'].astype(str).values
    op = split_df['original_price'].values

    pred_min_pct = np.zeros(len(split_df), dtype=float)
    pred_max_pct = np.zeros(len(split_df), dtype=float)
    route_used = np.array(['tier_split_tier1to4'] * len(split_df), dtype=object)

    idx_t5 = np.where(tier == 'Tier 5')[0]
    idx_non_t5 = np.where(tier != 'Tier 5')[0]
    residual_scale = float((models.get('metadata') or {}).get('residual_scale', 1.0))

    if len(idx_t5) > 0:
        x_t5 = X.iloc[idx_t5].copy()
        pred_min_pct[idx_t5] = models['min_t5'].predict(x_t5)
        base_max = models['max_t5'].predict(x_t5)

        if 'max_t5_residual' in models:
            x_res = x_t5.copy()
            x_res['base_pred_max_pct'] = base_max
            base_max = base_max + (residual_scale * models['max_t5_residual'].predict(x_res[RESIDUAL_FEATURE_COLS]))

        pred_max_pct[idx_t5] = base_max
        route_used[idx_t5] = 'tier_split_tier5'

    if len(idx_non_t5) > 0:
        x_non = X.iloc[idx_non_t5]
        pred_min_pct[idx_non_t5] = models['min_non_t5'].predict(x_non)
        pred_max_pct[idx_non_t5] = models['max_non_t5'].predict(x_non)

    pred_min = []
    pred_max = []
    for a, b, price in zip(pred_min_pct, pred_max_pct, op):
        mn, mx = postprocess_range_from_pct(a, b, price)
        pred_min.append(mn)
        pred_max.append(mx)

    out = split_df.copy()
    out['model_route'] = route_used
    out['pred_min_pct_raw'] = pred_min_pct
    out['pred_max_pct_raw'] = pred_max_pct
    out['pred_min'] = np.array(pred_min)
    out['pred_max'] = np.array(pred_max)
    out['abs_error_min'] = np.abs(out['pred_min'] - out['target_rule_min'])
    out['abs_error_max'] = np.abs(out['pred_max'] - out['target_rule_max'])
    out['pct_error_min'] = np.where(out['target_rule_min'] == 0, np.nan, out['abs_error_min'] / out['target_rule_min'])
    out['pct_error_max'] = np.where(out['target_rule_max'] == 0, np.nan, out['abs_error_max'] / out['target_rule_max'])
    out['range_violation_after_fix'] = (out['pred_min'] > out['pred_max']).astype(int)
    return out


def make_tier_summary(df):
    return (
        df.groupby('tier_primary', as_index=False)
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


def main():
    df = pd.read_csv(DATA_PATH)
    bm = load_brand_master(str(BRAND_MASTER_PATH))
    df = attach_brand_master_features(df, bm)
    models = load_models()

    all_preds = []
    metrics = {}

    for split in ['train', 'val', 'test']:
        part = df[df['split_set'] == split].copy()
        pred = predict_split(part, models)
        all_preds.append(pred)

        metrics[f'{split}_mae_min'] = mae(pred['target_rule_min'].values, pred['pred_min'].values)
        metrics[f'{split}_rmse_min'] = rmse(pred['target_rule_min'].values, pred['pred_min'].values)
        metrics[f'{split}_mae_max'] = mae(pred['target_rule_max'].values, pred['pred_max'].values)
        metrics[f'{split}_rmse_max'] = rmse(pred['target_rule_max'].values, pred['pred_max'].values)
        metrics[f'{split}_range_violations_after_fix'] = int(pred['range_violation_after_fix'].sum())

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all.to_csv(REPORT_ANALYSIS_DIR / 'model_a_full_predictions_all_splits.csv', index=False)

    val_df = pred_all[pred_all['split_set'] == 'val'].copy()
    val_tier_summary = make_tier_summary(val_df)
    val_tier_summary.to_csv(REPORT_ANALYSIS_DIR / 'model_a_full_val_tier_summary.csv', index=False)

    top20 = val_df.sort_values('abs_error_max', ascending=False).head(20)
    top20.to_csv(REPORT_ANALYSIS_DIR / 'model_a_full_val_top20_errors.csv', index=False)

    with open(REPORT_METRICS_DIR / 'model_a_full_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Chart 1: Parity (VAL max)
    plt.figure(figsize=(7, 6))
    plt.scatter(val_df['target_rule_max'], val_df['pred_max'], alpha=0.6, s=20)
    m = max(val_df['target_rule_max'].max(), val_df['pred_max'].max())
    plt.plot([0, m], [0, m], 'r--', linewidth=1)
    plt.xlabel('Actual target_rule_max')
    plt.ylabel('Predicted pred_max')
    plt.title('Model A Tier-Split: Parity Plot (Validation, Max)')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'model_a_tier_split_parity_val_max.png', dpi=180)
    plt.close()

    # Chart 2: Residual hist
    residual = val_df['pred_max'] - val_df['target_rule_max']
    plt.figure(figsize=(7, 5))
    plt.hist(residual, bins=30)
    plt.xlabel('Residual (pred_max - target_rule_max)')
    plt.ylabel('Count')
    plt.title('Model A Tier-Split: Residual Distribution (Validation, Max)')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'model_a_tier_split_residual_hist_val_max.png', dpi=180)
    plt.close()

    # Chart 3: Tier MAE (VAL max)
    plt.figure(figsize=(7, 5))
    plt.bar(val_tier_summary['tier_primary'], val_tier_summary['mae_max'])
    plt.xlabel('Tier')
    plt.ylabel('MAE (Max Price)')
    plt.title('Model A Tier-Split: Tier-wise MAE (Validation, Max)')
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'model_a_tier_split_tier_mae_val_max.png', dpi=180)
    plt.close()

    # Chart 4: model comparison (val_mae_max + val_rmse_max)
    metric_files = {
        'Linear': REPORT_METRICS_DIR / 'model_a_baseline_metrics.json',
        'RF_abs': REPORT_METRICS_DIR / 'model_a_rf_abs_metrics.json',
        'RF_pct': REPORT_METRICS_DIR / 'model_a_rf_pct_metrics.json',
        'RF_pct_tier_split': REPORT_METRICS_DIR / 'model_a_rf_pct_tier_split_metrics.json',
    }
    rows = []
    for name, path in metric_files.items():
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                mvals = json.load(f)
            rows.append({
                'model': name,
                'val_mae_max': mvals.get('val_mae_max', np.nan),
                'val_rmse_max': mvals.get('val_rmse_max', np.nan),
            })

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(REPORT_ANALYSIS_DIR / 'model_a_full_model_comparison_metrics.csv', index=False)

    if not cmp_df.empty:
        x = np.arange(len(cmp_df))
        w = 0.35
        plt.figure(figsize=(9, 5))
        plt.bar(x - w / 2, cmp_df['val_mae_max'], width=w, label='val_mae_max')
        plt.bar(x + w / 2, cmp_df['val_rmse_max'], width=w, label='val_rmse_max')
        plt.xticks(x, cmp_df['model'])
        plt.ylabel('Error')
        plt.title('Model A Comparison (Validation)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(CHART_DIR / 'model_a_tier_split_comparison_val.png', dpi=180)
        plt.close()

    print('Saved metrics:', REPORT_METRICS_DIR / 'model_a_full_metrics.json')
    print('Saved analysis CSVs in:', REPORT_ANALYSIS_DIR)
    print('Saved charts in:', CHART_DIR)
    print('Overall metrics:', metrics)


if __name__ == '__main__':
    main()
