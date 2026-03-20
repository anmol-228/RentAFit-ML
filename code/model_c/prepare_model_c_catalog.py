from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

import sys
REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
ROOT_CODE_DIR = REPO_ROOT / 'code'
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_b.runtime import load_artifacts as load_model_b_artifacts, predict_one as predict_model_b
from model_c.policy import (
    FILTERED_OUT_STATUS,
    PRIMARY_POOL_STATUS,
    REVIEW_FALLBACK_STATUS,
    recommendation_pool_status_from_moderation,
)
from shared.gender_utils import resolve_gender_fields

BASE = REPO_ROOT
DATA_DIR = BASE / 'data/generated'
REPORT_DIR = BASE / 'reports/model_c'


def score_with_model_b(df: pd.DataFrame) -> pd.DataFrame:
    loaded = load_model_b_artifacts()

    predicted_decision = []
    approve_prob = []
    review_prob = []
    reject_prob = []

    for _, row in df.iterrows():
        result = predict_model_b(
            brand=row['brand'],
            category=row['category'],
            gender=row.get('gender'),
            material=row['material'],
            size=row['size'],
            condition=row['condition'],
            garment_age_months=int(row['age_months']),
            original_price=float(row['original_price']),
            provider_price=float(row['provider_price']),
            current_status='PENDING_REVIEW',
            loaded=loaded,
        )
        probs = result['prediction']['class_probabilities']
        predicted_decision.append(result['prediction']['predicted_decision'])
        approve_prob.append(float(probs['Approve']))
        review_prob.append(float(probs['Review']))
        reject_prob.append(float(probs['Reject']))

    out = df.copy()
    out['model_b_predicted_decision'] = predicted_decision
    out['model_b_approve_probability'] = approve_prob
    out['model_b_review_probability'] = review_prob
    out['model_b_reject_probability'] = reject_prob
    return out


def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    gender_fields = out.apply(
        lambda row: resolve_gender_fields(
            gender=row.get('gender'),
            category=row.get('category'),
            gender_source=row.get('gender_source'),
        ),
        axis=1,
        result_type='expand',
    )
    gender_fields.columns = ['gender', 'gender_source']
    out['gender'] = gender_fields['gender']
    out['gender_source'] = gender_fields['gender_source']

    out['quality_score_norm'] = (out['rule_quality_score'] / 100.0).clip(lower=0.0, upper=1.0)
    out['item_freshness_score'] = 1.0 - (out['age_months'].clip(lower=0, upper=18) / 18.0)
    out['freshness_score'] = out['item_freshness_score']

    provider_pct = np.where(
        out['original_price'] > 0,
        out['provider_price'] / out['original_price'],
        0.0,
    )
    out['provider_price_pct_of_original'] = pd.Series(provider_pct, index=out.index).clip(lower=0.0, upper=1.0)

    out = score_with_model_b(out)

    out['recommendation_pool_status'] = out.apply(
        lambda row: recommendation_pool_status_from_moderation(
            predicted_decision=row['model_b_predicted_decision'],
            rule_quality_score=row['rule_quality_score'],
            deviation_m=row['deviation_M'],
            provider_price=row['provider_price'],
            original_price=row['original_price'],
        ),
        axis=1,
    )
    out['recommendable_flag'] = (out['recommendation_pool_status'] != FILTERED_OUT_STATUS).astype(int)
    out['review_fallback_eligible'] = (out['recommendation_pool_status'] == REVIEW_FALLBACK_STATUS).astype(int)

    recommendable = out[out['recommendable_flag'] == 1].copy()
    category_avg_provider_price = recommendable.groupby('category')['provider_price'].mean().to_dict()
    fallback_category_avg = out.groupby('category')['provider_price'].mean().to_dict()
    out['category_avg_provider_price'] = out['category'].map(category_avg_provider_price).fillna(out['category'].map(fallback_category_avg)).fillna(0.0)
    out['price_vs_category_avg_ratio'] = np.where(
        out['category_avg_provider_price'] > 0,
        out['provider_price'] / out['category_avg_provider_price'],
        1.0,
    )

    pool_priority = out['recommendation_pool_status'].map({
        PRIMARY_POOL_STATUS: 1.0,
        REVIEW_FALLBACK_STATUS: 0.55,
        FILTERED_OUT_STATUS: 0.0,
    }).fillna(0.0)

    out['catalog_priority_score'] = (
        0.40 * out['quality_score_norm']
        + 0.20 * out['item_freshness_score']
        + 0.20 * out['model_b_approve_probability']
        + 0.20 * pool_priority
    ).round(6)

    return out


def main():
    parser = argparse.ArgumentParser(description='Build Model C catalog with gender, moderation, budget, and freshness policy fields.')
    parser.add_argument(
        '--source-path',
        default=str(BASE / 'data/frozen/v2_gender/pricing_features_augmented_1500.csv'),
        help='Source pricing features CSV.',
    )
    parser.add_argument(
        '--catalog-path',
        default=str(DATA_DIR / 'model_c_catalog.csv'),
        help='Output full catalog CSV.',
    )
    parser.add_argument(
        '--recommendable-path',
        default=str(DATA_DIR / 'model_c_catalog_recommendable.csv'),
        help='Output recommendable catalog CSV.',
    )
    parser.add_argument(
        '--summary-path',
        default=str(REPORT_DIR / 'model_c_catalog_summary.json'),
        help='Output JSON summary path.',
    )
    args = parser.parse_args()

    Path(args.catalog_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.recommendable_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.source_path)
    catalog = build_catalog(df)
    recommendable = catalog[catalog['recommendable_flag'] == 1].copy().reset_index(drop=True)

    catalog.to_csv(args.catalog_path, index=False)
    recommendable.to_csv(args.recommendable_path, index=False)

    summary = {
        'source_rows': int(len(df)),
        'catalog_rows': int(len(catalog)),
        'recommendable_rows': int(len(recommendable)),
        'filtered_out_rows': int((catalog['recommendable_flag'] == 0).sum()),
        'recommendable_rate': float((catalog['recommendable_flag'] == 1).mean()),
        'pool_status_counts': catalog['recommendation_pool_status'].value_counts().to_dict(),
        'category_counts_recommendable': recommendable['category'].value_counts().to_dict(),
        'gender_counts_recommendable': recommendable['gender'].value_counts().to_dict(),
        'size_counts_recommendable': recommendable['size'].value_counts().to_dict(),
        'avg_category_budget_recommendable': recommendable.groupby('category')['category_avg_provider_price'].mean().round(2).to_dict(),
    }

    with open(args.summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Saved catalog:', args.catalog_path)
    print('Saved recommendable catalog:', args.recommendable_path)
    print('Saved summary:', args.summary_path)
    print('Recommendable rows:', len(recommendable))
    print('Pool status counts:', summary['pool_status_counts'])


if __name__ == '__main__':
    main()
