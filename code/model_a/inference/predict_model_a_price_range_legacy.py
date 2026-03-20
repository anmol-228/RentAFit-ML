import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_a.utils.feature_builder import build_model_a_features, load_brand_master, postprocess_range_from_pct

MIN_MODEL = Path('/Users/mypc/RentAFit/models/model_a/rf_pct/model_a_min_rf_pct.pkl')
MAX_MODEL = Path('/Users/mypc/RentAFit/models/model_a/rf_pct/model_a_max_rf_pct.pkl')
BRAND_MASTER_PATH = Path('/Users/mypc/RentAFit/data/frozen/v1_final/brand_tier_master_project_final.csv')

FEATURE_COLS = [
    'brand', 'category', 'material', 'size', 'condition', 'tier_primary',
    'age_months', 'original_price',
    'base_min_pct', 'base_max_pct', 'cond_mult', 'age_mult', 'cat_mult', 'mat_mult',
]

CONDITION_CANONICAL = {
    'new': 'New',
    'like new': 'Like New',
    'likenew': 'Like New',
    'used': 'Used',
}


def normalize_condition(value: str) -> str:
    key = str(value).strip().lower()
    return CONDITION_CANONICAL.get(key, value)


def predict_from_minimal_input(
    brand: str,
    category: str,
    material: str,
    size: str,
    condition: str,
    age_months: int,
    original_price=None,
):
    if not MIN_MODEL.exists() or not MAX_MODEL.exists():
        raise FileNotFoundError(
            f'Model files not found. Expected:\n- {MIN_MODEL}\n- {MAX_MODEL}'
        )

    brand_df = load_brand_master(str(BRAND_MASTER_PATH))
    condition = normalize_condition(condition)

    features = build_model_a_features(
        brand=brand,
        category=category,
        material=material,
        age_months=int(age_months),
        size=size,
        condition=condition,
        original_price=original_price,
        brand_df=brand_df,
    )

    X = pd.DataFrame([features])[FEATURE_COLS]

    model_min = joblib.load(MIN_MODEL)
    model_max = joblib.load(MAX_MODEL)

    pred_min_pct = float(model_min.predict(X)[0])
    pred_max_pct = float(model_max.predict(X)[0])

    pred_min, pred_max = postprocess_range_from_pct(
        pred_min_pct,
        pred_max_pct,
        float(features['original_price']),
    )

    return {
        'input': {
            'brand': brand,
            'category': category,
            'material': material,
            'size': size,
            'condition': condition,
            'age_months': int(age_months),
            'original_price': int(features['original_price']),
            'original_price_source': features['original_price_source'],
            'original_price_reason': features['original_price_reason'],
        },
        'derived': {
            'tier_primary': features['tier_primary'],
            'base_min_pct': features['base_min_pct'],
            'base_max_pct': features['base_max_pct'],
            'cond_mult': features['cond_mult'],
            'age_mult': features['age_mult'],
            'cat_mult': features['cat_mult'],
            'mat_mult': features['mat_mult'],
        },
        'raw_model_output_pct': {
            'pred_min_pct': pred_min_pct,
            'pred_max_pct': pred_max_pct,
        },
        'final_price_range': {
            'min_price': int(pred_min),
            'max_price': int(pred_max),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='Legacy Model A predictor with minimal user input. Derived fields are computed internally.'
    )
    parser.add_argument('--brand', required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--material', required=True)
    parser.add_argument('--size', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--age_months', required=True, type=int)
    parser.add_argument('--original_price', required=False, type=float)
    parser.add_argument('--json', action='store_true')

    args = parser.parse_args()

    out = predict_from_minimal_input(
        brand=args.brand,
        category=args.category,
        material=args.material,
        size=args.size,
        condition=args.condition,
        age_months=args.age_months,
        original_price=args.original_price,
    )

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print('Predicted Rental Price Range')
    print('Input:', out['input'])
    print('Derived:', out['derived'])
    print('Raw pct:', out['raw_model_output_pct'])
    print('Final range:', out['final_price_range'])


if __name__ == '__main__':
    main()
