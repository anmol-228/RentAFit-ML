import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import sys
REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
ROOT_CODE_DIR = REPO_ROOT / 'code'
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_a.utils.feature_builder import (
    build_model_a_features,
    compute_rule_range_from_features,
    load_brand_master,
    postprocess_range_from_pct,
)

BASE_DIR = REPO_ROOT
BRAND_MASTER_PATH = BASE_DIR / 'data/frozen/v1_final/brand_tier_master_project_final.csv'

# Primary (best) model set: tier-split % models.
TIER_SPLIT_MODEL_DIR = BASE_DIR / 'models/model_a/rf_pct_tier_split'
FALLBACK_MODEL_DIR = BASE_DIR / 'models/model_a/rf_pct'

FEATURE_COLS = [
    'brand', 'category', 'material', 'size', 'condition', 'tier_primary',
    'age_months', 'original_price',
    'base_min_pct', 'base_max_pct', 'cond_mult', 'age_mult', 'cat_mult', 'mat_mult',
    'brand_avg_price_min', 'brand_avg_price_max', 'is_open_ended_brand_price', 'brand_known_in_master',
]

RESIDUAL_FEATURE_COLS = FEATURE_COLS + ['base_pred_max_pct']

CONDITION_CANONICAL = {
    'new': 'New',
    'like new': 'Like New',
    'likenew': 'Like New',
    'used': 'Used',
}


def load_models():
    tier_split_paths = {
        'min_non_t5': TIER_SPLIT_MODEL_DIR / 'model_a_min_rf_pct_tier1to4.pkl',
        'max_non_t5': TIER_SPLIT_MODEL_DIR / 'model_a_max_rf_pct_tier1to4.pkl',
        'min_t5': TIER_SPLIT_MODEL_DIR / 'model_a_min_rf_pct_tier5.pkl',
        'max_t5': TIER_SPLIT_MODEL_DIR / 'model_a_max_rf_pct_tier5.pkl',
    }
    residual_path = TIER_SPLIT_MODEL_DIR / 'model_a_max_rf_pct_tier5_residual.pkl'
    metadata_path = TIER_SPLIT_MODEL_DIR / 'model_a_rf_pct_tier_split_metadata.json'

    if all(p.exists() for p in tier_split_paths.values()):
        models = {k: joblib.load(v) for k, v in tier_split_paths.items()}
        if residual_path.exists():
            models['max_t5_residual'] = joblib.load(residual_path)
        meta = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        return {
            'mode': 'tier_split',
            'models': models,
            'metadata': meta,
        }

    fallback_paths = {
        'min': FALLBACK_MODEL_DIR / 'model_a_min_rf_pct.pkl',
        'max': FALLBACK_MODEL_DIR / 'model_a_max_rf_pct.pkl',
    }
    if all(p.exists() for p in fallback_paths.values()):
        return {
            'mode': 'single',
            'models': {k: joblib.load(v) for k, v in fallback_paths.items()},
            'metadata': {},
        }

    raise FileNotFoundError(
        'No suitable model files found. Train tier-split models first, or ensure rf_pct fallback models exist.'
    )


def normalize_condition(value: str) -> str:
    key = str(value).strip().lower()
    return CONDITION_CANONICAL.get(key, value)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def compute_confidence(features: dict, model_route: str, brand: str, pred_min_pct: float, pred_max_pct: float, metadata: dict):
    score = 1.0
    reasons = []

    conf = (metadata or {}).get('confidence', {}) if isinstance(metadata, dict) else {}
    threshold = float(conf.get('confidence_threshold', 0.55))

    brand_norm = str(brand).strip().lower()
    if model_route == 'tier_split_tier5':
        seen = set((metadata or {}).get('seen_brands_tier_split_tier5', []))
    else:
        seen = set((metadata or {}).get('seen_brands_tier_split_tier1to4', []))

    if brand_norm not in seen:
        score -= 0.35
        reasons.append('brand_unseen_in_route_training')

    if str(features.get('original_price_source')) != 'provided_by_user':
        score -= 0.35
        reasons.append('original_price_not_user_provided')

    if int(features.get('is_open_ended_brand_price', 0)) == 1:
        score -= 0.10
        reasons.append('open_ended_brand_price_band')

    if int(features.get('brand_known_in_master', 1)) == 0:
        score -= 0.20
        reasons.append('brand_missing_in_brand_master')

    raw_width_pct = abs(float(pred_max_pct) - float(pred_min_pct))
    route_p90 = (conf.get('route_p90_pct_error_max') or {}).get(model_route)
    if route_p90 is not None:
        if raw_width_pct > max(0.10, float(route_p90) * 2.0):
            score -= 0.15
            reasons.append('predicted_range_width_unusually_high')

    score = _clamp01(score)
    fallback = score < threshold

    return {
        'score': score,
        'threshold': threshold,
        'fallback_to_rule_range': bool(fallback),
        'reasons': reasons,
    }


def predict_one(
    brand: str,
    category: str,
    material: str,
    age_months: int,
    size: str,
    condition: str,
    original_price: float,
):
    brand_df = load_brand_master(str(BRAND_MASTER_PATH))
    loaded = load_models()
    condition_norm = normalize_condition(condition)

    features = build_model_a_features(
        brand=brand,
        category=category,
        material=material,
        age_months=int(age_months),
        size=size,
        condition=condition_norm,
        original_price=float(original_price),
        brand_df=brand_df,
    )

    X = pd.DataFrame([features])[FEATURE_COLS]

    residual_applied = False
    if loaded['mode'] == 'tier_split':
        if features['tier_primary'] == 'Tier 5':
            model_min = loaded['models']['min_t5']
            model_max = loaded['models']['max_t5']
            model_route = 'tier_split_tier5'
        else:
            model_min = loaded['models']['min_non_t5']
            model_max = loaded['models']['max_non_t5']
            model_route = 'tier_split_tier1to4'
    else:
        model_min = loaded['models']['min']
        model_max = loaded['models']['max']
        model_route = 'single_rf_pct_fallback'

    pred_min_pct = float(model_min.predict(X)[0])
    pred_max_pct = float(model_max.predict(X)[0])

    # Tier 5 residual correction for max_pct.
    if loaded['mode'] == 'tier_split' and model_route == 'tier_split_tier5' and 'max_t5_residual' in loaded['models']:
        residual_scale = float((loaded.get('metadata') or {}).get('residual_scale', 1.0))
        x_res = X.copy()
        x_res['base_pred_max_pct'] = pred_max_pct
        pred_max_pct += residual_scale * float(loaded['models']['max_t5_residual'].predict(x_res[RESIDUAL_FEATURE_COLS])[0])
        residual_applied = True

    model_min_price, model_max_price = postprocess_range_from_pct(pred_min_pct, pred_max_pct, float(features['original_price']))
    rule_min_price, rule_max_price = compute_rule_range_from_features(features)

    confidence = compute_confidence(
        features=features,
        model_route=model_route,
        brand=brand,
        pred_min_pct=pred_min_pct,
        pred_max_pct=pred_max_pct,
        metadata=loaded.get('metadata', {}),
    )

    if confidence['fallback_to_rule_range']:
        final_min_price, final_max_price = rule_min_price, rule_max_price
        final_source = 'rule_fallback'
    else:
        final_min_price, final_max_price = model_min_price, model_max_price
        final_source = 'model_output'

    out = {
        'input': {
            'brand': brand,
            'category': category,
            'material': material,
            'age_months': int(age_months),
            'size': size,
            'condition': condition_norm,
            'original_price': int(features['original_price']),
        },
        'derived_features': {
            'tier_primary': features['tier_primary'],
            'base_min_pct': features['base_min_pct'],
            'base_max_pct': features['base_max_pct'],
            'cond_mult': features['cond_mult'],
            'age_mult': features['age_mult'],
            'cat_mult': features['cat_mult'],
            'mat_mult': features['mat_mult'],
            'brand_avg_price_min': features['brand_avg_price_min'],
            'brand_avg_price_max': features['brand_avg_price_max'],
            'is_open_ended_brand_price': features['is_open_ended_brand_price'],
            'brand_known_in_master': features['brand_known_in_master'],
            'original_price_source': features['original_price_source'],
            'original_price_reason': features['original_price_reason'],
            'brand_master_tier': features['brand_master_tier'],
        },
        'raw_model_output_pct': {
            'pred_min_pct': pred_min_pct,
            'pred_max_pct': pred_max_pct,
            'tier5_residual_applied': residual_applied,
        },
        'candidate_ranges': {
            'model_range': {
                'min_price': int(model_min_price),
                'max_price': int(model_max_price),
            },
            'rule_range': {
                'min_price': int(rule_min_price),
                'max_price': int(rule_max_price),
            },
        },
        'confidence': confidence,
        'final_price_range': {
            'min_price': int(final_min_price),
            'max_price': int(final_max_price),
            'source': final_source,
        },
        'model_route': model_route,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description='Predict rental price range from simple listing input.')
    parser.add_argument('--brand', required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--material', required=True)
    parser.add_argument('--age_months', required=True, type=int)
    parser.add_argument('--size', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--original_price', required=True, type=float)
    parser.add_argument('--json', action='store_true', help='Print JSON output only')

    args = parser.parse_args()

    result = predict_one(
        brand=args.brand,
        category=args.category,
        material=args.material,
        age_months=args.age_months,
        size=args.size,
        condition=args.condition,
        original_price=args.original_price,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print('Model A Price Prediction')
    print('------------------------')
    print('Input:', result['input'])
    print('Derived:', result['derived_features'])
    print('Raw pct:', result['raw_model_output_pct'])
    print('Candidate ranges:', result['candidate_ranges'])
    print('Confidence:', result['confidence'])
    print('Final range:', result['final_price_range'])
    print('Model route:', result['model_route'])


if __name__ == '__main__':
    main()
