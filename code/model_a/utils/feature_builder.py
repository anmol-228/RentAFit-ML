import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Reuse canonical rule maps from your project.
import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from pricing_rules import BASE_MIN, BASE_MAX, tier_primary, cond_mult, age_mult, cat_mult, mat_mult

DEFAULT_ORIGINAL_PRICE_BY_TIER = {
    'Tier 1': 900,
    'Tier 2': 1800,
    'Tier 3': 3500,
    'Tier 4': 9000,
    'Tier 5': 30000,
}


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        if isinstance(value, str):
            value = value.replace(',', '').strip()
            if value == '':
                return None
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def load_brand_master(path: str = '/Users/mypc/RentAFit/data/frozen/v1_final/brand_tier_master_project_final.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {'brand', 'tier', 'avg_price_min', 'avg_price_max'}
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f'brand master missing columns: {missing}')

    df['brand_norm'] = df['brand'].astype(str).str.strip().str.lower()
    df['avg_price_min_num'] = df['avg_price_min'].apply(_to_float)
    df['avg_price_max_num'] = df['avg_price_max'].apply(_to_float)
    df['is_open_ended_brand_price'] = ((df['avg_price_min_num'].notna()) & (df['avg_price_max_num'].isna())).astype(int)
    return df


def estimate_original_price_from_brand(brand: str, brand_df: pd.DataFrame) -> Tuple[int, str, str]:
    brand_norm = str(brand).strip().lower()
    row = brand_df.loc[brand_df['brand_norm'] == brand_norm]

    if row.empty:
        # Unknown brand fallback.
        est = 3000
        return int(est), 'Tier 3', 'brand_not_found_default_tier3'

    r = row.iloc[0]
    tier = str(r['tier']).strip() if pd.notna(r['tier']) else 'Tier 3'
    if tier not in DEFAULT_ORIGINAL_PRICE_BY_TIER:
        tier = 'Tier 3'

    # Use normalized numeric columns prepared during load.
    pmin = _to_float(r.get('avg_price_min_num'))
    pmax = _to_float(r.get('avg_price_max_num'))

    if pmin and pmax and pmax >= pmin:
        est = (pmin + pmax) / 2.0
        reason = 'brand_avg_midpoint'
    elif pmin and not pmax:
        if tier == 'Tier 5':
            est = pmin * 1.45
        elif tier == 'Tier 4':
            est = pmin * 1.30
        else:
            est = pmin * 1.20
        reason = 'brand_min_scaled_due_to_missing_max'
    elif pmax and not pmin:
        est = pmax * 0.80
        reason = 'brand_max_scaled_due_to_missing_min'
    else:
        est = DEFAULT_ORIGINAL_PRICE_BY_TIER[tier]
        reason = 'tier_default_price'

    est = max(120, int(round(est)))
    return est, tier, reason


def get_brand_features(brand: str, brand_df: pd.DataFrame) -> Dict:
    brand_norm = str(brand).strip().lower()
    row = brand_df.loc[brand_df['brand_norm'] == brand_norm]
    if row.empty:
        return {
            'brand_avg_price_min': 0.0,
            'brand_avg_price_max': 0.0,
            'is_open_ended_brand_price': 0,
            'brand_known_in_master': 0,
        }

    r = row.iloc[0]
    pmin = _to_float(r.get('avg_price_min_num'))
    pmax = _to_float(r.get('avg_price_max_num'))
    open_ended = int(r.get('is_open_ended_brand_price', 0)) if pd.notna(r.get('is_open_ended_brand_price', 0)) else 0

    return {
        'brand_avg_price_min': float(pmin) if pmin is not None else 0.0,
        'brand_avg_price_max': float(pmax) if pmax is not None else 0.0,
        'is_open_ended_brand_price': int(open_ended),
        'brand_known_in_master': 1,
    }


def build_model_a_features(
    brand: str,
    category: str,
    material: str,
    age_months: int,
    size: str,
    condition: str,
    original_price: Optional[float],
    brand_df: pd.DataFrame,
) -> Dict:
    # Use provided original price if available; else estimate from brand master.
    if original_price is not None:
        op = max(120, int(round(float(original_price))))
        est_tier_from_brand = None
        original_price_source = 'provided_by_user'
        original_price_reason = 'direct_input'
    else:
        op, est_tier_from_brand, original_price_reason = estimate_original_price_from_brand(brand, brand_df)
        original_price_source = 'estimated_from_brand_master'

    # Manual-v1 policy: tier_primary derived from original_price (price-tier fallback).
    tier = tier_primary(op)
    brand_feats = get_brand_features(brand, brand_df)

    base_min_pct = float(BASE_MIN[tier])
    base_max_pct = float(BASE_MAX[tier])
    c_mult = float(cond_mult(condition))
    a_mult = float(age_mult(int(age_months)))
    cat_multiplier = float(cat_mult(category))
    mat_multiplier = float(mat_mult(material))

    return {
        'brand': brand,
        'category': category,
        'material': material,
        'size': size,
        'condition': condition,
        'age_months': int(age_months),
        'original_price': int(op),
        'tier_primary': tier,
        'base_min_pct': base_min_pct,
        'base_max_pct': base_max_pct,
        'cond_mult': c_mult,
        'age_mult': a_mult,
        'cat_mult': cat_multiplier,
        'mat_mult': mat_multiplier,
        'brand_avg_price_min': brand_feats['brand_avg_price_min'],
        'brand_avg_price_max': brand_feats['brand_avg_price_max'],
        'is_open_ended_brand_price': brand_feats['is_open_ended_brand_price'],
        'brand_known_in_master': brand_feats['brand_known_in_master'],
        'original_price_source': original_price_source,
        'original_price_reason': original_price_reason,
        'brand_master_tier': est_tier_from_brand,
    }


def bucket_round(v: float) -> int:
    if v < 200:
        return round(v / 10) * 10
    if v <= 1000:
        return round(v / 50) * 50
    return round(v / 100) * 100


def postprocess_range_from_pct(pred_min_pct: float, pred_max_pct: float, original_price: float) -> Tuple[int, int]:
    a = max(0.0, min(float(pred_min_pct), 0.20))
    b = max(0.0, min(float(pred_max_pct), 0.20))

    pred_min = a * original_price
    pred_max = b * original_price

    if pred_min > pred_max:
        pred_min, pred_max = pred_max, pred_min

    pred_min = bucket_round(pred_min)
    pred_max = bucket_round(pred_max)

    if pred_max < pred_min:
        pred_max = pred_min

    return int(pred_min), int(pred_max)


def compute_rule_range_from_features(features: Dict) -> Tuple[int, int]:
    op = float(features['original_price'])
    raw_min = (
        op
        * float(features['base_min_pct'])
        * float(features['cond_mult'])
        * float(features['age_mult'])
        * float(features['cat_mult'])
        * float(features['mat_mult'])
    )
    raw_max = (
        op
        * float(features['base_max_pct'])
        * float(features['cond_mult'])
        * float(features['age_mult'])
        * float(features['cat_mult'])
        * float(features['mat_mult'])
    )

    cap_min = min(raw_min, 0.20 * op, op)
    cap_max = min(raw_max, 0.20 * op, op)

    rule_min = bucket_round(cap_min)
    rule_max = bucket_round(cap_max)
    if rule_max < rule_min:
        rule_max = rule_min
    return int(rule_min), int(rule_max)
