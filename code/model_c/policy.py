from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
ROOT_CODE_DIR = REPO_ROOT / 'code'
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from shared.gender_utils import normalize_gender


SIZE_ORDER = {
    'XS': 0,
    'S': 1,
    'M': 2,
    'L': 3,
    'XL': 4,
    'XXL': 5,
}

PRIMARY_POOL_STATUS = 'PRIMARY_APPROVE'
REVIEW_FALLBACK_STATUS = 'REVIEW_FALLBACK'
FILTERED_OUT_STATUS = 'FILTERED_OUT'
MAX_REVIEW_FILL_ITEMS = 2


def normalize_size(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text


def size_distance(query_size: Optional[str], candidate_size: Optional[str]) -> int:
    q = SIZE_ORDER.get(normalize_size(query_size))
    c = SIZE_ORDER.get(normalize_size(candidate_size))
    if q is None or c is None:
        return 99
    return abs(q - c)


def size_match_label(distance: int) -> str:
    if distance == 0:
        return 'exact_size_match'
    if distance == 1:
        return 'nearest_size_match'
    if distance == 99:
        return 'unknown_size_match'
    return 'extended_size_match'


def size_match_score(distance: int) -> float:
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.8
    if distance == 2:
        return 0.55
    if distance == 99:
        return 0.20
    return 0.35


def gender_compatible(query_gender: Optional[str], candidate_gender: Optional[str]) -> bool:
    q = normalize_gender(query_gender) or 'Unisex'
    c = normalize_gender(candidate_gender) or 'Unisex'
    if q == 'Women':
        return c in {'Women', 'Unisex'}
    if q == 'Men':
        return c in {'Men', 'Unisex'}
    return c == 'Unisex'


def gender_match_label(query_gender: Optional[str], candidate_gender: Optional[str]) -> str:
    q = normalize_gender(query_gender) or 'Unisex'
    c = normalize_gender(candidate_gender) or 'Unisex'
    if q == c:
        return 'exact_gender_match'
    if c == 'Unisex':
        return 'unisex_gender_match'
    return 'cross_gender_match'


def derive_budget_reference(
    candidates: pd.DataFrame,
    category: str,
    explicit_budget: Optional[float] = None,
) -> tuple[float, str]:
    if explicit_budget is not None and float(explicit_budget) > 0:
        return float(explicit_budget), 'explicit_user_budget'

    category_rows = candidates[candidates['category'] == category]
    if not category_rows.empty and 'category_avg_provider_price' in category_rows.columns:
        values = category_rows['category_avg_provider_price'].dropna()
        if not values.empty and float(values.iloc[0]) > 0:
            return float(values.iloc[0]), 'category_average_budget'

    if not category_rows.empty:
        return float(category_rows['provider_price'].mean()), 'category_average_budget_fallback'

    return 0.0, 'missing_budget_reference'


def budget_alignment_score(series: pd.Series, reference_budget: float) -> pd.Series:
    if reference_budget is None or float(reference_budget) <= 0:
        return pd.Series(0.0, index=series.index)
    ref = max(float(reference_budget), 1.0)
    return 1.0 - ((series - ref).abs() / ref).clip(lower=0.0, upper=1.0)


def within_budget_band(series: pd.Series, reference_budget: float, explicit_budget: Optional[float] = None) -> pd.Series:
    if explicit_budget is not None and float(explicit_budget) > 0:
        return series <= float(explicit_budget)

    if reference_budget is None or float(reference_budget) <= 0:
        return pd.Series(True, index=series.index)

    band = max(150.0, 0.45 * float(reference_budget))
    return (series - float(reference_budget)).abs() <= band


def safety_score(pool_status: str) -> float:
    if pool_status == PRIMARY_POOL_STATUS:
        return 1.0
    if pool_status == REVIEW_FALLBACK_STATUS:
        return 0.60
    return 0.0


def recommendation_pool_status_from_moderation(
    predicted_decision: str,
    rule_quality_score: float,
    deviation_m: float,
    provider_price: float,
    original_price: float,
) -> str:
    if provider_price <= 0 or original_price <= 0:
        return FILTERED_OUT_STATUS

    if str(predicted_decision).strip() == 'Approve' and float(rule_quality_score) >= 55:
        return PRIMARY_POOL_STATUS

    if (
        str(predicted_decision).strip() == 'Review'
        and float(rule_quality_score) >= 60
        and float(deviation_m) <= 40
    ):
        return REVIEW_FALLBACK_STATUS

    return FILTERED_OUT_STATUS
