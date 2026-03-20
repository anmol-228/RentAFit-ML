from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from shared.gender_utils import gender_conflict_flag, gender_from_category


DECISION_LABEL_MAP = {
    'Approve': 0,
    'Review': 1,
    'Reject': 2,
}

UNISEX_EXPLICIT_VARIANTS = ('Women', 'Men')
WOMEN_SPECIFIC_CONFLICT_VARIANTS = ('Men',)


def _apply_gender_policy(row: pd.Series) -> pd.Series:
    out = row.copy()
    out['gender_conflict_flag'] = int(gender_conflict_flag(out.get('gender'), out.get('category')))
    out['gender_policy_override_applied'] = 0
    out['gender_policy_reason'] = 'allowed_gender_selection'

    effective_decision = str(out.get('effective_listing_decision')).strip()
    if out['gender_conflict_flag'] == 1 and effective_decision != 'Reject':
        out['effective_listing_decision'] = 'Review'
        out['effective_decision_label'] = DECISION_LABEL_MAP['Review']
        out['gender_policy_override_applied'] = 1
        out['gender_policy_reason'] = 'gender_category_conflict_review'

    if out['gender_conflict_flag'] == 1 and effective_decision == 'Reject':
        out['gender_policy_reason'] = 'gender_conflict_but_reject_already_applied'

    return out


def expand_gender_variants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows = []
    for _, base_row in df.iterrows():
        parent_id = str(base_row['listing_id'])
        category = str(base_row['category']).strip()
        category_gender = gender_from_category(category)

        base = base_row.copy()
        base['listing_parent_id'] = parent_id
        base['gender_variant_type'] = 'base'
        if pd.isna(base.get('gender_source')) or str(base.get('gender_source')).strip() == '':
            base['gender_source'] = 'category_derived'
        rows.append(_apply_gender_policy(base))

        if category_gender == 'Unisex':
            for explicit_gender in UNISEX_EXPLICIT_VARIANTS:
                variant = base_row.copy()
                variant['listing_id'] = f'{parent_id}__G_{explicit_gender.upper()}'
                variant['listing_parent_id'] = parent_id
                variant['gender'] = explicit_gender
                variant['gender_source'] = 'user_selected'
                variant['gender_variant_type'] = f'explicit_{explicit_gender.lower()}_allowed'
                rows.append(_apply_gender_policy(variant))

        elif category_gender == 'Women':
            for explicit_gender in WOMEN_SPECIFIC_CONFLICT_VARIANTS:
                variant = base_row.copy()
                variant['listing_id'] = f'{parent_id}__G_{explicit_gender.upper()}'
                variant['listing_parent_id'] = parent_id
                variant['gender'] = explicit_gender
                variant['gender_source'] = 'user_selected'
                variant['gender_variant_type'] = f'explicit_{explicit_gender.lower()}_conflict'
                rows.append(_apply_gender_policy(variant))

    out = pd.DataFrame(rows)
    out['listing_id'] = out['listing_id'].astype(str)
    out['listing_parent_id'] = out['listing_parent_id'].astype(str)
    out['gender_policy_override_applied'] = out['gender_policy_override_applied'].astype(int)
    out['gender_conflict_flag'] = out['gender_conflict_flag'].astype(int)
    out['effective_decision_label'] = out['effective_decision_label'].astype(int)
    return out


def build_gender_variant_summary(df: pd.DataFrame) -> dict:
    return {
        'rows_total': int(len(df)),
        'gender_counts': df['gender'].value_counts().to_dict() if 'gender' in df.columns else {},
        'gender_source_counts': df['gender_source'].value_counts().to_dict() if 'gender_source' in df.columns else {},
        'gender_variant_type_counts': df['gender_variant_type'].value_counts().to_dict() if 'gender_variant_type' in df.columns else {},
        'gender_conflict_flag_counts': df['gender_conflict_flag'].value_counts().sort_index().to_dict() if 'gender_conflict_flag' in df.columns else {},
        'gender_policy_override_applied': int(df['gender_policy_override_applied'].sum()) if 'gender_policy_override_applied' in df.columns else 0,
        'effective_decision_counts': df['effective_listing_decision'].value_counts().to_dict() if 'effective_listing_decision' in df.columns else {},
    }
