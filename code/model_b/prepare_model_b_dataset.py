from pathlib import Path
import argparse
import pandas as pd

import sys
REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
BASE = REPO_ROOT
ROOT_CODE_DIR = BASE / 'code'
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from shared.gender_utils import derive_gender, gender_conflict_flag, resolve_gender_fields

DATA_DIR = BASE / 'data/frozen/v1_final'

CONDITION_TOKEN_MAP = {
    'New': 1,
    'Like New': 2,
    'Used': 3,
}

DECISION_LABEL_MAP = {
    'Approve': 0,
    'Review': 1,
    'Reject': 2,
}

STALE_AGE_THRESHOLD_MONTHS = 10


def age_bin_token(age_months: float) -> int:
    age = float(age_months)
    if age <= 3:
        return 1
    if age <= 6:
        return 2
    if age <= 9:
        return 3
    return 4


def older_listing_flag(age_months: float) -> int:
    return int(float(age_months) >= STALE_AGE_THRESHOLD_MONTHS)


def listing_lifecycle_action(age_months: float) -> str:
    if float(age_months) >= STALE_AGE_THRESHOLD_MONTHS:
        return 'needs_reapproval_or_removal'
    return 'active_current_cycle'


def effective_listing_decision(final_decision: str, age_months: float) -> str:
    if float(age_months) >= STALE_AGE_THRESHOLD_MONTHS and final_decision == 'Approve':
        return 'Review'
    return final_decision


def build_model_b_dataset(pricing_path: Path, gold_path: Path) -> pd.DataFrame:
    pricing = pd.read_csv(pricing_path)
    gold = pd.read_csv(gold_path)

    gold_small = gold[[
        'listing_id',
        'final_quality_score_0_100',
        'final_listing_decision',
        'final_reviewer_note',
    ]]

    merged = pricing.merge(gold_small, on='listing_id', how='inner', validate='one_to_one')

    gender_fields = merged.apply(
        lambda row: resolve_gender_fields(
            gender=row.get('gender'),
            category=row.get('category'),
            gender_source=row.get('gender_source'),
        ) if pd.notna(row.get('gender')) else derive_gender(explicit_gender=row.get('gender'), category=row.get('category')),
        axis=1,
        result_type='expand',
    )
    gender_fields.columns = ['gender', 'gender_source']
    merged['gender'] = gender_fields['gender']
    merged['gender_source'] = gender_fields['gender_source']
    merged['gender_conflict_flag'] = merged.apply(
        lambda row: gender_conflict_flag(row['gender'], row['category']),
        axis=1,
    )

    merged['condition_token'] = merged['condition'].map(CONDITION_TOKEN_MAP)
    merged['age_bin_token'] = merged['age_months'].apply(age_bin_token)
    merged['older_listing_flag'] = merged['age_months'].apply(older_listing_flag)
    merged['listing_lifecycle_action'] = merged['age_months'].apply(listing_lifecycle_action)
    merged['age_policy_override_applied'] = ((merged['older_listing_flag'] == 1) & (merged['final_listing_decision'] == 'Approve')).astype(int)
    merged['effective_listing_decision'] = merged.apply(
        lambda row: effective_listing_decision(row['final_listing_decision'], row['age_months']),
        axis=1,
    )
    merged['decision_label'] = merged['final_listing_decision'].map(DECISION_LABEL_MAP)
    merged['effective_decision_label'] = merged['effective_listing_decision'].map(DECISION_LABEL_MAP)

    final_cols = [
        'listing_id',
        'brand',
        'category',
        'gender',
        'gender_source',
        'gender_conflict_flag',
        'material',
        'size',
        'condition',
        'condition_token',
        'age_months',
        'age_bin_token',
        'older_listing_flag',
        'listing_lifecycle_action',
        'age_policy_override_applied',
        'original_price',
        'provider_price',
        'tier_primary',
        'deviation_M',
        'condition_penalty',
        'age_penalty',
        'deviation_penalty',
        'total_penalty',
        'final_quality_score_0_100',
        'final_listing_decision',
        'decision_label',
        'effective_listing_decision',
        'effective_decision_label',
        'final_reviewer_note',
    ]

    model_b = merged[final_cols].copy()

    assert len(model_b) == 375
    assert model_b['listing_id'].nunique() == 375
    assert model_b['condition_token'].notna().all()
    assert model_b['age_bin_token'].notna().all()
    assert model_b['decision_label'].notna().all()
    assert model_b['effective_decision_label'].notna().all()

    return model_b


def main():
    parser = argparse.ArgumentParser(description='Build Model B dataset with shared gender fields.')
    parser.add_argument(
        '--pricing-path',
        default=str(DATA_DIR / 'pricing_features.csv'),
        help='Pricing features CSV.',
    )
    parser.add_argument(
        '--gold-path',
        default=str(DATA_DIR / 'gold_labels_model_b_full_manual.csv'),
        help='Manual gold labels CSV.',
    )
    parser.add_argument(
        '--out-path',
        default=str(DATA_DIR / 'model_b_train_manual.csv'),
        help='Output Model B training CSV.',
    )
    args = parser.parse_args()

    model_b = build_model_b_dataset(Path(args.pricing_path), Path(args.gold_path))
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    model_b.to_csv(args.out_path, index=False)

    print('Saved:', args.out_path)
    print('rows:', len(model_b))
    print('final_decision_counts:', model_b['final_listing_decision'].value_counts().to_dict())
    print('effective_decision_counts:', model_b['effective_listing_decision'].value_counts().to_dict())
    print('older_listing_flag_counts:', model_b['older_listing_flag'].value_counts().sort_index().to_dict())
    print('age_policy_override_applied:', int(model_b['age_policy_override_applied'].sum()))
    print('gender_counts:', model_b['gender'].value_counts().to_dict())
    print('gender_conflict_flag_counts:', model_b['gender_conflict_flag'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
