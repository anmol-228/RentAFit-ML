from pathlib import Path
import argparse
import pandas as pd

BASE = Path('/Users/mypc/RentAFit')
ROOT_CODE_DIR = BASE / 'code'
import sys
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from shared.gender_utils import derive_gender, gender_conflict_flag, resolve_gender_fields

FROZEN_DIR = BASE / 'data/frozen/v1_final'
GENERATED_DIR = BASE / 'data/generated'

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


def build_dataset(pricing_path: Path, gold_expanded_path: Path) -> pd.DataFrame:
    pricing = pd.read_csv(pricing_path)
    gold = pd.read_csv(gold_expanded_path)

    gold_small = gold[[
        'listing_id',
        'final_quality_score_0_100',
        'final_listing_decision',
        'final_reviewer_note',
    ]].copy()

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
        'listing_id', 'source_listing_id', 'data_source',
        'brand', 'category', 'gender', 'gender_source', 'gender_conflict_flag', 'material', 'size', 'condition',
        'condition_token', 'age_months', 'age_bin_token',
        'older_listing_flag', 'listing_lifecycle_action', 'age_policy_override_applied',
        'original_price', 'provider_price', 'tier_primary',
        'deviation_M', 'condition_penalty', 'age_penalty', 'deviation_penalty', 'total_penalty',
        'final_quality_score_0_100', 'final_listing_decision', 'decision_label',
        'effective_listing_decision', 'effective_decision_label', 'final_reviewer_note'
    ]
    out = merged[final_cols].copy()

    assert out['listing_id'].nunique() == len(out)
    assert out['condition_token'].notna().all()
    assert out['age_bin_token'].notna().all()
    assert out['effective_decision_label'].notna().all()
    return out


def main():
    parser = argparse.ArgumentParser(description='Build expanded Model B dataset with shared gender fields.')
    parser.add_argument(
        '--pricing-path',
        default=str(FROZEN_DIR / 'pricing_features_augmented_1500.csv'),
        help='Augmented pricing features CSV.',
    )
    parser.add_argument(
        '--gold-expanded-path',
        default=str(GENERATED_DIR / 'gold_labels_model_b_expanded.csv'),
        help='Expanded human-reviewed Model B gold labels CSV.',
    )
    parser.add_argument(
        '--out-path',
        default=str(GENERATED_DIR / 'model_b_train_expanded.csv'),
        help='Output expanded Model B dataset CSV.',
    )
    args = parser.parse_args()

    df = build_dataset(Path(args.pricing_path), Path(args.gold_expanded_path))
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False)
    print('Saved:', args.out_path)
    print('rows:', len(df))
    print('effective_decision_counts:', df['effective_listing_decision'].value_counts().to_dict())
    print('older_listing_counts:', df['older_listing_flag'].value_counts().sort_index().to_dict())
    print('gender_counts:', df['gender'].value_counts().to_dict())
    print('gender_conflict_flag_counts:', df['gender_conflict_flag'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
