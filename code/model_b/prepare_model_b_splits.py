from pathlib import Path
import argparse
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
ROOT_CODE_DIR = REPO_ROOT / 'code'
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_b.gender_policy import build_gender_variant_summary, expand_gender_variants

BASE = REPO_ROOT
GENERATED_DIR = BASE / 'data/generated'
REPORT_DIR = BASE / 'reports/model_b'

INPUT_PATH = GENERATED_DIR / 'model_b_train_expanded_gender.csv'
OUTPUT_PATH = GENERATED_DIR / 'model_b_train_expanded_gender_ready.csv'
SUMMARY_PATH = REPORT_DIR / 'model_b_split_summary.json'

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def build_split_summary(df: pd.DataFrame, base_rows_total: int) -> dict:
    summary = {
        'base_rows_total_before_gender_expansion': int(base_rows_total),
        'rows_total': int(len(df)),
        'random_state': RANDOM_STATE,
        'fractions': {'train': TRAIN_FRACTION, 'val': VAL_FRACTION, 'test': TEST_FRACTION},
        'split_counts': df['split_set'].value_counts().sort_index().to_dict(),
        'effective_decision_overall': df['effective_listing_decision'].value_counts().sort_index().to_dict(),
        'gender_counts': df['gender'].value_counts().sort_index().to_dict(),
        'gender_variant_type_counts': df['gender_variant_type'].value_counts().sort_index().to_dict(),
        'gender_conflict_flag_counts': df['gender_conflict_flag'].value_counts().sort_index().to_dict(),
        'gender_policy_override_applied': int(df['gender_policy_override_applied'].sum()),
        'per_split_effective_decision': {},
        'per_split_gender_variant_summary': {},
    }
    for split in ['train', 'val', 'test']:
        part = df[df['split_set'] == split]
        summary['per_split_effective_decision'][split] = part['effective_listing_decision'].value_counts().sort_index().to_dict()
        summary['per_split_gender_variant_summary'][split] = build_gender_variant_summary(part)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Create Model B train/val/test splits and expand gender variants inside each split.')
    parser.add_argument('--input-path', default=str(INPUT_PATH), help='Base expanded Model B dataset path.')
    parser.add_argument('--out-path', default=str(OUTPUT_PATH), help='Output split-ready gender-aware Model B dataset.')
    parser.add_argument('--summary-path', default=str(SUMMARY_PATH), help='Output JSON split summary.')
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    assert df['listing_id'].nunique() == len(df)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_FRACTION),
        random_state=RANDOM_STATE,
        stratify=df['effective_decision_label'],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION)),
        random_state=RANDOM_STATE,
        stratify=temp_df['effective_decision_label'],
    )

    train_df = train_df.copy(); train_df['split_set'] = 'train'
    val_df = val_df.copy(); val_df['split_set'] = 'val'
    test_df = test_df.copy(); test_df['split_set'] = 'test'

    train_aug = expand_gender_variants(train_df)
    train_aug['split_set'] = 'train'
    val_aug = expand_gender_variants(val_df)
    val_aug['split_set'] = 'val'
    test_aug = expand_gender_variants(test_df)
    test_aug['split_set'] = 'test'

    out = pd.concat([train_aug, val_aug, test_aug], ignore_index=True)
    out = out.sort_values('listing_id').reset_index(drop=True)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    summary = build_split_summary(out, base_rows_total=len(df))
    with open(args.summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Saved:', args.out_path)
    print('Saved summary:', args.summary_path)
    print('split_counts:', summary['split_counts'])
    print('per_split_effective_decision:', summary['per_split_effective_decision'])
    print('gender_counts:', summary['gender_counts'])
    print('gender_policy_override_applied:', summary['gender_policy_override_applied'])


if __name__ == '__main__':
    main()
