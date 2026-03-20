from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_b.runtime import load_artifacts, predict_one

BASE = Path('/Users/mypc/RentAFit')
DEFAULT_INPUT = BASE / 'data/generated/model_b_live_listings_sample.csv'
DEFAULT_SCORED = BASE / 'reports/model_b/model_b_live_listings_scored_sample.csv'
DEFAULT_QUEUE = BASE / 'reports/model_b/model_b_review_queue_sample.csv'
DEFAULT_SUMMARY = BASE / 'reports/model_b/model_b_live_ops_summary.json'


def flatten_result(source_row: pd.Series, result: dict) -> dict:
    probs = result['prediction']['class_probabilities']
    lifecycle = result['lifecycle']
    row = dict(source_row)
    row.update({
        'model_input_age_months': result['derived_features']['model_input_age_months'],
        'listing_age_months': lifecycle['listing_age_months'],
        'listing_age_source': lifecycle['listing_age_source'],
        'listing_age_reference_date': lifecycle['listing_age_reference_date'],
        'as_of_date': lifecycle['as_of_date'],
        'predicted_decision': result['prediction']['predicted_decision'],
        'prob_approve': probs['Approve'],
        'prob_review': probs['Review'],
        'prob_reject': probs['Reject'],
        'recommended_status': lifecycle['next_status'],
        'visible_to_renters': int(lifecycle['visible_to_renters']),
        'review_required': int(lifecycle['review_required']),
        'review_reason': lifecycle['review_reason'],
        'review_priority': lifecycle['review_priority'],
        'assigned_reviewer_role': lifecycle['assigned_reviewer_role'],
        'stale_listing_flag': int(lifecycle['stale_listing_flag']),
        'removal_recommended': int(lifecycle['removal_recommended']),
        'auto_removed': int(lifecycle['auto_removed']),
        'rule_quality_score': result['derived_features']['rule_quality_score'],
        'rule_decision': result['derived_features']['rule_decision'],
        'deviation_M': result['derived_features']['deviation_M'],
        'total_penalty': result['derived_features']['total_penalty'],
    })
    return row


def main():
    parser = argparse.ArgumentParser(description='Run Model B operational scoring on a live listing CSV and generate a review queue.')
    parser.add_argument('--input_csv', default=str(DEFAULT_INPUT))
    parser.add_argument('--output_scored_csv', default=str(DEFAULT_SCORED))
    parser.add_argument('--output_queue_csv', default=str(DEFAULT_QUEUE))
    parser.add_argument('--summary_json', default=str(DEFAULT_SUMMARY))
    parser.add_argument('--as_of_date', required=False)
    parser.add_argument('--auto_remove_stale', action='store_true')
    parser.add_argument('--removal_grace_months', type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    scored_path = Path(args.output_scored_csv)
    queue_path = Path(args.output_queue_csv)
    summary_path = Path(args.summary_json)

    scored_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    artifacts = load_artifacts()

    rows = []
    for _, src in df.iterrows():
        result = predict_one(
            brand=src['brand'],
            category=src['category'],
            material=src['material'],
            size=src['size'],
            condition=src['condition'],
            garment_age_months=src.get('garment_age_months', src.get('age_months')),
            original_price=src['original_price'],
            provider_price=src['provider_price'],
            current_status=src.get('current_status', 'ACTIVE'),
            listing_created_at=src.get('listing_created_at'),
            last_approved_at=src.get('last_approved_at'),
            last_reapproved_at=src.get('last_reapproved_at'),
            as_of_date=args.as_of_date,
            auto_remove_stale=args.auto_remove_stale,
            removal_grace_months=args.removal_grace_months,
            loaded=artifacts,
        )
        rows.append(flatten_result(src, result))

    out_df = pd.DataFrame(rows)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    out_df['_priority_rank'] = out_df['review_priority'].map(priority_order).fillna(9)

    queue_cols = [
        'listing_id', 'provider_id', 'current_status', 'recommended_status',
        'predicted_decision', 'listing_age_months', 'stale_listing_flag',
        'removal_recommended', 'review_reason', 'review_priority',
        'assigned_reviewer_role', 'visible_to_renters', 'deviation_M',
        'total_penalty', 'rule_quality_score'
    ]
    queue_df = out_df[out_df['review_required'] == 1][queue_cols + ['_priority_rank']].sort_values(
        ['_priority_rank', 'listing_age_months', 'listing_id'], ascending=[True, False, True]
    ).drop(columns=['_priority_rank'])

    out_df = out_df.drop(columns=['_priority_rank'])
    out_df.to_csv(scored_path, index=False)
    queue_df.to_csv(queue_path, index=False)

    summary = {
        'input_rows': int(len(df)),
        'scored_rows': int(len(out_df)),
        'review_queue_rows': int(len(queue_df)),
        'recommended_status_counts': out_df['recommended_status'].value_counts().sort_index().to_dict(),
        'predicted_decision_counts': out_df['predicted_decision'].value_counts().sort_index().to_dict(),
        'stale_listing_flag_counts': out_df['stale_listing_flag'].value_counts().sort_index().to_dict(),
        'removal_recommended_count': int(out_df['removal_recommended'].sum()),
        'auto_removed_count': int(out_df['auto_removed'].sum()),
        'review_priority_counts': out_df['review_priority'].value_counts().sort_index().to_dict(),
        'input_csv': str(input_path),
        'output_scored_csv': str(scored_path),
        'output_queue_csv': str(queue_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print('Saved scored listings:', scored_path)
    print('Saved review queue:', queue_path)
    print('Saved summary:', summary_path)
    print('Review queue rows:', len(queue_df))


if __name__ == '__main__':
    main()
