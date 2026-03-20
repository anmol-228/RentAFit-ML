from pathlib import Path
import argparse
import json

import sys
REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
ROOT_CODE_DIR = REPO_ROOT / 'code'
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_b.runtime import predict_one


def main():
    parser = argparse.ArgumentParser(description='Predict Model B listing decision and lifecycle status using the trained LSTM hybrid model.')
    parser.add_argument('--brand', required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--gender', required=False, help='Optional explicit gender. If omitted, Model B derives it from category.')
    parser.add_argument('--material', required=True)
    parser.add_argument('--size', required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--age_months', required=False, type=int, help='Backward-compatible garment age input in months.')
    parser.add_argument('--garment_age_months', required=False, type=int, help='Preferred garment age input in months.')
    parser.add_argument('--original_price', required=True, type=float)
    parser.add_argument('--provider_price', required=True, type=float)
    parser.add_argument('--current_status', required=False, default='PENDING_REVIEW')
    parser.add_argument('--listing_created_at', required=False)
    parser.add_argument('--last_approved_at', required=False)
    parser.add_argument('--last_reapproved_at', required=False)
    parser.add_argument('--as_of_date', required=False, help='Optional YYYY-MM-DD date for lifecycle calculation.')
    parser.add_argument('--auto_remove_stale', action='store_true', help='If set, stale listings beyond grace period can move to REMOVED.')
    parser.add_argument('--removal_grace_months', required=False, type=int, default=3)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    if args.age_months is None and args.garment_age_months is None:
        parser.error('Provide either --age_months or --garment_age_months.')

    result = predict_one(
        brand=args.brand,
        category=args.category,
        gender=args.gender,
        material=args.material,
        size=args.size,
        condition=args.condition,
        age_months=args.age_months,
        garment_age_months=args.garment_age_months,
        original_price=args.original_price,
        provider_price=args.provider_price,
        current_status=args.current_status,
        listing_created_at=args.listing_created_at,
        last_approved_at=args.last_approved_at,
        last_reapproved_at=args.last_reapproved_at,
        as_of_date=args.as_of_date,
        auto_remove_stale=args.auto_remove_stale,
        removal_grace_months=args.removal_grace_months,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('Model B Listing Moderation')
        print('-------------------------')
        print('Input:', result['input'])
        print('Predicted decision:', result['prediction']['predicted_decision'])
        print('Raw predicted decision:', result['prediction']['raw_predicted_decision'])
        print('Suggested status:', result['prediction']['suggested_listing_status'])
        print('Gender context:', {
            'gender': result['input']['gender'],
            'gender_source': result['input']['gender_source'],
            'gender_conflict_flag': result['derived_features']['gender_conflict_flag'],
            'frontend_popup_recommended': result['lifecycle']['frontend_popup_recommended'],
        })
        print('Listing age context:', {
            'listing_age_months': result['lifecycle']['listing_age_months'],
            'listing_age_source': result['lifecycle']['listing_age_source'],
            'listing_age_reference_date': result['lifecycle']['listing_age_reference_date'],
            'as_of_date': result['lifecycle']['as_of_date'],
        })
        print('Lifecycle flags:', {
            'stale_listing_flag': result['lifecycle']['stale_listing_flag'],
            'removal_recommended': result['lifecycle']['removal_recommended'],
            'auto_removed': result['lifecycle']['auto_removed'],
            'review_required': result['lifecycle']['review_required'],
            'visible_to_renters': result['lifecycle']['visible_to_renters'],
        })
        if result['lifecycle']['frontend_popup_recommended']:
            print('Popup message:', result['lifecycle']['frontend_popup_message'])
        print('Class probabilities:', result['prediction']['class_probabilities'])


if __name__ == '__main__':
    main()
