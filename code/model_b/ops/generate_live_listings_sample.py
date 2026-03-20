from __future__ import annotations

from datetime import date
from pathlib import Path
import argparse

import pandas as pd

BASE = Path('/Users/mypc/RentAFit')
SOURCE = BASE / 'data/generated/model_b_train_expanded_ready.csv'
OUTPUT = BASE / 'data/generated/model_b_live_listings_sample.csv'

AS_OF = date(2026, 3, 14)


def subtract_months(d: date, months: int) -> date:
    year = d.year
    month = d.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(d.day, 28)
    return date(year, month, day)


def main():
    parser = argparse.ArgumentParser(description='Generate a sample live-listings CSV for operational Model B scoring.')
    parser.add_argument('--output_csv', default=str(OUTPUT))
    parser.add_argument('--rows_per_class', type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(SOURCE)
    parts = []
    for decision in ['Approve', 'Review', 'Reject']:
        part = df[df['effective_listing_decision'] == decision].head(args.rows_per_class)
        parts.append(part)
    sample = pd.concat(parts, ignore_index=True)

    listing_age_pattern = [1, 3, 5, 8, 10, 11, 12, 13]
    rows = []
    for i, row in sample.reset_index(drop=True).iterrows():
        listing_age_months = listing_age_pattern[i % len(listing_age_pattern)]
        listing_created_at = subtract_months(AS_OF, listing_age_months + 1).isoformat()
        last_approved_at = subtract_months(AS_OF, listing_age_months).isoformat()

        if i % 6 == 0:
            last_reapproved_at = subtract_months(AS_OF, max(1, listing_age_months - 2)).isoformat()
        else:
            last_reapproved_at = ''

        if row['effective_listing_decision'] == 'Reject':
            current_status = 'PENDING_REVIEW'
        elif listing_age_months >= 10 and i % 2 == 0:
            current_status = 'ACTIVE'
        elif listing_age_months >= 10:
            current_status = 'REAPPROVAL_REQUIRED'
        else:
            current_status = 'ACTIVE'

        rows.append({
            'listing_id': row['listing_id'],
            'provider_id': f'P{1000 + i}',
            'current_status': current_status,
            'brand': row['brand'],
            'category': row['category'],
            'material': row['material'],
            'size': row['size'],
            'condition': row['condition'],
            'garment_age_months': int(row['age_months']),
            'original_price': int(row['original_price']),
            'provider_price': int(row['provider_price']),
            'listing_created_at': listing_created_at,
            'last_approved_at': last_approved_at,
            'last_reapproved_at': last_reapproved_at,
            'last_condition_update_at': '',
            'last_price_update_at': '',
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print('Saved sample live listings:', out_path)
    print('Rows:', len(out))


if __name__ == '__main__':
    main()
