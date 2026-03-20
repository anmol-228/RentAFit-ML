from pathlib import Path
import argparse
import json

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_c.runtime import recommend_from_item, recommend_from_profile


def main():
    parser = argparse.ArgumentParser(description='Recommend similar RentAFit items using the Model C content-based baseline.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seed_item_id', help='Recommend items similar to a single viewed item.')
    group.add_argument('--liked_item_ids', help='Comma-separated liked item IDs to build a renter preference profile.')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_filter', required=False)
    parser.add_argument('--max_provider_price', type=float, required=False)
    parser.add_argument('--exclude_same_brand', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    if args.seed_item_id:
        result = recommend_from_item(
            seed_item_id=args.seed_item_id,
            top_k=args.top_k,
            category_filter=args.category_filter,
            max_provider_price=args.max_provider_price,
            exclude_same_brand=args.exclude_same_brand,
        )
    else:
        liked_ids = [x.strip() for x in str(args.liked_item_ids).split(',') if x.strip()]
        result = recommend_from_profile(
            liked_item_ids=liked_ids,
            top_k=args.top_k,
            category_filter=args.category_filter,
            max_provider_price=args.max_provider_price,
            exclude_same_brand=args.exclude_same_brand,
        )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == '__main__':
    main()
