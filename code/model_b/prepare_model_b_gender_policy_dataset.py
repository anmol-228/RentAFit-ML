from pathlib import Path
import argparse
import json

import pandas as pd

from gender_policy import build_gender_variant_summary, expand_gender_variants


def main():
    parser = argparse.ArgumentParser(description='Expand Model B datasets with explicit gender variants and policy overrides.')
    parser.add_argument('--input-path', required=True, help='Base Model B dataset with gender columns.')
    parser.add_argument('--out-path', required=True, help='Output augmented Model B dataset path.')
    parser.add_argument('--summary-path', required=False, help='Optional JSON summary path.')
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    out = expand_gender_variants(df)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    summary = build_gender_variant_summary(out)
    if args.summary_path:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    print('Saved:', args.out_path)
    if args.summary_path:
        print('Saved summary:', args.summary_path)
    print('rows:', len(out))
    print('gender_counts:', summary['gender_counts'])
    print('gender_variant_type_counts:', summary['gender_variant_type_counts'])
    print('gender_policy_override_applied:', summary['gender_policy_override_applied'])


if __name__ == '__main__':
    main()
