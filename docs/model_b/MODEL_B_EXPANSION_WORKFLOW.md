# Model B Expansion Workflow

## Goal

Expand the trusted human-reviewed Model B dataset without manually creating every row from scratch.

## Core idea

We do not auto-generate final gold labels for Model B the way we did for Model A.
Instead, we:

1. generate candidate rows automatically,
2. prefill rule-based suggestions,
3. manually review only the final human label fields.

This keeps Model B academically valid because the final truth still comes from human review.

## Source file used

- `data/frozen/v1_final/pricing_features_augmented_1500.csv`

## Existing gold labels not reused

Rows already present in:
- `data/frozen/v1_final/gold_labels_model_b_full_manual.csv`

are excluded from the expansion batch.

## Candidate selection priorities

The selection script prioritizes rows that are most useful for training:

1. older listings (`age_months >= 10`)
2. high deviation (`deviation_M > 35`)
3. used-condition listings
4. premium tiers (`Tier 4`, `Tier 5`)
5. borderline review-like cases

## Why these rows were prioritized

These are the rows where Model B learns the most:
- difficult cases,
- edge cases,
- reapproval/removal cases,
- pricing-risk cases,
- premium and high-value cases.

## Review file created

- `data/generated/model_b_expansion_candidates_manual_review.csv`

## Columns to review manually

Humans should fill these columns:

- `reviewer_name`
- `review_date`
- `final_quality_score_0_100`
- `final_listing_decision`
- `final_reviewer_note`

## Columns that are suggestions only

These are helper fields, not final truth:

- `rule_quality_score`
- `rule_decision`
- `auto_suggested_quality_score_0_100`
- `auto_suggested_listing_decision`

## Important review rule

If a listing is `10+ months` old, it should not remain freely approved for the current cycle.
It should be treated as requiring reapproval or possible removal.

## After manual review

The reviewed rows can later be merged into a new expanded gold label file, for example:

- `gold_labels_model_b_expanded.csv`

That expanded file can then be used to retrain Model B.
