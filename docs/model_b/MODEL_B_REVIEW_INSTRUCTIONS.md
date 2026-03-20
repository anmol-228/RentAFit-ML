# Model B Review Instructions

Use the expansion candidate file:

- `data/generated/model_b_expansion_candidates_manual_review.csv`

## What to do row by row

1. read listing details and pricing behavior
2. look at:
   - condition
   - age_months
   - deviation_M
   - penalties
   - rule suggestions
3. decide final human score and decision
4. fill the final columns

## Fill these columns

- `reviewer_name`
- `review_date`
- `final_quality_score_0_100`
- `final_listing_decision`
- `final_reviewer_note`

## Do not treat suggestions as mandatory

The following are only guidance:

- `rule_decision`
- `auto_suggested_listing_decision`
- `rule_quality_score`
- `auto_suggested_quality_score_0_100`

Humans are allowed to disagree when needed.

## Strong priority while reviewing

Pay extra attention to rows where:

- `older_listing_flag = 1`
- `high_deviation_flag = 1`
- `used_flag = 1`
- `premium_tier_flag = 1`

## Current age policy

If `age_months >= 10`, the listing is considered older and should be reviewed carefully for reapproval or removal.
