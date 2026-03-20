# Model B (Listing Quality / Reapproval)

This folder contains the working RentAFit **Model B** pipeline.

## What Model B does

Model B is the listing moderation model. It predicts whether a listing should be:

- `Approve`
- `Review`
- `Reject`

It also now includes the operational lifecycle layer for:

- stale-listing detection,
- reapproval-required status,
- review-queue generation,
- reviewer-role assignment,
- website-ready listing status recommendations,
- gender-aware moderation policy.

## Completed components

### Data and labels
- manual gold dataset,
- reviewed expansion batch,
- expanded gold labels,
- expanded train/val/test dataset,
- gender-aware training variants for unisex and conflict cases.

### ML model
- hybrid **LSTM + tabular** PyTorch classifier,
- trained weights,
- saved sklearn preprocessor,
- metrics and charts.

### Operational lifecycle layer
- date-based listing age calculation,
- separation of garment age and listing age,
- stale listing detection at `10+ months`,
- status mapping to:
  - `ACTIVE`
  - `PENDING_REVIEW`
  - `REAPPROVAL_REQUIRED`
  - `REJECTED`
  - optional `REMOVED`
- live-listing batch scoring,
- review-queue generation.

## Main files

### Preparation
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_dataset.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_expansion_candidates.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_expanded_dataset.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_gender_policy_dataset.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_splits.py`
- `/Users/mypc/RentAFit/code/model_b/gender_policy.py`

### Training
- `/Users/mypc/RentAFit/code/model_b/training/train_model_b_lstm.py`

### Runtime and lifecycle
- `/Users/mypc/RentAFit/code/model_b/lifecycle.py`
- `/Users/mypc/RentAFit/code/model_b/runtime.py`

### Inference and ops
- `/Users/mypc/RentAFit/code/model_b/inference/predict_model_b_decision.py`
- `/Users/mypc/RentAFit/code/model_b/ops/generate_live_listings_sample.py`
- `/Users/mypc/RentAFit/code/model_b/ops/run_model_b_live_ops.py`

## Age logic

### Model input age
- `garment_age_months`
- used by the trained LSTM + tabular classifier

### Lifecycle age
- `listing_age_months`
- computed from:
  - `last_reapproved_at`
  - `last_approved_at`
  - `listing_created_at`
- used to decide stale-listing reapproval flow

## Key policy

If a listing is stale (`listing_age_months >= 10`), it should not remain automatically active.
It should move to:
- `REAPPROVAL_REQUIRED`

Gender policy:
- if user leaves gender blank, derive it from category
- unisex categories allow both `Men` and `Women` without penalty
- women-specific categories currently keep `Women` as the expected value
- if user-selected gender conflicts with category policy, the listing should go to `Review`
- runtime also returns a frontend popup recommendation for those conflict cases

Review ownership in queue output:
- `assigned_reviewer_role = admin_reviewer`

## Runtime output shape

Single-listing prediction now returns:

- `prediction`
  predicted decision, class probabilities, suggested status
- `lifecycle`
  listing age, stale flags, removal recommendation, reviewer routing, visibility
- `summary`
  compact fields for API/frontend use

This makes Model B easier to integrate without reading multiple nested structures manually.

## Main saved outputs

- model: `/Users/mypc/RentAFit/models/model_b/model_b_lstm.pt`
- preprocessor: `/Users/mypc/RentAFit/models/model_b/model_b_tabular_preprocessor.joblib`
- metrics: `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_metrics.json`
- confusion matrix: `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_confusion_matrix_test.png`
- training history: `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_training_history.png`
- live ops summary: `/Users/mypc/RentAFit/reports/model_b/model_b_live_ops_summary.json`
- review queue sample: `/Users/mypc/RentAFit/reports/model_b/model_b_review_queue_sample.csv`
- cross-check report: `/Users/mypc/RentAFit/reports/validation/model_crosscheck_report.md`

## Supporting docs

- `/Users/mypc/RentAFit/docs/model_b/Model_B_Master_Document.md`
- `/Users/mypc/RentAFit/docs/model_b/MODEL_B_OPERATIONAL_LIFECYCLE_GUIDE.md`
- `/Users/mypc/RentAFit/docs/model_b/MODEL_B_LSTM_RESULTS.md`

## Commands

Single-listing inference:

```bash
python3 /Users/mypc/RentAFit/code/model_b/inference/predict_model_b_decision.py \
  --brand "Zara" \
  --category "Top" \
  --gender "Women" \
  --material "Cotton" \
  --size "S" \
  --condition "Like New" \
  --garment_age_months 3 \
  --original_price 2599 \
  --provider_price 180 \
  --listing_created_at 2026-02-01 \
  --as_of_date 2026-03-14 \
  --json
```

Operational batch scoring:

```bash
python3 /Users/mypc/RentAFit/code/model_b/ops/run_model_b_live_ops.py --as_of_date 2026-03-14
```
