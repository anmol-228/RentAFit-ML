# Model B Operational Lifecycle Guide

## Purpose

This guide explains the non-training part of Model B: how the trained moderation model is used to drive listing status, stale-listing detection, and review-queue generation in a website-ready workflow.

## What is now implemented

### 1. Date-based listing age calculation
Implemented in:
- `code/model_b/lifecycle.py`

The system can now compute `listing_age_months` from:
- `last_reapproved_at`
- `last_approved_at`
- `listing_created_at`

Priority order:
1. `last_reapproved_at`
2. `last_approved_at`
3. `listing_created_at`
4. fallback to provided age input if no dates are available

### 2. Separation of garment age and listing age
Implemented in:
- `code/model_b/runtime.py`
- `code/model_b/inference/predict_model_b_decision.py`

Current logic:
- `garment_age_months` is used as the model input age.
- `listing_age_months` is computed for lifecycle decisions.

This means pricing/quality behavior and stale-listing workflow can now be discussed separately.

### 3. Website-ready lifecycle status mapping
Implemented in:
- `code/model_b/lifecycle.py`

Current status logic:
- `Reject` -> `REJECTED`
- stale listing -> `REAPPROVAL_REQUIRED`
- `Approve` and not stale -> `ACTIVE`
- otherwise -> `PENDING_REVIEW`

Optional support also exists for:
- `REMOVED`

If `auto_remove_stale` is enabled and stale grace period is crossed, the lifecycle logic can move a stale listing to `REMOVED`.

### 4. Reviewer ownership and queue generation
Implemented in:
- `code/model_b/ops/run_model_b_live_ops.py`

Rows needing review are assigned:
- `assigned_reviewer_role = admin_reviewer`

Queue fields include:
- `review_reason`
- `review_priority`
- `stale_listing_flag`
- `removal_recommended`

### 5. Live-listing batch scoring
Implemented in:
- `code/model_b/ops/run_model_b_live_ops.py`

This script reads a CSV of current listings, runs Model B, computes lifecycle age, and writes:
- scored listing snapshot
- review queue
- summary JSON

## Files added for the operational layer

### Code
- `code/model_b/lifecycle.py`
- `code/model_b/runtime.py`
- `code/model_b/ops/generate_live_listings_sample.py`
- `code/model_b/ops/run_model_b_live_ops.py`

### Template and sample input
- `data/templates/model_b_live_listings_template.csv`
- `data/generated/model_b_live_listings_sample.csv`

### Sample outputs
- `reports/model_b/model_b_live_listings_scored_sample.csv`
- `reports/model_b/model_b_review_queue_sample.csv`
- `reports/model_b/model_b_live_ops_summary.json`

## Sample operational run

### Generate sample live listings
```bash
python3 code/model_b/ops/generate_live_listings_sample.py
```

### Run live operational scoring
```bash
python3 code/model_b/ops/run_model_b_live_ops.py --as_of_date 2026-03-14
```

## Current sample output summary

From:
- `reports/model_b/model_b_live_ops_summary.json`

Counts:
- input rows: `60`
- review queue rows: `47`
- recommended status counts:
  - `ACTIVE: 13`
  - `PENDING_REVIEW: 9`
  - `REAPPROVAL_REQUIRED: 18`
  - `REJECTED: 20`
- stale listing count: `26`
- removal recommended count: `7`

## What is still future backend work

The Python-side operational logic is now implemented, but these pieces still belong to the actual website/backend layer:

- storing listing dates in a real database,
- running scheduled re-check jobs automatically,
- auto-hiding listings from renter UI based on status,
- building an admin review dashboard,
- exposing this as a backend API endpoint.

So Model B is now complete on the **ML + lifecycle logic side**, while final website automation still needs backend integration.
