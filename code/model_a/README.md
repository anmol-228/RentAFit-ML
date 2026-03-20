# Model A (Pricing Range) - Organized Pipeline

This folder contains the pricing-range pipeline used by RentAFit listings.

## Folder layout
- `training/`
  - `train_model_a_baseline.py`
  - `train_model_a_rf_abs.py`
  - `train_model_a_rf_pct.py`
  - `train_model_a_rf_pct_tier_split.py` (current production candidate)
- `inference/`
  - `predict_price_range_simple_input.py` (production-style input)
  - `predict_model_a_price_range_legacy.py`
- `analysis/`
  - `model_a_full_report.py`
- `utils/`
  - `feature_builder.py`

## Data input
- Training dataset: `data/frozen/v1_final/model_a_train_ready.csv`
- Brand master: `data/frozen/v1_final/brand_tier_master_project_final.csv`

## Output locations
- Models: `models/model_a/...`
- Metrics: `reports/model_a/metrics/...`
- Analysis CSVs: `reports/model_a/analysis/...`
- Charts: `reports/model_a/charts/...`

## Best current training command
```bash
python3 code/model_a/training/train_model_a_rf_pct_tier_split.py
```

## What the current production candidate does
- Uses percentage targets (`min_pct`, `max_pct`).
- Uses separate models for Tier 5 vs Tier 1-4.
- Adds Tier-5 residual correction model for `max_pct`.
- Adds Tier-5 sparse brand oversampling during training.
- Uses hard post-rules (cap/order/rounding).
- Uses confidence fallback to deterministic rule range on low-confidence cases.

## Production-style prediction input
User must provide:
- `brand, category, material, age_months, size, condition, original_price`

Model derives internally:
- `tier_primary`
- `base_min_pct`, `base_max_pct`
- `cond_mult`, `age_mult`, `cat_mult`, `mat_mult`
- Tier-5 brand features from brand master (`brand_avg_price_min/max`, open-ended flag)

```bash
python3 code/model_a/inference/predict_price_range_simple_input.py \
  --brand "Prada" \
  --category "Dress" \
  --material "Silk" \
  --age_months 6 \
  --size "M" \
  --condition "Like New" \
  --original_price 95000 \
  --json
```

Expected headline output for the sample above:

```json
{
  "final_price_range": {
    "min_price": 9300,
    "max_price": 11900,
    "source": "model_output"
  },
  "confidence": {
    "score": 0.9,
    "fallback_to_rule_range": false
  },
  "model_route": "tier_split_tier5"
}
```
