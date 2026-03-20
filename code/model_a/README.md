# Model A (Pricing Range) - Organized Pipeline

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
- Training dataset: `/Users/mypc/RentAFit/data/frozen/v1_final/model_a_train_ready.csv`
- Brand master: `/Users/mypc/RentAFit/data/frozen/v1_final/brand_tier_master_project_final.csv`

## Output locations
- Models: `/Users/mypc/RentAFit/models/model_a/...`
- Metrics: `/Users/mypc/RentAFit/reports/model_a/metrics/...`
- Analysis CSVs: `/Users/mypc/RentAFit/reports/model_a/analysis/...`
- Charts: `/Users/mypc/RentAFit/reports/model_a/charts/...`

## Best current training command
```bash
python3 "/Users/mypc/RentAFit/code/model_a/training/train_model_a_rf_pct_tier_split.py"
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
python3 "/Users/mypc/RentAFit/code/model_a/inference/predict_price_range_simple_input.py" \
  --brand "Prada" \
  --category "Dress" \
  --material "Silk" \
  --age_months 6 \
  --size "M" \
  --condition "Like New" \
  --original_price 95000 \
  --json
```
