# Model A Integration Guide (Website/API)

## Required website input (provider flow)
- `brand`
- `category`
- `material`
- `age_months`
- `size`
- `condition`
- `original_price` (mandatory)

## Why original_price is mandatory now
- Tier 5 quality strongly depends on accurate scale.
- Estimating original price increases risk for luxury brands.
- Mandatory original price significantly improves stability and predictability.

## Internal model derivations (not entered by user)
- `tier_primary`
- `base_min_pct`, `base_max_pct`
- `cond_mult`, `age_mult`, `cat_mult`, `mat_mult`
- `brand_avg_price_min`, `brand_avg_price_max`, `is_open_ended_brand_price`

## Runtime safeguards
1. Tier-specific routing: Tier 5 model vs Tier 1-4 model
2. Tier 5 residual correction on max prediction
3. Hard post-rules:
- cap to 20% of original price
- enforce min <= max
- bucket rounding (10/50/100)
4. Confidence gate:
- if low confidence, final output falls back to deterministic rule range

## CLI test command
```bash
python3 "code/model_a/inference/predict_price_range_simple_input.py" \
  --brand "Prada" \
  --category "Dress" \
  --material "Silk" \
  --age_months 6 \
  --size "M" \
  --condition "Like New" \
  --original_price 95000 \
  --json
```

## Suggested API contract (Python ML service)
### Request JSON
```json
{
  "brand": "Prada",
  "category": "Dress",
  "material": "Silk",
  "age_months": 6,
  "size": "M",
  "condition": "Like New",
  "original_price": 95000
}
```

### Response JSON (minimal)
```json
{
  "min_price": 9300,
  "max_price": 11900
}
```

### Response JSON (debug/audit mode)
- include `derived_features`, `candidate_ranges`, `confidence`, `model_route`, and `final_price_range.source`.
