# Model A Multi-Input Validation Report

## Scope
- Validated model inference on multiple inputs (dataset-backed + custom simple input).
- Verified manual rule compliance in outputs:
  - `min_price <= max_price`
  - bucket rounding rule (10/50/100)
  - postprocess consistency check
  - non-negative outputs

## Files Generated
- Detailed results: `reports/model_a/analysis/model_a_multi_input_validation_details.csv`
- Summary JSON: `reports/model_a/analysis/model_a_multi_input_validation_summary.json`
- Known-tier summary: `reports/model_a/analysis/model_a_multi_input_known_tier_summary.csv`

## Test Coverage
- Total tests: **78**
- Dataset-backed tests (known targets): **70**
- Custom simple-input tests: **8**

## Rule Compliance Result
- All-rule pass count: **78**
- Rule check fail count: **0**

## Accuracy on Known-Target Tests
- MAE(min): **13.4286**
- MAE(max): **15.5714**
- Worst absolute error on max: **400.0000**

## Model Route Usage
- `tier_split_tier1to4`: 61
- `tier_split_tier5`: 17

## Original Price Source Usage
- `provided_by_user`: 70
- `estimated_from_brand_master`: 8

## Known-Target Tier Summary
| tier_primary | rows | mae_min | mae_max | max_abs_error_max |
|---|---:|---:|---:|---:|
| Tier 1 | 13 | 0.0000 | 0.7692 | 10.0000 |
| Tier 2 | 13 | 0.7692 | 1.5385 | 10.0000 |
| Tier 3 | 14 | 5.7143 | 4.2857 | 50.0000 |
| Tier 4 | 16 | 15.6250 | 6.2500 | 50.0000 |
| Tier 5 | 14 | 42.8571 | 64.2857 | 400.0000 |

## Conclusion
- Output rules are being enforced consistently for all tested inputs.