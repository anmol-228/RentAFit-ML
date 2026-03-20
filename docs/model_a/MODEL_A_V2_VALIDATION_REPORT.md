# Model A v2 Validation Report (Tier5 Fix Pack)

## Scope
- Validated inference after implementing Tier5 improvements:
  - mandatory `original_price`
  - Tier5 brand-master features
  - Tier5 sparse-brand oversampling (training)
  - Tier5 max residual correction
  - confidence-based fallback to deterministic rule range

## Files Generated
- Details: `reports/model_a/analysis/model_a_v2_multi_input_validation_details.csv`
- Summary: `reports/model_a/analysis/model_a_v2_multi_input_validation_summary.json`
- Tier summary: `reports/model_a/analysis/model_a_v2_known_tier_summary.csv`

## Coverage
- Total tests: **88**
- Known-target tests: **80**
- Custom tests: **8**

## Rule Compliance
- Rule pass: **88**
- Rule fail: **0**

## Runtime Safety Routing
- Final model output count: **87**
- Final rule-fallback count: **1**

## Accuracy (Known Targets)
- MAE(min): **8.3750**
- MAE(max): **15.5000**
- Worst max error: **400.0000**

## Conclusion
- Output rule consistency is preserved across all tested inputs.