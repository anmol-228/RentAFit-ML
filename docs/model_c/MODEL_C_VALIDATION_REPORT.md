# Model C Validation Report

## 1. Purpose

This report records the latest validation pass for RentAFit **Model C** after the policy-aware rewrite.

It complements:
- `/Users/mypc/RentAFit/docs/model_c/Model_C_Master_Document.md`

and focuses specifically on:
- smoke-test behavior,
- proxy metrics,
- policy enforcement,
- artifact presence,
- and documentation / visual completeness.

Validation date:
- `2026-03-18`

---

## 2. Validation Scope

The validation covered:

1. item-to-item recommendation flow
2. profile-from-liked-items recommendation flow
3. hard category filter
4. hard gender compatibility
5. exact-size-first then nearest-size fallback
6. explicit-budget and category-budget behavior
7. limited review fallback behavior
8. artifact and visual availability

Primary cross-check report:
- `/Users/mypc/RentAFit/reports/validation/model_crosscheck_report.md`
- `/Users/mypc/RentAFit/reports/validation/model_crosscheck_report.json`

---

## 3. Current Model C Policy Confirmed

Confirmed rules:

- same category only
- hard gender compatibility
- exact size first, then nearest size
- explicit budget if provided
- otherwise category-average budget
- primary approved items first
- up to 2 review-fallback items only if needed
- top 5 recommendations with reason tags

---

## 4. Current Proxy Metrics

From:
- `/Users/mypc/RentAFit/reports/model_c/model_c_proxy_metrics.json`

Current values:
- `fill_rate_at_5 = 0.9350`
- `same_category_at_5 = 1.0000`
- `gender_compatible_at_5 = 1.0000`
- `exact_size_at_5 = 0.6560`
- `size_compatible_at_5 = 0.8881`
- `same_material_at_5 = 0.8531`
- `same_tier_at_5 = 0.5563`
- `avg_rule_quality_score_top5 = 84.17`
- `avg_similarity_score_top5 = 0.6731`
- `avg_budget_alignment_top5 = 0.6900`
- `avg_final_score_top5 = 0.7331`
- `review_fallback_rate_top5 = 0.1296`

Important note:
- category and gender metrics are perfect because they are enforced by policy filters
- the stronger ranking evidence is in material match, tier match, quality, similarity, and budget alignment

---

## 5. Policy-Aware Random Baseline Comparison

From:
- `/Users/mypc/RentAFit/reports/model_c/model_c_proxy_vs_random_metrics.json`

Random baseline values:
- `same_material_at_5 = 0.8069`
- `same_tier_at_5 = 0.5313`
- `avg_rule_quality_score_top5 = 83.56`
- `avg_similarity_score_top5 = 0.6247`
- `avg_budget_alignment_top5 = 0.6727`
- `avg_final_score_top5 = 0.7023`

Conclusion:
- Model C is stronger than the policy-aware random baseline on the ranking-sensitive metrics that matter.

---

## 6. Smoke Tests Confirmed

### 6.1 Women item query
- mode: item-to-item
- category: `Dress`
- gender: `Women`
- result: `5/5`
- review fallback used: `0`

### 6.2 Unisex item query
- mode: item-to-item
- category: `Shirt`
- gender: `Unisex`
- result: `5/5`
- review fallback used: `0`

### 6.3 Profile query
- mode: liked-items profile
- category: `Dress`
- gender: `Women`
- result: `5/5`
- review fallback used: `0`

### 6.4 Fallback edge case
- mode: item-to-item
- category: `Activewear`
- gender: `Unisex`
- result: `4/5`
- review fallback used: `2`

Interpretation:
- the recommender is allowed to return fewer than 5 results when the fallback cap is reached
- this is correct behavior, not a failure

### 6.5 Explicit-budget case
- mode: item-to-item
- category: `Dress`
- gender: `Women`
- result: `5/5`
- budget source: `explicit_user_budget`

---

## 7. Artifact And Visual Audit

Confirmed artifacts exist:
- `/Users/mypc/RentAFit/data/generated/model_c_catalog.csv`
- `/Users/mypc/RentAFit/data/generated/model_c_catalog_recommendable.csv`
- `/Users/mypc/RentAFit/models/model_c/content_based/model_c_preprocessor.joblib`
- `/Users/mypc/RentAFit/models/model_c/content_based/model_c_nearest_neighbors.joblib`
- `/Users/mypc/RentAFit/models/model_c/content_based/model_c_feature_matrix.joblib`
- `/Users/mypc/RentAFit/models/model_c/content_based/model_c_metadata.json`

Confirmed visuals exist:
- `/Users/mypc/RentAFit/reports/model_c/model_c_data_pipeline.png`
- `/Users/mypc/RentAFit/reports/model_c/model_c_architecture.png`
- `/Users/mypc/RentAFit/reports/model_c/model_c_similarity_flow.png`
- `/Users/mypc/RentAFit/reports/model_c/model_c_catalog_distribution_chart.png`
- `/Users/mypc/RentAFit/reports/model_c/model_c_proxy_metrics_chart.png`
- `/Users/mypc/RentAFit/reports/model_c/model_c_proxy_vs_random_chart.png`

Word handbook visual check:
- `/Users/mypc/RentAFit/docs/model_c/Model_C_Master_Document.docx`
- embedded images confirmed: `6`

---

## 8. Improvement Check

A small safe tuning search was run on the Model C re-ranking weights.

Result:
- no material improvement worth adopting was found
- the best alternative improved the composite objective by only about `0.0003`

Decision:
- keep current weights
- avoid changing the system for a negligible gain

---

## 9. Final Validation Verdict

Model C is currently:

- working,
- policy-consistent,
- artifact-complete,
- documented,
- and stable after the rewrite.

No validation anomalies remained in the final combined cross-check.
