# RentAFit Model A Handbook (V2 Rerun, Fully Synced)

## 1) Document objective

This document is the complete technical record of **Model A (Rental Price Range Prediction)** in RentAFit, updated after a fresh **v2 rerun**.

It covers:
- problem statement,
- datasets and rule logic,
- model evolution from baseline to v2,
- exact formulas,
- libraries and implementation details,
- fresh run outputs,
- graph generation and graph-by-graph interpretation,
- inference behavior and safety controls.

This version is intended to be both beginner-readable and technically correct for academic review.

---

## 2) Problem definition

Model A predicts a rental **price range** for a listing:
- `min_price`
- `max_price`

Input attributes:
- `brand`
- `category`
- `material`
- `size`
- `condition`
- `age_months`
- `original_price`

Why range (min/max) and not one value:
- rental prices are naturally uncertain,
- a range is safer for platform moderation,
- a range supports provider flexibility while keeping prices realistic.

---

## 3) Code and artifact locations

### 3.1 Main code
- `code/model_a/training/train_model_a_baseline.py`
- `code/model_a/training/train_model_a_rf_abs.py`
- `code/model_a/training/train_model_a_rf_pct.py`
- `code/model_a/training/train_model_a_rf_pct_tier_split.py` (current v2 model)
- `code/model_a/inference/predict_price_range_simple_input.py`
- `code/model_a/analysis/model_a_full_report.py`
- `code/model_a/utils/feature_builder.py`
- `code/pricing_rules.py`

### 3.2 Data snapshot used
- `data/frozen/v1_final/model_a_train_ready.csv`
- `data/frozen/v1_final/brand_tier_master_project_final.csv`

### 3.3 Model outputs
- `models/model_a/rf_pct_tier_split/*.pkl`
- `models/model_a/rf_pct_tier_split/model_a_rf_pct_tier_split_metadata.json`

### 3.4 Reports and charts
- `reports/model_a/metrics/*.json`
- `reports/model_a/analysis/*.csv`
- `reports/model_a/charts/*.png`

---

## 4) Dataset profile (used in this rerun)

From `model_a_train_ready.csv`:
- Total rows: **1500**
- Total columns: **20**
- Split:
  - train: **1082**
  - val: **287**
  - test: **131**
- Tier distribution:
  - Tier 1: **269**
  - Tier 2: **275**
  - Tier 3: **292**
  - Tier 4: **380**
  - Tier 5: **284**
- Data source mix:
  - manual: **375**
  - synthetic: **1125**

From `brand_tier_master_project_final.csv`:
- Brand rows: **79**
- Tier distribution:
  - Tier 1: 5
  - Tier 2: 13
  - Tier 3: 21
  - Tier 4: 21
  - Tier 5: 19
- Missing `avg_price_max` rows (open-ended price bands): **13**

---

## 5) Pricing rule foundation (supervised target source)

Model A is trained on labels generated from deterministic rules.

Rule file:
- `code/pricing_rules.py`

### 5.1 Tier assignment by original price
- `<= 1200` -> Tier 1
- `<= 2500` -> Tier 2
- `<= 5000` -> Tier 3
- `<= 15000` -> Tier 4
- `> 15000` -> Tier 5

### 5.2 Base percentages
`BASE_MIN`
- Tier 1: 0.05
- Tier 2: 0.06
- Tier 3: 0.07
- Tier 4: 0.09
- Tier 5: 0.11

`BASE_MAX`
- Tier 1: 0.06
- Tier 2: 0.07
- Tier 3: 0.09
- Tier 4: 0.11
- Tier 5: 0.14

### 5.3 Multipliers
Condition multiplier:
- New -> 1.00
- Like New -> 0.90
- Used -> 0.70

Age multiplier:
- <=3 -> 1.00
- <=6 -> 0.90
- <=10 -> 0.80
- <=15 -> 0.70
- >15 -> 0.60

Category multiplier:
- Saree / Lehenga / Ethnic Wear -> 1.10
- Dress / Jacket -> 1.05
- Shirt / Top / Kurta -> 1.00
- Activewear / Jeans -> 0.95

Material multiplier:
- Silk / Leather -> 1.05
- Linen -> 1.03
- Others -> 1.00

### 5.4 Core rule formulas
Let:
- `OP = original_price`
- `bmin = BASE_MIN[tier_primary(OP)]`
- `bmax = BASE_MAX[tier_primary(OP)]`
- `cm = cond_mult(condition)`
- `am = age_mult(age_months)`
- `catm = cat_mult(category)`
- `matm = mat_mult(material)`

Then:
- `raw_min = OP * bmin * cm * am * catm * matm`
- `raw_max = OP * bmax * cm * am * catm * matm`

Capping rule:
- `cap_min = min(raw_min, 0.20*OP, OP)`
- `cap_max = min(raw_max, 0.20*OP, OP)`

Rounding buckets:
- value < 200 -> nearest 10
- 200 <= value <= 1000 -> nearest 50
- value > 1000 -> nearest 100

Final constraints:
- `rule_min = bucket_round(cap_min)`
- `rule_max = bucket_round(cap_max)`
- if `rule_max < rule_min`, force `rule_max = rule_min`

Model targets:
- `target_rule_min = rule_min`
- `target_rule_max = rule_max`

---

## 6) Why Model A evolved in multiple stages

### Stage 1: Linear Regression baseline
Purpose:
- establish a baseline reference.

Issue:
- weak non-linear fit,
- many invalid range order cases before post-fix.

### Stage 2: RF with absolute targets
Purpose:
- capture non-linear interactions better.

Issue:
- price scales across tiers are very different (Tier 1 vs Tier 5), reducing consistency.

### Stage 3: RF with percentage targets
Key idea:
- predict `% of original price` instead of absolute price directly.

Benefit:
- scale normalization,
- major MAE drop.

### Stage 4 (current): RF % tier split + Tier5 residual + oversampling
Why:
- error concentration in Tier 5 remained higher in absolute terms.

V2 improvements:
- separate models for Tier 5 and Tier 1-4,
- Tier 5 sparse-brand oversampling,
- Tier 5 residual correction model for max% prediction,
- confidence-based fallback to deterministic rule range.

---


## 7) Libraries used and why

### Core Python
- `json`: saves metrics, metadata, and route summaries.
- `argparse`: creates command-line style inference scripts.
- `pathlib`: stable path handling for models, datasets, and reports.
- `math`: supports utility-style numeric logic.

### Data handling
- `pandas`: reads CSV files, merges brand master information, prepares feature tables, and builds report summaries.
- `numpy`: handles numeric transformations, vectorized operations, quantiles, and error calculations.

### Machine learning
- `scikit-learn` is the main ML framework for Model A.

Used components:
- `LinearRegression`: baseline model for simple first comparison.
- `RandomForestRegressor`: main non-linear regression engine.
- `ColumnTransformer`: separates categorical and numeric preprocessing cleanly.
- `OneHotEncoder(handle_unknown='ignore')`: safely encodes brand/category/material/size/condition/tier without crashing on unseen values.
- `Pipeline`: keeps preprocessing and model together as one trainable and savable unit.
- `mean_absolute_error` and `mean_squared_error`: used for evaluation.

### Serialization
- `joblib`: saves trained sklearn pipelines as `.pkl` files so the same preprocessing and model structure can be reused during inference.

### Visualization
- `matplotlib`: generates parity, residual, tier-error, and model-comparison charts.

### Why this technology stack is suitable
This stack is practical for a student ML pricing project because:
- sklearn is strong for tabular regression,
- pandas and numpy make structured data easy to manage,
- joblib keeps deployment simple,
- matplotlib makes the results explainable.

### One-hot encoding used in Model A
One-hot encoding is used for all categorical Model A inputs:
- `brand`
- `category`
- `material`
- `size`
- `condition`
- `tier_primary`

These encoded columns are stored inside the fitted sklearn pipelines and were also exported separately for inspection in:
- `data/frozen/v1_final/model_a_onehot_exports/model_a_final_onehot_combined.csv`
- `data/frozen/v1_final/model_a_onehot_exports/model_a_final_onehot_tier1to4.csv`
- `data/frozen/v1_final/model_a_onehot_exports/model_a_final_onehot_tier5.csv`

## 8) Feature engineering in v2

### 8.1 Main feature set
- `brand`
- `category`
- `material`
- `size`
- `condition`
- `tier_primary`
- `age_months`
- `original_price`
- `base_min_pct`
- `base_max_pct`
- `cond_mult`
- `age_mult`
- `cat_mult`
- `mat_mult`
- `brand_avg_price_min`
- `brand_avg_price_max`
- `is_open_ended_brand_price`
- `brand_known_in_master`

### 8.2 Target transformation
- `target_min_pct = target_rule_min / original_price`
- `target_max_pct = target_rule_max / original_price`

### 8.3 Tier 5 residual correction
- base Tier 5 max model predicts `base_pred_max_pct`
- residual target: `target_max_pct - base_pred_max_pct`
- residual model input includes all features + `base_pred_max_pct`
- final correction:
  - `pred_max_pct = base_pred_max_pct + residual_scale * residual_pred`
  - current `residual_scale = 0.35`

### 8.4 Oversampling in Tier 5
- sparse Tier 5 brands are duplicated with replacement in training only.
- current run:
  - added rows: **113**
  - target per brand: **22**
  - Tier 5 train rows: 213 -> 326


### 8.5 Model A training architecture

File:
- `reports/model_a/model_a_training_architecture.png`

![Model A Training Architecture](reports/model_a/model_a_training_architecture.png)

Quick pointers:
- Read this left to right: data and feature building happen first, then route-specific training happens, then outputs are postprocessed and saved.
- The Tier 5 path is separated because luxury items behaved differently and needed extra correction.
- This image explains why the final Model A system is more structured than a single plain regressor.

How to read this diagram:
- the left side is the frozen training dataset and feature-building layer.
- the middle separates the two main learning routes: Tier 1-4 and Tier 5.
- the lower Tier 5 residual block exists because luxury/max-price behavior is harder and needed a correction layer.
- the right side shows that predictions are still postprocessed before artifacts are saved.

Why this matters:
- it shows that Model A is not a single random regressor.
- it is a structured system built around percentage targets, route-specific training, and rule-safe postprocessing.
- this is the clearest visual explanation for why the final v2 model outperformed the earlier simpler versions.

---

## 9) Fresh v2 rerun execution logs

### 9.1 Training command executed
```bash
python3 code/model_a/training/train_model_a_rf_pct_tier_split.py
```

Training output:
```text
Model A Tier-Split RF (% target, v2 with Tier5 residual + oversampling) Metrics
train_mae_min: 2.855822550831793
train_rmse_min: 23.62484046244874
train_mae_max: 3.068391866913124
train_rmse_max: 25.871143796339364
train_range_violations_after_fix: 0
val_mae_min: 13.066202090592334
val_rmse_min: 34.69629306731831
val_mae_max: 15.993031358885018
val_rmse_max: 47.53505949857537
val_range_violations_after_fix: 0
test_mae_min: 8.931297709923664
test_rmse_min: 25.06480151146544
test_mae_max: 13.816793893129772
test_rmse_max: 31.634844061234535
test_range_violations_after_fix: 0

Oversample info: {'oversample_added_rows': 113, 'target_per_brand': 22, 'brands_in_tier5_train': 14}
Saved models in: models/model_a/rf_pct_tier_split
Saved metrics: reports/model_a/metrics/model_a_rf_pct_tier_split_metrics.json
Saved analysis files in: reports/model_a/analysis
```

### 9.2 Analysis/report command executed
```bash
python3 code/model_a/analysis/model_a_full_report.py
```

Analysis output:
```text
Saved metrics: reports/model_a/metrics/model_a_full_metrics.json
Saved analysis CSVs in: reports/model_a/analysis
Saved charts in: reports/model_a/charts
Overall metrics: {'train_mae_min': 2.855822550831793, 'train_rmse_min': 23.62484046244874, 'train_mae_max': 3.068391866913124, 'train_rmse_max': 25.871143796339364, 'train_range_violations_after_fix': 0, 'val_mae_min': 13.066202090592334, 'val_rmse_min': 34.69629306731831, 'val_mae_max': 15.993031358885018, 'val_rmse_max': 47.53505949857537, 'val_range_violations_after_fix': 0, 'test_mae_min': 8.931297709923664, 'test_rmse_min': 25.06480151146544, 'test_mae_max': 13.816793893129772, 'test_rmse_max': 31.634844061234535, 'test_range_violations_after_fix': 0}
```

---


## 9.1) What improvements were made across Model A versions

Model A did not become strong in one step. It improved through a series of engineering decisions.

### Improvement 1: Move beyond the linear baseline
The linear baseline was useful only as a reference point.
It could not represent the non-linear relationship between:
- tier,
- age,
- condition,
- category,
- material,
- original price.

This is why Random Forest was introduced.

### Improvement 2: Shift from absolute rupee targets to percentage targets
Predicting direct rupee values made the model struggle across very different scales.
For example:
- a Tier 1 listing may be in the hundreds,
- a Tier 5 listing may be in the tens of thousands.

Using percentage targets normalized the learning problem:
- `target_min_pct = rule_min / original_price`
- `target_max_pct = rule_max / original_price`

This was one of the biggest improvements in the whole Model A journey.

### Improvement 3: Split Tier 1-4 and Tier 5 into separate routes
Even after percentage modeling, Tier 5 remained harder because:
- luxury prices vary more,
- open-ended price bands are more common,
- fewer rows exist per luxury brand.

So the model was split into:
- Tier 1-4 route,
- Tier 5 route.

This reduced interference between normal listings and luxury listings.

### Improvement 4: Oversample sparse Tier 5 brands
Some Tier 5 brands had too few examples.
Instead of leaving them weakly represented, we oversampled sparse Tier 5 brands during training.
This improved route stability for luxury predictions.

### Improvement 5: Add Tier 5 residual correction for max percent
Tier 5 max-price prediction was still the hardest part.
So a second residual model was added to learn the remaining error after the base Tier 5 max model.
This is why the current v2 setup is not just one model; it is a coordinated tiered system.

### Improvement 6: Add confidence scoring and rule fallback
Even a strong model should not always be trusted blindly.
So confidence logic was added to penalize risky situations such as:
- unseen brand in route training,
- missing user-provided original price,
- open-ended brand price bands,
- unusually wide predicted ranges.

If confidence falls below threshold, the system switches from model output to deterministic rule output.
This makes the pricing engine safer for website use.

### Why these improvements matter together
The final Model A works well because it combines:
- rule-grounded supervision,
- scale normalization,
- tier-aware routing,
- luxury-route stabilization,
- postprocessing safety,
- confidence-based fallback.

So the final system is stronger than a plain regression model.

## 10) Stage-wise results (all model versions)

### 10.1 Linear baseline (`model_a_baseline_metrics.json`)
| metric | value |
|---|---:|
| val_mae_min | 380.973257442866 |
| val_rmse_min | 697.9598748378612 |
| val_mae_max | 457.769222575888 |
| val_rmse_max | 874.1685110901807 |
| val_range_violations_before_fix | 54 |
| test_mae_min | 224.4191169316718 |
| test_rmse_min | 286.98259137636444 |
| test_mae_max | 253.29067785758937 |
| test_rmse_max | 323.0178603027541 |
| test_range_violations_before_fix | 41 |

### 10.2 RF absolute targets (`model_a_rf_abs_metrics.json`)
| metric | value |
|---|---:|
| val_mae_min | 165.50994438947623 |
| val_rmse_min | 729.2948217566039 |
| val_mae_max | 211.5428165615841 |
| val_rmse_max | 934.5921128864004 |
| val_range_violations_before_fix | 0 |
| test_mae_min | 23.771652002577547 |
| test_rmse_min | 46.32132928670161 |
| test_mae_max | 28.588010248339458 |
| test_rmse_max | 51.83592785172331 |
| test_range_violations_before_fix | 1 |

### 10.3 RF percentage targets (`model_a_rf_pct_metrics.json`)
| metric | value |
|---|---:|
| val_mae_min | 13.066202090592334 |
| val_rmse_min | 36.26749247539677 |
| val_mae_max | 18.32752613240418 |
| val_rmse_max | 51.817153738919195 |
| val_range_violations_after_fix | 0 |
| test_mae_min | 9.16030534351145 |
| test_rmse_min | 25.412625340911895 |
| test_mae_max | 12.290076335877863 |
| test_rmse_max | 29.122012345534007 |
| test_range_violations_after_fix | 0 |

### 10.4 RF percentage tier split v2 (`model_a_rf_pct_tier_split_metrics.json`)
| metric | value |
|---|---:|
| train_mae_min | 2.855822550831793 |
| train_rmse_min | 23.62484046244874 |
| train_mae_max | 3.068391866913124 |
| train_rmse_max | 25.871143796339364 |
| train_range_violations_after_fix | 0 |
| val_mae_min | 13.066202090592334 |
| val_rmse_min | 34.69629306731831 |
| val_mae_max | 15.993031358885018 |
| val_rmse_max | 47.53505949857537 |
| val_range_violations_after_fix | 0 |
| test_mae_min | 8.931297709923664 |
| test_rmse_min | 25.06480151146544 |
| test_mae_max | 13.816793893129772 |
| test_rmse_max | 31.634844061234535 |
| test_range_violations_after_fix | 0 |

Key interpretation:
- baseline to v2 shows large MAE reduction,
- postprocessing keeps range violations at zero in v2,
- Tier 5 still has the highest absolute error due price scale, which is expected and managed by fallback logic.

---


## 11) Graphs (regenerated after v2) and detailed interpretation

### 11.1) Parity plot: predicted max vs actual max
File:
- `reports/model_a/charts/model_a_tier_split_parity_val_max.png`

![Model A Parity Plot](reports/model_a/charts/model_a_tier_split_parity_val_max.png)

Quick pointers:
- Each point compares predicted max price to true max price on validation data.
- Points near the diagonal indicate strong agreement between prediction and target.
- Wider spread at the high end is expected because Tier 5 prices magnify small percentage errors.

What it shows:
- each point compares predicted `max_price` with true `target_rule_max` on the validation set.
- the closer the points are to the diagonal line, the better the prediction alignment.

How to interpret it:
- dense clustering near the diagonal means the model captures the main pricing relationship well.
- wider spread is expected for high-value Tier 5 items because a small percentage error becomes a larger rupee error.
- this plot is useful for checking whether the model is biased too high or too low.

Why it matters:
- it gives a direct visual answer to the question: “Are predicted max prices matching target max prices?”

### 11.2) Residual histogram: validation max error distribution
File:
- `reports/model_a/charts/model_a_tier_split_residual_hist_val_max.png`

![Model A Residual Histogram](reports/model_a/charts/model_a_tier_split_residual_hist_val_max.png)

Quick pointers:
- This chart shows how prediction error is distributed around zero.
- A center near zero means the model is not strongly biased high or low overall.
- The tails reveal that only a smaller set of cases remain difficult.

What it shows:
- distribution of `predicted_max - actual_max` error on the validation set.

How to interpret it:
- values near zero mean good predictions.
- a symmetric distribution around zero suggests low systematic bias.
- long tails indicate a small number of harder cases.

Why it matters:
- it shows whether the model is generally overpredicting or underpredicting.
- it also shows whether remaining errors are concentrated in a small tail instead of everywhere.

### 11.3) Tier-wise MAE chart
File:
- `reports/model_a/charts/model_a_tier_split_tier_mae_val_max.png`

![Model A Tier-wise MAE](reports/model_a/charts/model_a_tier_split_tier_mae_val_max.png)

Quick pointers:
- This compares error by tier instead of averaging everything together.
- Higher Tier 5 error is expected because luxury prices have larger absolute scale.
- The chart helps justify why separate Tier 5 handling was necessary.

What it shows:
- average absolute rupee error in `max_price` per tier.

How to interpret it:
- lower tiers should naturally have lower absolute rupee error because prices are smaller.
- Tier 5 usually has the highest absolute error because price magnitude is much larger.

Why it matters:
- it tells us where the model is most difficult to stabilize.
- this is one of the main reasons the Tier 5 split and residual correction were introduced.

### 11.4) Model-comparison chart
File:
- `reports/model_a/charts/model_a_tier_split_comparison_val.png`

![Model A Model Comparison](reports/model_a/charts/model_a_tier_split_comparison_val.png)

Quick pointers:
- This is the improvement story of Model A in one view.
- It shows why the project moved from linear baseline to RF absolute, then RF percentage, then tier-split v2.
- The final bar being best is the evidence behind the final architecture choice.

What it shows:
- comparison of major validation metrics across:
  - linear baseline,
  - RF absolute,
  - RF percentage,
  - RF percentage tier-split v2.

How to interpret it:
- each later stage should reduce validation error and improve stability.
- the chart proves the journey was evidence-based, not random experimentation.

Why it matters:
- this is the clearest graph for explaining why the project moved from baseline to the current final version.

## 12) Route-level validation interpretation (v2)

From `model_a_rf_pct_tier_split_val_route_summary.csv`:

- `tier_split_tier1to4`
  - rows: 230
  - mae_max: 6.4783
  - p90_pct_error_max: 0.090909
  - p95_pct_error_max: 0.162719

- `tier_split_tier5`
  - rows: 57
  - mae_max: 54.3860
  - p90_pct_error_max: 0.041391
  - p95_pct_error_max: 0.048095

Important nuance:
- Tier 5 absolute error is high because amounts are high,
- relative percentile errors are still tight,
- this confirms the need to evaluate both absolute and relative metrics.

---

## 13) Inference pipeline behavior (runtime)

Inference script:
- `code/model_a/inference/predict_price_range_simple_input.py`

Pipeline steps:
1. accept simple input fields,
2. normalize condition label,
3. derive engineered features internally,
4. route to Tier 5 or Tier 1-4 model pair,
5. apply Tier 5 residual correction if applicable,
6. compute model candidate range,
7. compute deterministic rule range,
8. compute confidence score,
9. if confidence below threshold, return rule fallback.

### 13.1 Confidence scoring logic
Start score = 1.0 and deduct:
- unseen brand in route training: -0.35
- original price not user-provided: -0.35
- open-ended brand price band: -0.10
- brand missing in brand master: -0.20
- unusually high predicted range width: -0.15

Threshold:
- 0.55

If score < 0.55:
- final source = `rule_fallback`


### 13.2 Runtime inference flow

File:
- `reports/model_a/model_a_inference_flow.png`

![Model A Inference Flow](reports/model_a/model_a_inference_flow.png)

Quick pointers:
- The user enters only simple listing fields; technical features are derived internally.
- Both model and rule outputs are computed so the system can compare them safely.
- Confidence logic decides whether to trust the model result or return a safe rule fallback.

How to read this diagram:
- user-facing inputs stay simple; hidden engineered fields are derived internally.
- the system routes the request into the correct pricing path based on tier.
- model and rule outputs are both computed so they can be compared.
- confidence logic decides whether the model output is trusted or whether the safe rule fallback should be returned.

Why this matters:
- it explains how Model A can be production-friendly without asking users for technical fields like multipliers or base percentages.
- it also shows why the final system is safer than a plain regression model: the rule engine still protects the output at runtime.

---

## 14) Fresh inference run outputs (v2)

### 14.1 Known Tier 5 brand example
Command:
```bash
python3 code/model_a/inference/predict_price_range_simple_input.py \
  --brand "Prada" --category "Dress" --material "Silk" --age_months 6 \
  --size "M" --condition "Like New" --original_price 95000 --json
```
Result summary:
- route: `tier_split_tier5`
- model output %: min 0.0981, max 0.1254
- final range: **9300 to 11900**
- confidence: 0.9
- source: `model_output`

### 14.2 Known Tier 3 brand example
Input: Zara / Jacket / Denim / 8 months / Like New / 4999
Result summary:
- route: `tier_split_tier1to4`
- final range: **250 to 350**
- confidence: 1.0
- source: `model_output`

### 14.3 Known Tier 5 aged/used example
Input: Anita Dongre / Lehenga / Silk / 12 months / Used / 65000
Result summary:
- route: `tier_split_tier5`
- residual correction applied: true
- final range: **4000 to 5200**
- source: `model_output`

### 14.4 Unknown brand fallback example
Input: Unknown Brand / Top / Cotton / 4 months / Used / 1800
Result summary:
- confidence: 0.45 (<0.55)
- reasons: unseen brand + brand missing in master
- source: `rule_fallback`
- final range: **70 to 80**

---

## 15) Safety constraints guaranteed in v2

After prediction, v2 enforces:
- clamp predicted percentages into [0, 0.20],
- `min <= max` ordering,
- bucket rounding (10/50/100),
- non-negative outputs,
- fallback to deterministic rule range under low confidence.

These constraints keep outputs aligned with business rules even in uncertain cases.

---

## 16) Why this v2 design is technically justified

1. **Rule-derived targets** provide stable supervision when public real rental labels are unavailable.
2. **Percentage targets** improve learning across wide price scales.
3. **Tier split** isolates premium behavior from mass-market behavior.
4. **Residual correction** reduces systematic Tier 5 max errors.
5. **Confidence fallback** prevents low-trust model responses from reaching final output unchecked.

This combination balances:
- predictive quality,
- explainability,
- operational safety.

---

## 17) Limitations and next engineering steps

Current limitations:
- Tier 5 absolute errors remain the largest (expected with high-value items).
- Brand master has open-ended price bands for some premium brands.
- Model A currently uses metadata only (no image input).

Next improvements:
- increase real Tier 5 labeled coverage,
- calibrate residual scaling by brand clusters,
- add uncertainty calibration/quantile layer,
- maintain periodic retraining with accepted listing feedback.

---

## 18) Final status after rerun

- v2 training rerun: completed successfully.
- v2 analysis rerun: completed successfully.
- charts regenerated from latest v2 outputs.
- document synchronized with current formulas, metrics, charts, and inference outputs.

This handbook now reflects the latest model state and can be used as the primary technical submission for Model A.

## 19) Cross-check test cases (2026-03-14)

Cases: 6
Anomalies: 0

| Case | Brand | Category | Condition | Age | Original | Final Min | Final Max | Source | Confidence | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Zara | Top | New | 2 | 1200 | 60 | 70 | model_output | 1.00 |  |
| A2 | H&M | Jeans | Like New | 6 | 2400 | 120 | 140 | model_output | 1.00 |  |
| A3 | Biba | Kurta | Like New | 4 | 4200 | 250 | 300 | model_output | 1.00 |  |
| A4 | Anita Dongre | Lehenga | New | 2 | 22000 | 2800 | 3500 | model_output | 0.90 |  |
| A5 | Uniqlo | Jacket | Used | 12 | 8000 | 400 | 450 | model_output | 1.00 |  |
| A6 | BrandX | Dress | Like New | 3 | 3000 | 200 | 250 | rule_fallback | 0.45 | rule fallback (unknown brand) |

---

