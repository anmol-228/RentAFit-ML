# RentAFit Model C Handbook

## 1. Purpose Of This Document

This handbook explains the final **Model C** recommendation system used in RentAFit.
It is written to be detailed enough for project defense, but clear enough that a student can learn how the recommender works from start to finish.

Model C is the renter-side model.
Its job is to recommend relevant clothes from the live catalog in a way that is:

- explainable,
- policy-aware,
- safe,
- size-conscious,
- gender-aware,
- and practical without real renter click history.

---

## 2. What Model C Does

Model C recommends clothes in two ways:

1. **Viewed item -> similar items**
2. **Liked items -> profile-based recommendations**

So if a renter opens one item, Model C finds similar catalog items.
If a renter likes multiple items, Model C builds a simple preference profile and recommends based on that combined preference.

The current v1 system is a **content-based recommender**.

That means it recommends using item features such as:

- category,
- gender,
- size,
- material,
- brand,
- tier,
- condition,
- price,
- quality,
- freshness.

It does **not** depend on real renter behavior logs yet.

---

## 3. Why A Content-Based Recommender Was Chosen

This was the right starting point because RentAFit currently has:

- a structured clothing catalog,
- pricing and quality features,
- moderation signals from Model B,
- but no real renter click / like / order history at scale.

If we tried to build collaborative filtering first, we would need real user-item interaction logs.
That data does not exist yet in a trustworthy form.

So the practical and honest v1 choice is:

- build recommendations from the **content of the items themselves**
- keep the logic explainable
- support a clean upgrade path later when real renter history exists

That is why Model C is content-based first, while still leaving room for:

- liked-item profile recommendations later,
- hybrid recommendation later,
- collaborative filtering later.

---

## 4. Final Policy Decisions For Model C

These are the final rules chosen for Model C.

### 4.1 Main recommendation goal
- recommend similar clothes from a selected item
- support liked-item based recommendations later

### 4.2 Core modeling approach
- content-based recommender using catalog metadata and quality context

### 4.3 Gender rule
- gender is a **hard compatibility filter**
- `Women` query can see `Women` and `Unisex`
- `Men` query can see `Men` and `Unisex`
- `Unisex` query sees only `Unisex`

### 4.4 Category rule
- same category only in v1

### 4.5 Size rule
- exact same size first
- nearest size next
- extended size can still appear if the pool is too small

### 4.6 Safety rule
- primary recommendations come from safe approved items
- up to **2** `Review` items may be used only as fallback if otherwise strong matches exist

### 4.7 Budget rule
- if renter gives explicit budget, use that
- otherwise use **category-average provider price** as the default budget reference

### 4.8 Output rule
- top `5` recommendations
- include reason tags explaining why each item was selected

### 4.9 Evaluation rule
- use **proxy metrics**
- because real renter interaction logs do not exist yet

---

## 5. What "Freshness" Means

Freshness was not a manually typed raw dataset field.

It is a **derived feature** built from `age_months`.

The intuition is simple:

- newer item -> higher freshness
- older item -> lower freshness

Current derived feature:
- `item_freshness_score`

This helps Model C prefer items that feel more current and rental-ready.

So freshness is basically:
- a normalized "how recent / less old is this item?" signal

---

## 6. Data Used By Model C

Primary source file:
- `data/frozen/v2_gender/pricing_features_augmented_1500.csv`

Generated catalog files:
- `data/generated/model_c_catalog.csv`
- `data/generated/model_c_catalog_recommendable.csv`

Saved training artifacts:
- `models/model_c/content_based/model_c_preprocessor.joblib`
- `models/model_c/content_based/model_c_nearest_neighbors.joblib`
- `models/model_c/content_based/model_c_feature_matrix.joblib`
- `models/model_c/content_based/model_c_catalog_recommendable.csv`
- `models/model_c/content_based/model_c_metadata.json`

Current dataset counts:
- source rows: `1500`
- full catalog rows: `1500`
- recommendable rows: `932`
- filtered out rows: `568`
- recommendable rate: `0.6213`

Current pool counts:
- `PRIMARY_APPROVE: 529`
- `REVIEW_FALLBACK: 403`
- `FILTERED_OUT: 568`

Current recommendable gender counts:
- `Unisex: 575`
- `Women: 357`

Current recommendable size counts:
- `M: 357`
- `S: 243`
- `L: 215`
- `XS: 60`
- `XL: 57`

---

## 7. Why Model B Is Part Of Model C

Model C does not recommend items blindly.

Before an item becomes recommendable, Model C uses **Model B moderation output** to help decide whether the item belongs in:

- the main safe pool,
- the review fallback pool,
- or the filtered-out pool.

Columns added during catalog preparation:
- `model_b_predicted_decision`
- `model_b_approve_probability`
- `model_b_review_probability`
- `model_b_reject_probability`
- `recommendation_pool_status`
- `recommendable_flag`
- `review_fallback_eligible`

This is important because recommendation quality is not just "find similar items".
It is also:

- avoid weak items,
- avoid risky items,
- keep the renter-facing catalog safer.

So Model C is not isolated from the rest of the ML system.
It is deliberately linked with Model B.

---

## 8. Recommendation Pool Rule

Model C first builds a catalog and then classifies items into recommendation pools.

### 8.1 Primary recommendation pool
An item goes into `PRIMARY_APPROVE` when:

- Model B predicts `Approve`
- and `rule_quality_score >= 55`

### 8.2 Review fallback pool
An item goes into `REVIEW_FALLBACK` when:

- Model B predicts `Review`
- and `rule_quality_score >= 60`
- and `deviation_M <= 40`

### 8.3 Filtered out pool
An item becomes `FILTERED_OUT` when:

- `provider_price <= 0`
- or `original_price <= 0`
- or it fails the pool rules above

This policy means:

- approved and strong items come first,
- review items are available only as a controlled fallback,
- clearly weak items are excluded entirely.

---

## 9. Data Pipeline Diagram

Image file:
- `reports/model_c/model_c_data_pipeline.png`

![Model C Data Pipeline](reports/model_c/model_c_data_pipeline.png)

Quick pointers:
- the v2 gender-aware pricing dataset is the starting point
- Model B moderation is used during catalog preparation
- the output is split into full catalog vs recommendable catalog
- training artifacts are then built from the recommendable pool
- charts and reports are generated after the artifacts are ready

Why this matters:
- it shows that Model C is a full pipeline, not just one ranking script

---

## 10. Feature Engineering Used In Model C

### 10.1 Categorical features
- `brand`
- `category`
- `gender`
- `material`
- `size`
- `condition`
- `tier_primary`

These are one-hot encoded.

### 10.2 Numeric features
- `age_months`
- `original_price`
- `provider_price`
- `rule_mid`
- `rule_quality_score`
- `quality_score_norm`
- `item_freshness_score`
- `provider_price_pct_of_original`
- `category_avg_provider_price`
- `price_vs_category_avg_ratio`
- `catalog_priority_score`
- `model_b_approve_probability`
- `model_b_review_probability`

These are median-imputed and standardized.

### 10.3 Derived recommendation features
These are not all used inside the preprocessor, but they are important during policy and re-ranking:

- `budget_alignment_score`
- `gender_match_label`
- `size_match_label`
- `recommendation_pool_status`

### 10.4 Why these features make sense

This feature set helps the recommender understand:

- what the item is,
- how premium it is,
- whether it is safe enough to recommend,
- whether it is fresh or stale,
- whether it fits expected price behavior,
- whether it belongs near the renter's size and budget zone.

---

## 11. Architecture Diagram

Image file:
- `reports/model_c/model_c_architecture.png`

![Model C Architecture](reports/model_c/model_c_architecture.png)

Quick pointers:
- the recommendable catalog is encoded into vectors
- a cosine nearest-neighbor index retrieves similar items
- hard policy filters are applied after retrieval
- re-ranking then uses size, budget, and safety signals
- limited review fallback is applied only at the end

Why this matters:
- it shows that Model C is not only nearest-neighbor retrieval
- it is a retrieval + policy + re-ranking system

---

## 12. Recommendation Flow Diagram

Image file:
- `reports/model_c/model_c_similarity_flow.png`

![Model C Recommendation Flow](reports/model_c/model_c_similarity_flow.png)

Quick pointers:
- similarity retrieval gives a candidate set first
- category and gender are enforced as hard filters
- exact size is preferred before nearest size
- budget and safety then reshape the ranking
- review fallback is capped at 2 items

Why this matters:
- it makes the runtime policy easy to understand step by step

---

## 13. How Training Works In Model C

Model C does not train a deep neural network.

Instead, it builds reusable recommender artifacts:

1. fit the preprocessor on the recommendable catalog
2. transform the catalog into an item feature matrix
3. fit a cosine `NearestNeighbors` model
4. save artifacts for runtime retrieval

Main scripts:
- `code/model_c/prepare_model_c_catalog.py`
- `code/model_c/training/train_model_c_content_based.py`

Saved metadata currently says:
- model type: `content_based_cosine_similarity`
- catalog rows: `932`
- neighbors fit: `150`

---

## 14. Recommendation Scoring Logic

Model C is a **two-stage recommender**:

1. **retrieve candidates by cosine similarity**
2. **re-rank candidates using policy-aware signals**

Current re-ranking weights:
- base similarity weight: `0.55`
- quality weight: `0.15`
- freshness weight: `0.08`
- budget alignment weight: `0.10`
- size match weight: `0.07`
- safety weight: `0.05`

### Why these weights make sense

- similarity remains the most important signal
- quality matters because renters should see better items first
- freshness matters because newer items are usually more attractive
- budget alignment matters because recommendations should stay realistic
- size matters because fit matters
- safety matters because primary approved items should outrank fallback review items

### Important clarification about budget

When the renter does **not** provide an explicit budget, Model C uses:

- `category_average_budget`

This is a **soft ranking signal**, not a hard removal rule.

That means:
- a luxury item can still rank well if similarity, size, quality, and tier are strong
- but price distance from the category average will reduce its budget alignment score

This behavior is intentional.

---

## 15. Query Modes

### 15.1 Item-to-item recommendation

Example:
```bash
python3 code/model_c/inference/recommend_model_c_items.py \
  --seed_item_id L0015 --top_k 5 --json
```

### 15.2 Profile-from-liked-items recommendation

Example:
```bash
python3 code/model_c/inference/recommend_model_c_items.py \
  --liked_item_ids L0015,L0021,L0100 --top_k 5 --json
```

In profile mode, Model C builds a small preference summary from the liked items, such as:

- dominant category,
- dominant gender,
- dominant material,
- dominant size,
- dominant brand,
- dominant tier,
- average provider price.

---

## 16. Runtime Policy Rules In Plain Language

At runtime, Model C follows this logic:

1. find candidate items using similarity
2. keep only same-category items
3. enforce gender compatibility
4. prefer exact size matches
5. then allow nearest sizes
6. use category-average or explicit budget for budget alignment
7. rank primary approved items first
8. allow up to 2 review fallback items if needed
9. return top 5 items with reason tags

This is exactly why the system feels safer and more intentional than a plain nearest-neighbor list.

---

## 17. Evaluation Method

Real renter interaction logs do not yet exist.
So Model C is evaluated using **policy-aware proxy metrics**.

That means we check whether recommended items are:

- same category,
- gender compatible,
- size compatible,
- materially similar,
- tier aligned,
- budget aligned,
- high enough quality,
- retrieved with strong similarity.

We also compare against a **policy-aware random baseline**.

This is important:
- the random baseline uses the same hard filters
- so metrics forced by policy may become identical by design
- the more meaningful comparison is on ranking-sensitive metrics such as:
  - material match
  - tier match
  - similarity
  - budget alignment
  - final score
  - average quality

---

## 18. Current Metrics

From:
- `reports/model_c/model_c_proxy_metrics.json`

Current Model C proxy metrics:
- `fill_rate_at_5 = 0.9350`
- `same_category_at_5 = 1.0000`
- `gender_compatible_at_5 = 1.0000`
- `exact_size_at_5 = 0.6560`
- `size_compatible_at_5 = 0.8881`
- `same_material_at_5 = 0.8531`
- `same_tier_at_5 = 0.5563`
- `avg_rule_quality_score_top5 = 84.1663`
- `avg_similarity_score_top5 = 0.6731`
- `avg_budget_alignment_top5 = 0.6900`
- `avg_final_score_top5 = 0.7331`
- `review_fallback_rate_top5 = 0.1296`

From:
- `reports/model_c/model_c_proxy_vs_random_metrics.json`

Policy-aware random baseline:
- `fill_rate_at_5 = 0.9350`
- `same_category_at_5 = 1.0000`
- `gender_compatible_at_5 = 1.0000`
- `exact_size_at_5 = 0.6560`
- `size_compatible_at_5 = 0.8881`
- `same_material_at_5 = 0.8069`
- `same_tier_at_5 = 0.5313`
- `avg_rule_quality_score_top5 = 83.5606`
- `avg_similarity_score_top5 = 0.6247`
- `avg_budget_alignment_top5 = 0.6727`
- `avg_final_score_top5 = 0.7023`
- `review_fallback_rate_top5 = 0.1296`

### What these numbers mean

Some metrics are identical because:

- category is a hard filter
- gender is a hard filter
- size behavior is heavily shaped by policy before ranking

So the real evidence that Model C is better than random is:

- better material alignment
- better tier alignment
- higher average quality
- higher average similarity
- better budget alignment
- higher final ranking score

That is exactly what we want from a ranking model.

---

## 19. Proxy Metrics Chart

Image file:
- `reports/model_c/model_c_proxy_metrics_chart.png`

![Model C Proxy Metrics](reports/model_c/model_c_proxy_metrics_chart.png)

Quick pointers:
- fill rate shows how often the model can fill the top 5 list
- gender compatibility stays perfect because of hard policy filtering
- exact size is lower than total size compatibility because nearest size fallback is allowed
- budget alignment and similarity are both reasonably strong
- average quality stays high, meaning recommendations are not low-grade items

---

## 20. Model C vs Random Baseline Chart

Image file:
- `reports/model_c/model_c_proxy_vs_random_chart.png`

![Model C vs Random Baseline](reports/model_c/model_c_proxy_vs_random_chart.png)

Quick pointers:
- comparison focuses on ranking-sensitive metrics
- the gain over random is strongest where real ranking matters
- Model C beats the random baseline on:
  - material similarity
  - tier similarity
  - budget alignment
  - average similarity
  - average final score
  - average quality

Why this matters:
- it proves the recommender is doing more than policy filtering
- it is actually ranking better candidates higher

---

## 21. Catalog Distribution Chart

Image file:
- `reports/model_c/model_c_catalog_distribution_chart.png`

![Model C Catalog Distribution](reports/model_c/model_c_catalog_distribution_chart.png)

Quick pointers:
- the chart shows the top categories in the recommendable pool
- it also shows gender balance in the pool
- it visualizes how many items are primary vs review fallback

Why this matters:
- it helps explain the actual recommendation search space before ranking happens

---

## 22. Real Smoke-Test Examples

### 22.1 Women-item query
Command used:
```bash
python3 code/model_c/inference/recommend_model_c_items.py --seed_item_id L0015 --top_k 5 --json
```

Observed behavior:
- category stayed `Dress`
- gender stayed `Women`
- exact size matches were preferred
- no review fallback was needed

### 22.2 Unisex-item query
Command used:
```bash
python3 code/model_c/inference/recommend_model_c_items.py --seed_item_id L0037 --top_k 5 --json
```

Observed behavior:
- category stayed `Shirt`
- gender stayed `Unisex`
- exact size matches appeared first when available
- extended sizes appeared only when the pool needed them

### 22.3 Profile query
Command used:
```bash
python3 code/model_c/inference/recommend_model_c_items.py --liked_item_ids L0015,L0021,L0100 --top_k 5 --json
```

Observed behavior:
- the profile summarized into `Dress`, `Women`, `Silk`, `L`, `Tier 5`
- the recommender respected category and gender
- exact size and extended size both appeared depending on availability

### 22.4 Review fallback edge case
Command used:
```bash
python3 code/model_c/inference/recommend_model_c_items.py --seed_item_id L0080 --top_k 5 --json
```

Observed behavior:
- only `2` results were returned
- both were `REVIEW_FALLBACK`
- `policy_summary.review_items_used = 2`

This proves:
- the review fallback cap is respected
- the system does not fill the list with weak items just to reach 5 results

---

## 23. Main Code Files

### Preparation
- `code/model_c/prepare_model_c_catalog.py`

### Policy helpers
- `code/model_c/policy.py`

### Training / artifact building
- `code/model_c/training/train_model_c_content_based.py`

### Runtime
- `code/model_c/runtime.py`

### Inference
- `code/model_c/inference/recommend_model_c_items.py`

### Analysis
- `code/model_c/analysis/model_c_report.py`

---

## 24. Why Model C Is Strong For This Stage

Model C is strong for the current stage of the project because it is:

- explainable,
- policy-aware,
- connected with Model B moderation,
- gender-aware,
- size-aware,
- budget-aware,
- and already able to serve both item and profile queries.

It is not pretending to be something it is not.

It is honestly a **well-engineered content-based recommender**, which is exactly the right v1 choice for the available data.

---

## 25. Honest Limitations

Model C still has important limitations:

1. it does not yet use real renter click / like / booking logs
2. proxy metrics are not the same as true production precision@K
3. budget defaults use category averages, which can underrepresent luxury seed preferences
4. the system is currently metadata-driven, not image-driven
5. broader cross-category styling recommendations are intentionally disabled in v1

These are acceptable limitations for a student project baseline.

---

## 26. Best Future Upgrades

Natural future upgrades are:

1. add real renter interaction logs
2. move to a hybrid recommender
3. personalize with renter style history
4. allow related-category recommendations
5. use image embeddings later for style similarity
6. integrate explicit budget and occasion filters in the frontend

---

## 27. Final Summary

Model C is the renter-side recommendation model for RentAFit.

It:

- builds a moderated and filtered recommendation pool,
- represents items using content features,
- retrieves candidates by cosine similarity,
- applies hard category and gender rules,
- prioritizes exact size then nearest size,
- uses category-average or explicit budget alignment,
- allows at most 2 review fallback items,
- and returns top 5 explainable recommendations.

That makes it a complete, defensible, and practical v1 recommender for the current project.
