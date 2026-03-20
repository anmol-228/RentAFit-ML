# Model C (Policy-Aware Content-Based Recommendation)

Model C is the renter-side recommendation system for RentAFit.

It recommends:

- similar items from a viewed item
- recommendations from a small liked-item profile

The current v1 is a **content-based recommender** built from catalog metadata, quality context, moderation signals, budget alignment, gender compatibility, and size matching.

## Final Model C policy

- same category only
- hard gender compatibility
- exact size first, then nearest size
- explicit user budget if provided
- otherwise use category-average budget
- primary approved items first
- allow up to 2 review-fallback items only if needed
- return top 5 recommendations with reason tags

## Why this design was chosen

RentAFit does not yet have real renter interaction logs.
So the most honest v1 recommender is content-based:

- it is explainable
- it works with current data
- it is easy to defend academically
- it can later be upgraded into a hybrid recommender

## Main files

### Policy helpers
- `code/model_c/policy.py`

### Catalog preparation
- `code/model_c/prepare_model_c_catalog.py`

### Training / artifact building
- `code/model_c/training/train_model_c_content_based.py`

### Runtime
- `code/model_c/runtime.py`

### Inference
- `code/model_c/inference/recommend_model_c_items.py`

### Analysis / reporting
- `code/model_c/analysis/model_c_report.py`

## Main data outputs

- `data/generated/model_c_catalog.csv`
- `data/generated/model_c_catalog_recommendable.csv`

## Saved artifacts

- `models/model_c/content_based/model_c_preprocessor.joblib`
- `models/model_c/content_based/model_c_nearest_neighbors.joblib`
- `models/model_c/content_based/model_c_feature_matrix.joblib`
- `models/model_c/content_based/model_c_catalog_recommendable.csv`
- `models/model_c/content_based/model_c_metadata.json`

## Current artifact summary

From the current saved metadata:

- model type: `content_based_cosine_similarity`
- recommendable catalog rows: `932`
- neighbors fit: `150`

Current ranking weights:

- base similarity: `0.55`
- quality: `0.15`
- freshness: `0.08`
- budget alignment: `0.10`
- size match: `0.07`
- safety: `0.05`

## Current evaluation highlights

Proxy metrics:

- `fill_rate_at_5 = 0.9350`
- `gender_compatible_at_5 = 1.0000`
- `exact_size_at_5 = 0.6560`
- `size_compatible_at_5 = 0.8881`
- `same_material_at_5 = 0.8531`
- `avg_similarity_score_top5 = 0.6731`
- `avg_budget_alignment_top5 = 0.6900`
- `avg_final_score_top5 = 0.7331`

Policy-aware random baseline:

- `same_material_at_5 = 0.8069`
- `avg_similarity_score_top5 = 0.6247`
- `avg_budget_alignment_top5 = 0.6727`
- `avg_final_score_top5 = 0.7023`

This means the model is doing more than just rule filtering.
It is ranking stronger candidates higher than a random selector working under the same hard constraints.

## Reports and visuals

- `reports/model_c/model_c_catalog_distribution_chart.png`
- `reports/model_c/model_c_proxy_metrics_chart.png`
- `reports/model_c/model_c_proxy_vs_random_chart.png`
- `reports/model_c/model_c_similarity_flow.png`
- `reports/model_c/model_c_data_pipeline.png`
- `reports/model_c/model_c_architecture.png`

## Commands

Rebuild Model C end to end:

```bash
python3 code/model_c/prepare_model_c_catalog.py
python3 code/model_c/training/train_model_c_content_based.py
python3 code/model_c/analysis/model_c_report.py
```

Run item-to-item inference:

```bash
python3 code/model_c/inference/recommend_model_c_items.py --seed_item_id L0015 --top_k 5 --json
```

Run profile-from-liked-items inference:

```bash
python3 code/model_c/inference/recommend_model_c_items.py --liked_item_ids L0015,L0021,L0100 --top_k 5 --json
```

Sample item-to-item run highlights:

```json
{
  "query_mode": "item_to_item",
  "seed_item": {
    "listing_id": "L0015",
    "brand": "Prada",
    "category": "Dress"
  },
  "policy_summary": {
    "same_category_only": true,
    "query_gender": "Women",
    "query_size": "S"
  }
}
```

## Important honest limitation

Model C is still a v1 recommender.

It does **not** yet use:

- real renter click logs
- booking history
- image embeddings
- collaborative filtering

That is acceptable for the current project stage.
