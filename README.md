# RentAFit-ML

![Repo status](https://img.shields.io/badge/status-active-7B1E2B)
![Models](https://img.shields.io/badge/models-A%20%7C%20B%20%7C%20C-2B2B2B)
![Docs](https://img.shields.io/badge/docs-complete-4CAF7D)
![Validation](https://img.shields.io/badge/validation-cross--checked-5A1420)

Machine learning workspace for the **RentAFit** platform.

This repository contains the complete ML layer for RentAFit: pricing, moderation, and recommendation systems, along with the datasets, trained artifacts, validation reports, and detailed technical documentation used to support the platform.

## Overview

### Model suite
- **Model A**: price range prediction for clothing rental listings
- **Model B**: listing moderation and lifecycle decision model
- **Model C**: content-based recommendation system for renter-side suggestions

### Included assets
- training and inference code
- frozen and generated datasets
- trained model artifacts
- validation reports and charts
- full model handbooks in Markdown and Word format

## Repository structure

```text
code/
  api/
  model_a/
  model_b/
  model_c/
  shared/
  validation/

data/
  frozen/
  generated/

docs/
  model_a/
  model_b/
  model_c/

models/
  model_a/
  model_b/
  model_c/

reports/
  model_a/
  model_b/
  model_c/
  validation/
```

## Highlights

### Model A
- tier-aware Random Forest pricing pipeline
- range prediction with fallback safeguards
- documented training path from baseline to final model

### Model B
- hybrid LSTM + tabular moderation model
- lifecycle-aware status logic for stale listings
- gender-aware moderation policy and validation

### Model C
- policy-aware content recommender
- category, gender, size, budget, and quality constraints
- ranked top-k output with explanation tags

## Visual previews

### Model A pricing flow
![Model A training architecture](reports/model_a/model_a_training_architecture.png)

### Model B hybrid moderation architecture
![Model B hybrid architecture](reports/model_b/model_b_hybrid_architecture.png)

### Model C recommendation architecture
![Model C architecture](reports/model_c/model_c_architecture.png)

## Key results

| Model | Focus | Current headline result |
| --- | --- | --- |
| Model A | Price range prediction | Stable tier-split pricing pipeline with safe fallback behavior |
| Model B | Moderation | Test macro F1 `0.9765` |
| Model C | Recommendation | Strong policy-aware ranking over random baseline |

## Documentation

### Core handbooks
- `docs/model_a/Model_A_Holy_Book.docx`
- `docs/model_b/Model_B_Master_Document.docx`
- `docs/model_c/Model_C_Master_Document.docx`

### Validation
- `reports/validation/model_crosscheck_report.md`
- `docs/model_c/MODEL_C_VALIDATION_REPORT.docx`

### API layer
- `code/api/app.py`

## Reproducibility notes

This repository includes:
- frozen datasets used to build and validate the models
- generated intermediate datasets required by the current training flows
- trained model files used by the present inference pipelines

These assets were intentionally kept in the repository because they remain within manageable GitHub limits and make the ML work easier to review and reproduce.

## Relationship to the main platform repository

This repository contains the ML layer only.

The separate main platform repository is intended to hold:
- frontend
- backend
- platform integration
- product flow implementation

## Current status

At the current snapshot:
- Model A is implemented, validated, and documented
- Model B is implemented, validated, and documented
- Model C is implemented, validated, and documented
- combined cross-check validation across all three models has been completed
