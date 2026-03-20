# RentAFit-ML

Machine learning and AI research workspace for the **RentAFit** platform.

This repository contains the ML side of the project, separated from the main integrated product repo for cleaner presentation.

## What this repository includes

### Core models
- **Model A**: price range prediction for clothing rental listings
- **Model B**: listing moderation and lifecycle decision model
- **Model C**: recommendation system for renter-side item suggestions

### Experimental module
- **Virtual Try-On**: CatVTON planning and local integration preparation

### Supporting assets
- training and inference code
- generated datasets
- frozen datasets
- trained model artifacts
- charts, evaluation outputs, and validation reports
- detailed model handbooks in Markdown and Word

## Repository structure

```text
code/
  api/
  model_a/
  model_b/
  model_c/
  shared/
  validation/
  virtual_tryon/

data/
  frozen/
  generated/
  virtual_tryon/

docs/
  model_a/
  model_b/
  model_c/
  virtual_tryon/

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

## Key documentation

### Model handbooks
- `docs/model_a/Model_A_Holy_Book.docx`
- `docs/model_b/Model_B_Master_Document.docx`
- `docs/model_c/Model_C_Master_Document.docx`

### Validation
- `reports/validation/model_crosscheck_report.md`
- `docs/model_c/MODEL_C_VALIDATION_REPORT.docx`

### Virtual try-on
- `docs/virtual_tryon/VIRTUAL_TRYON_PLAN.md`
- `docs/virtual_tryon/Virtual_TryOn_CatVTON_Windows_Setup_Guide.pdf`

## ML API

The Python ML service used for integration lives in:

- `code/api/app.py`

This exposes endpoints for:
- Model A prediction
- Model B moderation
- Model C recommendation

## Notes on datasets and artifacts

This repository includes:
- frozen datasets used for the project
- generated intermediate datasets needed for Model B and Model C
- trained model files currently used by the project

The included files were kept because they are still within manageable GitHub size limits and make the ML work easier to review and reproduce.

## Relationship to the main platform repo

This repository is the **ML/research repo**.

The separate main application repo is intended to hold:
- frontend
- backend
- platform integration
- product flow implementation

## Current status

At the time of upload:
- Model A is implemented, validated, and documented
- Model B is implemented, validated, and documented
- Model C is implemented, validated, and documented
- cross-check validation across all three models has been completed
- virtual try-on is documented as the next experimental AI module
