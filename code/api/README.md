# RentAFit ML API

This folder exposes Model A, Model B, and Model C via a simple FastAPI service.

## Endpoints

- `GET /health`
- `POST /api/predict-price` (Model A)
- `POST /api/model-b/predict` (Model B)
- `POST /api/model-c/recommend` (Model C)
- `GET /api/model-c/samples` (sample listing IDs for demo)

## Run locally

```bash
cd code/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

If you prefer to work from the repository root instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn code.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Example payloads

Model A:
```json
{
  "brand": "Zara",
  "category": "Top",
  "material": "Cotton",
  "size": "S",
  "condition": "Like New",
  "age_months": 3,
  "original_price": 2599
}
```

Model B:
```json
{
  "brand": "Zara",
  "category": "Top",
  "material": "Cotton",
  "size": "S",
  "condition": "Like New",
  "garment_age_months": 3,
  "original_price": 2599,
  "provider_price": 180,
  "listing_created_at": "2026-02-01",
  "as_of_date": "2026-03-14"
}
```

Model C:
```json
{
  "seed_item_id": "L0008",
  "top_k": 5
}
```

## Notes

- The API depends on the full ML stack, not only FastAPI.
- Model B requires `torch` because the moderation pipeline uses a PyTorch LSTM + tabular model.
