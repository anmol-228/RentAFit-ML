from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_a.inference.predict_price_range_simple_input import predict_one as predict_model_a
from model_b.runtime import predict_one as predict_model_b
from model_c.runtime import recommend_from_item, recommend_from_profile, load_artifacts

app = FastAPI(title='RentAFit ML API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class ModelAPredictRequest(BaseModel):
    brand: str
    category: str
    material: str
    size: str
    condition: str
    age_months: int = Field(..., ge=0)
    original_price: float = Field(..., gt=0)


class ModelBPredictRequest(BaseModel):
    brand: str
    category: str
    gender: Optional[str] = None
    material: str
    size: str
    condition: str
    garment_age_months: int = Field(..., ge=0)
    original_price: float = Field(..., gt=0)
    provider_price: float = Field(..., ge=0)
    current_status: Optional[str] = 'PENDING_REVIEW'
    listing_created_at: Optional[str] = None
    last_approved_at: Optional[str] = None
    last_reapproved_at: Optional[str] = None
    as_of_date: Optional[str] = None
    auto_remove_stale: bool = False
    removal_grace_months: int = 3


class ModelCRecommendRequest(BaseModel):
    seed_item_id: Optional[str] = None
    liked_item_ids: Optional[List[str]] = None
    top_k: int = Field(5, ge=1, le=20)
    category_filter: Optional[str] = None
    max_provider_price: Optional[float] = None
    exclude_same_brand: bool = False


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/api/predict-price')
async def predict_price(payload: ModelAPredictRequest):
    try:
        result = predict_model_a(
            brand=payload.brand,
            category=payload.category,
            material=payload.material,
            age_months=payload.age_months,
            size=payload.size,
            condition=payload.condition,
            original_price=payload.original_price,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    final_range = result['final_price_range']
    min_price = float(final_range['min_price'])
    max_price = float(final_range['max_price'])
    predicted_mid = round((min_price + max_price) / 2.0, 2)

    return {
        'model': 'ModelA',
        'final_price_range': final_range,
        'predicted_price_mid': predicted_mid,
        'confidence': result.get('confidence'),
        'candidate_ranges': result.get('candidate_ranges'),
        'model_route': result.get('model_route'),
    }


@app.post('/api/model-b/predict')
async def predict_model_b_endpoint(payload: ModelBPredictRequest):
    try:
        result = predict_model_b(
            brand=payload.brand,
            category=payload.category,
            gender=payload.gender,
            material=payload.material,
            size=payload.size,
            condition=payload.condition,
            garment_age_months=payload.garment_age_months,
            original_price=payload.original_price,
            provider_price=payload.provider_price,
            current_status=payload.current_status,
            listing_created_at=payload.listing_created_at,
            last_approved_at=payload.last_approved_at,
            last_reapproved_at=payload.last_reapproved_at,
            as_of_date=payload.as_of_date,
            auto_remove_stale=payload.auto_remove_stale,
            removal_grace_months=payload.removal_grace_months,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        'model': 'ModelB',
        'prediction': result.get('prediction'),
        'lifecycle': result.get('lifecycle'),
        'age_context': result.get('age_context'),
        'derived_features': result.get('derived_features'),
    }


@app.post('/api/model-c/recommend')
async def recommend_model_c(payload: ModelCRecommendRequest):
    if payload.seed_item_id:
        try:
            result = recommend_from_item(
                seed_item_id=payload.seed_item_id,
                top_k=payload.top_k,
                category_filter=payload.category_filter,
                max_provider_price=payload.max_provider_price,
                exclude_same_brand=payload.exclude_same_brand,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return result

    if payload.liked_item_ids:
        try:
            result = recommend_from_profile(
                liked_item_ids=payload.liked_item_ids,
                top_k=payload.top_k,
                category_filter=payload.category_filter,
                max_provider_price=payload.max_provider_price,
                exclude_same_brand=payload.exclude_same_brand,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return result

    raise HTTPException(status_code=400, detail='Provide seed_item_id or liked_item_ids')


@app.get('/api/model-c/samples')
async def model_c_samples():
    loaded = load_artifacts()
    catalog = loaded['catalog']
    sample = catalog.head(12)[['listing_id', 'brand', 'category', 'material', 'size', 'tier_primary']]
    return {
        'count': int(len(sample)),
        'items': sample.to_dict(orient='records'),
    }
