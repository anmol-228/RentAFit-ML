from __future__ import annotations

from pathlib import Path
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from pricing_rules import compute_pricing_fields
from model_b.lifecycle import (
    DEFAULT_REMOVAL_GRACE_MONTHS,
    DEFAULT_STALE_THRESHOLD_MONTHS,
    derive_listing_age_context,
    effective_status_from_prediction,
)
from shared.gender_utils import derive_gender, gender_conflict_flag as compute_gender_conflict_flag

BASE = Path('/Users/mypc/RentAFit')
MODEL_PATH = BASE / 'models/model_b/model_b_lstm.pt'
PREPROCESSOR_PATH = BASE / 'models/model_b/model_b_tabular_preprocessor.joblib'

CLASS_NAMES = ['Approve', 'Review', 'Reject']
CAT_COLS = ['brand', 'category', 'gender', 'material', 'size', 'tier_primary']
NUM_COLS = [
    'age_months', 'older_listing_flag', 'age_policy_override_applied',
    'gender_conflict_flag', 'gender_policy_override_applied',
    'original_price', 'provider_price', 'deviation_M',
    'condition_penalty', 'age_penalty', 'deviation_penalty', 'total_penalty'
]
SEQ_COLS = ['condition_token', 'age_bin_token']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONDITION_CANONICAL = {
    'new': 'New',
    'like new': 'Like New',
    'likenew': 'Like New',
    'used': 'Used',
}


class ModelBHybridLSTM(nn.Module):
    def __init__(self, tabular_dim: int, vocab_size: int = 8, embed_dim: int = 8, lstm_hidden: int = 16, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, batch_first=True)
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, num_classes),
        )

    def forward(self, seq_x, tab_x):
        emb = self.embedding(seq_x)
        _, (h_n, _) = self.lstm(emb)
        seq_out = h_n[-1]
        tab_out = self.tabular_branch(tab_x)
        combined = torch.cat([seq_out, tab_out], dim=1)
        return self.head(combined)



def normalize_condition(value: str) -> str:
    key = str(value).strip().lower()
    return CONDITION_CANONICAL.get(key, str(value).strip())


def age_bin_token(age_months: float) -> int:
    age = float(age_months)
    if age <= 3:
        return 1
    if age <= 6:
        return 2
    if age <= 9:
        return 3
    return 4


def condition_token(condition: str) -> int:
    condition_norm = normalize_condition(condition)
    if condition_norm == 'New':
        return 1
    if condition_norm == 'Like New':
        return 2
    return 3


def older_listing_flag_from_model_age(age_months: float) -> int:
    return int(float(age_months) >= 10)


def build_feature_row(brand, category, material, size, condition, model_input_age_months, original_price, provider_price, gender=None):
    condition_norm = normalize_condition(condition)
    resolved_gender, gender_source = derive_gender(explicit_gender=gender, category=category)
    gender_conflict = int(compute_gender_conflict_flag(resolved_gender, category))
    rule = compute_pricing_fields(
        brand=brand,
        category=category,
        material=material,
        size=size,
        condition=condition_norm,
        age_months=int(model_input_age_months),
        original_price=int(original_price),
        provider_price=int(provider_price),
    )

    older_flag = older_listing_flag_from_model_age(model_input_age_months)
    age_override = int(older_flag == 1 and rule['rule_decision'] == 'Approve')
    gender_policy_override = int(gender_conflict == 1 and rule['rule_decision'] != 'Reject')

    return {
        'brand': brand,
        'category': category,
        'gender': resolved_gender,
        'gender_source': gender_source,
        'gender_conflict_flag': gender_conflict,
        'gender_policy_override_applied': gender_policy_override,
        'material': material,
        'size': size,
        'condition': condition_norm,
        'condition_token': condition_token(condition_norm),
        'model_input_age_months': int(model_input_age_months),
        'age_months': int(model_input_age_months),
        'age_bin_token': age_bin_token(model_input_age_months),
        'older_listing_flag': older_flag,
        'age_policy_override_applied': age_override,
        'original_price': int(original_price),
        'provider_price': int(provider_price),
        'tier_primary': rule['tier_primary'],
        'deviation_M': rule['deviation_M'],
        'condition_penalty': rule['condition_penalty'],
        'age_penalty': rule['age_penalty'],
        'deviation_penalty': rule['deviation_penalty'],
        'total_penalty': rule['total_penalty'],
        'rule_quality_score': rule['rule_quality_score'],
        'rule_decision': rule['rule_decision'],
    }



def load_artifacts():
    payload = torch.load(MODEL_PATH, map_location=DEVICE)
    pre = joblib.load(PREPROCESSOR_PATH)
    model = ModelBHybridLSTM(tabular_dim=payload['tabular_dim']).to(DEVICE)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    return {
        'model': model,
        'preprocessor': pre,
        'payload': payload,
    }



def predict_one(
    brand: str,
    category: str,
    material: str,
    size: str,
    condition: str,
    original_price: float,
    provider_price: float,
    gender: Optional[str] = None,
    age_months: Optional[int] = None,
    garment_age_months: Optional[int] = None,
    current_status: Optional[str] = None,
    listing_created_at: Optional[str] = None,
    last_approved_at: Optional[str] = None,
    last_reapproved_at: Optional[str] = None,
    as_of_date: Optional[str] = None,
    auto_remove_stale: bool = False,
    stale_threshold_months: int = DEFAULT_STALE_THRESHOLD_MONTHS,
    removal_grace_months: int = DEFAULT_REMOVAL_GRACE_MONTHS,
    loaded: Optional[dict] = None,
):
    if garment_age_months is None and age_months is None:
        raise ValueError('Provide either garment_age_months or age_months.')

    model_input_age_months = int(garment_age_months if garment_age_months is not None else age_months)
    age_context = derive_listing_age_context(
        listing_created_at=listing_created_at,
        last_approved_at=last_approved_at,
        last_reapproved_at=last_reapproved_at,
        as_of_date=as_of_date,
        fallback_age_months=model_input_age_months,
    )

    row = build_feature_row(
        brand=brand,
        category=category,
        material=material,
        size=size,
        condition=condition,
        gender=gender,
        model_input_age_months=model_input_age_months,
        original_price=original_price,
        provider_price=provider_price,
    )

    artifacts = loaded or load_artifacts()
    model = artifacts['model']
    pre = artifacts['preprocessor']

    tab_df = pd.DataFrame([row])[CAT_COLS + NUM_COLS]
    tab_x = pre.transform(tab_df)
    tab_x = tab_x.astype(np.float32).toarray() if hasattr(tab_x, 'toarray') else tab_x.astype(np.float32)
    seq_x = np.array([[row['condition_token'], row['age_bin_token']]], dtype=np.int64)

    tab_t = torch.tensor(tab_x, dtype=torch.float32, device=DEVICE)
    seq_t = torch.tensor(seq_x, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(seq_t, tab_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    raw_predicted_decision = CLASS_NAMES[pred_idx]
    predicted_decision = raw_predicted_decision
    policy_adjustments = []
    if row['gender_conflict_flag'] == 1 and raw_predicted_decision != 'Reject':
        predicted_decision = 'Review'
        policy_adjustments.append('gender_category_conflict_review')

    lifecycle = effective_status_from_prediction(
        predicted_decision=predicted_decision,
        listing_age_months=age_context['listing_age_months'],
        current_status=current_status,
        stale_threshold_months=stale_threshold_months,
        removal_grace_months=removal_grace_months,
        auto_remove_stale=auto_remove_stale,
    )
    lifecycle.update(age_context)
    if row['gender_conflict_flag'] == 1:
        lifecycle['review_reason'] = '; '.join(filter(None, [lifecycle.get('review_reason', ''), 'gender_category_conflict']))
        lifecycle['frontend_popup_recommended'] = True
        lifecycle['frontend_popup_message'] = 'Selected gender does not match the current category policy. The listing should be reviewed before activation.'
    else:
        lifecycle['frontend_popup_recommended'] = False
        lifecycle['frontend_popup_message'] = ''

    return {
        'input': {
            'brand': brand,
            'category': category,
            'gender': row['gender'],
            'gender_source': row['gender_source'],
            'material': material,
            'size': size,
            'condition': normalize_condition(condition),
            'garment_age_months': int(model_input_age_months),
            'original_price': int(original_price),
            'provider_price': int(provider_price),
            'current_status': current_status,
            'listing_created_at': listing_created_at,
            'last_approved_at': last_approved_at,
            'last_reapproved_at': last_reapproved_at,
            'as_of_date': age_context['as_of_date'],
        },
        'age_context': age_context,
        'derived_features': row,
        'prediction': {
            'raw_predicted_decision': raw_predicted_decision,
            'predicted_decision': predicted_decision,
            'class_probabilities': {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            },
            'policy_adjustments': policy_adjustments,
            'suggested_listing_status': lifecycle['next_status'],
        },
        'lifecycle': lifecycle,
        'summary': {
            'predicted_decision': predicted_decision,
            'raw_predicted_decision': raw_predicted_decision,
            'suggested_listing_status': lifecycle['next_status'],
            'listing_age_months': lifecycle['listing_age_months'],
            'stale_listing_flag': lifecycle['stale_listing_flag'],
            'auto_removed': lifecycle['auto_removed'],
            'frontend_popup_recommended': lifecycle['frontend_popup_recommended'],
        },
    }
