from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
BASE = REPO_ROOT
INPUT_PATH = BASE / 'data/generated/model_c_catalog_recommendable.csv'
MODEL_DIR = BASE / 'models/model_c/content_based'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_PATH = MODEL_DIR / 'model_c_preprocessor.joblib'
NEIGHBORS_PATH = MODEL_DIR / 'model_c_nearest_neighbors.joblib'
MATRIX_PATH = MODEL_DIR / 'model_c_feature_matrix.joblib'
CATALOG_REF_PATH = MODEL_DIR / 'model_c_catalog_recommendable.csv'
METADATA_PATH = MODEL_DIR / 'model_c_metadata.json'

CAT_COLS = ['brand', 'category', 'gender', 'material', 'size', 'condition', 'tier_primary']
NUM_COLS = [
    'age_months',
    'original_price',
    'provider_price',
    'rule_mid',
    'rule_quality_score',
    'quality_score_norm',
    'item_freshness_score',
    'provider_price_pct_of_original',
    'category_avg_provider_price',
    'price_vs_category_avg_ratio',
    'catalog_priority_score',
    'model_b_approve_probability',
    'model_b_review_probability',
]


def make_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    return ColumnTransformer([
        ('cat', cat_pipe, CAT_COLS),
        ('num', num_pipe, NUM_COLS),
    ])


def main():
    df = pd.read_csv(INPUT_PATH)
    if df.empty:
        raise ValueError('Recommendable Model C catalog is empty. Run prepare_model_c_catalog.py first.')

    preprocessor = make_preprocessor()
    X = preprocessor.fit_transform(df[CAT_COLS + NUM_COLS])

    n_neighbors = min(150, len(df))
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    nn.fit(X)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(nn, NEIGHBORS_PATH)
    joblib.dump(X, MATRIX_PATH)
    df.to_csv(CATALOG_REF_PATH, index=False)

    metadata = {
        'model_type': 'content_based_cosine_similarity',
        'catalog_rows': int(len(df)),
        'n_neighbors_fit': int(n_neighbors),
        'categorical_columns': CAT_COLS,
        'numeric_columns': NUM_COLS,
        'recommendation_logic': {
            'base_similarity_weight': 0.55,
            'quality_weight': 0.15,
            'freshness_weight': 0.08,
            'budget_alignment_weight': 0.10,
            'size_match_weight': 0.07,
            'safety_weight': 0.05,
        },
        'policy': {
            'same_category_only': True,
            'hard_gender_filter': True,
            'exact_size_then_nearest': True,
            'review_fallback_limit': 2,
            'default_budget_strategy': 'category_average_budget',
        },
    }
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Saved:', PREPROCESSOR_PATH)
    print('Saved:', NEIGHBORS_PATH)
    print('Saved:', MATRIX_PATH)
    print('Saved:', CATALOG_REF_PATH)
    print('Saved:', METADATA_PATH)


if __name__ == '__main__':
    main()
