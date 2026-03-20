from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_c.policy import (
    FILTERED_OUT_STATUS,
    MAX_REVIEW_FILL_ITEMS,
    PRIMARY_POOL_STATUS,
    REVIEW_FALLBACK_STATUS,
    budget_alignment_score,
    derive_budget_reference,
    gender_compatible,
    gender_match_label,
    size_distance,
    size_match_label,
    size_match_score,
    safety_score,
)
from shared.gender_utils import normalize_gender

BASE = Path('/Users/mypc/RentAFit')
MODEL_DIR = BASE / 'models/model_c/content_based'

PREPROCESSOR_PATH = MODEL_DIR / 'model_c_preprocessor.joblib'
NEIGHBORS_PATH = MODEL_DIR / 'model_c_nearest_neighbors.joblib'
MATRIX_PATH = MODEL_DIR / 'model_c_feature_matrix.joblib'
CATALOG_PATH = MODEL_DIR / 'model_c_catalog_recommendable.csv'
FULL_CATALOG_PATH = BASE / 'data/generated/model_c_catalog.csv'
METADATA_PATH = MODEL_DIR / 'model_c_metadata.json'

_CACHE = None


def _normalize_top_k(top_k: int) -> int:
    return max(1, min(int(top_k), 20))


def load_artifacts():
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    catalog = pd.read_csv(CATALOG_PATH)
    full_catalog = pd.read_csv(FULL_CATALOG_PATH) if FULL_CATALOG_PATH.exists() else catalog.copy()
    matrix = joblib.load(MATRIX_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    neighbors = joblib.load(NEIGHBORS_PATH)
    metadata = json.loads(METADATA_PATH.read_text())

    _CACHE = {
        'catalog': catalog,
        'full_catalog': full_catalog,
        'matrix': matrix,
        'preprocessor': preprocessor,
        'neighbors': neighbors,
        'metadata': metadata,
    }
    return _CACHE


def _profile_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    if len(mode) == 0:
        return None
    return mode.iloc[0]


def _profile_gender(series: pd.Series) -> str:
    normalized = [(normalize_gender(v) or 'Unisex') for v in series.tolist()]
    specific = {v for v in normalized if v in {'Women', 'Men'}}
    if len(specific) == 1:
        return list(specific)[0]
    return 'Unisex'


def _metadata_feature_cols(loaded: dict) -> List[str]:
    return loaded['metadata']['categorical_columns'] + loaded['metadata']['numeric_columns']


def _query_vector_from_row(row: pd.Series, loaded: dict):
    preprocessor = loaded['preprocessor']
    feature_cols = _metadata_feature_cols(loaded)
    return preprocessor.transform(pd.DataFrame([row[feature_cols].to_dict()]))


def _sort_or_shuffle(pool: pd.DataFrame, randomize: bool = False, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    if pool.empty:
        return pool
    if randomize:
        if rng is None:
            rng = np.random.default_rng(42)
        shuffled_idx = rng.permutation(pool.index.to_numpy())
        return pool.loc[shuffled_idx]
    return pool.sort_values(
        ['final_score', 'similarity_score', 'catalog_priority_score', 'model_b_approve_probability'],
        ascending=False,
    )


def prepare_policy_candidates(
    candidates: pd.DataFrame,
    query_context: Dict,
    explicit_budget: Optional[float] = None,
    exclude_same_brand: bool = False,
    excluded_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    out = candidates.copy()

    if excluded_ids:
        out = out[~out['listing_id'].isin(excluded_ids)]

    out = out[out['category'] == query_context['category']]
    out = out[out['gender'].apply(lambda value: gender_compatible(query_context['gender'], value))]

    if exclude_same_brand and query_context.get('brand') is not None:
        out = out[out['brand'] != query_context['brand']]

    if out.empty:
        query_context = dict(query_context)
        query_context.update({
            'budget_reference_price': 0.0,
            'budget_source': 'missing_budget_reference',
        })
        return out, query_context

    reference_budget, budget_source = derive_budget_reference(
        out,
        category=query_context['category'],
        explicit_budget=explicit_budget,
    )
    query_context = dict(query_context)
    query_context.update({
        'budget_reference_price': float(reference_budget),
        'budget_source': budget_source,
    })

    out['size_distance'] = out['size'].apply(lambda value: size_distance(query_context.get('size'), value))
    out['size_match_label'] = out['size_distance'].apply(size_match_label)
    out['size_match_score'] = out['size_distance'].apply(size_match_score)
    out['gender_match_label'] = out['gender'].apply(lambda value: gender_match_label(query_context.get('gender'), value))
    out['budget_alignment_score'] = budget_alignment_score(out['provider_price'], reference_budget)
    if explicit_budget is not None and float(explicit_budget) > 0:
        out['within_budget_band'] = out['provider_price'] <= float(explicit_budget)
    else:
        from model_c.policy import within_budget_band
        out['within_budget_band'] = within_budget_band(out['provider_price'], reference_budget, explicit_budget=None)
    out['safety_score'] = out['recommendation_pool_status'].map(safety_score).fillna(0.0)

    out['final_score'] = (
        0.55 * out['similarity_score']
        + 0.15 * out['quality_score_norm']
        + 0.08 * out['item_freshness_score']
        + 0.10 * out['budget_alignment_score']
        + 0.07 * out['size_match_score']
        + 0.05 * out['safety_score']
    )
    return out, query_context


def select_recommendations_from_candidates(
    candidates: pd.DataFrame,
    top_k: int = 5,
    randomize: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    top_k = _normalize_top_k(top_k)
    if candidates.empty:
        return candidates.head(0).copy()

    selected_parts = []
    selected_ids = set()
    review_used = 0

    def add_pool(pool: pd.DataFrame, is_review: bool):
        nonlocal review_used
        ordered = _sort_or_shuffle(pool, randomize=randomize, rng=rng)
        for _, row in ordered.iterrows():
            if len(selected_ids) >= top_k:
                break
            if row['listing_id'] in selected_ids:
                continue
            if is_review and review_used >= MAX_REVIEW_FILL_ITEMS:
                break
            selected_parts.append(row.to_frame().T)
            selected_ids.add(row['listing_id'])
            if is_review:
                review_used += 1

    primary = candidates[candidates['recommendation_pool_status'] == PRIMARY_POOL_STATUS].copy()
    review = candidates[candidates['recommendation_pool_status'] == REVIEW_FALLBACK_STATUS].copy()

    for pool, is_review in [(primary, False), (review, True)]:
        if pool.empty or len(selected_ids) >= top_k:
            continue
        for budget_pref in [True, False]:
            budget_pool = pool[pool['within_budget_band'] == budget_pref]
            if budget_pool.empty:
                continue
            for distance in sorted(budget_pool['size_distance'].dropna().unique()):
                add_pool(budget_pool[budget_pool['size_distance'] == distance], is_review=is_review)
                if len(selected_ids) >= top_k:
                    break
            if len(selected_ids) >= top_k:
                break

    if not selected_parts:
        return candidates.head(0).copy()
    out = pd.concat(selected_parts, ignore_index=True)
    return out.head(top_k)


def _reason_tags(query_context: Dict, rec_row: pd.Series) -> List[str]:
    reasons = ['same_category']

    reasons.append(rec_row['gender_match_label'])
    reasons.append(rec_row['size_match_label'])

    if str(query_context.get('material')) == str(rec_row['material']):
        reasons.append('same_material')
    if str(query_context.get('tier_primary')) == str(rec_row['tier_primary']):
        reasons.append('same_tier')
    if str(query_context.get('brand')) == str(rec_row['brand']):
        reasons.append('same_brand')
    if str(query_context.get('condition')) == str(rec_row['condition']):
        reasons.append('same_condition')

    if bool(rec_row.get('within_budget_band')):
        if query_context.get('budget_source') == 'explicit_user_budget':
            reasons.append('within_user_budget')
        else:
            reasons.append('near_category_budget')

    if float(rec_row.get('quality_score_norm', 0.0)) >= 0.75:
        reasons.append('high_quality_item')
    if float(rec_row.get('item_freshness_score', 0.0)) >= 0.60:
        reasons.append('fresh_item')
    if rec_row.get('recommendation_pool_status') == REVIEW_FALLBACK_STATUS:
        reasons.append('review_fallback_item')

    deduped = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped[:5]


def _build_result_rows(candidates: pd.DataFrame, query_context: Dict) -> List[Dict]:
    rows = []
    for _, row in candidates.iterrows():
        rows.append({
            'listing_id': row['listing_id'],
            'brand': row['brand'],
            'category': row['category'],
            'gender': row['gender'],
            'material': row['material'],
            'size': row['size'],
            'condition': row['condition'],
            'age_months': int(row['age_months']),
            'provider_price': float(row['provider_price']),
            'category_avg_provider_price': float(row['category_avg_provider_price']),
            'rule_mid': float(row['rule_mid']),
            'tier_primary': row['tier_primary'],
            'rule_quality_score': float(row['rule_quality_score']),
            'recommendation_pool_status': row['recommendation_pool_status'],
            'gender_match_label': row['gender_match_label'],
            'size_match_label': row['size_match_label'],
            'similarity_score': round(float(row['similarity_score']), 6),
            'budget_alignment_score': round(float(row['budget_alignment_score']), 6),
            'final_score': round(float(row['final_score']), 6),
            'explanation_reasons': _reason_tags(query_context, row),
        })
    return rows


def build_item_candidate_pool(
    seed_item_id: str,
    max_provider_price: float = None,
    exclude_same_brand: bool = False,
    loaded: Optional[dict] = None,
):
    loaded = loaded or load_artifacts()
    catalog = loaded['catalog']
    full_catalog = loaded['full_catalog']
    matrix = loaded['matrix']
    preprocessor = loaded['preprocessor']
    neighbors = loaded['neighbors']

    matches = catalog.index[catalog['listing_id'] == seed_item_id].tolist()
    if matches:
        seed_idx = matches[0]
        seed_row = catalog.iloc[seed_idx]
        query_vec = matrix[seed_idx]
    else:
        full_matches = full_catalog.index[full_catalog['listing_id'] == seed_item_id].tolist()
        if not full_matches:
            raise ValueError(f'Seed item not found in Model C catalog: {seed_item_id}')
        seed_row = full_catalog.iloc[full_matches[0]]
        query_vec = preprocessor.transform(pd.DataFrame([seed_row[_metadata_feature_cols(loaded)].to_dict()]))

    n_query = min(max(neighbors.n_neighbors, 200), len(catalog))
    distances, indices = neighbors.kneighbors(query_vec, n_neighbors=n_query, return_distance=True)
    sims = 1.0 - distances.ravel()
    candidates = catalog.iloc[indices.ravel()].copy()
    candidates['similarity_score'] = sims

    query_context = {
        'listing_id': seed_row['listing_id'],
        'brand': seed_row['brand'],
        'category': seed_row['category'],
        'gender': seed_row['gender'],
        'material': seed_row['material'],
        'size': seed_row['size'],
        'condition': seed_row['condition'],
        'tier_primary': seed_row['tier_primary'],
        'provider_price': float(seed_row['provider_price']),
    }

    candidates, query_context = prepare_policy_candidates(
        candidates=candidates,
        query_context=query_context,
        explicit_budget=max_provider_price,
        exclude_same_brand=exclude_same_brand,
        excluded_ids=[seed_item_id],
    )
    return seed_row, candidates, query_context


def build_profile_candidate_pool(
    liked_item_ids: List[str],
    max_provider_price: float = None,
    exclude_same_brand: bool = False,
    loaded: Optional[dict] = None,
):
    loaded = loaded or load_artifacts()
    catalog = loaded['catalog']
    full_catalog = loaded['full_catalog']
    matrix = loaded['matrix']
    preprocessor = loaded['preprocessor']
    neighbors = loaded['neighbors']

    transformed_rows = []
    liked_row_frames = []
    liked_found_ids = []

    for item_id in liked_item_ids:
        matches = catalog.index[catalog['listing_id'] == item_id].tolist()
        if matches:
            idx = matches[0]
            liked_row_frames.append(catalog.iloc[[idx]])
            transformed_rows.append(matrix[idx])
            liked_found_ids.append(item_id)
            continue

        full_matches = full_catalog.index[full_catalog['listing_id'] == item_id].tolist()
        if full_matches:
            row = full_catalog.iloc[[full_matches[0]]].copy()
            liked_row_frames.append(row)
            transformed_rows.append(preprocessor.transform(row[_metadata_feature_cols(loaded)]))
            liked_found_ids.append(item_id)

    if not transformed_rows:
        raise ValueError('None of the liked_item_ids were found in the Model C catalog.')

    liked_rows = pd.concat(liked_row_frames, ignore_index=True)
    query_vec = sparse.vstack(transformed_rows).mean(axis=0)
    query_vec = sparse.csr_matrix(query_vec)

    n_query = min(max(neighbors.n_neighbors, 200), len(catalog))
    distances, indices = neighbors.kneighbors(query_vec, n_neighbors=n_query, return_distance=True)
    sims = 1.0 - distances.ravel()
    candidates = catalog.iloc[indices.ravel()].copy()
    candidates['similarity_score'] = sims

    query_context = {
        'listing_id': '|'.join(liked_found_ids),
        'brand': _profile_mode(liked_rows['brand']),
        'category': _profile_mode(liked_rows['category']),
        'gender': _profile_gender(liked_rows['gender']),
        'material': _profile_mode(liked_rows['material']),
        'size': _profile_mode(liked_rows['size']),
        'condition': _profile_mode(liked_rows['condition']),
        'tier_primary': _profile_mode(liked_rows['tier_primary']),
        'provider_price': float(liked_rows['provider_price'].mean()),
        'liked_item_ids_found': liked_found_ids,
    }

    candidates, query_context = prepare_policy_candidates(
        candidates=candidates,
        query_context=query_context,
        explicit_budget=max_provider_price,
        exclude_same_brand=exclude_same_brand,
        excluded_ids=liked_rows['listing_id'].tolist(),
    )
    return liked_rows, candidates, query_context


def recommend_from_item(
    seed_item_id: str,
    top_k: int = 5,
    category_filter: str = None,
    max_provider_price: float = None,
    exclude_same_brand: bool = False,
):
    top_k = _normalize_top_k(top_k)
    loaded = load_artifacts()
    seed_row, candidates, query_context = build_item_candidate_pool(
        seed_item_id=seed_item_id,
        max_provider_price=max_provider_price,
        exclude_same_brand=exclude_same_brand,
        loaded=loaded,
    )

    if category_filter and category_filter != query_context['category']:
        candidates = candidates.head(0).copy()

    selected = select_recommendations_from_candidates(candidates, top_k=top_k)
    review_items_used = int((selected['recommendation_pool_status'] == REVIEW_FALLBACK_STATUS).sum()) if not selected.empty else 0

    return {
        'query_mode': 'item_to_item',
        'seed_item': {
            'listing_id': seed_row['listing_id'],
            'brand': seed_row['brand'],
            'category': seed_row['category'],
            'gender': seed_row['gender'],
            'material': seed_row['material'],
            'size': seed_row['size'],
            'condition': seed_row['condition'],
            'provider_price': float(seed_row['provider_price']),
            'tier_primary': seed_row['tier_primary'],
        },
        'policy_summary': {
            'same_category_only': True,
            'query_gender': query_context['gender'],
            'query_size': query_context['size'],
            'budget_reference_price': round(float(query_context['budget_reference_price']), 2),
            'budget_source': query_context['budget_source'],
            'max_review_fallback_items': MAX_REVIEW_FILL_ITEMS,
            'review_items_used': review_items_used,
        },
        'filters': {
            'category_filter': query_context['category'],
            'max_provider_price': max_provider_price,
            'exclude_same_brand': bool(exclude_same_brand),
            'top_k': top_k,
        },
        'recommendations': _build_result_rows(selected, query_context),
    }


def recommend_from_profile(
    liked_item_ids: List[str],
    top_k: int = 5,
    category_filter: str = None,
    max_provider_price: float = None,
    exclude_same_brand: bool = False,
):
    top_k = _normalize_top_k(top_k)
    liked_rows, candidates, query_context = build_profile_candidate_pool(
        liked_item_ids=liked_item_ids,
        max_provider_price=max_provider_price,
        exclude_same_brand=exclude_same_brand,
    )

    if category_filter and category_filter != query_context['category']:
        candidates = candidates.head(0).copy()

    selected = select_recommendations_from_candidates(candidates, top_k=top_k)
    review_items_used = int((selected['recommendation_pool_status'] == REVIEW_FALLBACK_STATUS).sum()) if not selected.empty else 0

    return {
        'query_mode': 'profile_from_liked_items',
        'profile_summary': {
            'liked_item_ids_found': query_context['liked_item_ids_found'],
            'category': query_context['category'],
            'gender': query_context['gender'],
            'material': query_context['material'],
            'size': query_context['size'],
            'brand': query_context['brand'],
            'tier_primary': query_context['tier_primary'],
            'condition': query_context['condition'],
            'avg_provider_price': round(float(query_context['provider_price']), 2),
        },
        'policy_summary': {
            'same_category_only': True,
            'query_gender': query_context['gender'],
            'query_size': query_context['size'],
            'budget_reference_price': round(float(query_context['budget_reference_price']), 2),
            'budget_source': query_context['budget_source'],
            'max_review_fallback_items': MAX_REVIEW_FILL_ITEMS,
            'review_items_used': review_items_used,
        },
        'filters': {
            'category_filter': query_context['category'],
            'max_provider_price': max_provider_price,
            'exclude_same_brand': bool(exclude_same_brand),
            'top_k': top_k,
        },
        'recommendations': _build_result_rows(selected, query_context),
    }
