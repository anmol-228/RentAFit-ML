from __future__ import annotations

from datetime import date
from pathlib import Path
import json
import zipfile

import pandas as pd

import sys

ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_a.inference.predict_price_range_simple_input import predict_one as predict_model_a
from model_b.runtime import predict_one as predict_model_b
from model_c.runtime import load_artifacts, recommend_from_item, recommend_from_profile
from model_c.policy import gender_compatible


BASE = Path('/Users/mypc/RentAFit')
REPORT_DIR = BASE / 'reports/validation'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = REPORT_DIR / 'model_crosscheck_report.json'
REPORT_MD = REPORT_DIR / 'model_crosscheck_report.md'

TODAY = date(2026, 3, 18)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _count_docx_images(path: Path) -> int:
    if not path.exists():
        return 0
    with zipfile.ZipFile(path, 'r') as zf:
        return sum(1 for name in zf.namelist() if name.startswith('word/media/'))


def _load_text(path: Path) -> str:
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8')


def run_model_a_cases():
    cases = [
        {
            'case_id': 'A1',
            'brand': 'Zara',
            'category': 'Top',
            'material': 'Cotton',
            'size': 'S',
            'condition': 'New',
            'age_months': 2,
            'original_price': 1200,
        },
        {
            'case_id': 'A2',
            'brand': 'H&M',
            'category': 'Jeans',
            'material': 'Denim',
            'size': 'M',
            'condition': 'Like New',
            'age_months': 6,
            'original_price': 2400,
        },
        {
            'case_id': 'A3',
            'brand': 'Biba',
            'category': 'Kurta',
            'material': 'Silk',
            'size': 'M',
            'condition': 'Like New',
            'age_months': 4,
            'original_price': 4200,
        },
        {
            'case_id': 'A4',
            'brand': 'Anita Dongre',
            'category': 'Lehenga',
            'material': 'Silk',
            'size': 'L',
            'condition': 'New',
            'age_months': 2,
            'original_price': 22000,
        },
        {
            'case_id': 'A5',
            'brand': 'Uniqlo',
            'category': 'Jacket',
            'material': 'Polyester',
            'size': 'M',
            'condition': 'Used',
            'age_months': 12,
            'original_price': 8000,
        },
        {
            'case_id': 'A6',
            'brand': 'BrandX',
            'category': 'Dress',
            'material': 'Linen',
            'size': 'S',
            'condition': 'Like New',
            'age_months': 3,
            'original_price': 3000,
        },
        {
            'case_id': 'A7',
            'brand': 'Louis Vuitton',
            'category': 'Shirt',
            'material': 'Linen',
            'size': 'XS',
            'condition': 'Like New',
            'age_months': 5,
            'original_price': 70499,
        },
    ]

    results = []
    anomalies = []
    for case in cases:
        res = predict_model_a(
            brand=case['brand'],
            category=case['category'],
            material=case['material'],
            age_months=case['age_months'],
            size=case['size'],
            condition=case['condition'],
            original_price=case['original_price'],
        )
        final_range = res['final_price_range']
        min_price = final_range['min_price']
        max_price = final_range['max_price']
        if min_price > max_price:
            anomalies.append({'case_id': case['case_id'], 'issue': 'min_price > max_price'})
        if min_price < 0 or max_price < 0:
            anomalies.append({'case_id': case['case_id'], 'issue': 'negative price in range'})
        results.append({
            'case_id': case['case_id'],
            'input': case,
            'model_route': res.get('model_route'),
            'confidence_score': _safe_float(res.get('confidence', {}).get('score', 0.0)),
            'confidence_fallback': bool(res.get('confidence', {}).get('fallback_to_rule_range', False)),
            'final_min_price': min_price,
            'final_max_price': max_price,
            'source': final_range.get('source'),
            'rule_min_price': res.get('candidate_ranges', {}).get('rule_range', {}).get('min_price'),
            'rule_max_price': res.get('candidate_ranges', {}).get('rule_range', {}).get('max_price'),
        })
    return {'cases': results, 'anomalies': anomalies}


def run_model_b_cases():
    cases = [
        {
            'case_id': 'B1',
            'brand': 'Zara',
            'category': 'Top',
            'gender': 'Women',
            'material': 'Cotton',
            'size': 'M',
            'condition': 'New',
            'garment_age_months': 2,
            'original_price': 2000,
            'provider_price': 180,
            'listing_created_at': '2026-02-01',
            'as_of_date': str(TODAY),
            'current_status': 'PENDING_REVIEW',
        },
        {
            'case_id': 'B2',
            'brand': 'Prada',
            'category': 'Dress',
            'gender': 'Men',
            'material': 'Silk',
            'size': 'S',
            'condition': 'Like New',
            'garment_age_months': 4,
            'original_price': 95000,
            'provider_price': 11650,
            'listing_created_at': '2026-01-20',
            'as_of_date': str(TODAY),
            'current_status': 'PENDING_REVIEW',
        },
        {
            'case_id': 'B3',
            'brand': 'Biba',
            'category': 'Kurta',
            'gender': 'Women',
            'material': 'Silk',
            'size': 'M',
            'condition': 'Like New',
            'garment_age_months': 8,
            'original_price': 4000,
            'provider_price': 320,
            'listing_created_at': '2025-01-01',
            'as_of_date': str(TODAY),
            'current_status': 'ACTIVE',
        },
        {
            'case_id': 'B4',
            'brand': 'Gucci',
            'category': 'Dress',
            'gender': 'Women',
            'material': 'Silk',
            'size': 'S',
            'condition': 'Used',
            'garment_age_months': 12,
            'original_price': 120000,
            'provider_price': 60000,
            'listing_created_at': '2025-04-01',
            'as_of_date': str(TODAY),
            'current_status': 'ACTIVE',
        },
        {
            'case_id': 'B5',
            'brand': 'H&M',
            'category': 'Jeans',
            'gender': 'Men',
            'material': 'Denim',
            'size': 'M',
            'condition': 'Like New',
            'garment_age_months': 4,
            'original_price': 3000,
            'provider_price': 260,
            'listing_created_at': '2024-12-01',
            'as_of_date': str(TODAY),
            'current_status': 'ACTIVE',
            'auto_remove_stale': True,
            'removal_grace_months': 3,
        },
        {
            'case_id': 'B6',
            'brand': 'Uniqlo',
            'category': 'Jacket',
            'gender': 'Men',
            'material': 'Polyester',
            'size': 'L',
            'condition': 'Like New',
            'garment_age_months': 5,
            'original_price': 8000,
            'provider_price': 1200,
            'listing_created_at': '2026-01-15',
            'as_of_date': str(TODAY),
            'current_status': 'PENDING_REVIEW',
        },
    ]

    results = []
    anomalies = []
    for case in cases:
        res = predict_model_b(
            brand=case['brand'],
            category=case['category'],
            material=case['material'],
            size=case['size'],
            condition=case['condition'],
            gender=case.get('gender'),
            garment_age_months=case['garment_age_months'],
            original_price=case['original_price'],
            provider_price=case['provider_price'],
            current_status=case.get('current_status'),
            listing_created_at=case.get('listing_created_at'),
            as_of_date=case.get('as_of_date'),
            auto_remove_stale=case.get('auto_remove_stale', False),
            removal_grace_months=case.get('removal_grace_months', 3),
        )
        pred = res['prediction']
        lifecycle = res['lifecycle']
        features = res['derived_features']

        if pred['predicted_decision'] == 'Approve' and lifecycle['next_status'] == 'REJECTED':
            anomalies.append({'case_id': case['case_id'], 'issue': 'Approve but REJECTED status'})
        if features['gender_conflict_flag'] == 1 and pred['predicted_decision'] != 'Review':
            anomalies.append({'case_id': case['case_id'], 'issue': 'gender conflict did not force Review'})
        if lifecycle.get('frontend_popup_recommended') and features['gender_conflict_flag'] != 1:
            anomalies.append({'case_id': case['case_id'], 'issue': 'popup recommended without gender conflict'})
        if lifecycle.get('auto_removed') and lifecycle.get('next_status') != 'REMOVED':
            anomalies.append({'case_id': case['case_id'], 'issue': 'auto_removed true but next_status is not REMOVED'})

        results.append({
            'case_id': case['case_id'],
            'input': case,
            'resolved_gender': features['gender'],
            'gender_conflict_flag': features['gender_conflict_flag'],
            'predicted_decision': pred['predicted_decision'],
            'suggested_status': pred['suggested_listing_status'],
            'listing_age_months': res['age_context']['listing_age_months'],
            'stale_flag': lifecycle.get('stale_listing_flag', False),
            'auto_removed': lifecycle.get('auto_removed', False),
            'popup_recommended': lifecycle.get('frontend_popup_recommended', False),
        })
    return {'cases': results, 'anomalies': anomalies}


def _pick_first(catalog: pd.DataFrame, *, category: str = None, gender: str = None):
    df = catalog
    if category is not None:
        df = df[df['category'] == category]
    if gender is not None:
        df = df[df['gender'] == gender]
    if df.empty:
        return None
    return df.iloc[0]['listing_id']


def _find_review_fallback_seed(catalog: pd.DataFrame):
    for item_id in catalog['listing_id'].head(250).tolist():
        try:
            res = recommend_from_item(item_id, top_k=5)
        except Exception:
            continue
        if res.get('policy_summary', {}).get('review_items_used', 0) > 0:
            return item_id
    return None


def _validate_model_c_recommendations(case_id: str, query_gender: str, query_category: str, recs: list[dict], max_provider_price: float = None):
    anomalies = []
    for rec in recs:
        if rec['category'] != query_category:
            anomalies.append({'case_id': case_id, 'issue': 'recommendation escaped same-category filter'})
        if not gender_compatible(query_gender, rec['gender']):
            anomalies.append({'case_id': case_id, 'issue': 'recommendation escaped gender filter'})
        if rec['size_match_label'] == 'unknown_size_match':
            anomalies.append({'case_id': case_id, 'issue': 'unknown size match in final recommendations'})
        if max_provider_price is not None and float(rec['provider_price']) > float(max_provider_price):
            anomalies.append({'case_id': case_id, 'issue': 'recommendation exceeded explicit max_provider_price'})
    return anomalies


def run_model_c_cases():
    loaded = load_artifacts()
    catalog = loaded['catalog']

    women_dress_seed = _pick_first(catalog, category='Dress', gender='Women')
    unisex_shirt_seed = _pick_first(catalog, category='Shirt', gender='Unisex')
    review_fallback_seed = _find_review_fallback_seed(catalog)

    women_profile_ids = catalog[(catalog['category'] == 'Dress') & (catalog['gender'] == 'Women')]['listing_id'].head(3).tolist()
    if len(women_profile_ids) < 3:
        women_profile_ids = catalog['listing_id'].head(3).tolist()

    cases = [
        {'case_id': 'C1', 'mode': 'item', 'seed_item_id': women_dress_seed, 'top_k': 5},
        {'case_id': 'C2', 'mode': 'item', 'seed_item_id': unisex_shirt_seed, 'top_k': 5},
        {'case_id': 'C3', 'mode': 'profile', 'liked_item_ids': women_profile_ids, 'top_k': 5},
    ]
    if review_fallback_seed is not None:
        cases.append({'case_id': 'C4', 'mode': 'item', 'seed_item_id': review_fallback_seed, 'top_k': 5})
    if women_dress_seed is not None:
        cases.append({'case_id': 'C5', 'mode': 'item', 'seed_item_id': women_dress_seed, 'top_k': 5, 'max_provider_price': 3000})

    results = []
    anomalies = []
    for case in cases:
        if case['mode'] == 'item':
            res = recommend_from_item(
                case['seed_item_id'],
                top_k=case['top_k'],
                max_provider_price=case.get('max_provider_price'),
            )
            recs = res['recommendations']
            policy = res['policy_summary']
            anomalies.extend(_validate_model_c_recommendations(
                case['case_id'],
                policy['query_gender'],
                res['seed_item']['category'],
                recs,
                max_provider_price=case.get('max_provider_price'),
            ))
            if policy.get('review_items_used', 0) > 2:
                anomalies.append({'case_id': case['case_id'], 'issue': 'review fallback exceeded cap of 2'})

            results.append({
                'case_id': case['case_id'],
                'query_mode': 'item_to_item',
                'seed_or_profile': res['seed_item']['listing_id'],
                'query_category': res['seed_item']['category'],
                'query_gender': policy['query_gender'],
                'top_k_requested': case['top_k'],
                'results_returned': len(recs),
                'budget_source': policy['budget_source'],
                'review_items_used': policy['review_items_used'],
                'top_recommendations': [r['listing_id'] for r in recs[:3]],
                'top_categories': [r['category'] for r in recs[:3]],
                'top_pool_statuses': [r['recommendation_pool_status'] for r in recs[:3]],
                'shortfall_is_valid': len(recs) < case['top_k'],
            })
        else:
            res = recommend_from_profile(case['liked_item_ids'], top_k=case['top_k'])
            recs = res['recommendations']
            policy = res['policy_summary']
            anomalies.extend(_validate_model_c_recommendations(
                case['case_id'],
                policy['query_gender'],
                res['profile_summary']['category'],
                recs,
            ))
            if policy.get('review_items_used', 0) > 2:
                anomalies.append({'case_id': case['case_id'], 'issue': 'review fallback exceeded cap of 2'})

            results.append({
                'case_id': case['case_id'],
                'query_mode': 'profile_from_liked_items',
                'seed_or_profile': ','.join(res['profile_summary']['liked_item_ids_found']),
                'query_category': res['profile_summary']['category'],
                'query_gender': policy['query_gender'],
                'top_k_requested': case['top_k'],
                'results_returned': len(recs),
                'budget_source': policy['budget_source'],
                'review_items_used': policy['review_items_used'],
                'top_recommendations': [r['listing_id'] for r in recs[:3]],
                'top_categories': [r['category'] for r in recs[:3]],
                'top_pool_statuses': [r['recommendation_pool_status'] for r in recs[:3]],
                'shortfall_is_valid': len(recs) < case['top_k'],
            })

    return {'cases': results, 'anomalies': anomalies}


def run_documentation_audit():
    model_a_md = BASE / 'docs/model_a/Model_A_Holy_Book.md'
    model_a_docx = BASE / 'docs/model_a/Model_A_Holy_Book.docx'
    model_b_md = BASE / 'docs/model_b/Model_B_Master_Document.md'
    model_b_docx = BASE / 'docs/model_b/Model_B_Master_Document.docx'
    model_c_md = BASE / 'docs/model_c/Model_C_Master_Document.md'
    model_c_docx = BASE / 'docs/model_c/Model_C_Master_Document.docx'
    project_md = BASE / 'docs/project/RentAFit_Final_Project_Master_Document.md'
    project_docx = BASE / 'docs/project/RentAFit_Final_Project_Master_Document.docx'

    required_files = {
        'model_a_md': model_a_md,
        'model_a_docx': model_a_docx,
        'model_b_md': model_b_md,
        'model_b_docx': model_b_docx,
        'model_c_md': model_c_md,
        'model_c_docx': model_c_docx,
        'project_md': project_md,
        'project_docx': project_docx,
    }

    required_visuals = {
        'model_a': [
            BASE / 'reports/model_a/model_a_training_architecture.png',
            BASE / 'reports/model_a/model_a_inference_flow.png',
            BASE / 'reports/model_a/charts/model_a_tier_split_parity_val_max.png',
            BASE / 'reports/model_a/charts/model_a_tier_split_residual_hist_val_max.png',
            BASE / 'reports/model_a/charts/model_a_tier_split_tier_mae_val_max.png',
            BASE / 'reports/model_a/charts/model_a_tier_split_comparison_val.png',
        ],
        'model_b': [
            BASE / 'reports/model_b/model_b_data_pipeline.png',
            BASE / 'reports/model_b/model_b_hybrid_architecture.png',
            BASE / 'reports/model_b/model_b_lifecycle_flow.png',
            BASE / 'reports/model_b/model_b_dataset_distribution_chart.png',
            BASE / 'reports/model_b/model_b_branch_comparison_chart.png',
            BASE / 'reports/model_b/model_b_lstm_training_history.png',
            BASE / 'reports/model_b/model_b_lstm_confusion_matrix_test.png',
        ],
        'model_c': [
            BASE / 'reports/model_c/model_c_data_pipeline.png',
            BASE / 'reports/model_c/model_c_architecture.png',
            BASE / 'reports/model_c/model_c_similarity_flow.png',
            BASE / 'reports/model_c/model_c_catalog_distribution_chart.png',
            BASE / 'reports/model_c/model_c_proxy_metrics_chart.png',
            BASE / 'reports/model_c/model_c_proxy_vs_random_chart.png',
        ],
        'project': [
            BASE / 'reports/project/project_system_architecture.png',
            BASE / 'reports/project/project_data_lineage.png',
            BASE / 'reports/project/project_model_suite_overview.png',
            BASE / 'reports/project/project_progress_roadmap.png',
        ],
    }

    doc_texts = {
        'model_a': _load_text(model_a_md),
        'model_b': _load_text(model_b_md),
        'model_c': _load_text(model_c_md),
        'project': _load_text(project_md),
    }

    required_sections = {
        'model_a': ['Purpose', 'Architecture', 'Stage-wise results', 'Quick pointers'],
        'model_b': ['Purpose', 'Architecture', 'Main Results', 'Quick pointers'],
        'model_c': ['Purpose', 'Current Metrics', 'Smoke-Test Examples', 'Quick pointers'],
        'project': ['Project', 'Model A', 'Model B', 'Model C'],
    }

    docx_counts = {
        'model_a_docx_images': _count_docx_images(model_a_docx),
        'model_b_docx_images': _count_docx_images(model_b_docx),
        'model_c_docx_images': _count_docx_images(model_c_docx),
        'project_docx_images': _count_docx_images(project_docx),
    }

    anomalies = []
    for label, path in required_files.items():
        if not path.exists():
            anomalies.append({'item': label, 'issue': 'required file missing'})

    visual_summary = {}
    for group, paths in required_visuals.items():
        missing = [str(path) for path in paths if not path.exists()]
        visual_summary[group] = {
            'required_visual_count': len(paths),
            'missing_visuals': missing,
        }
        for path in paths:
            if not path.exists():
                anomalies.append({'item': group, 'issue': f'missing visual: {path.name}'})

    for name, markers in required_sections.items():
        text = doc_texts[name]
        missing_markers = [marker for marker in markers if marker.lower() not in text.lower()]
        if missing_markers:
            anomalies.append({'item': f'{name}_md', 'issue': f'missing expected sections/phrases: {missing_markers}'})

    minimum_images = {
        'model_a_docx_images': 6,
        'model_b_docx_images': 7,
        'model_c_docx_images': 6,
        'project_docx_images': 10,
    }
    for key, min_expected in minimum_images.items():
        if docx_counts[key] < min_expected:
            anomalies.append({'item': key, 'issue': f'docx embeds too few images ({docx_counts[key]} < {min_expected})'})

    return {
        'required_files_present': {k: v.exists() for k, v in required_files.items()},
        'docx_image_counts': docx_counts,
        'visual_summary': visual_summary,
        'anomalies': anomalies,
    }


def build_markdown(report: dict) -> str:
    lines = []
    lines.append('# RentAFit Model Cross-Check Report')
    lines.append('')
    lines.append(f'Generated: {TODAY.isoformat()}')
    lines.append('')

    lines.append('## Model A Summary')
    lines.append('')
    lines.append(f"Cases: {len(report['model_a']['cases'])}")
    lines.append(f"Anomalies: {len(report['model_a']['anomalies'])}")
    lines.append('')
    lines.append('| Case | Brand | Category | Condition | Age | Original | Final Min | Final Max | Source | Confidence | Fallback |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in report['model_a']['cases']:
        inp = row['input']
        lines.append(
            f"| {row['case_id']} | {inp['brand']} | {inp['category']} | {inp['condition']} | {inp['age_months']} | {inp['original_price']} | {row['final_min_price']} | {row['final_max_price']} | {row['source']} | {row['confidence_score']:.2f} | {row['confidence_fallback']} |"
        )

    lines.append('')
    lines.append('## Model B Summary')
    lines.append('')
    lines.append(f"Cases: {len(report['model_b']['cases'])}")
    lines.append(f"Anomalies: {len(report['model_b']['anomalies'])}")
    lines.append('')
    lines.append('| Case | Category | Input Gender | Resolved Gender | Conflict | Decision | Status | Listing Age | Stale | Auto Removed | Popup |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in report['model_b']['cases']:
        inp = row['input']
        lines.append(
            f"| {row['case_id']} | {inp['category']} | {inp.get('gender', '')} | {row['resolved_gender']} | {row['gender_conflict_flag']} | {row['predicted_decision']} | {row['suggested_status']} | {row['listing_age_months']} | {row['stale_flag']} | {row['auto_removed']} | {row['popup_recommended']} |"
        )

    lines.append('')
    lines.append('## Model C Summary')
    lines.append('')
    lines.append(f"Cases: {len(report['model_c']['cases'])}")
    lines.append(f"Anomalies: {len(report['model_c']['anomalies'])}")
    lines.append('')
    lines.append('| Case | Mode | Seed/Profile | Category | Gender | Returned | Budget Source | Review Used | Top Recs | Top Pools |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in report['model_c']['cases']:
        lines.append(
            f"| {row['case_id']} | {row['query_mode']} | {row['seed_or_profile']} | {row['query_category']} | {row['query_gender']} | {row['results_returned']}/{row['top_k_requested']} | {row['budget_source']} | {row['review_items_used']} | {', '.join(row['top_recommendations'])} | {', '.join(row['top_pool_statuses'])} |"
        )

    lines.append('')
    lines.append('## Documentation And Visual Audit')
    lines.append('')
    doc_audit = report['documentation_audit']
    lines.append(f"Documentation anomalies: {len(doc_audit['anomalies'])}")
    lines.append('')
    lines.append('| Document | Exists | Embedded Images |')
    lines.append('| --- | --- | --- |')
    lines.append(f"| Model A handbook | {doc_audit['required_files_present']['model_a_docx']} | {doc_audit['docx_image_counts']['model_a_docx_images']} |")
    lines.append(f"| Model B handbook | {doc_audit['required_files_present']['model_b_docx']} | {doc_audit['docx_image_counts']['model_b_docx_images']} |")
    lines.append(f"| Model C handbook | {doc_audit['required_files_present']['model_c_docx']} | {doc_audit['docx_image_counts']['model_c_docx_images']} |")
    lines.append(f"| Project master handbook | {doc_audit['required_files_present']['project_docx']} | {doc_audit['docx_image_counts']['project_docx_images']} |")
    lines.append('')
    lines.append('| Visual Group | Required Visuals | Missing Visuals |')
    lines.append('| --- | --- | --- |')
    for group, summary in doc_audit['visual_summary'].items():
        missing = ', '.join(Path(x).name for x in summary['missing_visuals']) if summary['missing_visuals'] else 'None'
        lines.append(f"| {group} | {summary['required_visual_count']} | {missing} |")

    all_anomalies = []
    for label, data in [
        ('Model A', report['model_a']['anomalies']),
        ('Model B', report['model_b']['anomalies']),
        ('Model C', report['model_c']['anomalies']),
    ]:
        for item in data:
            all_anomalies.append(f"- {label} {item['case_id']}: {item['issue']}")
    for item in report['documentation_audit']['anomalies']:
        all_anomalies.append(f"- Documentation {item['item']}: {item['issue']}")

    if all_anomalies:
        lines.append('')
        lines.append('## Anomalies')
        lines.extend(all_anomalies)

    return '\n'.join(lines)


def main():
    report = {
        'generated_on': TODAY.isoformat(),
        'model_a': run_model_a_cases(),
        'model_b': run_model_b_cases(),
        'model_c': run_model_c_cases(),
        'documentation_audit': run_documentation_audit(),
        'improvement_check': {
            'model_c_weight_tuning_attempted': True,
            'result': 'No material gain found; current weights retained because the best safe alternative improved the composite objective by only about 0.0003.',
        },
    }

    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    REPORT_MD.write_text(build_markdown(report), encoding='utf-8')

    print('Saved:', REPORT_JSON)
    print('Saved:', REPORT_MD)


if __name__ == '__main__':
    main()
