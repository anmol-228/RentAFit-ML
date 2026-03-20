from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys
ROOT_CODE_DIR = Path('/Users/mypc/RentAFit/code')
if str(ROOT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_CODE_DIR))

from model_c.runtime import (
    build_item_candidate_pool,
    load_artifacts,
    recommend_from_item,
    select_recommendations_from_candidates,
)

BASE = Path('/Users/mypc/RentAFit')
REPORT_DIR = BASE / 'reports/model_c'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = REPORT_DIR / 'model_c_proxy_metrics.json'
RANDOM_COMPARE_PATH = REPORT_DIR / 'model_c_proxy_vs_random_metrics.json'
CAT_DIST_PATH = REPORT_DIR / 'model_c_catalog_distribution_chart.png'
PROXY_CHART_PATH = REPORT_DIR / 'model_c_proxy_metrics_chart.png'
COMPARE_CHART_PATH = REPORT_DIR / 'model_c_proxy_vs_random_chart.png'
FLOW_PATH = REPORT_DIR / 'model_c_similarity_flow.png'
PIPELINE_PATH = REPORT_DIR / 'model_c_data_pipeline.png'
ARCH_PATH = REPORT_DIR / 'model_c_architecture.png'
SAMPLES_PATH = REPORT_DIR / 'model_c_sample_recommendations.csv'


def make_distribution_chart(catalog: pd.DataFrame):
    cat_counts = catalog['category'].value_counts().sort_values(ascending=False).head(10)
    gender_counts = catalog['gender'].value_counts().sort_index()
    pool_counts = catalog['recommendation_pool_status'].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].bar(cat_counts.index, cat_counts.values, color='#86b5d8')
    axes[0].set_title('Top Categories In Recommendation Pool')
    axes[0].tick_params(axis='x', rotation=35)
    axes[0].set_ylabel('Item count')

    axes[1].bar(gender_counts.index, gender_counts.values, color='#f4b183')
    axes[1].set_title('Gender Distribution In Recommendation Pool')
    axes[1].set_ylabel('Item count')

    axes[2].bar(pool_counts.index, pool_counts.values, color=['#9fd39f', '#f7d794'])
    axes[2].set_title('Primary vs Review Fallback Pool')
    axes[2].tick_params(axis='x', rotation=20)
    axes[2].set_ylabel('Item count')

    fig.suptitle('Model C Recommendation Pool Distribution')
    fig.tight_layout()
    fig.savefig(CAT_DIST_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_proxy_chart(metrics: dict):
    labels = [
        'fill_rate@5',
        'gender_ok@5',
        'exact_size@5',
        'budget_align@5',
        'avg_similarity@5',
        'avg_quality@5',
    ]
    values = [
        metrics['fill_rate_at_5'],
        metrics['gender_compatible_at_5'],
        metrics['exact_size_at_5'],
        metrics['avg_budget_alignment_top5'],
        metrics['avg_similarity_score_top5'],
        metrics['avg_rule_quality_score_top5'] / 100.0,
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color=['#7db4d8', '#9fd39f', '#f4b183', '#d6b4e8', '#f2a6b3', '#8bd3c7'])
    ax.set_ylim(0, 1.05)
    ax.set_title('Model C Policy-Aware Proxy Metrics')
    ax.set_ylabel('Normalized score')
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(PROXY_CHART_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_comparison_chart(model_metrics: dict, random_metrics: dict):
    labels = ['same_material@5', 'same_tier@5', 'budget_align@5', 'avg_similarity@5', 'avg_final_score@5', 'avg_quality@5']
    model_vals = [
        model_metrics['same_material_at_5'],
        model_metrics['same_tier_at_5'],
        model_metrics['avg_budget_alignment_top5'],
        model_metrics['avg_similarity_score_top5'],
        model_metrics['avg_final_score_top5'],
        model_metrics['avg_rule_quality_score_top5'] / 100.0,
    ]
    random_vals = [
        random_metrics['same_material_at_5'],
        random_metrics['same_tier_at_5'],
        random_metrics['avg_budget_alignment_top5'],
        random_metrics['avg_similarity_score_top5'],
        random_metrics['avg_final_score_top5'],
        random_metrics['avg_rule_quality_score_top5'] / 100.0,
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, model_vals, width, label='Model C', color='#7db4d8')
    ax.bar(x + width / 2, random_vals, width, label='Policy-aware random baseline', color='#f4b183')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Normalized score')
    ax.set_title('Model C vs Policy-Aware Random Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(COMPARE_CHART_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_flow_diagram():
    fig, ax = plt.subplots(figsize=(17, 7.2))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 7.2)
    ax.axis('off')

    def box(x, y, w, h, text, fc, fontsize=11, weight='normal'):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.12', linewidth=1.7, edgecolor='#444', facecolor=fc)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, weight=weight)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=14, linewidth=1.7, color='#555'))

    ax.text(8.5, 6.6, 'Model C Policy-Aware Recommendation Flow', fontsize=20, weight='bold', ha='center')
    ax.text(8.5, 6.2, 'Similarity retrieval is followed by hard category/gender filtering, size fallback, budget alignment, and limited review fallback.', fontsize=10.8, ha='center', color='#555')

    box(0.6, 2.8, 3.0, 1.6, 'Recommendable catalog\n(primary + review fallback)', '#d9edf7', weight='bold')
    box(4.2, 2.6, 3.4, 2.0, 'Feature encoding\none-hot categorical\nscaled numeric\ncontent vector per item', '#e8f5e9', weight='bold')
    box(8.2, 2.8, 3.2, 1.6, 'Cosine similarity\nnearest neighbor retrieval', '#fff3cd', weight='bold')
    box(11.9, 4.2, 2.8, 1.2, 'Hard filters\nsame category\ngender compatible', '#ffe0b2', weight='bold')
    box(11.9, 2.0, 2.8, 1.2, 'Ranking policy\nexact size -> nearest\nbudget alignment', '#fde2e4', weight='bold')
    box(11.9, 0.2, 2.8, 1.2, 'Fallback policy\nup to 2 review items\nif strong matches', '#e0f2f1', weight='bold')
    box(15.1, 2.4, 1.8, 2.0, 'Top-5 ranked\nrecommendations\nwith reason tags', '#e2f0d9', weight='bold')

    arr(3.6, 3.6, 4.2, 3.6)
    arr(7.6, 3.6, 8.2, 3.6)
    arr(11.4, 4.8, 11.9, 4.8)
    arr(11.4, 2.6, 11.9, 2.6)
    arr(11.4, 1.2, 11.9, 1.2)
    arr(14.7, 3.2, 15.1, 3.4)

    fig.tight_layout()
    fig.savefig(FLOW_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(18, 6.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    def box(x, y, w, h, text, fc, fontsize=11, weight='normal'):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.12', linewidth=1.6, edgecolor='#444', facecolor=fc)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, weight=weight)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=14, linewidth=1.6, color='#555'))

    ax.text(9, 6.0, 'Model C Data Pipeline And Artifacts', fontsize=20, weight='bold', ha='center')
    ax.text(9, 5.6, 'Catalog rows are scored with Model B moderation, split into primary vs review fallback pools, then encoded for retrieval.', fontsize=10.8, ha='center', color='#555')

    box(0.5, 2.8, 3.0, 1.6, 'Pricing features\nv2_gender augmented dataset\n(1500 rows)', '#d9edf7', weight='bold')
    box(4.0, 2.8, 3.2, 1.6, 'Prepare catalog\nmoderation scoring +\nbudget/freshness fields', '#e8f5e9', weight='bold')
    box(7.8, 3.6, 3.3, 1.2, 'model_c_catalog.csv', '#fff3cd', weight='bold', fontsize=10.2)
    box(7.8, 2.0, 3.3, 1.2, 'model_c_catalog_recommendable.csv', '#fff3cd', weight='bold', fontsize=9.6)
    box(11.6, 2.8, 3.2, 1.6, 'Build retrieval artifacts\npreprocessor + NN index', '#e0f2f1', weight='bold')
    box(15.3, 3.6, 2.2, 1.2, 'Saved artifacts\n(joblib + metadata)', '#e2f0d9', weight='bold', fontsize=10)
    box(15.3, 1.6, 2.2, 1.2, 'Reports + charts\n(proxy metrics)', '#fde2e4', weight='bold', fontsize=10)

    arr(3.5, 3.6, 4.0, 3.6)
    arr(7.2, 3.6, 7.8, 4.2)
    arr(7.2, 3.2, 7.8, 2.6)
    arr(11.1, 3.6, 11.6, 3.6)
    arr(14.8, 3.8, 15.3, 4.2)
    arr(14.8, 3.0, 15.3, 2.2)

    fig.tight_layout()
    fig.savefig(PIPELINE_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 6.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    def box(x, y, w, h, text, fc, fontsize=11, weight='normal'):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.12', linewidth=1.6, edgecolor='#444', facecolor=fc)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, weight=weight)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=14, linewidth=1.6, color='#555'))

    ax.text(9, 6.0, 'Model C Architecture (Training + Inference)', fontsize=20, weight='bold', ha='center')
    ax.text(9, 5.6, 'Gender, size, quality, freshness, and category-budget context all contribute to the retrieval and re-ranking process.', fontsize=10.8, ha='center', color='#555')

    box(0.6, 2.8, 2.7, 1.6, 'Recommendable\nitem catalog', '#d9edf7', weight='bold')
    box(3.7, 2.8, 3.0, 1.6, 'Preprocessing\none-hot + scaling', '#e8f5e9', weight='bold')
    box(7.1, 2.8, 2.7, 1.6, 'Feature matrix\nX (items)', '#fff3cd', weight='bold')
    box(10.2, 2.8, 3.0, 1.6, 'NearestNeighbors\ncosine index', '#e0f2f1', weight='bold')
    box(13.8, 3.9, 3.4, 1.0, 'Query: viewed item / liked items', '#ffe0b2', weight='bold', fontsize=10.5)
    box(13.8, 2.4, 3.4, 1.0, 'Hard filters\ncategory + gender', '#fde2e4', weight='bold', fontsize=10.5)
    box(13.8, 0.7, 3.4, 1.2, 'Re-rank\nsize + budget + safety\nthen limited review fallback', '#e2f0d9', weight='bold', fontsize=10.5)

    arr(3.3, 3.6, 3.7, 3.6)
    arr(6.7, 3.6, 7.1, 3.6)
    arr(9.8, 3.6, 10.2, 3.6)
    arr(13.2, 4.4, 13.8, 4.4)
    arr(13.2, 2.9, 13.8, 2.9)
    arr(15.5, 2.4, 15.5, 1.9)

    fig.tight_layout()
    fig.savefig(ARCH_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _stratified_sample(catalog: pd.DataFrame) -> pd.DataFrame:
    sample_parts = [group.head(min(len(group), 8)) for _, group in catalog.groupby('category', sort=False)]
    return pd.concat(sample_parts, ignore_index=True) if sample_parts else catalog.head(0).copy()


def _metric_from_recs(seed: dict, rec_df: pd.DataFrame):
    return {
        'same_category': float((rec_df['category'] == seed['category']).mean()),
        'gender_compatible': float(rec_df['gender_match_label'].isin(['exact_gender_match', 'unisex_gender_match']).mean()),
        'exact_size': float((rec_df['size_match_label'] == 'exact_size_match').mean()),
        'size_compatible': float(rec_df['size_match_label'].isin(['exact_size_match', 'nearest_size_match']).mean()),
        'same_material': float((rec_df['material'] == seed['material']).mean()),
        'same_tier': float((rec_df['tier_primary'] == seed['tier_primary']).mean()),
        'avg_quality': float(rec_df['rule_quality_score'].mean()),
        'avg_similarity': float(rec_df['similarity_score'].mean()),
        'avg_budget_alignment': float(rec_df['budget_alignment_score'].mean()),
        'avg_final_score': float(rec_df['final_score'].mean()),
        'review_fallback_rate': float((rec_df['recommendation_pool_status'] == 'REVIEW_FALLBACK').mean()),
    }


def compute_model_metrics(sample: pd.DataFrame):
    rows = []
    fill_rates = []
    same_cat = []
    gender_ok = []
    exact_size = []
    size_ok = []
    same_mat = []
    same_tier = []
    avg_quality = []
    avg_sim = []
    avg_budget = []
    avg_final = []
    review_fallback = []

    for item_id in sample['listing_id'].tolist():
        result = recommend_from_item(item_id, top_k=5)
        recs = result['recommendations']
        if not recs:
            continue

        fill_rates.append(len(recs) / 5.0)
        rec_df = pd.DataFrame(recs)
        seed = result['seed_item']
        metric = _metric_from_recs(seed, rec_df)

        same_cat.append(metric['same_category'])
        gender_ok.append(metric['gender_compatible'])
        exact_size.append(metric['exact_size'])
        size_ok.append(metric['size_compatible'])
        same_mat.append(metric['same_material'])
        same_tier.append(metric['same_tier'])
        avg_quality.append(metric['avg_quality'])
        avg_sim.append(metric['avg_similarity'])
        avg_budget.append(metric['avg_budget_alignment'])
        avg_final.append(metric['avg_final_score'])
        review_fallback.append(metric['review_fallback_rate'])

        for rec in recs:
            rows.append({
                'seed_item_id': seed['listing_id'],
                'seed_category': seed['category'],
                'seed_gender': seed['gender'],
                'seed_size': seed['size'],
                'recommended_item_id': rec['listing_id'],
                'recommended_category': rec['category'],
                'recommended_gender': rec['gender'],
                'recommended_size': rec['size'],
                'recommended_brand': rec['brand'],
                'recommended_provider_price': rec['provider_price'],
                'recommendation_pool_status': rec['recommendation_pool_status'],
                'similarity_score': rec['similarity_score'],
                'budget_alignment_score': rec['budget_alignment_score'],
                'final_score': rec['final_score'],
                'explanation_reasons': '|'.join(rec['explanation_reasons']),
            })

    metrics = {
        'evaluation_type': 'policy_aware_proxy_metrics',
        'sample_query_count': int(len(fill_rates)),
        'fill_rate_at_5': float(np.mean(fill_rates)) if fill_rates else 0.0,
        'same_category_at_5': float(np.mean(same_cat)) if same_cat else 0.0,
        'gender_compatible_at_5': float(np.mean(gender_ok)) if gender_ok else 0.0,
        'exact_size_at_5': float(np.mean(exact_size)) if exact_size else 0.0,
        'size_compatible_at_5': float(np.mean(size_ok)) if size_ok else 0.0,
        'same_material_at_5': float(np.mean(same_mat)) if same_mat else 0.0,
        'same_tier_at_5': float(np.mean(same_tier)) if same_tier else 0.0,
        'avg_rule_quality_score_top5': float(np.mean(avg_quality)) if avg_quality else 0.0,
        'avg_similarity_score_top5': float(np.mean(avg_sim)) if avg_sim else 0.0,
        'avg_budget_alignment_top5': float(np.mean(avg_budget)) if avg_budget else 0.0,
        'avg_final_score_top5': float(np.mean(avg_final)) if avg_final else 0.0,
        'review_fallback_rate_top5': float(np.mean(review_fallback)) if review_fallback else 0.0,
        'important_note': 'These are proxy metrics because real renter interaction logs are not available yet.',
    }
    return metrics, rows


def compute_random_metrics(sample: pd.DataFrame, rng: np.random.Generator):
    fill_rates = []
    same_cat = []
    gender_ok = []
    exact_size = []
    size_ok = []
    same_mat = []
    same_tier = []
    avg_quality = []
    avg_sim = []
    avg_budget = []
    avg_final = []
    review_fallback = []

    for item_id in sample['listing_id'].tolist():
        seed_row, candidates, _ = build_item_candidate_pool(item_id)
        if candidates.empty:
            continue
        selected = select_recommendations_from_candidates(candidates, top_k=5, randomize=True, rng=rng)
        if selected.empty:
            continue
        rec_df = selected.copy()
        fill_rates.append(len(rec_df) / 5.0)
        metric = _metric_from_recs(seed_row.to_dict(), rec_df)
        same_cat.append(metric['same_category'])
        gender_ok.append(metric['gender_compatible'])
        exact_size.append(metric['exact_size'])
        size_ok.append(metric['size_compatible'])
        same_mat.append(metric['same_material'])
        same_tier.append(metric['same_tier'])
        avg_quality.append(metric['avg_quality'])
        avg_sim.append(metric['avg_similarity'])
        avg_budget.append(metric['avg_budget_alignment'])
        avg_final.append(metric['avg_final_score'])
        review_fallback.append(metric['review_fallback_rate'])

    return {
        'evaluation_type': 'policy_aware_random_baseline',
        'sample_query_count': int(len(fill_rates)),
        'fill_rate_at_5': float(np.mean(fill_rates)) if fill_rates else 0.0,
        'same_category_at_5': float(np.mean(same_cat)) if same_cat else 0.0,
        'gender_compatible_at_5': float(np.mean(gender_ok)) if gender_ok else 0.0,
        'exact_size_at_5': float(np.mean(exact_size)) if exact_size else 0.0,
        'size_compatible_at_5': float(np.mean(size_ok)) if size_ok else 0.0,
        'same_material_at_5': float(np.mean(same_mat)) if same_mat else 0.0,
        'same_tier_at_5': float(np.mean(same_tier)) if same_tier else 0.0,
        'avg_rule_quality_score_top5': float(np.mean(avg_quality)) if avg_quality else 0.0,
        'avg_similarity_score_top5': float(np.mean(avg_sim)) if avg_sim else 0.0,
        'avg_budget_alignment_top5': float(np.mean(avg_budget)) if avg_budget else 0.0,
        'avg_final_score_top5': float(np.mean(avg_final)) if avg_final else 0.0,
        'review_fallback_rate_top5': float(np.mean(review_fallback)) if review_fallback else 0.0,
    }


def main():
    loaded = load_artifacts()
    catalog = loaded['catalog']

    sample = _stratified_sample(catalog)

    make_distribution_chart(catalog)
    make_flow_diagram()
    make_pipeline_diagram()
    make_architecture_diagram()

    model_metrics, sample_rows = compute_model_metrics(sample)
    rng = np.random.default_rng(42)
    random_metrics = compute_random_metrics(sample, rng)

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_metrics, f, indent=2)
    with open(RANDOM_COMPARE_PATH, 'w', encoding='utf-8') as f:
        json.dump(random_metrics, f, indent=2)

    pd.DataFrame(sample_rows).to_csv(SAMPLES_PATH, index=False)

    make_proxy_chart(model_metrics)
    make_comparison_chart(model_metrics, random_metrics)

    print('Saved metrics:', METRICS_PATH)
    print('Saved random baseline metrics:', RANDOM_COMPARE_PATH)
    print('Saved distribution chart:', CAT_DIST_PATH)
    print('Saved proxy chart:', PROXY_CHART_PATH)
    print('Saved comparison chart:', COMPARE_CHART_PATH)
    print('Saved flow diagram:', FLOW_PATH)
    print('Saved pipeline diagram:', PIPELINE_PATH)
    print('Saved architecture diagram:', ARCH_PATH)
    print('Saved sample recommendations:', SAMPLES_PATH)


if __name__ == '__main__':
    main()
