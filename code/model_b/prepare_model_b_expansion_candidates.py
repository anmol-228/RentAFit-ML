from pathlib import Path
import json

import pandas as pd

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
BASE = REPO_ROOT
DATA_FROZEN = BASE / 'data/frozen/v1_final'
DATA_GENERATED = BASE / 'data/generated'
REPORT_DIR = BASE / 'reports/model_b'

AUG_PATH = DATA_FROZEN / 'pricing_features_augmented_1500.csv'
GOLD_PATH = DATA_FROZEN / 'gold_labels_model_b_full_manual.csv'
OUT_PATH = DATA_GENERATED / 'model_b_expansion_candidates_manual_review.csv'
SUMMARY_PATH = REPORT_DIR / 'model_b_expansion_candidate_summary.json'

TARGET_ROWS = 400
STALE_AGE_THRESHOLD = 10
HIGH_DEVIATION_THRESHOLD = 35.0


def older_listing_flag(age_months: float) -> int:
    return int(float(age_months) >= STALE_AGE_THRESHOLD)


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
    if condition == 'New':
        return 1
    if condition == 'Like New':
        return 2
    return 3


def auto_suggested_decision(rule_decision: str, age_months: float) -> str:
    if float(age_months) >= STALE_AGE_THRESHOLD and rule_decision == 'Approve':
        return 'Review'
    return rule_decision


def selection_reason(row) -> str:
    reasons = []
    if row['older_listing_flag'] == 1:
        reasons.append('older_listing_10plus')
    if row['high_deviation_flag'] == 1:
        reasons.append('high_deviation_over_35')
    if row['used_flag'] == 1:
        reasons.append('used_condition')
    if row['premium_tier_flag'] == 1:
        reasons.append('premium_tier_4_or_5')
    if row['borderline_flag'] == 1:
        reasons.append('borderline_review_case')
    if not reasons:
        reasons.append('general_fill_case')
    return '|'.join(reasons)


def add_priority_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['condition_token'] = out['condition'].apply(condition_token)
    out['age_bin_token'] = out['age_months'].apply(age_bin_token)
    out['older_listing_flag'] = out['age_months'].apply(older_listing_flag)
    out['high_deviation_flag'] = (out['deviation_M'] > HIGH_DEVIATION_THRESHOLD).astype(int)
    out['used_flag'] = (out['condition'] == 'Used').astype(int)
    out['premium_tier_flag'] = out['tier_primary'].isin(['Tier 4', 'Tier 5']).astype(int)
    out['borderline_flag'] = (
        (out['rule_decision'] == 'Review') |
        (out['rule_quality_score'].between(55, 85, inclusive='both'))
    ).astype(int)
    out['priority_score'] = (
        3 * out['older_listing_flag'] +
        3 * out['high_deviation_flag'] +
        2 * out['used_flag'] +
        2 * out['premium_tier_flag'] +
        2 * out['borderline_flag']
    )
    out['auto_suggested_quality_score_0_100'] = out['rule_quality_score']
    out['auto_suggested_listing_decision'] = out.apply(
        lambda row: auto_suggested_decision(row['rule_decision'], row['age_months']),
        axis=1,
    )
    out['age_policy_applied_to_suggestion'] = (
        (out['older_listing_flag'] == 1) & (out['rule_decision'] == 'Approve')
    ).astype(int)
    out['selection_reason'] = out.apply(selection_reason, axis=1)
    return out


def select_group(df: pd.DataFrame, current_ids: set, mask, limit: int) -> pd.DataFrame:
    pool = df[mask & (~df['listing_id'].isin(current_ids))].copy()
    if pool.empty:
        return pool.head(0)
    pool = pool.sort_values(
        ['priority_score', 'deviation_M', 'age_months', 'rule_quality_score', 'original_price'],
        ascending=[False, False, False, True, False],
    )
    return pool.head(limit)


def build_candidates() -> tuple[pd.DataFrame, dict]:
    aug = pd.read_csv(AUG_PATH)
    gold = pd.read_csv(GOLD_PATH)

    gold_ids = set(gold['listing_id'].astype(str))
    work = aug[~aug['listing_id'].astype(str).isin(gold_ids)].copy()
    work = work[work['data_source'] == 'synthetic'].copy()
    work = add_priority_features(work)

    selected_parts = []
    selected_ids = set()

    quotas = [
        ('older_listing_priority', work['older_listing_flag'] == 1, 120),
        ('high_deviation_priority', work['high_deviation_flag'] == 1, 120),
        ('used_condition_priority', work['used_flag'] == 1, 80),
        ('premium_or_borderline_priority', (work['premium_tier_flag'] == 1) | (work['borderline_flag'] == 1), 80),
    ]

    quota_log = {}
    for name, mask, limit in quotas:
        picked = select_group(work, selected_ids, mask, limit)
        selected_parts.append(picked)
        selected_ids.update(picked['listing_id'].astype(str).tolist())
        quota_log[name] = int(len(picked))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else work.head(0).copy()

    if len(selected) < TARGET_ROWS:
        remaining = work[~work['listing_id'].astype(str).isin(selected_ids)].copy()
        remaining = remaining.sort_values(
            ['priority_score', 'deviation_M', 'age_months', 'rule_quality_score', 'original_price'],
            ascending=[False, False, False, True, False],
        )
        filler = remaining.head(TARGET_ROWS - len(selected))
        selected = pd.concat([selected, filler], ignore_index=True)

    selected = selected.drop_duplicates(subset=['listing_id']).head(TARGET_ROWS).copy()
    selected = selected.sort_values(['priority_score', 'listing_id'], ascending=[False, True]).reset_index(drop=True)

    selected['review_batch'] = 'batch_01'
    selected['review_status'] = 'pending'
    selected['reviewer_name'] = ''
    selected['review_date'] = ''
    selected['final_quality_score_0_100'] = ''
    selected['final_listing_decision'] = ''
    selected['final_reviewer_note'] = ''
    selected['label_source'] = 'manual_review_pending'

    final_cols = [
        'listing_id', 'source_listing_id', 'data_source',
        'brand', 'category', 'material', 'size', 'condition',
        'condition_token', 'age_months', 'age_bin_token',
        'older_listing_flag', 'original_price', 'provider_price', 'tier_primary',
        'deviation_M', 'condition_penalty', 'age_penalty', 'deviation_penalty', 'total_penalty',
        'rule_quality_score', 'rule_decision',
        'auto_suggested_quality_score_0_100', 'auto_suggested_listing_decision',
        'age_policy_applied_to_suggestion',
        'high_deviation_flag', 'used_flag', 'premium_tier_flag', 'borderline_flag',
        'priority_score', 'selection_reason',
        'review_batch', 'review_status', 'reviewer_name', 'review_date',
        'final_quality_score_0_100', 'final_listing_decision', 'final_reviewer_note', 'label_source',
    ]

    selected = selected[final_cols].copy()

    summary = {
        'target_rows': TARGET_ROWS,
        'selected_rows': int(len(selected)),
        'quota_log': quota_log,
        'selected_rule_decision_counts': selected['rule_decision'].value_counts().to_dict(),
        'selected_auto_suggested_decision_counts': selected['auto_suggested_listing_decision'].value_counts().to_dict(),
        'selected_tier_counts': selected['tier_primary'].value_counts().sort_index().to_dict(),
        'selected_condition_counts': selected['condition'].value_counts().to_dict(),
        'selected_older_listing_flag_counts': selected['older_listing_flag'].value_counts().sort_index().to_dict(),
        'selected_high_deviation_flag_counts': selected['high_deviation_flag'].value_counts().sort_index().to_dict(),
        'selected_borderline_flag_counts': selected['borderline_flag'].value_counts().sort_index().to_dict(),
    }

    return selected, summary


def main():
    DATA_GENERATED.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    selected, summary = build_candidates()
    selected.to_csv(OUT_PATH, index=False)
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Saved candidates:', OUT_PATH)
    print('Saved summary:', SUMMARY_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
