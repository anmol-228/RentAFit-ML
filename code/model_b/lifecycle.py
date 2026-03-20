from __future__ import annotations

from datetime import date
import math
from typing import Any, Optional

DEFAULT_STALE_THRESHOLD_MONTHS = 10
DEFAULT_REMOVAL_GRACE_MONTHS = 3
DEFAULT_REVIEWER_ROLE = 'admin_reviewer'


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == '' or text.lower() in {'nan', 'nat', 'none', 'null'}


def parse_optional_date(value: Any) -> Optional[date]:
    if _is_missing(value):
        return None
    text = str(value).strip()[:10]
    return date.fromisoformat(text)


def iso_or_none(value: Optional[date]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def months_between(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(0, months)


def derive_listing_age_context(
    listing_created_at: Any = None,
    last_approved_at: Any = None,
    last_reapproved_at: Any = None,
    as_of_date: Any = None,
    fallback_age_months: Optional[int] = None,
) -> dict:
    as_of = parse_optional_date(as_of_date) or date.today()

    candidates = [
        ('last_reapproved_at', parse_optional_date(last_reapproved_at)),
        ('last_approved_at', parse_optional_date(last_approved_at)),
        ('listing_created_at', parse_optional_date(listing_created_at)),
    ]

    source_name = None
    reference_date = None
    for name, candidate in candidates:
        if candidate is not None:
            source_name = name
            reference_date = candidate
            break

    if reference_date is not None:
        return {
            'listing_age_months': months_between(reference_date, as_of),
            'listing_age_source': source_name,
            'listing_age_reference_date': iso_or_none(reference_date),
            'as_of_date': iso_or_none(as_of),
        }

    if fallback_age_months is not None:
        return {
            'listing_age_months': int(fallback_age_months),
            'listing_age_source': 'fallback_input_age_months',
            'listing_age_reference_date': None,
            'as_of_date': iso_or_none(as_of),
        }

    return {
        'listing_age_months': None,
        'listing_age_source': 'missing',
        'listing_age_reference_date': None,
        'as_of_date': iso_or_none(as_of),
    }


def review_reason(predicted_decision: str, stale_listing_flag: bool, removal_recommended: bool) -> str:
    reasons = []
    if predicted_decision == 'Reject':
        reasons.append('model_reject')
    elif predicted_decision == 'Review':
        reasons.append('model_review')
    if stale_listing_flag:
        reasons.append('stale_listing_reapproval')
    if removal_recommended:
        reasons.append('removal_recommended_after_stale_grace')
    return '; '.join(reasons)


def review_priority(predicted_decision: str, stale_listing_flag: bool, removal_recommended: bool) -> str:
    if predicted_decision == 'Reject' or removal_recommended:
        return 'high'
    if stale_listing_flag or predicted_decision == 'Review':
        return 'medium'
    return 'low'


def effective_status_from_prediction(
    predicted_decision: str,
    listing_age_months: Optional[int],
    current_status: Optional[str] = None,
    stale_threshold_months: int = DEFAULT_STALE_THRESHOLD_MONTHS,
    removal_grace_months: int = DEFAULT_REMOVAL_GRACE_MONTHS,
    auto_remove_stale: bool = False,
) -> dict:
    current_status = str(current_status).strip().upper() if current_status else None
    stale_listing_flag = listing_age_months is not None and int(listing_age_months) >= int(stale_threshold_months)
    removal_recommended = listing_age_months is not None and int(listing_age_months) >= int(stale_threshold_months + removal_grace_months)

    if current_status == 'REMOVED':
        next_status = 'REMOVED'
    else:
        if predicted_decision == 'Reject':
            next_status = 'REJECTED'
        elif stale_listing_flag and auto_remove_stale and removal_recommended:
            next_status = 'REMOVED'
        elif stale_listing_flag:
            next_status = 'REAPPROVAL_REQUIRED'
        elif predicted_decision == 'Approve':
            next_status = 'ACTIVE'
        else:
            next_status = 'PENDING_REVIEW'

        return {
            'current_status': current_status,
            'next_status': next_status,
            'listing_age_months': listing_age_months,
            'stale_threshold_months': int(stale_threshold_months),
            'removal_grace_months': int(removal_grace_months),
            'stale_listing_flag': bool(stale_listing_flag),
            'removal_recommended': bool(removal_recommended),
            'auto_removed': bool(next_status == 'REMOVED'),
            'review_required': next_status in {'PENDING_REVIEW', 'REAPPROVAL_REQUIRED', 'REJECTED'},
            'visible_to_renters': next_status == 'ACTIVE',
            'assigned_reviewer_role': DEFAULT_REVIEWER_ROLE if next_status in {'PENDING_REVIEW', 'REAPPROVAL_REQUIRED', 'REJECTED'} else '',
            'review_reason': review_reason(predicted_decision, stale_listing_flag, removal_recommended),
            'review_priority': review_priority(predicted_decision, stale_listing_flag, removal_recommended),
        }

    return {
        'current_status': current_status,
        'next_status': next_status,
        'listing_age_months': listing_age_months,
        'stale_threshold_months': int(stale_threshold_months),
        'removal_grace_months': int(removal_grace_months),
        'stale_listing_flag': bool(stale_listing_flag),
        'removal_recommended': bool(removal_recommended or current_status == 'REMOVED'),
        'auto_removed': False,
        'review_required': False,
        'visible_to_renters': False,
        'assigned_reviewer_role': '',
        'review_reason': 'already_removed',
        'review_priority': 'low',
    }
