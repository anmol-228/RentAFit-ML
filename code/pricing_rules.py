import math
from typing import Dict, Tuple

# Canonical manual-v1 pricing rules.
# If you want rule_mid rounded to nearest 10, set MID_MODE="nearest10".
MID_MODE = "avg"  # allowed: "avg", "nearest10"

BASE_MIN = {"Tier 1": 0.05, "Tier 2": 0.06, "Tier 3": 0.07, "Tier 4": 0.09, "Tier 5": 0.11}
BASE_MAX = {"Tier 1": 0.06, "Tier 2": 0.07, "Tier 3": 0.09, "Tier 4": 0.11, "Tier 5": 0.14}


def tier_primary(original_price: float) -> str:
    if original_price <= 1200:
        return "Tier 1"
    if original_price <= 2500:
        return "Tier 2"
    if original_price <= 5000:
        return "Tier 3"
    if original_price <= 15000:
        return "Tier 4"
    return "Tier 5"


def cond_mult(condition: str) -> float:
    if condition == "New":
        return 1.0
    if condition == "Like New":
        return 0.9
    return 0.7


def age_mult(age_months: float) -> float:
    if age_months <= 3:
        return 1.0
    if age_months <= 6:
        return 0.9
    if age_months <= 10:
        return 0.8
    if age_months <= 15:
        return 0.7
    return 0.6


def cat_mult(category: str) -> float:
    if category in ("Saree", "Lehenga", "Ethnic Wear"):
        return 1.10
    if category in ("Dress", "Jacket"):
        return 1.05
    if category in ("Shirt", "Top", "Kurta"):
        return 1.00
    if category in ("Activewear", "Jeans"):
        return 0.95
    return 1.00


def mat_mult(material: str) -> float:
    if material in ("Silk", "Leather"):
        return 1.05
    if material == "Linen":
        return 1.03
    return 1.00


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def bucket_round(value: float) -> int:
    if value < 200:
        return round_half_up(value / 10) * 10
    if value <= 1000:
        return round_half_up(value / 50) * 50
    return round_half_up(value / 100) * 100


def round_to_nearest_10(value: float) -> int:
    return round_half_up(value / 10) * 10


def compute_rule_mid(rule_min: int, rule_max: int, mid_mode: str = MID_MODE):
    raw = (rule_min + rule_max) / 2.0
    if mid_mode == "nearest10":
        return int(round_to_nearest_10(raw))
    # avg mode: keep exact midpoint (usually ends with .0 or .5)
    return round(raw, 1)


def deviation_m(provider_price: float, rule_min: float, rule_max: float) -> float:
    if provider_price < rule_min:
        return ((rule_min - provider_price) / rule_min) * 100 if rule_min else 0.0
    if provider_price > rule_max:
        return ((provider_price - rule_max) / rule_max) * 100 if rule_max else 0.0
    return 0.0


def penalties(condition: str, age_months: float, deviation: float) -> Tuple[int, int, int]:
    cond_penalty = 0 if condition == "New" else 8 if condition == "Like New" else 18

    if age_months <= 3:
        age_penalty = 0
    elif age_months <= 6:
        age_penalty = 4
    elif age_months <= 9:
        age_penalty = 8
    elif age_months <= 12:
        age_penalty = 12
    elif age_months <= 18:
        age_penalty = 16
    else:
        age_penalty = 20

    if deviation == 0:
        deviation_penalty = 0
    elif deviation <= 10:
        deviation_penalty = 5
    elif deviation <= 20:
        deviation_penalty = 12
    elif deviation <= 35:
        deviation_penalty = 20
    elif deviation <= 50:
        deviation_penalty = 30
    else:
        deviation_penalty = 45

    return cond_penalty, age_penalty, deviation_penalty


def rule_decision(original_price: float, provider_price: float, quality_score: int, deviation: float) -> str:
    if provider_price <= 0 or original_price <= 0 or provider_price > 0.35 * original_price or quality_score < 40 or deviation > 60:
        return "Reject"
    if quality_score >= 70 and deviation <= 20:
        return "Approve"
    return "Review"


def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _validate_bucket_value(v: int) -> bool:
    if v < 200:
        return v % 10 == 0
    if v <= 1000:
        return v % 50 == 0
    return v % 100 == 0


def compute_pricing_fields(
    brand: str,
    category: str,
    material: str,
    size: str,
    condition: str,
    age_months: int,
    original_price: int,
    provider_price: int,
    mid_mode: str = MID_MODE,
) -> Dict:
    tier = tier_primary(original_price)
    base_min_pct = BASE_MIN[tier]
    base_max_pct = BASE_MAX[tier]

    condition_multiplier = cond_mult(condition)
    age_multiplier = age_mult(age_months)
    category_multiplier = cat_mult(category)
    material_multiplier = mat_mult(material)

    raw_min = original_price * base_min_pct * condition_multiplier * age_multiplier * category_multiplier * material_multiplier
    raw_max = original_price * base_max_pct * condition_multiplier * age_multiplier * category_multiplier * material_multiplier

    capped_min = min(raw_min, 0.20 * original_price, original_price)
    capped_max = min(raw_max, 0.20 * original_price, original_price)

    rule_min = bucket_round(capped_min)
    rule_max = bucket_round(capped_max)
    if rule_max < rule_min:
        rule_max = rule_min

    rule_mid = compute_rule_mid(rule_min, rule_max, mid_mode=mid_mode)
    deviation = deviation_m(provider_price, rule_min, rule_max)

    condition_penalty, age_penalty, deviation_penalty = penalties(condition, age_months, deviation)
    total_penalty = condition_penalty + age_penalty + deviation_penalty

    quality_score = max(0, 100 - total_penalty)
    decision = rule_decision(original_price, provider_price, quality_score, deviation)

    penalty_text = f"cond={condition_penalty}|age={age_penalty}|dev={deviation_penalty}|total={total_penalty}"

    out = {
        "brand": brand,
        "category": category,
        "material": material,
        "size": size,
        "condition": condition,
        "age_months": int(age_months),
        "original_price": int(original_price),
        "provider_price": int(provider_price),
        "tier_primary": tier,
        "base_min_pct": base_min_pct,
        "base_max_pct": base_max_pct,
        "cond_mult": condition_multiplier,
        "age_mult": age_multiplier,
        "cat_mult": category_multiplier,
        "mat_mult": material_multiplier,
        "rule_min": int(rule_min),
        "rule_max": int(rule_max),
        "rule_mid": rule_mid,
        "deviation_M": round(deviation, 4),
        "condition_penalty": int(condition_penalty),
        "age_penalty": int(age_penalty),
        "deviation_penalty": int(deviation_penalty),
        "total_penalty": int(total_penalty),
        "penalties": penalty_text,
        "rule_quality_score": int(quality_score),
        "rule_decision": decision,
    }
    validate_pricing_fields(out, mid_mode=mid_mode)
    return out


def validate_pricing_fields(row: Dict, mid_mode: str = MID_MODE) -> None:
    original_price = float(row["original_price"])
    provider_price = float(row["provider_price"])
    age_months = float(row["age_months"])
    condition = row["condition"]
    category = row["category"]
    material = row["material"]

    exp_tier = tier_primary(original_price)
    if row["tier_primary"] != exp_tier:
        raise ValueError(f"Invalid tier_primary for row {row.get('listing_id', '?')}")

    if not _is_close(float(row["base_min_pct"]), BASE_MIN[exp_tier]) or not _is_close(float(row["base_max_pct"]), BASE_MAX[exp_tier]):
        raise ValueError(f"Invalid base pct for row {row.get('listing_id', '?')}")

    exp_cond = cond_mult(condition)
    exp_age = age_mult(age_months)
    exp_cat = cat_mult(category)
    exp_mat = mat_mult(material)
    if not _is_close(float(row["cond_mult"]), exp_cond):
        raise ValueError(f"Invalid cond_mult for row {row.get('listing_id', '?')}")
    if not _is_close(float(row["age_mult"]), exp_age):
        raise ValueError(f"Invalid age_mult for row {row.get('listing_id', '?')}")
    if not _is_close(float(row["cat_mult"]), exp_cat):
        raise ValueError(f"Invalid cat_mult for row {row.get('listing_id', '?')}")
    if not _is_close(float(row["mat_mult"]), exp_mat):
        raise ValueError(f"Invalid mat_mult for row {row.get('listing_id', '?')}")

    raw_min = original_price * BASE_MIN[exp_tier] * exp_cond * exp_age * exp_cat * exp_mat
    raw_max = original_price * BASE_MAX[exp_tier] * exp_cond * exp_age * exp_cat * exp_mat
    cap_min = min(raw_min, 0.20 * original_price, original_price)
    cap_max = min(raw_max, 0.20 * original_price, original_price)
    exp_rule_min = bucket_round(cap_min)
    exp_rule_max = bucket_round(cap_max)
    if exp_rule_max < exp_rule_min:
        exp_rule_max = exp_rule_min

    rule_min = int(float(row["rule_min"]))
    rule_max = int(float(row["rule_max"]))
    if rule_min != exp_rule_min or rule_max != exp_rule_max:
        raise ValueError(f"Invalid rule range for row {row.get('listing_id', '?')}")
    if rule_max < rule_min:
        raise ValueError(f"rule_max < rule_min for row {row.get('listing_id', '?')}")
    if not _validate_bucket_value(rule_min) or not _validate_bucket_value(rule_max):
        raise ValueError(f"Invalid bucket rounding for row {row.get('listing_id', '?')}")

    exp_mid = compute_rule_mid(rule_min, rule_max, mid_mode=mid_mode)
    if float(row["rule_mid"]) != float(exp_mid):
        raise ValueError(f"Invalid rule_mid for row {row.get('listing_id', '?')}")

    exp_dev = deviation_m(provider_price, rule_min, rule_max)
    if not _is_close(float(row["deviation_M"]), exp_dev, tol=1e-4):
        raise ValueError(f"Invalid deviation_M for row {row.get('listing_id', '?')}")

    exp_cond_penalty, exp_age_penalty, exp_dev_penalty = penalties(condition, age_months, exp_dev)
    if int(row["condition_penalty"]) != exp_cond_penalty:
        raise ValueError(f"Invalid condition_penalty for row {row.get('listing_id', '?')}")
    if int(row["age_penalty"]) != exp_age_penalty:
        raise ValueError(f"Invalid age_penalty for row {row.get('listing_id', '?')}")
    if int(row["deviation_penalty"]) != exp_dev_penalty:
        raise ValueError(f"Invalid deviation_penalty for row {row.get('listing_id', '?')}")

    exp_total_penalty = exp_cond_penalty + exp_age_penalty + exp_dev_penalty
    if int(row["total_penalty"]) != exp_total_penalty:
        raise ValueError(f"Invalid total_penalty for row {row.get('listing_id', '?')}")

    exp_quality_score = max(0, 100 - exp_total_penalty)
    if int(row["rule_quality_score"]) != exp_quality_score:
        raise ValueError(f"Invalid rule_quality_score for row {row.get('listing_id', '?')}")

    exp_decision = rule_decision(original_price, provider_price, exp_quality_score, exp_dev)
    if row["rule_decision"] != exp_decision:
        raise ValueError(f"Invalid rule_decision for row {row.get('listing_id', '?')}")
