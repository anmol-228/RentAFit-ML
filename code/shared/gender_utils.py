from typing import Optional, Tuple


VALID_GENDERS = ("Women", "Men", "Unisex")
VALID_GENDER_SOURCES = ("user_selected", "category_derived", "manual_override")

_NORMALIZED_GENDER_MAP = {
    "women": "Women",
    "woman": "Women",
    "female": "Women",
    "ladies": "Women",
    "lady": "Women",
    "men": "Men",
    "man": "Men",
    "male": "Men",
    "gents": "Men",
    "gent": "Men",
    "unisex": "Unisex",
    "uni sex": "Unisex",
    "uni-sex": "Unisex",
}

_CATEGORY_GENDER_MAP = {
    "Saree": "Women",
    "Lehenga": "Women",
    "Dress": "Women",
    "Top": "Women",
    "Shirt": "Unisex",
    "Jacket": "Unisex",
    "Jeans": "Unisex",
    "Activewear": "Unisex",
    "Ethnic Wear": "Unisex",
    "Kurta": "Unisex",
}


def normalize_gender(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = " ".join(text.replace("-", " ").split()).lower()
    if lowered in _NORMALIZED_GENDER_MAP:
        return _NORMALIZED_GENDER_MAP[lowered]

    title = text.title()
    if title in VALID_GENDERS:
        return title
    return None


def gender_from_category(category: Optional[str]) -> str:
    if category is None:
        return "Unisex"
    return _CATEGORY_GENDER_MAP.get(str(category).strip(), "Unisex")


def derive_gender(explicit_gender: Optional[str] = None, category: Optional[str] = None) -> Tuple[str, str]:
    normalized = normalize_gender(explicit_gender)
    if normalized is not None:
        return normalized, "user_selected"
    return gender_from_category(category), "category_derived"


def resolve_gender_fields(
    gender: Optional[str] = None,
    category: Optional[str] = None,
    gender_source: Optional[str] = None,
) -> Tuple[str, str]:
    normalized = normalize_gender(gender)
    if normalized is None:
        return gender_from_category(category), "category_derived"

    source = str(gender_source).strip() if gender_source is not None else ""
    if source in VALID_GENDER_SOURCES:
        return normalized, source
    return normalized, "user_selected"


def gender_conflict_flag(gender: Optional[str], category: Optional[str]) -> int:
    normalized = normalize_gender(gender)
    if normalized is None:
        return 0

    category_gender = gender_from_category(category)
    if category_gender == "Unisex":
        return 0
    return int(normalized != category_gender)
