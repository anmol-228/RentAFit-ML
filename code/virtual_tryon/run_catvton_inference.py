"""Starter wrapper for RentAFit virtual try-on inference.

Current status:
- project structure created
- CatVTON integration not wired yet

Planned job of this script:
1. accept person image path
2. accept garment image path
3. validate category support
4. call CatVTON inference
5. save generated output image
6. print or return output path
"""

from pathlib import Path
import argparse
import json

SUPPORTED_CATEGORIES = {"Shirt", "Top", "Jacket"}


def validate_inputs(person_image: str, garment_image: str, category: str):
    errors = []
    if category not in SUPPORTED_CATEGORIES:
        errors.append(f"Unsupported category for V1: {category}")
    if not Path(person_image).exists():
        errors.append(f"Person image not found: {person_image}")
    if not Path(garment_image).exists():
        errors.append(f"Garment image not found: {garment_image}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="RentAFit CatVTON wrapper placeholder")
    parser.add_argument("--person_image", required=True)
    parser.add_argument("--garment_image", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output_path", required=False)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    errors = validate_inputs(args.person_image, args.garment_image, args.category)

    result = {
        "status": "not_implemented",
        "model": "CatVTON",
        "category": args.category,
        "supported_categories": sorted(SUPPORTED_CATEGORIES),
        "person_image": args.person_image,
        "garment_image": args.garment_image,
        "output_path": args.output_path,
        "validation_errors": errors,
        "message": "CatVTON runtime is not connected yet. Use this script after the model environment is set up."
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
