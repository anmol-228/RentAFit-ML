# Virtual Try-On Module

This folder is the working area for the RentAFit virtual try-on feature.

## Goal

Enable a renter to:
- upload their own image,
- choose a supported clothing listing,
- generate a virtual try-on output image.

## V1 scope

Supported categories only:
- Shirt
- Top
- Jacket

Not supported in V1:
- Saree
- Lehenga
- Ethnic Wear
- Dress
- Kurta
- Activewear
- Jeans

## Main files

- `run_catvton_inference.py`
  Main wrapper script that will call CatVTON inference.
- `preprocess.py`
  Input validation and image-preparation helper logic.
- `postprocess.py`
  Output save/path formatting helper logic.
- `sample_inputs/`
  Small local examples for testing.
- `sample_outputs/`
  Saved outputs from early test runs.

## Practical rule

This module is separate from Model A, Model B, and Model C.
Do not mix virtual try-on logic into those folders.
