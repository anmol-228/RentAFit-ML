# RentAFit Virtual Try-On Plan

## Objective

Add an experimental renter-side virtual try-on feature using a pretrained diffusion model.

Chosen model:
- CatVTON

## V1 scope

Supported categories only:
- Shirt
- Top
- Jacket

Feature flow:
1. renter uploads image
2. renter chooses supported clothing item
3. system runs CatVTON inference
4. system returns generated try-on image

## Why CatVTON was chosen

- easier than IDM-VTON for a beginner team
- lighter inference path
- good enough output quality for a college project
- realistic for a 3-month deadline

## What is not included in V1

- training a custom try-on model
- saree / lehenga / kurta try-on
- 3D fitting
- video try-on
- multi-angle try-on
