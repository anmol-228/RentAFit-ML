# RentAFit Model Cross-Check Report

Generated: 2026-03-18

## Model A Summary

Cases: 7
Anomalies: 0

| Case | Brand | Category | Condition | Age | Original | Final Min | Final Max | Source | Confidence | Fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Zara | Top | New | 2 | 1200 | 60 | 70 | model_output | 1.00 | False |
| A2 | H&M | Jeans | Like New | 6 | 2400 | 120 | 140 | model_output | 1.00 | False |
| A3 | Biba | Kurta | Like New | 4 | 4200 | 250 | 300 | model_output | 1.00 | False |
| A4 | Anita Dongre | Lehenga | New | 2 | 22000 | 2800 | 3500 | model_output | 0.90 | False |
| A5 | Uniqlo | Jacket | Used | 12 | 8000 | 400 | 450 | model_output | 1.00 | False |
| A6 | BrandX | Dress | Like New | 3 | 3000 | 200 | 250 | rule_fallback | 0.45 | True |
| A7 | Louis Vuitton | Shirt | Like New | 5 | 70499 | 6500 | 8300 | model_output | 0.90 | False |

## Model B Summary

Cases: 6
Anomalies: 0

| Case | Category | Input Gender | Resolved Gender | Conflict | Decision | Status | Listing Age | Stale | Auto Removed | Popup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | Top | Women | Women | 0 | Review | PENDING_REVIEW | 1 | False | False | False |
| B2 | Dress | Men | Men | 1 | Review | PENDING_REVIEW | 1 | False | False | True |
| B3 | Kurta | Women | Women | 0 | Review | REAPPROVAL_REQUIRED | 14 | True | False | False |
| B4 | Dress | Women | Women | 0 | Reject | REJECTED | 11 | True | False | False |
| B5 | Jeans | Men | Men | 0 | Review | REMOVED | 15 | True | True | False |
| B6 | Jacket | Men | Men | 0 | Reject | REJECTED | 2 | False | False | False |

## Model C Summary

Cases: 5
Anomalies: 0

| Case | Mode | Seed/Profile | Category | Gender | Returned | Budget Source | Review Used | Top Recs | Top Pools |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | item_to_item | L0007 | Dress | Women | 5/5 | category_average_budget | 0 | L1499, L1189, L0107 | PRIMARY_APPROVE, PRIMARY_APPROVE, PRIMARY_APPROVE |
| C2 | item_to_item | L0037 | Shirt | Unisex | 5/5 | category_average_budget | 0 | L0470, L0295, L0142 | PRIMARY_APPROVE, PRIMARY_APPROVE, PRIMARY_APPROVE |
| C3 | profile_from_liked_items | L0007,L0015,L0051 | Dress | Women | 5/5 | category_average_budget | 0 | L0884, L0862, L0350 | PRIMARY_APPROVE, PRIMARY_APPROVE, PRIMARY_APPROVE |
| C4 | item_to_item | L0003 | Activewear | Unisex | 4/5 | category_average_budget | 2 | L0185, L1020, L0603 | PRIMARY_APPROVE, PRIMARY_APPROVE, REVIEW_FALLBACK |
| C5 | item_to_item | L0007 | Dress | Women | 5/5 | explicit_user_budget | 0 | L1499, L1189, L0107 | PRIMARY_APPROVE, PRIMARY_APPROVE, PRIMARY_APPROVE |

## Documentation And Visual Audit

Documentation anomalies: 0

| Document | Exists | Embedded Images |
| --- | --- | --- |
| Model A handbook | True | 6 |
| Model B handbook | True | 7 |
| Model C handbook | True | 6 |
| Project master handbook | True | 12 |

| Visual Group | Required Visuals | Missing Visuals |
| --- | --- | --- |
| model_a | 6 | None |
| model_b | 7 | None |
| model_c | 6 | None |
| project | 4 | None |