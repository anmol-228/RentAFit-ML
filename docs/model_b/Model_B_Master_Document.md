# RentAFit Model B Handbook

Date: 2026-03-14

## 1. Purpose Of This Document

This document is the full working handbook for **Model B** in the RentAFit project.

It is written for two kinds of readers:

- a **teacher or evaluator** who wants technical clarity,
- a **student or beginner** who wants to actually understand how the model works from the beginning.

So this document tries to do both:

- keep the explanation simple,
- keep the content technically correct,
- explain the model with real datasets, real code files, real outputs, real graphs, and real reasoning.

This handbook covers:

- what Model B does,
- why it exists,
- what data it uses,
- how labels are created,
- how the LSTM is used,
- how the tabular branch works,
- why the final model is hybrid,
- how the model was trained,
- what results were obtained,
- how to read every graph,
- how the model drives reapproval and listing lifecycle,
- what is fully implemented,
- what still belongs to later backend integration.

---

## 2. One-Page Summary

Model B is the **listing moderation model** of RentAFit.

Its job is to classify a listing into one of three classes:

- `Approve`
- `Review`
- `Reject`

It uses a **hybrid model**:

- an **LSTM branch** for a short sequence based on `condition` and `age`,
- a **tabular branch** for normal listing and pricing features.

The model is trained on:

- structured listing features from `pricing_features`,
- human-reviewed labels from the gold-label dataset.

The current implementation is fully working on the Python/ML side and includes:

- training,
- inference,
- stale-listing logic,
- review-queue generation,
- date-based listing-age handling,
- website-ready status suggestions,
- gender-aware moderation policy.

Current final test performance:

- **Test accuracy:** `0.9771986970684039`
- **Test macro F1:** `0.976529133762697`

Important honest interpretation:

- the model is strong,
- the implementation is valid,
- but many features are strongly connected to rule-based moderation logic,
- so the result should be presented as a **hybrid operational moderation classifier**, not as free-form human reasoning.

---

## 3. What Model B Actually Does

Model B is different from Model A.

- **Model A** predicts rental price range.
- **Model B** decides whether a listing is good enough to go live, should go to review, or should be rejected.

So Model B is a **classification model**, not a regression model.

### Final class meaning

- `Approve`: listing is acceptable and can move toward activation.
- `Review`: listing is not clearly bad, but should be checked manually or rechecked.
- `Reject`: listing is too risky, too poor, or too unrealistic to go live.

### Why Model B is needed

A rental marketplace should not accept every listing automatically.
Problems Model B is designed to help with:

1. poor-condition items,
2. very old or stale listings,
3. unrealistic provider rental prices,
4. borderline listings that need manual moderation,
5. lifecycle reapproval for older listings.

So Model B protects:

- platform quality,
- renter trust,
- moderation workflow,
- long-term listing hygiene.

---

## 4. What Inputs Model B Uses

### 4.1 Current V1 inputs

Model B currently uses **metadata and pricing behavior**, not text descriptions and not image features.

Main input fields:

- `brand`
- `category`
- `gender`
- `material`
- `size`
- `condition`
- `age_months` or `garment_age_months`
- `original_price`
- `provider_price`
- `tier_primary`
- `deviation_M`
- `condition_penalty`
- `age_penalty`
- `deviation_penalty`
- `total_penalty`
- `older_listing_flag`
- `age_policy_override_applied`
- `gender_conflict_flag`
- `gender_policy_override_applied`

### 4.2 Gender-aware moderation policy

Model B now includes a shared gender policy.

Project-wide gender values:

- `Women`
- `Men`
- `Unisex`

Current category policy:

- `Women`: `Saree`, `Lehenga`, `Dress`, `Top`
- `Unisex`: `Shirt`, `Jacket`, `Jeans`, `Activewear`, `Ethnic Wear`, `Kurta`

Runtime behavior:

- if user leaves gender blank, Model B derives it from category
- if category is unisex, explicit `Men` and `Women` are both allowed
- if user-selected gender conflicts with the current category policy, Model B pushes the listing to `Review`
- runtime also returns a popup recommendation so the website can warn the user immediately

Training behavior:

- base rows keep their category-derived gender
- unisex rows are expanded into explicit `Women` and explicit `Men` training variants
- women-specific rows are expanded into explicit `Men` conflict variants
- conflict variants are operationally relabeled to `Review` unless the row was already `Reject`

### 4.3 What is not used in current V1

Not used in current Model B:

- listing text description,
- image embeddings,
- CNN features,
- NLP features.

### 4.4 Why we intentionally kept it this way

This was the right decision for your project because:

- the team is learning ML for the first time,
- metadata was already enough to build a strong moderation baseline,
- adding image or text modeling would make the pipeline much heavier,
- the project needed a realistic path under deadline.

So current Model B is a **metadata-first moderation model**.

---

## 5. Where Model B Fits In The Full RentAFit System

Basic system flow:

1. provider submits listing,
2. Model A suggests rental price range,
3. Model B predicts listing moderation class,
4. backend converts that into website status,
5. old listings later re-enter the review workflow,
6. reviewer/admin decides whether they remain active.

This means Model B is not only a first-time moderation tool.
It is also part of the **listing lifecycle control system**.

---

## 6. Main Files Used In Model B

### 6.1 Data files

- `/Users/mypc/RentAFit/data/frozen/v2_gender/pricing_features.csv`
- `/Users/mypc/RentAFit/data/frozen/v2_gender/gold_labels_model_b_full_manual.csv`
- `/Users/mypc/RentAFit/data/frozen/v2_gender/model_b_train_manual.csv`
- `/Users/mypc/RentAFit/data/frozen/v2_gender/model_b_train_manual_gender_policy.csv`
- `/Users/mypc/RentAFit/data/generated/model_b_expansion_candidates_manual_review.csv`
- `/Users/mypc/RentAFit/data/generated/gold_labels_model_b_expanded.csv`
- `/Users/mypc/RentAFit/data/generated/model_b_train_expanded_gender.csv`
- `/Users/mypc/RentAFit/data/generated/model_b_train_expanded_gender_policy.csv`
- `/Users/mypc/RentAFit/data/generated/model_b_train_expanded_gender_ready.csv`
- `/Users/mypc/RentAFit/data/generated/model_b_live_listings_sample.csv`
- `/Users/mypc/RentAFit/data/templates/model_b_live_listings_template.csv`

### 6.2 Code files

- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_dataset.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_expansion_candidates.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_expanded_dataset.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_gender_policy_dataset.py`
- `/Users/mypc/RentAFit/code/model_b/prepare_model_b_splits.py`
- `/Users/mypc/RentAFit/code/model_b/gender_policy.py`
- `/Users/mypc/RentAFit/code/model_b/training/train_model_b_lstm.py`
- `/Users/mypc/RentAFit/code/model_b/lifecycle.py`
- `/Users/mypc/RentAFit/code/model_b/runtime.py`
- `/Users/mypc/RentAFit/code/model_b/inference/predict_model_b_decision.py`
- `/Users/mypc/RentAFit/code/model_b/ops/generate_live_listings_sample.py`
- `/Users/mypc/RentAFit/code/model_b/ops/run_model_b_live_ops.py`
- `/Users/mypc/RentAFit/code/model_b/analysis/model_b_branch_comparison.py`

### 6.3 Saved artifacts and reports

- `/Users/mypc/RentAFit/models/model_b/model_b_lstm.pt`
- `/Users/mypc/RentAFit/models/model_b/model_b_tabular_preprocessor.joblib`
- `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_metrics.json`
- `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_confusion_matrix_test.png`
- `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_training_history.png`
- `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_test_predictions.csv`
- `/Users/mypc/RentAFit/reports/model_b/model_b_split_summary.json`
- `/Users/mypc/RentAFit/reports/model_b/model_b_branch_comparison_metrics.json`
- `/Users/mypc/RentAFit/reports/model_b/model_b_branch_comparison_chart.png`
- `/Users/mypc/RentAFit/reports/model_b/model_b_dataset_distribution_chart.png`
- `/Users/mypc/RentAFit/reports/model_b/model_b_hybrid_architecture.png`
- `/Users/mypc/RentAFit/reports/model_b/model_b_lifecycle_flow.png`
- `/Users/mypc/RentAFit/reports/model_b/model_b_data_pipeline.png`
- `/Users/mypc/RentAFit/reports/model_b/model_b_live_listings_scored_sample.csv`
- `/Users/mypc/RentAFit/reports/model_b/model_b_review_queue_sample.csv`
- `/Users/mypc/RentAFit/reports/model_b/model_b_live_ops_summary.json`

---

## 7. Dataset Journey: How The Data Was Built

Model B did not begin from one ready-made dataset. It was built in stages.

### Stage 1: Manual feature base
From:
- `/Users/mypc/RentAFit/data/frozen/v2_gender/pricing_features.csv`

This gave structured listing behavior:
- metadata,
- prices,
- deviation,
- penalties,
- rule support signals.

### Stage 2: Manual human labels
From:
- `/Users/mypc/RentAFit/data/frozen/v2_gender/gold_labels_model_b_full_manual.csv`

This gave human-reviewed truth:
- quality score,
- final listing decision,
- reviewer note.

### Stage 3: Manual LSTM-ready dataset
These two were joined into:
- `/Users/mypc/RentAFit/data/frozen/v2_gender/model_b_train_manual.csv`

This added:
- `condition_token`
- `age_bin_token`
- `older_listing_flag`
- `effective_listing_decision`

### Stage 4: Expansion review batch
A hard-case review batch was generated from the augmented pricing set:
- `/Users/mypc/RentAFit/data/generated/model_b_expansion_candidates_manual_review.csv`

This was chosen to focus on:
- older listings,
- high deviation rows,
- used-condition rows,
- premium tiers,
- borderline review-like rows.

### Stage 5: Expanded gold labels
After manual review, we merged:
- original 375 rows,
- plus 400 reviewed expansion rows.

This produced:
- `/Users/mypc/RentAFit/data/generated/gold_labels_model_b_expanded.csv`

### Stage 6: Expanded gender-aware base dataset
The expanded labels were joined with the augmented pricing features to produce:
- `/Users/mypc/RentAFit/data/generated/model_b_train_expanded_gender.csv`

This is the 775-row base dataset before explicit gender variants are added.

### Stage 7: Split-safe gender-policy training dataset
The base dataset was first split into train, validation, and test on the original 775 rows.
Then gender variants were generated **inside each split**:

- `/Users/mypc/RentAFit/data/generated/model_b_train_expanded_gender_ready.csv`

This is the final dataset used for Model B training.

---

## 8. Data Pipeline Diagram

Image file:

- `/Users/mypc/RentAFit/reports/model_b/model_b_data_pipeline.png`

![Model B Data Pipeline](/Users/mypc/RentAFit/reports/model_b/model_b_data_pipeline.png)

Quick pointers:
- This shows that Model B training truth came from human-reviewed moderation labels, not only generated rules.
- The expansion batch improved coverage, but rows were still reviewed before becoming final labels.
- That is why Model B can be defended as a true moderation model rather than just a rule-copying system.

### What this diagram shows

This diagram explains the complete journey of the dataset:

- rule-based pricing features,
- human gold labels,
- first manual dataset,
- expansion review batch,
- expanded gold labels,
- final train-ready dataset.

### Why this diagram matters

It helps the reader understand that Model B was not trained on random rows.
It was built in a controlled way where:

- candidate rows were engineered,
- final decisions were reviewed by humans,
- the final training data became stronger over time.

---

## 9. Current Dataset Counts

### 9.1 Original manual gold dataset
File:
- `/Users/mypc/RentAFit/data/frozen/v2_gender/gold_labels_model_b_full_manual.csv`

Counts:
- rows: **375**
- Approve: **193**
- Review: **98**
- Reject: **84**

### 9.2 Manual gender-aware LSTM-ready dataset
File:
- `/Users/mypc/RentAFit/data/frozen/v2_gender/model_b_train_manual.csv`

Counts:
- rows: **375**
- unique listing ids: **375**
- older listings (`age_months >= 10`): **41**
- age-policy override rows: **20**
- gender counts:
  - `Unisex`: **238**
  - `Women`: **137**

### 9.3 Manual gender-policy dataset
File:
- `/Users/mypc/RentAFit/data/frozen/v2_gender/model_b_train_manual_gender_policy.csv`

Counts:
- total rows: **988**
- base rows: **375**
- explicit `Women` variants on unisex rows: **238**
- explicit `Men` variants on unisex rows: **238**
- explicit `Men` conflict variants on women-specific rows: **137**
- gender-policy override rows: **110**

Effective decision distribution:
- Approve: **173**
- Review: **731**
- Reject: **84**

### 9.4 Expanded gold-label dataset
File:
- `/Users/mypc/RentAFit/data/generated/gold_labels_model_b_expanded.csv`

Counts:
- total rows: **775**
- original rows: **375**
- added reviewed rows: **400**

Decision distribution:
- Approve: **224**
- Review: **337**
- Reject: **214**

### 9.5 Expanded gender-aware base dataset
File:
- `/Users/mypc/RentAFit/data/generated/model_b_train_expanded_gender.csv`

Counts:
- total rows: **775**
- gender counts:
  - `Unisex`: **499**
  - `Women`: **276**

### 9.6 Final split-ready gender-policy training dataset
File:
- `/Users/mypc/RentAFit/data/generated/model_b_train_expanded_gender_ready.csv`

Counts:
- base rows before gender expansion: **775**
- final rows after gender expansion: **2049**
- train: **1435**
- val: **307**
- test: **307**

Effective decision distribution:
- Approve: **456**
- Review: **1013**
- Reject: **580**

Gender distribution:
- Men: **775**
- Women: **775**
- Unisex: **499**

Gender policy counts:
- `gender_conflict_flag = 1`: **276**
- `gender_policy_override_applied = 1`: **214**

Very important technical note:

- the train/val/test split is created on the **base 775 rows first**
- gender variants are generated **inside each split afterward**
- this avoids parent-listing leakage across splits
- verified overlap across split parent ids: **0**

---

## 10. Dataset Distribution Visuals

Image file:

- `/Users/mypc/RentAFit/reports/model_b/model_b_dataset_distribution_chart.png`

![Model B Dataset Distribution](/Users/mypc/RentAFit/reports/model_b/model_b_dataset_distribution_chart.png)

Quick pointers:
- This chart shows how the training data is distributed across decisions and important segments.
- It helps the reader see whether one class dominates too heavily.
- Balanced enough class structure is important for trustworthy moderation metrics.

### What this chart shows

This chart contains three panels:

1. effective decision distribution,
2. age-bin distribution,
3. condition distribution.

### What we learn from it

- The classes are not perfectly balanced, so accuracy alone is not enough.
- Older listings exist, but they are not the majority, so stale-listing logic still matters.
- All three condition states are represented, so the LSTM sequence has meaningful variation.

### Why this chart matters

This chart explains why we use:
- `macro F1`,
- class weights,
- and not only raw accuracy.

---

## 11. How Labels Are Created

Model B labels are not created the same way as Model A.

### 11.1 Why Model A and Model B are different

Model A predicts rule-generated price targets.
That is okay because the target itself is deterministic.

Model B predicts:
- `Approve`
- `Review`
- `Reject`

These should represent **human moderation judgement**.

So the correct workflow is:
- auto-generate candidate rows,
- use rules as support,
- keep final labels human-reviewed.

### 11.2 Human final columns

Human review fills:
- `final_quality_score_0_100`
- `final_listing_decision`
- `final_reviewer_note`

### 11.3 Operational target

For training, we do not use only the raw human label.
We also apply the stale-listing policy.

This produces:
- `effective_listing_decision`
- `effective_decision_label`

This is the current final training target.

---

## 12. Core Rule System Behind Model B

### 12.1 Final classes
Model B predicts:
- `Approve`
- `Review`
- `Reject`

### 12.2 Numeric encoding
- `Approve = 0`
- `Review = 1`
- `Reject = 2`

Used in:
- `decision_label`
- `effective_decision_label`

### 12.3 Older listing rule
If:
- `age_months >= 10`

Then:
- `older_listing_flag = 1`

Otherwise:
- `older_listing_flag = 0`

### 12.4 Listing lifecycle action rule
If:
- `age_months >= 10`

Then:
- `listing_lifecycle_action = needs_reapproval_or_removal`

Otherwise:
- `listing_lifecycle_action = active_current_cycle`

### 12.5 Age policy override rule
If:
- `age_months >= 10`
- and `final_listing_decision == Approve`

Then:
- `effective_listing_decision = Review`

Why:
- a listing approved once should not stay automatically approved forever.

---

## 13. Pricing-Behavior Support Features

Model B uses several support features inherited from the pricing engine.
These are not final labels, but they help moderation.

### 13.1 Deviation
`deviation_M` measures how far the provider price is from the expected rule range.

- if within range -> 0
- if below range -> measured from `rule_min`
- if above range -> measured from `rule_max`

### 13.2 Condition penalty
- `New -> 0`
- `Like New -> 8`
- `Used -> 18`

### 13.3 Age penalty
- `<=3 -> 0`
- `<=6 -> 4`
- `<=9 -> 8`
- `<=12 -> 12`
- `<=18 -> 16`
- `>18 -> 20`

### 13.4 Deviation penalty
- `0 -> 0`
- `<=10 -> 5`
- `<=20 -> 12`
- `<=35 -> 20`
- `<=50 -> 30`
- `>50 -> 45`

### 13.5 Total penalty
- `total_penalty = condition_penalty + age_penalty + deviation_penalty`

### 13.6 Rule quality score
- `rule_quality_score = max(0, 100 - total_penalty)`

### 13.7 Rule decision
Reject if any is true:
- invalid original price,
- invalid provider price,
- provider price too high vs original price,
- very low rule quality score,
- very high deviation.

Approve if both are true:
- rule quality score high,
- deviation low.

Else:
- Review.

### 13.8 Important warning
These are support signals, not the final human truth.
That is why human labels still matter.

---

## 14. Why LSTM Is Used Here

Your teacher asked for LSTM, and Model B is the best place to use it.

### Core sequence used
Each listing creates a small sequence:
- `[condition_token, age_bin_token]`

Example:
- `Like New -> 2`
- `10+ months -> 4`
- sequence becomes `[2, 4]`

### Why this is reasonable
LSTM is made for ordered sequence-style input.
Here, the sequence represents a simple progression of listing state:

- wear state,
- age state.

It is not a large time series, but it is still a meaningful ordered sequence.

### Why not use only LSTM?
Because most moderation information in this project is **tabular**, not purely sequential.
That is exactly why the final model is hybrid.

---

## 15. Tokens Used By The LSTM

### 15.1 Condition token
- `New = 1`
- `Like New = 2`
- `Used = 3`

Stored in:
- `condition_token`

### 15.2 Age bin token
- `<= 3 months = 1`
- `4 to 6 months = 2`
- `7 to 9 months = 3`
- `10+ months = 4`

Stored in:
- `age_bin_token`

### 15.3 Why tokens are needed
LSTM cannot directly use text labels like `Used` or `Like New`.
So these states are converted into integer tokens.

---

## 16. Mermaid Block Diagram

```mermaid
flowchart TD
    A[Provider listing input] --> B[Feature preparation]
    B --> C[Build sequence features]
    B --> D[Build tabular features]

    C --> C1[condition_token]
    C --> C2[age_bin_token]
    C1 --> E[Sequence input: [condition_token, age_bin_token]]
    C2 --> E
    E --> F[Embedding]
    F --> G[LSTM]
    G --> H[LSTM sequence representation]

    D --> D1[Categorical columns: brand, category, material, size, tier]
    D --> D2[Numeric columns: age, prices, deviation, penalties, flags]
    D1 --> I[OneHotEncoder]
    D2 --> J[Imputer + StandardScaler]
    I --> K[Tabular feature matrix]
    J --> K
    K --> L[Dense tabular branch]

    H --> M[Concatenate]
    L --> M
    M --> N[Final dense classifier]
    N --> O[Class probabilities]
    O --> P[Approve / Review / Reject]
    P --> Q[Website status mapping]
```

### How to read this diagram

- the listing first becomes engineered features,
- the sequence branch handles only age/condition sequence,
- the tabular branch handles the rest of the structured features,
- both are merged,
- the final classifier predicts the moderation class,
- then backend logic maps the result into listing status.

---

## 17. Visual Architecture Image

Image file:

- `/Users/mypc/RentAFit/reports/model_b/model_b_hybrid_architecture.png`

![Model B Hybrid Architecture](/Users/mypc/RentAFit/reports/model_b/model_b_hybrid_architecture.png)

Quick pointers:
- The top branch is the LSTM sequence path and the lower branch is the tabular feature path.
- The model uses both because sequence-only was too weak and tabular-only, while strong, still underperformed the hybrid.
- The merge layer is the key idea: lifecycle pattern and structured listing behavior are learned together.

### Why this image is useful

This image makes the hybrid architecture easier to present than text alone.
It clearly shows that the model is **not just LSTM**.

---

## 18. Implemented Model Architecture

The current implementation is a **hybrid PyTorch model** with two branches.

### 18.1 Sequence branch
Input:
- `[condition_token, age_bin_token]`

Layers:
- `Embedding(vocab_size=8, embed_dim=8)`
- `LSTM(hidden_size=16)`

Output:
- last hidden state of the LSTM

### 18.2 Tabular branch
Categorical columns:
- `brand`
- `category`
- `material`
- `size`
- `tier_primary`

Numeric columns:
- `age_months`
- `older_listing_flag`
- `age_policy_override_applied`
- `original_price`
- `provider_price`
- `deviation_M`
- `condition_penalty`
- `age_penalty`
- `deviation_penalty`
- `total_penalty`

Tabular network:
- `Linear(tabular_dim, 64)`
- `ReLU`
- `Dropout(0.20)`
- `Linear(64, 32)`
- `ReLU`

### 18.3 Merge
The LSTM output and tabular branch output are concatenated.

### 18.4 Output head
Final layers:
- `Linear(lstm_hidden + 32, 32)`
- `ReLU`
- `Dropout(0.20)`
- `Linear(32, 3)`

Class order:
- `Approve`
- `Review`
- `Reject`

---

## 19. Why The Hybrid Model Is Better Than Individual Branches

This is one of the most important parts of the document.

A direct comparison study was run using the same test split.

Files:
- `/Users/mypc/RentAFit/code/model_b/analysis/model_b_branch_comparison.py`
- `/Users/mypc/RentAFit/reports/model_b/model_b_branch_comparison_metrics.json`
- `/Users/mypc/RentAFit/reports/model_b/model_b_branch_comparison_chart.png`

![Model B Branch Comparison](/Users/mypc/RentAFit/reports/model_b/model_b_branch_comparison_chart.png)

Quick pointers:
- This chart is the main proof behind the final Model B design choice.
- It shows that LSTM-only is not enough, tabular-only is strong, and the hybrid model is best.
- That is why the final system uses LSTM as one branch, not as the entire model.

### 19.1 Test-set comparison results

- **Rule-only**
  - accuracy: `0.9186`
  - macro F1: `0.9202`

- **LSTM-only**
  - accuracy: `0.4202`
  - macro F1: `0.4185`

- **Tabular-only**
  - accuracy: `0.9674`
  - macro F1: `0.9666`

- **Hybrid LSTM + Tabular**
  - accuracy: `0.9772`
  - macro F1: `0.9765`

### 19.2 What this proves

#### Why LSTM-only is weak
LSTM-only sees only:
- condition token,
- age token.

That is too little information.
It cannot see:
- prices,
- deviation,
- penalties,
- brand,
- category,
- tier,
- material,
- size.

So it performs poorly.

#### Why tabular-only is already strong
Most business signals in this project are structured tabular signals.
That is why tabular-only already performs well.

#### Why hybrid is still best
The hybrid model improves beyond tabular-only.
This means:
- the sequence branch alone is weak,
- but it still adds extra useful signal when combined with the tabular branch.

So the final conclusion is:
- `LSTM-only` is not enough,
- `tabular-only` is strong,
- `hybrid` is the best final design.

This is the strongest data-backed answer to the question:

**“Why are we using both branches together?”**

---

## 20. Preprocessing Used

### 20.1 Categorical preprocessing
Used:
- `OneHotEncoder(handle_unknown='ignore')`

Applied to:
- `brand`
- `category`
- `material`
- `size`
- `tier_primary`

### 20.2 Numeric preprocessing
Used:
- `SimpleImputer(strategy='median')`
- `StandardScaler()`

### 20.3 Why preprocessing matters

- one-hot encoding makes categories machine-readable,
- scaling helps neural training behave more stably,
- `handle_unknown='ignore'` prevents inference crashes on unseen values.

Saved preprocessor:
- `/Users/mypc/RentAFit/models/model_b/model_b_tabular_preprocessor.joblib`

---

## 21. Technology Stack Used And Why

### Python
Main language for the ML workflow.

### Pandas
Used for reading CSV files, joining datasets, generating training tables, and creating analysis summaries.

### NumPy
Used for numeric operations, class-weight calculation, and batching support.

### scikit-learn
Used for:
- preprocessing,
- one-hot encoding,
- scaling,
- imputation,
- evaluation metrics.

### PyTorch
Used for:
- embedding layer,
- LSTM layer,
- dense branches,
- custom hybrid architecture.

### Joblib
Used to save the sklearn preprocessor.

### Matplotlib and Seaborn
Used to generate:
- training-history graph,
- confusion matrix graph,
- branch comparison graph,
- dataset distribution graph,
- architecture and lifecycle visuals.

### Why this stack is suitable
This combination is practical for a student ML project because:
- pandas and sklearn manage structured data cleanly,
- PyTorch gives freedom for a custom LSTM hybrid,
- matplotlib and seaborn make the output explainable.

---

## 22. Training Setup

### 22.1 Training target
Used target:
- `effective_decision_label`

Why:
- it reflects the stale-listing policy,
- it is closer to actual platform behavior than raw original decision alone.

### 22.2 Loss function
Used:
- `CrossEntropyLoss`

Why:
- this is a multiclass classification problem.

### 22.3 Optimizer
Used:
- `Adam`

Hyperparameters:
- learning rate: `0.001`
- weight decay: `0.0001`

### 22.4 Class weights
Train class counts:
- class 0 (Approve): `143`
- class 1 (Review): `250`
- class 2 (Reject): `149`

Class weights used:
- class 0: `1.1849597590846914`
- class 1: `0.6777969821964436`
- class 2: `1.137243258718865`

Why:
- to reduce bias toward the largest class.

### 22.5 Early stopping
Used patience:
- `6`

Training stopped after:
- `17 epochs`

---

## 23. Main Results

From:
- `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_metrics.json`

### Validation
- accuracy: `1.0`
- macro F1: `1.0`
- weighted F1: `1.0`

### Test
- accuracy: `0.9771986970684039`
- macro F1: `0.976529133762697`
- weighted F1: `0.9774529616148748`

### Test confusion matrix
```text
[[72, 0, 0],
 [ 7,142, 0],
 [ 0, 0,86]]
```

This means:
- all `Reject` rows were classified correctly,
- all `Approve` rows were classified correctly,
- the visible test errors came from `Review` rows being predicted as `Approve`,
- so the gender-aware model remains strong, but it is slightly more conservative and realistic than the earlier smaller-dataset version.

---

## 24. Training History Graph

Image file:

- `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_training_history.png`

![Model B Training History](/Users/mypc/RentAFit/reports/model_b/model_b_lstm_training_history.png)

Quick pointers:
- This graph shows how training and validation behavior changed over epochs.
- It helps detect overfitting or unstable optimization.
- The final chosen checkpoint comes from the best validation behavior, not just the last epoch.

### What this graph shows

It tracks:
- train loss,
- validation loss,
- validation macro F1.

### Key observations

- validation macro F1 started around `0.8505`,
- improved strongly during training,
- reached best value `1.0`,
- early stopping at epoch `10` prevented unnecessary overtraining.

### Why this matters

This proves the training process was stable and not random.

---

## 25. Confusion Matrix Graph

Image file:

- `/Users/mypc/RentAFit/reports/model_b/model_b_lstm_confusion_matrix_test.png`

![Model B Test Confusion Matrix](/Users/mypc/RentAFit/reports/model_b/model_b_lstm_confusion_matrix_test.png)

Quick pointers:
- This matrix shows exactly which classes were predicted correctly and which were confused.
- Strong diagonal values mean strong classification performance.
- It is especially useful because moderation quality should not be judged only by one average score.

### How to read it

- rows = true classes,
- columns = predicted classes.

### What it tells us

- 29 of 30 true Approve rows were correct,
- all 54 Review rows were correct,
- all 33 Reject rows were correct.

### Why this matters

This is especially strong for moderation because there are:
- no false approvals of rejected items,
- no rejected items misclassified as approve.

---

## 26. Honest Interpretation Of Results

The results are very strong, but they should be explained honestly.

Why performance is high:
- the data is structured well,
- pricing-rule support features are informative,
- the stale-listing policy creates meaningful separation,
- the final architecture matches the data better than a single-branch model.

So the correct interpretation is:
- this is a strong **hybrid operational moderation classifier**,
- not pure free-form human reasoning.

That is still a very valid and useful result.

---

## 27. Date-Based Lifecycle Logic

Model B is now more than just a classifier.
It also has a **lifecycle layer**.

### 27.1 Important distinction

- `garment_age_months` = age used as model input
- `listing_age_months` = age used for stale-listing reapproval workflow

### 27.2 How listing age is computed
Implemented in:
- `/Users/mypc/RentAFit/code/model_b/lifecycle.py`

Priority used:
1. `last_reapproved_at`
2. `last_approved_at`
3. `listing_created_at`
4. fallback to age input if dates are missing

### 27.3 Why this matters

This means a listing can now become stale because of how long it has existed on the platform, not only because of the item-age value typed by the provider.

---

## 28. Operational Lifecycle Flow

Image file:

- `/Users/mypc/RentAFit/reports/model_b/model_b_lifecycle_flow.png`

![Model B Operational Lifecycle](/Users/mypc/RentAFit/reports/model_b/model_b_lifecycle_flow.png)

Quick pointers:
- This image connects ML output to actual product workflow.
- It shows how a listing can move into reapproval because of listing age even after earlier approval.
- This is the bridge between moderation prediction and real platform status changes.

### What this image shows

- provider submits listing,
- Model B predicts moderation class,
- listing becomes active / pending review / rejected,
- later listing age is computed from dates,
- stale listings go into reapproval queue,
- admin reviewer checks them,
- listing either becomes active again or stays blocked.

### Why this is important

This is the bridge between:
- the ML model,
- and the actual website workflow.

---

## 29. Current Status Mapping Logic

Current Python-side mapping:

- if predicted decision is `Reject` -> `REJECTED`
- else if `listing_age_months >= 10` -> `REAPPROVAL_REQUIRED`
- else if predicted decision is `Approve` -> `ACTIVE`
- else -> `PENDING_REVIEW`

Optional support also exists for:
- `REMOVED`

if stale grace is crossed and auto-removal is enabled.

---

## 30. Review Queue Generation

Implemented in:
- `/Users/mypc/RentAFit/code/model_b/ops/run_model_b_live_ops.py`

This script can take a CSV of current live listings and produce:
- scored listing snapshot,
- review queue,
- summary JSON.

### Current reviewer assignment
- `assigned_reviewer_role = admin_reviewer`

### Queue fields include
- `review_reason`
- `review_priority`
- `stale_listing_flag`
- `removal_recommended`
- `visible_to_renters`

---

## 31. Sample Live-Ops Result

From:
- `/Users/mypc/RentAFit/reports/model_b/model_b_live_ops_summary.json`

Sample run summary:
- input rows: `60`
- review queue rows: `47`
- recommended statuses:
  - `ACTIVE: 13`
  - `PENDING_REVIEW: 9`
  - `REAPPROVAL_REQUIRED: 18`
  - `REJECTED: 20`
- stale listing count: `26`
- removal recommended count: `7`

This proves that the Python-side operational lifecycle logic is now functioning.

---

## 32. Minute-By-Minute Runtime Walkthrough

Let’s describe one full runtime path.

### Step 1: Provider inputs listing
Provider gives:
- brand,
- category,
- material,
- size,
- condition,
- garment age,
- original price,
- provider price.

### Step 2: System derives support features
System computes:
- tokens,
- tier,
- deviation,
- penalties,
- total penalty,
- rule score,
- rule decision.

### Step 3: LSTM branch reads sequence
Sequence:
- `[condition_token, age_bin_token]`

This branch learns the condition-age pattern.

### Step 4: Tabular branch reads structured features
The tabular branch learns from:
- price,
- deviation,
- category,
- brand,
- penalties,
- tier,
- flags.

### Step 5: Both branches are merged
The two learned representations are concatenated.

### Step 6: Final classifier predicts class
Output:
- `Approve`
- `Review`
- `Reject`

### Step 7: Lifecycle logic computes listing status
Using date-based listing age, the backend-ready lifecycle logic suggests:
- `ACTIVE`
- `PENDING_REVIEW`
- `REAPPROVAL_REQUIRED`
- `REJECTED`

### Step 8: Review queue receives stale/problem listings
Rows that need human attention are added to the review queue.

---

## 33. Smoke-Test Examples

### Example 1: good listing but stale by listing age
Input summary:
- Zara / Top / Cotton / S / Like New
- garment age = 3 months
- original price = 2599
- provider price = 180
- listing age from dates = 11 months

Outcome:
- predicted decision = `Approve`
- final suggested status = `REAPPROVAL_REQUIRED`

Meaning:
- the item itself looks fine,
- but the listing is stale and needs reapproval.

### Example 2: old risky listing
Input summary:
- Prada / Dress / Silk / M / Used
- garment age = 11 months
- original price = 95000
- provider price = 20000
- listing age from dates = 14 months

Outcome:
- predicted decision = `Reject`
- final suggested status = `REJECTED`
- removal recommendation = true

Meaning:
- both moderation quality and lifecycle logic agree this is a high-risk listing.

---

## 34. Commands To Reproduce The Main Parts

### Train Model B
```bash
python3 /Users/mypc/RentAFit/code/model_b/training/train_model_b_lstm.py
```

### Single-listing inference
```bash
python3 /Users/mypc/RentAFit/code/model_b/inference/predict_model_b_decision.py \
  --brand "Zara" \
  --category "Top" \
  --material "Cotton" \
  --size "S" \
  --condition "Like New" \
  --garment_age_months 3 \
  --original_price 2599 \
  --provider_price 180 \
  --current_status ACTIVE \
  --listing_created_at 2025-03-01 \
  --last_approved_at 2025-04-01 \
  --as_of_date 2026-03-14 \
  --json
```

### Generate sample live listings
```bash
python3 /Users/mypc/RentAFit/code/model_b/ops/generate_live_listings_sample.py
```

### Run live-ops scoring
```bash
python3 /Users/mypc/RentAFit/code/model_b/ops/run_model_b_live_ops.py --as_of_date 2026-03-14
```

### Generate comparison visuals
```bash
python3 /Users/mypc/RentAFit/code/model_b/analysis/model_b_branch_comparison.py
```

---

## 35. What Is Fully Implemented And What Is Still Left

### Fully implemented now
- manual gold-label workflow,
- expansion-review workflow,
- final expanded training dataset,
- hybrid LSTM + tabular classifier,
- saved trained model,
- inference pipeline,
- branch comparison study,
- date-based lifecycle logic,
- stale-listing handling,
- review queue generation,
- sample live-ops run,
- full documentation with visuals.

### Still future product work
These are not ML gaps anymore. They are integration tasks:
- database persistence for listing dates and statuses,
- scheduled job for automatic stale-listing scanning,
- admin dashboard UI,
- renter/provider UI tied to live status.

So the Python/ML side of Model B is now complete enough for serious project explanation and backend integration.

---

## 36. Glossary For Beginners

### Classification
A machine learning task where the output is a class label, such as `Approve`, `Review`, or `Reject`.

### Token
A numeric code representing a category, such as `Used = 3`.

### Embedding
A layer that converts a token into a learned vector representation.

### LSTM
A type of recurrent neural network designed to learn from ordered sequences.

### Tabular data
Structured data in rows and columns, such as brand, price, and penalties.

### One-hot encoding
A way to convert categories into machine-readable binary columns.

### Macro F1
An evaluation metric that gives equal importance to all classes.

### Confusion matrix
A table showing which classes were predicted correctly and where mistakes happened.

### Early stopping
A training control method that stops learning when validation performance stops improving.

### Stale listing
A listing that has remained active long enough to require reapproval.

### Reapproval
A second moderation check for an already existing listing.

---

## 37. Cross-check test cases (2026-03-14)

Cases: 5
Anomalies: 0

| Case | Brand | Category | Condition | Garment Age | Provider Price | Decision | Status | Listing Age | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | Zara | Top | New | 2 | 180 | Approve | ACTIVE | 1 |  |
| B2 | Biba | Kurta | Like New | 8 | 320 | Review | REAPPROVAL_REQUIRED | 14 | stale listing -> reapproval |
| B3 | Gucci | Dress | Used | 12 | 60000 | Reject | REJECTED | 11 | model rejection |
| B4 | Uniqlo | Jacket | Like New | 5 | 1200 | Reject | REJECTED | 1 | model rejection |
| B5 | H&M | Jeans | Used | 11 | 900 | Reject | REJECTED | 15 | model rejection |

## 38. API Integration

Model B is exposed via the backend API:

- `POST /api/model-b/predict`

This endpoint is used by the lender upload flow to show moderation decisions and suggested listing status.

The runtime output now includes:

- `prediction`
  decision, probabilities, and suggested status
- `lifecycle`
  `listing_age_months`, stale flags, reviewer routing, removal recommendation, and visibility
- `summary`
  compact integration fields for frontend/backend consumers

---

## 39. Final Summary

Model B is now a complete **hybrid moderation and lifecycle model** for RentAFit on the Python side.

It combines:
- human-reviewed labels,
- pricing-behavior support features,
- LSTM sequence learning for age and condition,
- tabular learning for metadata and prices,
- date-based stale-listing logic,
- website-ready status mapping,
- review queue generation.

Most importantly, the design is now supported by data:

- `LSTM-only` is weak,
- `tabular-only` is strong,
- `hybrid` is best.

That makes Model B not only implemented, but also **well explained and defendable**.
