# Model B LSTM Results

## Current implementation

A first PyTorch-based hybrid Model B has been trained with:

- LSTM sequence branch using `[condition_token, age_bin_token]`
- tabular branch using listing metadata and pricing-behavior features
- final target: `effective_decision_label`

## Main files

- Training script: `code/model_b/training/train_model_b_lstm.py`
- Inference script: `code/model_b/inference/predict_model_b_decision.py`
- Metrics: `reports/model_b/model_b_lstm_metrics.json`
- Confusion matrix: `reports/model_b/model_b_lstm_confusion_matrix_test.png`
- Training history: `reports/model_b/model_b_lstm_training_history.png`
- Test predictions: `reports/model_b/model_b_lstm_test_predictions.csv`

## Current metrics

- Best validation macro F1: `0.9915343915343916`
- Validation accuracy: `0.9913793103448276`
- Test accuracy: `0.9914529914529915`
- Test macro F1: `0.9912921785103407`
- Test weighted F1: `0.9914197656383942`

## Test confusion matrix

```text
[[29, 1, 0],
 [ 0,54, 0],
 [ 0, 0,33]]
```

This means there was only one test misclassification in the current split.

## Important interpretation note

These very strong results are useful, but they should be interpreted honestly.
The current Model B uses features such as:

- deviation
- penalties
- total penalty
- older listing flag

These features are strongly connected to the moderation rules and the effective label policy.
So the current model is a strong operational classifier, but it is still partly learning rule-shaped behavior.

This means:

- the implementation is valid,
- the model is useful,
- but the high score should not be overclaimed as fully independent human-style reasoning.

The best future improvement is to expand more human-reviewed labels and gradually reduce overdependence on rule-derived signals if needed.
