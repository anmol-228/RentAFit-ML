from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
BASE = REPO_ROOT
DATA_PATH = BASE / 'data/generated/model_b_train_expanded_gender_ready.csv'
MODEL_DIR = BASE / 'models/model_b'
REPORT_DIR = BASE / 'reports/model_b'

PREPROCESSOR_PATH = MODEL_DIR / 'model_b_tabular_preprocessor.joblib'
MODEL_PATH = MODEL_DIR / 'model_b_lstm.pt'
METRICS_PATH = REPORT_DIR / 'model_b_lstm_metrics.json'
CONFUSION_PATH = REPORT_DIR / 'model_b_lstm_confusion_matrix_test.png'
TRAIN_HISTORY_PATH = REPORT_DIR / 'model_b_lstm_training_history.png'
PREDICTIONS_PATH = REPORT_DIR / 'model_b_lstm_test_predictions.csv'

SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMBED_DIM = 8
LSTM_HIDDEN = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SEQ_COLS = ['condition_token', 'age_bin_token']
CAT_COLS = ['brand', 'category', 'gender', 'material', 'size', 'tier_primary']
NUM_COLS = [
    'age_months', 'older_listing_flag', 'age_policy_override_applied',
    'gender_conflict_flag', 'gender_policy_override_applied',
    'original_price', 'provider_price', 'deviation_M',
    'condition_penalty', 'age_penalty', 'deviation_penalty', 'total_penalty'
]
TARGET_COL = 'effective_decision_label'
TARGET_NAME_COL = 'effective_listing_decision'


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelBHybridLSTM(nn.Module):
    def __init__(self, tabular_dim: int, vocab_size: int = 8, embed_dim: int = EMBED_DIM, lstm_hidden: int = LSTM_HIDDEN, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, batch_first=True)

        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, num_classes),
        )

    def forward(self, seq_x, tab_x):
        emb = self.embedding(seq_x)
        _, (h_n, _) = self.lstm(emb)
        seq_out = h_n[-1]
        tab_out = self.tabular_branch(tab_x)
        combined = torch.cat([seq_out, tab_out], dim=1)
        return self.head(combined)


def make_preprocessor():
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    return ColumnTransformer([
        ('cat', cat_pipe, CAT_COLS),
        ('num', num_pipe, NUM_COLS),
    ])


def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=DEVICE)


def make_batches(seq, tab, y, batch_size=BATCH_SIZE, shuffle=False):
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        part = idx[start:start + batch_size]
        yield seq[part], tab[part], y[part]


def evaluate(model, seq, tab, y_true):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb_seq, xb_tab, _ in make_batches(seq, tab, y_true, shuffle=False):
            logits = model(xb_seq, xb_tab)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
    y_pred = np.concatenate(preds)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=['Approve', 'Review', 'Reject'], output_dict=True, zero_division=0),
        'y_pred': y_pred,
    }


def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    train_df = df[df['split_set'] == 'train'].copy()
    val_df = df[df['split_set'] == 'val'].copy()
    test_df = df[df['split_set'] == 'test'].copy()

    pre = make_preprocessor()
    X_train_tab = pre.fit_transform(train_df[CAT_COLS + NUM_COLS])
    X_val_tab = pre.transform(val_df[CAT_COLS + NUM_COLS])
    X_test_tab = pre.transform(test_df[CAT_COLS + NUM_COLS])

    X_train_tab = X_train_tab.astype(np.float32).toarray() if hasattr(X_train_tab, 'toarray') else X_train_tab.astype(np.float32)
    X_val_tab = X_val_tab.astype(np.float32).toarray() if hasattr(X_val_tab, 'toarray') else X_val_tab.astype(np.float32)
    X_test_tab = X_test_tab.astype(np.float32).toarray() if hasattr(X_test_tab, 'toarray') else X_test_tab.astype(np.float32)

    X_train_seq = train_df[SEQ_COLS].astype(int).values
    X_val_seq = val_df[SEQ_COLS].astype(int).values
    X_test_seq = test_df[SEQ_COLS].astype(int).values

    y_train = train_df[TARGET_COL].astype(int).values
    y_val = val_df[TARGET_COL].astype(int).values
    y_test = test_df[TARGET_COL].astype(int).values

    train_seq_t = to_tensor(X_train_seq, dtype=torch.long)
    val_seq_t = to_tensor(X_val_seq, dtype=torch.long)
    test_seq_t = to_tensor(X_test_seq, dtype=torch.long)

    train_tab_t = to_tensor(X_train_tab)
    val_tab_t = to_tensor(X_val_tab)
    test_tab_t = to_tensor(X_test_tab)

    y_train_t = to_tensor(y_train, dtype=torch.long)
    y_val_t = to_tensor(y_val, dtype=torch.long)
    y_test_t = to_tensor(y_test, dtype=torch.long)

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = to_tensor(class_weights, dtype=torch.float32)

    model = ModelBHybridLSTM(tabular_dim=X_train_tab.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    best_state = None
    best_val_f1 = -1.0
    epochs_without_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb_seq, xb_tab, yb in make_batches(train_seq_t, train_tab_t, y_train_t, shuffle=True):
            optimizer.zero_grad()
            logits = model(xb_seq, xb_tab)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float('nan')

        model.eval()
        with torch.no_grad():
            val_logits = model(val_seq_t, val_tab_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_f1 = float(f1_score(y_val, val_pred, average='macro'))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print(f'Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_macro_f1={val_f1:.4f}')

        if epochs_without_improve >= PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_eval = evaluate(model, val_seq_t, val_tab_t, y_val)
    test_eval = evaluate(model, test_seq_t, test_tab_t, y_test)

    torch.save({
        'model_state_dict': model.state_dict(),
        'tabular_dim': X_train_tab.shape[1],
        'seq_cols': SEQ_COLS,
        'cat_cols': CAT_COLS,
        'num_cols': NUM_COLS,
        'target_col': TARGET_COL,
        'class_names': ['Approve', 'Review', 'Reject'],
    }, MODEL_PATH)
    joblib.dump(pre, PREPROCESSOR_PATH)

    metrics = {
        'device': DEVICE,
        'rows_total': int(len(df)),
        'split_counts': df['split_set'].value_counts().sort_index().to_dict(),
        'train_class_counts': {str(i): int(v) for i, v in enumerate(class_counts)},
        'class_weights': {str(i): float(v) for i, v in enumerate(class_weights)},
        'best_val_macro_f1': best_val_f1,
        'val_accuracy': val_eval['accuracy'],
        'val_macro_f1': val_eval['macro_f1'],
        'val_weighted_f1': val_eval['weighted_f1'],
        'test_accuracy': test_eval['accuracy'],
        'test_macro_f1': test_eval['macro_f1'],
        'test_weighted_f1': test_eval['weighted_f1'],
        'val_confusion_matrix': val_eval['confusion_matrix'],
        'test_confusion_matrix': test_eval['confusion_matrix'],
        'val_classification_report': val_eval['classification_report'],
        'test_classification_report': test_eval['classification_report'],
        'history': history,
    }

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    test_out = test_df[['listing_id', 'effective_listing_decision', 'effective_decision_label']].copy()
    test_out['pred_label'] = test_eval['y_pred']
    label_map = {0: 'Approve', 1: 'Review', 2: 'Reject'}
    test_out['pred_decision'] = test_out['pred_label'].map(label_map)
    test_out.to_csv(PREDICTIONS_PATH, index=False)

    cm = np.array(test_eval['confusion_matrix'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Approve', 'Review', 'Reject'], yticklabels=['Approve', 'Review', 'Reject'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Model B LSTM Confusion Matrix (Test)')
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.plot(history['val_macro_f1'], label='val_macro_f1')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Model B LSTM Training History')
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAIN_HISTORY_PATH, dpi=180)
    plt.close()

    print('Saved model:', MODEL_PATH)
    print('Saved preprocessor:', PREPROCESSOR_PATH)
    print('Saved metrics:', METRICS_PATH)
    print('Saved confusion matrix:', CONFUSION_PATH)
    print('Saved training history:', TRAIN_HISTORY_PATH)
    print('Saved test predictions:', PREDICTIONS_PATH)
    print('Best val macro F1:', best_val_f1)
    print('Test macro F1:', test_eval['macro_f1'])
    print('Test accuracy:', test_eval['accuracy'])


if __name__ == '__main__':
    main()
