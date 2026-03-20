from pathlib import Path
import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = next(parent.parent for parent in Path(__file__).resolve().parents if parent.name == 'code')
BASE = REPO_ROOT
DATA_PATH = BASE / 'data/generated/model_b_train_expanded_gender_ready.csv'
HYBRID_METRICS_PATH = BASE / 'reports/model_b/model_b_lstm_metrics.json'
REPORT_DIR = BASE / 'reports/model_b'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = REPORT_DIR / 'model_b_branch_comparison_metrics.json'
CHART_PATH = REPORT_DIR / 'model_b_branch_comparison_chart.png'
DATASET_DIST_PATH = REPORT_DIR / 'model_b_dataset_distribution_chart.png'
ARCH_PATH = REPORT_DIR / 'model_b_hybrid_architecture.png'
LIFECYCLE_PATH = REPORT_DIR / 'model_b_lifecycle_flow.png'
DATA_PIPELINE_PATH = REPORT_DIR / 'model_b_data_pipeline.png'

SEED = 42
BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
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


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceOnlyLSTM(nn.Module):
    def __init__(self, vocab_size: int = 8, embed_dim: int = 8, hidden: int = 16, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )

    def forward(self, seq_x):
        emb = self.embedding(seq_x)
        _, (h_n, _) = self.lstm(emb)
        return self.head(h_n[-1])


class TabularOnlyMLP(nn.Module):
    def __init__(self, tabular_dim: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, tab_x):
        return self.net(tab_x)


class RuleOnlyClassifier:
    def predict(self, df: pd.DataFrame):
        rule_quality_score = (100 - df['total_penalty']).clip(lower=0)
        reject_mask = (df['provider_price'] <= 0) | (df['original_price'] <= 0) | (df['provider_price'] > 0.35 * df['original_price']) | (rule_quality_score < 40) | (df['deviation_M'] > 60)
        approve_mask = (~reject_mask) & (rule_quality_score >= 70) & (df['deviation_M'] <= 20)
        out = pd.Series('Review', index=df.index)
        out.loc[approve_mask] = 'Approve'
        out.loc[reject_mask] = 'Reject'
        return out.map({'Approve': 0, 'Review': 1, 'Reject': 2}).astype(int).values


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


def make_batches(*arrays, batch_size=BATCH_SIZE, shuffle=False):
    n = len(arrays[-1])
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        part = idx[start:start + batch_size]
        yield [arr[part] for arr in arrays]


def eval_predictions(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def train_sequence_only(train_seq, y_train, val_seq, y_val, class_weights):
    model = SequenceOnlyLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=to_tensor(class_weights, dtype=torch.float32))

    best_state = None
    best_f1 = -1.0
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        for xb_seq, yb in make_batches(train_seq, y_train, batch_size=BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(xb_seq), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb_seq, yb in make_batches(val_seq, y_val, batch_size=BATCH_SIZE, shuffle=False):
                preds.append(torch.argmax(model(xb_seq), dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)
        f1 = f1_score(y_val.cpu().numpy(), y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_f1


def train_tabular_only(train_tab, y_train, val_tab, y_val, class_weights):
    model = TabularOnlyMLP(tabular_dim=train_tab.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=to_tensor(class_weights, dtype=torch.float32))

    best_state = None
    best_f1 = -1.0
    stale = 0
    for _ in range(EPOCHS):
        model.train()
        for xb_tab, yb in make_batches(train_tab, y_train, batch_size=BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(xb_tab), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for xb_tab, yb in make_batches(val_tab, y_val, batch_size=BATCH_SIZE, shuffle=False):
                preds.append(torch.argmax(model(xb_tab), dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)
        f1 = f1_score(y_val.cpu().numpy(), y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_f1


def predict_seq_model(model, seq_t):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb_seq in make_batches(seq_t, np.arange(len(seq_t)), batch_size=BATCH_SIZE, shuffle=False):
            seq_batch = xb_seq[0]
            preds.append(torch.argmax(model(seq_batch), dim=1).cpu().numpy())
    return np.concatenate(preds)


def predict_tab_model(model, tab_t):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb_tab in make_batches(tab_t, np.arange(len(tab_t)), batch_size=BATCH_SIZE, shuffle=False):
            tab_batch = xb_tab[0]
            preds.append(torch.argmax(model(tab_batch), dim=1).cpu().numpy())
    return np.concatenate(preds)


def make_dataset_charts(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.countplot(data=df, x='effective_listing_decision', order=['Approve', 'Review', 'Reject'], hue='effective_listing_decision', hue_order=['Approve', 'Review', 'Reject'], ax=axes[0], palette='Set2', legend=False)
    axes[0].set_title('Effective Decision Distribution')
    axes[0].set_xlabel('Decision')
    axes[0].set_ylabel('Count')

    sns.countplot(data=df, x='age_bin_token', order=[1,2,3,4], hue='age_bin_token', hue_order=[1,2,3,4], ax=axes[1], palette='Blues', legend=False)
    axes[1].set_title('Age Bin Token Distribution')
    axes[1].set_xlabel('Age Bin Token')
    axes[1].set_ylabel('Count')

    sns.countplot(data=df, x='condition', order=['New', 'Like New', 'Used'], hue='condition', hue_order=['New', 'Like New', 'Used'], ax=axes[2], palette='Oranges', legend=False)
    axes[2].set_title('Condition Distribution')
    axes[2].set_xlabel('Condition')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    fig.savefig(DATASET_DIST_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    boxes = {
        'input': (0.05, 0.70, 0.16, 0.12, 'Provider listing\ninput'),
        'seq': (0.28, 0.78, 0.20, 0.10, 'Sequence branch\ncondition_token + age_bin_token'),
        'emb': (0.53, 0.80, 0.12, 0.08, 'Embedding'),
        'lstm': (0.70, 0.80, 0.12, 0.08, 'LSTM'),
        'tab': (0.28, 0.50, 0.22, 0.12, 'Tabular branch\nbrand/category/material/size\nprices, deviation, penalties'),
        'prep': (0.56, 0.52, 0.18, 0.10, 'One-hot + scaling'),
        'dense': (0.79, 0.52, 0.12, 0.08, 'Dense MLP'),
        'merge': (0.68, 0.30, 0.16, 0.10, 'Concatenate\nsequence + tabular'),
        'head': (0.68, 0.14, 0.16, 0.08, 'Final classifier'),
        'out': (0.68, 0.02, 0.18, 0.08, 'Approve / Review / Reject'),
    }
    for x,y,w,h,label in boxes.values():
        rect = plt.Rectangle((x,y), w,h, facecolor='#f5f7fb', edgecolor='#34495e', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    arrows = [
        ('input','seq'), ('seq','emb'), ('emb','lstm'),
        ('input','tab'), ('tab','prep'), ('prep','dense'),
        ('lstm','merge'), ('dense','merge'), ('merge','head'), ('head','out')
    ]
    centers = {k:(v[0]+v[2]/2, v[1]+v[3]/2) for k,v in boxes.items()}
    for a,b in arrows:
        ax.annotate('', xy=centers[b], xytext=centers[a], arrowprops=dict(arrowstyle='->', lw=1.8, color='#2c3e50'))
    ax.set_title('Model B Hybrid Architecture', fontsize=16, weight='bold')
    fig.savefig(ARCH_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_data_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis('off')
    boxes = {
        'pf': (0.02, 0.62, 0.18, 0.16, 'pricing_features.csv\n(manual feature base)'),
        'gold': (0.02, 0.26, 0.18, 0.16, 'gold_labels_model_b_full_manual.csv\n(human-reviewed labels)'),
        'manual': (0.28, 0.44, 0.20, 0.18, 'model_b_train_manual.csv\nLSTM-ready manual dataset'),
        'cand': (0.55, 0.62, 0.22, 0.16, '400-row expansion candidates\nmanual review batch'),
        'expgold': (0.55, 0.26, 0.22, 0.16, 'gold_labels_model_b_expanded.csv\n375 + 400 reviewed rows'),
        'ready': (0.82, 0.44, 0.16, 0.18, 'model_b_train_expanded_ready.csv\ntrain/val/test ready')
    }
    for x,y,w,h,label in boxes.values():
        rect = plt.Rectangle((x,y), w,h, facecolor='#eef5ff', edgecolor='#28527a', linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    centers = {k:(v[0]+v[2]/2, v[1]+v[3]/2) for k,v in boxes.items()}
    arrows = [('pf','manual'),('gold','manual'),('manual','cand'),('gold','expgold'),('cand','expgold'),('expgold','ready')]
    for a,b in arrows:
        ax.annotate('', xy=centers[b], xytext=centers[a], arrowprops=dict(arrowstyle='->', lw=2, color='#1f3c5b'))
    ax.set_title('Model B Data Pipeline', fontsize=16, weight='bold')
    fig.savefig(DATA_PIPELINE_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_lifecycle_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis('off')
    nodes = {
        'submit': (0.04, 0.50, 0.14, 0.14, 'Provider submits\nlisting'),
        'model': (0.24, 0.50, 0.16, 0.14, 'Model B predicts\nApprove / Review / Reject'),
        'active': (0.49, 0.72, 0.16, 0.12, 'ACTIVE'),
        'review': (0.49, 0.48, 0.20, 0.12, 'PENDING_REVIEW /\nREAPPROVAL_REQUIRED'),
        'reject': (0.49, 0.24, 0.16, 0.12, 'REJECTED'),
        'age': (0.76, 0.70, 0.18, 0.14, 'Listing age computed\nfrom dates'),
        'queue': (0.78, 0.44, 0.18, 0.14, 'Admin reviewer\nqueue'),
        'final': (0.78, 0.18, 0.18, 0.14, 'Back to ACTIVE\nor stay blocked')
    }
    for x,y,w,h,label in nodes.values():
        rect = plt.Rectangle((x,y), w,h, facecolor='#fff8ef', edgecolor='#8e5a2b', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    centers = {k:(v[0]+v[2]/2, v[1]+v[3]/2) for k,v in nodes.items()}
    arrows = [('submit','model'),('model','active'),('model','review'),('model','reject'),('active','age'),('age','review'),('review','queue'),('queue','final')]
    for a,b in arrows:
        ax.annotate('', xy=centers[b], xytext=centers[a], arrowprops=dict(arrowstyle='->', lw=1.8, color='#6b3f1f'))
    ax.set_title('Model B Operational Lifecycle', fontsize=16, weight='bold')
    fig.savefig(LIFECYCLE_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    set_seed()
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

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()

    train_seq_t = to_tensor(X_train_seq, dtype=torch.long)
    val_seq_t = to_tensor(X_val_seq, dtype=torch.long)
    test_seq_t = to_tensor(X_test_seq, dtype=torch.long)
    train_tab_t = to_tensor(X_train_tab)
    val_tab_t = to_tensor(X_val_tab)
    test_tab_t = to_tensor(X_test_tab)
    y_train_t = to_tensor(y_train, dtype=torch.long)
    y_val_t = to_tensor(y_val, dtype=torch.long)

    seq_model, seq_best_val_f1 = train_sequence_only(train_seq_t, y_train_t, val_seq_t, y_val_t, class_weights)
    tab_model, tab_best_val_f1 = train_tabular_only(train_tab_t, y_train_t, val_tab_t, y_val_t, class_weights)

    seq_test_pred = predict_seq_model(seq_model, test_seq_t)
    tab_test_pred = predict_tab_model(tab_model, test_tab_t)
    rule_test_pred = RuleOnlyClassifier().predict(test_df)

    hybrid_metrics = json.loads(HYBRID_METRICS_PATH.read_text())

    metrics = {
        'sequence_only': {
            'best_val_macro_f1': float(seq_best_val_f1),
            **eval_predictions(y_test, seq_test_pred),
        },
        'tabular_only': {
            'best_val_macro_f1': float(tab_best_val_f1),
            **eval_predictions(y_test, tab_test_pred),
        },
        'rule_only': eval_predictions(y_test, rule_test_pred),
        'hybrid_lstm_tabular': {
            'best_val_macro_f1': float(hybrid_metrics['best_val_macro_f1']),
            'accuracy': float(hybrid_metrics['test_accuracy']),
            'macro_f1': float(hybrid_metrics['test_macro_f1']),
            'weighted_f1': float(hybrid_metrics['test_weighted_f1']),
            'confusion_matrix': hybrid_metrics['test_confusion_matrix'],
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    models = ['Rule-only', 'LSTM-only', 'Tabular-only', 'Hybrid']
    macro_f1 = [metrics['rule_only']['macro_f1'], metrics['sequence_only']['macro_f1'], metrics['tabular_only']['macro_f1'], metrics['hybrid_lstm_tabular']['macro_f1']]
    accuracy = [metrics['rule_only']['accuracy'], metrics['sequence_only']['accuracy'], metrics['tabular_only']['accuracy'], metrics['hybrid_lstm_tabular']['accuracy']]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#8ecae6')
    ax.bar(x + width/2, macro_f1, width, label='Macro F1', color='#ffb703')
    for i, v in enumerate(accuracy):
        ax.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(macro_f1):
        ax.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.08)
    ax.set_title('Model B Branch Comparison on Test Set')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    fig.savefig(CHART_PATH, dpi=180, bbox_inches='tight')
    plt.close(fig)

    make_dataset_charts(df)
    make_architecture_diagram()
    make_lifecycle_diagram()
    make_data_pipeline_diagram()

    joblib.dump(pre, REPORT_DIR / 'model_b_branch_comparison_preprocessor.joblib')
    print('Saved metrics:', METRICS_PATH)
    print('Saved comparison chart:', CHART_PATH)
    print('Saved dataset chart:', DATASET_DIST_PATH)
    print('Saved architecture diagram:', ARCH_PATH)
    print('Saved lifecycle diagram:', LIFECYCLE_PATH)
    print('Saved data pipeline diagram:', DATA_PIPELINE_PATH)


if __name__ == '__main__':
    main()
