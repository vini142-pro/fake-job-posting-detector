"""
models/trainer.py
Model training, evaluation, and comparison for Fake Job Detection
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score
)
from scipy.sparse import hstack, csr_matrix

PALETTE = {
    'legit': '#2ecc71', 'fraud': '#e74c3c',
    'primary': '#2c3e50', 'accent': '#3498db',
    'bg': '#f8f9fa', 'grid': '#dee2e6'
}
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  Saved: {path}")


# ── Feature Preparation ───────────────────────────────────────────────────────
def prepare_features(df_train, df_test, eng_train, eng_test, vectorizer_type='tfidf'):
    """Build TF-IDF + engineered feature matrix."""
    print(f"\n🔧 Building features using {vectorizer_type.upper()}...")

    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    else:
        vectorizer = CountVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2
        )

    X_text_train = vectorizer.fit_transform(df_train['cleaned_text'].fillna(''))
    X_text_test = vectorizer.transform(df_test['cleaned_text'].fillna(''))

    # Combine with engineered features
    X_eng_train = csr_matrix(eng_train.fillna(0).values)
    X_eng_test = csr_matrix(eng_test.fillna(0).values)

    X_train = hstack([X_text_train, X_eng_train])
    X_test = hstack([X_text_test, X_eng_test])

    print(f"  Training matrix: {X_train.shape}")
    print(f"  Test matrix:     {X_test.shape}")

    return X_train, X_test, vectorizer


# ── Model Definitions ─────────────────────────────────────────────────────────
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs'
        ),
        "Naive Bayes (Complement)": ComplementNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        ),
        "Linear SVM": LinearSVC(
            C=1.0, class_weight='balanced', max_iter=2000, random_state=42
        ),
    }


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    has_proba = hasattr(model, 'predict_proba')
    has_decision = hasattr(model, 'decision_function')

    if has_proba:
        y_prob = model.predict_proba(X_test)[:, 1]
    elif has_decision:
        scores = model.decision_function(X_test)
        # normalize to [0,1]
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        'name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model': model
    }
    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(results, y_test):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    fig.suptitle('Confusion Matrices – All Models', fontsize=14, fontweight='bold', color=PALETTE['primary'])

    for i, r in enumerate(results):
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                    cmap='RdYlGn', linewidths=0.5,
                    xticklabels=['Legit', 'Fraud'],
                    yticklabels=['Legit', 'Fraud'])
        axes[i].set_title(f"{r['name']}\nF1={r['f1']:.3f} | AUC={r['roc_auc']:.3f}", fontsize=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_fig("07_confusion_matrices.png")


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PALETTE['fraud'], PALETTE['legit'], PALETTE['accent'], '#9b59b6', '#e67e22']

    for r, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        ax.plot(fpr, tpr, label=f"{r['name']} (AUC={r['roc_auc']:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curves – Model Comparison', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.legend(loc='lower right', fontsize=9)
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    plt.tight_layout()
    save_fig("08_roc_curves.png")


def plot_precision_recall(results, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PALETTE['fraud'], PALETTE['legit'], PALETTE['accent'], '#9b59b6', '#e67e22']

    for r, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, r['y_prob'])
        ax.plot(rec, prec, label=f"{r['name']} (AP={r['avg_precision']:.3f})",
                color=color, linewidth=2)

    baseline = y_test.mean()
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.2f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    save_fig("09_precision_recall_curves.png")


def plot_metric_comparison(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=metric.upper().replace('_', ' '),
                      color=color, alpha=0.85, edgecolor='white')

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    plt.tight_layout()
    save_fig("10_model_comparison.png")


def plot_feature_importance(model, vectorizer, eng_feature_names, top_n=20):
    """For Logistic Regression: show top fraud/legit features."""
    if not hasattr(model, 'coef_'):
        return

    coef = model.coef_[0]
    tfidf_names = vectorizer.get_feature_names_out().tolist()
    all_names = tfidf_names + eng_feature_names

    if len(coef) != len(all_names):
        return

    idx_top_fraud = np.argsort(coef)[-top_n:][::-1]
    idx_top_legit = np.argsort(coef)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Top Feature Importances (Logistic Regression Coefficients)',
                 fontsize=13, fontweight='bold', color=PALETTE['primary'])

    for ax, idx, title, color in [
        (axes[0], idx_top_fraud, 'Top Fraud Indicators', PALETTE['fraud']),
        (axes[1], idx_top_legit, 'Top Legitimate Indicators', PALETTE['legit'])
    ]:
        feats = [all_names[i] for i in idx]
        vals = [abs(coef[i]) for i in idx]
        ax.barh(feats[::-1], vals[::-1], color=color, alpha=0.8, edgecolor='white')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('|Coefficient|')

    plt.tight_layout()
    save_fig("11_feature_importance.png")


def plot_vectorizer_comparison(df_train, df_test, eng_train, eng_test, y_train, y_test):
    """Compare TF-IDF vs CountVectorizer on Logistic Regression."""
    print("\n🔬 Comparing TF-IDF vs CountVectorizer...")
    results = {}

    for vtype in ['tfidf', 'count']:
        X_train, X_test, vec = prepare_features(df_train, df_test, eng_train, eng_test, vtype)
        model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[vtype] = {
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'accuracy': accuracy_score(y_test, y_pred)
        }

    labels = list(results.keys())
    metrics = ['f1', 'roc_auc', 'accuracy']
    x = np.arange(len(metrics))
    w = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, [results['tfidf'][m] for m in metrics], w,
           label='TF-IDF', color=PALETTE['accent'], edgecolor='white')
    ax.bar(x + w/2, [results['count'][m] for m in metrics], w,
           label='CountVectorizer', color='#e67e22', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics])
    ax.set_ylabel('Score')
    ax.set_ylim(0.7, 1.05)
    ax.set_title('TF-IDF vs CountVectorizer (Logistic Regression)',
                 fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.legend()
    plt.tight_layout()
    save_fig("12_vectorizer_comparison.png")
    return results


# ── Main Training Pipeline ────────────────────────────────────────────────────
def run_training(df: pd.DataFrame, eng_features: pd.DataFrame):
    print("\n🤖 Starting Model Training Pipeline...")

    y = df['fraudulent'].values

    # Train/test split
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    df_train, df_test = df.iloc[idx_train], df.iloc[idx_test]
    eng_train, eng_test = eng_features.iloc[idx_train], eng_features.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")
    print(f"  Train fraud rate: {y_train.mean():.1%} | Test fraud rate: {y_test.mean():.1%}")

    # Build features
    X_train, X_test, vectorizer = prepare_features(df_train, df_test, eng_train, eng_test, 'tfidf')

    # Train all models
    models = get_models()
    results = []
    print("\n📈 Training models...")

    for name, model in models.items():
        print(f"  ▸ {name}...")
        r = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(r)
        print(f"    F1={r['f1']:.4f} | AUC={r['roc_auc']:.4f} | Acc={r['accuracy']:.4f}")

    # Plots
    print("\n📊 Generating evaluation plots...")
    plot_confusion_matrix(results, y_test)
    plot_roc_curves(results, y_test)
    plot_precision_recall(results, y_test)
    plot_metric_comparison(results)

    # Feature importance for best model (LR)
    lr_result = next(r for r in results if 'Logistic' in r['name'])
    plot_feature_importance(
        lr_result['model'], vectorizer,
        list(eng_features.columns)
    )

    # Vectorizer comparison
    plot_vectorizer_comparison(df_train, df_test, eng_train, eng_test, y_train, y_test)

    # Save best model
    best = max(results, key=lambda r: r['f1'])
    print(f"\n🏆 Best Model: {best['name']} (F1={best['f1']:.4f})")

    joblib.dump(best['model'], os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))

    # Print summary table
    print("\n" + "═" * 65)
    print(f"{'Model':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("─" * 65)
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        print(f"{r['name']:<28} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f} {r['roc_auc']:>6.4f}")
    print("═" * 65)

    return results, best['model'], vectorizer, eng_features.columns.tolist()
