"""
utils/eda.py
Exploratory Data Analysis and Visualization for Fake Job Detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    'legit': '#2ecc71',
    'fraud': '#e74c3c',
    'primary': '#2c3e50',
    'accent': '#3498db',
    'bg': '#f8f9fa',
    'grid': '#dee2e6'
}
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': PALETTE['bg'],
    'axes.facecolor': 'white',
    'axes.edgecolor': PALETTE['grid'],
    'axes.grid': True,
    'grid.color': PALETTE['grid'],
    'grid.alpha': 0.5
})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"  Saved: {path}")


# ── 1. Class Distribution ─────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame):
    counts = df['fraudulent'].value_counts()
    labels = ['Legitimate', 'Fraudulent']
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [PALETTE['legit'], PALETTE['fraud']]
    pct = [v / sum(values) * 100 for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Class Distribution – Job Postings Dataset', fontsize=14, fontweight='bold', color=PALETTE['primary'])

    # Bar chart
    bars = axes[0].bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val, p in zip(bars, values, pct):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f'{val}\n({p:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_title('Count per Class', fontsize=12)
    axes[0].set_ylabel('Number of Postings')
    axes[0].set_ylim(0, max(values) * 1.2)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        values, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    axes[1].set_title('Class Proportion', fontsize=12)

    plt.tight_layout()
    save_fig("01_class_distribution.png")


# ── 2. Missing Values ─────────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame):
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'location']
    miss_pct = {col: (df[col].fillna('').str.strip() == '').mean() * 100 for col in text_cols if col in df.columns}

    fig, ax = plt.subplots(figsize=(10, 5))
    cols = list(miss_pct.keys())
    vals = list(miss_pct.values())
    bar_colors = [PALETTE['fraud'] if v > 20 else PALETTE['accent'] for v in vals]
    bars = ax.barh(cols, vals, color=bar_colors, edgecolor='white', linewidth=1)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('Missing / Empty (%)')
    ax.set_title('Missing Value Analysis by Column', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.set_xlim(0, max(vals) + 15)
    plt.tight_layout()
    save_fig("02_missing_values.png")


# ── 3. Text Length Distribution ──────────────────────────────────────────────
def plot_text_lengths(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Text Length by Class', fontsize=13, fontweight='bold', color=PALETTE['primary'])

    for ax, col in zip(axes, ['description', 'requirements', 'company_profile']):
        for label, color, grp in [(0, PALETTE['legit'], 'Legitimate'), (1, PALETTE['fraud'], 'Fraudulent')]:
            lengths = df[df['fraudulent'] == label][col].fillna('').apply(len)
            ax.hist(lengths, bins=30, alpha=0.6, color=color, label=grp, edgecolor='white')
        ax.set_title(col.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Character Length')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig("03_text_length_distribution.png")


# ── 4. Fraud Keyword Analysis ─────────────────────────────────────────────────
def plot_fraud_keywords(df: pd.DataFrame):
    from utils.text_preprocessor import FRAUD_SIGNALS, combine_text_features

    combined = df.apply(combine_text_features, axis=1).str.lower()

    legit_text = ' '.join(combined[df['fraudulent'] == 0].tolist())
    fraud_text = ' '.join(combined[df['fraudulent'] == 1].tolist())

    top_n = 15
    fraud_kw_counts = Counter()
    legit_kw_counts = Counter()
    for kw in FRAUD_SIGNALS:
        fraud_kw_counts[kw] = fraud_text.count(kw)
        legit_kw_counts[kw] = legit_text.count(kw)

    top_fraud = sorted(fraud_kw_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, fraud_vals = zip(*top_fraud)
    legit_vals = [legit_kw_counts[w] for w in words]

    x = np.arange(len(words))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, fraud_vals, w, label='Fraudulent', color=PALETTE['fraud'], alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, legit_vals, w, label='Legitimate', color=PALETTE['legit'], alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Keyword Frequency')
    ax.set_title('Fraud Signal Keyword Frequency: Fraud vs Legitimate', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.legend()
    plt.tight_layout()
    save_fig("04_fraud_keywords.png")


# ── 5. Structural Feature Analysis ───────────────────────────────────────────
def plot_structural_features(df: pd.DataFrame):
    features = ['has_company_logo', 'has_questions', 'telecommuting']
    fig, axes = plt.subplots(1, len(features), figsize=(14, 5))
    fig.suptitle('Structural Feature Distribution by Class', fontsize=13, fontweight='bold', color=PALETTE['primary'])

    for ax, feat in zip(axes, features):
        if feat not in df.columns:
            continue
        ct = pd.crosstab(df[feat], df['fraudulent'])
        ct.columns = ['Legitimate', 'Fraudulent']
        ct.plot(kind='bar', ax=ax, color=[PALETTE['legit'], PALETTE['fraud']],
                edgecolor='white', rot=0)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig("05_structural_features.png")


# ── 6. Exclamation Mark Analysis ─────────────────────────────────────────────
def plot_exclamation_analysis(df: pd.DataFrame):
    df2 = df.copy()
    df2['exclamation_count'] = df2['description'].fillna('').apply(lambda x: x.count('!'))

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, grp in [(0, PALETTE['legit'], 'Legitimate'), (1, PALETTE['fraud'], 'Fraudulent')]:
        data = df2[df2['fraudulent'] == label]['exclamation_count']
        ax.hist(data, bins=20, alpha=0.7, color=color, label=grp, edgecolor='white')
    ax.set_xlabel('Number of Exclamation Marks in Description')
    ax.set_ylabel('Count')
    ax.set_title('Exclamation Mark Usage by Class', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.legend()
    plt.tight_layout()
    save_fig("06_exclamation_analysis.png")


def run_full_eda(df: pd.DataFrame):
    print("\n📊 Running Exploratory Data Analysis...")
    print(f"  Dataset shape: {df.shape}")
    print(f"  Fraudulent: {df['fraudulent'].sum()} | Legitimate: {(df['fraudulent']==0).sum()}")

    plot_class_distribution(df)
    plot_missing_values(df)
    plot_text_lengths(df)
    plot_fraud_keywords(df)
    plot_structural_features(df)
    plot_exclamation_analysis(df)

    print("  EDA Complete. All plots saved to outputs/")
