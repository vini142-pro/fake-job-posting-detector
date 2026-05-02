"""
main.py — Fake Job Posting Detection System
============================================
A complete ML pipeline for detecting fraudulent job postings.

Usage:
  python main.py            # Full pipeline: EDA + Train + Evaluate + Predict
  python main.py --train    # Train only
  python main.py --predict  # Prediction demo only (requires trained model)
  python main.py --eda      # EDA only
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# ── Step 0: ensure data dir & generate dataset if needed ──────────────────────
DATA_PATH = os.path.join("data", "fake_job_postings.csv")

def ensure_dataset():
    if not os.path.exists(DATA_PATH):
        print("📦 Dataset not found. Generating synthetic dataset...")
        os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
        from data.generate_dataset import generate_dataset
        os.makedirs("data", exist_ok=True)
        df = generate_dataset(2000)
        df.to_csv(DATA_PATH, index=False)
        print(f"   Generated {len(df)} records → {DATA_PATH}")
    else:
        print(f"📦 Dataset found: {DATA_PATH}")


def load_and_preprocess():
    from utils.text_preprocessor import preprocess_dataframe, extract_engineered_features

    print("\n⚙️  Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}")
    print(f"   Fraud: {df['fraudulent'].sum()} | Legit: {(df['fraudulent']==0).sum()}")

    df = preprocess_dataframe(df)
    eng = extract_engineered_features(df)

    print(f"   Engineered features: {eng.shape[1]}")
    return df, eng


def run_full_pipeline():
    print("\n" + "╔" + "═"*54 + "╗")
    print("║   FAKE JOB POSTING DETECTION SYSTEM — Full Pipeline   ║")
    print("╚" + "═"*54 + "╝")

    ensure_dataset()
    df, eng = load_and_preprocess()

    # EDA
    from utils.eda import run_full_eda
    run_full_eda(df)

    # Training
    from models.trainer import run_training
    results, best_model, vectorizer, eng_cols = run_training(df, eng)

    # Prediction demo
    from models.predictor import run_demo_predictions
    run_demo_predictions()

    print("\n✅ Pipeline complete!")
    print(f"   📁 Visualizations → outputs/")
    print(f"   💾 Models         → models/")
    print("\n📌 Interpretation & Limitations:")
    print("   • Logistic Regression and Complement Naive Bayes offer")
    print("     strong baselines with high interpretability.")
    print("   • Recall for fraud is prioritized over precision to")
    print("     minimize missed fraudulent postings (false negatives).")
    print("   • Limitations: model is trained on synthetic data and")
    print("     may not generalize to all real-world fraud patterns.")
    print("   • Adversarial postings (e.g., well-written fraud) may")
    print("     evade detection — human review remains essential.")


def main():
    # Ensure working directory is the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    parser = argparse.ArgumentParser(description='Fake Job Posting Detection')
    parser.add_argument('--train', action='store_true', help='Train models only')
    parser.add_argument('--predict', action='store_true', help='Run prediction demo only')
    parser.add_argument('--eda', action='store_true', help='Run EDA only')
    args = parser.parse_args()

    if args.eda:
        ensure_dataset()
        df, eng = load_and_preprocess()
        from utils.eda import run_full_eda
        run_full_eda(df)

    elif args.train:
        ensure_dataset()
        df, eng = load_and_preprocess()
        from models.trainer import run_training
        run_training(df, eng)

    elif args.predict:
        from models.predictor import run_demo_predictions
        run_demo_predictions()

    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
