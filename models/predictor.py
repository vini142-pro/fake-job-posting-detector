"""
models/predictor.py
Prediction system for classifying new job postings as Fraudulent or Legitimate
"""

import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

from utils.text_preprocessor import (
    clean_text, combine_text_features, extract_engineered_features
)

MODEL_DIR = "models"


class JobFraudPredictor:
    """
    End-to-end prediction system for fake job detection.
    Loads trained model + vectorizer and classifies new job postings.
    """

    def __init__(self):
        model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
        vec_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')

        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            raise FileNotFoundError(
                "Trained model not found. Please run main.py first to train the model."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)
        print("✅ Predictor loaded successfully.")

    def _build_row(self, title='', company_profile='', description='',
                   requirements='', benefits='', location='',
                   employment_type='Full-time', required_experience='',
                   required_education='', industry='', function='',
                   telecommuting=0, has_company_logo=1, has_questions=0):
        """Build a single-row DataFrame from inputs."""
        return pd.DataFrame([{
            'title': title,
            'company_profile': company_profile,
            'description': description,
            'requirements': requirements,
            'benefits': benefits,
            'location': location,
            'employment_type': employment_type,
            'required_experience': required_experience,
            'required_education': required_education,
            'industry': industry,
            'function': function,
            'telecommuting': telecommuting,
            'has_company_logo': has_company_logo,
            'has_questions': has_questions,
        }])

    def predict(self, **kwargs) -> dict:
        """
        Predict whether a job posting is fraudulent.

        Parameters: title, company_profile, description, requirements,
                    benefits, location, employment_type, ...

        Returns: dict with 'prediction', 'confidence', 'risk_level', 'explanation'
        """
        df = self._build_row(**kwargs)

        # Preprocess
        df['combined_text'] = df.apply(combine_text_features, axis=1)
        df['cleaned_text'] = df['combined_text'].apply(clean_text)

        # Features
        X_text = self.vectorizer.transform(df['cleaned_text'].fillna(''))
        eng = extract_engineered_features(df)
        X_eng = csr_matrix(eng.fillna(0).values)
        X = hstack([X_text, X_eng])

        # Predict
        prediction = self.model.predict(X)[0]

        # Confidence
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            fraud_prob = proba[1]
        elif hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(X)[0]
            fraud_prob = 1 / (1 + np.exp(-score))  # sigmoid
        else:
            fraud_prob = float(prediction)

        # Risk level
        if fraud_prob >= 0.80:
            risk = "HIGH RISK 🔴"
        elif fraud_prob >= 0.50:
            risk = "MEDIUM RISK 🟡"
        elif fraud_prob >= 0.30:
            risk = "LOW RISK 🟢"
        else:
            risk = "LEGITIMATE ✅"

        # Explanation signals
        signals = []
        desc = kwargs.get('description', '')
        title = kwargs.get('title', '')
        company = kwargs.get('company_profile', '')
        req = kwargs.get('requirements', '')

        if desc.count('!') >= 3:
            signals.append("⚠️  Excessive exclamation marks in description")
        if any(kw in (desc + title).lower() for kw in ['earn', 'fast', 'guaranteed', 'unlimited']):
            signals.append("⚠️  Fraud keywords detected (earn/fast/guaranteed/unlimited)")
        if len(company.strip()) < 20:
            signals.append("⚠️  Missing or very short company profile")
        if len(req.strip()) < 30:
            signals.append("⚠️  Missing or very short requirements section")
        if kwargs.get('has_company_logo', 1) == 0:
            signals.append("⚠️  No company logo")
        if sum(1 for c in desc if c.isupper()) / max(len(desc), 1) > 0.15:
            signals.append("⚠️  Unusual CAPS usage in description")

        if not signals:
            signals.append("✅ No major fraud signals detected")

        return {
            'prediction': 'FRAUDULENT' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': round(float(fraud_prob), 4),
            'confidence': round(float(max(fraud_prob, 1 - fraud_prob)) * 100, 1),
            'risk_level': risk,
            'signals': signals
        }

    def predict_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Batch prediction from a CSV file."""
        df = pd.read_csv(csv_path)
        results = []

        for _, row in df.iterrows():
            result = self.predict(**row.to_dict())
            results.append({
                'title': row.get('title', ''),
                'prediction': result['prediction'],
                'fraud_probability': result['fraud_probability'],
                'risk_level': result['risk_level']
            })

        return pd.DataFrame(results)


# ── Demo predictions ──────────────────────────────────────────────────────────
DEMO_JOBS = [
    {
        "name": "Legitimate: Software Engineer",
        "title": "Senior Software Engineer",
        "company_profile": "TechNova Solutions is a leading software company with 500+ employees founded in 2010.",
        "description": "We are looking for an experienced software engineer to join our product team. You will design and implement scalable backend services, work with modern cloud infrastructure, and mentor junior developers.",
        "requirements": "Bachelor's in Computer Science. 4+ years Python/Java experience. Familiarity with AWS and microservices. Strong communication skills.",
        "benefits": "Competitive salary, health insurance, flexible work from home, annual bonus.",
        "location": "Bangalore, India",
        "employment_type": "Full-time",
        "has_company_logo": 1,
        "has_questions": 1,
        "telecommuting": 0
    },
    {
        "name": "Fraudulent: Work From Home Scam",
        "title": "EARN $500/DAY Working From Home!! URGENT HIRING!!",
        "company_profile": "",
        "description": "AMAZING OPPORTUNITY!! 🌟 Earn $500-$1000 PER DAY working from home!! No experience required!! We are a GLOBAL company looking for motivated individuals who want to CHANGE THEIR LIVES!! Just complete easy online tasks and get PAID DAILY!! Limited spots - APPLY NOW!!!",
        "requirements": "Must be 18+. No degree needed. Just motivation!",
        "benefits": "Unlimited earning potential!! Be your own boss!! Financial freedom!!",
        "location": "Remote/Worldwide",
        "employment_type": "Other",
        "has_company_logo": 0,
        "has_questions": 0,
        "telecommuting": 1
    },
    {
        "name": "Borderline: Commission Sales",
        "title": "Sales Representative - Commission Based",
        "company_profile": "FastGrowth Sales is expanding our team.",
        "description": "Looking for motivated sales professionals. You will contact potential clients, present our products, and close deals. Work from home available.",
        "requirements": "Good communication skills. Previous sales experience preferred.",
        "benefits": "Commission only. Unlimited potential. Flexible hours.",
        "location": "Remote",
        "employment_type": "Contract",
        "has_company_logo": 1,
        "has_questions": 0,
        "telecommuting": 1
    }
]


def run_demo_predictions():
    """Run demo predictions on sample jobs."""
    print("\n" + "═" * 60)
    print("🔍  FAKE JOB DETECTION – PREDICTION SYSTEM DEMO")
    print("═" * 60)

    try:
        predictor = JobFraudPredictor()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    for job in DEMO_JOBS:
        name = job.pop('name')
        print(f"\n{'─' * 60}")
        print(f"📋 {name}")
        print(f"   Title: {job['title'][:60]}")

        result = predictor.predict(**job)

        print(f"   ┌─ Prediction  : {result['prediction']}")
        print(f"   ├─ Fraud Prob  : {result['fraud_probability']:.1%}")
        print(f"   ├─ Confidence  : {result['confidence']}%")
        print(f"   ├─ Risk Level  : {result['risk_level']}")
        print(f"   └─ Signals     :")
        for sig in result['signals']:
            print(f"        {sig}")

    print(f"\n{'═' * 60}\n")
