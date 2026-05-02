"""
utils/text_preprocessor.py
Text cleaning and preprocessing utilities for job posting data
"""

import re
import string
import pandas as pd
import numpy as np


# Common stopwords (lightweight, no NLTK dependency)
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'that', 'this', 'these', 'those', 'it', 'its', 'they', 'their',
    'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her', 'i', 'my',
    'me', 'us', 'them', 'who', 'which', 'what', 'when', 'where', 'how',
    'if', 'so', 'as', 'not', 'no', 'nor', 'yet', 'both', 'either',
    'also', 'just', 'than', 'then', 'too', 'very', 'more', 'most',
    'other', 'some', 'any', 'all', 'each', 'every', 'into', 'through',
    'about', 'above', 'after', 'before', 'between', 'during', 'without',
    'within', 'along', 'following', 'across', 'behind', 'beyond', 'plus',
    'except', 'up', 'out', 'around', 'down', 'off', 'again', 'further',
    'once', 'here', 'there', 'while', 'although', 'because', 'since',
    'unless', 'until', 'whether', 'among', 'per', 'via', 'vs', 'etc'
}

# Fraud signal keywords (for feature engineering)
FRAUD_SIGNALS = [
    'earn', 'money', 'fast', 'quick', 'easy', 'guaranteed', 'unlimited',
    'passive', 'income', 'home', 'immediate', 'urgent', 'now', 'today',
    'free', 'risk', 'invest', 'join', 'network', 'marketing', 'commission',
    'referral', 'recruit', 'bonus', 'daily', 'payout', 'amazing', 'opportunity',
    'exclusive', 'limited', 'boss', 'freedom', 'financial', 'rich', 'dream'
]


def clean_text(text: str) -> str:
    """Clean and normalize text for ML processing."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters and digits, keep spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    
    return ' '.join(tokens)


def combine_text_features(row: pd.Series) -> str:
    """Combine all text columns into a single feature."""
    fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    parts = []
    for field in fields:
        val = row.get(field, '')
        if isinstance(val, str) and val.strip():
            parts.append(val)
    return ' '.join(parts)


def extract_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hand-crafted features that signal fraud."""
    feats = pd.DataFrame(index=df.index)
    
    # Text length features
    for col in ['description', 'requirements', 'company_profile', 'benefits']:
        text_col = df[col].fillna('')
        feats[f'{col}_len'] = text_col.apply(len)
        feats[f'{col}_word_count'] = text_col.apply(lambda x: len(x.split()))
        feats[f'{col}_is_empty'] = (text_col.str.strip() == '').astype(int)
    
    # Exclamation mark count (fraud signal)
    for col in ['description', 'title', 'benefits']:
        feats[f'{col}_exclamation'] = df[col].fillna('').apply(lambda x: x.count('!'))
    
    # Capitalization ratio (fraud signal - ALL CAPS)
    feats['desc_caps_ratio'] = df['description'].fillna('').apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    
    # Fraud keyword count
    def count_fraud_keywords(text):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        return sum(1 for kw in FRAUD_SIGNALS if kw in text_lower)
    
    combined = df.apply(combine_text_features, axis=1)
    feats['fraud_keyword_count'] = combined.apply(count_fraud_keywords)
    
    # Structural features
    feats['has_company_logo'] = df.get('has_company_logo', pd.Series(0, index=df.index)).fillna(0).astype(int)
    feats['has_questions'] = df.get('has_questions', pd.Series(0, index=df.index)).fillna(0).astype(int)
    feats['telecommuting'] = df.get('telecommuting', pd.Series(0, index=df.index)).fillna(0).astype(int)
    
    # Missing value indicators
    feats['missing_requirements'] = df['requirements'].fillna('').apply(lambda x: int(len(x.strip()) == 0))
    feats['missing_company_profile'] = df['company_profile'].fillna('').apply(lambda x: int(len(x.strip()) == 0))
    feats['missing_benefits'] = df['benefits'].fillna('').apply(lambda x: int(len(x.strip()) == 0))
    
    # Employment type encoding
    emp_map = {'Full-time': 0, 'Part-time': 1, 'Contract': 2, 'Internship': 3, 'Other': 4, '': 5}
    feats['employment_type_enc'] = df.get('employment_type', '').fillna('').map(emp_map).fillna(5).astype(int)
    
    return feats


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline on the raw dataframe."""
    df = df.copy()
    
    # Fill NaN text fields
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'location']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Create combined cleaned text
    df['combined_text'] = df.apply(combine_text_features, axis=1)
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    return df
