"""
app.py — Streamlit UI for Fake Job Posting Detection System
Run from the project root:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from models.predictor import JobFraudPredictor

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Fake Job Posting Detection System")
st.markdown("Enter the job posting details below and click **Predict** to check if it's legitimate or fraudulent.")

# ── Load predictor (cached so it loads only once) ────────────────────────────
@st.cache_resource
def load_predictor():
    return JobFraudPredictor()

try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("📋 Job Posting Details")

col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Job Title *", placeholder="e.g. Senior Software Engineer")
    location = st.text_input("Location", placeholder="e.g. Bangalore, India")
    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Contract", "Internship", "Other"]
    )
    required_experience = st.text_input("Required Experience", placeholder="e.g. Mid-Senior level")
    required_education = st.text_input("Required Education", placeholder="e.g. Bachelor's Degree")

with col2:
    industry = st.text_input("Industry", placeholder="e.g. Information Technology")
    function = st.text_input("Function", placeholder="e.g. Engineering")
    telecommuting = st.checkbox("Remote / Telecommuting")
    has_company_logo = st.checkbox("Has Company Logo", value=True)
    has_questions = st.checkbox("Has Screening Questions")

st.markdown("---")

company_profile = st.text_area(
    "Company Profile",
    placeholder="Describe the company — its mission, size, and background...",
    height=100
)
description = st.text_area(
    "Job Description *",
    placeholder="Describe the role, responsibilities, and what the candidate will do...",
    height=150
)
requirements = st.text_area(
    "Requirements",
    placeholder="List skills, qualifications, and experience required...",
    height=100
)
benefits = st.text_area(
    "Benefits",
    placeholder="e.g. Health insurance, flexible hours, annual bonus...",
    height=80
)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", type="primary"):
    if not title.strip() and not description.strip():
        st.warning("⚠️ Please enter at least a Job Title or Description.")
    else:
        with st.spinner("Analyzing job posting..."):
            result = predictor.predict(
                title=title,
                company_profile=company_profile,
                description=description,
                requirements=requirements,
                benefits=benefits,
                location=location,
                employment_type=employment_type,
                required_experience=required_experience,
                required_education=required_education,
                industry=industry,
                function=function,
                telecommuting=int(telecommuting),
                has_company_logo=int(has_company_logo),
                has_questions=int(has_questions),
            )

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        # Main verdict
        if result['prediction'] == 'FRAUDULENT':
            st.error(f"⚠️ **FRAUDULENT Job Posting Detected**")
        else:
            st.success(f"✅ **Legitimate Job Posting**")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Level", result['risk_level'])
        m2.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
        m3.metric("Confidence", f"{result['confidence']}%")

        # Signals
        st.markdown("#### 🔎 Signals Detected")
        for signal in result['signals']:
            st.markdown(f"- {signal}")
