"""
Dataset Generator for Fake Job Posting Detection
Generates a realistic synthetic dataset mimicking the EMSI/Kaggle fake jobs dataset
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# --- Legitimate job templates ---
LEGIT_TITLES = [
    "Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager",
    "Business Analyst", "Frontend Developer", "Backend Developer", "DevOps Engineer",
    "Data Analyst", "UX Designer", "HR Manager", "Sales Executive",
    "Machine Learning Engineer", "Cybersecurity Analyst", "Cloud Architect",
    "Full Stack Developer", "Financial Analyst", "Operations Manager",
    "Content Writer", "QA Engineer", "Project Manager", "Network Engineer",
    "Database Administrator", "Mobile Developer", "AI Research Scientist"
]

LEGIT_COMPANIES = [
    "TechNova Solutions", "DataBridge Corp", "CloudSphere Inc", "Meridian Analytics",
    "Pinnacle Systems", "NexGen Software", "Vertex Technologies", "Sapient Global",
    "Orion Data Labs", "Catalyst Innovations", "Apex Digital", "Fusion Works",
    "Quantum Dynamics", "Helix Technologies", "Stratos Group"
]

LEGIT_LOCATIONS = [
    "Bangalore, India", "Mumbai, India", "Delhi, India", "Hyderabad, India",
    "Pune, India", "Chennai, India", "San Francisco, CA", "New York, NY",
    "London, UK", "Remote", "Berlin, Germany", "Singapore"
]

LEGIT_DESCRIPTIONS = [
    "We are looking for an experienced {title} to join our dynamic team. You will be responsible for developing scalable solutions, collaborating with cross-functional teams, and contributing to our technical roadmap. The ideal candidate has strong problem-solving skills and a passion for technology.",
    "Join our growing team as a {title}. In this role, you will work closely with stakeholders to deliver high-quality products. You will participate in the full software development lifecycle, from requirements gathering to deployment and maintenance.",
    "As a {title}, you will design and implement robust systems that serve millions of users. We value innovation, teamwork, and continuous learning. You will have the opportunity to work with cutting-edge technologies in a collaborative environment.",
    "We're hiring a {title} to help shape the future of our platform. You'll collaborate with engineers, designers, and product teams to build features that delight users. Strong communication skills and technical expertise are essential.",
    "An exciting opportunity for a talented {title} to join our engineering team. You will architect and develop solutions for complex business problems, mentor junior team members, and contribute to best practices and coding standards."
]

LEGIT_REQUIREMENTS = [
    "Bachelor's degree in Computer Science or related field. 3+ years of relevant experience. Proficiency in Python, Java, or related technologies. Strong analytical and problem-solving skills. Excellent communication abilities.",
    "Master's or Bachelor's degree in a technical field. 2-5 years of industry experience. Hands-on experience with cloud platforms (AWS/GCP/Azure). Strong understanding of data structures and algorithms. Team player with good interpersonal skills.",
    "Proven experience in a similar role. Strong portfolio demonstrating past work. Familiarity with agile methodologies. Experience with version control systems like Git. Ability to work in a fast-paced environment.",
    "Minimum 3 years of experience in the relevant domain. Proficiency with relevant tools and frameworks. Strong debugging and testing skills. Good understanding of software design patterns. Willingness to learn new technologies.",
]

LEGIT_BENEFITS = [
    "Competitive salary and performance bonuses. Health, dental, and vision insurance. Flexible working hours and remote options. Professional development budget. Annual paid leave and sick days.",
    "Market-leading compensation. Employee stock options. Comprehensive health coverage. Learning and development opportunities. Modern office with recreational facilities.",
    "Attractive CTC with annual increments. Medical insurance for family. Flexible work-from-home policy. Career growth opportunities. Team outings and company events.",
]

# --- Fraudulent job templates ---
FRAUD_TITLES = [
    "Work From Home - Earn $500/Day", "Easy Money Online - No Experience Needed",
    "Get Paid To Take Surveys", "Data Entry Clerk - Instant Hire",
    "Home-Based Customer Service - $800/Week", "Online Tutor - Earn Big",
    "Mystery Shopper - Immediate Start", "Envelope Stuffing Job",
    "Make Money Fast - Flexible Hours", "Commission Only Sales Agent",
    "Urgently Hiring - No Interview Required", "Financial Freedom Opportunity"
]

FRAUD_COMPANIES = [
    "GlobalPay Solutions", "EasyEarn Network", "QuickCash Ltd",
    "HomeWork Inc", "OnlineProfits Group", "FastMoney Corp",
    "WorkFromAnywhere", "InstantIncome Hub", ""  # some have no company
]

FRAUD_DESCRIPTIONS = [
    "AMAZING OPPORTUNITY!! 🌟 Earn $500-$1000 PER DAY working from home!! No experience required!! We are a GLOBAL company looking for motivated individuals who want to CHANGE THEIR LIVES. You will be working on simple tasks that anyone can do. Limited spots available - APPLY NOW before it's too late!!!",
    "Are you tired of your 9-5 job? Do you want FINANCIAL FREEDOM? This is the PERFECT opportunity for you! We offer guaranteed income with NO risk. Just complete easy online tasks and get PAID DAILY. We accept everyone regardless of experience or education. Join our team of THOUSANDS of successful members!",
    "Urgent hiring!! We need 50 people to start IMMEDIATELY. No interview, no background check. Just sign up and start earning TODAY. Our proven system has helped thousands achieve financial independence. Send us your details and we'll send you your first payment within 24 hours.",
    "Work from home processing orders and handling data entry. Earn $25-$40 per hour. No experience needed, we provide full training. Part time or full time available. Must have computer and internet connection. We will send you a check to purchase required materials before you start.",
    "BE YOUR OWN BOSS!! Join our exclusive network marketing team and earn passive income. We are expanding globally and need team leaders. There is NO LIMIT to what you can earn!! Invest just $50 to get started and watch your income MULTIPLY every week!!"
]

FRAUD_REQUIREMENTS = [
    "No experience necessary! Must be 18+. Have computer/phone with internet. Willing to work hard. That's it - we'll train you!",
    "Just need to be motivated! No degree required. Must be able to follow simple instructions. Basic computer skills helpful but not required.",
    "",  # many fake jobs have no requirements
    "Must send processing fee of $30 to receive your starter kit. No criminal background (self-reported). Must have bank account for direct deposit.",
    "Must purchase starter pack ($99 value) at discounted price of $49. Must recruit at least 2 friends to join the network."
]

FRAUD_BENEFITS = [
    "Unlimited earning potential!! Work when you want!! Be your own boss!! 100% risk-free guarantee!!",
    "Earn money fast! Daily payouts! No boss! No commute!",
    "",
    "Get rich working from home! Set your own hours! Join thousands of happy members!"
]


def generate_text(templates, **kwargs):
    t = random.choice(templates)
    try:
        return t.format(**kwargs)
    except:
        return t


def generate_dataset(n=2000):
    records = []
    n_legit = int(n * 0.83)
    n_fraud = n - n_legit

    # Legitimate jobs
    for i in range(n_legit):
        title = random.choice(LEGIT_TITLES)
        records.append({
            "job_id": i + 1,
            "title": title,
            "company_profile": f"{random.choice(LEGIT_COMPANIES)} is a leading technology company established in {random.randint(2000,2020)} with a team of {random.randint(50,5000)} professionals.",
            "description": generate_text(LEGIT_DESCRIPTIONS, title=title),
            "requirements": random.choice(LEGIT_REQUIREMENTS),
            "benefits": random.choice(LEGIT_BENEFITS),
            "location": random.choice(LEGIT_LOCATIONS),
            "employment_type": random.choice(["Full-time", "Part-time", "Contract", "Internship"]),
            "required_experience": random.choice(["Entry level", "Mid-Senior level", "Director", "Executive", "Not Applicable"]),
            "required_education": random.choice(["Bachelor's Degree", "Master's Degree", "High School", "Some College", "Doctorate"]),
            "industry": random.choice(["Information Technology", "Finance", "Healthcare", "Education", "Marketing", "Engineering"]),
            "function": random.choice(["Engineering", "Sales", "Marketing", "Finance", "Human Resources", "Operations"]),
            "telecommuting": random.randint(0, 1),
            "has_company_logo": 1,
            "has_questions": random.randint(0, 1),
            "fraudulent": 0
        })

    # Fraudulent jobs
    for i in range(n_fraud):
        records.append({
            "job_id": n_legit + i + 1,
            "title": random.choice(FRAUD_TITLES),
            "company_profile": random.choice(["", "", "We are a fast-growing global company with unlimited opportunities!", "Join our team of 10,000+ successful members worldwide!"]),
            "description": random.choice(FRAUD_DESCRIPTIONS),
            "requirements": random.choice(FRAUD_REQUIREMENTS),
            "benefits": random.choice(FRAUD_BENEFITS),
            "location": random.choice(["Remote", "Worldwide", "Work From Home", "", "Anywhere"]),
            "employment_type": random.choice(["Part-time", "Contract", "Other", ""]),
            "required_experience": random.choice(["Not Applicable", "Entry level", ""]),
            "required_education": random.choice(["High School", "Unspecified", ""]),
            "industry": random.choice(["", "Marketing", "Other", "Financial Services"]),
            "function": random.choice(["", "Sales", "Other", "Business Development"]),
            "telecommuting": 1,
            "has_company_logo": random.choice([0, 0, 1]),
            "has_questions": random.randint(0, 1),
            "fraudulent": 1
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(2000)
    df.to_csv("data/fake_job_postings.csv", index=False)
    print(f"Dataset generated: {len(df)} records")
    print(f"Fraudulent: {df['fraudulent'].sum()} | Legitimate: {(df['fraudulent']==0).sum()}")
