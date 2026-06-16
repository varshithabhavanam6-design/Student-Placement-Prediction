"""
Generate a realistic synthetic dataset for Student Placement Prediction.
This mimics a Kaggle-style dataset with 1000 student records.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

cgpa            = np.round(np.random.uniform(5.0, 10.0, n), 2)
attendance      = np.round(np.random.uniform(50, 100, n), 1)
communication   = np.random.randint(1, 11, n)          # 1–10 scale
aptitude_score  = np.random.randint(40, 100, n)        # out of 100
internship      = np.random.randint(0, 4, n)           # 0‑3 internships
technical_skills= np.random.randint(1, 11, n)          # 1–10 scale
projects        = np.random.randint(0, 6, n)           # 0‑5 projects
backlogs        = np.random.randint(0, 5, n)           # number of backlogs
soft_skills     = np.random.randint(1, 11, n)          # 1–10 scale
mock_interviews = np.random.randint(0, 11, n)          # 0‑10 mocks attended

# Placement logic — weighted score to simulate realistic outcomes
score = (
    cgpa            * 0.30 +
    attendance      * 0.05 +
    communication   * 0.15 +
    (aptitude_score / 10) * 0.15 +
    internship      * 0.10 +
    technical_skills* 0.15 +
    projects        * 0.05 +
    soft_skills     * 0.05 -
    backlogs        * 0.08
)

# Add noise and threshold
score_norm = (score - score.min()) / (score.max() - score.min())
noise = np.random.normal(0, 0.07, n)
prob = np.clip(score_norm + noise, 0, 1)
placed = (prob > 0.45).astype(int)

df = pd.DataFrame({
    "CGPA":              cgpa,
    "Attendance":        attendance,
    "CommunicationSkills": communication,
    "AptitudeScore":     aptitude_score,
    "InternshipCount":   internship,
    "TechnicalSkills":   technical_skills,
    "ProjectCount":      projects,
    "Backlogs":          backlogs,
    "SoftSkills":        soft_skills,
    "MockInterviews":    mock_interviews,
    "Placed":            placed,
})

out = "/home/claude/Student-Placement-Prediction/dataset/student_placement.csv"
df.to_csv(out, index=False)
print(f"Dataset saved → {out}")
print(df["Placed"].value_counts().to_string())
print(df.head())
