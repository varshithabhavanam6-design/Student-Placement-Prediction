"""
generate_dataset.py — Synthetic Student Placement Dataset Generator
=====================================================================
Generates a realistic, synthetic dataset of student academic & skill
metrics, and a binary "Placed" target that depends on those metrics
(with some noise so the problem isn't trivially separable).

Run with:  python src/generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000  # number of student records

# ── Feature generation ──────────────────────────────────────────────────────
CGPA            = np.clip(np.random.normal(7.2, 1.0, N), 5.0, 10.0).round(2)
Attendance      = np.clip(np.random.normal(80, 10, N), 50, 100).round(0)
Communication   = np.clip(np.random.normal(6.8, 1.8, N), 1, 10).round(0)
Aptitude_Score  = np.clip(np.random.normal(68, 14, N), 40, 100).round(0)
Internships     = np.clip(np.random.poisson(1.0, N), 0, 3)
Technical_Skills= np.clip(np.random.normal(6.8, 1.8, N), 1, 10).round(0)
Projects        = np.clip(np.random.poisson(1.8, N), 0, 5)
Backlogs        = np.clip(np.random.poisson(0.5, N), 0, 4)
Soft_Skills     = np.clip(np.random.normal(6.8, 1.6, N), 1, 10).round(0)
Mock_Interviews = np.clip(np.random.poisson(4, N), 0, 10)

df = pd.DataFrame({
    "CGPA": CGPA,
    "Attendance": Attendance,
    "Communication": Communication,
    "Aptitude_Score": Aptitude_Score,
    "Internships": Internships,
    "Technical_Skills": Technical_Skills,
    "Projects": Projects,
    "Backlogs": Backlogs,
    "Soft_Skills": Soft_Skills,
    "Mock_Interviews": Mock_Interviews,
})

# ── Target generation ────────────────────────────────────────────────────────
# Build a weighted "placement score" from normalized features, then turn it
# into a probability with a sigmoid, then sample the binary outcome from it.
# This keeps the relationship realistic but not perfectly deterministic.

def normalize(s, lo, hi):
    return (s - lo) / (hi - lo)

score = (
    0.30 * normalize(df["CGPA"], 5.0, 10.0) +
    0.10 * normalize(df["Attendance"], 50, 100) +
    0.12 * normalize(df["Communication"], 1, 10) +
    0.10 * normalize(df["Aptitude_Score"], 40, 100) +
    0.10 * normalize(df["Internships"], 0, 3) +
    0.15 * normalize(df["Technical_Skills"], 1, 10) +
    0.08 * normalize(df["Projects"], 0, 5) +
    0.07 * normalize(df["Soft_Skills"], 1, 10) +
    0.06 * normalize(df["Mock_Interviews"], 0, 10) -
    0.18 * normalize(df["Backlogs"], 0, 4)
)

# Center & scale, then squash with a sigmoid for a smooth probability.
# A small positive bias shifts the overall placement rate up toward the
# ~62% historically seen in campus placement datasets.
score_scaled = (score - score.mean()) / score.std()
prob_placed = 1 / (1 + np.exp(-2.2 * score_scaled - 0.85))

# Add a touch of irreducible noise so the model can't reach 100% accuracy
noise = np.random.normal(0, 0.06, N)
prob_placed = np.clip(prob_placed + noise, 0.02, 0.98)

df["Placed"] = (np.random.rand(N) < prob_placed).astype(int)

# Nudge class balance toward the README's documented ~62/38 split, if needed
target_rate = 0.62
current_rate = df["Placed"].mean()
print(f"Generated placement rate: {current_rate*100:.1f}% (target ~{target_rate*100:.0f}%)")

# ── Save ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "dataset", "student_placement.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)

print(f"✅ Dataset saved to: {OUT}")
print(f"   Shape: {df.shape}")
print(df.head())
