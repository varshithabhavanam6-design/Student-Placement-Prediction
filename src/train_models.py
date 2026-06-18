"""
train_models.py — EDA, training & evaluation for Student Placement Prediction
================================================================================
- Loads dataset/student_placement.csv
- Generates EDA charts (distributions, correlation heatmap)
- Trains Logistic Regression, Decision Tree, Random Forest
- Evaluates with train/test split + 5-fold cross-validation
- Saves comparison charts (accuracy, confusion matrices, ROC, feature importance)
- Picks the best model by CV accuracy and saves model + scaler + feature list

Run with:  python src/train_models.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "dataset", "student_placement.csv")
SS   = os.path.join(BASE, "screenshots")
SRC  = os.path.join(BASE, "src")
os.makedirs(SS, exist_ok=True)

# Dark theme to match the Streamlit app's look
BG, PANEL, GRID, TEXT, MUTED = "#0D1117", "#161B22", "#30363D", "#E6EDF3", "#8B949E"
PALETTE = ["#58A6FF", "#3FB950", "#F85149", "#D29922", "#BC8CFF", "#39C5CF"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "figure.dpi": 130,
})

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Loaded dataset: {df.shape}")

features = [c for c in df.columns if c != "Placed"]
X = df[features]
y = df["Placed"]

# ═══════════════════════════════════════════════════════════════
# EDA CHARTS
# ═══════════════════════════════════════════════════════════════
print("Generating EDA charts...")

# 1) Feature distributions by placement status
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    ax = axes[i]
    sns.kdeplot(data=df, x=col, hue="Placed", fill=True, common_norm=False,
                palette=[PALETTE[2], PALETTE[1]], alpha=0.4, ax=ax, legend=(i == 0))
    ax.set_title(col, fontsize=10, color=TEXT)
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
fig.suptitle("Feature Distributions by Placement Status", fontsize=15, color=TEXT, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(SS, "eda_distributions.png"), bbox_inches="tight", facecolor=BG)
plt.close(fig)

# 2) Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            linewidths=0.5, linecolor=GRID, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("Feature Correlation Heatmap", fontsize=14, color=TEXT)
fig.tight_layout()
fig.savefig(os.path.join(SS, "correlation_heatmap.png"), bbox_inches="tight", facecolor=BG)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT + SCALING
# ═══════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
roc_data = {}

print("Training models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    cm = confusion_matrix(y_test, y_pred)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

    results[name] = {
        "model": model,
        "test_acc": test_acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cm": cm,
    }
    print(f"  {name:22s} | Test Acc: {test_acc*100:5.2f}% | CV Acc: {cv_mean*100:5.2f}% ± {cv_std*100:4.2f}%")

best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best model: {best_name}")

# ═══════════════════════════════════════════════════════════════
# COMPARISON CHARTS
# ═══════════════════════════════════════════════════════════════
print("Generating model comparison charts...")

# 3) Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
names = list(results.keys())
test_accs = [results[n]["test_acc"] * 100 for n in names]
cv_accs   = [results[n]["cv_mean"] * 100 for n in names]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, test_accs, w, label="Test Accuracy", color=PALETTE[0])
ax.bar(x + w/2, cv_accs,   w, label="CV Accuracy",   color=PALETTE[1])
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Model Accuracy Comparison", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
for i, v in enumerate(test_accs):
    ax.text(i - w/2, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=TEXT)
for i, v in enumerate(cv_accs):
    ax.text(i + w/2, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=TEXT)
fig.tight_layout()
fig.savefig(os.path.join(SS, "accuracy_comparison.png"), bbox_inches="tight", facecolor=BG)
plt.close(fig)

# 4) Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, name in zip(axes, names):
    cm = results[name]["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Not Placed", "Placed"], yticklabels=["Not Placed", "Placed"])
    ax.set_title(name, color=TEXT, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
fig.suptitle("Confusion Matrices", fontsize=14, color=TEXT)
fig.tight_layout()
fig.savefig(os.path.join(SS, "confusion_matrices.png"), bbox_inches="tight", facecolor=BG)
plt.close(fig)

# 5) ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for i, name in enumerate(names):
    fpr, tpr, roc_auc = roc_data[name]
    ax.plot(fpr, tpr, color=PALETTE[i], lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves", color=TEXT)
ax.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(SS, "roc_curves.png"), bbox_inches="tight", facecolor=BG)
plt.close(fig)

# 6) Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importances.index, importances.values, color=PALETTE[0])
ax.set_xlabel("Importance")
ax.set_title("Feature Importance (Random Forest)", color=TEXT)
fig.tight_layout()
fig.savefig(os.path.join(SS, "feature_importance.png"), bbox_inches="tight", facecolor=BG)
plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# SAVE BEST MODEL + SCALER + FEATURES
# ═══════════════════════════════════════════════════════════════
joblib.dump(best_model, os.path.join(SRC, "best_model.pkl"))
joblib.dump(scaler,     os.path.join(SRC, "scaler.pkl"))
joblib.dump(features,   os.path.join(SRC, "features.pkl"))

# Also save a small JSON summary so README / app numbers can be kept in sync
summary = {
    name: {
        "test_accuracy": round(results[name]["test_acc"] * 100, 2),
        "cv_accuracy": round(results[name]["cv_mean"] * 100, 2),
        "cv_std": round(results[name]["cv_std"] * 100, 2),
    } for name in names
}
summary["best_model"] = best_name
with open(os.path.join(SRC, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Saved best_model.pkl, scaler.pkl, features.pkl to {SRC}")
print(f"✅ Saved 6 chart images to {SS}")
print(f"✅ Saved results_summary.json")
