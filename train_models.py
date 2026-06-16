"""
train_models.py
---------------
Trains Logistic Regression, Decision Tree, and Random Forest on the
student placement dataset, compares accuracy, saves the best model,
and exports visualisation charts.
"""

import os, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "dataset", "student_placement.csv")
MODEL_DIR = os.path.join(BASE, "src")
SS_DIR    = os.path.join(BASE, "screenshots")
os.makedirs(SS_DIR, exist_ok=True)

# ── 1. Load & split ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
X  = df.drop("Placed", axis=1)
y  = df["Placed"]
FEATURES = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 2. Train models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    use_scaled = name == "Logistic Regression"
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s  if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy")
    results[name] = {
        "model":    model,
        "accuracy": accuracy_score(y_test, y_pred),
        "cv_mean":  cv.mean(),
        "cv_std":   cv.std(),
        "y_pred":   y_pred,
        "y_prob":   y_prob,
    }
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Test Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  CV Accuracy   : {cv.mean():.4f} ± {cv.std():.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Placed", "Placed"]))

# ── 3. Save best model + scaler ──────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES,   os.path.join(MODEL_DIR, "features.pkl"))
print(f"\n✅  Best model: {best_name}  "
      f"(accuracy={results[best_name]['accuracy']:.4f})")

# ── 4. Palette ────────────────────────────────────────────────────────────────
CLRS = {
    "Logistic Regression": "#4361EE",
    "Decision Tree":       "#F72585",
    "Random Forest":       "#4CC9F0",
}
BG    = "#0D1117"
PANEL = "#161B22"
TEXT  = "#E6EDF3"
ACC   = "#58A6FF"

def dark_fig(*args, **kw):
    fig = plt.figure(*args, **kw)
    fig.patch.set_facecolor(BG)
    return fig

def dark_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    return ax

# ── 5. Chart A — Accuracy Comparison ─────────────────────────────────────────
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]
cvs   = [results[n]["cv_mean"]  * 100 for n in names]
stds  = [results[n]["cv_std"]   * 100 for n in names]
colors = [CLRS[n] for n in names]

fig, ax = dark_fig(figsize=(9, 5)), plt.gca()
dark_ax(ax)
x     = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, accs, width, color=colors, alpha=0.9,  label="Test Accuracy")
bars2 = ax.bar(x + width/2, cvs,  width, color=colors, alpha=0.55, label="CV Accuracy",
               yerr=stds, capsize=5, ecolor=TEXT)
ax.set_ylim(60, 105)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=14)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{bar.get_height():.1f}%", ha="center", va="bottom",
            fontsize=9, color=TEXT)
legend = ax.legend(facecolor=PANEL, edgecolor="#30363D", labelcolor=TEXT)
plt.tight_layout()
plt.savefig(os.path.join(SS_DIR, "accuracy_comparison.png"), dpi=150)
plt.close()
print("📊 Saved: accuracy_comparison.png")

# ── 6. Chart B — Confusion Matrices (3-panel) ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor(BG)
for ax, name in zip(axes, names):
    cm = confusion_matrix(y_test, results[name]["y_pred"])
    dark_ax(ax)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=False, linewidths=0.5, linecolor="#30363D",
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"])
    ax.set_title(name, color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", color=TEXT); ax.set_ylabel("Actual", color=TEXT)
    ax.tick_params(colors=TEXT)
plt.tight_layout()
plt.savefig(os.path.join(SS_DIR, "confusion_matrices.png"), dpi=150)
plt.close()
print("📊 Saved: confusion_matrices.png")

# ── 7. Chart C — ROC Curves ──────────────────────────────────────────────────
fig, ax = dark_fig(figsize=(7, 5)), plt.gca()
dark_ax(ax)
for name in names:
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_prob"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=CLRS[name], lw=2,
            label=f"{name} (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1],"--", color="#6E7681", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold", pad=14)
legend = ax.legend(facecolor=PANEL, edgecolor="#30363D", labelcolor=TEXT)
plt.tight_layout()
plt.savefig(os.path.join(SS_DIR, "roc_curves.png"), dpi=150)
plt.close()
print("📊 Saved: roc_curves.png")

# ── 8. Chart D — Feature Importance (Random Forest) ──────────────────────────
rf_model   = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)

fig, ax = dark_fig(figsize=(8, 6)), plt.gca()
dark_ax(ax)
bars = ax.barh(
    [FEATURES[i] for i in sorted_idx],
    importances[sorted_idx],
    color=ACC, alpha=0.85
)
ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title("Feature Importance — Random Forest", fontsize=14,
             fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig(os.path.join(SS_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("📊 Saved: feature_importance.png")

# ── 9. Chart E — EDA: Distribution plots ─────────────────────────────────────
num_feats = ["CGPA", "Attendance", "AptitudeScore",
             "CommunicationSkills", "TechnicalSkills", "SoftSkills"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor(BG)
axes = axes.flatten()
for ax, feat in zip(axes, num_feats):
    dark_ax(ax)
    for val, color, label in [(1, "#4CC9F0", "Placed"), (0, "#F72585", "Not Placed")]:
        subset = df[df["Placed"] == val][feat]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor="none")
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.set_xlabel(feat); ax.set_ylabel("Count")
    handles = [mpatches.Patch(color="#4CC9F0", label="Placed"),
               mpatches.Patch(color="#F72585", label="Not Placed")]
    ax.legend(handles=handles, facecolor=PANEL, edgecolor="#30363D", labelcolor=TEXT)
plt.suptitle("Feature Distributions by Placement Status",
             color=TEXT, fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SS_DIR, "eda_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved: eda_distributions.png")

# ── 10. Chart F — Correlation Heatmap ────────────────────────────────────────
fig, ax = dark_fig(figsize=(10, 8)), plt.gca()
dark_ax(ax)
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.4, linecolor="#30363D",
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14,
             fontweight="bold", pad=14, color=TEXT)
ax.tick_params(colors=TEXT, labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print("📊 Saved: correlation_heatmap.png")

print("\n✅  All charts generated. Training complete!")
