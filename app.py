"""
app.py — Student Placement Prediction System
============================================
Run with:  streamlit run app.py
"""

import os, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0D1117; color: #E6EDF3; }

.hero {
    background: linear-gradient(135deg, #1C2333 0%, #161B22 100%);
    border: 1px solid #30363D;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
}
.hero h1 { font-size: 2.2rem; font-weight: 700; color: #58A6FF; margin: 0; }
.hero p  { color: #8B949E; margin-top: .4rem; font-size: 1.05rem; }

.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #58A6FF; }
.metric-card .lbl { font-size: .82rem; color: #8B949E; margin-top: .2rem; }

.result-placed {
    background: linear-gradient(135deg, #0D2B0D, #0F2D1A);
    border: 2px solid #3FB950;
    border-radius: 14px;
    padding: 1.4rem 2rem;
    text-align: center;
}
.result-placed h2 { color: #3FB950; font-size: 1.8rem; margin: 0; }
.result-placed p  { color: #7EE787; margin-top: .3rem; }

.result-notplaced {
    background: linear-gradient(135deg, #2D1B1B, #2D0F0F);
    border: 2px solid #F85149;
    border-radius: 14px;
    padding: 1.4rem 2rem;
    text-align: center;
}
.result-notplaced h2 { color: #F85149; font-size: 1.8rem; margin: 0; }
.result-notplaced p  { color: #FF7B72; margin-top: .3rem; }

.tip-box {
    background: #161B22;
    border-left: 4px solid #58A6FF;
    border-radius: 0 10px 10px 0;
    padding: .8rem 1rem;
    margin: .5rem 0;
    color: #C9D1D9;
    font-size: .93rem;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #E6EDF3;
    border-bottom: 2px solid #58A6FF;
    padding-bottom: .4rem;
    margin: 1.4rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE, "src")
SS   = os.path.join(BASE, "screenshots")
DATA = os.path.join(BASE, "dataset", "student_placement.csv")

@st.cache_resource
def load_model():
    model    = joblib.load(os.path.join(SRC, "best_model.pkl"))
    scaler   = joblib.load(os.path.join(SRC, "scaler.pkl"))
    features = joblib.load(os.path.join(SRC, "features.pkl"))
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

model, scaler, features = load_model()
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=72)
    st.title("Student Placement\nPredictor")
    st.markdown("---")
    st.markdown("**About this app**")
    st.info(
        "Enter your academic and skill details to predict your campus placement "
        "probability using Machine Learning."
    )
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("🐍 Python · 🤖 Scikit-learn\n\n📊 Pandas · 📈 Matplotlib\n\n🌐 Streamlit")
    st.markdown("---")
    st.caption("AICTE ML Internship Project")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 Student Placement Prediction System</h1>
  <p>Powered by Machine Learning · Logistic Regression · Decision Tree · Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ───────────────────────────────────────────────────────────────
placed_pct = df["Placed"].mean() * 100
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [len(df), f"{placed_pct:.1f}%", "85.5%", "3"],
    ["Total Students", "Placement Rate", "Best Accuracy", "ML Models"],
):
    col.markdown(f"""
    <div class="metric-card">
      <div class="val">{val}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Placement", "📊 Model Analysis", "📂 Dataset"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Enter Your Profile</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        cgpa           = st.slider("📚 CGPA",                5.0, 10.0, 7.5, 0.1)
        attendance     = st.slider("🏫 Attendance (%)",       50,  100,  80,  1)
        communication  = st.slider("🗣️ Communication Skills", 1,   10,   7,   1)
        aptitude_score = st.slider("🧠 Aptitude Score",       40,  100,  70,  1)
        internship     = st.slider("💼 Internships",          0,   3,    1,   1)

    with col_b:
        technical      = st.slider("💻 Technical Skills",     1,   10,   7,   1)
        projects       = st.slider("🗂️ Projects",             0,   5,    2,   1)
        backlogs       = st.slider("⚠️ Backlogs",             0,   4,    0,   1)
        soft_skills    = st.slider("🤝 Soft Skills",          1,   10,   7,   1)
        mock_interviews= st.slider("🎤 Mock Interviews",      0,   10,   4,   1)

    st.markdown("")
    predict_btn = st.button("🔮 Predict My Placement", use_container_width=True)

    if predict_btn:
        input_df = pd.DataFrame([[
            cgpa, attendance, communication, aptitude_score,
            internship, technical, projects, backlogs, soft_skills, mock_interviews
        ]], columns=features)

        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]
        placed_prob  = probability[1] * 100

        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
            <div class="result-placed">
              <h2>✅ Likely to be Placed!</h2>
              <p>Placement Probability: <strong>{placed_prob:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-notplaced">
              <h2>❌ Needs Improvement</h2>
              <p>Placement Probability: <strong>{placed_prob:.1f}%</strong></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Probability bar
        fig, ax = plt.subplots(figsize=(7, 1.2))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        ax.barh([""], [placed_prob],      color="#3FB950", height=0.5, label="Placed")
        ax.barh([""], [100-placed_prob],  color="#F85149", height=0.5,
                left=[placed_prob], label="Not Placed")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", color="#E6EDF3")
        ax.tick_params(colors="#E6EDF3")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        ax.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3",
                  loc="upper right", fontsize=8)
        ax.axvline(50, color="#8B949E", linestyle="--", lw=1)
        ax.set_title(f"Placement Probability: {placed_prob:.1f}%",
                     color="#E6EDF3", fontsize=11)
        st.pyplot(fig, use_container_width=False)
        plt.close()

        # Tips
        st.markdown('<div class="section-title">💡 Improvement Tips</div>',
                    unsafe_allow_html=True)
        tips = []
        if cgpa < 7.5:          tips.append("📚 Aim for a CGPA above 7.5 — it's the primary filter for most recruiters.")
        if attendance < 75:     tips.append("🏫 Maintain at least 75% attendance to be eligible for campus drives.")
        if communication < 7:   tips.append("🗣️ Improve communication skills via GDs, debates, and presentations.")
        if internship == 0:     tips.append("💼 Complete at least one internship to gain industry exposure.")
        if technical < 7:       tips.append("💻 Sharpen technical skills — practice DSA and build domain projects.")
        if backlogs > 0:        tips.append("⚠️ Clear all backlogs; many companies have a zero-backlog policy.")
        if mock_interviews < 3: tips.append("🎤 Attend more mock interviews to reduce anxiety and improve performance.")
        if projects < 2:        tips.append("🗂️ Build 2–3 solid projects to showcase on your resume and GitHub.")

        if not tips:
            tips = ["🌟 Great profile! Keep practicing and apply to as many companies as possible."]

        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    chart_map = {
        "Accuracy Comparison":     "accuracy_comparison.png",
        "Confusion Matrices":      "confusion_matrices.png",
        "ROC Curves":              "roc_curves.png",
        "Feature Importance":      "feature_importance.png",
        "EDA: Distributions":      "eda_distributions.png",
        "Correlation Heatmap":     "correlation_heatmap.png",
    }

    sel = st.selectbox("Choose a chart:", list(chart_map.keys()))
    path = os.path.join(SS, chart_map[sel])
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning("Chart not found. Run `python src/train_models.py` first.")

    st.markdown('<div class="section-title">Model Comparison Summary</div>',
                unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Model":              ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Test Accuracy":      ["85.50%", "77.50%", "83.00%"],
        "CV Accuracy":        ["86.25%", "79.25%", "85.00%"],
        "CV Std Dev":         ["±1.98%", "±3.98%", "±3.16%"],
        "Best For":           ["Interpretability", "Visual explanation", "Robustness"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — DATASET
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Dataset Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Features", len(df.columns) - 1)
    c3.metric("Placed", f"{df['Placed'].sum()} ({df['Placed'].mean()*100:.1f}%)")

    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="section-title">Descriptive Statistics</div>',
                unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    # Download
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Full Dataset (CSV)", csv,
                       "student_placement.csv", "text/csv",
                       use_container_width=True)
