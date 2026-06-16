# 🎓 Student Placement Prediction System

> **AICTE Machine Learning Internship Project**  
> Predicts whether a student will be placed based on academic performance and skill metrics.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [ML Models & Results](#-ml-models--results)
- [Screenshots](#-screenshots)
- [Dataset](#-dataset)
- [Author](#-author)

---

## 🌟 Project Overview

This end-to-end Machine Learning project predicts student campus placement outcomes using 10 key features including CGPA, communication skills, internship experience, and more.

It covers the complete data science workflow:
1. Dataset generation / collection  
2. Data cleaning & preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Model training & comparison  
5. Streamlit web application deployment  

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 Real-time Prediction | Enter your details and get an instant placement prediction |
| 📊 Model Comparison | Compare Logistic Regression, Decision Tree & Random Forest |
| 📈 Visual Analytics | ROC curves, confusion matrices, feature importance charts |
| 💡 Smart Tips | Personalised improvement recommendations |
| ⬇️ Dataset Download | Export the full dataset as CSV |

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Library | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | Joblib |
| Version Control | Git & GitHub |

---

## 📁 Project Structure

```
Student-Placement-Prediction/
│
├── dataset/
│   └── student_placement.csv      # 1000-student synthetic dataset
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb     # Jupyter notebook (optional exploration)
│
├── src/
│   ├── generate_dataset.py        # Creates the synthetic dataset
│   ├── train_models.py            # Trains all 3 models + exports charts
│   ├── best_model.pkl             # Saved best model (Logistic Regression)
│   ├── scaler.pkl                 # Saved StandardScaler
│   └── features.pkl               # Feature list
│
├── screenshots/
│   ├── accuracy_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── eda_distributions.png
│   └── correlation_heatmap.png
│
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Student-Placement-Prediction.git
cd Student-Placement-Prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Generate dataset
```bash
python src/generate_dataset.py
```

### Step 2 — Train models & generate charts
```bash
python src/train_models.py
```

### Step 3 — Launch the Streamlit app
```bash
streamlit run app.py
```
Then open your browser at **http://localhost:8501**

---

## 🤖 ML Models & Results

| Model | Test Accuracy | CV Accuracy | CV Std Dev |
|---|---|---|---|
| **Logistic Regression** ✅ | **85.50%** | **86.25%** | ±1.98% |
| Random Forest | 83.00% | 85.00% | ±3.16% |
| Decision Tree | 77.50% | 79.25% | ±3.98% |

**Winner: Logistic Regression** — highest accuracy with lowest variance, making it the most reliable model for this dataset.

### Input Features

| Feature | Range | Description |
|---|---|---|
| CGPA | 5.0 – 10.0 | Cumulative Grade Point Average |
| Attendance | 50 – 100% | Class attendance percentage |
| Communication Skills | 1 – 10 | Self-rated communication ability |
| Aptitude Score | 40 – 100 | Score on aptitude test |
| Internship Count | 0 – 3 | Number of internships completed |
| Technical Skills | 1 – 10 | Programming / domain expertise |
| Project Count | 0 – 5 | Number of academic/personal projects |
| Backlogs | 0 – 4 | Number of pending subjects |
| Soft Skills | 1 – 10 | Teamwork, leadership, etc. |
| Mock Interviews | 0 – 10 | Mock interviews attended |

---

## 📸 Screenshots

| Chart | Description |
|---|---|
| `accuracy_comparison.png` | Bar chart comparing Test vs CV accuracy |
| `confusion_matrices.png` | Side-by-side confusion matrices |
| `roc_curves.png` | ROC / AUC curves for all models |
| `feature_importance.png` | Random Forest feature ranking |
| `eda_distributions.png` | Feature distributions by placement status |
| `correlation_heatmap.png` | Feature correlation matrix |

---

## 📂 Dataset

- **Source:** Synthetically generated for demonstration (mirrors real Kaggle datasets)
- **Size:** 1,000 student records
- **Target:** `Placed` (1 = placed, 0 = not placed)
- **Class balance:** ~62% placed, ~38% not placed

---

## 👤 Author

**Your Name**  
AICTE Machine Learning Internship  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_USERNAME)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
