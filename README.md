# Heart Disease Prediction using Machine Learning 

An interactive web application that predicts the presence of heart disease using patient health data (UCI Heart Disease dataset). The project trains and compares four classification algorithms, validates results using cross-validation, dynamically selects the best-performing model, and provides a Streamlit-based UI for exploring the data, comparing models, and predicting risk for a new patient in real time.

---

## Overview

Heart disease is one of the leading causes of death worldwide, and early detection through data-driven risk assessment can support faster clinical decision-making. This project builds a complete, reproducible classification pipeline — from raw data to an interactive web app — while applying sound machine learning practices such as leakage prevention, threshold-aware evaluation, and reliability checks via cross-validation.

---

## Features

- **Interactive Web App (Streamlit)** — navigate between Data Overview, Model Comparison, Feature Importance, and Prediction pages from the sidebar
- **Exploratory Data Analysis (EDA)** — missing values, duplicates, class distribution visualization
- **Data Preprocessing** — duplicate removal and feature scaling using `StandardScaler`
- **Four Classification Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost
- **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices for every model
- **5-Fold Cross-Validation** — checks reliability of each model's recall score beyond a single train/test split
- **Dynamic Best Model Selection** — automatically selects the best model using a combined Accuracy + Cross-Validated Recall score, rather than hardcoding a "winner"
- **Feature Importance Visualization** — for tree-based models (Random Forest / XGBoost)
- **Real-Time Prediction Form** — enter a new patient's details through the UI and get an instant prediction with confidence score
- **Cached Model Training** — models are trained once per dataset (`st.cache_resource`), so the app stays fast when navigating between pages

---

## Dataset

The project uses the **UCI Heart Disease dataset** (`heart.csv`), which contains 13 clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, and more, along with a binary `target` column (1 = heart disease present, 0 = no heart disease).

---

## Repository Contents

This repo contains two versions of the project:

- **`app.py`** — the main deliverable: an interactive Streamlit web app (see below for how to run it)
- **`disease_prediction.py`** — the original standalone script version with console-based (CLI) output and matplotlib pop-up charts, kept to show the project's progression from a script into a full interactive app

For the best experience, run `app.py`.

---

## Tech Stack

- **Language**: Python
- **Web Framework**: Streamlit
- **Libraries**: pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn

---

## Project Workflow

1. Upload the dataset through the app's sidebar
2. Explore the data (shape, nulls, duplicates, class balance) on the **Data Overview** page
3. Models are trained automatically in the background (Logistic Regression, SVM, Random Forest, XGBoost)
4. View and compare model performance on the **Model Comparison** page (metrics table, accuracy/recall charts, confusion matrix, classification report, cross-validation results)
5. View the **Feature Importance** page for the best-selected model (Random Forest / XGBoost)
6. Go to the **Predict** page, fill in a patient's details, and get an instant heart disease risk prediction with a confidence score

---

## Key Learnings & Design Decisions

- **Leakage Prevention**: The scaler is fit only on training data and reused (never refit) on test/new data, and cross-validation is done using a pipeline so scaling happens fresh within each fold.
- **Why Recall Matters Here**: In a medical screening context, missing an actual heart disease case (false negative) is more costly than a false alarm — so recall is prioritized alongside accuracy when selecting the best model.
- **Why ROC-AUC**: Reported for tree-based models since it evaluates prediction probabilities rather than just hard class labels, giving a fuller picture of model performance.
- **Dynamic Model Selection**: Instead of assuming one algorithm is always best, the "best model" is chosen programmatically each run based on combined metrics.
- **Caching for Performance**: Data loading and model training are cached using Streamlit's `@st.cache_data` and `@st.cache_resource`, so the app doesn't retrain models on every user interaction.

---

## How to Run Locally

1. Clone the repository
   ```bash
   git clone https://github.com/Jinsipatel/heart-disease-prediction-ml.git
   cd heart-disease-prediction-ml
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```
   (Do **not** run it with `python app.py` — Streamlit apps must be launched with the `streamlit run` command so the browser-based UI and interactivity work correctly.)

4. The app will open automatically in your browser at `http://localhost:8501`. Upload the `heart.csv` dataset from the sidebar to get started.

---

## App Pages

| Page | Description |
|------|-------------|
| **Data Overview** | Dataset preview, shape, missing values, statistical summary, and target class distribution |
| **Model Comparison** | Metrics table for all 4 models, accuracy/recall bar charts, confusion matrix, classification report, and cross-validation results |
| **Feature Importance** | Feature importance chart for the best-selected model (Random Forest / XGBoost) |
| **Predict** | Interactive form to enter patient details and get a real-time heart disease prediction with confidence score |

---

## Author & Contact

Jinsi patel

E_mail:jinsipatel1307@gmail.com
