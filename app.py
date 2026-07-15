# ============================================================
# Heart Disease Prediction - Streamlit Web App
# ============================================================
# Interactive version of the original script. Loads data, trains
# multiple models (cached so training runs only once), lets the
# user explore EDA + model comparison, and predicts heart disease
# risk for a new patient entered through the UI.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.drop_duplicates()
    return df


# ------------------------------------------------------------
# Model Training (cached so this heavy step runs only once per dataset)
# ------------------------------------------------------------
@st.cache_resource
def train_models(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_objects = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            random_state=42, eval_metric="logloss",
            n_estimators=150, max_depth=3, learning_rate=0.1
        ),
    }

    results = {}
    predictions = {}

    for name, model in model_objects.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        predictions[name] = pred

        metrics = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
        }
        if hasattr(model, "predict_proba"):
            metrics["ROC-AUC"] = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

        results[name] = metrics

    # 5-fold cross-validation (recall) using a pipeline to avoid leakage across folds
    cv_results = {}
    for name, model in model_objects.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="recall")
        cv_results[name] = scores

    comparison = pd.DataFrame(results).T
    comparison["CV_Recall_Mean"] = [cv_results[name].mean() for name in comparison.index]
    comparison["Score"] = (comparison["Accuracy"] + comparison["CV_Recall_Mean"]) / 2

    best_model_name = comparison["Score"].idxmax()
    final_model = model_objects[best_model_name]

    return {
        "X": X, "y": y,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "model_objects": model_objects,
        "predictions": predictions,
        "comparison": comparison,
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "final_model": final_model,
    }


# ------------------------------------------------------------
# Sidebar - Data Source
# ------------------------------------------------------------
st.sidebar.title("Heart Disease Prediction")
uploaded_file = st.sidebar.file_uploader("Upload heart.csv", type=["csv"])

if uploaded_file is None:
    st.info("Upload the heart.csv dataset from the sidebar to get started.")
    st.stop()

df = load_data(uploaded_file)

page = st.sidebar.radio(
    "Navigate",
    ["Data Overview", "Model Comparison", "Feature Importance", "Predict"]
)

with st.spinner("Training models... (only runs once per dataset)"):
    state = train_models(df)

# ------------------------------------------------------------
# Page 1: Data Overview / EDA
# ------------------------------------------------------------
if page == "Data Overview":
    st.title("Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preview")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
    with col2:
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().rename("Missing Count"))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Target Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax)
    ax.set_title("Heart Disease Distribution")
    st.pyplot(fig)

# ------------------------------------------------------------
# Page 2: Model Comparison
# ------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")

    st.subheader("Metrics Table")
    st.dataframe(state["comparison"].style.highlight_max(axis=0, color="lightgreen"))

    st.success(f"Best Model (Accuracy + CV Recall): **{state['best_model_name']}**")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Accuracy by Model")
        fig, ax = plt.subplots()
        ax.bar(state["comparison"].index, state["comparison"]["Accuracy"])
        ax.set_ylabel("Accuracy")
        plt.xticks(rotation=20)
        st.pyplot(fig)
    with colB:
        st.subheader("Recall by Model")
        fig, ax = plt.subplots()
        ax.bar(state["comparison"].index, state["comparison"]["Recall"], color="purple")
        ax.set_ylabel("Recall")
        plt.xticks(rotation=20)
        st.pyplot(fig)

    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select model", list(state["model_objects"].keys()))
    cm = confusion_matrix(state["y_test"], state["predictions"][selected_model])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(state["y_test"], state["predictions"][selected_model]))

    st.subheader("Cross-Validation (5-Fold Recall)")
    for name, scores in state["cv_results"].items():
        st.write(f"**{name}** — Mean Recall: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")

# ------------------------------------------------------------
# Page 3: Feature Importance
# ------------------------------------------------------------
elif page == "Feature Importance":
    st.title("Feature Importance")

    best_model_name = state["best_model_name"]
    final_model = state["final_model"]

    if best_model_name in ["Random Forest", "XGBoost"]:
        importance_values = final_model.feature_importances_
        feature_importance = pd.DataFrame({
            "Feature": state["X"].columns,
            "Importance": importance_values
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(feature_importance)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feature_importance["Feature"], feature_importance["Importance"], color="teal")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"{best_model_name} Feature Importance")
        st.pyplot(fig)
    else:
        st.warning(f"Feature importance is only available for Random Forest and XGBoost. Best model here is {best_model_name}.")

# ------------------------------------------------------------
# Page 4: Predict - Interactive Patient Input
# ------------------------------------------------------------
elif page == "Predict":
    st.title(f"Predict Heart Disease Risk (using {state['best_model_name']})")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
            trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
            thal = st.selectbox("Thal", options=[0, 1, 2, 3])

        with col2:
            chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)

        with col3:
            exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2])
            ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3, 4])

        submitted = st.form_submit_button("Predict")

    if submitted:
        patient_data = [age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]

        new_patient = pd.DataFrame([patient_data], columns=state["X"].columns)
        new_patient_scaled = state["scaler"].transform(new_patient)

        final_model = state["final_model"]
        prediction = final_model.predict(new_patient_scaled)

        if hasattr(final_model, "predict_proba"):
            probability = final_model.predict_proba(new_patient_scaled)[0][1]
        else:
            probability = None

        st.divider()
        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")

        if probability is not None:
            st.metric("Model Confidence (Heart Disease Probability)", f"{probability * 100:.2f}%")
