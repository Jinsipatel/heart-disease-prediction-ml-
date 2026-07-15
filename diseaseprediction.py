# ============================================================
# Heart Disease Prediction - Multi-Model Classification Pipeline
# ============================================================
# This script loads the UCI Heart Disease dataset, performs EDA,
# trains multiple classifiers, compares them using accuracy/recall
# and cross-validation, selects the best model, and finally allows
# real-time prediction based on user-entered patient data.

# ---------------- Imports ----------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score


# ---------------- Load Dataset ----------------
df = pd.read_csv("heart.csv")


# ---------------- Initial Data Exploration ----------------
print(df.head())               # Preview first 5 rows

print(df.shape)                 # Number of rows and columns

print(df.info())                # Data types and non-null counts

print(df.describe())            # Statistical summary of numeric columns

# check the null value
print(df.isnull().sum())        # Count missing values per column

print(df.duplicated().sum())    # Count duplicate rows

# check the 
df = df.drop_duplicates()       # Remove duplicate rows to avoid data leakage/bias

print(df['target'].value_counts())   # Class distribution (disease vs no disease)

# Visualize target class balance
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# ---------------- Feature/Target Split ----------------
X = df.drop("target", axis=1)   # Features (all columns except target)

y = df["target"]                # Target label (1 = disease, 0 = no disease)

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ---------------- Feature Scaling ----------------
# Standardize features (mean=0, std=1) - important for distance/gradient based models like LR & SVM
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)   # Fit scaler on training data only (avoids data leakage)

X_test = scaler.transform(X_test)         # Apply same transformation to test data

# ============================================================
# Model 1: Logistic Regression
# ============================================================
lr = LogisticRegression(random_state=42)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Logistic Regression Results")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision :", precision_score(y_test, lr_pred))
print("Recall :", recall_score(y_test, lr_pred))
print("F1 Score :", f1_score(y_test, lr_pred))

cm = confusion_matrix(y_test, lr_pred)
print(cm)

# Visualize confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, lr_pred))

# ============================================================
# Model 2: Support Vector Machine (SVM)
# ============================================================
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

print("SVM Results")
print("Accuracy :", accuracy_score(y_test, svm_pred))
print("Precision :", precision_score(y_test, svm_pred))
print("Recall :", recall_score(y_test, svm_pred))
print("F1 Score :", f1_score(y_test, svm_pred))

cm = confusion_matrix(y_test, svm_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, svm_pred))

# ============================================================
# Model 3: Random Forest
# ============================================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Results")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision :", precision_score(y_test, rf_pred))
print("Recall :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))
# ROC-AUC uses predicted probabilities rather than hard class labels
print("ROC-AUC Score :", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, rf_pred))

# ============================================================
# Model 4: XGBoost
# ============================================================
xgb = XGBClassifier(random_state=42,eval_metric='logloss',n_estimators=150,max_depth=3,learning_rate=0.1)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("XGBoost Results")
print("Accuracy :", accuracy_score(y_test, xgb_pred))
print("Precision :", precision_score(y_test, xgb_pred))
print("Recall :", recall_score(y_test, xgb_pred))
print("F1 Score :", f1_score(y_test, xgb_pred))
print("ROC-AUC Score :", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, xgb_pred))

# ============================================================
# Model Comparison (Accuracy & Recall)
# ============================================================
models = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]
model_objects = {
    "Logistic Regression": lr,
    "SVM": svm,
    "Random Forest": rf,
    "XGBoost": xgb
}

accuracy = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, svm_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, xgb_pred)
]
recall = [
    recall_score(y_test, lr_pred),
    recall_score(y_test, svm_pred),
    recall_score(y_test, rf_pred),
    recall_score(y_test, xgb_pred)
]
comparison = pd.DataFrame({"Model": models, "Accuracy": accuracy, "Recall": recall})
print(comparison)

# ============================================================
# Cross-Validation for Reliability Check
# ============================================================
# Recall is prioritized here since missing a heart disease case (false negative)
# is more costly than a false alarm in a medical diagnosis context.
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

print("\n--- Cross-Validation Results (5-Fold) ---")

cv_results = {}

for name, model in model_objects.items():
    # Wrap scaler + model in a pipeline so scaling is refit correctly within each CV fold
    # (prevents data leakage from test folds into training folds)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='recall')
    cv_results[name] = scores
    print(f"{name}: Mean Recall = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")

# ---------------- Visualization: Accuracy Comparison ----------------
plt.figure(figsize=(8,5))
plt.bar(comparison["Model"], comparison["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()

# ---------------- Visualization: Recall Comparison ----------------
plt.figure(figsize=(8,5))
plt.bar(comparison["Model"], comparison["Recall"], color='purple')
plt.title("Model Recall Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Recall")
plt.show()

# ---------------- Dynamic Best Model Selection ----------------
# Combine test accuracy with cross-validated recall to pick the most reliable model,
# rather than relying on a single train/test split (which can be misleading)
cv_means = [cv_results[name].mean() for name in models]
comparison["CV_Recall_Mean"] = cv_means

comparison["Score"] = (comparison["Accuracy"] + comparison["CV_Recall_Mean"]) / 2
best_model = comparison.loc[comparison["Score"].idxmax()]
print("Best Model")
print(best_model)

best_model_name = best_model["Model"]           
final_model = model_objects[best_model_name]

# ============================================================
# Feature Importance - Best Model
# ============================================================
# Only tree-based models (Random Forest, XGBoost) expose feature_importances_
if best_model_name in ["Random Forest", "XGBoost"]:
    importance_values = final_model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance_values
    }).sort_values(by="Importance", ascending=False)

    print(f"\n{best_model_name} Feature Importance")
    print(feature_importance)

    plt.figure(figsize=(8,5))
    plt.barh(feature_importance["Feature"], feature_importance["Importance"], color='teal')
    plt.title(f"{best_model_name} Feature Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.show()

else:
    print(f"\nFeature importance visualization not available for {best_model_name} (only supported for Random Forest and XGBoost).")

# ---------------- Final Summary of Best Model ----------------
print("Best Performing Model :", best_model["Model"])
print("Accuracy :", best_model["Accuracy"])
print("Recall :", best_model["Recall"])
print("Score :", best_model["Score"])

print(f"\nCross-Validation Reliability for {best_model_name}:")
print(f"Mean Recall : {cv_results[best_model_name].mean():.4f}")
print(f"Std Dev : {cv_results[best_model_name].std():.4f}")


# ============================================================
# Interactive Prediction - Real-Time Patient Input
# ============================================================
def get_patient_input():
    """
    Collects patient health details via console input and returns
    them as a list matching the feature order used during training.
    Returns None if invalid (non-numeric) input is entered, so the
    caller can prompt again.
    """
    print("\n--- Enter Patient Details for Prediction ---")
    try:
        age = int(input("Age: "))
        sex = int(input("Sex (1 = Male, 0 = Female): "))
        cp = int(input("Chest Pain Type (0-3): "))
        trestbps = int(input("Resting Blood Pressure: "))
        chol = int(input("Cholesterol: "))
        fbs = int(input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No): "))
        restecg = int(input("Resting ECG Result (0-2): "))
        thalach = int(input("Max Heart Rate Achieved: "))
        exang = int(input("Exercise Induced Angina (1 = Yes, 0 = No): "))
        oldpeak = float(input("Oldpeak (ST depression): "))
        slope = int(input("Slope of ST Segment (0-2): "))
        ca = int(input("Number of Major Vessels (0-4): "))
        thal = int(input("Thal (0-3): "))

        return [age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]

    except ValueError:
        print("Invalid input! Please enter numeric values only.\n")
        return None


# Keep asking until valid input is provided
patient_data = None
while patient_data is None:
    patient_data = get_patient_input()

# Convert input into a DataFrame with the same column names as training data
new_patient = pd.DataFrame([patient_data], columns=X.columns)
# Apply the SAME scaler fitted on training data (not a new one) for consistent scaling
new_patient = scaler.transform(new_patient)

# Predict using the dynamically selected best model
prediction = final_model.predict(new_patient)
print("\nPrediction :", prediction)

# Show prediction confidence if the model supports probability outputs
if hasattr(final_model, "predict_proba"):
    probability = final_model.predict_proba(new_patient)[0][1]
    print(f"Model Confidence (Heart Disease Probability): {probability*100:.2f}%")


if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")


print("Project Completed Successfully")
