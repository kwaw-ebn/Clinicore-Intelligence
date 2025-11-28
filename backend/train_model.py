# ============================================================
# TRAIN XGBOOST MODELS FOR CLINIC-ASSIST
# Builds:
#   • disease_model.joblib  (multi-class disease prediction)
#   • outcome_model.joblib  (binary risk prediction)
#   • feature_importance.json
#   • roc_data.json
#   • confusion_matrix.json
#
# Dataset required:
#   Disease_symptom_and_patient_profile_dataset.csv
#
# Written for Clinic-Assist / MedIntel Assist
# ============================================================

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

print("Dataset loaded:", df.shape)


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.dropna()
df = df.reset_index(drop=True)

# Expected columns
expected_cols = [
    "age", "gender", "blood_pressure", "cholesterol",
    "fever", "cough", "difficulty_breathing",
    "disease", "risk"
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise Exception(f"Dataset is missing required columns: {missing}")

print("All required columns found.")


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

label_encoders = {}

categorical_columns = ["gender", "blood_pressure", "cholesterol"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# ============================================================
# FEATURES & TARGETS
# ============================================================

X = df[[
    "age", "gender", "blood_pressure", "cholesterol",
    "fever", "cough", "difficulty_breathing"
]]

y_disease = df["disease"]
y_risk = df["risk"]  # binary


# Encode disease labels
disease_encoder = LabelEncoder()
y_disease_encoded = disease_encoder.fit_transform(y_disease)


# ============================================================
# SPLIT TRAIN/TEST
# ============================================================

X_train, X_test, y_train_d, y_test_d = train_test_split(
    X, y_disease_encoded, test_size=0.2, random_state=42, stratify=y_disease_encoded
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
)


# ============================================================
# TRAIN DISEASE MODEL (MULTI-CLASS XGBOOST)
# ============================================================

print("\nTraining disease model...")

disease_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(disease_encoder.classes_)
)

disease_model.fit(X_train, y_train_d)

print("Disease model trained!")


# ============================================================
# TRAIN OUTCOME MODEL (BINARY XGBOOST)
# ============================================================

print("\nTraining outcome model...")

outcome_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss"
)

outcome_model.fit(X_train_r, y_train_r)

print("Outcome model trained!")


# ============================================================
# SAVE MODELS
# ============================================================

bundle_disease = {
    "model": disease_model,
    "encoder": disease_encoder,
    "feature_labels": list(X.columns),
    "categorical_label_encoders": label_encoders
}

bundle_outcome = {
    "model": outcome_model,
    "feature_labels": list(X.columns),
    "categorical_label_encoders": label_encoders
}

joblib.dump(bundle_disease, "disease_model.joblib")
joblib.dump(bundle_outcome, "outcome_model.joblib")

print("\nModels saved:")
print(" - disease_model.joblib")
print(" - outcome_model.joblib")


# ============================================================
# FEATURE IMPORTANCE EXPORT
# ============================================================

importance = disease_model.feature_importances_
feature_importance = {
    "features": list(X.columns),
    "importance": importance.tolist()
}

with open("feature_importance.json", "w") as f:
    json.dump(feature_importance, f, indent=4)

print("Feature importance exported.")


# ============================================================
# ROC CURVE DATA (OUTCOME MODEL)
# ============================================================

print("\nGenerating ROC curve...")

y_proba = outcome_model.predict_proba(X_test_r)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_r, y_proba)
auc_score = roc_auc_score(y_test_r, y_proba)

roc_output = {
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "thresholds": thresholds.tolist(),
    "auc": float(auc_score)
}

with open("roc_data.json", "w") as f:
    json.dump(roc_output, f, indent=4)

print("ROC exported: auc =", auc_score)


# ============================================================
# CONFUSION MATRIX (OUTCOME MODEL)
# ============================================================

print("\nGenerating confusion matrix...")

y_pred = outcome_model.predict(X_test_r)
cm = confusion_matrix(y_test_r, y_pred)

cm_output = {
    "matrix": cm.tolist(),
    "labels": ["Low Risk", "High Risk"]
}

with open("confusion_matrix.json", "w") as f:
    json.dump(cm_output, f, indent=4)

print("Confusion matrix exported.")


# ============================================================
# DONE
# ============================================================

print("\n=============================================")
print("TRAINING COMPLETE — ALL FILES GENERATED")
print("=============================================")
print(" Files created:")
print("  - disease_model.joblib")
print("  - outcome_model.joblib")
print("  - feature_importance.json")
print("  - roc_data.json")
print("  - confusion_matrix.json")
print("=============================================")
