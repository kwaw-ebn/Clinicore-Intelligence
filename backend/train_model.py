"""Train synthetic-only CliniCore MVP models for software testing."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "synthetic" / "Clinicore_Synthetic_Clinical_Dataset_v2.xlsx"
OUTPUT = Path(__file__).resolve().parent

PRE_LAB_FEATURES = [
    "age_years", "sex_at_birth", "pregnancy_status", "symptom_duration_days",
    "fever_reported", "chills", "headache", "cough", "difficulty_breathing",
    "fatigue", "sore_throat", "runny_nose", "nausea", "vomiting", "diarrhea",
    "abdominal_pain", "painful_urination", "urinary_frequency", "flank_pain",
    "rash", "itching", "confusion", "diabetes_history", "hypertension_history",
    "asthma_history", "temperature_c", "heart_rate_bpm", "respiratory_rate_bpm",
    "spo2_percent", "systolic_bp_mmhg", "diastolic_bp_mmhg", "weight_kg",
    "height_cm", "bmi_kg_m2",
]
LAB_FEATURES = ["malaria_rdt", "hemoglobin_g_dl", "wbc_10e9_l",
                "glucose_test_type", "glucose_mmol_l",
                "urine_leukocyte_esterase", "urine_nitrite"]
CATEGORICAL = {"sex_at_birth", "pregnancy_status", "malaria_rdt",
               "glucose_test_type", "urine_leukocyte_esterase", "urine_nitrite"}


def make_pipeline(features, class_count, binary=False):
    categorical = [x for x in features if x in CATEGORICAL]
    numeric = [x for x in features if x not in CATEGORICAL]
    transform = ColumnTransformer([
        ("number", SimpleImputer(strategy="median"), numeric),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    settings = dict(n_estimators=140, max_depth=4, learning_rate=0.06,
                    subsample=0.9, colsample_bytree=0.9, random_state=4545,
                    n_jobs=2, eval_metric="logloss" if binary else "mlogloss")
    if not binary:
        settings.update(objective="multi:softprob", num_class=class_count)
    return Pipeline([("preprocess", transform), ("model", XGBClassifier(**settings))])


def train_multiclass(df, features, target, filename):
    train, test = df[df.split == "train"], df[df.split == "test"]
    labels = sorted(train[target].unique())
    ids = {label: index for index, label in enumerate(labels)}
    model = make_pipeline(features, len(labels))
    model.fit(train[features], train[target].map(ids))
    predicted = model.predict(test[features])
    joblib.dump({"model": model, "labels": labels, "features": features,
                 "model_kind": target, "model_version": "synthetic-v2-2026-09",
                 "synthetic_only": True, "clinical_use": False}, OUTPUT / filename)
    return {"accuracy": round(float(accuracy_score(test[target].map(ids), predicted)), 4),
            "report": classification_report(test[target].map(ids), predicted,
                labels=list(range(len(labels))), target_names=labels,
                zero_division=0, output_dict=True)}


def train_risk(df):
    features = PRE_LAB_FEATURES + LAB_FEATURES
    train, test = df[df.split == "train"], df[df.split == "test"]
    model = make_pipeline(features, 2, binary=True)
    model.fit(train[features], train.urgent_clinician_review)
    probabilities = model.predict_proba(test[features])[:, 1]
    joblib.dump({"model": model, "features": features,
                 "model_kind": "urgent_clinician_review",
                 "model_version": "synthetic-v2-2026-09", "synthetic_only": True,
                 "clinical_use": False}, OUTPUT / "outcome_model.joblib")
    return {"roc_auc": round(float(roc_auc_score(test.urgent_clinician_review,
                                                  probabilities)), 4)}


def main():
    df = pd.read_excel(DATASET, sheet_name="Synthetic encounters")
    required = set(PRE_LAB_FEATURES + LAB_FEATURES + ["split", "pre_lab_target",
                   "post_lab_support_category", "urgent_clinician_review"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    metrics = {
        "warning": "Synthetic software-test metrics; not clinical performance.",
        "pre_lab": train_multiclass(df, PRE_LAB_FEATURES, "pre_lab_target",
                                    "prelab_model.joblib"),
        "post_lab": train_multiclass(df, PRE_LAB_FEATURES + LAB_FEATURES,
                                     "post_lab_support_category",
                                     "disease_model.joblib"),
        "risk": train_risk(df),
    }
    (OUTPUT / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"pre_lab_accuracy": metrics["pre_lab"]["accuracy"],
                      "post_lab_accuracy": metrics["post_lab"]["accuracy"],
                      "risk_roc_auc": metrics["risk"]["roc_auc"]}, indent=2))


if __name__ == "__main__":
    main()
