"""CliniCore Intelligence MVP API.

Research prototype only. Predictions are decision-support signals and must not be
used as a diagnosis or as the sole basis for treatment.
"""
import os
import traceback
import uuid
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.metrics import auc, confusion_matrix, roc_curve

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "Frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
allowed_origins = [value.strip() for value in os.getenv("ALLOWED_ORIGINS", "").split(",") if value.strip()]
CORS(app, resources={r"/*": {"origins": allowed_origins or "*"}})
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


def load_bundle(filename):
    path = BASE_DIR / filename
    try:
        return joblib.load(path) if path.exists() else None
    except Exception as exc:
        app.logger.error("Could not load %s: %s", filename, exc)
        return None


disease_bundle = load_bundle("disease_model.joblib")
outcome_bundle = load_bundle("outcome_model.joblib")
disease_model = disease_bundle.get("model") if disease_bundle else None
disease_labels = disease_bundle.get("labels", []) if disease_bundle else []
disease_features = disease_bundle.get("features", []) if disease_bundle else []
outcome_model = outcome_bundle.get("model") if outcome_bundle else None
API_VERSION = "1.2.0"
MODEL_VERSION = os.getenv("MODEL_VERSION", "mvp-2026-09")


def json_body():
    if not request.is_json:
        raise ValueError("Content-Type must be application/json")
    return request.get_json(silent=True) or {}


def yes(value):
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def preprocess_input(payload):
    try:
        age = float(payload.get("Age", payload.get("age")))
    except (TypeError, ValueError):
        raise ValueError("Age must be a number between 0 and 120")
    if not 0 <= age <= 120:
        raise ValueError("Age must be between 0 and 120")

    bp = str(payload.get("BloodPressure", payload.get("Blood Pressure", "Normal"))).title()
    chol = str(payload.get("Cholesterol", payload.get("Cholesterol Level", "Normal"))).title()
    if bp not in {"Low", "Normal", "High"} or chol not in {"Low", "Normal", "High"}:
        raise ValueError("Blood pressure and cholesterol must be Low, Normal, or High")

    values = [
        yes(payload.get("Fever", "No")), yes(payload.get("Cough", "No")),
        yes(payload.get("Fatigue", "No")),
        yes(payload.get("DifficultyBreathing", payload.get("Difficulty Breathing", "No"))),
        age, str(payload.get("Gender", "Female")).lower() in {"male", "m", "1"},
        {"Low": 0, "Normal": 1, "High": 2}[bp],
        {"Low": 0, "Normal": 1, "High": 2}[chol],
    ]
    return np.asarray(values, dtype=float).reshape(1, -1)


def prototype_meta():
    return {"prototype": True, "clinical_use": False, "api_version": API_VERSION,
            "model_version": MODEL_VERSION, "request_id": str(uuid.uuid4()),
            "disclaimer": "For supervised MVP testing only. Not a diagnosis or treatment recommendation."}


@app.get("/health")
def health():
    ready = disease_model is not None and outcome_model is not None
    return jsonify({"status": "ok" if ready else "degraded", "models_loaded": ready, **prototype_meta()}), 200 if ready else 503


@app.get("/meta")
def metadata():
    ready = disease_model is not None and outcome_model is not None
    return jsonify({"models_loaded": ready, "features": len(disease_features), **prototype_meta()})


@app.post("/predict-disease")
def predict_disease():
    try:
        if disease_model is None:
            return jsonify(error="Disease model is unavailable", **prototype_meta()), 503
        probabilities = disease_model.predict_proba(preprocess_input(json_body()))[0]
        ranked = sorted(zip(disease_labels, probabilities), key=lambda item: item[1], reverse=True)[:3]
        return jsonify(top3=[{"condition": str(name), "confidence": round(float(score), 4)} for name, score in ranked], **prototype_meta())
    except ValueError as exc:
        return jsonify(error=str(exc), **prototype_meta()), 400
    except Exception:
        app.logger.exception("Disease prediction failed")
        return jsonify(error="Prediction could not be completed", **prototype_meta()), 500


@app.post("/predict-outcome")
def predict_outcome():
    try:
        if outcome_model is None:
            return jsonify(error="Outcome model is unavailable", **prototype_meta()), 503
        probability = float(outcome_model.predict_proba(preprocess_input(json_body()))[0][1])
        return jsonify(risk="Higher model-estimated risk" if probability >= 0.5 else "Lower model-estimated risk",
                       probability=round(probability, 4), **prototype_meta())
    except ValueError as exc:
        return jsonify(error=str(exc), **prototype_meta()), 400
    except Exception:
        app.logger.exception("Outcome prediction failed")
        return jsonify(error="Risk estimate could not be completed", **prototype_meta()), 500


@app.get("/feature-importance")
def feature_importance():
    values = getattr(disease_model, "feature_importances_", []) if disease_model else []
    return jsonify([{"feature": str(f), "importance": round(float(v), 4)} for f, v in zip(disease_features, values)])


@app.post("/roc-data")
def roc_data():
    try:
        body = json_body(); y_true = body.get("y_true", []); y_prob = body.get("y_prob", [])
        if len(y_true) != len(y_prob) or len(set(y_true)) < 2:
            raise ValueError("Equal-length arrays containing both outcome classes are required")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return jsonify(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=round(float(auc(fpr, tpr)), 4))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/confusion-matrix")
def matrix():
    try:
        body = json_body(); y_true = body.get("y_true", []); y_pred = body.get("y_pred", [])
        if not y_true or len(y_true) != len(y_pred):
            raise ValueError("Equal-length non-empty arrays are required")
        return jsonify(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


def openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)


def demo_chat_reply(message):
    """Return deterministic, non-diagnostic guidance for free MVP testing."""
    text = message.lower()
    urgent_terms = (
        "severe difficulty breathing", "unresponsive", "unconscious",
        "seizure", "heavy bleeding", "chest pain"
    )
    if any(term in text for term in urgent_terms):
        return (
            "FREE DEMO MODE — This description may require urgent assessment. "
            "Follow the facility's emergency protocol and contact the appropriate "
            "local emergency service. Do not rely on this demo response for triage."
        )

    if any(term in text for term in ("fever", "cough", "breathing", "fatigue")):
        checklist = (
            "review onset and duration; record temperature, respiratory rate and "
            "oxygen saturation when available; check hydration and relevant history; "
            "screen for red flags; and apply the approved local clinical guideline."
        )
    elif any(term in text for term in ("blood pressure", "hypertension", "bp")):
        checklist = (
            "repeat the measurement using correct technique; review symptoms, "
            "medicines, pregnancy status when relevant and cardiovascular risk; "
            "then apply the approved local clinical guideline."
        )
    else:
        checklist = (
            "clarify the presenting concern, onset, duration, severity, relevant "
            "history, medicines, allergies, vital signs and red flags."
        )

    return (
        "FREE DEMO MODE — Suggested information-gathering checklist: "
        + checklist
        + " This is a fixed prototype response, not AI-generated advice, a diagnosis "
          "or a treatment recommendation. A qualified clinician must verify all decisions."
    )


def demo_note(transcript):
    """Create a transparent draft without inferring facts absent from the transcript."""
    return (
        "FREE DEMO DRAFT — CLINICIAN REVIEW REQUIRED\n\n"
        "Reported conversation\n"
        + transcript
        + "\n\nRelevant history\nNot structured in free demo mode.\n\n"
          "Objective information\nNot provided unless explicitly stated above.\n\n"
          "Assessment considerations\nNot generated in free demo mode.\n\n"
          "Follow-up\nClinician to verify the transcript, complete missing fields and "
          "apply the appropriate local protocol. This draft is not part of the medical "
          "record until reviewed and approved."
    )


@app.post("/chat")
@app.post("/api/chat")
def chat():
    try:
        message = str(json_body().get("message", "")).strip()
        if not message or len(message) > 4000:
            raise ValueError("Message must contain 1 to 4,000 characters")
        client = openai_client()
        if client is None:
            return jsonify(reply=demo_chat_reply(message), mode="demo", **prototype_meta())
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a clinician-facing prototype assistant. Do not diagnose, prescribe, or invent facts. State uncertainty, recommend clinical verification, and direct urgent or emergency concerns to local emergency services."},
                {"role": "user", "content": message},
            ],
        )
        return jsonify(reply=response.choices[0].message.content.strip(), mode="openai", **prototype_meta())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Chat request failed")
        return jsonify(error="AI assistant request failed"), 502


@app.post("/generate-note")
def generate_note():
    try:
        transcript = str(json_body().get("chat", "")).strip()
        if not transcript or len(transcript) > 12000:
            raise ValueError("Chat transcript must contain 1 to 12,000 characters")
        client = openai_client()
        if client is None:
            return jsonify(note=demo_note(transcript), mode="demo", **prototype_meta())
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=600,
            messages=[
                {"role": "system", "content": "Convert the transcript into a draft note with: Reported symptoms, Relevant history, Objective information, Assessment considerations, and Follow-up. Never add missing facts. Mark unknown information as not provided. Add: Draft for clinician review; not part of the medical record until verified."},
                {"role": "user", "content": transcript},
            ],
        )
        return jsonify(note=response.choices[0].message.content.strip(), mode="openai", **prototype_meta())
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Note generation failed")
        return jsonify(error="Note generation failed"), 502


@app.get("/swagger.yaml")
def swagger_yaml():
    return send_from_directory(BASE_DIR, "swagger.yaml")


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:path>")
def frontend_file(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
