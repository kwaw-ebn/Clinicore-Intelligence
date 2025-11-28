"""
backend/server.py
- Loads XGBoost models saved as joblib bundles:
  {'model': model, 'labels': [...], 'features': [...]}
- Endpoints:
  POST /predict-disease   -> returns top3 disease candidates
  POST /predict-outcome   -> returns risk probability
  GET  /feature-importance-> returns list of feature importance
  POST /roc-data          -> computes ROC from provided arrays (y_true,y_prob)
  POST /confusion-matrix  -> computes confusion matrix from provided arrays
  POST /log-metrics       -> stores metrics in Firestore (if service account configured)
  POST /api/chat          -> proxies to OpenAI (if OPENAI_API_KEY set)
  GET  /swagger.yaml      -> serves swagger docs (simple)
"""
import os, traceback, json
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Optional OpenAI
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    import openai
    openai.api_key = OPENAI_KEY

# Optional Firebase Admin
FIREBASE_ENABLED = False
db = None
try:
    svc = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if svc and os.path.exists(svc):
        import firebase_admin
        from firebase_admin import credentials, firestore
        cred = credentials.Certificate(svc)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        FIREBASE_ENABLED = True
        print("Firebase Admin initialized for server metrics logging.")
except Exception as e:
    print("Firebase admin not enabled:", e)

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

# MODEL LOAD
DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "disease_model.joblib")
OUTCOME_MODEL_PATH = os.path.join(os.path.dirname(__file__), "outcome_model.joblib")

disease_bundle = None
outcome_bundle = None

def safe_load(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        print("Failed loading model", path, e)
    return None

disease_bundle = safe_load(DISEASE_MODEL_PATH)
outcome_bundle = safe_load(OUTCOME_MODEL_PATH)

if disease_bundle:
    disease_model = disease_bundle.get("model")
    disease_labels = disease_bundle.get("labels")
    disease_features = disease_bundle.get("features")
else:
    disease_model = None
    disease_labels = []
    disease_features = []

if outcome_bundle:
    outcome_model = outcome_bundle.get("model")
    outcome_labels = outcome_bundle.get("labels")
    outcome_features = outcome_bundle.get("features")
else:
    outcome_model = None
    outcome_labels = []
    outcome_features = []

# PREPROCESS helpers (accepts keys used in frontend)
def preprocess_input(payload):
    bp_map = {"Low": 0, "Normal": 1, "High": 2}
    chol_map = {"Low": 0, "Normal": 1, "High": 2}

    # tolerate slightly different keys
    fever = payload.get("Fever") or payload.get("fever") or "No"
    cough = payload.get("Cough") or payload.get("cough") or "No"
    fatigue = payload.get("Fatigue") or payload.get("fatigue") or "No"
    dbreath = payload.get("DifficultyBreathing") or payload.get("Difficulty Breathing") or payload.get("dbreath") or "No"
    age = float(payload.get("Age") or payload.get("age") or 0)
    gender = payload.get("Gender") or payload.get("gender") or "Female"
    bp = payload.get("BloodPressure") or payload.get("Blood Pressure") or payload.get("bp_cat") or "Normal"
    chol = payload.get("Cholesterol") or payload.get("Cholesterol Level") or payload.get("chol") or "Normal"

    arr = [
        1 if str(fever).strip().lower() in ("yes","y","true","1") else 0,
        1 if str(cough).strip().lower() in ("yes","y","true","1") else 0,
        1 if str(fatigue).strip().lower() in ("yes","y","true","1") else 0,
        1 if str(dbreath).strip().lower() in ("yes","y","true","1") else 0,
        age,
        1 if str(gender).strip().lower() in ("male","m","1") else 0,
        bp_map.get(str(bp).strip(), 1),
        chol_map.get(str(chol).strip(), 1)
    ]
    return np.array(arr).reshape(1, -1)

# ENDPOINTS
@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    payload = request.get_json() or {}
    try:
        X = preprocess_input(payload)
        if disease_model is None:
            return jsonify({"error":"Disease model not loaded"}), 500
        probs = disease_model.predict_proba(X)[0]
        ranked = sorted(zip(disease_labels, probs), key=lambda x: x[1], reverse=True)
        top3 = [{"disease": name, "confidence": float(round(score,4))} for name,score in ranked[:3]]
        return jsonify({"top3": top3})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict-outcome", methods=["POST"])
def predict_outcome():
    payload = request.get_json() or {}
    try:
        X = preprocess_input(payload)
        if outcome_model is None:
            return jsonify({"error":"Outcome model not loaded"}), 500
        prob = float(outcome_model.predict_proba(X)[0][1])
        label = "High Risk" if prob >= 0.5 else "Low Risk"
        return jsonify({"risk": label, "probability": round(prob,4)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        if disease_model is None:
            return jsonify([])
        importance = getattr(disease_model, "feature_importances_", None)
        if importance is None:
            # XGBoost may have get_booster().get_score()
            try:
                booster = disease_model.get_booster()
                scores = booster.get_score(importance_type='weight')
                # map to features list order
                importance = [scores.get(f, 0.0) for f in disease_features]
            except Exception:
                importance = [0.0]*len(disease_features)
        result = [{"feature": f, "importance": float(round(float(v),4))} for f,v in zip(disease_features, importance)]
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify([])

@app.route("/roc-data", methods=["POST"])
def roc_data():
    try:
        body = request.get_json() or {}
        y_true = list(body.get("y_true", []))
        y_prob = list(body.get("y_prob", []))
        if len(y_true) < 2:
            return jsonify({"error":"Not enough samples"}), 400
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        return jsonify({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(round(auc_val,4))})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/confusion-matrix", methods=["POST"])
def conf_matrix():
    try:
        body = request.get_json() or {}
        y_true = list(body.get("y_true", []))
        y_pred = list(body.get("y_pred", []))
        if len(y_true) == 0:
            return jsonify({"error":"y_true empty"}), 400
        cm = confusion_matrix(y_true, y_pred).tolist()
        return jsonify(cm)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/log-metrics", methods=["POST"])
def log_metrics():
    """
    Stores prediction / model metrics to Firestore collection "model_metrics"
    Expected JSON fields: model (str), payload (dict), prediction (dict), user (optional)
    """
    body = request.get_json() or {}
    if not FIREBASE_ENABLED:
        return jsonify({"ok": False, "message": "Firestore not enabled on this server"}), 500
    try:
        doc = {
            "model": body.get("model"),
            "payload": body.get("payload"),
            "prediction": body.get("prediction"),
            "user": body.get("user"),
            "ts": firestore.SERVER_TIMESTAMP if 'firestore' in globals() else None
        }
        db.collection("model_metrics").add(doc)
        return jsonify({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not OPENAI_KEY:
        return jsonify({"reply":"OpenAI key not configured"}), 500
    try:
        body = request.get_json() or {}
        msg = body.get("message","")
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a clinical assistant. Provide concise, evidence-aware guidance. Mention uncertainty and recommend clinician judgement."},
                {"role":"user","content":msg}
            ],
            max_tokens=512,
            temperature=0.2
        )
        reply = completion.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"AI error: {str(e)}"}), 500

# Simple swagger serve (static YAML)
@app.route("/swagger.yaml", methods=["GET"])
def swagger_yaml():
    swagger_path = os.path.join(os.path.dirname(__file__), "swagger.yaml")
    if os.path.exists(swagger_path):
        return send_from_directory(os.path.dirname(__file__), "swagger.yaml")
    return Response("openapi: '3.0.0'\ninfo:\n  title: Clinic Assist API\n  version: '1.0'\n", mimetype="text/yaml")

# Serve frontend
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "dashboard.html")

# ==========================================
# CHATBOT ENDPOINT + NOTE GENERATION
# ==========================================
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical assistant AI. Provide safe medical explanations. "
                    "Do NOT diagnose. "
                    "Always advise seeing a clinician. "
                    "After answering, ask one relevant follow-up question."
                )
            },
            {"role": "user", "content": message}
        ],
        max_tokens=200
    )

    reply = response.choices[0].message["content"]
    return jsonify({"reply": reply})


# ==========================================
# STRUCTURED DOCTOR NOTE GENERATION
# ==========================================
@app.route("/generate-note", methods=["POST"])
def generate_note():
    data = request.json
    chat = data.get("chat", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Turn the conversation into a structured medical note.\n"
                    "Do NOT diagnose. Use the following format:\n\n"
                    "Symptoms:\n"
                    "- ...\n"
                    "Relevant History:\n"
                    "- ...\n"
                    "AI Observations:\n"
                    "- ...\n"
                    "Recommendations (safe):\n"
                    "- ...\n"
                    "Follow-up advice: Seek clinician confirmation.\n"
                )
            },
            {"role": "user", "content": chat}
        ],
        max_tokens=250
    )

    note = response.choices[0].message["content"]
    return jsonify({"note": note})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
