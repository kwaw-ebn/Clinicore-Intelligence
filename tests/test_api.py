import os

import numpy as np
import pytest

import backend.server as server


class FakeDiseaseModel:
    feature_importances_ = np.array([0.1] * 8)

    def predict_proba(self, _features):
        return np.array([[0.7, 0.2, 0.1]])


class FakeOutcomeModel:
    def predict_proba(self, _features):
        return np.array([[0.35, 0.65]])


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "disease_model", FakeDiseaseModel())
    monkeypatch.setattr(server, "outcome_model", FakeOutcomeModel())
    monkeypatch.setattr(server, "disease_labels", ["Condition A", "Condition B", "Condition C"])
    monkeypatch.setattr(server, "disease_features", ["Fever", "Cough", "Fatigue", "Breathing", "Age", "Gender", "BP", "Cholesterol"])
    server.app.config.update(TESTING=True)
    return server.app.test_client()


def valid_payload():
    return {"Age": 35, "Gender": "Female", "Fever": "Yes", "Cough": "No", "Fatigue": "No", "DifficultyBreathing": "No", "BloodPressure": "Normal", "Cholesterol": "Normal"}


def test_health_reports_models_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["models_loaded"] is True
    assert response.json["prototype"] is True


def test_metadata_exposes_version_and_not_clinical_use(client):
    response = client.get("/meta")
    assert response.status_code == 200
    assert response.json["models_loaded"] is True
    assert response.json["clinical_use"] is False
    assert response.json["api_version"] == "1.3.0"
    assert response.json["request_id"]


def test_disease_prediction_contract(client):
    response = client.post("/predict-disease", json=valid_payload())
    assert response.status_code == 200
    assert len(response.json["top3"]) == 3
    assert response.json["top3"][0] == {"condition": "Condition A", "confidence": 0.7}
    assert response.json["clinical_use"] is False


def test_outcome_prediction_contract(client):
    response = client.post("/predict-outcome", json=valid_payload())
    assert response.status_code == 200
    assert response.json["risk"] == "Higher model-estimated risk"
    assert response.json["probability"] == 0.65


@pytest.mark.parametrize("age", [-1, 121, "not-a-number", None])
def test_invalid_age_is_rejected(client, age):
    payload = valid_payload(); payload["Age"] = age
    response = client.post("/predict-disease", json=payload)
    assert response.status_code == 400
    assert "Age" in response.json["error"]


def test_non_json_request_is_rejected(client):
    response = client.post("/predict-disease", data="Age=30")
    assert response.status_code == 400


def test_chat_uses_safe_demo_without_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/chat", json={"message": "What should be reviewed?"})
    assert response.status_code == 200
    assert response.json["mode"] == "demo"
    assert "FREE DEMO MODE" in response.json["reply"]
    assert response.json["clinical_use"] is False


def test_demo_chat_flags_urgent_language(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/chat", json={"message": "Patient reports chest pain"})
    assert response.status_code == 200
    assert "urgent assessment" in response.json["reply"]


def test_note_uses_demo_without_key_and_does_not_infer(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/generate-note", json={"chat": "Clinician: fever for two days"})
    assert response.status_code == 200
    assert response.json["mode"] == "demo"
    assert "fever for two days" in response.json["note"]
    assert "Not structured in free demo mode" in response.json["note"]
