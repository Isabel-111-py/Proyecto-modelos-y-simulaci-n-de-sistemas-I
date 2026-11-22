from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
import model_registry as registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_MODEL = PROJECT_ROOT / "models" / "v3" / "model.pkl"

PATIENT = {
    "Age": 40, "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0, "MaxHR": 170,
    "Oldpeak": 0.0, "Sex": "M", "ChestPainType": "ATA", "RestingECG": "Normal",
    "ExerciseAngina": "N", "ST_Slope": "Up",
}
RECORDS = [{**PATIENT, "HeartDisease": i % 2} for i in range(100)]

client = TestClient(main.app)


def isolated_registry(tmp_path, monkeypatch, active="v3", metrics=(0.86, 0.86), with_model=True):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(registry, "MODELS_DIR", models_dir)
    monkeypatch.setattr(registry, "CURRENT_MODEL_FILE", models_dir / "current_model.txt")
    if with_model:
        (models_dir / active).mkdir()
        (models_dir / active / "model.pkl").write_bytes(b"modelo activo")
        (models_dir / active / "metadata.json").write_text(
            json.dumps({"metrics": {"accuracy": metrics[0], "macro_f1": metrics[1]}}), encoding="utf-8"
        )
    registry.set_current_version(active)
    return models_dir


def use_real_v3(tmp_path, monkeypatch):
    """Apunta el registro al modelo v3 real sin depender de (ni modificar) models/current_model.txt."""
    current = tmp_path / "current_model.txt"
    current.write_text("v3\n", encoding="utf-8")
    monkeypatch.setattr(registry, "CURRENT_MODEL_FILE", current)


def stub_predict(monkeypatch, prediction=1, probability=0.9):
    def fake(data, model_path):
        result = data.copy()
        result["prediction"] = prediction
        result["probability"] = probability
        return result

    monkeypatch.setattr(main, "predict", fake)


def fake_trainer(models_dir, accuracy, macro_f1):
    def fake_train(data: pd.DataFrame, version: str, output_dir, force: bool = False):
        path = models_dir / version / "model.pkl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"modelo candidato")
        (path.parent / "metadata.json").write_text(
            json.dumps({"metrics": {"accuracy": accuracy, "macro_f1": macro_f1}}), encoding="utf-8"
        )
        return path

    return fake_train


# ------------------------------------------------------------------ general
def test_root_describes_service():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/health"


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="El modelo v3 no está disponible")
def test_health_reports_loaded_model(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    use_real_v3(tmp_path, monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["model_version"] == "v3"


def test_health_is_degraded_when_model_is_missing(tmp_path, monkeypatch):
    isolated_registry(tmp_path, monkeypatch, with_model=False)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


# ------------------------------------------------------------------ /predict
@pytest.mark.skipif(not REAL_MODEL.exists(), reason="El modelo v3 no está disponible")
def test_predict_with_real_model(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    use_real_v3(tmp_path, monkeypatch)
    response = client.post("/predict", json=PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert body["model_version"] == "v3"


def test_predict_returns_prediction_with_stub_model(tmp_path, monkeypatch):
    isolated_registry(tmp_path, monkeypatch)
    stub_predict(monkeypatch, prediction=1, probability=0.9)
    response = client.post("/predict", json=PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["probability"] == 0.9
    assert body["model_version"] == "v3"
    assert body["latency_ms"] >= 0


def test_predict_returns_503_without_active_model(tmp_path, monkeypatch):
    isolated_registry(tmp_path, monkeypatch, with_model=False)
    assert client.post("/predict", json=PATIENT).status_code == 503


def test_predict_returns_500_when_the_model_fails(tmp_path, monkeypatch):
    isolated_registry(tmp_path, monkeypatch)

    def broken(data, model_path):
        raise RuntimeError("modelo dañado")

    monkeypatch.setattr(main, "predict", broken)
    response = client.post("/predict", json=PATIENT)
    assert response.status_code == 500
    assert "modelo dañado" in response.json()["detail"]


def test_predict_rejects_invalid_category():
    assert client.post("/predict", json={**PATIENT, "Sex": "unknown"}).status_code == 422


def test_predict_rejects_out_of_range_value():
    assert client.post("/predict", json={**PATIENT, "Age": 200}).status_code == 422


def test_predict_rejects_unknown_field():
    assert client.post("/predict", json={**PATIENT, "not_a_feature": 10}).status_code == 422


def test_predict_rejects_missing_field():
    incomplete = {key: value for key, value in PATIENT.items() if key != "Age"}
    assert client.post("/predict", json=incomplete).status_code == 422


# ------------------------------------------------- extensión para la Fase 4
def test_prediction_hook_is_called_and_its_failure_is_tolerated(tmp_path, monkeypatch):
    isolated_registry(tmp_path, monkeypatch)
    stub_predict(monkeypatch)
    calls = []
    app = main.create_app(on_prediction=lambda **kwargs: calls.append(kwargs))
    assert TestClient(app).post("/predict", json=PATIENT).status_code == 200
    assert calls[0]["prediction"] == 1 and calls[0]["model_version"] == "v3"

    def failing_hook(**kwargs):
        raise OSError("disco lleno")

    broken = TestClient(main.create_app(on_prediction=failing_hook))
    assert broken.post("/predict", json=PATIENT).status_code == 200


def test_metrics_exists_only_when_a_provider_is_given():
    assert client.get("/metrics").status_code == 404
    with_metrics = TestClient(main.create_app(metrics_provider=lambda: {"total_predictions": 7}))
    assert with_metrics.get("/metrics").json() == {"total_predictions": 7}


# ------------------------------------------------------------------- /train
def test_train_activates_new_version_when_it_is_good_enough(tmp_path, monkeypatch):
    models_dir = isolated_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "train_model", fake_trainer(models_dir, 0.85, 0.84))
    response = client.post("/train", json={"version": "v4", "data": RECORDS})
    assert response.status_code == 200
    assert response.json()["model_version"] == "v4"
    assert response.json()["activation"]["activated"] is True
    assert registry.get_current_version() == "v4"


def test_train_keeps_active_model_when_candidate_drops_too_much(tmp_path, monkeypatch):
    models_dir = isolated_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "train_model", fake_trainer(models_dir, 0.70, 0.68))
    response = client.post("/train", json={"version": "v4", "data": RECORDS})
    assert response.status_code == 200
    assert response.json()["activation"]["activated"] is False
    assert response.json()["active_model_version"] == "v3"
    assert registry.get_current_version() == "v3"


def test_train_assigns_next_version_when_omitted(tmp_path, monkeypatch):
    models_dir = isolated_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "train_model", fake_trainer(models_dir, 0.86, 0.86))
    response = client.post("/train", json={"data": RECORDS})
    assert response.status_code == 200
    assert response.json()["model_version"] == "v4"


def test_train_rejects_too_few_records():
    assert client.post("/train", json={"data": RECORDS[:10]}).status_code == 422


def test_train_rejects_invalid_version_name():
    assert client.post("/train", json={"version": "nuevo", "data": RECORDS}).status_code == 422


def test_train_returns_409_when_version_already_exists(tmp_path, monkeypatch):
    isolated_registry(tmp_path, monkeypatch)

    def already_there(data, version, output_dir, force=False):
        raise FileExistsError(f"La versión '{version}' ya existe.")

    monkeypatch.setattr(main, "train_model", already_there)
    assert client.post("/train", json={"version": "v3", "data": RECORDS}).status_code == 409


def test_train_returns_422_when_training_data_is_invalid(tmp_path, monkeypatch):
    isolated_registry(tmp_path, monkeypatch)

    def invalid(data, version, output_dir, force=False):
        raise ValueError("Se requieren al menos 2 registros de cada clase")

    monkeypatch.setattr(main, "train_model", invalid)
    assert client.post("/train", json={"data": RECORDS}).status_code == 422
