import json

import numpy as np
import pandas as pd
import pytest

import model_registry as registry
import monitoring
from preprocessing import FEATURE_COLUMNS, reference_statistics

PATIENT = {
    "Age": 40, "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0, "MaxHR": 170,
    "Oldpeak": 0.0, "Sex": "M", "ChestPainType": "ATA", "RestingECG": "Normal",
    "ExerciseAngina": "N", "ST_Slope": "Up",
}


def make_inputs(rows=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Age": rng.integers(30, 77, rows),
            "Sex": rng.choice(["M", "F"], rows),
            "ChestPainType": rng.choice(["TA", "ATA", "NAP", "ASY"], rows),
            "RestingBP": rng.integers(90, 190, rows),
            "Cholesterol": rng.integers(120, 400, rows),
            "FastingBS": rng.integers(0, 2, rows),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], rows),
            "MaxHR": rng.integers(80, 200, rows),
            "ExerciseAngina": rng.choice(["Y", "N"], rows),
            "Oldpeak": np.round(rng.uniform(0, 4, rows), 1),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], rows),
        }
    )


def isolate(tmp_path, monkeypatch):
    """Redirige el registro de predicciones, los reportes y los modelos a una carpeta temporal."""
    monkeypatch.setattr(monitoring, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(monitoring, "PREDICTIONS_FILE", tmp_path / "logs" / "predictions.csv")
    monkeypatch.setattr(monitoring, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(monitoring, "REPORT_JSON", tmp_path / "reports" / "monitoring_report.json")
    monkeypatch.setattr(monitoring, "REPORT_MD", tmp_path / "reports" / "monitoring_report.md")
    models_dir = tmp_path / "models"
    (models_dir / "v3").mkdir(parents=True)
    (models_dir / "v3" / "model.pkl").write_bytes(b"x")
    (models_dir / "v3" / "metadata.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.86, "macro_f1": 0.86}}), encoding="utf-8"
    )
    monkeypatch.setattr(registry, "MODELS_DIR", models_dir)
    monkeypatch.setattr(registry, "CURRENT_MODEL_FILE", models_dir / "current_model.txt")
    registry.set_current_version("v3")
    return models_dir


def log_predictions(prediction_values, versions=None):
    for index, prediction in enumerate(prediction_values):
        monitoring.record_prediction(
            model_version=(versions or ["v3"] * len(prediction_values))[index],
            prediction=prediction, probability=0.8 if prediction else 0.2,
            latency_ms=10 + index, input_data=PATIENT,
        )


def test_summary_is_empty_without_predictions(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    summary = monitoring.build_summary()
    assert summary["total_predictions"] == 0
    assert summary["average_latency_ms"] is None


def test_predictions_are_logged_and_summarized(tmp_path, monkeypatch):
    isolate(tmp_path, monkeypatch)
    log_predictions([1, 0, 1, 1], versions=["v3", "v3", "v4", "v4"])
    summary = monitoring.build_summary()
    assert summary["total_predictions"] == 4
    assert summary["predictions_by_class"] == {"0": 1, "1": 3}
    assert summary["positive_rate"] == 0.75
    assert summary["predictions_by_model_version"] == {"v3": 2, "v4": 2}
    assert summary["average_latency_ms"] == 11.5
    assert summary["latency_p95_ms"] == 13.0
    assert monitoring.PREDICTIONS_FILE.read_text(encoding="utf-8").splitlines()[0].startswith("timestamp,")


def test_drift_is_stable_for_the_same_distribution():
    data = make_inputs()
    reference = reference_statistics(data[FEATURE_COLUMNS])
    drift = monitoring.compute_drift(data.sample(60, random_state=1), reference)
    assert drift["status"] == "estable"
    assert drift["drifted_features"] == []


def test_drift_detects_shifted_numeric_and_categorical_inputs():
    data = make_inputs()
    reference = reference_statistics(data[FEATURE_COLUMNS])
    live = data.sample(60, random_state=1).copy()
    live["Age"] = live["Age"] + 20
    live["Sex"] = "M"
    drift = monitoring.compute_drift(live, reference)
    assert drift["status"] == "drift"
    assert "Age" in drift["drifted_features"]
    assert "Sex" in drift["drifted_features"]
    assert monitoring.retraining_recommendation(drift)["evaluate_retraining"] is True


def test_drift_needs_enough_rows_and_a_reference():
    data = make_inputs()
    reference = reference_statistics(data[FEATURE_COLUMNS])
    assert monitoring.compute_drift(data.head(5), reference)["status"] == "datos_insuficientes"
    assert monitoring.compute_drift(data, None)["status"] == "sin_referencia"


def test_report_is_written_as_json_and_markdown(tmp_path, monkeypatch):
    models_dir = isolate(tmp_path, monkeypatch)
    reference = reference_statistics(make_inputs()[FEATURE_COLUMNS])
    metadata = json.loads((models_dir / "v3" / "metadata.json").read_text(encoding="utf-8"))
    metadata["reference_stats"] = reference
    (models_dir / "v3" / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    log_predictions([1, 0, 1])

    json_path, md_path = monitoring.generate_report()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["metrics"]["total_predictions"] == 3
    assert report["active_model_version"] == "v3"
    assert report["drift"]["status"] == "datos_insuficientes"
    assert report["model_versions"][0]["version"] == "v3"
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Reporte de monitoreo" in markdown
    assert "Versiones del modelo" in markdown


def test_monitored_api_logs_predictions_and_exposes_metrics(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import main
    import monitored_api

    isolate(tmp_path, monkeypatch)

    def fake_predict(data, model_path):
        result = data.copy()
        result["prediction"] = 1
        result["probability"] = 0.9
        return result

    monkeypatch.setattr(main, "predict", fake_predict)
    client = TestClient(monitored_api.app)
    assert client.post("/predict", json=PATIENT).status_code == 200
    assert client.post("/predict", json=PATIENT).status_code == 200
    metrics = client.get("/metrics").json()
    assert metrics["total_predictions"] == 2
    assert metrics["predictions_by_class"]["1"] == 2
    assert monitoring.PREDICTIONS_FILE.exists()
