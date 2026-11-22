"""Monitoreo básico: registro de predicciones, métricas operativas, drift y reporte (Fase 4)."""

from __future__ import annotations

import csv
import json
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_PHASE2 = str(ROOT / "fase-2")
if _PHASE2 not in sys.path:
    sys.path.insert(0, _PHASE2)

import pandas as pd  # noqa: E402

import model_registry as registry  # noqa: E402
from preprocessing import FEATURE_COLUMNS, reference_statistics, replace_invalid_zeros  # noqa: E402

LOG_DIR = HERE / "logs"
REPORT_DIR = HERE / "reports"
PREDICTIONS_FILE = LOG_DIR / "predictions.csv"
REPORT_JSON = REPORT_DIR / "monitoring_report.json"
REPORT_MD = REPORT_DIR / "monitoring_report.md"
CSV_FIELDS = ["timestamp", "status_http", "model_version", "prediction", "probability", "latency_ms", "input_json"]

# Umbrales de la política (ver policy.md)
MIN_DRIFT_ROWS = 30          # predicciones mínimas para evaluar drift
NUMERIC_SHIFT_THRESHOLD = 0.5  # desplazamiento de la media, en desviaciones estándar de entrenamiento
CATEGORY_SHIFT_THRESHOLD = 0.20  # cambio máximo en la proporción de una categoría (20 puntos)

_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------- registro
def record_request(
    *, status_http: int, model_version: str = "", prediction: int | None = None,
    probability: float | None = None, latency_ms: float = 0.0,
    input_data: dict[str, Any] | None = None,
) -> None:
    """Agrega cualquier solicitud de predicción, exitosa o rechazada."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_http": int(status_http),
        "model_version": model_version,
        "prediction": "" if prediction is None else int(prediction),
        "probability": "" if probability is None else float(probability),
        "latency_ms": round(float(latency_ms), 4),
        "input_json": json.dumps(input_data or {}, ensure_ascii=False, separators=(",", ":")),
    }
    with _WRITE_LOCK:
        needs_header = not PREDICTIONS_FILE.exists() or PREDICTIONS_FILE.stat().st_size == 0
        with PREDICTIONS_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


def record_prediction(
    *, model_version: str, prediction: int, probability: float | None,
    latency_ms: float, input_data: dict[str, Any],
) -> None:
    """Compatibilidad para registrar una predicción exitosa."""
    record_request(
        status_http=200, model_version=model_version, prediction=prediction,
        probability=probability, latency_ms=latency_ms, input_data=input_data,
    )


def read_log() -> list[dict[str, str]]:
    if not PREDICTIONS_FILE.exists() or PREDICTIONS_FILE.stat().st_size == 0:
        return []
    with PREDICTIONS_FILE.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# ---------------------------------------------------------------- métricas
def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_summary(rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Métricas operativas calculadas a partir del registro de predicciones."""
    rows = read_log() if rows is None else rows
    if not rows:
        return {
            "total_requests": 0,
            "failed_requests": 0,
            "failure_rate": None,
            "total_predictions": 0,
            "predictions_by_class": {"0": 0, "1": 0},
            "positive_rate": None,
            "average_probability": None,
            "predictions_by_model_version": {},
            "average_latency_ms": None,
            "latency_p95_ms": None,
            "first_prediction_at": None,
            "last_prediction_at": None,
        }

    classes = Counter(row.get("prediction", "") for row in rows if row.get("prediction", "") != "")
    versions = Counter(row.get("model_version", "") for row in rows)
    latencies = sorted(v for v in (_to_float(row.get("latency_ms")) for row in rows) if v is not None)
    probabilities = [v for v in (_to_float(row.get("probability")) for row in rows) if v is not None]
    p95 = latencies[int(round(0.95 * (len(latencies) - 1)))] if latencies else None

    statuses = [int(row.get("status_http", "200")) for row in rows]
    failed = sum(400 <= status <= 599 for status in statuses)
    successful = len(rows) - failed
    return {
        "total_requests": len(rows),
        "failed_requests": failed,
        "failure_rate": round(failed / len(rows), 4) if rows else None,
        "total_predictions": successful,
        "predictions_by_class": {"0": classes.get("0", 0), "1": classes.get("1", 0)},
        "positive_rate": round(classes.get("1", 0) / successful, 4) if successful else None,
        "average_probability": round(sum(probabilities) / len(probabilities), 4) if probabilities else None,
        "predictions_by_model_version": dict(versions),
        "average_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "latency_p95_ms": round(p95, 4) if p95 is not None else None,
        "first_prediction_at": rows[0].get("timestamp"),
        "last_prediction_at": rows[-1].get("timestamp"),
    }


# ------------------------------------------------------------------- drift
def load_inputs(rows: list[dict[str, str]]) -> pd.DataFrame:
    records = []
    for row in rows:
        try:
            records.append(json.loads(row["input_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    return pd.DataFrame(records)


def load_reference(reference_csv: str | Path | None = None) -> tuple[dict | None, str]:
    """Referencia de entrenamiento: un CSV indicado o las estadísticas guardadas en el modelo activo."""
    if reference_csv:
        frame = pd.read_csv(reference_csv)
        return reference_statistics(frame[FEATURE_COLUMNS]), f"CSV: {reference_csv}"
    try:
        version = registry.get_current_version()
        metadata = json.loads(registry.get_model_path(version).with_name("metadata.json").read_text(encoding="utf-8"))
        if "reference_stats" in metadata:
            return metadata["reference_stats"], f"metadatos del modelo {version}"
    except (OSError, ValueError):
        pass
    return None, "no disponible"


def compute_drift(
    inputs: pd.DataFrame,
    reference: dict | None,
    *,
    min_rows: int = MIN_DRIFT_ROWS,
    numeric_threshold: float = NUMERIC_SHIFT_THRESHOLD,
    category_threshold: float = CATEGORY_SHIFT_THRESHOLD,
) -> dict[str, Any]:
    """Compara la distribución de las entradas recientes con la de entrenamiento."""
    if reference is None:
        return {"status": "sin_referencia", "rows": int(len(inputs)), "features": {}, "drifted_features": []}
    if len(inputs) < min_rows:
        return {
            "status": "datos_insuficientes", "rows": int(len(inputs)), "min_rows": min_rows,
            "features": {}, "drifted_features": [],
        }

    cleaned = replace_invalid_zeros(inputs)
    features: dict[str, dict[str, Any]] = {}
    drifted: list[str] = []

    for column, ref in reference.get("numeric", {}).items():
        if column not in cleaned.columns:
            continue
        live_mean = pd.to_numeric(cleaned[column], errors="coerce").mean()
        ref_mean, ref_std = ref.get("mean"), ref.get("std")
        if pd.isna(live_mean) or ref_mean is None or not ref_std:
            features[column] = {"type": "numeric", "shift_in_std": None, "drift": False}
            continue
        shift = abs(float(live_mean) - ref_mean) / ref_std
        is_drift = bool(shift > numeric_threshold)
        features[column] = {
            "type": "numeric", "reference_mean": round(ref_mean, 4), "live_mean": round(float(live_mean), 4),
            "shift_in_std": round(shift, 4), "drift": is_drift,
        }
        if is_drift:
            drifted.append(column)

    for column, ref_props in reference.get("categorical", {}).items():
        if column not in cleaned.columns:
            continue
        live_props = cleaned[column].value_counts(normalize=True)
        categories = set(ref_props) | {str(k) for k in live_props.index}
        diffs = {c: abs(float(live_props.get(c, 0.0)) - float(ref_props.get(c, 0.0))) for c in categories}
        worst = max(diffs, key=diffs.get)
        is_drift = bool(diffs[worst] > category_threshold)
        features[column] = {
            "type": "categorical", "max_proportion_change": round(diffs[worst], 4),
            "category": worst, "drift": is_drift,
        }
        if is_drift:
            drifted.append(column)

    return {
        "status": "drift" if drifted else "estable",
        "rows": int(len(inputs)),
        "features": features,
        "drifted_features": drifted,
    }


def retraining_recommendation(drift: dict[str, Any]) -> dict[str, Any]:
    """Aplica los criterios automáticos de la política. Los criterios con etiquetas reales son manuales."""
    reasons = []
    if drift.get("status") == "drift":
        reasons.append("Cambio de distribución en: " + ", ".join(drift["drifted_features"]))
    return {
        "evaluate_retraining": bool(reasons),
        "reasons": reasons,
        "manual_criteria": [
            "Acumular al menos 100 registros nuevos con etiqueta real verificada.",
            "Caída de 5 puntos porcentuales o más en accuracy o macro-F1 con datos etiquetados recientes.",
        ],
    }


# ----------------------------------------------------------------- reporte
def build_report(reference_csv: str | Path | None = None) -> dict[str, Any]:
    rows = read_log()
    reference, reference_source = load_reference(reference_csv)
    drift = compute_drift(load_inputs(rows), reference)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_model_version": registry.get_current_version(),
        "metrics": build_summary(rows),
        "drift": {"reference_source": reference_source, **drift},
        "retraining": retraining_recommendation(drift),
        "model_versions": registry.list_versions(),
        "notes": [
            "Las métricas describen el uso y el tiempo de respuesta de la API.",
            "Sin etiquetas reales posteriores no es posible calcular accuracy, precision o recall en producción.",
        ],
    }


def _fmt(value: Any) -> str:
    return "n/d" if value is None else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    metrics, drift, retraining = report["metrics"], report["drift"], report["retraining"]
    lines = [
        "# Reporte de monitoreo",
        "",
        f"- Generado: {report['generated_at']}",
        f"- Modelo activo: **{report['active_model_version']}**",
        "",
        "## Uso de la API",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Solicitudes totales | {metrics['total_requests']} |",
        f"| Solicitudes fallidas (4xx/5xx) | {metrics['failed_requests']} |",
        f"| Tasa de fallos | {_fmt(metrics['failure_rate'])} |",
        f"| Predicciones registradas | {metrics['total_predictions']} |",
        f"| Clase 0 (normal) | {metrics['predictions_by_class']['0']} |",
        f"| Clase 1 (enfermedad) | {metrics['predictions_by_class']['1']} |",
        f"| Proporción de clase 1 | {_fmt(metrics['positive_rate'])} |",
        f"| Probabilidad media de clase 1 | {_fmt(metrics['average_probability'])} |",
        f"| Latencia media (ms) | {_fmt(metrics['average_latency_ms'])} |",
        f"| Latencia p95 (ms) | {_fmt(metrics['latency_p95_ms'])} |",
        f"| Primera predicción | {_fmt(metrics['first_prediction_at'])} |",
        f"| Última predicción | {_fmt(metrics['last_prediction_at'])} |",
        "",
        f"Uso por versión: {json.dumps(metrics['predictions_by_model_version'], ensure_ascii=False)}",
        "",
        "## Cambio en la distribución de las entradas (drift)",
        "",
        f"- Referencia: {drift['reference_source']}",
        f"- Estado: **{drift['status']}** (registros evaluados: {drift['rows']})",
    ]
    if drift["status"] == "datos_insuficientes":
        lines.append(f"- Se necesitan al menos {drift['min_rows']} predicciones para evaluar drift.")
    if drift["features"]:
        lines += ["", "| Variable | Tipo | Cambio | Drift |", "|---|---|---|---|"]
        for name, info in drift["features"].items():
            if info["type"] == "numeric":
                change = f"{_fmt(info['shift_in_std'])} desv. estándar"
            else:
                change = f"{info['max_proportion_change']} (categoría {info['category']})"
            lines.append(f"| {name} | {info['type']} | {change} | {'sí' if info['drift'] else 'no'} |")
    lines += ["", "## Recomendación de reentrenamiento", ""]
    if retraining["evaluate_retraining"]:
        lines += [f"- Evaluar reentrenamiento: {reason}" for reason in retraining["reasons"]]
    else:
        lines.append("- Ningún criterio automático se cumple.")
    lines.append("- Criterios manuales (requieren etiquetas reales): " + " ".join(retraining["manual_criteria"]))
    lines += ["", "## Versiones del modelo", "", "| Versión | Activa | Accuracy | Macro-F1 | Entrenada |", "|---|---|---|---|---|"]
    for item in report["model_versions"]:
        m = item["metrics"]
        lines.append(
            f"| {item['version']} | {'sí' if item['active'] else 'no'} | {_fmt(round(m['accuracy'], 4) if 'accuracy' in m else None)} "
            f"| {_fmt(round(m['macro_f1'], 4) if 'macro_f1' in m else None)} | {_fmt(item['trained_at'])} |"
        )
    lines += ["", "## Notas", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def generate_report(reference_csv: str | Path | None = None) -> tuple[Path, Path]:
    """Escribe el reporte en JSON y Markdown y devuelve ambas rutas."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(reference_csv)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    return REPORT_JSON, REPORT_MD
