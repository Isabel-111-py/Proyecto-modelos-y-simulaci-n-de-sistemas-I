"""API REST de predicción de enfermedad cardíaca (Fase 3).

Endpoints: GET /health, POST /predict, POST /train.
``create_app`` permite que la Fase 4 amplíe esta misma API con monitoreo
(registro de predicciones y /metrics) sin duplicar código.

Ejecución local (desde la raíz del repositorio):
    uvicorn main:app --app-dir fase-3 --reload
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("fase-2", "fase-3"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import model_registry as registry  # noqa: E402
from predict import load_artifact_cached, predict  # noqa: E402
from schemas import HealthResponse, PatientFeatures, PredictionResponse, TrainRequest  # noqa: E402
from train import train_model  # noqa: E402

logger = logging.getLogger("heart-disease-api")
_TRAIN_LOCK = threading.Lock()


def create_app(
    *,
    on_prediction: Callable[..., None] | None = None,
    on_request: Callable[..., None] | None = None,
    metrics_provider: Callable[[], dict[str, Any]] | None = None,
    version: str = "1.0.0",
) -> FastAPI:
    """Construye la aplicación. Los parámetros opcionales los usa la Fase 4."""
    app = FastAPI(
        title="Heart Disease Prediction API",
        description="API REST para consultar el modelo XGBoost de predicción de enfermedad cardíaca.",
        version=version,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        if request.url.path == "/predict" and on_request is not None:
            try:
                body = await request.json()
            except Exception:
                body = {}
            on_request(status_http=422, input_data=body)
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        if request.url.path == "/predict" and on_request is not None:
            try:
                body = await request.json()
            except Exception:
                body = {}
            on_request(status_http=exc.status_code, input_data=body)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/", tags=["general"])
    def root() -> dict:
        return {"message": "Heart Disease Prediction API", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse, tags=["general"])
    def health() -> dict:
        model_version = registry.get_current_version()
        model_path = registry.get_model_path(model_version)
        loaded = False
        error = None
        if model_path.exists():
            try:
                load_artifact_cached(model_path)
                loaded = True
            except Exception as exc:  # /health informa el fallo en lugar de caerse
                error = str(exc)
        else:
            error = f"No se encontró el archivo del modelo: {model_path}"
        return {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "model_version": model_version,
            "error": error,
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict_endpoint(patient: PatientFeatures) -> PredictionResponse:
        model_version = registry.get_current_version()
        model_path = registry.get_model_path(model_version)
        if not model_path.exists():
            raise HTTPException(status_code=503, detail=f"No se encontró el modelo activo: {model_path}")

        started = time.perf_counter()
        try:
            frame = pd.DataFrame([patient.model_dump()])
            result = predict(frame, model_path)
            prediction = int(result.iloc[0]["prediction"])
            probability = float(result.iloc[0]["probability"]) if "probability" in result else None
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"No fue posible generar la predicción: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if on_prediction is not None:
            try:
                on_prediction(
                    model_version=model_version,
                    prediction=prediction,
                    probability=probability,
                    latency_ms=latency_ms,
                    input_data=patient.model_dump(),
                )
            except Exception:  # un fallo del registro no debe impedir devolver la predicción
                logger.exception("No fue posible registrar la predicción")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=model_version,
            latency_ms=round(latency_ms, 4),
        )

    @app.post("/train", tags=["training"])
    def train_endpoint(request: TrainRequest) -> dict:
        if not _TRAIN_LOCK.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Ya hay un entrenamiento en curso. Intente de nuevo cuando termine.")
        try:
            new_version = request.version or registry.next_version()
            frame = pd.DataFrame([record.model_dump() for record in request.data])
            try:
                artifact_path = train_model(
                    frame, version=new_version, output_dir=registry.MODELS_DIR, force=False
                )
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except (ValueError, KeyError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"No fue posible entrenar el modelo: {exc}") from exc

            # Nunca se reemplaza el modelo activo a ciegas: se compara con sus métricas guardadas.
            activation = registry.activate_if_acceptable(new_version)
        finally:
            _TRAIN_LOCK.release()

        return {
            "message": "Entrenamiento completado. La activación depende de la evaluación.",
            "model_version": new_version,
            "model_path": str(Path(artifact_path)),
            "activation": activation,
            "active_model_version": registry.get_current_version(),
        }

    if metrics_provider is not None:

        @app.get("/metrics", tags=["monitoring"])
        def metrics_endpoint() -> dict:
            return metrics_provider()

    return app


app = create_app()
