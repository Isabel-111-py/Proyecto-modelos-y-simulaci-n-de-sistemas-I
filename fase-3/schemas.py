"""Esquemas de entrada y salida de la API (validación con Pydantic)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_PHASE2 = str(Path(__file__).resolve().parents[1] / "fase-2")
if _PHASE2 not in sys.path:
    sys.path.insert(0, _PHASE2)
_PHASE3 = str(Path(__file__).resolve().parent)
if _PHASE3 not in sys.path:
    sys.path.insert(0, _PHASE3)

from train import MIN_TRAINING_ROWS  # noqa: E402

try:
    from sample_training_data import SAMPLE_100_PATIENTS  # noqa: E402
except Exception:
    SAMPLE_100_PATIENTS = []


class PatientFeatures(BaseModel):
    """Las 11 variables predictoras que espera el modelo."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Age": 40, "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0, "MaxHR": 170,
                "Oldpeak": 0.0, "Sex": "M", "ChestPainType": "ATA", "RestingECG": "Normal",
                "ExerciseAngina": "N", "ST_Slope": "Up",
            }
        },
    )

    Age: int = Field(ge=1, le=120)
    RestingBP: float = Field(ge=0, le=300)       # 0 = no informado
    Cholesterol: float = Field(ge=0, le=1000)    # 0 = no informado
    FastingBS: int = Field(ge=0, le=1)
    MaxHR: int = Field(ge=0, le=250)
    Oldpeak: float = Field(ge=-10, le=20)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingECG: Literal["Normal", "ST", "LVH"]
    ExerciseAngina: Literal["Y", "N"]
    ST_Slope: Literal["Up", "Flat", "Down"]


class TrainingRecord(PatientFeatures):
    HeartDisease: int = Field(ge=0, le=1)


class TrainRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "version": "v4",
                "data": SAMPLE_100_PATIENTS,
            }
        },
    )

    version: str | None = Field(default=None, pattern=r"^v[0-9]+$")
    data: list[TrainingRecord] = Field(min_length=MIN_TRAINING_ROWS)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "model_loaded": True,
                "model_version": "v3",
                "error": None,
            }
        }
    )

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    error: str | None = None

