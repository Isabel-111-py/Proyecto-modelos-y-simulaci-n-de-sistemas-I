"""API de la Fase 4: la API de la Fase 3 más registro de predicciones y GET /metrics.

Ejecución local (desde la raíz del repositorio):
    uvicorn monitored_api:app --app-dir fase-4 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("fase-2", "fase-3", "fase-4"):
    _path = str(ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from main import create_app  # noqa: E402
from monitoring import build_summary, record_prediction, record_request  # noqa: E402

app = create_app(
    on_prediction=record_prediction,
    on_request=record_request,
    metrics_provider=build_summary,
    version="1.1.0",
)
