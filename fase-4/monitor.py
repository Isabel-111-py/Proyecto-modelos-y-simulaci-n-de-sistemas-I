"""Script de monitoreo: genera el reporte de la Fase 4 a partir del registro de predicciones.

Ejemplos:
    python fase-4/monitor.py
    python fase-4/monitor.py --reference data/heart.csv   # referencia de drift desde el CSV de entrenamiento
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import monitoring  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera el reporte de monitoreo.")
    parser.add_argument(
        "--reference",
        help="CSV de entrenamiento para calcular la distribución de referencia "
        "(si se omite se usan los metadatos del modelo activo, cuando existen)",
    )
    args = parser.parse_args()
    json_path, md_path = monitoring.generate_report(args.reference)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"Reporte JSON: {json_path}")
    print(f"Reporte Markdown: {md_path}")
    print(json.dumps({"metrics": report["metrics"], "drift_status": report["drift"]["status"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
