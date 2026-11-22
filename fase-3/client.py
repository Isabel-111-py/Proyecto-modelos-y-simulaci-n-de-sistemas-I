"""Cliente de prueba de la API (solo biblioteca estándar).

Ejemplos:
    python fase-3/client.py
    python fase-3/client.py --input paciente.json
    python fase-3/client.py --train data/heart.csv     # entrena una versión nueva (cambia el estado)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SAMPLE_PATIENT = {
    "Age": 40, "RestingBP": 120, "Cholesterol": 200, "FastingBS": 0, "MaxHR": 170,
    "Oldpeak": 0.0, "Sex": "M", "ChestPainType": "ATA", "RestingECG": "Normal",
    "ExerciseAngina": "N", "ST_Slope": "Up",
}
INT_COLUMNS = {"Age", "FastingBS", "MaxHR", "HeartDisease"}
FLOAT_COLUMNS = {"RestingBP", "Cholesterol", "Oldpeak"}


def call_api(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 30) -> tuple[int, dict]:
    """Devuelve (código HTTP, cuerpo JSON). No lanza excepción con códigos 4xx/5xx."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body}
    except URLError as exc:
        raise RuntimeError(f"No fue posible conectar con la API: {exc.reason}") from exc


def load_training_rows(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row = {}
            for key, value in raw.items():
                if key in INT_COLUMNS:
                    row[key] = int(value)
                elif key in FLOAT_COLUMNS:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def show(title: str, status: int, body: dict) -> None:
    print(f"\n{title} -> HTTP {status}")
    print(json.dumps(body, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cliente de prueba de Heart Disease Prediction API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", help="Archivo JSON con los 11 campos de un paciente")
    parser.add_argument("--train", metavar="CSV", help="CSV etiquetado para probar POST /train")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    failures = 0

    status, body = call_api(f"{base}/health")
    show("1) GET /health", status, body)
    failures += status != 200

    patient = SAMPLE_PATIENT
    if args.input:
        # utf-8-sig acepta JSON creado por PowerShell con o sin BOM.
        with open(args.input, encoding="utf-8-sig") as file:
            patient = json.load(file)
    status, body = call_api(f"{base}/predict", "POST", patient)
    show("2) POST /predict (entrada válida)", status, body)
    failures += status != 200

    status, body = call_api(f"{base}/predict", "POST", {**patient, "Sex": "desconocido"})
    show("3) POST /predict (entrada inválida, se espera 422)", status, body)
    failures += status != 422

    status, body = call_api(f"{base}/metrics")
    if status == 404:
        print("\n4) GET /metrics -> no disponible (esta API no incluye monitoreo; es la Fase 3)")
    else:
        show("4) GET /metrics", status, body)

    if args.train:
        status, body = call_api(f"{base}/train", "POST", {"data": load_training_rows(args.train)}, timeout=300)
        show("5) POST /train", status, body)
        failures += status != 200

    print("\nResultado:", "OK" if not failures else f"{failures} comprobación(es) fallida(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
