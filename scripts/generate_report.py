"""Genera los reportes de monitoreo de la Fase 4 con un solo comando (sin Docker).

Pasos:
  1. Respalda el registro de predicciones anterior (fase-4/logs/predictions.csv).
  2. Envía tráfico SINTÉTICO normal a la API monitoreada (en el mismo proceso, con TestClient)
     y genera el reporte  -> fase-4/reports/monitoring_report_estable.{md,json}
  3. Envía tráfico SINTÉTICO desplazado (drift) y genera el reporte final
     -> fase-4/reports/monitoring_report.{md,json}

Las predicciones las calcula el modelo activo real; lo sintético son las entradas (valores al azar
en rangos plausibles, NO pacientes reales). Los reportes lo indican.

Requiere las dependencias de requirements.txt y data/heart.csv (python scripts/download_data.py),
salvo que el modelo activo ya guarde `reference_stats` en su metadata.json.

Ejemplo (desde la raíz del repositorio):
    python scripts/generate_report.py
    python scripts/generate_report.py --n 80 --reference data/heart.csv
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("fase-2", "fase-3", "fase-4"):
    sys.path.insert(0, str(ROOT / _folder))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import monitoring  # noqa: E402
from simulate_traffic import make_patient  # noqa: E402

SYNTHETIC_NOTE = (
    "Reporte generado con tráfico sintético (scripts/simulate_traffic.py): las entradas son valores al azar "
    "en rangos plausibles, no pacientes reales. Las predicciones sí provienen del modelo activo."
)


def get_client():
    """Cliente en proceso de la API de la Fase 4 (no necesita levantar el servidor)."""
    from fastapi.testclient import TestClient

    import monitored_api

    return TestClient(monitored_api.app)


def send_traffic(client, n: int, drifted: bool, rng: random.Random) -> int:
    ok = 0
    for _ in range(n):
        response = client.post("/predict", json=make_patient(rng, drifted))
        if response.status_code != 200:
            raise RuntimeError(f"POST /predict respondió {response.status_code}: {response.text}")
        ok += 1
    return ok


def resolve_reference(argument: str | None) -> str | None:
    """CSV de referencia si existe; si no, exige que el modelo activo guarde sus estadísticas."""
    candidate = Path(argument) if argument else ROOT / "data" / "heart.csv"
    if candidate.exists():
        return str(candidate)
    reference, _ = monitoring.load_reference(None)
    if reference is not None:
        return None
    raise SystemExit(
        f"No se encontró {candidate} y el modelo activo no incluye 'reference_stats'.\n"
        "Descargue el dataset con:  pip install kagglehub && python scripts/download_data.py"
    )


def finalize_report(suffix: str | None, reference: str | None) -> Path:
    """Genera el reporte actual, le agrega la nota de datos sintéticos y opcionalmente lo copia con otro nombre."""
    json_path, md_path = monitoring.generate_report(reference)

    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["notes"].append(SYNTHETIC_NOTE)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with md_path.open("a", encoding="utf-8") as file:
        file.write(f"- {SYNTHETIC_NOTE}\n")

    if suffix:
        shutil.copy(json_path, json_path.with_name(f"monitoring_report_{suffix}.json"))
        shutil.copy(md_path, md_path.with_name(f"monitoring_report_{suffix}.md"))
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=60, help="Peticiones por etapa (mínimo 30 para evaluar drift)")
    parser.add_argument("--reference", help="CSV de entrenamiento (por defecto data/heart.csv)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true", help="No respaldar ni vaciar el registro existente")
    args = parser.parse_args(argv)
    if args.n < monitoring.MIN_DRIFT_ROWS:
        parser.error(f"--n debe ser al menos {monitoring.MIN_DRIFT_ROWS} para poder evaluar drift")

    reference = resolve_reference(args.reference)
    log = monitoring.PREDICTIONS_FILE
    if not args.append and log.exists() and log.stat().st_size > 0:
        backup = log.with_name("predictions.backup.csv")
        shutil.move(str(log), str(backup))
        print(f"Registro anterior respaldado en: {backup}")

    client = get_client()
    rng = random.Random(args.seed)

    sent = send_traffic(client, args.n, drifted=False, rng=rng)
    print(f"Etapa 1: {sent} predicciones con tráfico normal.")
    stable_md = finalize_report("estable", reference)
    stable = json.loads(stable_md.with_suffix(".json").read_text(encoding="utf-8"))["drift"]["status"]
    print(f"  Drift: {stable}  -> {stable_md.with_name('monitoring_report_estable.md')}")

    sent = send_traffic(client, args.n, drifted=True, rng=rng)
    print(f"Etapa 2: {sent} predicciones con tráfico desplazado.")
    final_md = finalize_report(None, reference)
    final = json.loads(final_md.with_suffix(".json").read_text(encoding="utf-8"))
    print(f"  Drift: {final['drift']['status']}  -> {final_md}")
    if final["drift"]["drifted_features"]:
        print("  Variables con drift:", ", ".join(final["drift"]["drifted_features"]))

    print("\nListo. Suba fase-4/reports/*.md y *.json al repositorio (el registro predictions.csv NO se sube).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
