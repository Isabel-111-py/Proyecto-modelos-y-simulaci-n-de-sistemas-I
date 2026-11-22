"""Envía pacientes sintéticos a POST /predict para generar registros de monitoreo.

Los valores se generan al azar dentro de rangos plausibles; NO son datos reales.

Ejemplos:
    python scripts/simulate_traffic.py --n 60
    python scripts/simulate_traffic.py --n 60 --drifted     # entradas desplazadas para provocar drift
"""

from __future__ import annotations

import argparse
import json
import random
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_patient(rng: random.Random, drifted: bool) -> dict:
    if drifted:  # población más mayor, con peor capacidad de esfuerzo y más colesterol
        age = clip(rng.gauss(68, 6), 50, 90)
        max_hr = clip(rng.gauss(110, 15), 70, 160)
        cholesterol = clip(rng.gauss(330, 40), 200, 500)
        chest_pain = rng.choices(["ASY", "NAP", "ATA", "TA"], [85, 8, 4, 3])[0]
        slope = rng.choices(["Flat", "Up", "Down"], [85, 10, 5])[0]
    else:
        age = clip(rng.gauss(53, 9), 28, 77)
        max_hr = clip(rng.gauss(137, 25), 70, 200)
        cholesterol = 0 if rng.random() < 0.15 else clip(rng.gauss(240, 55), 120, 500)
        chest_pain = rng.choices(["ASY", "NAP", "ATA", "TA"], [54, 22, 19, 5])[0]
        slope = rng.choices(["Flat", "Up", "Down"], [50, 43, 7])[0]
    return {
        "Age": int(age),
        "RestingBP": round(clip(rng.gauss(132, 17), 90, 200)),
        "Cholesterol": round(cholesterol),
        "FastingBS": int(rng.random() < 0.23),
        "MaxHR": int(max_hr),
        "Oldpeak": round(max(0.0, rng.gauss(0.9, 1.0)), 1),
        "Sex": rng.choices(["M", "F"], [79, 21])[0],
        "ChestPainType": chest_pain,
        "RestingECG": rng.choices(["Normal", "LVH", "ST"], [60, 20, 20])[0],
        "ExerciseAngina": rng.choices(["N", "Y"], [60, 40])[0],
        "ST_Slope": slope,
    }


def post(url: str, payload: dict) -> int:
    request = Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        return response.status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=60, help="Cantidad de peticiones")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drifted", action="store_true", help="Genera entradas con una distribución distinta")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    url = args.base_url.rstrip("/") + "/predict"
    sent = 0
    try:
        for _ in range(args.n):
            if post(url, make_patient(rng, args.drifted)) == 200:
                sent += 1
    except HTTPError as exc:
        print(f"La API respondió HTTP {exc.code}: {exc.read().decode(errors='replace')}")
        return 1
    except URLError as exc:
        print(f"No fue posible conectar con la API: {exc.reason}")
        return 1
    print(f"{sent}/{args.n} predicciones registradas. Genere el reporte con: python fase-4/monitor.py --reference data/heart.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
