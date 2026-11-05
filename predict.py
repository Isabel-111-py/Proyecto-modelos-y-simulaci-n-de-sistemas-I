from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from preprocessing import prepare_features, validate_input


def load_artifact(path: str | Path) -> tuple[object, bool, list[str] | None]:
    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        return loaded["pipeline"], True, None
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"], False, loaded.get("feature_columns")
    # Compatibility with the original Phase 1 modelo_xgboost.pkl.
    return loaded, False, list(getattr(loaded, "feature_names_in_", [])) or None


def predict(data: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    model, is_pipeline, feature_columns = load_artifact(model_path)
    validate_input(data)
    X = data if is_pipeline else prepare_features(data, feature_columns)
    result = data.copy()
    result["prediction"] = model.predict(X)
    if hasattr(model, "predict_proba"):
        result["probability"] = [row[1] for row in model.predict_proba(X)]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera predicciones de enfermedad cardíaca.")
    parser.add_argument("--data", required=True, help="CSV con los 11 campos originales")
    parser.add_argument("--model", default="models/v1/model.pkl", help="Artefacto del pipeline")
    parser.add_argument("--output", help="CSV de salida; si se omite, imprime el resultado")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = predict(pd.read_csv(args.data), args.model)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Predicciones guardadas en: {args.output}")
    else:
        print(result.to_csv(index=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
