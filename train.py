"""Train and version the Heart Failure XGBoost pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from preprocessing import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    build_pipeline,
    replace_invalid_zeros,
    validate_input,
)


def train_model(
    data: pd.DataFrame,
    version: str = "v1",
    output_dir: str | Path = "models",
    force: bool = False,
) -> Path:
    """Fit the complete pipeline and save it with evaluation metadata."""
    validate_input(data, include_target=True)
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("La columna HeartDisease contiene valores nulos.")

    X = replace_invalid_zeros(data[FEATURE_COLUMNS])
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost no está instalado. Ejecute: pip install -r requirements.txt") from exc

    classifier = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
    )
    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    majority_accuracy = float(y_test.value_counts().max() / len(y_test))
    version_dir = Path(output_dir) / version
    version_dir.mkdir(parents=True, exist_ok=True)
    artifact = version_dir / "model.pkl"
    if artifact.exists() and not force:
        raise FileExistsError(
            f"La versión '{version}' ya existe. Use otra versión o --force si desea reemplazarla."
        )
    joblib.dump({"pipeline": pipeline, "version": version, "target_column": TARGET_COLUMN}, artifact)

    metadata = {
        "version": version,
        "target_column": TARGET_COLUMN,
        "input_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "zero_values_treated_as_missing": ["Cholesterol", "RestingBP"],
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "test_rows": len(y_test),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
            "precision_class_1": float(precision_score(y_test, predictions, zero_division=0)),
            "recall_class_1": float(recall_score(y_test, predictions, zero_division=0)),
            "majority_baseline_accuracy": majority_accuracy,
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena el pipeline de enfermedad cardíaca.")
    parser.add_argument("--data", required=True, help="Ruta al CSV de entrenamiento")
    parser.add_argument("--version", default="v1", help="Versión del modelo, por ejemplo v1 o v2")
    parser.add_argument("--output-dir", default="models", help="Directorio de versiones")
    parser.add_argument("--force", action="store_true", help="Permite reemplazar una versión existente")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = train_model(pd.read_csv(args.data), args.version, args.output_dir, args.force)
    print(f"Modelo guardado en: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
