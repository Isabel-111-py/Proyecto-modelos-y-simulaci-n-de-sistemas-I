from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "HeartDisease"
CATEGORICAL_COLUMNS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC_COLUMNS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def validate_input(data: pd.DataFrame, include_target: bool = False) -> None:
    required = FEATURE_COLUMNS + ([TARGET_COLUMN] if include_target else [])
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas de entrada: {', '.join(missing)}")


def replace_invalid_zeros(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in ("Cholesterol", "RestingBP"):
        result[column] = result[column].replace(0, np.nan)
    return result


def build_pipeline(classifier) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    preprocessor = ColumnTransformer(
        [("numeric", numeric, NUMERIC_COLUMNS), ("categorical", categorical, CATEGORICAL_COLUMNS)]
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def prepare_features(data: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Legacy helper for reading the original Phase 1 artifact."""
    validate_input(data)
    encoded = pd.get_dummies(replace_invalid_zeros(data), columns=CATEGORICAL_COLUMNS, drop_first=True)
    if feature_columns is not None:
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded
