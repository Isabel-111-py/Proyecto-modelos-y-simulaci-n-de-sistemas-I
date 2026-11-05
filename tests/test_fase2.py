import joblib
import pandas as pd

from predict import predict
from preprocessing import replace_invalid_zeros, validate_input


class DummyPipeline:
    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return [[0.8, 0.2] for _ in range(len(X))]


def sample_data():
    return pd.DataFrame(
        {
            "Age": [40, 60], "Sex": ["M", "F"], "ChestPainType": ["ATA", "ASY"],
            "RestingBP": [120, 140], "Cholesterol": [200, 250], "FastingBS": [0, 1],
            "RestingECG": ["Normal", "LVH"], "MaxHR": [170, 120],
            "ExerciseAngina": ["N", "Y"], "Oldpeak": [0.0, 2.0], "ST_Slope": ["Up", "Flat"],
        }
    )


def test_only_invalid_measurement_zeros_become_missing():
    data = sample_data()
    data.loc[0, "Cholesterol"] = 0
    data.loc[0, "RestingBP"] = 0
    data.loc[0, "FastingBS"] = 0
    cleaned = replace_invalid_zeros(data)
    assert pd.isna(cleaned.loc[0, "Cholesterol"])
    assert pd.isna(cleaned.loc[0, "RestingBP"])
    assert cleaned.loc[0, "FastingBS"] == 0


def test_prediction_uses_original_input_columns(tmp_path):
    path = tmp_path / "model.pkl"
    joblib.dump({"pipeline": DummyPipeline(), "version": "test"}, path)
    result = predict(sample_data(), path)
    assert list(result["prediction"]) == [0, 0]
    assert list(result["probability"]) == [0.2, 0.2]


def test_missing_original_column_is_rejected():
    data = sample_data().drop(columns=["Age"])
    try:
        validate_input(data)
    except ValueError as error:
        assert "Age" in str(error)
    else:
        raise AssertionError("Se esperaba un error por columna faltante")
