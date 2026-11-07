import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from predict import predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "v3" / "model.pkl"


def raw_patient_data():
    return pd.DataFrame(
        {
            "Age": [40, 60],
            "Sex": ["M", "F"],
            "ChestPainType": ["ATA", "ASY"],
            "RestingBP": [120, 140],
            "Cholesterol": [200, 250],
            "FastingBS": [0, 1],
            "RestingECG": ["Normal", "LVH"],
            "MaxHR": [170, 120],
            "ExerciseAngina": ["N", "Y"],
            "Oldpeak": [0.0, 2.0],
            "ST_Slope": ["Up", "Flat"],
        }
    )


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="El modelo v3 no está disponible"
)
def test_model_accepts_original_11_fields():
    result = predict(raw_patient_data(), MODEL_PATH)

    assert len(result) == 2
    assert "prediction" in result.columns
    assert "probability" in result.columns
    assert result["prediction"].isin([0, 1]).all()
    assert result["probability"].between(0, 1).all()


def test_predict_help_runs():
    result = subprocess.run(
        [sys.executable, "predict.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Genera predicciones" in result.stdout