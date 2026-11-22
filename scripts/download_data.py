"""Descarga el dataset Heart Failure Prediction de Kaggle y lo guarda en data/heart.csv.

Requiere:  pip install kagglehub
"""

import shutil
from pathlib import Path

import kagglehub

destination = Path(__file__).resolve().parents[1] / "data" / "heart.csv"
folder = Path(kagglehub.dataset_download("fedesoriano/heart-failure-prediction"))
source = next(folder.glob("*.csv"))
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(source, destination)
print(f"Dataset guardado en: {destination}")
