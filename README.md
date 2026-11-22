# Proyecto integrador: predicción de enfermedad cardíaca

Modelos y Simulación de Sistemas I. Prototipo reproducible que lleva un modelo de Machine Learning desde un
notebook hasta una API REST contenerizada con monitoreo básico.

> Prototipo académico. No es un dispositivo médico ni debe usarse para decisiones clínicas.


# Maria Isabel Trejos Largo
# Juan Esteban Obando Soto
# Angel Simon Yepes Jaramillo



## Problema y datos

- **Dataset:** [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (Kaggle). El docente autorizó su uso para el proyecto.
- **Tarea:** clasificación binaria. Variable objetivo `HeartDisease` (1 = enfermedad cardíaca, 0 = normal).
- **Tamaño:** 918 registros y 11 variables predictoras (6 numéricas y 5 categóricas). No es una serie de tiempo.
- **Valores faltantes:** el dataset no trae celdas nulas, pero `Cholesterol` tiene 172 ceros y `RestingBP` tiene 1, que no son
  mediciones válidas. Se convierten en `NaN` y se imputan con la mediana dentro del `Pipeline` (solo con datos de entrenamiento).
- **Datos:** ver [data/README.md](data/README.md) (el CSV se descarga con `scripts/download_data.py`).

## Cumplimiento de los requisitos del dataset

Comparación con la sección 4 de la guía del proyecto (cifras calculadas sobre `heart.csv`):

| Requisito de la guía | Estado | Detalle |
|---|---|---|
| Competición/dataset de Kaggle | Cumple | Heart Failure Prediction (Kaggle) |
| Variable objetivo claramente definida | Cumple | `HeartDisease` (0/1) |
| Clasificación o regresión | Cumple | Clasificación binaria |
| Al menos 5 variables predictoras | Cumple | 11 (6 numéricas, 5 categóricas) |
| No es una serie de tiempo | Cumple | Cada fila es un paciente independiente |
| Se procesa en un computador personal | Cumple | 918 filas; el entrenamiento tarda segundos |
| Al menos 1.000 observaciones | **No cumple (918)** | Excepción, ver nota |
| Nulos del 0.1 % al 2 % en alguna predictora | **Cumple solo parcialmente** | No hay celdas vacías. Al tratar los ceros inválidos como faltantes, `RestingBP` tiene 1 de 918 (0.11 %, dentro del rango) y `Cholesterol` tiene 172 de 918 (18.7 %, fuera del rango). Ver nota |

**Nota sobre las excepciones.** El dataset original del equipo (rendimiento académico) se descartó porque ningún modelo superó
aproximadamente 60 % de accuracy (ver [fase-1](fase-1/README.md)). Se cambió a Heart Failure Prediction con autorización del
docente. Las dos diferencias frente a la guía (918 observaciones en lugar de 1.000 y la forma de los valores faltantes) se
declaran aquí de forma explícita. Evidencia de la autorización: _[fecha y asunto del correo del docente — completar]_.

## Estructura del repositorio

Cada fase conserva y amplía el trabajo de la anterior: la Fase 3 importa el código de la Fase 2 y la Fase 4 amplía la API de la Fase 3.

```
fase-1/   Modelo_Fase_1.ipynb, modelo_xgboost.pkl, README.md          Modelo predictivo
fase-2/   preprocessing.py, train.py, predict.py, model_registry.py,   Scripts y Docker
          Dockerfile, Modelo_Fase_2.ipynb, tests/, README.md
fase-3/   main.py, schemas.py, client.py, Dockerfile, tests/, README.md   API REST
fase-4/   monitoring.py, monitor.py, monitored_api.py, policy.md,        Monitoreo
          Dockerfile, reports/, logs/, tests/, README.md
models/   v3/ (model.pkl + metadata.json) y current_model.txt (versión activa)
data/     sample_patients.csv (y heart.csv, ver data/README.md)
scripts/  download_data.py, simulate_traffic.py, generate_report.py, docker_smoke_test.{sh,ps1}
compose.yaml, requirements.txt, pytest.ini
```

## Inicio rápido

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-notebooks.txt                   # solo para los notebooks y la descarga de datos
python scripts/download_data.py                             # crea data/heart.csv
python -m pytest                                            # pruebas de las fases 2, 3 y 4
```

| Quiero... | Comando | Detalle |
|---|---|---|
| Ver la exploración y el modelo | abrir `fase-1/Modelo_Fase_1.ipynb` y `fase-2/Modelo_Fase_2.ipynb` | [fase-1](fase-1/README.md), [fase-2](fase-2/README.md) |
| Entrenar una versión nueva | `python fase-2/train.py --data data/heart.csv --version v4` | [fase-2](fase-2/README.md) |
| Predecir desde la terminal | `python fase-2/predict.py --data data/sample_patients.csv` | [fase-2](fase-2/README.md) |
| Levantar la API (con monitoreo) | `docker compose up --build -d` y abrir <http://localhost:8000/docs> | [fase-3](fase-3/README.md), [fase-4](fase-4/README.md) |
| Probar la API | `python fase-3/client.py` | [fase-3](fase-3/README.md) |
| Generar el reporte de monitoreo (con tráfico sintético, sin Docker) | `python scripts/generate_report.py` | [fase-4](fase-4/README.md) |
| Generar el reporte con el registro actual | `python fase-4/monitor.py --reference data/heart.csv` | [fase-4](fase-4/README.md) |

## Resultados

El modelo desplegado es **v3**: el pipeline de la Fase 2 (imputación y codificación dentro del `Pipeline`, XGBoost).
Todas las cifras se calculan sobre una única partición estratificada 80/20 (`random_state=42`, 184 pacientes de prueba).

| Modelo | Accuracy | Macro-F1 | Notas |
|---|---|---|---|
| Referencia (siempre la clase mayoritaria) | 0.5543 | 0.3566 | Fase 2 |
| Random Forest (Fase 2) | 0.8587 | 0.8570 | |
| **XGBoost v3 (Fase 2, desplegado)** | **0.8641** | **0.8623** | Precisión clase 1: 0.8738. Recall clase 1: 0.8824. TN=69, FP=13, FN=12, TP=90 |
| XGBoost (Fase 1, sin tratar ceros) | 0.8859 | 0.88 | TN=72, FP=10, FN=11, TP=91. No es el modelo desplegado |

La diferencia entre la Fase 1 y la Fase 2 son 4 aciertos de 184 pacientes. Con una sola partición no es una diferencia
concluyente; se adoptó la Fase 2 porque trata los ceros inválidos y evita ajustar el preprocesamiento con datos de prueba.
El objetivo del curso no es el mejor modelo, sino un proceso reproducible.
