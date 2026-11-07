# Heart Failure Prediction — Fase 2

Aplicación reutilizable para entrenar y generar predicciones de enfermedad
cardíaca desde la línea de comandos. El proyecto usa el dataset **Heart
Failure Prediction** de 918 registros, con 11 predictores y `HeartDisease`
como objetivo binario.

# Angel Simon Yepes Jaramillo
# Juan Esteban Obando Soto
# Maria Isabel Trejos Largo

## 1. Problema y variables

El dataset **Heart Failure Prediction** contiene 918 registros y 11 variables
clínicas. El objetivo `HeartDisease` es binario: `0` significa ausencia de
enfermedad y `1` significa presencia de enfermedad.

Los predictores son: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`,
`FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak` y `ST_Slope`.

## 2. Exploración y preparación

### Decisiones de datos

Las columnas categóricas son `Sex`, `ChestPainType`, `RestingECG`,
`ExerciseAngina` y `ST_Slope`. `FastingBS=0` es un valor válido y se conserva.
La fuente utiliza `0` como valor no informado para `Cholesterol` en 172 filas
(18,7 %) y para `RestingBP` en una fila (0,11 %). Por eso la aplicación
convierte únicamente esos ceros a `NaN` y realiza imputación por mediana dentro
del pipeline, después de separar entrenamiento y prueba. Así se evita fuga de
información.

El pipeline persistido contiene, en un solo artefacto, la limpieza de ceros,
imputación, codificación one-hot y clasificador XGBoost. Por tanto, `predict.py`
recibe los 11 campos originales y no columnas codificadas manualmente.

## 3. Instalación local

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si PowerShell permite activar el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 4. Pruebas

```powershell
python -m pytest -q
```

## 5. Entrenamiento y versionado

El CSV debe llamarse, por ejemplo, `heart.csv` y contener las 11 variables más
`HeartDisease`.

```powershell
python train.py --data heart.csv --version v3
```

Se generan:

```text
models/v3/model.pkl       # pipeline completo
models/v3/metadata.json   # métricas, split, baseline y matriz de confusión
```

Cada versión se guarda en una carpeta diferente. No se debe sobrescribir una
versión publicada; use `v3`, `v4`, etc. si cambia el tratamiento o el modelo.
El script rechaza por defecto una versión existente; `--force` solo debe usarse
para una corrección local consciente.

La prueba usa `test_size=0.2`, `random_state=42` y estratificación. `metadata.json`
registra accuracy, macro-F1, precisión/recall de la clase 1, matriz de
confusión y la referencia de predecir siempre la clase mayoritaria.

## 6. Predicción

`pacientes.csv` debe contener solamente los 11 campos originales:

```powershell
python predict.py --data pacientes.csv --model models/v3/model.pkl --output predicciones.csv
```

La salida agrega `prediction` y, cuando está disponible, `probability`.

## 7. Docker

```powershell
docker build -t heart-failure-model .
docker run --rm `
  -v "${PWD}:/app" `
  heart-failure-model `
  --data pacientes.csv `
  --model models/v3/model.pkl `
  --output predicciones-docker.csv
```

El smoke test de Docker automatiza la construcción, ejecución y validación de
las columnas de salida:

```powershell
powershell -ExecutionPolicy Bypass `
  -File .\scripts\docker_smoke_test.ps1
```

Para entrenar dentro del contenedor:

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  --entrypoint python `
  heart-failure-model train.py --data heart.csv --version v3
```

## 8. Resultados y diferencias frente a la Fase 1

En la evaluación corregida de Fase 2 se obtuvieron estos resultados:

| Modelo | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline mayoritario | 55,43 % | 35,66 % |
| Random Forest | 85,87 % | 85,70 % |
| XGBoost | 86,41 % | 86,23 % |

La matriz de confusión de XGBoost fue:

```text
[[69, 13],
 [12, 90]]
```

Es decir: TN=69, FP=13, FN=12 y TP=90.

La comparación debe incluir una referencia simple: en la partición original,
la clase mayoritaria acertó 102 de 184 casos (55,4 %). El resultado reportado
para XGBoost fue 163/184 (88,6 %): 91 de 102 positivos detectados, 11 falsos
negativos, y 10 falsos positivos entre 82 negativos. Estos números describen
esa partición y no garantizan un bajo número de falsos negativos en nuevos
datos.

Random Forest y XGBoost deben entrenarse sobre los mismos pliegues estratificados
y compararse con una métrica definida antes de mirar la prueba (por ejemplo,
macro-F1). Si la elección original usó el conjunto de prueba, debe describirse
como exploratoria.
