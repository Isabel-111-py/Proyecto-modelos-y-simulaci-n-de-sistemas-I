# Heart Failure Prediction — Fase 3

API REST que expone el modelo de la Fase 2 para consultar predicciones y
entrenar nuevas versiones mediante solicitudes HTTP. La API utiliza FastAPI,
reutiliza el pipeline completo de la Fase 2 y conserva las versiones anteriores.

# Angel Simon Yepes Jaramillo
# Juan Esteban Obando Soto
# Maria Isabel Trejos Largo

## 1. Objetivo y estructura

La API trabaja con los 11 campos originales del dataset:

`Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`,
`RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak` y `ST_Slope`.

No se envía `HeartDisease` en una predicción, porque es la variable objetivo.
El preprocesamiento, la imputación y la codificación los realiza el pipeline
guardado en `models/`.

## 2. Instalación local

Desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest fase-3
```

Las pruebas verifican `/health`, una predicción válida, validaciones `422`,
errores del modelo, entrenamiento, conflictos de versión y el uso del gancho
de monitoreo de la Fase 4.

## 4. Ejecutar la API

Sin Docker:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir fase-3 --reload
```

Documentación interactiva:

```text
http://127.0.0.1:8000/docs
```

## 5. Operaciones disponibles

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Informa si la API y el modelo activo están disponibles. |
| POST | `/predict` | Recibe un paciente y devuelve predicción, probabilidad y versión. |
| POST | `/train` | Entrena una versión nueva y decide si se activa. |

### `GET /health`

Respuesta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v4",
  "error": null
}
```

### `POST /predict`

Ejemplo de entrada:

```json
{
  "Age": 40,
  "RestingBP": 120,
  "Cholesterol": 200,
  "FastingBS": 0,
  "MaxHR": 170,
  "Oldpeak": 0.0,
  "Sex": "M",
  "ChestPainType": "ATA",
  "RestingECG": "Normal",
  "ExerciseAngina": "N",
  "ST_Slope": "Up"
}
```

Respuesta:

```json
{
  "prediction": 0,
  "probability": 0.0242,
  "model_version": "v4",
  "latency_ms": 24.4
}
```

### `POST /train`

El cliente puede entrenar con el CSV completo:

```powershell
.\.venv\Scripts\python.exe fase-3\client.py --train data\heart.csv
```

La API guarda el artefacto en `models/vN/`, conserva las versiones anteriores
y activa la nueva solamente si cumple la regla de evaluación configurada.

## 6. Validación y errores

- `200`: solicitud válida.
- `422`: campo faltante, tipo incorrecto, categoría inválida o dato fuera de rango.
- `409`: versión existente o entrenamiento concurrente.
- `500`: error interno de predicción o entrenamiento.
- `503`: el modelo activo no está disponible.

## 7. Cliente de prueba

Con la API encendida:

```powershell
.\.venv\Scripts\python.exe fase-3\client.py
.\.venv\Scripts\python.exe fase-3\client.py --input paciente.json
```

El cliente prueba `/health`, una predicción válida, una entrada inválida y
`/metrics` cuando la API de Fase 4 está activa.

## 8. Docker

La API final se ejecuta junto con el monitoreo mediante Compose desde la raíz:

```powershell
docker compose up --build -d
docker compose ps
```

Espere a que el contenedor aparezca como `healthy`. La API queda disponible en:

```text
http://localhost:8000
http://localhost:8000/docs
```

Para detenerla:

```powershell
docker compose down
```
