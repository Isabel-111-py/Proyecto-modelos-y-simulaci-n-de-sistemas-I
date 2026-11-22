# Heart Failure Prediction — Fase 4

Monitoreo básico de la API de la Fase 3. Registra solicitudes, calcula métricas
operativas, identifica cambios en la distribución de entradas y genera reportes
para apoyar la decisión de reentrenamiento.

# Angel Simon Yepes Jaramillo
# Juan Esteban Obando Soto
# Maria Isabel Trejos Largo

## 1. Objetivo y componentes

La Fase 4 conserva la API de la Fase 3 y agrega:

- Registro persistente de solicitudes a `/predict`.
- Registro de respuestas exitosas y rechazadas.
- Métricas de uso y latencia.
- Detección básica de drift.
- Reporte Markdown y JSON.
- Política de reentrenamiento y versionado.

Los archivos principales son:

```text
monitoring.py       Registro, métricas, drift y reportes
monitored_api.py    API de Fase 3 conectada al monitoreo
monitor.py          Generador de reportes
policy.md           Política de reentrenamiento
tests/              Pruebas del monitoreo
reports/            Reportes generados y muestra del registro
```

## 2. Instalación local

Desde la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest fase-4
```

Para ejecutar las pruebas de las dos fases:

```powershell
.\.venv\Scripts\python.exe -m pytest fase-3 fase-4
```

Resultado verificado:

```text
27 passed
```

## 4. Ejecutar la API monitoreada

La forma recomendada es Docker Compose:

```powershell
docker compose up --build -d
docker compose ps
```

Espere a que `heart-disease-api` aparezca como `healthy` antes de enviar
solicitudes.

La API queda disponible en:

```text
http://localhost:8000
http://localhost:8000/docs
```

Para detenerla:

```powershell
docker compose down
```

## 5. Registro de solicitudes

Cada solicitud a `/predict`, válida o rechazada, se agrega a:

```text
fase-4/logs/predictions.csv
```

El archivo contiene:

```text
timestamp
status_http
model_version
prediction
probability
latency_ms
input_json
```

Las solicitudes `4xx` y `5xx` se registran con la predicción y la versión
vacías cuando el modelo no llegó a utilizarse.

## 6. Métricas operativas

Consultar:

```powershell
Invoke-RestMethod http://localhost:8000/metrics
```

La respuesta incluye:

- Total de solicitudes.
- Solicitudes fallidas.
- Tasa de fallos.
- Total de predicciones exitosas.
- Predicciones por clase.
- Uso por versión del modelo.
- Latencia promedio y p95.

En una ejecución de ejemplo se obtuvieron:

```text
Solicitudes totales: 6
Solicitudes fallidas: 3
Tasa de fallos: 50 %
Predicciones exitosas: 3
Versión utilizada: v4
```

Estas métricas describen el funcionamiento del servicio, no la calidad clínica
del modelo, porque no contienen la etiqueta real de cada paciente.

## 7. Generar el reporte

```powershell
.\.venv\Scripts\python.exe fase-4\monitor.py `
  --reference data\heart.csv
```

Se generan:

```text
fase-4/reports/monitoring_report.md
fase-4/reports/monitoring_report.json
```

Con menos de 30 predicciones el reporte muestra `datos_insuficientes` para
drift. Esto es esperado. Para evaluar drift con tráfico sintético:

```powershell
.\.venv\Scripts\python.exe scripts\simulate_traffic.py --n 60
.\.venv\Scripts\python.exe fase-4\monitor.py --reference data\heart.csv

.\.venv\Scripts\python.exe scripts\simulate_traffic.py --n 60 --drifted
.\.venv\Scripts\python.exe fase-4\monitor.py --reference data\heart.csv
```

El primer reporte debe indicar `estable` y el segundo puede indicar `drift`.

## 8. Versiones y política de reentrenamiento

Las versiones disponibles se consultan con:

```powershell
.\.venv\Scripts\python.exe fase-2\model_registry.py list
```

En la ejecución de entrega se conservaron:

```text
v3: versión anterior
v4: versión activa después de POST /train
```

Para volver a una versión anterior:

```powershell
.\.venv\Scripts\python.exe fase-2\model_registry.py use v3
```

La política del equipo es revisar el modelo cuando se acumulen al menos 100
registros nuevos con etiqueta real o cuando accuracy o macro-F1 disminuyan
cinco puntos porcentuales o más. El equipo compara la candidata y la versión
activa sobre los mismos datos reservados y solo activa la candidata si cumple
el umbral configurado.

## 9. Evidencias

La carpeta `reports/` contiene reportes generados y una muestra del registro:

```text
monitoring_report.md
monitoring_report.json
predictions_sample.csv
```

Los registros completos de ejecución se mantienen en `logs/`, pero no deben
subirse si contienen datos de entrada reales.
