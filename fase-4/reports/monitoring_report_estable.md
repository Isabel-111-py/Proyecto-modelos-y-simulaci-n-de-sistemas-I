# Reporte de monitoreo

- Generado: 2026-11-21T23:45:28.580504+00:00
- Modelo activo: **v3**

## Uso de la API

| Métrica | Valor |
|---|---|
| Predicciones registradas | 60 |
| Clase 0 (normal) | 19 |
| Clase 1 (enfermedad) | 41 |
| Proporción de clase 1 | 0.6833 |
| Probabilidad media de clase 1 | 0.6216 |
| Latencia media (ms) | 75.5384 |
| Latencia p95 (ms) | 144.32 |
| Primera predicción | 2026-11-21T23:45:22.133422+00:00 |
| Última predicción | 2026-11-21T23:45:28.464998+00:00 |

Uso por versión: {"v3": 60}

## Cambio en la distribución de las entradas (drift)

- Referencia: CSV: C:\Modelos_I\modelosentrega3\Proyecto-corregido-v3\Proyecto-modelos-y-simulaci-n-de-sistemas-I\data\heart.csv
- Estado: **estable** (registros evaluados: 60)

| Variable | Tipo | Cambio | Drift |
|---|---|---|---|
| Age | numeric | 0.0766 desv. estándar | no |
| RestingBP | numeric | 0.0644 desv. estándar | no |
| Cholesterol | numeric | 0.0843 desv. estándar | no |
| FastingBS | numeric | 0.0783 desv. estándar | no |
| MaxHR | numeric | 0.0206 desv. estándar | no |
| Oldpeak | numeric | 0.0634 desv. estándar | no |
| Sex | categorical | 0.0269 (categoría F) | no |
| ChestPainType | categorical | 0.0264 (categoría ASY) | no |
| RestingECG | categorical | 0.168 (categoría Normal) | no |
| ExerciseAngina | categorical | 0.0041 (categoría N) | no |
| ST_Slope | categorical | 0.0303 (categoría Up) | no |

## Recomendación de reentrenamiento

- Ningún criterio automático se cumple.
- Criterios manuales (requieren etiquetas reales): Acumular al menos 100 registros nuevos con etiqueta real verificada. Caída de 5 puntos porcentuales o más en accuracy o macro-F1 con datos etiquetados recientes.

## Versiones del modelo

| Versión | Activa | Accuracy | Macro-F1 | Entrenada |
|---|---|---|---|---|
| v3 | sí | 0.8641 | 0.8623 | n/d |

## Notas

- Las métricas describen el uso y el tiempo de respuesta de la API.
- Sin etiquetas reales posteriores no es posible calcular accuracy, precision o recall en producción.
- Reporte generado con tráfico sintético (scripts/simulate_traffic.py): las entradas son valores al azar en rangos plausibles, no pacientes reales. Las predicciones sí provienen del modelo activo.
