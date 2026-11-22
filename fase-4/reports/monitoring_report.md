# Reporte de monitoreo

- Generado: 2026-11-22T04:25:12.534834+00:00
- Modelo activo: **v4**

## Uso de la API

| Métrica | Valor |
|---|---|
| Solicitudes totales | 6 |
| Solicitudes fallidas (4xx/5xx) | 3 |
| Tasa de fallos | 0.5 |
| Predicciones registradas | 3 |
| Clase 0 (normal) | 3 |
| Clase 1 (enfermedad) | 0 |
| Proporción de clase 1 | 0.0 |
| Probabilidad media de clase 1 | 0.0242 |
| Latencia media (ms) | 20.3494 |
| Latencia p95 (ms) | 50.1614 |
| Primera predicción | 2026-11-22T04:09:43.353154+00:00 |
| Última predicción | 2026-11-22T04:24:41.075112+00:00 |

Uso por versión: {"v4": 3, "": 3}

## Cambio en la distribución de las entradas (drift)

- Referencia: CSV: data\heart.csv
- Estado: **datos_insuficientes** (registros evaluados: 6)
- Se necesitan al menos 30 predicciones para evaluar drift.

## Recomendación de reentrenamiento

- Ningún criterio automático se cumple.
- Criterios manuales (requieren etiquetas reales): Acumular al menos 100 registros nuevos con etiqueta real verificada. Caída de 5 puntos porcentuales o más en accuracy o macro-F1 con datos etiquetados recientes.

## Versiones del modelo

| Versión | Activa | Accuracy | Macro-F1 | Entrenada |
|---|---|---|---|---|
| v3 | no | 0.8641 | 0.8623 | n/d |
| v4 | sí | 0.8641 | 0.8623 | 2026-11-22T03:49:24.297994+00:00 |

## Notas

- Las métricas describen el uso y el tiempo de respuesta de la API.
- Sin etiquetas reales posteriores no es posible calcular accuracy, precision o recall en producción.
