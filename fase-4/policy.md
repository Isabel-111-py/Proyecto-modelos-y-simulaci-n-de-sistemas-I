# Política de monitoreo y reentrenamiento

## 1. Registro

Cada predicción exitosa se registra en `fase-4/logs/predictions.csv` (fecha UTC, versión, clase, probabilidad, latencia y entrada).
El archivo contiene datos de pacientes: no se sube al repositorio.

## 2. Revisión

Se genera y revisa el reporte (`python fase-4/monitor.py --reference data/heart.csv`) una vez por semana, antes de cada entrega
y cada vez que se acumulen 100 predicciones nuevas. Se revisan:

- Total de predicciones y proporción de clase 1 (un cambio brusco puede indicar un problema en los datos de entrada).
- Uso por versión del modelo y latencia (media y p95).
- Estado de drift por variable.

Estas métricas describen el uso técnico y la distribución de las entradas. No son accuracy, precision ni recall en producción,
porque la etiqueta real de cada paciente no está disponible automáticamente.

## 3. Cuándo evaluar un reentrenamiento

| Criterio | Umbral | Cómo se detecta |
|---|---|---|
| Drift en las entradas | Media numérica desplazada más de 0.5 desviaciones estándar, o una categoría con cambio mayor a 20 puntos, con al menos 30 predicciones | Automático (reporte) |
| Datos nuevos etiquetados | Al menos 100 registros nuevos con la etiqueta real `HeartDisease` verificada | Manual |
| Deterioro del desempeño | Caída de 5 puntos porcentuales o más en accuracy o macro-F1 al evaluar con datos etiquetados recientes frente a la evaluación de referencia (v3: accuracy 0.8641, macro-F1 0.8623) | Manual |

Si se cumple cualquiera, el equipo decide si reentrenar. Un drift detectado solo recomienda evaluar: no activa nada por sí solo.

## 4. Procedimiento de reentrenamiento

1. Reunir el conjunto completo: datos originales más los nuevos registros con etiqueta verificada.
2. Entrenar una versión nueva sin sobrescribir las anteriores:
   `python fase-2/train.py --data <datos.csv> --version vN` (o `POST /train` con ese conjunto).
3. Revisar `models/vN/metadata.json` (métricas, matriz de confusión, tamaño de entrenamiento).
4. Activar solo si cumple la regla de la sección 5.

## 5. Regla de activación

`POST /train` activa la versión nueva solo si **accuracy y macro-F1 no caen más de 5 puntos porcentuales** frente a la versión activa.
Si no cumple, o no existen metadatos comparables, se conserva la versión activa y la respuesta explica el motivo.
Con `train.py` la activación es manual (`--activate`), aplicando el mismo criterio.

## 6. Versionamiento y reversión

- Cada versión vive en `models/vN/` con su `model.pkl` y `metadata.json`; ninguna se elimina durante el reentrenamiento.
- La versión activa está en `models/current_model.txt`.
- Reversión: `python fase-2/model_registry.py use v3` y reiniciar la API (con Docker: `docker compose restart`).
- Tras activar o revertir, generar un reporte nuevo para dejar constancia de la versión en uso.
