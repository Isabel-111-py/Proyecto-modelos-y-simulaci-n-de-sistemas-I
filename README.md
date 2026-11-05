# Datset: Heart Failure Prediction

Durante la fase exploratoria inicial, se evaluó un dataset previo llamado academic performance, sin embargo, a la hora de aplicar diversos modelos como: Regresión logistica, random forest y XGBoost, sin embargo en todas alcanzaba un techo de varianza irreducible de un 60% de precisión. 
De modo que llegamos a la conclusión de que el data set "academic performance" podría tener datos sesgados que le impiden al modelo aprender patrones suficiente para realizar una predicción precisa. 

Tras llegar a esta conclusión, decidimos seleccionar el Heart Failure Prediction Dataset, un conjunto de datos clínico con variables fisiológicas directas.
Aunque, bien podriamos permanecer con el data "academic performance" y limitarnos a explicar porque falló en los tres intentos con diversos modelos, creemos que la idea de realizar este proyecto es poner a prueba un modelo que funcione, esa seria otra de las razones por la cuales cambiamos el data set.

El dataset consta de 918 registros y 11 variables clínicas principales:

1. Age (Numérica): Edad del paciente.
2. Sex (Categórica): Género del paciente (M/F).
3. hestPainType (Categórica): Tipo de dolor torácico (TA, ATA, NAP, ASY).
4. RestingBP (Numérica): Presión arterial en reposo medida en mm Hg.
5. Cholesterol (Numérica): Colesterol sérico medido en mm/dl.
6. FastingBS (Binaria): Glucemia en ayunas (> 120 mg/dl).
7. RestingECG (Categórica): Resultados del electrocardiograma en reposo.
8. MaxHR (Numérica): Frecuencia cardíaca máxima alcanzada.
9. ExerciseAngina (Categórica): Angina inducida por ejercicio (Y/N).
10. Oldpeak (Numérica): Depresión del segmento ST provocada por el ejercicio.
11. ST_Slope (Categórica): Pendiente del segmento ST en esfuerzo.
12. Variable: HeartDisease (Binaria / Objetivo): Presencia de enfermedad cardíaca (1: Enfermedad, 0: Normal).

Metodología y Preprocesamiento:

1. Verificación de datos nulos: se aplico `df.isnull().sum()` para contabilizar la cantidad de celdas nulas, es decir celdas que hacen parte de un registro y que no tienen un valor registrado. Afortunadamente, el data set no tenia este tipo de valores. 
2. Codificación Categórica: Se aplicó`pd.get_dummies` sobre las características cualitativas (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`).
3. División de Datos: 80/20 para entrenamiento y prueba.
4. Modelado: XGBoost (Nota: también probamos random forest, sin embargo, decidimos quedarnos con XGBoost, porque era mas preciso que el random forest)

# Resultados del Modelado

Como fue mencionado anteriormente, experimentamos con dos modelos: Random forest y XGBoost ammbos modelos mostraron un resultado bastante preciso: 

* Modelo: Random Forest
Accuracy: 0.85
F1-Score (Clase 0): 0.83
F1-Score (Clase 1):0.87
Macro F1: 0.85

* Modelo: XGBoost
Accuracy: 0.89
F1-Score (Clase 0): 0.87
F1-Score (Clase 1):0.90
Macro F1:0.88


## Modelo eligido: XGBoost
* Accuracy Global: 89%
* Sensibilidad / Recall (Clase 1):89%
* Precisión (Clase 1): 90%

Resultado: El modelo demostró un equilibrio sólido entre precisión y sensibilidad, garantizando un bajo nivel de falsos negativos en la detección de patologías cardíacas.
