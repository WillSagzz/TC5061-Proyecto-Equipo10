# Modelos y métricas

Para evaluar la viabilidad técnica, se establece una estrategia de modelado incremental para cada pregunta de negocio. En primer lugar, se define un modelo base (*baseline*) de baja complejidad que se ejecuta y registra en MLflow Tracking como punto de referencia mínimo. Posteriormente, se proponen modelos candidatos con mayor capacidad para capturar relaciones no lineales en los datos. Finalmente, se seleccionan métricas de evaluación de la librería scikit-learn (Pedregosa et al., 2011) alineadas tanto con la estructura técnica de cada problema como con el impacto operativo de las decisiones en MiBici. Los modelos base y candidatos provienen de las guías de aprendizaje supervisado y de *clustering* de la misma librería (scikit-learn, s.f.-a, s.f.-c). Siguiendo su guía de evaluación de modelos (scikit-learn, s.f.-b), todas las métricas se calculan sobre el conjunto de prueba cronológico (P1 y P2) o sobre la matriz de perfiles (P3) y se registran en MLflow con el conjunto como prefijo (`test_mae`, `test_recall_vaciarse`, `train_silhouette`).

## P1: Predicción de viajes iniciados por estación y hora (Regresión)

### Modelo baseline

**`DummyRegressor` por estación y hora:** se entrena un `DummyRegressor` con `strategy="mean"` para cada combinación de estación y hora del día. Cada uno aprende el promedio de viajes de su estación a esa hora en el conjunto de entrenamiento y predice ese valor. Por ejemplo, si la estación 35 tuvo en promedio 11.6 viajes de 8:00 a 9:00, predice 11.6 viajes para cualquier día a esa hora. Para combinaciones que no aparecen en entrenamiento, se usa un `DummyRegressor` general con el promedio de todos los viajes. No usa ninguna otra variable, por lo que representa el patrón habitual de cada estación: los modelos candidatos deben superarlo para justificar su complejidad.

### Modelos candidatos

- **Regresión Ridge:** modelo lineal sencillo que estima el número de viajes que iniciarán en cada estación y hora del día siguiente, a partir de los rezagos básicos de demanda (viajes a la misma hora del día anterior y de la semana anterior), el promedio histórico de la estación a esa hora y las variables de calendario. Las variables se estandarizan antes de entrenar. Se elige como primer candidato porque es rápido de entrenar sobre millones de registros, requiere un ajuste mínimo de hiperparámetros y es interpretable. Además, permite comprobar si incorporar la inercia reciente de cada estación mejora el pronóstico frente al *baseline*, que solo usa un promedio estático por estación y hora.
- **`HistGradientBoostingRegressor` (o LightGBM):** algoritmo de árboles de decisión ensamblados de forma secuencial, diseñado específicamente para grandes volúmenes de datos tabulares. Es ideal para P1 porque maneja de forma nativa relaciones no lineales complejas (por ejemplo, el efecto combinado de ser viernes por la tarde en una estación de alta capacidad) y procesa millones de filas en pocos minutos.

### Métricas de evaluación

- **Error absoluto medio (`mean_absolute_error`, `test_mae`):** mide el promedio de la diferencia absoluta entre los viajes reales y los viajes predichos por el modelo. Se expresa exactamente en las mismas unidades del negocio (número de viajes por estación y hora), lo que permite al operador del sistema interpretar directamente cuántos viajes de desviación promedio tiene el pronóstico al dimensionar la flota activa diaria y planear ventanas de mantenimiento. Además, no se distorsiona de forma exagerada por un día atípico (como un evento masivo en la ciudad).
- **Raíz del error cuadrático medio (`root_mean_squared_error`, `test_rmse`):** mide la magnitud del error dando un peso mucho mayor a las desviaciones grandes antes de calcular el promedio. En la operación de MiBici, equivocarse por 1 viaje durante 10 horas seguidas es manejable, pero equivocarse por 15 viajes en una sola hora pico provoca el colapso inmediato de la estación. Esta métrica penaliza precisamente esos errores grandes y concentrados, por lo que complementa al error absoluto medio: si el RMSE crece mucho más que el MAE, el modelo falla en las horas pico, que son las que más importan para la planeación de la flota.

Se descarta el error porcentual (MAPE) porque la mayoría de las combinaciones de estación y hora tienen cero o muy pocos viajes, lo que vuelve indefinido o inestable cualquier error relativo. Ambas métricas se registran para los tres conjuntos (`train_`, `val_` y `test_`), de modo que la diferencia entre entrenamiento y prueba haga visible el sobreajuste o el cambio de la demanda en los meses recientes.

## P2: Riesgo de vaciarse o llenarse por estación e intervalo de 3 horas (Clasificación)

### Modelo baseline

**`DummyClassifier` por estación, intervalo del día y tipo de día:** análogo al *baseline* de P1, se entrena un `DummyClassifier` con `strategy="most_frequent"` para cada combinación de estación, intervalo de 3 horas y tipo de día (laboral o fin de semana). Cada uno predice la clase más frecuente que tuvo su grupo en el conjunto de entrenamiento; por ejemplo, si la estación 25 de 9:00 a 12:00 en días laborales terminó la mayoría de las veces en *riesgo de vaciarse*, predice esa clase. Representa lo que el equipo de rebalanceo puede hacer hoy con el patrón habitual de cada estación, sin usar el comportamiento reciente. Es un *baseline* más exigente que predecir siempre la clase mayoritaria (*sin riesgo*), que tendría exactitud alta pero sensibilidad cero en las clases de riesgo; aun así, hace visible el desbalance de clases, porque en los grupos donde el riesgo no es mayoritario predice *sin riesgo*.

### Modelos candidatos

- **Regresión logística multinomial (`LogisticRegression` con `class_weight="balanced"`):** modelo lineal que estima la probabilidad de cada una de las tres clases a partir de las variables de calendario, la capacidad de la estación y el comportamiento reciente (salidas, llegadas y desbalance de hace 1 hora, 24 horas y 7 días). Se elige porque es rápida, interpretable (los coeficientes indican qué variables empujan a una estación hacia vaciarse o llenarse), entrega probabilidades calibrables para ordenar las estaciones por prioridad y, con el peso balanceado de clases, compensa que los intervalos de riesgo sean mucho menos frecuentes que los normales.
- **`HistGradientBoostingClassifier`:** ensamble de árboles que captura interacciones no lineales (por ejemplo, una estación pequeña, en hora pico de un día laboral, que ya viene perdiendo bicicletas). Maneja de forma nativa millones de filas y el desbalance de clases mediante pesos por muestra. Es el candidato con mayor capacidad esperada y permite medir cuánto se gana frente al modelo lineal.

### Métricas de evaluación

- **Sensibilidad por clase de riesgo (`recall_score`, `test_recall_vaciarse` y `test_recall_llenarse`):** fracción de los intervalos que realmente terminaron en *riesgo de vaciarse* o en *riesgo de llenarse* que el modelo detectó. Es la métrica principal porque el costo de no detectar una estación que se vacía o se llena (un usuario sin bicicleta o sin anclaje) es mucho mayor que el de una falsa alarma (un recorrido innecesario de la camioneta), como se estableció en el ML Canvas. Se reporta por clase, y no promediada con *sin riesgo*, para que la clase mayoritaria no oculte el desempeño en las dos clases que importan al equipo de rebalanceo.
- **Precisión y precisión promedio por clase de riesgo (`precision_score` y `average_precision_score`, `test_precision_vaciarse`, `test_avg_precision_vaciarse` y sus equivalentes para *llenarse*):** la precisión indica qué fracción de las alertas emitidas fue correcta, es decir, cuántos recorridos de camioneta no fueron en vano; la precisión promedio resume la curva de precisión-sensibilidad y permite comparar modelos con independencia del umbral de probabilidad que se elija para despachar una camioneta (un parámetro del ML Canvas).
- **F1 macro (`f1_score` con `average="macro"`, `test_f1_macro`):** media armónica de precisión y sensibilidad promediada por clase, como resumen único para ordenar los *runs* en MLflow. Un modelo que marca todas las estaciones como en riesgo tendría sensibilidad perfecta pero enviaría a las camionetas a todas partes; el F1 castiga ese exceso de falsas alarmas, que agota la capacidad operativa del equipo de rebalanceo.
- **Matriz de confusión (`confusion_matrix`, artefacto):** se registra como figura en cada *run* para distinguir los dos tipos de error que tienen distinto remedio operativo: confundir *vaciarse* con *llenarse* manda la camioneta con la acción contraria, mientras que confundir cualquiera de ellas con *sin riesgo* la deja sin ir.
- **Exactitud:** no se usa como criterio de decisión porque con más del 95 % de los intervalos en la clase *sin riesgo* un modelo trivial la obtiene alta sin detectar ningún riesgo.

## P3: Grupos de estaciones por patrón de uso (Clustering)

### Modelo baseline

**`KMeans` con `k = 4`:** se estandarizan las cinco variables del perfil de cada estación (sección 4, Tabla 4) con `StandardScaler` y se agrupa con K-medias en cuatro grupos, número elegido a priori porque corresponde a las vocaciones que el área de planeación espera encontrar (residencial emisora, receptora de oficinas y escuelas, recreativa y periférica de baja rotación). La semilla fija de `config.py` hace reproducible la inicialización. Este *baseline* establece la cohesión mínima que un candidato debe mejorar y produce el primer mapa de grupos.

### Modelos candidatos

- **`KMeans` con selección de `k`:** se evalúan valores de `k` entre 3 y 8 y se elige el que maximiza el coeficiente de silueta sin sacrificar la interpretabilidad de los grupos. Permite comprobar si cuatro vocaciones son suficientes o si la ciudad presenta perfiles adicionales.
- **`AgglomerativeClustering` (enlace de Ward):** agrupamiento jerárquico que no supone grupos esféricos de tamaño similar, como sí lo hace K-medias, y cuyo dendrograma permite al área de planeación ver cómo se van fusionando las estaciones. Se elige como alternativa estructuralmente distinta para verificar que los grupos encontrados no dependen del algoritmo.

### Métricas de evaluación

Al no existir una etiqueta verdadera, se usan las métricas internas de la guía de *clustering* de scikit-learn (s.f.-a), calculadas sobre la matriz estandarizada de perfiles de estación:

- **Coeficiente de silueta (`silhouette_score`, `train_silhouette`; `train_silhouette_por_k` para cada `k` evaluado):** mide, entre −1 y 1, qué tan parecida es cada estación a las de su propio grupo frente a las del grupo más cercano. Es la métrica principal porque responde directamente a la pregunta de negocio: los grupos deben ser cohesivos y distinguibles entre sí para que planeación urbana pueda asignarles una vocación y actuar sobre ella.
- **Índice de Davies-Bouldin (`davies_bouldin_score`, `train_davies_bouldin`):** promedio de la similitud de cada grupo con su grupo más parecido; valores menores indican grupos más compactos y separados. Se usa como contraste de la silueta porque penaliza de manera distinta los grupos solapados y es más barato de calcular al comparar muchos valores de `k`.
- **Interpretabilidad y mapa (artefactos):** para cada *run* se registran el tamaño de cada grupo, el perfil promedio de las cinco variables por grupo y un mapa de las estaciones coloreadas por grupo usando las coordenadas del catálogo. Un agrupamiento con buena silueta pero sin lectura funcional ni coherencia territorial no sirve para decidir dónde expandir la red.

**Tabla 7.** Resumen de modelos y métricas por pregunta.

| Pregunta | Modelo base | Candidatos | Métricas principales | Decisión que apoyan |
|----------|-------------|------------|----------------------|---------------------|
| P1 (regresión) | `DummyRegressor` (promedio por estación y hora) | Ridge; `HistGradientBoostingRegressor` | `test_mae`, `test_rmse` | Tamaño de la flota activa y ventanas de mantenimiento |
| P2 (clasificación, 3 clases) | `DummyClassifier` (clase más frecuente por estación, intervalo y tipo de día) | Regresión logística balanceada; `HistGradientBoostingClassifier` | `test_recall_vaciarse`, `test_recall_llenarse`, precisión y precisión promedio por clase, `test_f1_macro`, matriz de confusión | Prioridad y ruta de rebalanceo |
| P3 (*clustering*) | `KMeans`, `k = 4` | `KMeans` con selección de `k`; `AgglomerativeClustering` | `train_silhouette`, `train_davies_bouldin`, mapa y perfil por grupo | Zonas de expansión y estaciones a ampliar |
