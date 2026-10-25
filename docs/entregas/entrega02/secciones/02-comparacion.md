# Comparación de experimentos y selección del modelo

<!-- Métricas con 3 decimales, copiadas de MLflow. Mejora vs. baseline sobre el conjunto de prueba. Las cifras del baseline son las de la Entrega 01 (versión de datos 1c3d603c990c); si la versión cambió, actualizarlas. -->

<div class="pendiente">

TODO. Dos líneas: se mantienen la partición temporal y las métricas de la Entrega 01 (o qué cambió y por qué); las corridas se filtran en MLflow con las etiquetas `question`, `model_family` y `stage_of_work`; cada pregunta incluye una búsqueda de hiperparámetros con *nested runs* (una corrida padre por pregunta), de la que se toma la mejor combinación como candidata.

</div>

## P1: viajes por estación y hora

**Tabla 1.** P1: *baseline* y corridas candidatas.

| Corrida | Modelo | Hiperparámetros clave | `val_mae` | `val_rmse` | `test_mae` | `test_rmse` | Duración | Mejora en `test_mae` | Mejora en `test_rmse` |
|---------|--------|-----------------------|-----------|------------|------------|-------------|----------|----------------------|-----------------------|
| `P1-baseline-dummy-estacion-hora` | `DummyRegressor` por estación y hora | `strategy=mean` | 0.864 | 1.678 | 0.939 | 1.721 | TODO | — | — |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO % | TODO % |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO % | TODO % |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO % | TODO % |

<!-- ![Figura 1. P1: viajes reales contra predichos del modelo elegido en los primeros 7 días del conjunto de prueba (artefacto graficas/ de la corrida).](../figuras/p1_real_vs_predicho.png) -->

<div class="pendiente">

TODO. Análisis y modelo elegido: mejora sobre el *baseline*, lectura de `test_mae` (flota activa) y `test_rmse` (horas pico), estabilidad validación-prueba, costo de entrenamiento e inferencia; cerrar con el modelo elegido y por qué.

</div>

## P2: riesgo de vaciarse o llenarse

**Tabla 2.** P2: *baseline* y corridas candidatas, validación.

| Corrida | Modelo | Hiperparámetros clave | `val_recall_vaciarse` | `val_recall_llenarse` | `val_f1_macro` | Duración |
|---------|--------|-----------------------|-----------------------|-----------------------|----------------|----------|
| `P2-baseline-dummy-estacion-intervalo-tipo-dia` | `DummyClassifier` por estación, intervalo y tipo de día | `strategy=prior` | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

**Tabla 3.** P2: *baseline* y corridas candidatas, prueba y mejora respecto al *baseline* (puntos porcentuales de sensibilidad).

| Corrida | `test_recall_vaciarse` | `test_precision_vaciarse` | `test_recall_llenarse` | `test_precision_llenarse` | `test_f1_macro` | Mejora vaciarse | Mejora llenarse |
|---------|------------------------|---------------------------|------------------------|---------------------------|-----------------|-----------------|-----------------|
| `P2-baseline-dummy-estacion-intervalo-tipo-dia` | 0.289 | 0.370 | 0.273 | 0.559 | TODO | — | — |
| TODO | TODO | TODO | TODO | TODO | TODO | +TODO pp | +TODO pp |
| TODO | TODO | TODO | TODO | TODO | TODO | +TODO pp | +TODO pp |
| TODO | TODO | TODO | TODO | TODO | TODO | +TODO pp | +TODO pp |

<!-- ![Figura 2. P2: matriz de confusión del modelo elegido sobre el conjunto de prueba (artefacto graficas/ de la corrida).](../figuras/p2_matriz_confusion.png) -->

<div class="pendiente">

TODO. Análisis y modelo elegido: sensibilidad en las clases de riesgo (no detectar cuesta más que una falsa alarma) contra precisión (viajes de camioneta en vano), matriz de confusión, estabilidad, costo; cerrar con el modelo elegido y por qué.

</div>

## P3: grupos de estaciones

**Tabla 4.** P3: *baseline* y corridas candidatas.

| Corrida | Modelo | Hiperparámetros clave | `train_silhouette` | `train_davies_bouldin` | Tamaño de los grupos | Duración | Mejora en silueta |
|---------|--------|-----------------------|--------------------|------------------------|----------------------|----------|-------------------|
| `P3-baseline-kmeans-k4` | `StandardScaler` + `KMeans` | `n_clusters=4` | 0.173 | 1.640 | 87 / 38 / 111 / 135 | TODO | — |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO % |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO % |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO % |

<!-- ![Figura 3. P3: mapa de las estaciones por grupo del modelo elegido (artefacto graficas/ de la corrida).](../figuras/p3_mapa_grupos.png) -->

<div class="pendiente">

TODO. Análisis y modelo elegido: silueta y Davies-Bouldin contra interpretabilidad de los grupos (vocación y coherencia en el mapa), estabilidad con otra semilla, costo; cerrar con el modelo elegido y por qué.

</div>
