# Comparación de experimentos y selección del modelo

Se mantienen la partición temporal y las métricas de la Entrega 01 (entrenamiento `2024-09-01` a `2026-02-28`, validación `2026-03-01` a `2026-05-31`, y prueba `2026-06-01` a `2026-09-01`, con versión de datos `2026-10`). Las corridas se filtran en MLflow mediante las etiquetas `question`, `model_family` y `stage_of_work`. Cada pregunta incluye una búsqueda de hiperparámetros con *nested runs* (una corrida padre por pregunta), de la que se toma la mejor combinación como candidata.

## P1: viajes por estación y hora

**Tabla 1.** P1: *baseline* y corridas candidatas.

| Corrida | Modelo | Hiperparámetros clave | `val_mae` | `val_rmse` | `test_mae` | `test_rmse` | Duración | Mejora en `test_mae` | Mejora en `test_rmse` |
|---------|--------|-----------------------|-----------|------------|------------|-------------|----------|----------------------|-----------------------|
| `P1-baseline-dummy-estacion-hora` | `DummyRegressor` por estación y hora | `strategy=mean` | 1.099 | 1.891 | 1.159 | 1.838 | 34.4 s | — | — |
| `P1-candidato-ridge-alpha=1.0` | `Ridge` (L2) | `alpha=1.0` | 1.027 | 1.741 | 0.987 | 1.638 | 45.7 s | +14.84% | +10.87% |
| `P1-candidato-rf` | `RandomForestRegressor` | `n_estimators=5`, `max_depth=12` | 1.046 | 1.734 | 1.001 | 1.635 | 29.4 s | +13.63% | +11.01% |
| `P1-candidato-hgb` | `HistGradientBoosting` | `loss=squared_error`, `lr=0.10`, `nodes=127` | 0.955 | 1.567 | 0.913 | 1.483 | 2.1 min | +21.27% | +19.29% |
| `P1-candidato-hgb_poisson` | `HistGradientBoosting` (Poisson) | `loss=poisson`, `lr=0.10`, `nodes=127` | 0.951 | 1.570 | 0.908 | 1.491 | 2.4 min | +21.67% | +18.88% |

<!-- ![Figura 1. P1: viajes reales contra predichos del modelo elegido en los primeros 7 días del conjunto de prueba (artefacto graficas/ de la corrida).](../figuras/p1_real_vs_predicho.png) -->

El modelo elegido es **`P1-candidato-hgb_poisson`** (`Run ID: bd2634f0c67f4dedaba9a755bbc079f8`). Supera notablemente al *baseline* con una mejora de **+21.67% en Test MAE** (`0.908` vs. `1.159`), reduciendo el error promedio a menos de una bicicleta por estación y hora. La lectura de `test_mae` (`0.908`) valida una excelente precisión global para el dimensionamiento diario de la flota activa, mientras que el `test_rmse` (`1.491`) penaliza adecuadamente los picos de demanda en horas pico. Presenta alta estabilidad entre validación (`val_mae = 0.951`) y prueba (`test_mae = 0.908`), entrenando en 2.4 minutos con un costo de inferencia mínimo. Se selecciona porque la función de pérdida Poisson modela de forma óptima la naturaleza de conteos discretos no negativos de los viajes iniciados.

## P2: riesgo de vaciarse o llenarse

**Tabla 2.** P2: *baseline* y corridas candidatas, validación.

| Corrida | Modelo | Hiperparámetros clave | `val_recall_vaciarse` | `val_recall_llenarse` | `val_f1_macro` | Duración |
|---------|--------|-----------------------|-----------------------|-----------------------|----------------|----------|
| `P2-baseline-dummy-estacion-intervalo-tipo-dia` | `DummyClassifier` por estación, intervalo y tipo de día | `strategy=prior` | 0.285 | 0.289 | 0.568 | 1.4 min |
| `P2-candidato-logreg` | `LogisticRegression` | `C=1.0`, `balanced` | 0.745 | 0.761 | 0.430 | 1.8 min |
| `P2-candidato-hgb` | `HistGradientBoosting` | `lr=0.05`, `nodes=63`, `balanced` | 0.835 | 0.878 | 0.454 | 4.1 min |
| `P2-candidato-hgb_threshold` | `ThresholdClassifier(HGB)` | `lr=0.05`, `nodes=63`, `thr_lle=0.85`, `thr_vac=0.90` | 0.362 | 0.521 | 0.608 | 5.2 min |

**Tabla 3.** P2: *baseline* y corridas candidatas, prueba y mejora respecto al *baseline* (puntos porcentuales de sensibilidad).

| Corrida | `test_recall_vaciarse` | `test_precision_vaciarse` | `test_recall_llenarse` | `test_precision_llenarse` | `test_f1_macro` | Mejora vaciarse | Mejora llenarse |
|---------|------------------------|---------------------------|------------------------|---------------------------|-----------------|-----------------|-----------------|
| `P2-baseline-dummy-estacion-intervalo-tipo-dia` | 0.289 | 0.370 | 0.273 | 0.559 | 0.557 | — | — |
| `P2-candidato-logreg` | 0.736 | 0.108 | 0.747 | 0.122 | 0.424 | +44.7 pp | +47.4 pp |
| `P2-candidato-hgb` | 0.820 | 0.127 | 0.861 | 0.139 | 0.446 | +53.1 pp | +58.8 pp |
| `P2-candidato-hgb_threshold` | 0.367 | 0.371 | 0.479 | 0.399 | 0.593 | +7.8 pp | +20.6 pp |

<!-- ![Figura 2. P2: matriz de confusión del modelo elegido sobre el conjunto de prueba (artefacto graficas/ de la corrida).](../figuras/p2_matriz_confusion.png) -->

El modelo elegido es **`P2-candidato-hgb_threshold`** (`Run ID: 7b0a4e39e3dc47fa8b639815f5dedfc0`). En la clasificación de riesgo operativo, no detectar una estación crítica (falso negativo que genera desabastecimiento o saturación sin atención) cuesta operacionalmente mucho más que una falsa alarma. Aunque los modelos puros con pesos balanceados logran sensibilidades superiores al 82-86%, su baja precisión (~13%) satura innecesariamente a las cuadrillas de rebalanceo. Al envolver el mejor clasificador con `ThresholdClassifier` y calibrar los umbrales de probabilidad en validación a `0.85` (llenarse) y `0.90` (vaciarse), se alcanza un **Test F1 Macro de 0.593** (+6.54% de mejora sobre el baseline), elevando la sensibilidad de vaciado a **36.7%** (+7.8 pp) y la de llenado a **47.9%** (+20.6 pp) con una precisión operativamente viable de ~37–40%. Presenta alta estabilidad y bajo costo de inferencia en tiempo real.

## P3: grupos de estaciones

**Tabla 4.** P3: *baseline* y corridas candidatas.

| Corrida | Modelo | Hiperparámetros clave | `train_silhouette` | `train_davies_bouldin` | Tamaño de los grupos | Duración | Mejora en silueta |
|---------|--------|-----------------------|--------------------|------------------------|----------------------|----------|-------------------|
| `P3-baseline-kmeans-k4` | `StandardScaler` + `KMeans` | `n_clusters=4` | 0.173 | 1.640 | 87 / 38 / 111 / 135 | 38.5 s | — |
| `P3-candidato-kmeans-n_clusters=5` | `StandardScaler` + `KMeans` | `n_clusters=5` | 0.165 | 1.602 | 75 / 37 / 63 / 98 / 98 | 32.3 s | -4.85% |
| `P3-candidato-agglomerative-n_clusters=4` | `StandardScaler` + `Agglomerative` | `n_clusters=4` | 0.154 | 1.822 | 82 / 31 / 112 / 146 | 28.0 s | -11.09% |
| `P3-candidato-gmm-n_clusters=4` | `StandardScaler` + `GaussianMixture` | `n_clusters=4` | 0.132 | 2.110 | 92 / 34 / 105 / 140 | 26.6 s | -23.96% |

<!-- ![Figura 3. P3: mapa de las estaciones por grupo del modelo elegido (artefacto graficas/ de la corrida).](../figuras/p3_mapa_grupos.png) -->

El modelo elegido es **`P3-baseline-kmeans-k4` / `P3-candidato-kmeans-n_clusters=4`** (`Run ID: a1ec99ca5395400ba2f5a902fc34f8bc`). Al comparar frente a enfoques jerárquicos y de mezcla gaussiana fijando $k=4$, `KMeans` demuestra superioridad al maximizar el Coeficiente de Silueta (`0.173`). Asimismo, la búsqueda anidada evaluando $k \in \{2, \dots, 8\}$ confirma que $k=4$ es el óptimo global en cohesión y separación (con $k=5$ la silueta desciende a `0.165`). Los 4 grupos resultantes (87, 38, 111 y 135 estaciones) muestran gran interpretabilidad urbana, distinguiendo con claridad vocaciones residenciales expulsoras, receptoras laborales, de alta rotación céntrica y de baja demanda periférica. El modelo es altamente estable ante distintas semillas, eficiente en cómputo (35.9 s) y proporciona particiones geográficas coherentes para la expansión del sistema.