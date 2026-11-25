# Model Card · modelos de MiBici (Equipo 10)

Documenta los tres modelos del proyecto que están registrados en el Model Registry de MLflow y
que sirve la API REST: qué predicen, para quién, con qué datos se entrenaron, cómo se
evaluaron, dónde fallan y cómo se mantienen. Sigue la estructura de Mitchell et al. (2019).
Los datos de cada versión salen del Model Registry del servidor del equipo y de los reportes
de las entregas 02 (selección de modelos) y 05 (drift y A/B testing).

Última actualización: 25 de noviembre de 2026. Esta tarjeta se actualiza en el mismo *Pull
Request* que registra una versión nueva o mueve un alias.

| Pregunta | Modelo registrado | Versión en producción (`@champion`) | Qué responde |
|---|---|---|---|
| P1 | `mibici-equipo10-p1-demanda` | 3 | Cuántos viajes saldrán de una estación en cada hora del día siguiente |
| P2 | `mibici-equipo10-p2-riesgo` | 1 | Si una estación tiende a vaciarse, a llenarse o no tiene riesgo en el siguiente intervalo de 3 horas |
| P3 | `mibici-equipo10-p3-grupos` | 1 | A qué grupo de uso pertenece una estación |

## 1. Datos (común a los tres modelos)

- **Fuente.** [Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/): un CSV por mes
  con cada viaje (identificador, inicio, fin, estación de origen y de destino) y el catálogo de
  estaciones con su capacidad de anclajes. Los archivos no entran al repositorio; `make data`
  los descarga y los verifica contra `datapackage.json` (SHA-256 de cada archivo).
- **Periodo.** Septiembre de 2024 a agosto de 2026 (24 meses, 8,702,429 viajes tras la
  limpieza). Septiembre de 2026 se reservó para el monitoreo y el A/B testing. La versión de
  datos es la etiqueta `version_datos` de cada versión registrada (`2026-10` y `2026-11`).
- **Partición cronológica.** Entrenamiento de septiembre de 2024 a febrero de 2026 (18 meses),
  validación de marzo a mayo de 2026 y prueba de junio a agosto de 2026. Los hiperparámetros y
  el umbral de P2 se eligieron solo con validación; prueba se usó una vez, al final.
- **Sin fuga de información.** Toda variable histórica se calcula con datos cerrados antes del
  momento de la predicción (en P1, al menos 24 horas antes); los escaladores se ajustan solo
  con entrenamiento; la serie de cada estación empieza el día en que abrió.
- **Datos personales.** La ingesta descarta `Usuario_Id`, `Genero` y `Año_de_nacimiento`
  antes de escribir cualquier tabla. Ningún modelo, artefacto ni *run* los contiene. Los tres
  modelos trabajan con agregados por estación, no con personas.
- **Condiciones de uso.** El portal no publica licencia; la AMIM declara la información
  pública y de libre acceso. Los datos se usan solo con fines académicos y no se
  redistribuyen. El repositorio no incluye un archivo de licencia: es el proyecto integrador
  del curso TC5061.

Detalle en el README: [Datos utilizados](../README.md#datos-utilizados) y
[Estrategia de partición](../README.md#estrategia-de-partición).

## 2. P1 · Demanda por estación y hora

### Detalles del modelo

| | |
|---|---|
| Modelo registrado | `mibici-equipo10-p1-demanda`, versión 3 (`@champion` desde el 15 de noviembre de 2026) |
| Tipo | `HistGradientBoostingRegressor` de scikit-learn con pérdida de Poisson |
| Hiperparámetros | `learning_rate=0.1`, `max_leaf_nodes=127`, `max_iter=200`; semilla 42 |
| Corrida de origen | `P1-reentrenado-hasta-2026-09-01` en el experimento `mibici-p1-demanda` |
| Linaje | `git_commit` `6458899b`, `version_datos` `2026-11`; reentrenamiento de la versión 1 (`P1-candidato-hgb_poisson-...`, `git_commit` `5a95a6f5`, `version_datos` `2026-10`) con los mismos hiperparámetros y seis meses más de datos |
| Variables de entrada | 10: `station_id`, `hour_of_day`, `day_of_week`, `is_weekend`, `capacity`, `departures_lag_24`, `departures_lag_168`, `arrivals_lag_24`, `net_flow_lag_24`, `departures_mean_3_lag_24` |
| Salida | `departures_pred`: número de viajes que saldrán de la estación en esa hora (nunca negativo) |

### Uso previsto

- **Quién y para qué.** El operador del sistema: al cierre de cada día recibe el pronóstico hora
  por hora del día siguiente para dimensionar la flota en calle, programar mantenimiento en
  horas valle y asignar turnos.
- **Horizonte y granularidad.** 24 horas, por estación y hora de servicio (de 5:00 a 0:59; la
  API rechaza las horas de 1:00 a 4:59, sin servicio).
- **Fuera de alcance.** Pronósticos a menos de 24 horas (sus variables históricas no existen
  antes del cierre del día), estaciones que no están en el catálogo o con capacidad 0, otras
  ciudades y cualquier uso que identifique personas.

### Datos de entrenamiento

Versión 3: septiembre de 2024 a agosto de 2026 (24 meses), 5,489,700 filas de estación × hora,
incluidas las estaciones que abrieron entre marzo y agosto de 2026. La versión 1 se entrenó con
los 18 meses de entrenamiento de la partición (3,998,020 filas, 371 estaciones).

### Evaluación

Métricas en viajes por estación y hora. La versión 1 se eligió con la partición de la sección 1;
la versión 3 se comparó con ella sobre septiembre de 2026 (280,760 filas), mes que ninguna de
las dos vio al entrenarse, en la corrida `P1-ab-test-v1-vs-v3`.

| Conjunto | *Baseline* (media por estación y hora) | Versión 1 | Versión 3 (`@champion`) |
|---|---:|---:|---:|
| Validación (mar–may 2026), MAE / RMSE | 1.099 / 1.891 | 0.951 / 1.570 | — |
| Prueba (jun–ago 2026), MAE / RMSE | 1.159 / 1.838 | 0.908 / 1.491 | — |
| Septiembre de 2026, todas las estaciones, MAE / RMSE | — | 1.017 / 1.761 | 1.004 / 1.749 |
| Septiembre de 2026, estaciones conocidas (221,400 filas) | — | 1.116 / 1.912 | 1.113 / 1.902 |
| Septiembre de 2026, estaciones nuevas (59,360 filas) | — | 0.648 / 1.021 | 0.599 / 0.983 |

La versión 3 reduce el MAE 1.2 % en todo septiembre (diferencia −0.0126, intervalo de confianza
del 95 % de −0.0151 a −0.0102 por *bootstrap* de 1,000 remuestreos de días completos) y 7.5 % en
las estaciones nuevas. Por eso se promovió a `@champion`. En el periodo de prueba de la versión 1
el error fue parecido en entrenamiento (0.968), validación y prueba, así que el modelo no se
aprendió de memoria el periodo de entrenamiento.

### Limitaciones y riesgos

- El error sube en meses distintos a los de prueba: el MAE de la versión 1 pasó de 0.908 en su
  periodo de prueba a 1.017 en septiembre de 2026. Parte de ese cambio es el mes, no el modelo.
- Las estaciones nuevas son el riesgo principal. El drift de septiembre de 2026 está en
  `station_id` (distancia de Jensen-Shannon 0.289) y `capacity` (Wasserstein 0.105), no en la
  demanda ni en sus rezagos. Una estación que abra después del último reentrenamiento se
  pronostica sin historial propio.
- Los movimientos de rebalanceo no están en los datos: el modelo pronostica viajes de
  usuarios, no el inventario de la estación.
- La mejor combinación de la búsqueda quedó en el borde de la rejilla (`learning_rate` y
  `max_leaf_nodes` más altos); ampliarla es una mejora pendiente.

## 3. P2 · Riesgo de vaciarse o llenarse

### Detalles del modelo

| | |
|---|---|
| Modelo registrado | `mibici-equipo10-p2-riesgo`, versión 1 (`@champion` desde el 25 de octubre de 2026) |
| Tipo | `ThresholdClassifier` (`mibici/modeling/thresholds.py`) sobre un `HistGradientBoostingClassifier` de scikit-learn con pesos por clase |
| Hiperparámetros | `learning_rate=0.05`, `max_leaf_nodes=63`, `max_iter=200`; umbrales de alerta 0.90 (`vaciarse`) y 0.85 (`llenarse`), elegidos con validación entre 0.05 y 0.95 y guardados dentro del modelo; semilla 42 |
| Corrida de origen | `P2-candidato-hgb_threshold-learning_rate=0.05-max_iter=200-max_leaf_nodes=63` en `mibici-p2-riesgo` |
| Linaje | `git_commit` `7a74fff3`, `version_datos` `2026-10` |
| Variables de entrada | 13: `station_id`, `interval_of_day` (0 a 7), `day_of_week`, `is_weekend`, `capacity`, `departures_lag_1`, `departures_lag_8`, `departures_lag_56`, `arrivals_lag_1`, `arrivals_lag_8`, `net_flow_lag_1`, `net_flow_lag_8`, `departures_mean_3` |
| Salida | `risk_pred`: `vaciarse`, `llenarse` o `sin_riesgo`, con la probabilidad de cada clase |

### Uso previsto

- **Quién y para qué.** El equipo de rebalanceo: al cierre de cada intervalo de 3 horas decide
  qué estaciones atender primero y cuántas bicicletas abastecer o retirar en el intervalo
  siguiente.
- **Qué significa el riesgo.** Los datos abiertos no incluyen el inventario de bicicletas, así
  que el riesgo se aproxima con el flujo neto (llegadas − salidas) del intervalo comparado con la
  capacidad de la estación: `vaciarse` si es ≤ −40 % de la capacidad, `llenarse` si es ≥ +40 % y
  `sin_riesgo` en el resto. Una alerta señala presión de flujo, no un inventario medido.
- **Cómo alerta.** Solo marca `vaciarse` o `llenarse` si la probabilidad de esa clase supera su
  umbral; si no, responde `sin_riesgo`. Los umbrales se eligieron para maximizar el F1 macro, es
  decir, para equilibrar alertas acertadas y alertas que no saturen a las cuadrillas.
- **Fuera de alcance.** Estimar cuántas bicicletas hay en una estación, estaciones con
  capacidad 0 (se excluyen de la tabla) y horizontes distintos de 3 horas.

### Datos de entrenamiento

Septiembre de 2024 a febrero de 2026: 1,599,208 filas de estación × intervalo de 3 horas;
validación con 271,920 y prueba con 319,664. El 94 % de los intervalos son `sin_riesgo`, por lo
que el modelo se entrena con pesos por clase y se evalúa con métricas por clase.

### Evaluación

| Métrica | *Baseline* (`DummyClassifier` por estación, intervalo y tipo de día) | `@champion` |
|---|---:|---:|
| `val_f1_macro` | 0.568 | 0.608 |
| `test_f1_macro` | 0.557 | 0.593 |
| `test_recall_vaciarse` / `test_precision_vaciarse` | 0.289 / 0.370 | 0.367 / 0.371 |
| `test_recall_llenarse` / `test_precision_llenarse` | 0.273 / 0.559 | 0.479 / 0.399 |
| `val_avg_precision_vaciarse` / `val_avg_precision_llenarse` (no dependen del umbral) | 0.320 / 0.470 | 0.353 / 0.491 |

Fue el único candidato que superó al *baseline* en F1 macro. En prueba detecta casi el doble de
los intervalos en que una estación se llena (47.9 % contra 27.3 %) y más de los que se vacía
(36.7 % contra 28.9 %), con la misma precisión en `vaciarse` y menor en `llenarse`. Para el
equipo de rebalanceo, una estación llena que no se atiende cuesta más que una falsa alarma. Como
el umbral se eligió con validación, su `val_f1_macro` es algo optimista; la caída pequeña en
prueba (de 0.608 a 0.593) muestra que no se sobreajustó.

### Limitaciones y riesgos

- La etiqueta de riesgo es una aproximación. Un flujo neto del −40 % de la capacidad no garantiza
  que la estación se vacíe, porque el inventario inicial y los movimientos de rebalanceo no
  están en los datos.
- Detecta poco más de un tercio de los intervalos `vaciarse` y menos de la mitad de los
  `llenarse`: sirve para priorizar, no para sustituir la supervisión del inventario.
- Sin los umbrales, el mismo clasificador detectaba más del 80 % de los intervalos en riesgo,
  pero solo 13 a 16 % de sus alertas eran ciertas. Cambiar los umbrales cambia ese intercambio y
  exige volver a elegirlos con validación.
- Igual que P1, el drift de septiembre de 2026 se concentra en `station_id` (0.287) y
  `capacity` (0.101): las estaciones nuevas o con capacidad cambiada tienen menos historial.
- P2 es el *endpoint* más lento de la API, porque evalúa el modelo dos veces por petición (clase y
  probabilidades).

## 4. P3 · Grupos de estaciones

### Detalles del modelo

| | |
|---|---|
| Modelo registrado | `mibici-equipo10-p3-grupos`, versión 1 (`@champion` desde el 25 de octubre de 2026); versión 2 (`@challenger`): K-Means con k = 5 |
| Tipo | `Pipeline` de scikit-learn: `StandardScaler` + `KMeans` con 4 grupos |
| Hiperparámetros | `n_clusters=4`; semilla 42. La búsqueda con k de 2 a 8 confirmó que k = 4 tiene la mayor silueta |
| Corrida de origen | `P3-baseline-kmeans-k4` en `mibici-p3-perfiles` |
| Linaje | `git_commit` `5a95a6f5`, `version_datos` `2026-10` |
| Variables de entrada | 8, el perfil de uso de la estación: `morning_pct`, `midday_pct`, `afternoon_pct`, `night_pct` (porcentaje de movimientos por periodo del día), `weekdays_pct`, `log_daily_use_per_dock` (log de movimientos diarios por anclaje), `balance_morning` y `balance_afternoon` ((llegadas − salidas) ÷ (llegadas + salidas); −1 si solo salen bicicletas, +1 si solo llegan) |
| Salida | `group`: entero de 0 a 3 |

### Uso previsto

- **Quién y para qué.** Planeación urbana y expansión: dónde instalar estaciones nuevas y cuáles
  ampliar, a partir de qué estaciones comparten hábitos de uso y dónde están en el mapa.
- **Granularidad.** Una estación, con su comportamiento consolidado del periodo; se recalcula
  mensualmente. La latitud y la longitud no entran al modelo; solo sirven para el mapa.
- **Fuera de alcance.** Pronosticar demanda o riesgo (eso lo hacen P1 y P2) y decidir por sí
  solo retirar el servicio de una estación.

### Datos de entrenamiento

371 estaciones con movimientos entre septiembre de 2024 y febrero de 2026, una fila por estación.
Las 99 estaciones que abrieron a partir de marzo de 2026 no tienen perfil en ese periodo y no
entraron al ajuste; el modelo les asigna grupo con su perfil cuando lo tienen.

### Evaluación

No hay variable objetivo: se evalúa la cohesión y la separación de los grupos sobre el periodo de
entrenamiento. `train_silhouette` 0.173 y `train_davies_bouldin` 1.640, contra 0.165 y 1.602 de la
versión 2 (k = 5) y 0.154 y 1.822 del agrupamiento jerárquico con k = 4.

Perfil promedio de cada grupo (artefacto `grupos/p3_perfil_grupos.csv` de la corrida de origen):

| Grupo | Estaciones | Mañana % | Mediodía % | Tarde % | Noche % | Entre semana % | Uso por anclaje (log) | Balance mañana | Balance tarde | Lectura |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 87 | 35.2 | 23.1 | 30.4 | 11.3 | 83.5 | 1.24 | +0.13 | −0.25 | Reciben bicicletas en la mañana y las expulsan en la tarde, casi solo entre semana: destinos laborales |
| 1 | 38 | 34.8 | 27.9 | 26.3 | 11.1 | 70.9 | 1.08 | −0.04 | −0.13 | Mayor uso en fin de semana y al mediodía |
| 2 | 111 | 32.0 | 21.5 | 30.6 | 15.9 | 79.0 | 0.93 | −0.30 | +0.14 | Expulsan bicicletas en la mañana y las reciben en la tarde, con más uso nocturno y el menor uso por anclaje: residenciales |
| 3 | 135 | 30.5 | 25.0 | 30.1 | 14.4 | 79.2 | 1.95 | +0.01 | +0.02 | El mayor uso por anclaje, con flujo equilibrado todo el día: alta rotación |

### Limitaciones y riesgos

- La silueta es baja (0.173): los grupos se traslapan y una estación cercana a la frontera puede
  cambiar de grupo al recalcular. Los perfiles promedio sí se distinguen y son los que se usan
  para interpretar.
- El drift de septiembre de 2026 alcanzó 5 de las 8 variables (`midday_pct`, `afternoon_pct`,
  `night_pct`, `weekdays_pct` y `balance_afternoon`, prueba K-S con p < 0.05). Parte se debe a
  comparar un perfil de 18 meses contra uno de un solo mes; aun así, los grupos pueden quedar
  desactualizados y conviene recalcularlos con cada periodo nuevo.
- El grupo de menor uso por anclaje (grupo 2) describe un patrón de uso, no la utilidad social
  de esas estaciones: no debe usarse por sí solo para justificar retirar servicio de una zona.

## 5. Operación y mantenimiento

- **Cómo se sirven.** `mibici/api.py` carga los tres modelos al arrancar por nombre y alias
  (`models:/<nombre>@champion`) desde el Model Registry, o desde `models/registry/` cuando el
  servidor no está disponible, y los deja en memoria. `POST /predict/p1`, `/predict/p2` y
  `/predict/p3` validan la entrada con los esquemas de `mibici/schemas.py` y responden 422 ante
  una petición inválida; `GET /health` indica qué versión responde en cada modelo. Ver
  [API REST](../README.md#api-rest).
- **Desempeño medido.** Con 2 CPU y un *worker*, la API responde con p95 de 17 ms a 10 usuarios
  y se satura en unas 38 peticiones por segundo; la demanda de MiBici (unas 400 mil predicciones
  al mes) usa el 0.4 % de esa capacidad. Costo estimado de USD 0.34 por millón de predicciones
  con la instancia saturada (reporte de la entrega 06).
- **Monitoreo.** `make monitor-data` construye los conjuntos de referencia (periodo de
  entrenamiento) y actual (mes más reciente) y `make drift` genera el reporte de Evidently por
  pregunta (`reports/evidently/`). Se declara drift del conjunto cuando al menos la mitad de las
  variables lo tienen. Ver [Monitoreo](../README.md#monitoreo-de-datos-y-preparación-para-detección-de-drift).
- **Reentrenamiento.** Cada mes, cuando MiBici publica datos nuevos: `make retrain` reentrena
  P1 con todos los meses anteriores y lo registra como `@challenger`; `make ab-test` hace que
  `@champion` y `@challenger` predigan el mes nuevo y compara MAE y RMSE con un intervalo de
  confianza por *bootstrap* de días. La versión nueva se propone solo si todo el intervalo de la
  diferencia de MAE es negativo. El mismo ciclo aplica a P2 con el F1 macro; P3 se recalcula y se
  revisa con el drift, porque no tiene valor real con qué comparar.
- **Promoción y reversión.** Una persona revisa la comparación y mueve el alias `champion`
  (`python -m mibici.modeling.ab_test compare --promote` o desde la interfaz de MLflow). La API
  sirve la versión nueva al reiniciar. Regresar a la versión anterior es mover el alias de
  vuelta, sin cambiar código ni reentrenar.
- **Responsables.** El equipo de la sección [Integrantes y roles](../README.md#integrantes-y-roles)
  del README; cada cambio de alias tiene un responsable y queda en el historial del Model
  Registry.

## 6. Historial de versiones

Estado del Model Registry al 25 de noviembre de 2026.

| Modelo | Versión | Alias | Corrida de origen | `git_commit` | `version_datos` | Registrada |
|---|---:|---|---|---|---|---|
| `mibici-equipo10-p1-demanda` | 1 | — | `P1-candidato-hgb_poisson-learning_rate=0.1-max_iter=200-max_leaf_nodes=127` | `5a95a6f5` | `2026-10` | 2026-10-25 |
| `mibici-equipo10-p1-demanda` | 2 | — | `P1-reentrenado-hasta-2026-09-01` (corrida anterior del mismo reentrenamiento) | `6458899b` | `2026-11` | 2026-11-15 |
| `mibici-equipo10-p1-demanda` | 3 | `champion`, `challenger` | `P1-reentrenado-hasta-2026-09-01` | `6458899b` | `2026-11` | 2026-11-15 |
| `mibici-equipo10-p2-riesgo` | 1 | `champion` | `P2-candidato-hgb_threshold-learning_rate=0.05-max_iter=200-max_leaf_nodes=63` | `7a74fff3` | `2026-10` | 2026-10-25 |
| `mibici-equipo10-p3-grupos` | 1 | `champion` | `P3-baseline-kmeans-k4` | `5a95a6f5` | `2026-10` | 2026-10-25 |
| `mibici-equipo10-p3-grupos` | 2 | `challenger` | `P3-candidato-kmeans-n_clusters=5` | `845bab87` | `2026-10` | 2026-10-25 |

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E.,
  Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the
  Conference on Fairness, Accountability, and Transparency (FAT\* '19)*, 220–229.
  https://doi.org/10.1145/3287560.3287596
- MLflow. (s.f.). *MLflow Model Registry* [Documentación]. https://mlflow.org/docs/latest/ml/model-registry/
- Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A.,
  Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the
  enterprise*. O'Reilly Media.
