# Model Registry y ciclo de vida

**Tabla 5.** Modelos registrados y su linaje. El *commit* y la versión de datos son las etiquetas `git_commit` y `version_datos` de cada versión.

| Pregunta | Nombre registrado | Versión | Alias | Corrida de origen | *Commit* | Versión de datos |
|----------|-------------------|---------|-------|-------------------|----------|------------------|
| P1 | `mibici-equipo10-p1-demanda` | 1 | `champion` | `P1-candidato-hgb_poisson-learning_rate=0.1-max_iter=200-max_leaf_nodes=127` (`bd2634f0`) | `5a95a6f5` | `2026-10` |
| P2 | `mibici-equipo10-p2-riesgo` | 1 | `champion` | `P2-candidato-hgb_threshold-learning_rate=0.05-max_iter=200-max_leaf_nodes=63` (`7b0a4e39`) | `7a74fff3` | `2026-10` |
| P3 | `mibici-equipo10-p3-grupos` | 1 | `champion` | `P3-baseline-kmeans-k4` (`52e80800`) | `5a95a6f5` | `2026-10` |
| P3 | `mibici-equipo10-p3-grupos` | 2 | `challenger` | `P3-candidato-kmeans-n_clusters=5` (`02118ac9`) | TODO | `2026-10` |

Cada modelo registrado tiene una descripción con la pregunta que responde, el modelo y sus hiperparámetros, el periodo de entrenamiento, sus métricas contra el *baseline* y el usuario al que sirve:

- **P1 · Demanda:** `HistGradientBoostingRegressor` con pérdida de Poisson para el operador del sistema (`val_mae` 0.951 y `test_mae` 0.908).
- **P2 · Riesgo:** `HistGradientBoostingClassifier` con pesos por clase y umbrales de alerta de 0.90 (`vaciarse`) y 0.85 (`llenarse`) dentro del modelo, para el equipo de rebalanceo (`val_f1_macro` 0.608 y `test_f1_macro` 0.593).
- **P3 · Grupos:** `StandardScaler` + `KMeans` con 4 grupos sobre 371 estaciones y 8 variables de perfil de uso, para planeación urbana (`train_silhouette` 0.173).

Cada versión conserva su linaje: apunta a su corrida de origen y, por ella, a su experimento, parámetros, métricas y artefactos; las etiquetas `git_commit` y `version_datos` permiten reconstruirla con el mismo código y los mismos datos.

Para el ciclo de vida se usó el esquema vigente de **alias** (MLflow, s.f.-b, s.f.-c): `champion` marca la versión en producción y `challenger`, la siguiente mejor. No se usaron las etapas clásicas (*Staging* y *Production*) porque MLflow las marcó como obsoletas desde la versión 2.9 y recomienda alias. Con alias, el código que consume un modelo siempre lo pide por nombre y alias (`models:/mibici-equipo10-p1-demanda@champion`), sin depender del número de versión. Esto separa el código de despliegue de los modelos, una práctica central de MLOps para versionar modelos sin romper a quien los usa (Lakshmanan et al., 2020; Treveil et al., 2020).

Para llevar a producción un modelo reentrenado, por ejemplo con una versión nueva de `datapackage.json`, se registra como versión nueva del mismo nombre y se le asigna `challenger`. Si supera a la versión vigente en las métricas de validación, se mueve `champion` a esa versión, desde la interfaz o con `MlflowClient().set_registered_model_alias()`. Si se detecta un problema en producción, la reversión es inmediata: basta con volver a apuntar `champion` a la versión anterior, sin cambiar código ni reentrenar.

Se verificó que los tres modelos en producción se cargan por su nombre y alias (Figura 11):

```bash
python -c "import os; os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'; import mlflow; from mibici.config import MLFLOW_TRACKING_URI; mlflow.set_tracking_uri(MLFLOW_TRACKING_URI); [print(n, '->', type(mlflow.sklearn.load_model(f'models:/{n}@champion')).__name__) for n in ['mibici-equipo10-p1-demanda', 'mibici-equipo10-p2-riesgo', 'mibici-equipo10-p3-grupos']]"
```

```text
mibici-equipo10-p1-demanda -> HistGradientBoostingRegressor
mibici-equipo10-p2-riesgo -> ThresholdClassifier
mibici-equipo10-p3-grupos -> Pipeline
```
