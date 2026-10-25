# Model Registry y ciclo de vida

**Tabla 5.** Modelos registrados y su linaje.

| Pregunta | Nombre registrado | Versión | Alias | Descripción | Corrida de origen | *Commit* | Versión de datos |
|----------|-------------------|---------|-------|-------------|-------------------|----------|------------------|
| P1 | `mibici-equipo10-p1-demanda` | 1 | `champion` | `HistGradientBoostingRegressor` con pérdida Poisson (`lr=0.1`, `max_iter=200`, `max_leaf_nodes=127`) para predicción de viajes por estación y hora (`test_mae=0.908`). | `P1-candidato-hgb_poisson-learning_rate=0.1-max_iter=200-max_leaf_nodes=127` (`bd2634f0`) | `5a95a6f5` | `2026-10` |
| P2 | `mibici-equipo10-p2-riesgo` | 1 | `champion` | `ThresholdClassifier` sobre `HistGradientBoostingClassifier` (`lr=0.05`, `max_leaf_nodes=63`) con umbrales calibrados (`llenarse=0.85`, `vaciarse=0.90`; `test_f1_macro=0.593`). | `P2-candidato-hgb_threshold-learning_rate=0.05-max_iter=200-max_leaf_nodes=63` (`7b0a4e39`) | `7a74fff3` | `2026-10` |
| P3 | `mibici-equipo10-p3-grupos` | 1 | `champion` | Pipeline `StandardScaler` + `KMeans` (`n_clusters=4`) sobre 371 estaciones activas con 8 variables de perfil horario y balance (`train_silhouette=0.173`). | `P3-candidato-kmeans-n_clusters=4` (`a1ec99ca`) | `5a95a6f5` | `2026-10` |

Para gestionar el ciclo de vida de los modelos en el **Model Registry**, se adoptó el esquema vigente de *aliases* (`champion` para la versión activa en producción y `challenger` para el segundo mejor candidato en evaluación), en sustitución del esquema clásico de etapas rígidas (*Staging* / *Production*, deprecado desde MLflow 2.9). Este mecanismo permite desacoplar el código de consumo de las versiones numéricas específicas, ya que los servicios de inferencia invocan siempre una referencia fija por alias (por ejemplo, `models:/mibici-equipo10-p1-demanda@champion`). Para promover una nueva versión tras un reentrenamiento mensual con un nuevo corte de `datapackage.json`, se registra el nuevo artefacto en el catálogo, se valida su desempeño asignándole inicialmente el alias `challenger` y, una vez confirmado que supera las métricas del modelo vigente, se reasigna el puntero `champion` a la nueva versión mediante la interfaz de MLflow o con `MlflowClient().set_registered_model_alias()`. En caso de detectar degradación operativa o anomalías en producción, la reversión (*rollback*) es inmediata y atómica: basta con apuntar nuevamente el alias `champion` a la versión anterior verificada, sin necesidad de modificar el código de despliegue ni reentrenar artefactos.

Carga del modelo en producción por nombre y alias:

```python
import mlflow

modelo = mlflow.pyfunc.load_model("models:/mibici-equipo10-p1-demanda@champion")
print(modelo.metadata.signature)