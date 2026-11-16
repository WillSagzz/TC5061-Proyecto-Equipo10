# Versión alternativa del modelo

La versión alternativa se generó para P1 (demanda por estación y hora), el modelo que sirve por defecto la API del proyecto. Es el mismo `HistGradientBoostingRegressor` con pérdida de Poisson del `@champion`, con los mismos hiperparámetros (`learning_rate=0.1`, `max_leaf_nodes=127`), reentrenado con seis meses más de datos. Está registrada en el Model Registry de MLflow como la versión 3 del mismo modelo, `mibici-equipo10-p1-demanda`, con el alias `challenger`; la versión 2 es una corrida anterior del mismo reentrenamiento. Septiembre de 2026 no entra al reentrenamiento porque es el conjunto con el que se comparan las dos versiones. P2 y P3 conservan su `@champion`.

**Tabla 2.** Versiones de `mibici-equipo10-p1-demanda` en el Model Registry.

| | Versión 1 (`@champion`) | Versión 3 (`@challenger`) |
|---|---|---|
| Entrenamiento | Septiembre de 2024 a febrero de 2026 (18 meses) | Septiembre de 2024 a agosto de 2026 (24 meses) |
| Filas de entrenamiento | 3,998,020 | 5,489,700 |
| Corrida de origen | `P1-candidato-hgb_poisson-learning_rate=0.1-max_iter=200-max_leaf_nodes=127` | `P1-reentrenado-hasta-2026-09-01` |
