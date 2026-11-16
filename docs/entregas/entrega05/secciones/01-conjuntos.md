# Conjuntos de referencia y actual

El drift se mide comparando dos periodos de los viajes de MiBici, definidos en `mibici/config.py` y construidos por `mibici/monitoring.py` (`make monitor-data`) para las tres preguntas del proyecto:

- **Referencia: del 1 de septiembre de 2024 al 28 de febrero de 2026** (`REFERENCE_START` y `REFERENCE_END`). Son los 18 meses con los que se entrenaron los modelos registrados como `@champion`, es decir, los datos que el modelo conoce. Los meses de validación (marzo a mayo de 2026) y de prueba (junio a agosto de 2026) quedan fuera porque los modelos no se entrenaron con ellos.
- **Actual: septiembre de 2026** (`CURRENT_START` y `CURRENT_END`). Es el primer mes posterior a todos los datos que el proyecto usó para entrenar, validar y probar, y estaba reservado en `config.py` para este fin: ningún modelo lo vio, así que representa lo que recibirían en producción. Al ser un mes completo, cubre todos los días de la semana y todas las horas de servicio. Son 471,643 viajes limpios de los 490,056 del archivo `datos_abiertos_2026_09.csv`, guardados en `data/processed/meses_produccion.parquet`, aparte de la versión de datos congelada en `datapackage.json`.

Entre el fin de la referencia y el periodo actual pasan seis meses, por lo que la comparación mide el cambio acumulado desde que se entrenó el modelo vigente.

**Tabla 1.** Conjuntos de referencia y actual por pregunta.

| Pregunta | Una fila es | Variables | Referencia (2024-09 a 2026-02) | Actual (2026-09) |
|---|---|---|---|---|
| P1 · Demanda | una estación en una hora | 10 y el objetivo `departures` | 4,005,940 filas | 280,760 filas |
| P2 · Riesgo | una estación en un intervalo de 3 horas | 13 y el objetivo `risk` | 1,602,376 filas | 112,304 filas |
| P3 · Perfiles | una estación | 8, sin objetivo | 371 estaciones | 468 estaciones |

En P1 y P2 las variables históricas de la primera semana de septiembre se calculan con el cierre de agosto, para que el conjunto actual no tenga valores faltantes artificiales. En P3 el perfil de uso de cada estación se recalcula con los movimientos de cada periodo. El reporte de Evidently usa una muestra determinista de hasta 50,000 filas por conjunto en P1 y P2 (`EVIDENTLY_SAMPLE_SIZE`) y todas las estaciones en P3.
