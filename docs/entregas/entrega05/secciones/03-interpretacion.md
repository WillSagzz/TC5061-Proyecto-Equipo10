# Interpretación del drift

## Interpretación por pregunta

| Pregunta | Variables con drift significativo | Implicación para el modelo |
| --- | ---- | --- |
| P1. Regresión de viajes | `station_id` (0.289) y `capacity` (0.105) | La demanda y sus rezagos son estables, pero el modelo puede desempeñarse peor en estaciones nuevas, retiradas o con capacidad cambiada. |
| P2. Riesgo de bicicletas/anclajes | `station_id` (0.287) y `capacity` (0.101) | `risk`, `arrivals...`, `departures...` y `net_flow...` son estables. El riesgo principal son estaciones nuevas o modificadas, cuyo historial puede ser insuficiente. |
| P3. Clustering de estaciones | `midday_pct` (p=0.00013), `afternoon_pct` (p=0.0089), `night_pct` (p=0.0024), `weekdays_pct` (p=0.0249), `balance_afternoon` (p=0.0043) | Cambió el patrón de horario y semanal de uso. Algunos clusters pueden quedar desactualizados y requerir reasignación o reentrenamiento. |

## Interpretación general

En el caso de `P1` y `P2` el drift está en la frecuencia relativa de cada estación dentro del dataset (`station_id`) y en la capacidad de estaciones, no en la demanda ni en el riesgo. El desempeño debería mantenerse en estaciones conocidas, pero puede degradarse en estaciones nuevas o modificadas.
Cabe resaltar que el drift de `capacity` es bajo, pues apenas supera el umbral predeterminado para detectar features con drift.

Se presenta un caso distinto con `P3`, para esta pregunta hay un cambio real en el uso. Los clusters basados en horarios podrían no representar bien septiembre de 2026 y conviene revisar la estabilidad de las asignaciones.

En un futuro se debe auditar `station_id` y `capacity`.  No se recomienda reentrenar P1 o P2 solo por estos reportes.
Para P3, hay que comparar los clusters originales con una nueva segmentación de septiembre.

Por las similitudes observadas entre los datos de referencia y actuales, se puede utilizar una ventana creciente para el entrenamiento del modelo. Al menos en el caso de `P1` y `P2`.
El factor que puede tener mayor impacto en la mejoría del modelo es el agregar las estaciones más nuevas a su entrenamiento para que adquiera conocimiento de estas.