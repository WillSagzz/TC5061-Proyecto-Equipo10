# Recomendación sobre *Continuous Training*

**Sí conviene reentrenar el modelo, una vez al mes.** El drift de septiembre viene del crecimiento de la red: cambian `station_id` y `capacity`, porque operan 468 estaciones y el `@champion` se entrenó con 371 (sección 3). El A/B mostró que reentrenar con los meses recientes reduce el error: la versión 2 tiene menor MAE que la versión 1, con un intervalo de confianza que no incluye el cero, y la mayor mejora está en las estaciones nuevas (Tabla 4).

El disparador es el calendario. MiBici publica los viajes una vez al mes, y el mismo archivo trae las variables y la demanda real, así que ninguna señal de drift o de desempeño llega antes que ese archivo. Cada mes publicado inicia el ciclo que se siguió en esta entrega: el reporte de drift del mes (`make monitor-data` y `make drift`), el reentrenamiento con todos los meses anteriores, registrado como `@challenger`, y la comparación de las dos versiones sobre el mes nuevo. El costo es bajo: la corrida de reentrenamiento de P1 duró un minuto.

Reentrenar no implica promover. El alias `champion` se mueve solo si la versión nueva tiene menor MAE en el mes nuevo y el intervalo de confianza de la diferencia no incluye el cero, como ocurrió en septiembre. La versión anterior queda como `challenger`, y regresar a ella es mover el alias.

P2 tiene drift en las mismas dos variables, así que entra al mismo ciclo con su métrica principal, el F1 macro. P3 no tiene un valor real con el cual comparar dos versiones. Tiene drift en 5 de sus 8 variables y sus grupos se calcularon sin las estaciones que abrieron después de febrero de 2026, así que conviene recalcularlos y comparar la silueta con la del modelo vigente (0.173).
