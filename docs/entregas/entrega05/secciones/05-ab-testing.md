# A/B testing

Las dos versiones predijeron el mismo conjunto: las 280,760 filas del conjunto actual de P1 (Tabla 1), una por estación y hora de servicio de septiembre de 2026, mes que ninguna vio al entrenarse. Se comparan con las métricas con las que se eligió el `@champion`: el MAE, que es la principal, y el RMSE, en viajes por estación y hora. El resultado se separa en estaciones conocidas, las que ya operaban en el periodo de referencia, y nuevas, las que abrieron después, porque el drift se concentró en `station_id` y `capacity` (sección 3). La comparación está en la corrida `P1-ab-test-v1-vs-v3` del experimento `mibici-p1-demanda` de MLflow.

**Tabla 4.** Versión vigente (`@champion`, versión 1) y versión nueva (`@challenger`, versión 3) de P1 sobre septiembre de 2026.

| Estaciones | Filas | MAE vigente | MAE nueva | Diferencia de MAE | RMSE vigente | RMSE nueva |
|---|---:|---:|---:|---:|---:|---:|
| Todas | 280,760 | 1.0168 | 1.0042 | −0.0126 (−1.2 %) | 1.7613 | 1.7485 |
| Conocidas | 221,400 | 1.1158 | 1.1129 | −0.0029 (−0.3 %) | 1.9117 | 1.9021 |
| Nuevas | 59,360 | 0.6478 | 0.5991 | −0.0487 (−7.5 %) | 1.0208 | 0.9828 |

En todas las estaciones, el intervalo de confianza del 95 % de la diferencia de MAE va de −0.0151 a −0.0102 (*bootstrap* con 1,000 remuestreos de días completos).

**Debe quedar vigente la versión nueva.** Tiene menor MAE y menor RMSE en los tres grupos, y el intervalo de confianza de la diferencia queda completo por debajo de cero. La mejora está donde se detectó el drift: en las estaciones nuevas, que son el 21 % de las filas, el MAE baja 7.5 %, contra 0.3 % en las conocidas. Se entrenó con los meses de marzo a agosto de 2026, en los que ya operaban estaciones que la versión vigente nunca vio. Es el mismo modelo con las mismas variables, así que la API no cambia. Promoverla es mover el alias `champion` a ella y dejar la anterior como `challenger`.
