# Recomendación sobre *Continuous Training*

<div class="pendiente">

**TODO (#59).** Primera frase de la sección: si se recomienda reentrenar o no, con la diferencia entre `@challenger` y `@champion` que midió el A/B y su intervalo, en todas las estaciones y en las nuevas (sección 5). Los demás `TODO` de la sección son cifras del A/B (#59) y de las variantes del drift (#55); el resto del texto se confirma contra ellas.

</div>

**Por qué envejece el modelo.** El drift no está en la demanda sino en la red: `departures`, `risk` y las variables históricas no cambian, y sí `station_id` y `capacity`, porque en septiembre operan 468 estaciones y el periodo de entrenamiento del `@champion` tiene 371 (sección 3). En su periodo de prueba, de junio a agosto, el `@champion` tuvo un MAE de 0.908 viajes en P1 y un F1 macro de 0.593 en P2; en septiembre obtiene TODO (#59) y TODO (#59).

**Qué dispara el reentrenamiento: el calendario.** MiBici publica los viajes una vez al mes, y el mismo archivo trae las variables y el valor real de la demanda. Con datos abiertos, ninguna señal de drift ni de desempeño llega antes que ese archivo, así que un disparador más fino no adelanta nada. Cada mes publicado inicia el mismo ciclo, con ese mes como periodo actual (`CURRENT_MONTH_DATE` en `config.py`):

1. `make monitor-data` y `make drift`: conjuntos y reporte de drift del mes.
2. `make retrain`: el mismo modelo con la partición recorrida un mes, registrado como `@challenger`. El costo es bajo: el `@champion` de P1 se entrenó en 2.4 minutos y el de P2 en 5.2.
3. `make ab-test`: las dos versiones predicen el mes nuevo.

**Cómo se promueve una versión.** Reentrenar no implica promover. El alias `champion` se mueve solo si se cumple la regla de decisión de `config.py`: el `@challenger` mejora la métrica principal (MAE en P1, F1 macro en P2), el intervalo de confianza del 95 % de la diferencia no incluye el cero y, en P2, el recall de `vaciarse` y de `llenarse` no baja más que la tolerancia. La versión anterior queda como `challenger`, y regresar a ella es mover el alias. Mientras no haya varios meses de historial, una persona aprueba el cambio con la tabla del A/B. El reporte de drift no decide la promoción, la explica; si declara drift del conjunto en P1 o P2 (la mitad de las variables o más), se revisan las variables antes de promover, porque el cambio ya no sería solo de estaciones.

**P3.** No entra al A/B porque no tiene un valor real con el cual comparar dos versiones. Sus grupos se definieron sin las estaciones que abrieron después, cerca de 100 de las 468 de septiembre, y 5 de sus 8 variables tienen drift (TODO (#55): si se sostiene contra septiembre de 2025). Se recomienda recalcular los grupos cuando más del 10 % de las estaciones activas no tenga perfil en la ventana de entrenamiento, condición que hoy ya se cumple, y aceptar la versión nueva comparando su silueta con la del modelo vigente (0.173).

**Límites de la evidencia.** La comparación usa un solo mes, así que no separa el efecto de las estaciones nuevas del de la temporada. La versión alternativa solo cambia la ventana de entrenamiento, con los mismos hiperparámetros. Las dos versiones predijeron las mismas filas de un mes ya ocurrido; no se repartieron peticiones reales entre ellas. Con dos o tres ciclos más se sabrá si la regla de promoción es demasiado estricta o demasiado laxa.
