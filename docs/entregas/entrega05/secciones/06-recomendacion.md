# Recomendación sobre *Continuous Training*

**Sí conviene reentrenar el modelo, una vez al mes, y dejar vigente la versión nueva solo cuando gane la comparación con la anterior.**

La evidencia apunta a una sola causa: la red creció. En el reporte de drift de P1 solo cambian `station_id` y `capacity`, porque el `@champion` se entrenó con 371 estaciones y en septiembre operaban 468; la demanda y las variables históricas no cambian (sección 3). El A/B dice lo mismo: la versión reentrenada mejora 7.5 % en las estaciones nuevas y 0.3 % en las conocidas (Tabla 4). El modelo sigue sirviendo para las estaciones que conoce. Lo que le falta es conocer las nuevas, y eso solo se resuelve reentrenando.

La mejora total es pequeña, de 1.2 % del MAE, pero el intervalo de confianza confirma que es real y reentrenar cuesta un minuto de cómputo. Mientras la red siga abriendo estaciones, el beneficio se repite. Se hace cada mes porque MiBici publica los datos una vez al mes: antes no hay con qué reentrenar.

El reporte de drift sí detectó el cambio: marcó `station_id` y `capacity`. Lo que no se activó fue la regla de drift del conjunto, que pide que cambie la mitad de las columnas, y en P1 cambiaron 2 de 11. Esa regla responde a otro tipo de cambio. Seis de las 11 columnas de P1 se calculan con los viajes, así que un cambio en la demanda las movería juntas y sí la activaría. El crecimiento de la red solo toca dos columnas y queda por debajo del umbral, aunque reentrenar ayuda. Por eso el reentrenamiento mensual no espera a esa alerta y el reporte se lee variable por variable. Si un mes la regla se activa, es señal de que cambió la demanda y no solo la red, y conviene revisar el modelo antes de dejar vigente una versión nueva.

El MAE de septiembre tampoco habría bastado como alerta. El del `@champion` subió de 0.908 en su periodo de prueba a 1.017, pero el de la versión reentrenada fue de 1.004, así que casi toda esa subida se debe al mes y no se corrige reentrenando.

La comparación se hizo con un solo mes, y por eso la versión nueva no debe reemplazar a la vigente de forma automática. Cada mes se comparan las dos con los datos nuevos, como en la sección 5, y la nueva queda vigente solo si su MAE es menor y el intervalo de confianza de la diferencia no incluye el cero. Si después falla, se regresa a la anterior moviendo el alias `champion`.

Lo mismo aplica a P2, que tiene drift en las mismas dos variables. En P3 conviene recalcular los grupos, porque se formaron sin las estaciones que abrieron después de febrero de 2026.
