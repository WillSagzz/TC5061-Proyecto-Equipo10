# Recomendación sobre *Continuous Training*

**Sí conviene reentrenar el modelo: cada mes, cuando MiBici publica datos nuevos, y dejando vigente la versión nueva solo si gana la comparación con la anterior.**

La evidencia apunta a una sola causa: la red crece. En P1 el drift está en `station_id` y `capacity`, porque el `@champion` se entrenó con 371 estaciones y en septiembre operaban 468; la demanda y las variables históricas no cambian (sección 3). El A/B dice lo mismo: la versión reentrenada mejora el MAE 7.5 % en las estaciones nuevas y 0.3 % en las conocidas (Tabla 4). En total baja 1.2 %, con un intervalo de confianza que no incluye el cero. El modelo sigue sirviendo para las estaciones que conoce. Lo que le falta es conocer las nuevas, y eso solo se resuelve reentrenando.

Con cada mes que publica MiBici se repetiría el ciclo de esta entrega:

1. Se descarga el mes y se reentrena el modelo con todos los meses anteriores a él. La versión nueva se registra en MLflow como `@challenger`. Reentrenar P1 tarda un minuto.
2. Las dos versiones predicen el mes nuevo, que ninguna vio, y se comparan como en la sección 5. El reporte de drift del mes acompaña a la comparación: dice qué variables cambiaron y, con eso, en qué grupos de estaciones conviene separar el resultado.
3. Si la versión nueva tiene menor MAE y el intervalo de confianza de la diferencia no incluye el cero, una persona mueve el alias `champion` a ella. La API carga al arrancar la versión que tiene ese alias, y regresar a la anterior es mover el alias de vuelta. Si la versión nueva no gana, la vigente se queda.

Hoy estos pasos se corren a mano. Los pasos 1 y 2 se pueden programar en un flujo de GitHub Actions, como el que el repositorio ya usa para sus chequeos. El paso 3 se deja a una persona porque la regla de promoción se ha aplicado una sola vez.

Las alternativas habituales son reentrenar solo cuando se detecta drift o solo cuando sube el error. Se descartan porque resuelven dos problemas que este proyecto no tiene. El primero es el costo: esperar una señal evita reentrenamientos caros, y aquí reentrenar cuesta un minuto. El segundo es la espera del valor real: el drift se usa como aviso mientras no se sabe cuánto se equivocó el modelo, pero el archivo mensual de MiBici trae juntas las variables y la demanda real. En cuanto se puede medir el drift, también se puede medir directamente si una versión reentrenada predice mejor.

Cada alternativa tiene además un defecto propio. El drift mide si cambiaron las variables, no si el modelo predice peor, y necesita un umbral. El del proyecto pide que cambie la mitad de las columnas, y el crecimiento de la red se concentra en dos de las once de P1, lejos de ese umbral. El error de un mes tampoco responde la pregunta, porque depende del mes además del modelo: hay meses más difíciles de predecir, y eso no se corrige reentrenando. Septiembre lo ilustra: el MAE del `@champion` subió de 0.908 a 1.017, y el de la versión reentrenada fue de 1.004. Lo que decide es la diferencia entre las dos versiones en el mismo mes, y eso es lo que mide el paso 2.

P2 usa las mismas variables de red y tuvo el mismo drift, así que sigue el mismo ciclo con su métrica, el F1 macro. P3 no tiene un valor real con el cual comparar versiones; sus grupos se formaron sin las estaciones que abrieron después de febrero de 2026 y conviene recalcularlos.
