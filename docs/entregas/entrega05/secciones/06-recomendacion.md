# Recomendación sobre *Continuous Training*

**Sí conviene reentrenar el modelo: cada mes, cuando MiBici publica datos nuevos, y dejando vigente la versión nueva solo si gana la comparación con la anterior.**

La razón es que la red crece. El `@champion` se entrenó con 371 estaciones y en septiembre operaban 468. Por eso el drift de P1 está en `station_id` y `capacity`, mientras que la demanda y las variables históricas no cambian (sección 3). El A/B lo confirma: la versión reentrenada mejora el MAE 7.5 % en las estaciones nuevas y 0.3 % en las conocidas (Tabla 4). El reporte no declara drift del conjunto en P1 porque esa regla cuenta columnas y pide que cambie la mitad, y el crecimiento de la red toca dos de once. Contar columnas no mide cuánto afecta un cambio al modelo; el A/B muestra que este sí lo afecta.

La mejora total es pequeña: 1.2 % del MAE, con un intervalo de confianza que no incluye el cero. Aun así conviene, porque reentrenar tarda un minuto y cada estación que abre es una más que la versión vigente no conoce.

Con cada mes que publica MiBici se repetiría el ciclo de esta entrega:

1. Se descarga el mes y se reentrena el modelo con todos los meses anteriores a él. La versión nueva se registra en MLflow como `@challenger`.
2. Las dos versiones predicen el mes nuevo, que ninguna vio, y se comparan como en la sección 5.
3. Si la versión nueva tiene menor MAE y el intervalo de confianza de la diferencia no incluye el cero, queda propuesta para ser la vigente. Si no gana, la vigente se queda.
4. Una persona revisa la comparación y aprueba el cambio moviendo el alias `champion`. La API carga al arrancar la versión que tiene ese alias, y regresar a la anterior es mover el alias de vuelta.

Hoy estos pasos los corre una persona en su computadora. La propuesta es programar los pasos 1 a 3 en un flujo de GitHub Actions, que el repositorio ya usa para sus chequeos, y que ese flujo revise si el portal ya publicó un mes nuevo. El paso 4 se queda manual por tres razones. La regla solo mira el MAE del mes: no ve si el archivo llegó incompleto ni si la versión nueva empeora en un grupo de estaciones, y el portal ya reemplazó una vez su catálogo de estaciones. El pronóstico lo usa quien opera el sistema, así que cada cambio debe tener un responsable. Y aprobar cuesta poco: ocurre a lo mucho una vez al mes y sin urgencia, porque los datos también llegan cada mes.

No hace falta decidir de antemano cada cuánto reentrenar: se intenta con cada mes, que es cuando hay datos nuevos, y la comparación decide si la versión cambia. Las alternativas habituales son reentrenar solo cuando se detecta drift o solo cuando sube el error. Las dos sirven para no reentrenar de más, y eso importa cuando reentrenar es caro; aquí no lo es. Además, ninguna mide lo que hay que saber, que es si una versión reentrenada predice mejor que la vigente. El drift dice que cambiaron las variables y se usa como aviso cuando el valor real tarda en llegar. El archivo mensual de MiBici trae juntas las variables y la demanda real, así que ese aviso no llega antes que la medición directa del paso 2. El error de un mes depende del mes además del modelo: hay meses más difíciles de predecir, y eso no se corrige reentrenando. En septiembre, por ejemplo, el MAE del `@champion` subió de 0.908 en su periodo de prueba a 1.017, y el de la versión reentrenada fue de 1.004.

El mismo ciclo aplica a P2, que comparte las variables y la fuente de datos, con el F1 macro como métrica. P3 queda fuera de este ciclo: es un agrupamiento sin valor real, así que no hay con qué comparar dos versiones, y su revisión es la que describe la sección 3.
