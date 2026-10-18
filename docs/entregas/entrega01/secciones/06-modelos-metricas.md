# Modelos y métricas

Para evaluar la viabilidad técnica, se establece una estrategia de modelado incremental para cada pregunta de negocio. En primer lugar, se define un modelo base (*baseline*) de baja complejidad que se ejecuta y registra en MLflow Tracking como punto de referencia mínimo. Posteriormente, se proponen modelos candidatos con mayor capacidad para capturar relaciones no lineales en los datos. Finalmente, se seleccionan métricas de evaluación de la librería scikit-learn alineadas tanto con la estructura técnica de cada problema como con el impacto operativo de las decisiones en MiBici.

## P1: Predicción de viajes iniciados por estación y hora (Regresión)

### Modelo baseline

**`DummyRegressor` por estación y hora:** se entrena un `DummyRegressor` con `strategy="mean"` para cada combinación de estación y hora del día. Cada uno aprende el promedio de viajes de su estación a esa hora en el conjunto de entrenamiento y predice ese valor. Por ejemplo, si la estación 35 tuvo en promedio 11.6 viajes de 8:00 a 9:00, predice 11.6 viajes para cualquier día a esa hora. Para combinaciones que no aparecen en entrenamiento, se usa un `DummyRegressor` general con el promedio de todos los viajes. No usa ninguna otra variable, por lo que representa el patrón habitual de cada estación: los modelos candidatos deben superarlo para justificar su complejidad.

### Modelos candidatos

- **HistGradientBoostingRegressor o LightGBM:** algoritmo de árboles de decisión ensamblados de forma secuencial, diseñado específicamente para grandes volúmenes de datos tabulares. Es ideal para P1 porque maneja de forma nativa relaciones no lineales complejas (por ejemplo, el efecto combinado de ser viernes por la tarde en una estación de alta capacidad) y procesa millones de filas en pocos minutos.
- **Regresión Ridge:** modelo lineal sencillo que estima el número de viajes que iniciarán en cada estación y hora del día siguiente, a partir de los rezagos básicos de demanda (viajes a la misma hora del día anterior y de la semana anterior), el promedio histórico de la estación a esa hora y las variables de calendario. Las variables se estandarizan antes de entrenar. Se elige como primer candidato porque es rápido de entrenar sobre millones de registros, requiere un ajuste mínimo de hiperparámetros y es interpretable. Además, permite comprobar si incorporar la inercia reciente de cada estación mejora el pronóstico frente al *baseline*, que solo usa un promedio estático por estación y hora.

### Métricas de evaluación

- **Error absoluto medio (`mean_absolute_error`):** mide el promedio de la diferencia absoluta entre los viajes reales y los viajes predichos por el modelo. Se expresa exactamente en las mismas unidades del negocio (número de viajes/bicicletas por estación y hora), lo que permite al operador del sistema interpretar directamente cuántos viajes de desviación promedio tiene el pronóstico al dimensionar la flota activa diaria y planear ventanas de mantenimiento. Además, no se distorsiona de forma exagerada por un día atípico (como un evento masivo en la ciudad).
- **Raíz del error cuadrático medio (`root_mean_squared_error`):** mide la magnitud del error dando un peso mucho mayor a las desviaciones grandes antes de calcular el promedio. En la operación de MiBici, equivocarse por 1 viaje durante 10 horas seguidas es manejable, pero equivocarse por 15 viajes en una sola hora pico provoca el colapso inmediato de la estación. Esta métrica penaliza