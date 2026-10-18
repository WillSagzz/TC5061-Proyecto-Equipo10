# Modelos y métricas

Para evaluar la viabilidad técnica se establece una estrategia de modelado incremental para cada pregunta de negocio. En primer lugar, se define un modelo base (*baseline*) de baja complejidad que se ejecuta y registra en MLflow Tracking como punto de referencia mínimo. Posteriormente, se proponen modelos candidatos con mayor capacidad para capturar relaciones no lineales en los datos. Finalmente, se seleccionan métricas de evaluación de la librería scikit-learn alineadas tanto con la estructura técnica de cada problema como con el impacto operativo de las decisiones en MiBici.

## P1: Predicción de demanda de viajes por estación y hora (Regresión)

### Modelo baseline

**Ridge:** modelo lineal sencillo que estima el número de viajes que iniciarán en la siguiente hora a partir de los rezagos básicos de demanda (viajes hace 1 hora, hace 24 horas y hace 7 días) y la hora del día. Se elige como *baseline* porque es rápido de entrenar sobre millones de registros, requiere un ajuste mínimo de hiperparámetros y ofrece una línea base interpretable que supera a un simple promedio estático al incorporar la inercia reciente de cada estación.

### Modelos candidatos

- **HistGradientBoostingRegressor o LightGBM:** algoritmo de árboles de decisión ensamblados de forma secuencial, diseñado específicamente para grandes volúmenes de datos tabulares. Es ideal para P1 porque maneja de forma nativa relaciones no lineales complejas (por ejemplo, el efecto combinado de ser viernes por la tarde en una estación de alta capacidad) y procesa millones de filas en pocos segundos.
- **RandomForestRegressor:** ensamble de múltiples árboles independientes que reduce la varianza de las predicciones y es robusto frente a valores atípicos en horas de demanda inusual.

### Métricas de evaluación

- **Error absoluto medio (`mean_absolute_error`):** mide el promedio de la diferencia absoluta entre los viajes reales y los predichos. Se expresa en las mismas unidades del negocio (viajes o bicicletas por hora), lo que permite al operador interpretar directamente cuántas bicicletas de desviación promedio tiene el pronóstico al dimensionar la flota activa diaria y planear ventanas de mantenimiento. Además, no se distorsiona de forma exagerada por un día atípico (como un evento masivo en la ciudad).
- **Raíz del error cuadrático medio (`root_mean_squared_error`):** mide la magnitud del error dando un peso mucho mayor a las desviaciones grandes. En la operación de MiBici, equivocarse por 1 bicicleta durante 10 horas seguidas es manejable, pero equivocarse por 15 bicicletas en una sola hora pico provoca el colapso inmediato de la estación. Esta métrica penaliza severamente a los modelos que fallan por márgenes amplios en las horas de mayor demanda.

## P2: Riesgo de estación vacía o sin anclajes disponibles (Clasificación)

### Modelo baseline

**LogisticRegression:** clasificador lineal que estima la probabilidad de que una estación entre en estado crítico. Con `class_weight="balanced"` se corrige la desproporción entre horas normales y horas críticas; es rápida de entrenar y entrega probabilidades calibradas, necesarias para ordenar las estaciones por urgencia de atención.

### Modelos candidatos

- **HistGradientBoostingClassifier o XGBoost:** capaz de detectar umbrales abruptos donde la combinación de pérdidas acumuladas en las últimas 3 horas, la hora pico y una capacidad reducida de anclajes disparan el riesgo de vaciado o saturación.
- **RandomForestClassifier:** con `class_weight="balanced_subsample"` permite capturar interacciones complejas entre variables temporales y físicas de la estación, manteniendo estabilidad ante el desbalance de clases.

### Métricas de evaluación

- **Sensibilidad o exhaustividad (`recall_score`):** del total de estaciones que realmente sufrieron un desbalance crítico, qué porcentaje logró detectar el modelo con anticipación. Para el equipo de rebalanceo, el error más costoso es el falso negativo (no enterarse de que una estación se iba a quedar sin bicicletas o sin anclajes libres, dejando a los usuarios sin servicio). Priorizar una alta sensibilidad asegura que la gran mayoría de las emergencias reales sean detectadas a tiempo.
- **F1 (`f1_score`):** equilibrio armónico entre la sensibilidad (detectar los riesgos reales) y la precisión (que las alertas emitidas sean verdaderas). Si el modelo alertara riesgo en todas las estaciones para no fallar ninguna, las camionetas de rebalanceo harían viajes innecesarios gastando combustible y tiempo. El puntaje F1 evalúa que el modelo detecte las estaciones críticas sin saturar al equipo operativo con falsas alarmas.

## P3: Agrupación de estaciones por patrones de uso (Clustering)

### Modelo baseline

**KMeans:** algoritmo que agrupa las estaciones en un número predefinido de perfiles (evaluando inicialmente entre 4 y 5 grupos funcionales) minimizando la variación interna de los hábitos de uso estandarizados. Es el estándar de referencia en segmentación no supervisada, converge rápidamente y genera centros de grupo fáciles de interpretar como perfiles urbanos (residencial, oficinas, recreativo, baja demanda).

### Modelos candidatos

- **AgglomerativeClustering:** construye una jerarquía de grupos de abajo hacia arriba sin depender de una inicialización aleatoria de centros. Permite a planeación urbana visualizar mediante un dendrograma cómo se relacionan las estaciones a distintas escalas.
- **GaussianMixture:** a diferencia de K-Means, permite formar grupos de distinta dispersión y asigna a cada estación un grado de pertenencia probabilística a cada perfil funcional (útil para estaciones de uso mixto comercial-habitacional).

### Métricas de evaluación

- **Coeficiente de silueta (`silhouette_score`):** mide qué tan parecidas son las estaciones dentro de un mismo grupo en comparación con las estaciones de los grupos vecinos. Garantiza al área de planeación urbana y expansión que las categorías de estaciones descubiertas tienen hábitos de movilidad internamente consistentes y claramente diferenciados entre sí antes de usarlos para diseñar políticas de infraestructura por zona.
- **Índice de Davies-Bouldin (`davies_bouldin_score`):** mide la relación promedio entre la dispersión interna de cada grupo y la distancia que separa a los distintos grupos entre sí. Funciona como criterio objetivo de desempate para elegir el número óptimo de grupos de estaciones en Guadalajara, evitando crear demasiados perfiles redundantes que dificulten la toma de decisiones territoriales.
