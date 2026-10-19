# Pipeline de ML

El diseño del *pipeline* de *Machine Learning* establece la secuencia de procesamiento que transforma los registros crudos de viajes y el catálogo de estaciones de MiBici en datos estructurados listos para entrenar y evaluar los modelos de regresión (P1), clasificación (P2) y *clustering* (P3). Para optimizar el desarrollo, las preguntas P1 y P2 comparten una misma base de conteos por estación y hora (P1 la usa directamente y P2 la agrega en intervalos de 3 horas), mientras que P3 consolida el comportamiento histórico a nivel de estación única. Todas las transformaciones se diseñan bajo el principio de estricto corte temporal, garantizando que ninguna variable utilice información posterior al momento de la predicción.

## Etapa 1: Ingesta, limpieza y homologación de datos

Esta etapa inicial es común para P1, P2 y P3 y se ejecuta de manera automatizada en el archivo `dataset.py` a partir de los archivos mensuales de viajes y el catálogo de estaciones:

- **Homologación:** estandarización de los nombres de columnas y corrección de la codificación de caracteres entre los distintos archivos mensuales, además de la conversión de las fechas de inicio y fin de viaje a un formato de tiempo uniforme.
- **Depuración:** eliminación de registros duplicados, viajes sin estación de origen o destino identificada, viajes con fechas inconsistentes y viajes atípicos que no representan traslados reales.
- **Cruce con catálogo de estaciones:** unión de los viajes limpios con el archivo de nomenclatura para incorporar la capacidad total de anclajes y las coordenadas geográficas (latitud y longitud), descartando registros asociados a estaciones dadas de baja o inexistentes en el catálogo oficial.
- **Almacenamiento:** guardado de la tabla limpia en formato Parquet y registro de su versión mediante un archivo manifiesto con conteo de filas y código identificador único.

## Etapa 2: Agregaciones y construcción de la tabla base

A partir de los viajes limpios, los datos se agrupan según la unidad de análisis de cada pregunta.

**Para P1 y P2 (nivel estación y ventana de tiempo):**

- Se agrupan los viajes por estación de origen y ventana de una hora para obtener el total de salidas por hora.
- Se agrupan los viajes por estación de destino y ventana de una hora para obtener el total de llegadas por hora.
- Se genera una malla completa de todas las estaciones activas por todas las horas del periodo de estudio, rellenando con cero aquellas horas en las que una estación no tuvo salidas o llegadas, asegurando así series de tiempo continuas sin huecos.
- Para P2, la malla horaria se agrega en 8 intervalos de 3 horas por día (0-3, 3-6, …, 21-24 h), sumando las salidas y las llegadas de cada estación en cada intervalo; la longitud del intervalo se define una sola vez en `config.py`.
- **Construcción de variables objetivo:** para P1, la variable a predecir es el conteo de salidas registradas en la hora futura correspondiente. Para P2, se calcula el flujo neto (llegadas menos salidas) del intervalo futuro, se compara contra la capacidad total de anclajes de la estación y se asigna una de tres clases: *riesgo de vaciarse* si el flujo es menor o igual a −40 % de la capacidad, *riesgo de llenarse* si es mayor o igual a +40 % y *sin riesgo* en cualquier otro caso, exactamente como se define en la sección 4 (Tabla 4).

**Para P3 (nivel estación única):**

- Utilizando únicamente el periodo histórico de entrenamiento, se agrupan todos los movimientos (cada viaje cuenta como una salida en su origen y una llegada en su destino) por estación para resumir su volumen, su distribución porcentual a lo largo de los periodos del día y entre días laborales y fines de semana, y su balance entre llegadas y salidas.
- Como es una tarea no supervisada, se construye una matriz donde cada fila representa una estación única caracterizada por sus hábitos de uso.

## Etapa 3: Ingeniería de características

En esta etapa se calculan las variables predictoras mediante funciones reutilizables y documentadas dentro del módulo `features.py`:

- **Para P1 y P2:** se generan variables de calendario (hora o intervalo, día de la semana, fin de semana), rezagos históricos (comportamiento de la estación horas o días antes) y promedios móviles recientes. Para evitar fuga de información del futuro (*data leakage*), toda variable histórica se calcula exclusivamente con datos cerrados antes de la hora (P1) o del intervalo (P2) en que se emite la predicción.
- **Para P3:** se calculan las cinco variables del perfil de cada estación (Tabla 4 de la sección 4): proporciones de movimientos por periodo del día y en días laborales, uso diario por anclaje y balances de la mañana y de la tarde. Posteriormente, todas las variables numéricas de P3 se estandarizan a una misma escala para evitar que las estaciones más grandes dominen el algoritmo de agrupación por simple volumen de viajes.

## Etapa 4: Partición de datos y entrenamiento de modelos

- **Para P1 y P2:** no se utiliza partición aleatoria, sino un corte estrictamente cronológico (septiembre de 2024 a febrero de 2026 para entrenamiento, marzo a mayo de 2026 para validación y junio a agosto de 2026 para prueba, según las fechas de corte de `config.py`), garantizando que el modelo siempre aprenda del pasado para predecir el futuro. En P2, adicionalmente, se contempla el manejo del desbalance de clases durante el entrenamiento (pesos por clase), dado que los intervalos con riesgo de vaciarse o llenarse son mucho menos frecuentes que los intervalos sin riesgo.
- **Para P3:** se entrena el algoritmo de agrupación sobre el perfil consolidado de las estaciones, evaluando distintos números de grupos para encontrar una segmentación interpretable.

## Etapa 5: Evaluación y registro en MLflow Tracking

- Los modelos base y candidatos se ejecutan desde el código del repositorio (`train.py`) y se evalúan sobre el conjunto de prueba temporal (en P1 y P2) o mediante métricas de cohesión interna y proyección en el mapa de Guadalajara (en P3), con las métricas de la sección 6.
- Cada ejecución registra automáticamente en MLflow Tracking los parámetros de reproducción (rango de fechas, versión de datos, fechas de corte, semilla aleatoria e hiperparámetros), las métricas con el conjunto como prefijo, las etiquetas de equipo, pregunta y *commit*, y al menos un artefacto (sección 8).

## Catálogo de variables a construir

**Tabla 5.** Variables para P1 y P2. En P2 las ventanas de "hora" se sustituyen por intervalos de 3 horas y los rezagos se expresan en intervalos (1 intervalo, 8 intervalos = 24 h y 56 intervalos = 7 días).

| Variable | Descripción | Objetivo en MiBici |
|----------|-------------|--------------------|
| Hora del día (P1) o intervalo del día (P2) | Transformación de la hora (0 a 23) o del intervalo de 3 horas (0 a 7). | La movilidad urbana sigue ciclos diarios muy marcados (horas pico de entrada/salida laboral y escolar frente a madrugadas sin actividad). |
| Día de la semana e indicador de fin de semana | Número del día de la semana (lunes a domingo) y variable binaria que distingue sábado y domingo de los días laborales. | El patrón de viajes cambia drásticamente entre días de oficina y fines de semana. |
| Salidas registradas 1 h, 24 h y 7 días antes | Cantidad de viajes que iniciaron en esa misma estación exactamente 1 hora antes, el día anterior a la misma hora y la semana anterior a la misma hora. | Es el predictor directo más fuerte del volumen de viajes: captura la inercia inmediata de la estación y su hábito semanal. |
| Promedio móvil de 3 h | Promedio de viajes iniciados en la estación durante las 3 horas cerradas previas al momento de predicción. | Suaviza variaciones bruscas de una sola hora y refleja si la demanda de la estación está acelerándose o desacelerándose en ese turno. |
| Llegadas y desbalance neto (1 h y 24 h antes) | Cantidad de bicicletas que llegaron y diferencia neta entre llegadas y salidas en esa estación 1 hora antes y 24 horas antes. | Permite detectar si la estación ya viene sufriendo una pérdida o acumulación rápida de bicicletas en la hora inmediata anterior, o si suele saturarse a esa hora del día. |
| Capacidad total de anclajes | Número total de espacios físicos instalados en la estación según el catálogo oficial de nomenclatura. | En P1 acota el volumen máximo que puede despachar una estación y en P2 determina qué tan sensible es la estación a quedarse vacía o llena ante pocos viajes. |

**Tabla 6.** Variables para P3 (las mismas cinco de la Tabla 4 de la sección 4, calculadas sobre los movimientos de la estación: cada viaje es una salida en su origen y una llegada en su destino).

| Variable | Descripción | Objetivo en MiBici |
|----------|-------------|--------------------|
| Porcentaje de movimientos por periodo del día | Porcentaje del total de movimientos de la estación que ocurre en la mañana (6-10 h), el mediodía (10-14 h), la tarde (14-19 h) y la noche (19-6 h). Son cuatro variables que suman 100 %. | Permite agrupar estaciones con ritmos horarios similares independientemente de si son estaciones grandes o pequeñas en volumen total. |
| Porcentaje de movimientos en días laborales | Movimientos de lunes a viernes divididos entre el total de movimientos de la estación. | Separa las estaciones de uso cotidiano (trabajo y escuela) de las de uso recreativo de fin de semana. |
| Uso diario por anclaje | Promedio de movimientos diarios de la estación dividido entre su número de anclajes según el catálogo. | Mide el nivel de presión física de la infraestructura: detecta estaciones pequeñas con altísima demanda que requieren ampliación prioritaria de anclajes por parte de planeación urbana. |
| Balance de la mañana | (llegadas − salidas) ÷ (llegadas + salidas) con los movimientos de 5:00 a 12:00; un valor negativo indica que salen más bicicletas de las que llegan. | Identifica el rol funcional de la estación: las zonas habitacionales expulsan viajes por la mañana y las zonas de oficinas y escuelas los reciben. |
| Balance de la tarde | El mismo cociente con los movimientos de 16:00 a 20:00. | Complementa al anterior: una estación receptora por la mañana y emisora por la tarde confirma su vocación de destino laboral. |
| Latitud y longitud | Coordenadas geográficas de cada estación. | No se introducen al algoritmo para no forzar grupos por simple cercanía de calles, sino que se usan al final para proyectar los grupos en el mapa de Guadalajara. |
