# Pipeline de ML

El diseño del *pipeline* de *Machine Learning* establece la secuencia de procesamiento que transforma los registros crudos de viajes y el catálogo de estaciones de MiBici en datos estructurados listos para entrenar y evaluar los modelos de regresión (P1), clasificación (P2) y *clustering* (P3). Para optimizar el desarrollo, las preguntas P1 y P2 comparten una misma base de procesamiento horario por estación, mientras que P3 consolida el comportamiento histórico a nivel de estación única. Todas las transformaciones se diseñan bajo el principio de estricto corte temporal, garantizando que ninguna variable utilice información posterior al momento de la predicción.

## Etapa 1: Ingesta, limpieza y homologación de datos

Esta etapa inicial es común para P1, P2 y P3 y se ejecuta de manera automatizada en el archivo `dataset.py` a partir de los archivos mensuales de viajes y el catálogo de estaciones:

- **Homologación:** estandarización de los nombres de columnas y corrección de la codificación de caracteres entre los distintos archivos mensuales, además de la conversión de las fechas de inicio y fin de viaje a un formato de tiempo uniforme.
- **Depuración:** eliminación de registros duplicados, viajes sin estación de origen o destino identificada, viajes con fechas inconsistentes y viajes atípicos que no representan traslados reales.
- **Cruce con catálogo de estaciones:** unión de los viajes limpios con el archivo de nomenclatura para incorporar la capacidad total de anclajes y las coordenadas geográficas (latitud y longitud), descartando registros asociados a estaciones dadas de baja o inexistentes en el catálogo oficial.
- **Almacenamiento:** guardado de la tabla limpia en formato Parquet y registro de su versión mediante un archivo manifiesto con conteo de filas y código identificador único.

## Etapa 2: Agregaciones y construcción de la tabla base

A partir de los viajes limpios, los datos se agrupan según la unidad de análisis de cada pregunta.

**Para P1 y P2 (nivel estación y hora):**

- Se agrupan los viajes por estación de origen y ventana de una hora para obtener el total de salidas por hora.
- Se agrupan los viajes por estación de destino y ventana de una hora para obtener el total de llegadas por hora.
- Se genera una malla completa de todas las estaciones activas por todas las horas del periodo de estudio, rellenando con cero aquellas horas en las que una estación no tuvo salidas o llegadas, asegurando así series de tiempo continuas sin huecos.
- **Construcción de variables objetivo:** para P1, la variable a predecir es el conteo de salidas registradas en la hora futura correspondiente. Para P2, se calcula la diferencia neta entre llegadas y salidas en esa hora futura, se compara contra la capacidad total de anclajes de la estación y se asigna la etiqueta binaria "En riesgo" cuando dicho desbalance iguala o supera el 40 % de su capacidad, o "Sin riesgo" en caso contrario.

**Para P3 (nivel estación única):**

- Utilizando únicamente el periodo histórico de entrenamiento, se agrupan todos los viajes por estación para resumir su volumen total de salidas, su volumen de llegadas y su distribución porcentual a lo largo de las horas del día y los días de la semana.
- Como es una tarea no supervisada, se construye una matriz donde cada fila representa una estación única caracterizada por sus hábitos de uso.

## Etapa 3: Ingeniería de características

En esta etapa se calculan las variables predictoras mediante funciones reutilizables y documentadas dentro del módulo `features.py`:

- **Para P1 y P2:** se generan variables de calendario (hora, día, fin de semana), rezagos históricos (comportamiento de la estación horas o días antes) y promedios móviles recientes. Para evitar fuga de información del futuro (*data leakage*), toda variable histórica se calcula exclusivamente con datos cerrados antes de la hora en que se emite la predicción.
- **Para P3:** se calculan proporciones relativas de uso por franja horaria, rotación por anclaje y balance entre salidas y llegadas. Posteriormente, todas las variables numéricas de P3 se estandarizan a una misma escala para evitar que las estaciones más grandes dominen el algoritmo de agrupación por simple volumen de viajes.

## Etapa 4: Partición de datos y entrenamiento de modelos

- **Para P1 y P2:** no se utiliza partición aleatoria, sino un corte estrictamente cronológico (los primeros meses del periodo para entrenamiento, un bloque intermedio para validación y el bloque más reciente para prueba), garantizando que el modelo siempre aprenda del pasado para predecir el futuro. En P2, adicionalmente, se contempla el manejo del desbalance de clases durante el entrenamiento, dado que las horas críticas son menos frecuentes que las horas de operación normal.
- **Para P3:** se entrena el algoritmo de agrupación sobre el perfil consolidado de las estaciones, evaluando distintos números de grupos para encontrar una segmentación interpretable.

## Etapa 5: Evaluación y registro en MLflow Tracking

- Los modelos base y candidatos se ejecutan desde el código del repositorio y se evalúan sobre el conjunto de prueba temporal (en P1 y P2) o mediante métricas de cohesión interna y proyección en el mapa de Guadalajara (en P3).
- Cada ejecución registra automáticamente en MLflow Tracking los parámetros de reproducción (rango de fechas, versión de datos, semilla aleatoria e hiperparámetros).

## Catálogo de variables a construir

**Tabla 5.** Variables para P1 y P2.

| Variable | Descripción | Objetivo en MiBici |
|----------|-------------|--------------------|
| Hora del día | Transformación de la hora (0 a 23). | La movilidad urbana sigue ciclos diarios muy marcados (horas pico de entrada/salida laboral y escolar frente a madrugadas sin actividad). |
| Día de la semana e indicador de fin de semana | Número del día de la semana (lunes a domingo) y variable binaria que distingue sábado y domingo de los días laborales. | El patrón de viajes cambia drásticamente entre días de oficina y fines de semana. |
| Salidas registradas 1 h, 24 h y 7 días antes | Cantidad de viajes que iniciaron en esa misma estación exactamente 1 hora antes, el día anterior a la misma hora y la semana anterior a la misma hora. | Es el predictor directo más fuerte del volumen de viajes: captura la inercia inmediata de la estación y su hábito semanal. |
| Promedio móvil de 3 h | Promedio de viajes iniciados en la estación durante las 3 horas cerradas previas al momento de predicción. | Suaviza variaciones bruscas de una sola hora y refleja si la demanda de la estación está acelerándose o desacelerándose en ese turno. |
| Llegadas y desbalance neto (1 h y 24 h antes) | Cantidad de bicicletas que llegaron y diferencia neta entre llegadas y salidas en esa estación 1 hora antes y 24 horas antes. | Permite detectar si la estación ya viene sufriendo una pérdida o acumulación rápida de bicicletas en la hora inmediata anterior, o si suele saturarse a esa hora del día. |
| Capacidad total de anclajes | Número total de espacios físicos instalados en la estación según el catálogo oficial de nomenclatura. | En P1 acota el volumen máximo que puede despachar una estación y en P2 determina qué tan sensible es la estación a quedarse vacía o llena ante pocos viajes. |

**Tabla 6.** Variables para P3.

| Variable | Descripción | Objetivo en MiBici |
|----------|-------------|--------------------|
| Proporción de viajes por franja del día y en fin de semana | Porcentaje del total de viajes de la estación que ocurre en el bloque matutino (6-10 h), mediodía (10-14 h), vespertino (14-19 h) y nocturno (19-6 h). | Permite agrupar estaciones con ritmos horarios similares independientemente de si son estaciones grandes o pequeñas en volumen total. |
| Índice de atracción y emisión de viajes matutinos | Proporción de salidas frente al total de movimientos (salidas más llegadas) de la estación durante las horas pico de la mañana. | Identifica el rol funcional de la estación en la ciudad y puede ayudar a distinguir zonas habitacionales de zonas de oficinas. |
| Tasa de rotación diaria por anclaje | Promedio de viajes diarios atendidos por la estación dividido entre su número total de anclajes instalados. | Mide el nivel de presión física de la infraestructura: detecta estaciones pequeñas con altísima demanda que requieren ampliación prioritaria de anclajes por parte de planeación urbana. |
| Latitud y longitud | Coordenadas geográficas de cada estación. | No se introducen al algoritmo para no forzar grupos por simple cercanía de calles, sino que se usan al final para proyectar los grupos en el mapa de Guadalajara. |
