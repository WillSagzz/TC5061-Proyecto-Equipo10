# Definición de los datos y partición

## Rango de fechas propuesto

Los datos de MiBici abarcan desde diciembre de 2014 hasta septiembre de 2026. Sin embargo, usar todo el historial no garantiza mejores resultados. Los hábitos de movilidad cambian con el tiempo: cambia el comportamiento ciudadano, se abren estaciones nuevas, cambia la capacidad de otras, se modifican vías y carreteras, etc. Por eso, los patrones de hace varios años pueden no representar el comportamiento actual de nuestro intervalo de estudio. Además, los años 2020 y 2021 están alterados por la pandemia e introducen patrones anómalos. Por estas razones se selecciona un periodo de estudio acotado: el rango escogido es el más reciente, correspondiente a los últimos 24 meses, desde **septiembre de 2024 hasta agosto de 2026**. Esto permite capturar patrones estacionales de datos actuales, con los que estamos más familiarizados, y evitar utilizar estaciones que hayan desaparecido.

## Fuente de los archivos

El portal donde se alojan los datos de MiBici es <https://mibici.net/es/datos-abiertos/>. Los viajes se publican mensualmente; la Tabla 1 lista los 24 archivos que se utilizarán. Además se utiliza el catálogo de estaciones (nomenclatura) alojado en <https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv>.

**Tabla 1.** Archivos mensuales de viajes utilizados en el proyecto.

| Mes | Archivo |
|-----|---------|
| 2024-09 | <https://www.mibici.net/site/assets/files/89736/datos_abiertos_2024_09.csv> |
| 2024-10 | <https://www.mibici.net/site/assets/files/91758/datos_abiertos_2024_10.csv> |
| 2024-11 | <https://www.mibici.net/site/assets/files/93516/datos_abiertos_2024_11.csv> |
| 2024-12 | <https://www.mibici.net/site/assets/files/94962/datos_abiertos_2024_12.csv> |
| 2025-01 | <https://www.mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv> |
| 2025-02 | <https://www.mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv> |
| 2025-03 | <https://www.mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv> |
| 2025-04 | <https://www.mibici.net/site/assets/files/103813/datos_abiertos_2025_04.csv> |
| 2025-05 | <https://www.mibici.net/site/assets/files/105513/datos_abiertos_2025_05.csv> |
| 2025-06 | <https://www.mibici.net/site/assets/files/107088/datos_abiertos_2025_06.csv> |
| 2025-07 | <https://www.mibici.net/site/assets/files/108898/datos_abiertos_2025_07.csv> |
| 2025-08 | <https://www.mibici.net/site/assets/files/111021/datos_abiertos_2025_08.csv> |
| 2025-09 | <https://www.mibici.net/site/assets/files/113216/datos_abiertos_2025_09.csv> |
| 2025-10 | <https://www.mibici.net/site/assets/files/115104/datos_abiertos_2025_10.csv> |
| 2025-11 | <https://www.mibici.net/site/assets/files/116773/datos_abiertos_2025_11.csv> |
| 2025-12 | <https://www.mibici.net/site/assets/files/118153/datos_abiertos_2025_12.csv> |
| 2026-01 | <https://www.mibici.net/site/assets/files/120163/datos_abiertos_2026_01.csv> |
| 2026-02 | <https://www.mibici.net/site/assets/files/121761/datos_abiertos_2026_02.csv> |
| 2026-03 | <https://www.mibici.net/site/assets/files/123236/datos_abiertos_2026_03.csv> |
| 2026-04 | <https://www.mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv> |
| 2026-05 | <https://www.mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv> |
| 2026-06 | <https://www.mibici.net/site/assets/files/127734/datos_abiertos_2026_06.csv> |
| 2026-07 | <https://www.mibici.net/site/assets/files/129006/datos_abiertos_2026_07.csv> |
| 2026-08 | <https://www.mibici.net/site/assets/files/130362/datos_abiertos_2026_08.csv> |

## Tamaño real obtenido

- **Meses.** El conjunto consolidado comprende la totalidad de los 24 meses analizados, del 1 de septiembre de 2024 al 31 de agosto de 2026.
- **Número de viajes.** Durante este periodo se registró un volumen acumulado de **8,943,941 viajes**, con un promedio de 372,664 viajes mensuales.
- **Número de estaciones.** En el catálogo oficial de estaciones (nomenclatura) se identificaron **484** estaciones; sin embargo, en el periodo estudiado solo **472** estaciones registran actividad. Al cruzar ambas fuentes:
  - De las 472 estaciones con actividad, **470** están en la nomenclatura.
  - Las **2** estaciones que registran viajes pero no están en la nomenclatura son los ID **405** y **407**. Al no contar con información sobre ellas (capacidad y coordenadas), sus registros se excluyen del proyecto en la etapa de limpieza.
  - **14** estaciones están en la nomenclatura pero no tienen registro de viajes en el periodo estudiado.

## Variables originales

**Tabla 2.** Variables originales de los archivos de MiBici que se utilizan en el proyecto.

| Variable original | Archivo | Uso |
|-------------------|---------|-----|
| `Viaje_Id` | `datos_abiertos_AAAA_MM.csv` | Detección de registros duplicados. |
| `Inicio_del_viaje` | `datos_abiertos_AAAA_MM.csv` | Fecha y hora de inicio del viaje. |
| `Fin_del_viaje` | `datos_abiertos_AAAA_MM.csv` | Fecha y hora de fin del viaje. |
| `Origen_Id` | `datos_abiertos_AAAA_MM.csv` | Estación de la que sale la bicicleta (inicio del viaje). |
| `Destino_Id` | `datos_abiertos_AAAA_MM.csv` | Estación a la que llega la bicicleta (fin del viaje). |
| `id` | `nomenclatura_2026_09.csv` | Asocia el número de `Origen_Id` y `Destino_Id` con su nombre de la nomenclatura o `name` (GDL-001). |
| `name` | `nomenclatura_2026_09.csv` | Indica el municipio. |
| `latitude`, `longitude` | `nomenclatura_2026_09.csv` | Localizar las estaciones en el mapa. |
| `dpcapacity` | `nomenclatura_2026_09.csv` | Capacidad de anclajes de cada estación. |

## Variables objetivo y cómo se construyen

**Tabla 3.** Construcción de la variable objetivo de cada pregunta.

| Pregunta | Variable objetivo | Construcción |
|----------|-------------------|--------------|
| **P1. Regresión.** ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Viajes iniciados (entero). | Sumatoria de los viajes iniciados en cada estación durante cada hora. La hora de inicio de cada viaje se redondea hacia abajo a la hora completa (por ejemplo, 08:47 pasa a 08:00) y se agrupan los viajes por estación de origen y hora. Las combinaciones de estación y hora sin viajes se registran como cero. |
| **P2. Clasificación.** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Riesgo (categórica) con tres clases: riesgo de vaciarse, riesgo de llenarse, sin riesgo. | El día se divide en 8 intervalos de 3 horas y cada viaje se asigna dos veces: como salida de su estación de origen y como llegada de su estación de destino, cada una en el intervalo que corresponda. Se suman las salidas y las llegadas de cada estación en cada intervalo y se calcula el flujo neto (llegadas menos salidas); las combinaciones sin viajes se registran como cero. El flujo neto se compara con el 40 % de la capacidad de la estación: (1) *riesgo de vaciarse* si el flujo es menor o igual a −40 % de la capacidad; (2) *riesgo de llenarse* si es mayor o igual a +40 %; (3) *sin riesgo* si está estrictamente entre ambos umbrales. |
| **P3. Clustering.** ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | No posee una variable objetivo. | |

## Condiciones de uso de MiBici y tratamiento de datos personales

Los datos se obtienen del portal de datos abiertos de MiBici, sistema administrado por la **Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara (AMIM)**. El portal no publica una licencia para los archivos. Sin embargo, el Reglamento Interno en materia de Transparencia de la AMIM establece que la Agencia gestiona su información bajo una política de datos abiertos que favorece la creación de conocimiento (art. 6, fr. IV), y que la información que genera es pública y de libre acceso, salvo la reservada o confidencial (art. 18). Con base en ello, los datos se usarán solo con fines académicos, no se redistribuirán y no se utilizarán directa o indirectamente para intentar identificar usuarios.

El mismo reglamento considera confidencial la información con datos personales de una persona identificada o identificable (art. 21). Los archivos de viajes incluyen los campos `Usuario_Id`, `Genero` y `Año_de_nacimiento` que, combinados con los horarios y estaciones de cada viaje, podrían permitir identificar a una persona. Como las preguntas de negocio no requieren esta información, su uso será descartado del proyecto.

## Estrategia de partición de los datos

La partición no puede ser aleatoria porque trabajamos con variables temporales: una partición al azar puede producir un entrenamiento sin representatividad de días o estaciones o, en el peor de los casos, entrenar con datos futuros e introducir fuga de información. Las prácticas que se aplican en el proyecto para evitar estos escenarios son:

- Utilizar variables construidas únicamente con datos disponibles antes del momento de la predicción. Si un modelo tiene como horizonte el día siguiente, solo se usan datos anteriores a esa fecha.
- Se utilizarán tres particiones para los conjuntos de entrenamiento, validación y prueba, cada una de ellas con fechas específicas de división, que serán establecidas en `config.py`.
- Las transformaciones sólo serán ajustadas con el conjunto de entrenamiento y se aplicarán, sin recalcular, a los conjuntos de validación y prueba.

TODO: definir las fechas específicas de división de cada conjunto en `config.py`.
