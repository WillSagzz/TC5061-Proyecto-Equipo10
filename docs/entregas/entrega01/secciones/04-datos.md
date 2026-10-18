# Definición de los datos y partición

## Rango de fechas propuesto

Los datos de MiBici abarcan desde diciembre de 2014 hasta septiembre de 2026. Sin embargo, usar todo el historial no garantiza mejores resultados. Los hábitos de movilidad cambian con el tiempo: cambian los hábitos del comportamiento ciudadano, se abren estaciones nuevas, cambia la capacidad de otras, se modifican vías, carreteras, etc. Por eso, los patrones de hace varios años pueden no representar el comportamiento actual de nuestro intervalo de estudio. Además, los años 2020 y 2021 están alterados por la pandemia e introducen patrones anómalos. Es por estas razones que se selecciona un periodo de estudio. El rango escogido es el más reciente, correspondiente a los últimos 24 meses, desde **septiembre de 2024 hasta agosto de 2026**. Esto nos permite capturar patrones estacionales de datos actuales, con los que estamos más familiarizados, y evitar utilizar estaciones que hayan desaparecido.

## Fuente de los archivos

El repositorio donde se alojan los datos de MiBici es <https://mibici.net/es/datos-abiertos/>. Se accede a los viajes mensualmente; la Tabla 1 lista los 24 archivos que se utilizarán. Además, se utilizará el archivo de nomenclatura de las estaciones, alojado en <https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv>.

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
- **Número de estaciones.** En el catálogo oficial de estaciones (nomenclatura) se identificaron **484** estaciones; sin embargo, en el periodo estudiado solo **472** estaciones poseen actividad registrada. Al cruzar estas estaciones con el catálogo:
  - De las 472 estaciones con actividad, solo **470** están en la nomenclatura.
  - Las **2** estaciones que registran viajes pero no están en la nomenclatura son los ID **405** y **407**. Al no contar con información sobre ellas, sus registros se excluirán del proyecto en la etapa de limpieza.
  - **14** estaciones están en la nomenclatura pero no tienen registro de viajes en el periodo estudiado.

## Variables originales

**Tabla 2.** Variables originales de los archivos de MiBici que se utilizan en el proyecto.

| Variable original | Archivo | Uso |
|-------------------|---------|-----|
| `Viaje_Id` | `datos_abiertos_AAAA_MM.csv` | Detección de registros duplicados. |
| `Inicio_del_viaje` | `datos_abiertos_AAAA_MM.csv` | Fecha y hora de inicio de los viajes. |
| `Fin_del_viaje` | `datos_abiertos_AAAA_MM.csv` | Fecha y hora de fin de los viajes. |
| `Destino_Id` | `datos_abiertos_AAAA_MM.csv` | Estación a donde llega la bicicleta. |
| `Origen_Id` | `datos_abiertos_AAAA_MM.csv` | Estación de donde sale la bicicleta. |
| `id` | `nomenclatura_2026_09.csv` | Identificador de la estación. Permite unir los viajes (`Origen_Id` y `Destino_Id`) con la capacidad, la ubicación y el nombre de cada estación. |
| `name` | `nomenclatura_2026_09.csv` | El prefijo de su código indica el municipio, por ejemplo: GDL-001. |
| `latitude`, `longitude` | `nomenclatura_2026_09.csv` | Localizar las estaciones en el mapa. |
| `dpcapacity` | `nomenclatura_2026_09.csv` | Capacidad de anclajes de cada estación. |

## Variables objetivo y cómo se construyen

**Tabla 3.** Construcción de la variable objetivo de cada pregunta.

| Pregunta | Variable objetivo | Construcción |
|----------|-------------------|--------------|
| **P1. Regresión.** ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Viajes iniciados (entero). | Sumatoria de los viajes iniciados en cada estación durante cada hora. La hora de inicio de cada viaje se redondea hacia abajo a la hora completa (por ejemplo, 08:47 pasa a 08:00) y se agrupan los viajes por estación de origen y hora. Las combinaciones de estación de origen y hora sin viajes se registran como cero. |
| **P2. Clasificación.** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Riesgo (categórica) con tres clases: riesgo de vaciarse, riesgo de llenarse, sin riesgo. | El día se divide en 8 intervalos de 3 horas y cada viaje se asigna dos veces: como salida de su estación de origen y como llegada de su estación de destino, cada una en el intervalo que corresponda. Se suman las salidas y las llegadas de cada estación en cada intervalo y se calcula el flujo neto (llegadas menos salidas); las combinaciones sin viajes se registran como cero. El flujo neto se compara con el 40 % de la capacidad de la estación: (1) *riesgo de vaciarse* si el flujo es menor o igual a −40 % de la capacidad; (2) *riesgo de llenarse* si es mayor o igual a +40 %; (3) *sin riesgo* si está entre −40 % y +40 %, sin llegar a ninguno de los dos. |
| **P3. Clustering.** ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | No posee una variable objetivo. Se buscan los patrones ocultos de uso similar entre estaciones que mostrará el *clustering*. | El algoritmo agrupa estaciones con perfiles parecidos según sus movimientos. Un movimiento es cada vez que una bicicleta sale de una estación o llega a ella, por lo que cada viaje genera dos movimientos. Con ellos se calculan las variables del *clustering*: (1) *porcentaje de movimientos por periodo del día* (mañana, mediodía, tarde y noche) de cada estación; (2) *porcentaje de movimientos en días laborales*, por ejemplo, 3 movimientos el lunes y 6 el sábado dan 3 / (3 + 6) = 33.33 %; (3) *uso diario por anclaje*, el número promedio de movimientos de la estación por día dividido entre su capacidad (número de anclajes); (4) *balance de la mañana*, que indica si la estación gana o pierde bicicletas entre las 5:00 y las 12:00 y se calcula como (llegadas − salidas) ÷ (llegadas + salidas), donde un valor negativo significa que salen más bicicletas de las que llegan; (5) *balance de la tarde*, el mismo cálculo con los movimientos de 16:00 a 20:00. |

## Condiciones de uso de MiBici y tratamiento de datos personales

Los datos se obtienen del portal de datos abiertos de MiBici, sistema administrado por la **Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara (AMIM)**. El portal no publica una licencia para los archivos. Sin embargo, el Reglamento Interno en materia de Transparencia de la AMIM establece que la Agencia gestiona su información bajo una política de datos abiertos que favorece la creación de conocimiento (art. 6, fr. IV), y que la información que genera es pública y de libre acceso, salvo la reservada o confidencial (art. 18). Con base en ello, los datos se usarán solo con fines académicos, no se redistribuirán y no se utilizarán directa o indirectamente para intentar identificar usuarios.

El mismo reglamento considera confidencial la información con datos personales de una persona identificada o identificable (art. 21). Los archivos de viajes incluyen los campos `Usuario_Id`, `Genero` y `Año_de_nacimiento` que, combinados con los horarios y estaciones de cada viaje, podrían permitir identificar a una persona. Como las preguntas de negocio no requieren esta información, su uso será descartado del proyecto.

## Estrategia de partición de los datos

La partición no puede ser aleatoria porque trabajamos con variables temporales: una partición al azar puede llevar a un entrenamiento sin representatividad de días o estaciones o, en el peor de los casos, a entrenar con datos futuros e introducir fuga de información. Las prácticas que se aplican en el proyecto para evitar estos escenarios son:

- Utilizar variables construidas únicamente con datos disponibles antes del momento de la predicción. Si un modelo tiene como horizonte el día siguiente, solo se usan datos anteriores a esa fecha. En P1 se utilizarán datos de viajes hasta el cierre del día anterior y, en P2, hasta el cierre del intervalo anterior.
- Se utilizarán tres particiones para los conjuntos de entrenamiento, validación y prueba, cada una de ellas con fechas específicas de división, que serán establecidas en `config.py`.
- Las transformaciones solo serán ajustadas con el conjunto de entrenamiento y se aplicarán, sin recalcular, a los conjuntos de validación y prueba.