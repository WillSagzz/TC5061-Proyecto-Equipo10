# Arquitectura del sistema

Esta sección describe los componentes de la solución, cómo se conectan desde el portal de datos abiertos de MiBici hasta el registro de experimentos en MLflow, y qué herramienta se usa en cada uno y por qué. Materializa el *pipeline* de la sección 5 y las decisiones sobre datos de la sección 4; el repositorio que la implementa se describe en la sección 9. Las cifras se midieron descargando y procesando los 24 archivos mensuales del periodo de estudio.

## Principios de diseño

**1. Procesamiento por lotes en una sola máquina.** Las tres preguntas se responden con predicciones por lotes (diarias en P1, cada tres horas en P2 y mensuales en P3), así que no se necesita un servicio de predicción en línea; Huyen (2022, cap. 7) recomienda la predicción por lotes como punto de partida cuando la latencia lo permite. El volumen tampoco exige cómputo distribuido: los 24 archivos suman 758 MB en CSV y 8,943,941 viajes (8,702,429 tras la limpieza), que ocupan 158 MB en Parquet y 718 MB en memoria, y la malla estación-hora de P1 tiene 6.86 millones de filas (470 estaciones, 20 horas de servicio al día y 730 días). Todo cabe en una computadora portátil, y Wilson (2022, cap. 13) advierte contra añadir infraestructura antes de que el problema la exija.

**2. El repositorio guarda recetas y huellas, nunca resultados.** En Git viven el código, la configuración, el manifiesto que describe los datos y la documentación. Los CSV, los Parquet, los modelos, la base y los artefactos de MLflow y las credenciales se regeneran en cada máquina y están excluidos por el `.gitignore`. Esto sigue las opiniones de Cookiecutter Data Science de que los datos crudos son inmutables y de que secretos y configuración local quedan fuera del control de versiones (DrivenData, s.f.), y cumple la restricción de la actividad de que ningún dato de MiBici, modelo ni carpeta de MLflow entre al historial.

**3. Todo *run* registra el commit, la versión de datos y el entorno con que se produjo.** Los *runs* no se versionan en Git: viven en MLflow y cada uno lleva como etiquetas y parámetros el SHA del commit, la versión de datos del manifiesto, las fechas de corte y la semilla, de modo que cualquier integrante puede volver a ese commit, reconstruir esa versión de datos y reproducirlo. Treveil et al. (2020, cap. 4) señalan que la reproducibilidad exige versionar por separado código, datos y entorno, y Huyen (2022, cap. 6) describe el *experiment tracking* como el mecanismo que une esas piezas.

## Diagrama de arquitectura

![Figura 2. Arquitectura del sistema: componentes, herramientas y flujo de datos desde el portal de MiBici hasta el registro de experimentos.](../../../../references/arquitectura.png)

La Figura 2 muestra los siete componentes. Los nodos azules son archivos versionados en Git, los cilindros naranjas son almacenamiento local fuera de Git y el cilindro verde punteado es el servidor en Azure planeado para la siguiente entrega. La fuente en Mermaid (`references/arquitectura.mmd`) y su exportación a imagen se guardan en el repositorio.

El flujo es el siguiente. `config.py` fija el rango de fechas, las fechas de corte de la partición, la semilla y los parámetros de negocio. `make data` ejecuta `dataset.py`, que descubre en el portal los enlaces de los meses del periodo y del catálogo, descarga los archivos que faltan en `data/raw/`, los normaliza y limpia, escribe dos Parquet en `data/processed/` y genera el manifiesto `data_manifest.json`, versionado en Git, con la versión de datos y el hash SHA-256 de cada archivo. `make train` ejecuta `train.py`, que carga los Parquet con `features.py` (validando que cumplan el contrato de columnas y tipos de `config.py`), construye en memoria las tablas de modelado (estación-hora para P1, estación-intervalo para P2 y perfil por estación para P3), las parte cronológicamente, entrena el *baseline* de cada pregunta, guarda el modelo en `models/` y registra parámetros, métricas, etiquetas y artefactos en MLflow, en el *backend* local o, cuando esté desplegado, en el servidor compartido del equipo, según la variable de entorno `MLFLOW_TRACKING_URI`.

## Componentes y herramientas

### 1. Fuentes de datos

La única fuente es el portal de datos abiertos de MiBici (MiBici, s.f.): un CSV por mes con los viajes iniciados en ese mes y un archivo de nomenclatura con identificador, nombre, coordenadas y capacidad de cada estación. El proyecto usa los 24 meses de septiembre de 2024 a agosto de 2026 (sección 4). Los enlaces se descubren leyendo la página del portal, porque cada archivo se sirve desde una URL con un identificador numérico que el portal asigna al publicarlo.

Tres propiedades de la fuente condicionan el diseño:

- **Los archivos se vuelven a publicar sin aviso y en la misma URL.** Dentro del periodo de estudio, el archivo de marzo de 2025 se reescribió 153 días después de su publicación y el de agosto de 2025, 36 días después. Solo un hash del contenido permite detectarlo.
- **El catálogo de estaciones no tiene historial.** Se sobrescribe en la misma URL cada vez que se actualiza, por lo que su hash se registra como el de cualquier otro archivo.
- **El formato cambia entre meses.** En el propio periodo conviven dos formatos de fecha (`DD/MM/AAAA H:MM` sin segundos en septiembre y octubre de 2024, `AAAA-MM-DD HH:MM:SS` en el resto), dos codificaciones (Latin-1 en ocho archivos, UTF-8 en los otros dieciséis) y dos estilos de comillas y saltos de línea.

### 2. Configuración y versionado de datos

`config.py` es la única fuente de rutas, fechas, semillas y parámetros de negocio: el primer y el último mes del periodo, las fechas de corte de entrenamiento, validación y prueba, la semilla, el contrato de datos (columnas y tipos que entrega la ingesta y leen los modelos), las reglas de limpieza (duración máxima de un viaje, duración mínima de un retiro y devolución en la misma estación, horas sin servicio), el umbral y la longitud del intervalo de P2, los periodos del día y el número de grupos de P3, y el nombre del experimento y el *backend* de MLflow.

La actividad pide un mecanismo de versionado que permita reconstruir exactamente los mismos datos en cualquier computadora y registrar la versión en MLflow. Como el repositorio no puede contener ningún byte de MiBici, lo versionable es una descripción de los datos suficiente para reconstruirlos y verificarlos. Es la opción que sugiere la actividad (manifiesto en Git con conteo y hash por archivo) y el mismo patrón con que Nix, Bazel, pip en modo de verificación de hashes y DVC fijan dependencias externas: URL y hash de contenido en el control de versiones, bytes fuera de él y una diferencia visible si el origen cambió.

**Estado actual: manifiesto generado por la ingesta.** En esta entrega el mecanismo es el manifiesto `data_manifest.json`, que escribe `dataset.py` en la raíz del repositorio y se versiona en Git:

- Contiene la versión de datos, la fecha de generación, el rango de meses, el número de viajes limpios y de estaciones, el hash SHA-256 de cada uno de los 25 archivos crudos (24 meses y la nomenclatura) y el de los dos Parquet procesados.
- La versión (`version`, por ejemplo `1c3d603c990c`) son los primeros 12 caracteres del SHA-256 de los hashes de los Parquet procesados. Depende solo del contenido: dos integrantes que corren la ingesta sobre los mismos archivos obtienen la misma versión, y un archivo republicado por el portal o un cambio en las reglas de limpieza producen una versión distinta. El historial de versiones es el historial de Git del manifiesto, como ocurre con cualquier *lockfile*, y cualquier cambio entra a `main` por *Pull Request*.
- `train.py` lee el campo `version` del manifiesto y lo registra como el parámetro `version_datos`, de modo que cada *run* queda ligado a los datos exactos con que se entrenó.

El manifiesto describe los datos después de descargarlos: la verificación ocurre al comparar el manifiesto regenerado con el que está en Git (`git diff`), no antes de procesar. Esa es su limitación: si el portal republica un archivo, la ingesta lo procesa igual y el cambio se detecta al revisar la diferencia del manifiesto.

**Diseño objetivo: descriptor Frictionless Data Package.** Para la siguiente entrega el manifiesto evolucionará a un descriptor que sigue el estándar Frictionless Data Package (Frictionless Data, s.f.), `mibici/datapackage.json`, con un recurso por archivo fuente (24 meses y la nomenclatura), cada uno con `path` (la URL), `bytes`, `hash` SHA-256, `encoding` y el esquema de columnas, más `version`, `created`, `sources` y `licenses` del paquete. Las diferencias con el estado actual son tres:

- **Verifica antes de procesar.** La ingesta comparará el tamaño y el hash de cada archivo descargado contra el descriptor y terminará con error si el portal lo republicó con otro contenido, en lugar de procesarlo y dejar que la diferencia aparezca después en el manifiesto. Un archivo presente con hash correcto no se vuelve a descargar.
- **Se actualiza de forma explícita.** Una versión nueva de los datos será un descriptor nuevo, generado con un comando propio (`make snapshot`) que descubre los enlaces, descarga, calcula tamaños y hashes y asigna como `version` el mes en que se congeló la instantánea (`2026-10` para la ventana de septiembre de 2024 a agosto de 2026). El cambio entra a `main` por *Pull Request* y se revisa como un cambio de código.
- **Usa un estándar.** La biblioteca `frictionless` valida tamaños, hashes y esquema sin código propio, y el descriptor sirve además como ficha del conjunto de datos; la ventana se mantiene congelada durante el curso para que las métricas de las entregas sean comparables.

**Tabla 8.** Comandos del versionado de datos, en el estado actual y en el diseño objetivo.

| Comando | Estado actual | Diseño objetivo |
|---------|---------------|-----------------|
| `make data` | Descubre los enlaces del periodo, descarga a `data/raw/` los archivos que falten, normaliza y limpia, escribe los Parquet y regenera `data_manifest.json`. Si el portal republicó un archivo, cambian su hash y la versión, y `git diff` lo muestra. | Reconstruye la versión descrita en `datapackage.json`: descarga las URL listadas, verifica tamaño y hash de cada archivo antes de procesarlo y termina con error si alguno difiere; no hay opción para omitir la verificación, el remedio es un descriptor nuevo. |
| `git log -- data_manifest.json` | Historial de versiones de datos: cada commit que cambió el manifiesto corresponde a una versión distinta, con fecha y autor. | Igual, sobre `mibici/datapackage.json`. |
| `make snapshot VERSION=... START=... END=...` | No existe. | Regenera el descriptor a partir del portal (descubre enlaces, descarga, calcula tamaños y hashes); el cambio entra a `main` por *Pull Request*. |

En ambos casos cada *run* registra la versión de datos junto con el commit. Huyen (2022, cap. 6) y Treveil et al. (2020, cap. 4) exigen versionar los datos para reproducir un modelo; el manifiesto actual ya lo logra sin un almacén remoto, y el descriptor añade la verificación previa y el estándar.

### 3. Ingesta reproducible

`dataset.py` es un comando de terminal construido con Typer, sin pasos manuales. Trabaja archivo por archivo, porque el formato varía entre meses, y después consolida:

1. **Descubre y descarga.** Lee la página de datos abiertos, filtra los enlaces por el rango de fechas de `config.py` y descarga a `data/raw/` solo los archivos que no estén ya descargados, con una pausa entre descargas para no saturar el portal.
2. **Normaliza por archivo.** Detecta la codificación de cada CSV (UTF-8 o Latin-1), convierte los nombres de columna a minúsculas sin acentos, conserva únicamente las columnas del contrato (identificador del viaje, inicio, fin, origen y destino) y convierte las fechas indicando si el archivo usa día primero; sin esa indicación pandas interpreta `01/09/2024` como 9 de enero sin emitir error. Las columnas de usuario, género y año de nacimiento se descartan en esta frontera, para que ninguna tabla ni artefacto posterior contenga datos personales.
3. **Limpia.** Elimina los viajes con fechas ilegibles, con duración negativa o mayor a 24 horas, los retiros y devoluciones en la misma estación en menos de un minuto, las salidas entre la 1:00 y las 4:59 (horas en que el sistema no presta bicicletas), los viajes con estaciones que no están en la nomenclatura (los identificadores 405 y 407 de la sección 4) y los duplicados por identificador de viaje. Las reglas y sus umbrales viven en `config.py`.
4. **Consolida la ventana y escribe Parquet.** El archivo mensual es la partición de la fuente, no la unidad de análisis: 622 viajes del periodo terminan en el mes siguiente al de su archivo, los rezagos de 24 horas y 7 días que usarán los candidatos cruzan los bordes de mes y 111 estaciones abrieron durante la ventana, así que los 24 meses se unen en una sola tabla de viajes (`meses.parquet`) junto con el catálogo (`nomenclatura.parquet`), ambos en `data/processed/`.
5. **Registra la versión.** Calcula el SHA-256 de cada archivo crudo y de cada Parquet y escribe el manifiesto.

Herramientas: Python 3.10, Typer, requests y BeautifulSoup para descubrir y descargar, pandas y pyarrow para transformar y escribir, hashlib para los hashes y loguru para el registro de eventos. Huyen (2022, cap. 3) contrapone el CSV, orientado a filas, con Parquet, orientado a columnas y comprimido, que pandas lee y escribe de forma nativa (pandas, s.f.); aquí leer los 24 CSV toma 135 segundos y leer el Parquet equivalente, 1.4. La frontera de datos personales responde a las condiciones de uso de la sección 4 y a la práctica obligatoria de no registrar datos personales en MLflow.

### 4. Almacenamiento

El almacenamiento es local, en carpetas excluidas de Git, con la separación de la plantilla de Cookiecutter Data Science (DrivenData, s.f.). No hay almacenamiento compartido por decisión: la actividad exige que los datos puedan reconstruirse en cualquier computadora, así que el sistema de registro es el portal más el manifiesto, no la copia de ningún integrante.

**Tabla 9.** Niveles de almacenamiento local.

| Carpeta | Contenido | Quién la escribe |
|---------|-----------|------------------|
| `data/raw/` | CSV tal como se descargaron; copia inmutable de la que se calculan los hashes del manifiesto | `make data` |
| `data/processed/` | `meses.parquet` (viajes limpios de los 24 meses, sin datos personales) y `nomenclatura.parquet` (catálogo de estaciones) | `make data` |
| `models/` | Modelos serializados con joblib; cada modelo se registra también como artefacto de su *run* | `make train` |
| `mlflow.db` y `mlartifacts/` | Base de datos SQLite con los experimentos y *runs*, y carpeta con sus artefactos, en el modo local de MLflow | `make train` |

Las tablas de modelado de P1, P2 y P3 no se guardan en disco: `train.py` las construye en memoria a partir de los Parquet cada vez que corre, en menos de dos minutos, de modo que siempre corresponden al código del commit y a la versión de datos del manifiesto.

### 5. Procesamiento y *feature engineering*

`features.py` contiene las transformaciones como funciones reutilizables con *docstring* que indica qué hacen y por qué son pertinentes para su pregunta (catálogo en la sección 5): la carga de los Parquet con validación del contrato de datos, la lectura de la versión del manifiesto, la partición cronológica, el conteo de salidas y llegadas por estación y periodo (la base común de P1 y P2), las variables de calendario, la etiqueta de riesgo de P2 y el perfil de uso de cada estación de P3. `train.py` las aplica sobre el Parquet limpio y produce la malla estación-hora de P1, la malla estación-intervalo de P2 a la granularidad definida en `config.py` y el perfil histórico por estación de P3.

La regla crítica es que ninguna variable use información posterior al momento de la predicción. Huyen (2022, cap. 5) enumera las fugas más frecuentes (particiones aleatorias en datos temporales, escalado ajustado antes de dividir, variables calculadas con datos del futuro) y la arquitectura las cierra en tres puntos: la partición es siempre por fechas de corte, el perfil de P3 se calcula únicamente con los movimientos del periodo de entrenamiento, y el escalador de P3 vive dentro de un `Pipeline` de scikit-learn que se ajusta con esos mismos datos; los rezagos y promedios móviles de los candidatos se calcularán con desplazamientos sobre ventanas cerradas. Mantener las funciones en el módulo y no en los *notebooks* sigue a Cookiecutter Data Science (DrivenData, s.f.) y a Wilson (2022, cap. 9), y garantiza que el futuro `predict.py` use las mismas funciones que el entrenamiento, evitando el *train-serving skew* (Huyen, 2022).

### 6. Entrenamiento

`train.py` expone un comando por pregunta (`p1`, `p2`, `p3`) y uno que corre los tres (`all`), de modo que `make train` ejecuta los tres *baselines* y cada uno es un *run* independiente. Antes de entrenar, `load_data` se niega a continuar si falta la ingesta o si los Parquet no cumplen el contrato de columnas y tipos de `config.py`. Los modelos base viven en `modeling/baselines.py`: para P1 y P2, un estimador `GroupedDummy` que entrena un `DummyRegressor` o `DummyClassifier` de scikit-learn por cada grupo (estación y hora; estación, intervalo y tipo de día) con un *dummy* general como respaldo para grupos nuevos; para P3, un `Pipeline` con `StandardScaler` y `KMeans`. Cada modelo se ajusta con el bloque de entrenamiento, se evalúa con validación y prueba (P1 y P2) o con las métricas internas del propio perfil (P3) y se guarda en `models/`. scikit-learn cubre los modelos base y candidatos y las métricas de la sección 6; LightGBM entraría como dependencia fijada solo si se usa como candidato en P1. Empezar por un *baseline* sencillo y comparar siempre contra él sigue a Huyen (2022, cap. 6).

### 7. Registro de experimentos

El registro se hace con MLflow Tracking (MLflow, s.f.) desde `train.py`, nunca solo desde un *notebook*. Un único experimento, `mibici-equipo10`, agrupa todos los *runs*; cada uno lleva un nombre descriptivo con pregunta, etapa y modelo (`P1-baseline-dummy-estacion-hora`, `P2-baseline-dummy-estacion-intervalo-tipo-dia`, `P3-baseline-kmeans-k4`) y registra lo que exige la actividad: como parámetros, `version_datos`, las fechas de inicio de entrenamiento, validación y prueba y el fin de los datos, `semilla`, el modelo y sus hiperparámetros, el número de filas de cada conjunto y el número de grupos; como métricas, las de la sección 6 con el conjunto como prefijo (`test_mae`, `test_recall_vaciarse`, `train_silhouette`); como etiquetas, `equipo`, `pregunta`, `git_commit`, `git_cambios_sin_commit` (que avisa si el *run* se ejecutó con cambios locales sin confirmar) y `tipo_modelo`; como artefactos, al menos una figura por pregunta, el modelo y, en P3, las tablas de grupos. Para la siguiente entrega se prevé registrar además el conjunto de datos como entrada del *run* con `mlflow.log_input`, con la versión del manifiesto y la URL del portal como fuente, de modo que la versión aparezca también en la pestaña de *datasets* de la interfaz de MLflow. Nunca credenciales ni datos personales, lo que queda garantizado porque las tablas de modelado ya no los contienen.

El código no nombra ningún servidor: lee `MLFLOW_TRACKING_URI` del archivo `.env` mediante python-dotenv y, si no está definida, usa una base SQLite local. Esto permite dos modos con el mismo código, el mismo experimento y las mismas etiquetas:

**Tabla 10.** Modos de registro de experimentos.

| Modo | Cuándo se usa | Configuración | Qué necesita el integrante |
|------|---------------|---------------|----------------------------|
| Local | Iterar y depurar; reproducir desde un clon limpio sin credenciales | Sin `.env`: MLflow escribe en `mlflow.db` (SQLite) y guarda los artefactos en `mlartifacts/`; se consulta con `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Nada |
| Compartido en Azure (planeado para la siguiente entrega) | *Runs* que el equipo quiere ver y comparar en tiempo real | `MLFLOW_TRACKING_URI` apunta al servidor del equipo | Tres variables en `.env`: URI, usuario y contraseña de MLflow |

En esta entrega todos los *runs* y las capturas provienen del modo local. El servidor compartido está diseñado y se desplegará para la siguiente entrega como una instancia propia de MLflow Tracking Server en Azure: correrá en un contenedor en Azure Container Apps, guardará los metadatos en Azure Database for PostgreSQL y los artefactos en Azure Blob Storage, y protegerá el acceso con la autenticación básica de MLflow; como los artefactos pasan por el servidor, los integrantes no necesitarán credenciales de Azure. Se prefiere a la integración de MLflow en Azure Machine Learning porque ofrece la misma interfaz que el modo local (la actividad pide capturas de la interfaz de MLflow) y habilita el Model Registry para entregas siguientes. Wilson (2022, cap. 8) presenta MLflow como la herramienta para registrar y comparar los experimentos de un producto mínimo viable; Treveil et al. (2020, cap. 4) describen este registro como lo que vuelve auditables las decisiones de modelado. `mlflow.db`, `mlartifacts/`, `mlruns/` y `.env` permanecen fuera de Git; en el repositorio solo entra un `.env.example` con el nombre de la variable.

La Figura 3 resume cómo se encadenan estas piezas: el manifiesto fija qué datos se leyeron, el commit fija qué se hizo con ellos, `load_data` comprueba localmente que los Parquet cumplen el contrato, y el *run* registra ambos identificadores, de modo que `version_datos` más `git_commit` más el entorno fijado en `requirements.txt` bastan para repetirlo en otra máquina.

<figure><img src="../../../../references/trazabilidad.png" alt="Cadena de trazabilidad" style="width: 70%;"><figcaption>Figura 3. Cadena de trazabilidad: del manifiesto de datos y el commit de Git al <em>run</em> de MLflow.</figcaption></figure>

## Resumen de herramientas y justificación

**Tabla 11.** Herramientas por componente, alternativa considerada y justificación.

| Componente | Herramienta | Alternativa considerada | Justificación |
|------------|-------------|-------------------------|---------------|
| Fuentes de datos | Portal de MiBici; enlaces descubiertos con requests y BeautifulSoup | URL escritas a mano | Única fuente oficial; el portal asigna los identificadores de las URL, así que descubrirlos evita una lista manual (MiBici, s.f.) |
| Ingesta | Python 3.10, Typer, pandas, pyarrow, hashlib, loguru | *Notebook* ejecutado a mano | La actividad exige un comando sin pasos manuales; Cookiecutter Data Science organiza el análisis como pasos reproducibles desde la terminal (DrivenData, s.f.) |
| Formato de almacenamiento | Parquet | CSV intermedios | Orientado a columnas, comprimido y tipado (Huyen, 2022, cap. 3); reduce la lectura de 135 s a 1.4 s |
| Motor de procesamiento | pandas | Spark o Dask | Los datos caben en memoria; el cómputo distribuido añadiría costo sin beneficio (Wilson, 2022, cap. 13) |
| Versionado de datos | Hoy, manifiesto `data_manifest.json` en Git con SHA-256 de cada CSV y de cada Parquet, conteos y versión derivada del contenido; planeado, descriptor Frictionless Data Package `datapackage.json` con verificación previa (Frictionless Data, s.f.); historial de versiones en Git | DVC `import-url`; manifiesto en formato propio sin estándar | Versionar datos es requisito de reproducibilidad (Huyen, 2022, cap. 6; Treveil et al., 2020, cap. 4); el manifiesto lo logra sin almacén remoto y es la opción que sugiere la actividad; el estándar aporta validación con `frictionless` |
| *Feature engineering* | Funciones en `features.py` con *docstring*; contrato de datos en `config.py`; `Pipeline` de scikit-learn | Transformaciones en el *notebook* | Evita las fugas que cataloga Huyen (2022, cap. 5) y el *train-serving skew*; código modular y comprobable (Wilson, 2022, cap. 9) |
| Modelado | scikit-learn (`DummyRegressor`, `DummyClassifier`, `KMeans`, métricas) | Otras bibliotecas | Cubre modelos y métricas de las secciones 5 y 6; empezar por el modelo más simple (Huyen, 2022, cap. 6) |
| Registro de experimentos | MLflow Tracking local (SQLite y `mlartifacts/`); servidor propio en Azure (Container Apps, PostgreSQL, Blob Storage) planeado para la siguiente entrega | MLflow de Azure Machine Learning | Herramienta del curso; registra parámetros, métricas, etiquetas y artefactos (MLflow, s.f.; Wilson, 2022, cap. 8; Treveil et al., 2020, cap. 4); misma interfaz en ambos modos |
| Configuración y entorno | `config.py`, `.env` con python-dotenv, `requirements.txt` con versiones fijas | Constantes repartidas; dependencias sin fijar | Una sola fuente de rutas, fechas y semillas y un entorno reproducible son requisitos de la actividad; Treveil et al. (2020, cap. 4) piden versionar el entorno |
| Orquestación | Makefile con `data` y `train` | Airflow o Prefect | Un grafo de dos pasos no justifica un orquestador; Huyen (2022, cap. 10) ubica cron y Make en el primer escalón |
| Colaboración | Git y GitHub con `main` protegida, *Pull Requests* y Ruff | Trabajo directo sobre `main` | Requisito de la actividad; estándares de código y mantenibilidad en equipo (Wilson, 2022, cap. 10); detalle en la sección 9 |
