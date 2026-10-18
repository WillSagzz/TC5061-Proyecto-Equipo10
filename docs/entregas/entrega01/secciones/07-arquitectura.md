# Arquitectura del sistema

Esta sección presenta los componentes de la solución y cómo se conectan, desde el portal de datos abiertos de MiBici hasta el registro de experimentos en MLflow, y declara y justifica la herramienta elegida para cada componente. La arquitectura materializa el *pipeline* descrito en la sección anterior y las decisiones sobre datos y partición de la sección 4; el repositorio que la implementa se describe en la sección 9. Las cifras que aparecen aquí se midieron el 18 de octubre de 2026 descargando y procesando los 24 archivos mensuales del periodo de estudio en una computadora portátil convencional.

## Principios de diseño

Tres decisiones gobiernan el resto de la arquitectura.

**1. Procesamiento por lotes en una sola máquina.** Las tres preguntas se responden con predicciones por lotes (diarias en P1, cada tres horas en P2 y mensuales en P3), por lo que ninguna requiere un servicio de predicción en línea; Huyen (2022, cap. 7) recomienda la predicción por lotes como punto de partida siempre que la latencia lo permita. El volumen tampoco exige cómputo distribuido: los 24 archivos mensuales suman 758 MB en CSV y 8,943,941 viajes, que ocupan 158 MB en Parquet sin las columnas personales y 718 MB en memoria con todas las columnas; la malla horaria por estación de P1 tiene 8.27 millones de filas. Todo cabe en la memoria de una computadora portátil, y Wilson (2022) advierte que añadir infraestructura antes de que el problema la exija es una de las formas más comunes de fracasar en un proyecto de *Machine Learning*.

**2. El repositorio guarda recetas y huellas, nunca resultados.** En Git viven el código, la configuración, los manifiestos que describen los datos y la documentación; los CSV, los Parquet, los modelos, las carpetas de MLflow y las credenciales se regeneran en cada máquina y están excluidos por el `.gitignore`. Este principio sigue las opiniones de Cookiecutter Data Science de que los datos crudos son inmutables y que los secretos y la configuración local quedan fuera del control de versiones (DrivenData, s.f.), y cumple la restricción de la actividad de que ningún dato de MiBici, modelo ni carpeta de MLflow entre al historial.

**3. Todo *run* registra el commit, la versión de datos y el entorno con que se produjo.** Los *runs* no se versionan en Git: viven en MLflow y cada uno lleva como etiquetas y parámetros el SHA del commit, el nombre de la versión de datos, las fechas de corte y la semilla, de modo que cualquier integrante puede volver a ese commit, reconstruir esa versión de datos y reproducir el *run*. Treveil et al. (2020, cap. 4) señalan que la reproducibilidad exige versionar por separado el código, los datos y el entorno, y registrar además las semillas y los hiperparámetros; Huyen (2022, cap. 6) describe el *experiment tracking* como el mecanismo que une esas piezas.

<!-- pagebreak -->

## Diagrama de arquitectura

![Figura 2. Arquitectura del sistema: componentes, herramientas y flujo de datos desde el portal de MiBici hasta el registro de experimentos.](../../../../references/arquitectura.png)

La Figura 2 muestra los siete componentes y su conexión. Los nodos azules son archivos versionados en Git, los cilindros naranjas son almacenamiento local excluido de Git y el cilindro verde es el servicio en Azure. La fuente del diagrama (`references/arquitectura.mmd`, en formato Mermaid) y su exportación a imagen (`references/arquitectura.png`) se guardan en el repositorio, como pide la actividad.

El flujo es el siguiente. `config.py` declara la versión de datos activa; el manifiesto de esa versión lista los 24 archivos mensuales y el catálogo de estaciones con su URL, tamaño, número de filas y hash SHA-256. El comando `make data` ejecuta `dataset.py`, que descarga cada archivo, comprueba que su hash coincide con el manifiesto, lo normaliza, lo limpia y guarda el resultado en Parquet dentro de `data/`. El comando `make features` ejecuta `features.py`, que construye a partir del Parquet limpio las tres tablas de modelado (estación-hora para P1, estación-intervalo para P2 y perfil por estación para P3) y escribe junto a ellas un recibo con la versión de datos y el commit que las produjeron. El comando `make train` ejecuta `train.py`, que verifica ese recibo, entrena el modelo base o candidato de la pregunta indicada con una partición cronológica y registra parámetros, métricas, etiquetas y artefactos en MLflow. El *backend* de MLflow se elige con una variable de entorno: por defecto una carpeta local `mlruns/`, y opcionalmente el servidor compartido del equipo en Azure.

## Componentes y herramientas

### 1. Fuentes de datos

La única fuente es el portal de datos abiertos de MiBici (MiBici, s.f.), que publica un archivo CSV por mes con los viajes iniciados en ese mes y un archivo de nomenclatura con el identificador, nombre, coordenadas y capacidad de anclajes de cada estación. El portal lista 142 archivos mensuales desde diciembre de 2014; el proyecto usa los 24 de septiembre de 2024 a agosto de 2026 (sección 4). Los enlaces se descubren leyendo la página del portal en lugar de escribirse a mano, porque cada archivo se sirve desde una URL con un identificador numérico que el portal asigna al publicarlo.

Al revisar la fuente se encontraron tres propiedades que condicionan el resto del diseño:

- **Los archivos se vuelven a publicar sin aviso y en la misma URL.** Dentro del periodo de estudio, el archivo de marzo de 2025 se reescribió 153 días después de su publicación original y el de agosto de 2025, 36 días después; dos archivos de 2023 se reescribieron el 15 de octubre de 2026. Solo un hash del contenido permite detectar estos cambios.
- **El catálogo de estaciones no tiene historial.** El archivo `nomenclatura_2026_09.csv` se sobrescribe en la misma URL cada vez que se actualiza (la última, el 2 de octubre de 2026), de modo que es la entrada más volátil del proyecto y debe fijarse por hash como cualquier otra.
- **El formato cambia entre meses.** Dentro del propio periodo de estudio conviven dos formatos de fecha (`DD/MM/AAAA H:MM` sin segundos en septiembre y octubre de 2024, `AAAA-MM-DD HH:MM:SS` en el resto), dos codificaciones de caracteres (Latin-1 en siete archivos y UTF-8 en el resto) y dos estilos de comillas y saltos de línea. Esto confirma la advertencia de la actividad de que manejar estas diferencias de forma reproducible es parte del reto.

### 2. Configuración y versionado de datos

**Configuración centralizada.** `config.py` es la única fuente de rutas, fechas, semillas y parámetros de negocio: la versión de datos activa (`DATA_VERSION`), las fechas de inicio y fin del periodo (derivadas del manifiesto de esa versión, para que no puedan editarse sin crear una versión nueva), las fechas de corte de entrenamiento, validación y prueba, la semilla aleatoria, el umbral y el intervalo con los que se construye la etiqueta de P2, el historial mínimo que debe tener una estación para participar en los rezagos y el nombre del experimento de MLflow. Ningún otro archivo repite estos valores, como exige la actividad.

**Estándar de versionado de datos.** La actividad pide un mecanismo que permita reconstruir exactamente los mismos datos en cualquier computadora y que la versión pueda registrarse en MLflow. Como el repositorio no puede contener ningún byte de MiBici, lo único versionable es una *descripción* de los datos suficientemente precisa para reconstruirlos y verificarlos. Esa descripción es el manifiesto, que sigue la opción que sugiere la propia actividad (manifiesto en Git con número de registros y hash por archivo) y la extiende para que sirva como contrato de reconstrucción:

- **Unidad de versionado: una instantánea inmutable.** Una versión es un archivo JSON en `mibici/data_versions/` que lista los archivos fuente con su URL, tamaño en bytes, número de filas y hash SHA-256, además del rango de meses que cubre y la fecha en que se congeló. Una vez en `main`, el archivo no se modifica; corregir algo significa crear una versión nueva.
- **Nombre: el mes en que se congeló la instantánea**, con un sufijo de revisión opcional. `2026-10` es "los datos tal como se congelaron en octubre de 2026" y cubre de septiembre de 2024 a agosto de 2026; `2026-10.1` sería una segunda instantánea del mismo mes, por ejemplo tras una corrección del portal. El nombre dice cuándo se congeló y el contenido dice qué cubre. La cadencia mensual coincide con la publicación del portal y con el reentrenamiento mensual previsto en el ML Canvas.
- **Ventana congelada durante el curso.** Las versiones siguientes solo se crearán si el portal reescribe un archivo o si el equipo decide extender el periodo; así las métricas de las distintas entregas son comparables.
- **Ubicación dentro del módulo.** El manifiesto lo lee el código en tiempo de ejecución (`config.py` toma de él las fechas, `dataset.py` verifica contra él y `train.py` registra su nombre), por lo que vive junto a `config.py` y no en `references/`, que Cookiecutter Data Science reserva para material que consultan las personas (DrivenData, s.f.).

**Tabla 7.** Comandos del versionado de datos.

| Comando | Qué hace |
|---------|----------|
| `make data` | Reconstruye la versión declarada en `config.py`: descarga exactamente las URL listadas, verifica cada hash, normaliza y escribe Parquet. Si un archivo difiere del manifiesto, termina con error indicando cuál. |
| `make data VERSION=2026-10` | Igual, para una versión explícita. |
| `make snapshot VERSION=2026-11 START=2024-09 END=2026-09` | Crea un manifiesto nuevo a partir del portal: descubre los enlaces, descarga, calcula hashes y conteos y escribe el JSON. El archivo entra a `main` por *Pull Request*, de modo que un cambio de versión de datos se revisa igual que un cambio de código. |

Dos integrantes que ejecuten `make data` en fechas distintas obtienen datos idénticos o un error explícito que nombra el archivo que cambió; nunca datos silenciosamente distintos. El manifiesto describe las entradas, no las salidas: el Parquet procesado es una función determinista de la versión de datos y del commit del código, y esa separación evita que un cambio en las reglas de limpieza invalide una versión cuyos bytes de entrada no cambiaron. La Tabla 8 resume los tres contratos.

**Tabla 8.** Contratos que identifican los datos en cada capa.

| Capa | Qué la fija | Dónde vive | Cambia cuando |
|------|-------------|------------|---------------|
| Archivos fuente (CSV) | Manifiesto: URL, bytes, filas y SHA-256 por archivo; nombre de la versión | `mibici/data_versions/<versión>.json`, en Git | MiBici reescribe un archivo o el equipo cambia el periodo |
| Tablas procesadas (viajes limpios, tablas P1, P2 y P3) | Versión de datos + commit de Git + entorno fijado | La combinación, registrada en cada *run* de MLflow | Cambian las entradas o cambia el código de limpieza o de variables |
| Esquema de las tablas procesadas (columnas, tipos, llaves, particiones) | Declarado en código y comprobado al escribir y al leer | `mibici/schemas.py`, en Git | Cambio deliberado de esquema mediante *Pull Request* |

### 3. Ingesta reproducible

`dataset.py` es un comando de terminal construido con Typer que se ejecuta con `make data` y no requiere pasos manuales. Opera archivo por archivo, porque las propiedades de formato varían entre meses, y después consolida:

1. **Resuelve la lista de archivos** a partir del manifiesto de la versión activa.
2. **Descarga y verifica.** Calcula el SHA-256 de cada archivo descargado y lo compara con el manifiesto; un archivo ya presente con hash correcto no se vuelve a descargar, y uno incompleto o distinto se descarga de nuevo. El portal exige una cabecera de navegador para servir los archivos, que el cliente HTTP envía.
3. **Normaliza por archivo.** Detecta la codificación (UTF-8 o Latin-1), unifica los nombres de columna a un esquema único en minúsculas y sin acentos, y convierte las fechas indicando explícitamente el formato detectado en cada archivo; nunca se deja que la biblioteca lo adivine, porque con el formato de día primero pandas interpreta `01/09/2024` como 9 de enero sin emitir error.
4. **Limpia.** Elimina duplicados por identificador de viaje (el archivo de septiembre de 2026, fuera del periodo, contiene 758), descarta las columnas de usuario, género y año de nacimiento en esta frontera para que ninguna tabla ni artefacto posterior contenga datos personales, convierte los identificadores a enteros y cruza con el catálogo de estaciones, registrando las estaciones sin catálogo (405 y 407 en el periodo) y las de capacidad cero.
5. **Consolida la ventana.** El archivo mensual es la partición de la fuente, no la unidad de análisis: 622 viajes del periodo terminan en el mes calendario siguiente al de su archivo, los rezagos de 24 horas y 7 días cruzan los bordes de mes, y 111 estaciones abrieron durante la ventana. Por eso los 24 meses se unen en una sola tabla de viajes antes de cualquier agregación.
6. **Escribe Parquet** particionado por mes en `data/interim/`, junto con el catálogo de estaciones, y registra en el log el número de filas por archivo y total.

Herramientas: Python 3.10, Typer para la interfaz de línea de comandos, requests y BeautifulSoup para descubrir y descargar los archivos, pandas y pyarrow para leer, transformar y escribir, hashlib para los hashes y loguru para el registro de eventos. Huyen (2022, cap. 3) describe el CSV como un formato orientado a filas, legible pero lento de procesar, y Parquet como un formato orientado a columnas, comprimido y eficiente para lecturas analíticas; en este proyecto la diferencia es de 135 segundos para leer los 24 CSV frente a 1.4 segundos para leer el Parquet equivalente. La frontera de datos personales en la ingesta responde a las condiciones de uso analizadas en la sección 4 y a la práctica obligatoria de no registrar datos personales en MLflow; Treveil et al. (2020, cap. 3) ubican este tipo de controles de gobierno de datos dentro del desarrollo del modelo, no después.

### 4. Almacenamiento

El almacenamiento es local, en la carpeta `data/` excluida de Git, con los tres niveles que propone la plantilla de Cookiecutter Data Science (DrivenData, s.f.) más los modelos:

**Tabla 9.** Niveles de almacenamiento local.

| Carpeta | Contenido | Quién la escribe | Propiedad |
|---------|-----------|------------------|-----------|
| `data/raw/` | Los CSV tal como se descargaron | `make data` | Inmutable; es la copia contra la que se verifica el manifiesto |
| `data/interim/` | `viajes.parquet` particionado por mes y `estaciones.parquet`, ya limpios y sin datos personales | `make data` | Determinista dada la versión de datos y el commit |
| `data/processed/` | Tablas de modelado P1, P2 y P3 y el recibo de construcción `build_receipt.json` | `make features` | Determinista; el recibo registra versión de datos, commit, filas y esquema |
| `models/` | Modelos serializados | `make train` | Regenerable; el modelo también se registra como artefacto del *run* |

No existe almacenamiento compartido en esta entrega por decisión, no por omisión: la actividad exige que los datos puedan reconstruirse en cualquier computadora, así que el sistema de registro es el portal más el manifiesto, no la copia de ningún integrante. La partición por mes del Parquet hace barata la siguiente instantánea: cuando se incorpore un mes nuevo, solo ese archivo se convierte y los demás se verifican por hash.

### 5. Procesamiento y *feature engineering*

`features.py` contiene las transformaciones como funciones puras, reutilizables y documentadas con *docstring* que indica qué hacen y por qué son pertinentes para la pregunta que atienden, tal como pide la actividad y como se detalla en el catálogo de variables de la sección 5. El comando `make features` las aplica sobre el Parquet limpio y produce tres tablas: la malla completa estación-hora para P1 (472 estaciones por 17,520 horas), la malla estación-intervalo para P2 a la granularidad e intervalo definidos en `config.py`, y el perfil histórico por estación para P3.

La regla crítica es que ninguna variable use información posterior al momento de la predicción. Huyen (2022, cap. 5) enumera las fugas más frecuentes: particiones aleatorias en datos correlacionados en el tiempo, escalado ajustado antes de dividir y variables calculadas con datos del futuro. La arquitectura las cierra en tres puntos: los rezagos y promedios móviles se calculan con desplazamientos sobre ventanas cerradas, los escaladores viven dentro de un `Pipeline` de scikit-learn que se ajusta solo con el conjunto de entrenamiento, y las estaciones con menos historial que el mínimo configurado no aportan filas de entrenamiento. Mantener estas funciones en el módulo y no en los *notebooks* sigue el principio de Cookiecutter Data Science de que los *notebooks* son para exploración y comunicación (DrivenData, s.f.) y la recomendación de Wilson (2022, cap. 9) de escribir código modular y comprobable en lugar de *scripts* monolíticos; además garantiza que el futuro `predict.py` use exactamente las mismas funciones que el entrenamiento, evitando la discrepancia entre entrenamiento e inferencia que Huyen (2022) describe como *train-serving skew*.

### 6. Entrenamiento

`train.py` es un comando con dos opciones, la pregunta (`P1`, `P2` o `P3`) y el modelo (`baseline` o un candidato), de modo que `make train` lo invoca una vez por combinación y cada ejecución es un *run* independiente. Lee el recibo de construcción y se niega a entrenar si la versión de datos o el esquema no coinciden con `config.py`, para que nadie entrene sobre Parquet obsoleto o editado a mano. Toma las fechas de corte y la semilla de `config.py`, construye un `Pipeline` de scikit-learn con el preprocesamiento y el estimador, lo ajusta con el bloque de entrenamiento, evalúa con el bloque de prueba y guarda el modelo en `models/`.

La herramienta es scikit-learn, que cubre los modelos base y candidatos de las tres preguntas y las métricas de la sección 6; LightGBM o XGBoost se contemplan como candidatos adicionales para P1 y P2 y entrarían como dependencias fijadas cuando se usen. La estrategia de un *baseline* sencillo antes de cualquier candidato sigue la recomendación de Huyen (2022, cap. 6) de empezar por el modelo más simple y comparar siempre contra una línea base, y el uso de `Pipeline` sigue la guía de evaluación de modelos de scikit-learn.

### 7. Registro de experimentos

El registro se hace con MLflow Tracking (MLflow, s.f.) desde el código de `train.py`, nunca solo desde un *notebook*. Un único experimento, `mibici-equipo10`, agrupa todos los *runs*; cada *run* lleva un nombre descriptivo con la pregunta, el modelo y la etapa (por ejemplo `P1-ridge-baseline`, `P2-logreg-baseline`, `P3-kmeans-baseline`), y registra:

- **Parámetros de reproducción:** `data_version`, `start_date`, `end_date`, las fechas de corte, `random_seed`, el nombre del modelo y sus hiperparámetros, y los conteos de filas tomados del recibo de construcción.
- **Métricas** con el conjunto en que se calculan como prefijo: `test_mae` y `test_rmse` en P1, `test_recall` y `test_f1` en P2, `train_silhouette` y `train_davies_bouldin` en P3.
- **Etiquetas:** `team`, `question` y `git_commit`, además de `stage` (`baseline` o `candidate`).
- **Artefactos:** al menos una figura por pregunta (predicción contra valor real, matriz de confusión, mapa de grupos de estaciones), el modelo serializado y una copia del manifiesto de datos.
- **Nunca** credenciales ni datos personales, lo que queda garantizado porque las tablas de modelado ya no los contienen.

El código no nombra ningún *backend*: lee `MLFLOW_TRACKING_URI` del archivo `.env` mediante python-dotenv, ya integrado en `config.py`. Esto permite dos modos con el mismo código, el mismo nombre de experimento y las mismas etiquetas (Tabla 10).

**Tabla 10.** Modos de registro de experimentos.

| Modo | Cuándo se usa | `MLFLOW_TRACKING_URI` | Qué necesita el integrante |
|------|---------------|------------------------|----------------------------|
| Local | Iterar y depurar; reproducir desde un clon limpio sin credenciales (es el modo que usará quien evalúe) | Ausente: MLflow escribe en `mlruns/`, que se consulta con `mlflow ui` | Nada |
| Compartido en Azure | *Runs* que el equipo quiere ver y comparar en tiempo real | `https://<servidor-del-equipo>` | Tres variables en `.env`: URI, usuario y contraseña de MLflow |

El servidor compartido es una instancia propia de MLflow Tracking Server en Azure: el servidor corre en un contenedor en Azure Container Apps (que escala a cero cuando no se usa), guarda los metadatos de experimentos y *runs* en Azure Database for PostgreSQL y los artefactos en Azure Blob Storage, y protege el acceso con la autenticación básica integrada de MLflow. Con el acceso a artefactos a través del servidor, los integrantes no necesitan credenciales de Azure, solo las de MLflow. Se eligió esta opción frente al punto de conexión MLflow de Azure Machine Learning porque ofrece exactamente la misma interfaz que el modo local (la actividad pide capturas de la interfaz de MLflow), evita una dependencia adicional de autenticación y habilita el Model Registry para las entregas siguientes, que el almacén local en archivos no soporta. Wilson (2022, cap. 8) muestra MLflow como la herramienta para registrar y comparar los experimentos de un producto mínimo viable, y Treveil et al. (2020, cap. 4) describen este registro como el componente que vuelve auditables las decisiones de modelado. Las carpetas `mlruns/` y `mlartifacts/` locales y el archivo `.env` permanecen excluidos de Git; en el repositorio solo entra un `.env.example` con los nombres de las variables y sin valores.

## Trazabilidad de un *run*

La Figura 3 resume cómo se encadenan los contratos anteriores para que cualquier *run* pueda rastrearse hasta sus entradas. El manifiesto fija qué se leyó, el commit fija qué se hizo con ello, el esquema declara qué salió y el recibo de construcción comprueba localmente que los tres coinciden antes de entrenar. El *run* registra los tres identificadores, de modo que la combinación `data_version` más `git_commit` más el entorno fijado basta para repetirlo en otra máquina.

![Figura 3. Cadena de trazabilidad: del manifiesto de datos y el commit de Git al run de MLflow.](../../../../references/trazabilidad.png)

## Resumen de herramientas y justificación

**Tabla 11.** Herramientas por componente, alternativas consideradas y justificación.

| Componente | Herramienta | Alternativa considerada | Justificación |
|------------|-------------|-------------------------|---------------|
| Fuentes de datos | Portal de datos abiertos de MiBici, enlaces descubiertos con requests y BeautifulSoup | URL escritas a mano | Es la única fuente oficial; los identificadores numéricos de las URL los asigna el portal, así que descubrirlos evita mantener una lista manual (MiBici, s.f.) |
| Ingesta | Python 3.10, Typer, pandas, pyarrow, hashlib, loguru | *Notebook* ejecutado a mano | La actividad exige un comando sin pasos manuales; Cookiecutter Data Science organiza el análisis como un grafo de pasos reproducibles desde la terminal (DrivenData, s.f.) |
| Formato de almacenamiento | Parquet | CSV intermedios | Formato orientado a columnas, comprimido y tipado; Huyen (2022, cap. 3) lo contrapone al CSV orientado a filas, y aquí reduce la lectura de 135 s a 1.4 s |
| Motor de procesamiento | pandas sobre Parquet | Spark o Dask | Los datos caben en memoria (718 MB), por lo que el cómputo distribuido añadiría costo sin beneficio; Wilson (2022, cap. 13) advierte contra la sobreingeniería |
| Versionado de datos | Manifiesto JSON en Git con URL, bytes, filas y SHA-256; instantáneas inmutables nombradas por mes | DVC con almacenamiento remoto | Huyen (2022, cap. 6) y Treveil et al. (2020, cap. 4) exigen versionar los datos para reproducir un modelo; el manifiesto lo logra sin un almacén remoto y es la opción que sugiere la actividad, extendida con verificación |
| *Feature engineering* | Funciones puras en `features.py` con *docstring*; `Pipeline` de scikit-learn | Transformaciones en el *notebook*; *feature store* | Evita las fugas de información que cataloga Huyen (2022, cap. 5) y la discrepancia entrenamiento-inferencia; Wilson (2022, cap. 9) recomienda código modular y comprobable |
| Modelado | scikit-learn (LightGBM o XGBoost como candidatos opcionales) | Redes neuronales u otras bibliotecas | Cubre los modelos y métricas de las secciones 5 y 6; Huyen (2022, cap. 6) recomienda empezar por el modelo más simple y comparar contra una línea base |
| Registro de experimentos | MLflow Tracking con *backend* local o servidor propio en Azure | Azure Machine Learning; hojas de cálculo | Herramienta del curso; registra parámetros, métricas, etiquetas y artefactos por *run* (MLflow, s.f.); Wilson (2022, cap. 8) y Treveil et al. (2020, cap. 4) la presentan como base de la reproducibilidad y la auditoría |
| Servidor compartido | Azure Container Apps, Azure Database for PostgreSQL y Azure Blob Storage con autenticación básica de MLflow | Punto de conexión MLflow de Azure Machine Learning | Misma interfaz que el modo local, sin credenciales de Azure para los integrantes y con Model Registry disponible para entregas futuras |
| Configuración | `config.py` y `.env` con python-dotenv | Constantes repartidas en los *scripts* | La actividad exige una sola fuente de rutas, fechas y semillas; Cookiecutter Data Science mantiene los secretos fuera del control de versiones (DrivenData, s.f.) |
| Entorno | `requirements.txt` con versiones fijas sobre Python 3.10 | Dependencias sin fijar | Un entorno reproducible desde un clon limpio es requisito de la actividad y una de las tres cosas que Treveil et al. (2020, cap. 4) piden versionar |
| Orquestación | Makefile con los pasos `data`, `features`, `train` y `snapshot` | Airflow o Prefect | Un grafo de tres pasos no justifica un orquestador; Huyen (2022, cap. 10) ubica cron y Make en el primer escalón de la gestión de recursos y los orquestadores en el siguiente |
| Colaboración y calidad | Git y GitHub con `main` protegida, *Pull Requests* revisados, Conventional Commits y Ruff | Trabajo directo sobre `main` | Requisito de la actividad; Wilson (2022, cap. 10) relaciona los estándares de código con la mantenibilidad del trabajo en equipo (detalle en la sección 9) |

## Qué vive en Git y qué se regenera

**Tabla 12.** Separación entre lo versionado y lo regenerable.

| En Git | Se regenera en cada máquina y está en `.gitignore` |
|--------|-----------------------------------------------------|
| Código del módulo (`dataset.py`, `features.py`, `modeling/`, `schemas.py`) | `data/raw/`, `data/interim/`, `data/processed/` |
| `config.py` y `.env.example` | `.env` con la URI y las credenciales de MLflow |
| Manifiestos en `mibici/data_versions/` | `models/` |
| `requirements.txt`, `pyproject.toml`, `Makefile`, `.gitignore` | `mlruns/` y `mlartifacts/` |
| Documentación, ML Canvas y diagramas en `references/` y `docs/` | Salidas compiladas de las entregas |

Desde un clon limpio, el camino hasta un *run* reproducido es: crear el entorno e instalar `requirements.txt`, ejecutar `make data` (que reconstruye y verifica la versión de datos), `make features` y `make train`, y abrir `mlflow ui`. Ninguno de esos pasos requiere credenciales ni archivos que no estén en el repositorio.

## Alternativas descartadas y ruta de evolución

Cada alternativa descartada se enuncia junto con la condición que la volvería necesaria, porque las entregas siguientes se construyen sobre esta arquitectura.

- **Cómputo distribuido (Spark, Dask).** Se adoptaría si el periodo creciera hasta superar la memoria de una máquina; con 24 meses no ocurre (Huyen, 2022, cap. 3).
- **DVC o almacenamiento remoto de datos.** Se adoptaría si el equipo necesitara compartir datos derivados costosos de recalcular, o conservar una copia privada de los CSV ante reescrituras del portal; en ese caso el espejo iría a Azure Blob Storage y su URL se registraría en el manifiesto, nunca en Git.
- **Orquestador (Airflow, Prefect).** Se adoptaría cuando el *pipeline* deba ejecutarse de forma programada, por ejemplo con el reentrenamiento mensual del ML Canvas (Huyen, 2022, cap. 10).
- **Model Registry y servicio de predicción.** El servidor de MLflow en Azure ya lo habilita; `predict.py` implementará la inferencia por lotes cuando una entrega lo requiera (Wilson, 2022, cap. 16; Huyen, 2022, cap. 7).
- **Contenedores (Docker).** Útiles para fijar el entorno más allá de `requirements.txt`; se incorporarían al desplegar el servidor de MLflow y, más adelante, el servicio de predicción (Treveil et al., 2020, cap. 6).
- **Punto de conexión MLflow de Azure Machine Learning.** Descartado por las razones de la Tabla 11; se reconsideraría si el equipo migrara el entrenamiento a cómputo gestionado de Azure.

## Estado actual frente a la arquitectura objetivo

Esta sección describe la arquitectura que el equipo seguirá durante el curso. Al cierre de esta entrega, la ingesta ya descarga, filtra por fechas y concatena los archivos y escribe un manifiesto con conteos de filas; falta incorporar la verificación por hash contra el manifiesto versionado en el módulo, el formato de fecha explícito por archivo y el recibo de construcción, diseñados aquí a partir de lo observado en la fuente. El módulo pasará de llamarse `feature_engineering` a `mibici`, nombre que ya usan el diagrama y este documento.
