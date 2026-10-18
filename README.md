# TC5061-Proyecto-Equipo10

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto MiBici — propuesta de solución de *Machine Learning* sobre los datos abiertos del
sistema de bicicleta pública del Área Metropolitana de Guadalajara.

## Propósito del proyecto

<!-- TODO: descripción del problema de negocio, alcance y qué entrega el proyecto. -->

## Preguntas del proyecto

| # | Pregunta de negocio | Tipo de tarea | Quién la usaría |
|---|---|---|---|
| **P1** | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema, para planear la flota |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Equipo de rebalanceo de bicicletas |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | *Clustering* | Planeación urbana y expansión del sistema |

<!-- TODO: formulación de cada pregunta como problema de ML (variable objetivo, granularidad,
     horizonte) y la aproximación del riesgo en P2. -->

## Datos utilizados

<!-- TODO: fuente, rango de fechas y su justificación, tamaño real obtenido (meses, viajes,
     estaciones), variables originales, licencia de uso y estrategia de partición. -->

## Instalación

<!-- TODO: versión de Python, creación del entorno e instalación de dependencias desde un
     clon limpio. -->

## Comandos

<!-- TODO: comando de ingesta reproducible y comando del experimento base en MLflow. -->

## Estructura del proyecto

```
├── Makefile              <- Comandos de conveniencia: `make data`, `make lint`, `make format`
├── README.md             <- Este documento
├── pyproject.toml        <- Metadatos del paquete `feature_engineering` y configuración de Ruff
├── requirements.txt      <- Dependencias para reproducir el entorno
│
├── data                  <- Excluida de Git; la genera el comando de ingesta
│   ├── external          <- Datos de terceros
│   ├── interim           <- Datos intermedios ya transformados
│   ├── processed         <- Conjuntos finales, listos para modelar
│   └── raw               <- Descarga original e inmutable de MiBici
│
├── docs                  <- Proyecto mkdocs por defecto de la plantilla
│
├── models                <- Modelos entrenados y serializados; excluidos de Git
│
├── notebooks             <- Solo exploración. Convención: `1.0-iniciales-descripcion.ipynb`
│
├── references            <- ML Canvas, diagrama de arquitectura y material explicativo
│
├── reports               <- Análisis generados (HTML, PDF, etc.)
│   └── figures           <- Gráficas y figuras para los reportes
│
└── feature_engineering   <- Código fuente del proyecto
    ├── __init__.py       <- Hace de `feature_engineering` un módulo de Python
    ├── config.py         <- Rutas, fechas y semillas: la única fuente de configuración
    ├── dataset.py        <- Ingesta reproducible: descarga, normaliza y guarda en Parquet
    ├── features.py       <- Funciones reutilizables de feature engineering
    ├── plots.py          <- Visualizaciones
    └── modeling
        ├── __init__.py
        ├── predict.py    <- Inferencia con modelos entrenados
        └── train.py      <- Entrenamiento de modelos
```

## Flujo de trabajo con Git/GitHub

El equipo trabaja con **[GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow)**:
una sola rama de larga duración (`main`), ramas de trabajo cortas y todo cambio entra por
*Pull Request* revisado. No hay rama de integración intermedia.

Los nombres de ramas siguen **[Conventional Branch](https://conventionalbranch.org/)** y los
mensajes de commit siguen **[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)**.

### 1. Rama principal protegida

`main` siempre debe estar en un estado funcional: cualquier integrante puede clonarla y
reproducir la ingesta y el experimento base. Por eso **nadie hace `push` directo a `main`**.

`main` está protegida en GitHub con una *classic branch protection rule* (*Settings → Branches*)
que usa `main` como patrón. La regla tiene activadas estas opciones:

- **Require a pull request before merging** — bloquea el `push` directo.
  - **Require approvals: 1** — otro integrante debe aprobar.
  - **Dismiss stale pull request approvals when new commits are pushed** — si llegan commits nuevos, la aprobación anterior deja de valer.
- **Require conversation resolution before merging** — no se mezcla con comentarios abiertos.
- **Do not allow bypassing the above settings** — la regla aplica también al dueño del repositorio.

Y mantiene desactivadas estas dos:

- **Allow force pushes** — con lo que el historial no puede reescribirse.
- **Allow deletions** — con lo que la rama no puede borrarse.

### 2. El ciclo de trabajo

```bash
# 1. Partir siempre de una main actualizada
git checkout main
git pull origin main

# 2. Crear la rama de trabajo
git checkout -b feature/ingesta-parquet

# 3. Trabajar y commitear en incrementos pequeños
git add feature_engineering/dataset.py
git commit -m "feat(dataset): normaliza nombres de columnas al leer los CSV"

# 4. Publicar la rama
git push -u origin feature/ingesta-parquet

# 5. Abrir el Pull Request hacia main desde GitHub y asignar revisor

# 6. Atender los comentarios de la revisión con commits nuevos sobre la misma rama

# 7. Tras la aprobación, hacer merge desde GitHub y borrar la rama remota

# 8. Limpiar en local
git checkout main
git pull origin main
git branch -d feature/ingesta-parquet
```

### 3. Convención de nombres de ramas

Se sigue la especificación **[Conventional Branch](https://conventionalbranch.org/)**.

**Formato:** `tipo/descripcion`

La única rama sin prefijo es `main`.

#### Tipos permitidos

| Tipo | Cuándo se usa |
|---|---|
| `feature/` (alias `feat/`) | funcionalidad nueva |
| `bugfix/` (alias `fix/`) | corrección de un defecto |
| `hotfix/` | corrección urgente sobre algo ya en `main` |
| `release/` | preparación de una entrega |
| `chore/` | tareas que no son código: dependencias, configuración y **documentación** |
| `ai/`, `claude/`, `codex/`, `copilot/`, `cursor/` | ramas generadas por un agente de IA |

#### Reglas de escritura

1. **Solo minúsculas, dígitos y guiones.** Los puntos se permiten en la descripción, normalmente para versiones.
2. **Sin guiones bajos, espacios, acentos ni caracteres especiales.**
3. **Sin separadores consecutivos** ni al inicio o al final, y un guion nunca va pegado a un punto.
4. **Descriptiva pero breve:** el nombre debe decir el propósito del trabajo.
5. **Incluye el número de issue** cuando exista: `feature/issue-123-nueva-ingesta`.

#### Ejemplos

| Válida | Inválida | Por qué falla |
|---|---|---|
| `feature/ingesta-parquet` | `Feature/Ingesta-Parquet` | mayúsculas |
| `bugfix/encoding-csv-2015` | `fix/encoding csv` | espacio |
| `chore/readme-flujo-trabajo` | `docs/readme_flujo` | tipo no definido y guion bajo |
| `chore/fijar-dependencias` | `chore/fijar--dependencias` | guiones consecutivos |
| `release/v1.0.0` | `release/v1.-0.0` | guion pegado a un punto |

**Reglas de ciclo de vida:**

- **Una rama, un propósito.** Si el cambio necesita dos títulos para describirse, son dos ramas.
- **Ramas cortas.** Nacen de una `main` actualizada y se borran al mezclarse. Mientras más viva una rama, más caro es el merge.

### 4. Convención de mensajes de commit

Se sigue **[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)**.

**Estructura:**

```
tipo(ámbito opcional): descripción

[cuerpo opcional]

[pies de página opcionales]
```

El cuerpo, si existe, empieza una línea en blanco después de la descripción; los pies de
página, una línea en blanco después del cuerpo.

#### Tipos

| Tipo | Cuándo se usa |
|---|---|
| `feat` | funcionalidad nueva |
| `fix` | corrección de un defecto |
| `docs` | documentación, incluido este README |
| `refactor` | reescritura sin cambiar el comportamiento |
| `test` | pruebas |
| `perf` | mejora de rendimiento |
| `build` | dependencias y empaquetado |
| `ci` | integración continua |
| `chore` | mantenimiento y tareas que no tocan el código fuente |
| `revert` | revierte un commit anterior, citando su SHA en el pie |

La especificación solo obliga a `feat` y `fix`; el resto son los tipos que recomienda
`@commitlint/config-conventional`, citado por la propia especificación.

El **ámbito** es un sustantivo entre paréntesis que nombra la parte del código afectada.
Ámbitos del proyecto: `dataset`, `features`, `config`, `modeling`, `mlflow`, `notebooks`,
`readme`, `deps`.

#### Ejemplos

```
feat(dataset): descarga los CSV mensuales y los guarda en Parquet
fix(features): evita usar información posterior al momento de la predicción
feat(mlflow): registra el baseline de P1 con sus parámetros y métricas
docs(readme): documenta el flujo de trabajo con Git/GitHub
chore(deps): fija las versiones de scikit-learn y mlflow
```

#### Cambios que rompen compatibilidad

Se marcan de dos formas, y el `!` puede acompañar a cualquier tipo:

```
feat(dataset)!: cambia el esquema del Parquet de salida
```

```
refactor(config): unifica las rutas del proyecto

BREAKING CHANGE: RAW_DATA_DIR se renombra a RAW_DIR; actualiza tus notebooks.
```

`BREAKING CHANGE` va siempre en mayúsculas.

#### Reglas de escritura

- Descripción en **imperativo** ("agrega", no "agregado" ni "agregando"), en minúsculas y sin punto final.
- Un commit = un cambio coherente. Nada de `varios cambios` ni `wip`.
- **Nunca** incluir credenciales, rutas absolutas ni datos personales en el mensaje.

### 5. Pull Requests y revisiones

Todo cambio a `main` entra por PR. Ningún PR se mezcla sin la aprobación de otro integrante.

**Título del PR:** mismo formato que un commit — `tipo(ámbito): descripción`.

**Descripción del PR:**

```markdown
## Qué cambia
Resumen en una o dos frases.

## Por qué
Qué problema resuelve o qué motiva el cambio.

## Cómo probarlo
Comandos o pasos para verificar el cambio desde un clon limpio.

## Pendientes
Lo que queda fuera de este PR, si aplica.
```

**Qué revisa quien aprueba:**

- El código corre desde un clon limpio siguiendo el README.
- `make lint` pasa sin errores.
- No se versionan datos de MiBici, `mlruns/`, modelos ni credenciales.
- Las rutas, fechas y semillas viven en `config.py`, no repetidas en varios archivos.
- Las funciones nuevas de `features.py` tienen *docstring* y no usan información del futuro.
- El nombre de la rama y los mensajes de commit siguen las convenciones.

### 6. Qué nunca entra al repositorio

Se versiona el **código que produce** los datos, los experimentos y los modelos; nunca el
**resultado de ejecutarlo**. Lo siguiente está excluido en el [`.gitignore`](.gitignore):

| Excluido | Motivo |
|---|---|
| `data/` | los datos de MiBici se reconstruyen con el comando de ingesta |
| `mlruns/`, `mlartifacts/` | son la salida de correr un experimento; MLflow los guarda localmente y cada *run* lleva como etiqueta el commit que lo produjo, así que el código de ese commit basta para reproducirlo |
| `*.pkl`, `*.joblib` y otros modelos serializados | se regeneran ejecutando el código de entrenamiento |
| `.env` y cualquier credencial | nunca deben salir del equipo local |

Si algo de esto llega a `main`, borrarlo en un commit posterior **no** lo quita del historial:
avisa al equipo antes de reescribir nada.

### Referencias del flujo de trabajo

- [Conventional Branch](https://conventionalbranch.org/) — especificación de nombres de ramas.
- [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) — especificación de mensajes de commit.
- [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) — modelo de ramificación.
- [About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches) — configuración de la rama principal.
- [About pull request reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews) — revisiones.

## Convenciones de código

El código sigue **[PEP 8](https://peps.python.org/pep-0008/)**, verificado automáticamente con
**[Ruff](https://docs.astral.sh/ruff/)**, que ya viene configurado en
[`pyproject.toml`](pyproject.toml). La versión de Python del proyecto es **3.10**.

### 1. Formateo automático

No se discute el formato en las revisiones: lo resuelve la herramienta.

```bash
make format   # ruff check --fix && ruff format  -> corrige y formatea
make lint     # ruff format --check && ruff check -> solo verifica
```

Ejecuta `make format` **antes de cada commit**. Un PR cuyo `make lint` falle no se aprueba.

**Configuración vigente** (en `pyproject.toml`, no la cambies sin acuerdo del equipo):

| Ajuste | Valor | Nota |
|---|---|---|
| `line-length` | `99` | PEP 8 sugiere 79; el equipo adopta los 99 de la plantilla, desviación que la propia PEP 8 permite cuando el equipo lo acuerda |
| `extend-select` | `["I"]` | activa el ordenamiento de imports |
| `known-first-party` | `["feature_engineering"]` | separa los imports del proyecto de los de terceros |
| `force-sort-within-sections` | `true` | mezcla `import x` y `from x import y` en un solo orden alfabético |

### 2. Nomenclatura

Todos los identificadores se escriben **en inglés** y sin acentos; los comentarios y
*docstrings*, en español.

| Elemento | Estilo | Ejemplo |
|---|---|---|
| Módulos y paquetes | `minusculas_cortas` | `dataset.py`, `features.py` |
| Funciones y métodos | `snake_case` | `build_hourly_trip_counts()` |
| Variables | `snake_case` | `trips_by_station` |
| Constantes | `UPPER_SNAKE_CASE` | `RAW_DATA_DIR`, `RANDOM_SEED` |
| Clases | `PascalCase` | `StationFeatureBuilder` |
| Uso interno | `_prefijo_guion_bajo` | `_normalize_column_names()` |

Nombres descriptivos sobre nombres cortos: `df` solo se acepta como parámetro de una función
que recibe un único DataFrame.

### 3. Orden de imports

Tres bloques separados por una línea en blanco, ordenados por Ruff:

```python
from pathlib import Path          # 1. biblioteca estándar

from loguru import logger         # 2. terceros
import pandas as pd
from tqdm import tqdm
import typer

from feature_engineering.config import PROCESSED_DATA_DIR, RAW_DATA_DIR   # 3. del proyecto
```

Prohibido `from modulo import *`.

### 4. Docstrings

Toda función pública lleva *docstring* siguiendo **[PEP 257](https://peps.python.org/pep-0257/)**
en **formato Google**. Las funciones de `features.py` indican, además,
**qué hacen y por qué son pertinentes para la pregunta que atienden**:

```python
def build_hourly_trip_counts(
    trips: pd.DataFrame,
    freq: str = "h",
) -> pd.DataFrame:
    """Agrega los viajes por estación de origen y ventana de tiempo.

    Atiende P1: el modelo de regresión predice el conteo de viajes iniciados,
    por lo que esta función construye la variable objetivo a su granularidad.

    Args:
        trips: Viajes limpios, con las columnas `origin_station_id` y `start_time`.
        freq: Frecuencia de agregación de pandas. Por defecto, horaria.

    Returns:
        DataFrame con una fila por estación y ventana, y la columna `n_trips`.

    Raises:
        KeyError: Si falta alguna de las columnas requeridas.
    """
```

**Regla crítica:** ninguna variable puede usar información posterior al momento de la
predicción. Si una función calcula una media móvil, un rezago o cualquier agregado temporal,
el *docstring* debe decir explícitamente qué ventana usa y por qué no hay fuga de información
del futuro.

### 5. Anotaciones de tipo

Las funciones del módulo llevan anotaciones de tipo
([PEP 484](https://peps.python.org/pep-0484/)) en parámetros y valor de retorno, como ya hace
la plantilla:

```python
def load_stations(path: Path) -> pd.DataFrame:
    ...
```

### 6. Configuración centralizada

**Ninguna ruta, fecha ni semilla se escribe a mano en más de un archivo.** Todo vive en
[`feature_engineering/config.py`](feature_engineering/config.py) y se importa desde ahí:

```python
# Correcto
from feature_engineering.config import RAW_DATA_DIR, RANDOM_SEED

# Incorrecto: ruta absoluta y semilla sueltas en el código
df = pd.read_csv("C:/Users/yo/Desktop/proyecto/data/raw/datos_2024_01.csv")
train_test_split(X, y, random_state=42)
```

Las rutas se construyen siempre con `pathlib.Path` y los operadores `/`, nunca concatenando
cadenas, para que el código corra igual en Windows, macOS y Linux.

### 7. Registro de eventos

Se usa `loguru`, ya incluido en las dependencias, **no `print()`**:

```python
from loguru import logger

logger.info("Descargando datos de {mes}...", mes=mes)
logger.success("Ingesta completa: {n} registros.", n=len(df))
```

Nunca registrar credenciales ni datos personales en los logs.

### 8. Notebooks

Los notebooks son **solo para exploración**; la lógica final vive en el módulo
`feature_engineering/`. Se nombran con la convención de la plantilla CCDS:

```
numero.version-iniciales-descripcion-corta.ipynb
```

Por ejemplo, `1.0-gsg-exploracion-viajes.ipynb`. Antes de hacer commit de un notebook,
limpia sus salidas para que el *diff* sea legible.

### Referencias de estilo

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 — Type Hints](https://peps.python.org/pep-0484/)
- [Ruff — formateador y linter](https://docs.astral.sh/ruff/)
- [Google Python Style Guide — docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## Integrantes y roles

| Integrante | Matrícula | Rol | Usuario de GitHub |
|---|---|---|---|
| Mariely Toribio Rodríguez | A01797848 | TBD | <!-- TODO --> |
| Alan Augusto Herrera Sánchez | A01338336 | TBD | <!-- TODO --> |
| Guillermo Sáenz González | A00823049 | TBD | WillSagzz |
| Victor Yoguel Salazar Alanis | A01797950 | TBD | CFOP2357 |

**Declaración de uso de IA.** El equipo usó asistentes de IA para edición de código y gestión
del repositorio bajo un esquema *human-in-the-loop*: las decisiones fueron dirigidas, revisadas
y aprobadas por los autores.

- Anthropic. (2026). *Claude Code (Claude Opus 5 y Claude Fable 5.1)* [Modelo de lenguaje grande]. https://claude.com/claude-code
- OpenAI. (2026). *ChatGPT* [Modelo de lenguaje grande]. https://chatgpt.com