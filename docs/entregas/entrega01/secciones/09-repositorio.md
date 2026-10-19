# Repositorio del equipo

El repositorio `TC5061-Proyecto-Equipo10` es público y está disponible en <https://github.com/WillSagzz/TC5061-Proyecto-Equipo10>. Contiene el código que reproduce los datos y el experimento base, la documentación del proyecto y la fuente de este documento; nunca contiene datos de MiBici, modelos, carpetas de MLflow ni credenciales. Esta sección resume lo que el README documenta con detalle: la estructura, las buenas prácticas, el flujo de trabajo y los integrantes.

## Estructura del repositorio

La estructura se generó con la plantilla Cookiecutter Data Science (DrivenData, s.f.) mediante `pipx install cookiecutter-data-science` y `ccds`, con la opción `include_code_scaffold` en `Yes`, y se subió como primer *commit*. Se respeta la separación de carpetas de la plantilla y se agregan únicamente la carpeta `docs/entregas/`, con la fuente de los entregables, y los archivos de `references/` que pide la actividad.

```text
TC5061-Proyecto-Equipo10/
├── .gitignore            <- Ampliado: datos, mlruns/, mlartifacts/, modelos serializados y .env
├── Makefile              <- make data, make features, make train, make lint, make format, make entrega01
├── README.md             <- Propósito, preguntas, datos, instalación, comandos, flujo de Git e integrantes
├── pyproject.toml        <- Metadatos del paquete mibici y configuración de Ruff
├── requirements.txt      <- Dependencias con versión fija
├── data/                 <- Excluida de Git; la reconstruye make data (raw, interim, processed, external)
├── docs/
│   └── entregas/entrega01/   <- Fuente en Markdown de este PDF y su compilador (build.py)
├── models/               <- Modelos serializados; excluidos de Git
├── notebooks/            <- Solo exploración; convención 1.0-iniciales-descripcion.ipynb
├── references/           <- ml_canvas.png, ml_canvas_mibici.pdf, arquitectura.mmd/.png, trazabilidad.mmd/.png
├── reports/figures/      <- Figuras generadas por el código
└── mibici/               <- Módulo del proyecto
    ├── config.py         <- Rutas, fechas, semilla y parámetros: única fuente de configuración
    ├── dataset.py        <- Ingesta reproducible: descarga, normaliza, limpia, escribe Parquet y versiona
    ├── features.py       <- Funciones reutilizables de feature engineering con docstring
    ├── plots.py          <- Figuras
    └── modeling/
        ├── train.py      <- Entrenamiento y registro de runs en MLflow
        └── predict.py    <- Inferencia con modelos entrenados
```

![Figura 14. Estructura del repositorio TC5061-Proyecto-Equipo10 en GitHub.](../figuras/repositorio_estructura.png)

## Buenas prácticas aplicadas

La Tabla 15 recorre las prácticas obligatorias de la actividad e indica cómo las aplica el equipo y dónde puede verificarse cada una.

**Tabla 15.** Buenas prácticas obligatorias y su aplicación en el proyecto.

| Ámbito | Práctica | Cómo la aplica el equipo | Dónde se verifica |
|--------|----------|--------------------------|-------------------|
| MLflow Tracking | Un experimento por proyecto y *runs* con nombre descriptivo | Experimento único `mibici-equipo10`; `train.py` asigna el nombre `pregunta-modelo-etapa` a cada *run* (Tabla 12) | Interfaz de MLflow |
| MLflow Tracking | Registrar todo lo necesario para reproducir el *run* | Parámetros `version_datos`, fechas del periodo y de corte, `semilla`, modelo e hiperparámetros, tomados de `config.py` (Tabla 13) | Parámetros del *run* |
| MLflow Tracking | Ligar cada *run* con el código y la pregunta | Etiquetas `git_commit`, `pregunta`, `equipo` y `tipo_modelo` escritas por `train.py` | Etiquetas del *run* |
| MLflow Tracking | Nombrar las métricas con el conjunto en que se calculan | `test_mae`, `test_recall_vaciarse`, `train_silhouette`; nunca `mae` a secas | Métricas del *run* |
| MLflow Tracking | No registrar credenciales ni datos personales | `dataset.py` elimina usuario, género y año de nacimiento antes de escribir Parquet; la URI del servidor vive en `.env`, excluido de Git | Interfaz de MLflow y código |
| Estructura y código | Notebooks solo para exploración, con la convención de la plantilla | La lógica final vive en `mibici/`; el notebook de exploración de la descarga se llama `1.0-gsg-descarga-dataset.ipynb` y la ingesta definitiva está en `dataset.py` | Carpeta `notebooks/` |
| Estructura y código | Configuración en un solo lugar | Rutas (`pathlib`), fechas del periodo y de corte, semilla, umbral e intervalo de P2 y nombre del experimento, todos en `config.py` | `mibici/config.py` |
| Estructura y código | Dependencias con versión fija y entorno reproducible | `requirements.txt` con versiones fijadas; Python 3.10; instalación documentada en el README desde un clon limpio | `requirements.txt` y README |
| Datos y Git | Ningún dato, modelo, carpeta de MLflow ni credencial en el historial | `.gitignore` excluye `data/`, `mlruns/`, `mlartifacts/`, `*.pkl`, `*.joblib` y `.env`; la revisión de cada PR lo comprueba | `.gitignore` e historial |
| Datos y Git | Todo cambio entra a la rama principal por *Pull Request* revisado | Regla de protección de `main` que exige PR con una aprobación; siete PR mezclados a la fecha, todos revisados por otro integrante (Tabla 16) | Pestaña de *Pull Requests* |

## Flujo de trabajo con Git/GitHub

El equipo sigue GitHub Flow (GitHub, s.f.-b): `main` es la única rama de larga duración, cada cambio nace en una rama corta y entra a `main` mediante un *Pull Request* revisado y aprobado por otro integrante. El flujo completo, con ejemplos, está documentado en el README del repositorio.

- **Rama principal protegida.** `main` tiene una regla de protección activa en GitHub (GitHub, s.f.-a) que exige *Pull Request* con al menos una aprobación, descarta las aprobaciones anteriores cuando llegan *commits* nuevos, exige resolver todas las conversaciones antes de mezclar, permite solo mezclas por *squash* y bloquea los *force push* y el borrado de la rama. La regla aplica también al dueño del repositorio.
- **Convención de nombres de ramas.** Conventional Branch (Conventional Branch, s.f.): `tipo/descripcion` en minúsculas y con guiones, con los tipos `feature/`, `bugfix/`, `hotfix/`, `release/` y `chore/`; por ejemplo, `feature/download-dataset` o `chore/entrega01-propuesta-pdf`.
- **Convención de mensajes de commit.** Conventional Commits 1.0.0 (Conventional Commits, s.f.): `tipo(ámbito): descripción` en imperativo, con los ámbitos `dataset`, `features`, `config`, `modeling`, `mlflow`, `notebooks`, `readme` y `deps`; por ejemplo, `docs(entregas): redacta la sección de arquitectura de la entrega 01`. Los cambios incompatibles se marcan con `!`.
- **Pull Requests y revisiones.** El título del PR sigue el formato de los *commits* y la descripción indica qué cambia, por qué y cómo probarlo desde un clon limpio. Quien revisa (GitHub, s.f.-c) comprueba que el código corre siguiendo el README, que `make lint` pasa, que no se versionan datos ni credenciales, que la configuración sigue centralizada en `config.py` y que las funciones nuevas de `features.py` tienen *docstring* y no usan información del futuro.
- **Evidencia de colaboración.** A la fecha de esta entrega, los cuatro integrantes han abierto al menos un *Pull Request* y han revisado y aprobado al menos uno de otro integrante (Tabla 16), y los cuatro tienen *commits* en el historial de `main`.

**Tabla 16.** *Pull Requests* mezclados a `main` a la fecha de la entrega.

| PR | Título | Rama | Autor | Revisor |
|----|--------|------|-------|---------|
| #1 | docs(readme): documenta el flujo de trabajo con Git/GitHub y las convenciones de código | `chore/readme-flujo-trabajo` | Victor Yoguel Salazar Alanis | Alan Augusto Herrera Sánchez |
| #2 | docs(entregas): migrar la propuesta de la entrega 01 de word a markdown | `chore/entrega01-propuesta-pdf` | Victor Yoguel Salazar Alanis | Guillermo Sáenz González |
| #3 | docs(references): agregar ml canvas del proyecto mibici | `docs/ml-canvas` | Alan Augusto Herrera Sánchez | Guillermo Sáenz González |
| #4 | feature(dataset): agregar script para extraccion y transformacion inicial de datos | `feature/download-dataset` | Guillermo Sáenz González | Mariely Toribio Rodríguez |
| #5 | docs: actualiza secciones de la entrega 01 y unifica P2 en tres clases | `chore/actualiza-secciones-entrega01` | Mariely Toribio Rodríguez | Victor Yoguel Salazar Alanis |
| #6 | refactor!: renombra el paquete feature_engineering a mibici | `fix/folder-naming` | Victor Yoguel Salazar Alanis | Guillermo Sáenz González |
| #7 | docs(entregas): redacta la sección de arquitectura de la entrega 01 | `docs/entrega01-arquitectura` | Victor Yoguel Salazar Alanis | Guillermo Sáenz González |

## Integrantes y roles

Todos los integrantes participan en la propuesta, revisan los *Pull Requests* de los demás y contribuyen código y documentación; el rol indica el frente del que cada uno es responsable principal.

**Tabla 17.** Integrantes del equipo y roles.

| Integrante | Matrícula | Usuario de GitHub | Rol principal |
|------------|-----------|-------------------|---------------|
| Mariely Toribio Rodríguez | A01797848 | `a01797848` | Formulación de las preguntas, definición de los datos, modelos y métricas |
| Alan Augusto Herrera Sánchez | A01338336 | `alanhsa9` | ML Canvas y experimento base en MLflow |
| Guillermo Sáenz González | A00823049 | `WillSagzz` | Administración del repositorio en GitHub, ingesta reproducible y versionado de datos |
| Victor Yoguel Salazar Alanis | A01797950 | `CFOP2357` | Flujo de trabajo con Git, arquitectura del sistema y compilación del entregable |
