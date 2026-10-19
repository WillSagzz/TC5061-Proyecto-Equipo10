# Repositorio del equipo

El repositorio `TC5061-Proyecto-Equipo10` es público y está disponible en <https://github.com/WillSagzz/TC5061-Proyecto-Equipo10>. Contiene el código que reproduce los datos y el experimento base, la documentación del proyecto y la fuente de este documento; nunca contiene datos de MiBici, modelos, carpetas de MLflow ni credenciales. Esta sección resume lo que el README documenta con detalle: la estructura, las buenas prácticas, el flujo de trabajo y los integrantes.

## Estructura del repositorio

La estructura se generó con la plantilla Cookiecutter Data Science (DrivenData, s.f.) mediante `pipx install cookiecutter-data-science` y `ccds`, con la opción `include_code_scaffold` en `Yes`, y se subió como primer *commit*. Se respeta la separación de carpetas de la plantilla y se agregan únicamente la carpeta `docs/entregas/`, con la fuente de los entregables, y los archivos de `references/` que pide la actividad.

```text
TC5061-Proyecto-Equipo10/
├── .gitignore            <- Excluye datos, mlruns/, modelos serializados y .env
├── Makefile              <- make data, features, train, lint, format, entrega01
├── README.md             <- Propósito, datos, instalación, comandos, flujo Git
├── pyproject.toml        <- Metadatos del paquete mibici y configuración de Ruff
├── requirements.txt      <- Dependencias con versión fija
├── data/                 <- Excluida de Git; la reconstruye make data
├── docs/entregas/        <- Fuente en Markdown de este PDF y su compilador
├── models/               <- Modelos serializados; excluidos de Git
├── notebooks/            <- Solo exploración: 1.0-gsg-descarga-dataset.ipynb
├── references/           <- ML Canvas y diagramas (html, mmd, pdf, png)
├── reports/figures/      <- Figuras generadas por el código
└── mibici/               <- Módulo del proyecto
    ├── config.py         <- Rutas, fechas, semilla y parámetros
    ├── dataset.py        <- Ingesta reproducible y versionado de datos
    ├── features.py       <- Funciones de feature engineering con docstring
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
| Datos y Git | Todo cambio entra a la rama principal por *Pull Request* revisado | Regla de protección de `main` que exige PR con una aprobación; todos los PR mezclados han sido revisados por otro integrante | Pestaña de *Pull Requests* |

## Flujo de trabajo con Git/GitHub

El equipo sigue GitHub Flow (GitHub, s.f.-b): `main` es la única rama de larga duración, cada cambio nace en una rama corta y entra a `main` mediante un *Pull Request* revisado y aprobado por otro integrante. El flujo completo, con ejemplos, está documentado en el README del repositorio.

- **Rama principal protegida.** `main` tiene una regla de protección activa en GitHub (GitHub, s.f.-a) que exige *Pull Request* con al menos una aprobación, descarta las aprobaciones anteriores cuando llegan *commits* nuevos, exige resolver todas las conversaciones antes de mezclar, permite solo mezclas por *squash* y bloquea los *force push* y el borrado de la rama. La regla aplica también al dueño del repositorio.
- **Convención de nombres de ramas.** Conventional Branch (Conventional Branch, s.f.): `tipo/descripcion` en minúsculas y con guiones, con los tipos `feature/`, `bugfix/`, `hotfix/`, `release/` y `chore/`; por ejemplo, `feature/download-dataset` o `chore/entrega01-propuesta-pdf`.
- **Convención de mensajes de commit.** Conventional Commits 1.0.0 (Conventional Commits, s.f.): `tipo(ámbito): descripción` en imperativo, con los ámbitos `dataset`, `features`, `config`, `modeling`, `mlflow`, `notebooks`, `readme` y `deps`; por ejemplo, `docs(entregas): redacta la sección de arquitectura de la entrega 01`. Los cambios incompatibles se marcan con `!`.
- **Pull Requests y revisiones.** El título del PR sigue el formato de los *commits* y la descripción indica qué cambia, por qué y cómo probarlo desde un clon limpio. Quien revisa (GitHub, s.f.-c) comprueba que el código corre siguiendo el README, que `make lint` pasa, que no se versionan datos ni credenciales, que la configuración sigue centralizada en `config.py` y que las funciones nuevas de `features.py` tienen *docstring* y no usan información del futuro.
- **Evidencia de colaboración.** Los cuatro integrantes han abierto al menos un *Pull Request*, han revisado y aprobado al menos uno de otro integrante y tienen *commits* en el historial de `main`. La evidencia vive en la pestaña de *Pull Requests* del repositorio (<https://github.com/WillSagzz/TC5061-Proyecto-Equipo10/pulls?q=is%3Apr>), que se mantiene actualizada conforme avanza el proyecto.

## Integrantes y roles

Todos los integrantes participan en la propuesta, revisan los *Pull Requests* de los demás y contribuyen código y documentación; el rol indica el frente del que cada uno es responsable principal.

**Tabla 16.** Integrantes del equipo y roles.

| Integrante | Matrícula | Usuario de GitHub | Rol principal |
|------------|-----------|-------------------|---------------|
| Mariely Toribio Rodríguez | A01797848 | `a01797848` | Formulación de las preguntas, definición de los datos, modelos y métricas |
| Alan Augusto Herrera Sánchez | A01338336 | `alanhsa9` | ML Canvas y experimento base en MLflow |
| Guillermo Sáenz González | A00823049 | `WillSagzz` | Administración del repositorio en GitHub, ingesta reproducible y versionado de datos |
| Victor Yoguel Salazar Alanis | A01797950 | `CFOP2357` | Flujo de trabajo con Git, arquitectura del sistema y compilación del entregable |
