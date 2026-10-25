"""Convención de MLflow del proyecto: experimentos, nombres de run, etiquetas y notas.

Todo lo que identifica un run vive aquí, en un solo lugar, para que ``train.py`` y
``dataset.py`` lo escriban igual y los runs se filtren con las mismas llaves:

- ``ensure_experiment``: crea (o completa) el experimento de una pregunta con su descripción y
  etiquetas. Los nombres salen de ``EXPERIMENTS`` en ``config.py``.
- ``run_name``, ``standard_tags`` y ``run_note``: nombre, etiquetas y nota de cada run. El
  nombre es una etiqueta para humanos (``P1-candidato-rf``); los parámetros distinguen dos
  runs con el mismo nombre y el identificador del run es su identidad.
- ``ProjectExperimentProvider``: elige el experimento cuando el proyecto se ejecuta con
  ``mlflow run``. MLflow crea el run antes de lanzar el código del punto de entrada, así que
  ningún comando del proyecto puede escoger el experimento; en su lugar MLflow consulta a los
  *default experiment providers* registrados (grupo ``mlflow.default_experiment_provider`` en
  ``pyproject.toml``) cuando no se pasó ``--experiment-name`` ni ``MLFLOW_EXPERIMENT_NAME``.
  Este proveedor lee la línea de comandos de ``mlflow run``, reconoce el proyecto y la
  pregunta (``-P question=p1``) o el punto de entrada ``ingest`` y devuelve el experimento que
  corresponde. Fuera de ``mlflow run`` de este proyecto no interviene.

El módulo lo importa el propio comando ``mlflow`` al arrancar (por el *entry point*), por eso
carga ``.env`` al importarse: así ``mlflow run`` ya conoce el servidor y las credenciales
antes de crear el run, sin exportar variables a mano. Lo demás se importa dentro de las
funciones para no encarecer cualquier otro comando de ``mlflow``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from mlflow.tracking.default_experiment.abstract_context import DefaultExperimentProvider

PROJ_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJ_ROOT / ".env", override=False)

# Valor de la etiqueta `pregunta` por experimento
QUESTION_LABELS = {"p1": "P1", "p2": "P2", "p3": "P3", "ingesta": "datos"}
STAGES = ("baseline", "candidato")  # valores de la etiqueta `tipo_modelo`
NOTE_TAG = "mlflow.note.content"  # descripción que MLflow muestra en la interfaz

# Opciones de `mlflow run` que llevan un valor en el token siguiente (para separar la ruta
# del proyecto de las opciones al leer la línea de comandos)
_RUN_OPTIONS_WITH_VALUE = {
    "-e",
    "--entry-point",
    "-v",
    "--version",
    "-P",
    "--param-list",
    "-A",
    "--docker-args",
    "--experiment-name",
    "--experiment-id",
    "-b",
    "--backend",
    "-c",
    "--backend-config",
    "--env-manager",
    "--storage-dir",
    "--run-id",
    "--run-name",
}


# -----------------------------------------------------------------------------
# Experimentos
# -----------------------------------------------------------------------------
def experiment_tags(key: str) -> dict[str, str]:
    """Descripción y etiquetas del experimento de una pregunta (o de la ingesta)."""
    from mibici.config import EXPERIMENT_DESCRIPTIONS, EXPERIMENT_TASKS, TEAM

    return {
        NOTE_TAG: EXPERIMENT_DESCRIPTIONS[key],
        "equipo": TEAM,
        "pregunta": QUESTION_LABELS[key],
        "tarea": EXPERIMENT_TASKS[key],
    }


def ensure_experiment(key: str) -> str:
    """Devuelve el id del experimento de ``key`` (``p1``, ``p2``, ``p3`` o ``ingesta``).

    Lo crea si no existe, con su descripción y etiquetas, y completa las etiquetas que falten
    si ya existía (por ejemplo, si lo creó el script de administración del servidor). En modo
    local (SQLite) los artefactos van a ``mlartifacts/``. Es idempotente y puede llamarse con
    cualquier usuario: si el usuario no puede etiquetar, lo avisa y sigue.
    """
    import mlflow
    from mlflow.exceptions import MlflowException

    from mibici.config import EXPERIMENTS, MLFLOW_ARTIFACTS_DIR

    name = EXPERIMENTS[key]
    client = mlflow.MlflowClient()
    tags = experiment_tags(key)
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        location = (
            MLFLOW_ARTIFACTS_DIR.as_uri()
            if mlflow.get_tracking_uri().startswith("sqlite")
            else None
        )
        return client.create_experiment(name, artifact_location=location, tags=tags)
    missing = {k: v for k, v in tags.items() if k not in experiment.tags}
    for tag_key, value in missing.items():
        try:
            client.set_experiment_tag(experiment.experiment_id, tag_key, value)
        except MlflowException as error:  # usuario de solo lectura
            print(f"No se pudo etiquetar el experimento {name}: {error}", file=sys.stderr)
            break
    return experiment.experiment_id


# -----------------------------------------------------------------------------
# Runs: nombre, etiquetas y nota
# -----------------------------------------------------------------------------
def git_state() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin commit (un run así no es reproducible)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, bool(changes)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "sin_git", True


def run_name(key: str, stage: str | None = None, label: str | None = None) -> str:
    """Nombre del run: ``P{n}-{etapa}-{modelo}`` al entrenar, ``ingesta-{version}`` al ingerir.

    Es una etiqueta estable y legible, no un identificador: no lleva hiperparámetros ni
    fechas, que ya son parámetros del run.

    Args:
        key: ``p1``, ``p2``, ``p3`` o ``ingesta``.
        stage: ``baseline`` o ``candidato`` (entrenamiento).
        label: Nombre del estimador en el catálogo (entrenamiento) o versión de los datos
            (ingesta).
    """
    if key == "ingesta":
        return f"ingesta-{label}"
    if stage not in STAGES:
        raise ValueError(f"Etapa desconocida: {stage}. Usa una de {STAGES}.")
    return f"{QUESTION_LABELS[key]}-{stage}-{label}"


def standard_tags(
    key: str, estimator: str | None = None, stage: str | None = None
) -> dict[str, str]:
    """Etiquetas comunes a todos los runs del proyecto.

    Solo lo que sirve para filtrar y MLflow no registra por su cuenta: ``equipo``, ``usuario``
    (el del servidor compartido, no el del sistema que MLflow guarda en ``mlflow.user``),
    ``pregunta``, ``git_commit`` y ``git_cambios_sin_commit`` (MLflow solo registra el commit
    bajo ``mlflow run``), y al entrenar ``estimador`` y ``tipo_modelo``. Lo que define el
    modelo (versión de datos, fechas, semilla, hiperparámetros) son parámetros, no etiquetas.
    """
    from mibici.config import MLFLOW_RUN_USER, TEAM

    commit, uncommitted = git_state()
    tags = {
        "equipo": TEAM,
        "usuario": MLFLOW_RUN_USER,
        "pregunta": QUESTION_LABELS[key],
        "git_commit": commit,
        "git_cambios_sin_commit": str(uncommitted),
    }
    if estimator is not None:
        tags["estimador"] = estimator
    if stage is not None:
        tags["tipo_modelo"] = stage
    return tags


def run_note(key: str, data_version: str, estimator: str | None = None, **details: object) -> str:
    """Una línea para la nota del run (``mlflow.note.content``), visible en la interfaz."""
    if key == "ingesta":
        return f"Ingesta de la versión {data_version} de los datos (datapackage.json)"
    parts = [QUESTION_LABELS[key]]
    if estimator is not None:
        parts.append(estimator)
    parts.extend(f"{name}={value}" for name, value in details.items())
    parts.append(f"datos {data_version}")
    return " · ".join(parts)


# -----------------------------------------------------------------------------
# Proveedor del experimento para `mlflow run`
# -----------------------------------------------------------------------------
def parse_mlflow_run_argv(argv: list[str]) -> dict[str, str] | None:
    """Lee la línea de comandos de ``mlflow run`` y devuelve ruta, punto de entrada y parámetros.

    Returns:
        ``{"uri": ..., "entry_point": ..., "<parámetro>": ...}`` o ``None`` si no es
        ``mlflow run``.
    """
    if len(argv) < 2 or argv[1] != "run":
        return None
    parsed: dict[str, str] = {"entry_point": "main"}
    tokens = argv[2:]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            if (
                not token.startswith("--")
                and len(token) > 2
                and token[:2] in _RUN_OPTIONS_WITH_VALUE
            ):
                option, value = token[:2], token[2:]  # forma pegada: -Pquestion=p1, -etrain
            else:
                option, _, value = token.partition("=")
            if not value and option in _RUN_OPTIONS_WITH_VALUE:
                index += 1
                value = tokens[index] if index < len(tokens) else ""
            if option in ("-e", "--entry-point"):
                parsed["entry_point"] = value
            elif option in ("-P", "--param-list"):
                name, _, param_value = value.partition("=")
                parsed[name] = param_value
        elif "uri" not in parsed:
            parsed["uri"] = token
        index += 1
    return parsed


class ProjectExperimentProvider(DefaultExperimentProvider):
    """Elige el experimento de ``mlflow run`` a partir de la pregunta o del punto de entrada.

    Solo actúa cuando el comando es ``mlflow run`` de este proyecto (la ruta contiene el
    ``MLproject`` del paquete) y la línea de comandos trae ``-P question=p1|p2|p3`` o el punto
    de entrada ``ingest``. Si se pasó ``--experiment-name`` o ``--experiment-id``, MLflow ni
    siquiera consulta a los proveedores y se respeta lo pedido.
    """

    def _experiment_key(self) -> str | None:
        parsed = parse_mlflow_run_argv(sys.argv)
        if parsed is None or not self._is_this_project(parsed.get("uri", ".")):
            return None
        if parsed["entry_point"] == "ingest":
            return "ingesta"
        question = parsed.get("question", "").lower()
        return question if question in ("p1", "p2", "p3") else None

    @staticmethod
    def _is_this_project(uri: str) -> bool:
        manifest = Path(uri).expanduser().resolve() / "MLproject"
        try:
            return manifest.samefile(PROJ_ROOT / "MLproject")
        except OSError:
            return False

    def in_context(self) -> bool:
        return self._experiment_key() is not None

    def get_experiment_id(self) -> str | None:
        key = self._experiment_key()
        return ensure_experiment(key) if key else None


__all__ = [
    "NOTE_TAG",
    "QUESTION_LABELS",
    "STAGES",
    "ProjectExperimentProvider",
    "ensure_experiment",
    "experiment_tags",
    "git_state",
    "parse_mlflow_run_argv",
    "run_name",
    "run_note",
    "standard_tags",
]
