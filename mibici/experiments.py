"""Experimentos de MLflow del proyecto y etiquetas comunes de los runs.

Hay un experimento por pregunta y uno para la ingesta (``EXPERIMENTS`` en ``config.py``).
``ProjectExperimentProvider`` lo elige cuando el proyecto corre con ``mlflow run``: MLflow crea
el run antes de lanzar el punto de entrada y, si no se pasó ``--experiment-name``, consulta a
los proveedores registrados en el grupo ``mlflow.default_experiment_provider``
(``pyproject.toml``). Este lee la línea de comandos, reconoce el proyecto y la pregunta o el
punto de entrada ``ingest``, y devuelve el experimento. Como el comando ``mlflow`` importa este
módulo al arrancar, aquí se carga ``.env``; lo demás se importa dentro de las funciones.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
from mlflow.tracking.default_experiment.abstract_context import DefaultExperimentProvider

PROJ_ROOT = Path(__file__).resolve().parents[1]

QUESTION_LABELS = {"p1": "P1", "p2": "P2", "p3": "P3", "ingesta": "datos"}  # etiqueta `pregunta`
NOTE_TAG = "mlflow.note.content"

# Opciones de `mlflow run` que llevan valor en el token siguiente
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


class ExperimentMismatchError(RuntimeError):
    """El run que creó ``mlflow run`` está en un experimento distinto al de la pregunta."""


# -----------------------------------------------------------------------------
# Experimentos
# -----------------------------------------------------------------------------
def experiment_tags(key: str) -> dict[str, str]:
    """Descripción y etiquetas del experimento de ``key`` (``p1``, ``p2``, ``p3`` o ``ingesta``)."""
    from mibici.config import EXPERIMENT_DESCRIPTIONS, EXPERIMENT_TASKS, TEAM

    return {
        NOTE_TAG: EXPERIMENT_DESCRIPTIONS[key],
        "equipo": TEAM,
        "pregunta": QUESTION_LABELS[key],
        "tarea": EXPERIMENT_TASKS[key],
    }


def ensure_experiment(key: str) -> str:
    """Devuelve el id del experimento de ``key``; lo crea con sus etiquetas si no existe.

    Si ya existía completa las etiquetas que falten. En modo local (SQLite) los artefactos van
    a ``mlartifacts/``. Un usuario sin permiso para etiquetar recibe un aviso y sigue.
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
    for tag_key, value in tags.items():
        if tag_key in experiment.tags:
            continue
        try:
            client.set_experiment_tag(experiment.experiment_id, tag_key, value)
        except MlflowException as error:
            print(f"No se pudo etiquetar el experimento {name}: {error}", file=sys.stderr)
            break
    return experiment.experiment_id


def resolve_experiment(key: str) -> str:
    """Deja MLflow apuntando al experimento de ``key`` y devuelve su nombre.

    Bajo ``mlflow run`` el run ya existe (``MLFLOW_RUN_ID``) en el experimento que eligió la
    línea de comandos; si no es el de la pregunta, lanza ``ExperimentMismatchError`` antes de
    registrar nada. Sin ``mlflow run`` crea o selecciona el experimento.
    """
    import mlflow

    from mibici.config import EXPERIMENTS, MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    expected = EXPERIMENTS[key]
    run_id = os.getenv("MLFLOW_RUN_ID")
    if run_id:
        client = mlflow.MlflowClient()
        actual = client.get_experiment(client.get_run(run_id).info.experiment_id).name
        if actual != expected:
            raise ExperimentMismatchError(
                f"El run de mlflow run está en el experimento '{actual}' y {key} pertenece a "
                f"'{expected}'. Quita --experiment-name (el proyecto lo elige solo) o usa "
                f"--experiment-name {expected}."
            )
    else:
        ensure_experiment(key)
        mlflow.set_experiment(expected)
    return expected


# -----------------------------------------------------------------------------
# Etiquetas comunes de los runs
# -----------------------------------------------------------------------------
def git_state() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin commit."""
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


def run_tags(key: str, stage: str, family: str | None = None) -> dict[str, str]:
    """Etiquetas de un run: equipo, usuario, pregunta, commit, etapa y familia del modelo.

    ``mlflow.user`` repite el usuario del servidor: MLflow la llena con la cuenta del sistema al
    crear el run y es lo que la interfaz muestra en la columna "User".
    """
    from mibici.config import MLFLOW_RUN_USER, TEAM

    commit, uncommitted = git_state()
    tags = {
        "equipo": TEAM,
        "usuario": MLFLOW_RUN_USER,
        "mlflow.user": MLFLOW_RUN_USER,
        "pregunta": QUESTION_LABELS[key],
        "git_commit": commit,
        "git_cambios_sin_commit": str(uncommitted),
        "etapa": stage,
    }
    if family is not None:
        tags["familia_modelo"] = family
    return tags


def run_note(key: str, data_version: str, model: str | None = None, **details: object) -> str:
    """Una línea para la nota del run (``mlflow.note.content``)."""
    if key == "ingesta":
        return f"Ingesta de la versión {data_version} de los datos (datapackage.json)"
    parts = [QUESTION_LABELS[key]]
    if model is not None:
        parts.append(model)
    parts.extend(f"{name}={value}" for name, value in details.items())
    parts.append(f"datos {data_version}")
    return " · ".join(parts)


# -----------------------------------------------------------------------------
# Proveedor del experimento para `mlflow run`
# -----------------------------------------------------------------------------
def parse_mlflow_run_argv(argv: list[str]) -> dict[str, str] | None:
    """Ruta, punto de entrada y parámetros de una línea de comandos ``mlflow run``; ``None`` si no lo es."""
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
                option, value = token[:2], token[2:]  # forma pegada: -Pquestion=p1
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


def _is_this_project(uri: str) -> bool:
    manifest = Path(uri).expanduser().resolve() / "MLproject"
    try:
        return manifest.samefile(PROJ_ROOT / "MLproject")
    except OSError:
        return False


def _default_question(entry_point: str) -> str | None:
    """Valor por defecto del parámetro ``question`` del punto de entrada, según el MLproject."""
    import yaml

    try:
        spec = yaml.safe_load((PROJ_ROOT / "MLproject").read_text(encoding="utf-8"))
        parameter = spec["entry_points"][entry_point]["parameters"]["question"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None
    return parameter.get("default") if isinstance(parameter, dict) else None


def is_project_run(argv: list[str]) -> bool:
    """``True`` si la línea de comandos es un ``mlflow run`` de este proyecto."""
    parsed = parse_mlflow_run_argv(argv)
    return parsed is not None and _is_this_project(parsed.get("uri", "."))


def project_run_key(argv: list[str]) -> str | None:
    """Experimento (``p1``, ``p2``, ``p3`` o ``ingesta``) que pide un ``mlflow run`` de este proyecto."""
    parsed = parse_mlflow_run_argv(argv)
    if parsed is None or not _is_this_project(parsed.get("uri", ".")):
        return None
    entry_point = parsed["entry_point"]
    if entry_point == "ingest":
        return "ingesta"
    question = parsed.get("question") or _default_question(entry_point) or ""
    return question if question in ("p1", "p2", "p3") else None


class ProjectExperimentProvider(DefaultExperimentProvider):
    """Elige el experimento de ``mlflow run`` a partir de la pregunta o del punto de entrada.

    Solo actúa en ``mlflow run`` de este proyecto; con ``--experiment-name`` MLflow no consulta
    a los proveedores.
    """

    def in_context(self) -> bool:
        return project_run_key(sys.argv) is not None

    def get_experiment_id(self) -> str | None:
        key = project_run_key(sys.argv)
        return ensure_experiment(key) if key else None


# El comando `mlflow` importa este módulo al arrancar: aquí conoce el .env y, en un `mlflow run`
# de este proyecto sin servidor configurado, la misma base SQLite que usa config.py (sin esto
# el run iría a ./mlruns, el valor por defecto de MLflow).
load_dotenv(PROJ_ROOT / ".env", override=False)
if is_project_run(sys.argv):
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")

__all__ = [
    "NOTE_TAG",
    "QUESTION_LABELS",
    "ExperimentMismatchError",
    "ProjectExperimentProvider",
    "ensure_experiment",
    "experiment_tags",
    "git_state",
    "is_project_run",
    "parse_mlflow_run_argv",
    "project_run_key",
    "resolve_experiment",
    "run_note",
    "run_tags",
]
