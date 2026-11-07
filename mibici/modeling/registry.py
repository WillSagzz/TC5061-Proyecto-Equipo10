"""Carga y gestión de modelos desde el Model Registry de MLflow.

Proporciona las funciones para construir la URI de un modelo registrado por nombre y alias
(o versión), recuperarlo desde el servidor de MLflow configurado en ``MLFLOW_TRACKING_URI``
(o desde el caché local empaquetado en la imagen del contenedor) y mantenerlo en memoria
para servir predicciones en la API REST de la Entrega 04.

Uso desde la raíz del repositorio:

    python -m mibici.modeling.registry check --question p1
    python -m mibici.modeling.registry check --question all
    python -m mibici.modeling.registry download --question p1
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import mlflow
import mlflow.artifacts
from mlflow.exceptions import MlflowException
import mlflow.sklearn
import typer

from mibici.config import (
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SERVING_QUESTION,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_CACHE_DIR,
    REGISTERED_MODELS,
)
from mibici.features import P1_FEATURES, P2_FEATURES, P3_FEATURES
from mibici.modeling.thresholds import ThresholdClassifier

app = typer.Typer(help="Recuperación y verificación de modelos del Model Registry de MLflow.")

QUESTION_FEATURES: dict[str, list[str]] = {
    "p1": P1_FEATURES,
    "p2": P2_FEATURES,
    "p3": P3_FEATURES,
}

METADATA_FILENAME = "registry_metadata.json"


class ModelRegistryError(RuntimeError):
    """Error al recuperar o validar un modelo desde el Model Registry de MLflow."""


@dataclass(frozen=True)
class LoadedModel:
    """Modelo recuperado del Model Registry listo para inferencia.

    Attributes:
        question: Pregunta del proyecto (``p1``, ``p2`` o ``p3``).
        model_name: Nombre del modelo en el Model Registry.
        model_uri: URI de MLflow o ruta local desde donde se cargó.
        alias: Alias utilizado para resolver el modelo (por ejemplo, ``champion``), o ``None``.
        version: Versión del modelo si se especificó o resolvió, o ``None``.
        source: Origen efectivo de carga (``registry`` o ``cache``).
        estimator_type: Nombre de la clase del estimador cargado.
        features: Lista ordenada de columnas de entrada que exige el modelo.
        model: Instancia del modelo de scikit-learn cargada en memoria.
    """

    question: str
    model_name: str
    model_uri: str
    alias: str | None
    version: str | None
    source: str
    estimator_type: str
    features: list[str]
    model: Any


_MODEL_CACHE: dict[tuple[str, str | None, str | None], LoadedModel] = {}


def _validate_question(question: str) -> str:
    """Normaliza el identificador de pregunta y verifica que exista en el catálogo."""
    normalized = question.strip().lower()
    if normalized not in REGISTERED_MODELS:
        valid = ", ".join(sorted(REGISTERED_MODELS))
        raise ValueError(f"Pregunta '{question}' desconocida. Opciones válidas: {valid}.")
    return normalized


def build_model_uri(
    question: str = DEFAULT_SERVING_QUESTION,
    alias: str | None = DEFAULT_MODEL_ALIAS,
    version: int | str | None = None,
) -> str:
    """Construye la URI ``models:/`` de MLflow para la pregunta indicada.

    Prioriza ``version`` cuando se proporciona explícitamente; en caso contrario utiliza
    ``alias`` (por defecto ``champion``).

    Args:
        question: Pregunta del proyecto (``p1``, ``p2`` o ``p3``).
        alias: Alias registrado en MLflow (por ejemplo, ``champion`` o ``challenger``).
        version: Número de versión en el Model Registry (por ejemplo, ``1``).

    Returns:
        URI con formato ``models:/<nombre>@<alias>`` o ``models:/<nombre>/<versión>``.

    Raises:
        ValueError: Si la pregunta no es válida o si no se indica ni alias ni versión.
    """
    normalized = _validate_question(question)
    model_name = REGISTERED_MODELS[normalized]

    if version is not None and str(version).strip() != "":
        clean_version = str(version).strip()
        return f"models:/{model_name}/{clean_version}"

    if alias is not None and alias.strip() != "":
        clean_alias = alias.strip().lstrip("@")
        return f"models:/{model_name}@{clean_alias}"

    raise ValueError(
        f"Debes indicar un alias o una versión para construir la URI de '{model_name}'."
    )


def _estimator_display_name(model: Any) -> str:
    """Devuelve el nombre legible de la clase del estimador o pipeline cargado."""
    name = type(model).__name__
    if isinstance(model, ThresholdClassifier):
        inner = getattr(model, "estimator_", None) or getattr(model, "estimator", None)
        if inner is not None:
            return f"{name}({type(inner).__name__})"
    return name


def _load_from_local_cache(
    question: str,
    model_name: str,
    alias: str | None,
    version: str | None,
    cache_dir: Path,
) -> LoadedModel:
    """Carga un modelo previamente descargado del Model Registry en ``cache_dir``."""
    model_dir = Path(cache_dir) / question
    mlmodel_file = model_dir / "MLmodel"
    if not mlmodel_file.exists():
        raise ModelRegistryError(
            f"No existe un modelo empaquetado en '{model_dir}'. "
            "Descárgalo antes con `python -m mibici.modeling.registry download` "
            "o configura MLFLOW_TRACKING_URI."
        )
    try:
        loaded = mlflow.sklearn.load_model(str(model_dir))
    except Exception as exc:
        raise ModelRegistryError(
            f"Error al cargar el modelo local de '{model_dir}': {exc}"
        ) from exc

    if not hasattr(loaded, "predict"):
        raise ModelRegistryError(
            f"El objeto cargado desde '{model_dir}' no expone un método 'predict'."
        )

    meta_path = model_dir / METADATA_FILENAME
    resolved_version = version
    resolved_alias = alias
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            resolved_version = metadata.get("version", resolved_version)
            resolved_alias = metadata.get("alias", resolved_alias)
        except (OSError, ValueError):
            pass

    estimator_type = _estimator_display_name(loaded)
    logger.info(
        f"[{question.upper()}] Modelo '{model_name}' cargado desde caché local "
        f"'{model_dir}' -> {estimator_type}"
    )
    return LoadedModel(
        question=question,
        model_name=model_name,
        model_uri=str(model_dir),
        alias=resolved_alias,
        version=resolved_version,
        source="cache",
        estimator_type=estimator_type,
        features=list(QUESTION_FEATURES[question]),
        model=loaded,
    )


def load_registered_model(
    question: str = DEFAULT_SERVING_QUESTION,
    alias: str | None = DEFAULT_MODEL_ALIAS,
    version: int | str | None = None,
    tracking_uri: str | None = None,
    cache_dir: Path = MODEL_REGISTRY_CACHE_DIR,
    allow_cache_fallback: bool = True,
) -> LoadedModel:
    """Recupera un modelo del Model Registry de MLflow por su nombre y alias (o versión).

    Intenta primero cargar el modelo desde el servidor configurado en ``tracking_uri`` (por
    defecto ``MLFLOW_TRACKING_URI``). Si ocurre un error de conexión o el servidor no está
    disponible y ``allow_cache_fallback`` es ``True``, intenta cargarlo desde el directorio
    local ``cache_dir / question`` donde se haya descargado previamente del Registry.

    Args:
        question: Pregunta del proyecto (``p1``, ``p2`` o ``p3``).
        alias: Alias del modelo en el Registry (por defecto ``champion``).
        version: Versión específica del modelo (si se indica, tiene prioridad sobre ``alias``).
        tracking_uri: URI del servidor de MLflow; si es ``None``, usa ``MLFLOW_TRACKING_URI``.
        cache_dir: Carpeta donde se buscan modelos empaquetados descargados del Registry.
        allow_cache_fallback: Si es ``True``, usa ``cache_dir`` como respaldo cuando falla
            la conexión al servidor de MLflow.

    Returns:
        Instancia de ``LoadedModel`` con el estimador y sus metadatos.

    Raises:
        ValueError: Si la pregunta o los identificadores de versión/alias son inválidos.
        ModelRegistryError: Si no se puede recuperar el modelo ni del Registry ni del caché.
    """
    normalized = _validate_question(question)
    model_name = REGISTERED_MODELS[normalized]
    clean_version = (
        str(version).strip() if version is not None and str(version).strip() != "" else None
    )
    clean_alias = (
        alias.strip().lstrip("@")
        if clean_version is None and alias is not None and alias.strip() != ""
        else None
    )
    model_uri = build_model_uri(normalized, alias=clean_alias, version=clean_version)

    effective_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    try:
        mlflow.set_tracking_uri(effective_uri)
        loaded = mlflow.sklearn.load_model(model_uri)
    except (MlflowException, OSError, RuntimeError) as exc:
        if allow_cache_fallback and (Path(cache_dir) / normalized / "MLmodel").exists():
            logger.warning(
                f"No fue posible cargar '{model_uri}' desde '{effective_uri}' ({exc}). "
                f"Usando copia empaquetada en '{Path(cache_dir) / normalized}'."
            )
            return _load_from_local_cache(
                normalized, model_name, clean_alias, clean_version, Path(cache_dir)
            )
        raise ModelRegistryError(
            f"No se pudo recuperar el modelo '{model_uri}' desde '{effective_uri}': {exc}"
        ) from exc

    if not hasattr(loaded, "predict"):
        raise ModelRegistryError(f"El modelo '{model_uri}' no tiene un método 'predict' válido.")

    estimator_type = _estimator_display_name(loaded)
    logger.info(
        f"[{normalized.upper()}] Modelo '{model_uri}' cargado desde Model Registry "
        f"-> {estimator_type}"
    )
    return LoadedModel(
        question=normalized,
        model_name=model_name,
        model_uri=model_uri,
        alias=clean_alias,
        version=clean_version,
        source="registry",
        estimator_type=estimator_type,
        features=list(QUESTION_FEATURES[normalized]),
        model=loaded,
    )


def get_registered_model(
    question: str = DEFAULT_SERVING_QUESTION,
    alias: str | None = DEFAULT_MODEL_ALIAS,
    version: int | str | None = None,
    force_reload: bool = False,
) -> LoadedModel:
    """Devuelve el modelo registrado desde memoria o lo carga la primera vez.

    Evita descargar repetidamente el artefacto desde el servidor de MLflow en cada petición
    HTTP de la API REST.

    Args:
        question: Pregunta del proyecto (``p1``, ``p2`` o ``p3``).
        alias: Alias del modelo (por defecto ``champion``).
        version: Versión específica del modelo, o ``None``.
        force_reload: Si es ``True``, fuerza la recarga aunque ya esté en memoria.

    Returns:
        Instancia de ``LoadedModel`` lista para inferencia.
    """
    normalized = _validate_question(question)
    clean_version = (
        str(version).strip() if version is not None and str(version).strip() != "" else None
    )
    clean_alias = (
        alias.strip().lstrip("@")
        if clean_version is None and alias is not None and alias.strip() != ""
        else None
    )
    cache_key = (normalized, clean_alias, clean_version)
    if force_reload or cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_registered_model(
            question=normalized, alias=clean_alias, version=clean_version
        )
    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Limpia el caché en memoria de modelos cargados."""
    _MODEL_CACHE.clear()


def download_registered_model(
    question: str = DEFAULT_SERVING_QUESTION,
    alias: str | None = DEFAULT_MODEL_ALIAS,
    version: int | str | None = None,
    output_dir: Path = MODEL_REGISTRY_CACHE_DIR,
    tracking_uri: str | None = None,
) -> Path:
    """Descarga un modelo del Model Registry a ``output_dir / question`` para empaquetarlo.

    El objetivo es:
    Descargar el artefacto desde el Model Registry de MLflow y empaquetarlo dentro de la
    imagen del contenedor para que arranque sin depender de credenciales remotas.

    Args:
        question: Pregunta del proyecto (``p1``, ``p2`` o ``p3``).
        alias: Alias del modelo (por defecto ``champion``).
        version: Versión específica del modelo, o ``None``.
        output_dir: Directorio base donde se guardará la carpeta de la pregunta.
        tracking_uri: URI del servidor de MLflow; si es ``None``, usa ``MLFLOW_TRACKING_URI``.

    Returns:
        Ruta del directorio local donde quedó descargado el modelo de MLflow.

    Raises:
        ModelRegistryError: Si falla la descarga desde el Model Registry.
    """
    normalized = _validate_question(question)
    model_name = REGISTERED_MODELS[normalized]
    clean_version = (
        str(version).strip() if version is not None and str(version).strip() != "" else None
    )
    clean_alias = (
        alias.strip().lstrip("@")
        if clean_version is None and alias is not None and alias.strip() != ""
        else None
    )
    model_uri = build_model_uri(normalized, alias=clean_alias, version=clean_version)
    effective_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    target_dir = Path(output_dir) / normalized
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        mlflow.set_tracking_uri(effective_uri)
        downloaded = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(target_dir))
        )
    except (MlflowException, OSError, RuntimeError) as exc:
        raise ModelRegistryError(
            f"No se pudo descargar '{model_uri}' desde '{effective_uri}': {exc}"
        ) from exc

    metadata = {
        "question": normalized,
        "model_name": model_name,
        "model_uri": model_uri,
        "alias": clean_alias,
        "version": clean_version,
    }
    (downloaded / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success(f"[{normalized.upper()}] Modelo '{model_uri}' descargado en: {downloaded}")
    return downloaded


@app.command("check")
def check_command(
    question: Annotated[
        str, typer.Option(help="Pregunta a verificar: p1, p2, p3 o all.")
    ] = DEFAULT_SERVING_QUESTION,
    alias: Annotated[str, typer.Option(help="Alias en el Model Registry.")] = DEFAULT_MODEL_ALIAS,
    version: Annotated[
        str, typer.Option(help="Versión específica (opcional, reemplaza al alias).")
    ] = "",
) -> None:
    """Verifica que el modelo registrado en MLflow se cargue correctamente en memoria."""
    questions = list(REGISTERED_MODELS) if question.strip().lower() == "all" else [question]
    for q in questions:
        loaded = load_registered_model(
            question=q,
            alias=alias if not version else None,
            version=version if version else None,
        )
        logger.success(
            f"[{loaded.question.upper()}] OK · {loaded.model_name} ({loaded.model_uri}) -> "
            f"{loaded.estimator_type} · {len(loaded.features)} variables ({loaded.source})"
        )


@app.command("download")
def download_command(
    question: Annotated[
        str, typer.Option(help="Pregunta a descargar: p1, p2, p3 o all.")
    ] = DEFAULT_SERVING_QUESTION,
    alias: Annotated[str, typer.Option(help="Alias en el Model Registry.")] = DEFAULT_MODEL_ALIAS,
    version: Annotated[
        str, typer.Option(help="Versión específica (opcional, reemplaza al alias).")
    ] = "",
    output_dir: Annotated[
        Path, typer.Option(help="Directorio destino para empaquetar el modelo.")
    ] = MODEL_REGISTRY_CACHE_DIR,
) -> None:
    """Descarga el modelo desde el Model Registry hacia una carpeta local."""
    questions = list(REGISTERED_MODELS) if question.strip().lower() == "all" else [question]
    for q in questions:
        download_registered_model(
            question=q,
            alias=alias if not version else None,
            version=version if version else None,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    app()
