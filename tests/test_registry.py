"""Pruebas unitarias para el cargador de modelos del Model Registry (registry.py).

Verifica la construcción de URIs por alias y versión, la carga desde el Model Registry,
la resolución de la versión detrás de un alias, el respaldo desde caché local empaquetado
(también como primera opción cuando no hay ``MLFLOW_TRACKING_URI``), el caché en memoria y
el manejo de errores, sin depender de conexión al servidor remoto de Azure durante la
ejecución en CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

from loguru import logger
import mlflow
import mlflow.artifacts
from mlflow.exceptions import MlflowException
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mibici.config import REGISTERED_MODELS
from mibici.features import P1_FEATURES, P2_FEATURES, P3_FEATURES
from mibici.modeling import registry
from mibici.modeling.registry import (
    ModelRegistryError,
    build_model_uri,
    clear_model_cache,
    download_registered_model,
    get_registered_model,
    load_registered_model,
)


def _fake_mlflow_client(
    version: str | None = None,
    error: Exception | None = None,
    calls: list[tuple[str, str]] | None = None,
) -> type:
    """Construye un sustituto de ``mlflow.MlflowClient`` para simular la consulta de alias.

    Args:
        version: Versión que devolverá ``get_model_version_by_alias``.
        error: Excepción que lanzará la consulta en lugar de responder.
        calls: Lista donde se anotan los pares ``(nombre, alias)`` consultados.
    """

    class _FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> SimpleNamespace:
            if calls is not None:
                calls.append((name, alias))
            if error is not None:
                raise error
            return SimpleNamespace(name=name, aliases=[alias], version=version)

    return _FakeClient


@contextmanager
def _capture_logs(level: str) -> Iterator[list[str]]:
    """Captura los mensajes de loguru de ``level`` o superior emitidos dentro del bloque."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level=level)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Limpia el caché en memoria antes y después de cada prueba."""
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture(autouse=True)
def _offline_registry_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita consultas reales al Model Registry al resolver la versión detrás de un alias.

    Por defecto la consulta falla como si no hubiera servidor; cada prueba que lo necesite
    sustituye ``mlflow.MlflowClient`` por un doble con la respuesta que quiera simular.
    """
    monkeypatch.setattr(
        mlflow,
        "MlflowClient",
        _fake_mlflow_client(error=MlflowException("Model Registry no alcanzable")),
    )


def _save_dummy_mlflow_model(target_dir: Path) -> Path:
    """Guarda un modelo ligero de scikit-learn en formato MLflow dentro de ``target_dir``."""
    X = pd.DataFrame(np.ones((4, len(P1_FEATURES))), columns=P1_FEATURES)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    estimator = DummyRegressor(strategy="mean").fit(X, y)
    mlflow.sklearn.save_model(
        estimator,
        path=str(target_dir),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    return target_dir


def test_build_model_uri_with_alias_and_version() -> None:
    """Construye la URI oficial models:/ por alias (@champion) o por número de versión."""
    assert build_model_uri("p1") == f"models:/{REGISTERED_MODELS['p1']}@champion"
    assert build_model_uri("P2", alias="@challenger") == (
        f"models:/{REGISTERED_MODELS['p2']}@challenger"
    )
    assert build_model_uri("p3", version=2) == f"models:/{REGISTERED_MODELS['p3']}/2"


def test_build_model_uri_rejects_invalid_question_or_empty_target() -> None:
    """Caso límite: rechaza preguntas inexistentes o llamadas sin alias ni versión."""
    with pytest.raises(ValueError, match="desconocida"):
        build_model_uri("p4")
    with pytest.raises(ValueError, match="alias o una versión"):
        build_model_uri("p1", alias="", version=None)


@pytest.mark.parametrize(
    ("question", "expected_features"),
    [("p1", P1_FEATURES), ("p2", P2_FEATURES), ("p3", P3_FEATURES)],
)
def test_load_registered_model_from_registry_mock(
    monkeypatch: pytest.MonkeyPatch, question: str, expected_features: list[str]
) -> None:
    """Verifica que load_registered_model recupere el modelo y asigne sus features."""
    dummy = DummyRegressor(strategy="constant", constant=3.0)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: dummy)

    loaded = load_registered_model(question=question, alias="champion")
    assert loaded.question == question
    assert loaded.model_name == REGISTERED_MODELS[question]
    assert loaded.model_uri == f"models:/{REGISTERED_MODELS[question]}@champion"
    assert loaded.source == "registry"
    assert loaded.estimator_type == "DummyRegressor"
    assert loaded.features == expected_features
    assert loaded.model is dummy


def test_load_registered_model_falls_back_to_local_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Usa el modelo empaquetado en cache_dir cuando el servidor MLflow no está disponible."""
    _save_dummy_mlflow_model(tmp_path / "p1")

    def _fail_remote(uri: str):
        raise MlflowException("Servidor MLflow no alcanzable")

    real_load = mlflow.sklearn.load_model
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        mlflow.sklearn,
        "load_model",
        lambda uri: _fail_remote(uri) if uri.startswith("models:/") else real_load(uri),
    )

    loaded = load_registered_model("p1", cache_dir=tmp_path, allow_cache_fallback=True)
    assert loaded.source == "cache"
    assert loaded.question == "p1"
    assert loaded.estimator_type == "DummyRegressor"


def test_load_registered_model_raises_when_unavailable_and_no_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso límite: lanza ModelRegistryError si falla el servidor y no hay caché local."""
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        mlflow.sklearn,
        "load_model",
        lambda uri: (_ for _ in ()).throw(MlflowException("Sin conexión")),
    )
    with pytest.raises(ModelRegistryError, match="No se pudo recuperar el modelo"):
        load_registered_model("p1", cache_dir=tmp_path, allow_cache_fallback=True)


def test_get_registered_model_caches_instance_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Garantiza que get_registered_model descargue el modelo una sola vez por proceso."""
    calls: list[str] = []
    dummy = DummyRegressor()

    def _fake_load(uri: str):
        calls.append(uri)
        return dummy

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.sklearn, "load_model", _fake_load)

    first = get_registered_model("p1", alias="champion")
    second = get_registered_model("p1", alias="champion")
    assert first is second
    assert len(calls) == 1

    reloaded = get_registered_model("p1", alias="champion", force_reload=True)
    assert reloaded.model is dummy
    assert len(calls) == 2


def test_download_registered_model_writes_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifica que download_registered_model descargue el artefacto y escriba metadatos."""

    def _fake_download(artifact_uri: str, dst_path: str) -> str:
        target = Path(dst_path)
        (target / "MLmodel").write_text("artifact_path: modelo\n", encoding="utf-8")
        return str(target)

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", _fake_download)

    path = download_registered_model("p1", alias="champion", output_dir=tmp_path)
    assert (path / "MLmodel").exists()
    assert (path / "registry_metadata.json").exists()


def test_load_registered_model_resolves_version_behind_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expone en LoadedModel.version el número de versión que el Registry asigna al alias."""
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: DummyRegressor())
    monkeypatch.setattr(mlflow, "MlflowClient", _fake_mlflow_client(version="7", calls=calls))

    loaded = load_registered_model("p2", alias="@champion")
    assert loaded.source == "registry"
    assert loaded.alias == "champion"
    assert loaded.version == "7"
    assert calls == [(REGISTERED_MODELS["p2"], "champion")]

    # Con versión explícita no hay alias que resolver: se conserva la indicada sin consultar.
    explicit = load_registered_model("p2", version=4)
    assert explicit.version == "4"
    assert explicit.alias is None
    assert len(calls) == 1


def test_load_registered_model_tolerates_version_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso límite: si get_model_version_by_alias falla, el modelo carga y version es None."""
    dummy = DummyRegressor()
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: dummy)
    monkeypatch.setattr(
        mlflow, "MlflowClient", _fake_mlflow_client(error=RuntimeError("alias inexistente"))
    )

    with _capture_logs("WARNING") as warnings:
        loaded = load_registered_model("p1", alias="champion")
    assert loaded.model is dummy
    assert loaded.source == "registry"
    assert loaded.version is None
    assert any("alias inexistente" in message for message in warnings)


def test_download_registered_model_writes_resolved_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """registry_metadata.json incluye la versión resuelta para que la caché la informe."""

    def _fake_download(artifact_uri: str, dst_path: str) -> str:
        target = Path(dst_path)
        (target / "MLmodel").write_text("artifact_path: modelo\n", encoding="utf-8")
        return str(target)

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", _fake_download)
    monkeypatch.setattr(registry, "_resolve_alias_version", lambda name, alias: "3")

    path = download_registered_model("p3", alias="champion", output_dir=tmp_path)
    metadata = json.loads((path / "registry_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_name"] == REGISTERED_MODELS["p3"]
    assert metadata["alias"] == "champion"
    assert metadata["version"] == "3"


def test_load_registered_model_uses_cache_first_without_tracking_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sin MLFLOW_TRACKING_URI y con caché presente, carga la caché sin tocar el Registry."""
    _save_dummy_mlflow_model(tmp_path / "p1")
    (tmp_path / "p1" / "registry_metadata.json").write_text(
        json.dumps({"alias": "champion", "version": "3"}), encoding="utf-8"
    )
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    requested: list[str] = []
    real_load = mlflow.sklearn.load_model

    def _record_load(uri: str):
        requested.append(uri)
        return real_load(uri)

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow.sklearn, "load_model", _record_load)

    with _capture_logs("WARNING") as warnings:
        loaded = load_registered_model("p1", cache_dir=tmp_path)
    assert loaded.source == "cache"
    assert loaded.alias == "champion"
    assert loaded.version == "3"
    assert requested == [str(tmp_path / "p1")]
    assert not any(uri.startswith("models:/") for uri in requested)
    assert warnings == []


def test_load_registered_model_without_tracking_uri_respects_no_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso límite: allow_cache_fallback=False sigue obligando a consultar el servidor."""
    _save_dummy_mlflow_model(tmp_path / "p1")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        mlflow.sklearn,
        "load_model",
        lambda uri: (_ for _ in ()).throw(MlflowException("Sin conexión")),
    )
    with pytest.raises(ModelRegistryError, match="No se pudo recuperar el modelo"):
        load_registered_model("p1", cache_dir=tmp_path, allow_cache_fallback=False)
