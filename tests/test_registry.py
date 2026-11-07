"""Pruebas unitarias para el cargador de modelos del Model Registry (registry.py).

Verifica la construcción de URIs por alias y versión, la carga desde el Model Registry,
el respaldo desde caché local empaquetado, el caché en memoria y el manejo de errores,
sin depender de conexión al servidor remoto de Azure durante la ejecución en CI.
"""

from __future__ import annotations

from pathlib import Path

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
from mibici.modeling.registry import (
    ModelRegistryError,
    build_model_uri,
    clear_model_cache,
    download_registered_model,
    get_registered_model,
    load_registered_model,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Limpia el caché en memoria antes y después de cada prueba."""
    clear_model_cache()
    yield
    clear_model_cache()


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
