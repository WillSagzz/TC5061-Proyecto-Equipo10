"""Pruebas unitarias para la inicialización automática del Model Registry (bootstrap.py).

Verifica el registro automático de modelos sintéticos cuando el servidor MLflow arranca vacío,
la idempotencia cuando el alias @champion ya existe, la preferencia por artefactos locales en
models/registry/ y el manejo de tiempos de espera ante servidores HTTP inaccesibles.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import mlflow.sklearn
import pytest
import requests

from mibici.config import EXPERIMENTS
from mibici.features import RISK_CLASSES
from mibici.modeling.bootstrap import (
    _fit_synthetic_estimator,
    ensure_registered_models,
    wait_for_mlflow_server,
)
from mibici.modeling.registry import clear_model_cache, load_registered_model


@pytest.fixture
def isolated_mlflow_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[str, None, None]:
    """Crea un backend SQLite temporal aislado con artefactos fuera de la raíz del repo."""
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "test_mlflow.db"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    for exp_name in EXPERIMENTS.values():
        client.create_experiment(exp_name, artifact_location=artifacts_dir.as_uri())

    clear_model_cache()
    yield tracking_uri
    clear_model_cache()
    mlflow.set_tracking_uri(previous_uri)


def test_ensure_registered_models_seeds_synthetic_and_is_idempotent(
    isolated_mlflow_uri: str, tmp_path: Path
) -> None:
    """Caso normal: puebla P1, P2 y P3 cuando faltan y no duplica versiones al repetir."""
    empty_cache = tmp_path / "empty_cache"
    first_outcomes = ensure_registered_models(
        tracking_uri=isolated_mlflow_uri,
        alias="champion",
        cache_dir=empty_cache,
    )
    assert first_outcomes == {"p1": "synthetic", "p2": "synthetic", "p3": "synthetic"}

    loaded_p1 = load_registered_model(
        "p1", alias="champion", tracking_uri=isolated_mlflow_uri, allow_cache_fallback=False
    )
    loaded_p2 = load_registered_model(
        "p2", alias="champion", tracking_uri=isolated_mlflow_uri, allow_cache_fallback=False
    )
    loaded_p3 = load_registered_model(
        "p3", alias="champion", tracking_uri=isolated_mlflow_uri, allow_cache_fallback=False
    )
    assert loaded_p1.version == "1"
    assert sorted(loaded_p2.model.classes_) == sorted(RISK_CLASSES)
    assert loaded_p3.source == "registry"

    second_outcomes = ensure_registered_models(
        tracking_uri=isolated_mlflow_uri,
        alias="champion",
        cache_dir=empty_cache,
    )
    assert second_outcomes == {"p1": "existing", "p2": "existing", "p3": "existing"}


def test_ensure_registered_models_uses_local_cache_when_present(
    isolated_mlflow_uri: str, tmp_path: Path
) -> None:
    """Si existe models/registry/<pregunta>/MLmodel, registra esa copia en lugar de entrenar."""
    cache_dir = tmp_path / "cache"
    estimator, _ = _fit_synthetic_estimator("p1")
    mlflow.sklearn.save_model(
        sk_model=estimator,
        path=str(cache_dir / "p1"),
        serialization_format="cloudpickle",
    )

    outcomes = ensure_registered_models(
        tracking_uri=isolated_mlflow_uri,
        alias="champion",
        cache_dir=cache_dir,
        questions=("p1",),
    )
    assert outcomes == {"p1": "cache"}


def test_wait_for_mlflow_server_handles_sqlite_http_ok_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso límite: omite sqlite, acepta HTTP 200 y lanza TimeoutError si no responde."""
    wait_for_mlflow_server("sqlite:///mlflow.db", timeout_seconds=0.05)

    ok_response = MagicMock(status_code=200)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: ok_response)
    wait_for_mlflow_server("http://mlflow:5000", timeout_seconds=0.2, poll_interval=0.01)

    def _fail(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("sin conexión")

    monkeypatch.setattr(requests, "get", _fail)
    with pytest.raises(TimeoutError, match="no respondió"):
        wait_for_mlflow_server("http://mlflow:5000", timeout_seconds=0.05, poll_interval=0.01)
