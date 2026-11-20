"""Inicialización del Model Registry local para orquestación con Docker Compose.

Garantiza que el servicio de MLflow levantado por ``docker-compose.yml`` cuente con los tres
modelos registrados (``mibici-equipo10-p1-demanda``, ``-p2-riesgo`` y ``-p3-grupos``) bajo el
alias configurado (por defecto ``@champion``) antes de que arranque FastAPI:

1. Espera a que el servidor HTTP de MLflow responda en ``/health``.
2. Si el modelo y el alias ya existen en el servidor (por ejemplo, en reinicios con volumen
   persistente o al apuntar a un servidor externo), no modifica nada.
3. Si no existen y hay copia local descargada en ``models/registry/<pregunta>/``, registra ese
   artefacto real en el servidor de MLflow.
4. Si tampoco existe copia local (clon limpio o ejecución en CI/CD sin secretos), entrena y
   registra un estimador ligero con datos sintéticos que respeta el contrato de ``schemas.py``.

Uso desde la raíz del repositorio o dentro del contenedor:

    python -m mibici.modeling.bootstrap
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Annotated, Any

from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
import typer

from mibici.config import (
    DEFAULT_MODEL_ALIAS,
    EXPERIMENTS,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_CACHE_DIR,
    RANDOM_SEED,
    REGISTERED_MODELS,
)
from mibici.features import P1_FEATURES, P2_FEATURES, P3_FEATURES, RISK_CLASSES
from mibici.modeling.candidates import get_spec

app = typer.Typer(help="Inicializa el Model Registry local cuando faltan modelos @champion.")


def wait_for_mlflow_server(
    tracking_uri: str,
    timeout_seconds: float = 60.0,
    poll_interval: float = 1.0,
) -> None:
    """Espera hasta que un servidor HTTP de MLflow responda 200 en ``/health``.

    Si ``tracking_uri`` no usa HTTP/HTTPS (por ejemplo ``sqlite:///``), regresa de inmediato.

    Args:
        tracking_uri: URI configurada para MLflow Tracking.
        timeout_seconds: Tiempo máximo de espera en segundos.
        poll_interval: Pausa entre intentos consecutivos en segundos.

    Raises:
        TimeoutError: Si el servidor no responde antes de ``timeout_seconds``.
    """
    if not tracking_uri.startswith(("http://", "https://")):
        return

    health_url = f"{tracking_uri.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Servidor MLflow disponible en {tracking_uri}")
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(poll_interval)

    raise TimeoutError(
        f"El servidor MLflow en '{tracking_uri}' no respondió tras {timeout_seconds:.0f}s "
        f"(último error: {last_error})."
    )


def _alias_exists(client: mlflow.MlflowClient, model_name: str, alias: str) -> bool:
    """Indica si ``model_name`` ya tiene asignado ``alias`` en el Model Registry."""
    try:
        client.get_model_version_by_alias(model_name, alias)
        return True
    except MlflowException:
        return False


def _build_synthetic_training_pair(
    question: str, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Construye una muestra sintética mínima que cumple el contrato de cada pregunta."""
    rng = np.random.default_rng(seed)
    n_rows = 120

    if question == "p1":
        hours = rng.choice([6, 7, 8, 9, 12, 15, 18, 20], size=n_rows)
        days = rng.integers(0, 7, size=n_rows)
        df = pd.DataFrame(
            {
                "station_id": rng.integers(1, 55, size=n_rows),
                "hour_of_day": hours,
                "day_of_week": days,
                "is_weekend": (days >= 5).astype(int),
                "capacity": rng.choice([15, 20, 25, 30], size=n_rows),
                "departures_lag_24": rng.uniform(1.0, 20.0, size=n_rows),
                "departures_lag_168": rng.uniform(1.0, 20.0, size=n_rows),
                "arrivals_lag_24": rng.uniform(1.0, 20.0, size=n_rows),
                "net_flow_lag_24": rng.uniform(-8.0, 8.0, size=n_rows),
                "departures_mean_3_lag_24": rng.uniform(1.0, 18.0, size=n_rows),
            }
        )[P1_FEATURES]
        target = rng.poisson(lam=8.0, size=n_rows).astype(float)
        return df, target

    if question == "p2":
        days = rng.integers(0, 7, size=n_rows)
        df = pd.DataFrame(
            {
                "station_id": rng.integers(1, 55, size=n_rows),
                "interval_of_day": rng.integers(0, 8, size=n_rows),
                "day_of_week": days,
                "is_weekend": (days >= 5).astype(int),
                "capacity": rng.choice([15, 20, 25, 30], size=n_rows),
                "departures_lag_1": rng.uniform(0.0, 15.0, size=n_rows),
                "departures_lag_8": rng.uniform(0.0, 25.0, size=n_rows),
                "departures_lag_56": rng.uniform(0.0, 25.0, size=n_rows),
                "arrivals_lag_1": rng.uniform(0.0, 15.0, size=n_rows),
                "arrivals_lag_8": rng.uniform(0.0, 25.0, size=n_rows),
                "net_flow_lag_1": rng.uniform(-10.0, 10.0, size=n_rows),
                "net_flow_lag_8": rng.uniform(-12.0, 12.0, size=n_rows),
                "departures_mean_3": rng.uniform(1.0, 14.0, size=n_rows),
            }
        )[P2_FEATURES]
        labels = np.tile(np.array(sorted(RISK_CLASSES)), n_rows // len(RISK_CLASSES))
        return df, labels

    raw_shares = rng.dirichlet([2.5, 2.0, 2.5, 1.0], size=n_rows) * 100.0
    df = pd.DataFrame(
        {
            "morning_pct": raw_shares[:, 0],
            "midday_pct": raw_shares[:, 1],
            "afternoon_pct": raw_shares[:, 2],
            "night_pct": raw_shares[:, 3],
            "weekdays_pct": rng.uniform(55.0, 90.0, size=n_rows),
            "log_daily_use_per_dock": rng.uniform(0.3, 2.2, size=n_rows),
            "balance_morning": rng.uniform(-0.6, 0.6, size=n_rows),
            "balance_afternoon": rng.uniform(-0.6, 0.6, size=n_rows),
        }
    )[P3_FEATURES]
    return df, None


def _fit_synthetic_estimator(question: str) -> tuple[Any, pd.DataFrame]:
    """Entrena un estimador ligero del catálogo del proyecto para ``question``."""
    features_df, target = _build_synthetic_training_pair(question)
    if question == "p1":
        spec = get_spec("p1", "hgb_poisson")
        estimator = spec.build({**spec.defaults, "max_iter": 30})
        estimator.fit(features_df, target)
        return estimator, features_df

    if question == "p2":
        spec = get_spec("p2", "hgb_threshold")
        estimator = spec.build({**spec.defaults, "max_iter": 30})
        estimator.fit(features_df, target)
        return estimator, features_df

    spec = get_spec("p3", "baseline")
    estimator = spec.build(spec.defaults)
    estimator.fit(features_df)
    return estimator, features_df


def ensure_registered_models(
    tracking_uri: str | None = None,
    alias: str = DEFAULT_MODEL_ALIAS,
    cache_dir: Path = MODEL_REGISTRY_CACHE_DIR,
    questions: tuple[str, ...] = ("p1", "p2", "p3"),
    wait_timeout: float = 60.0,
) -> dict[str, str]:
    """Verifica y registra los modelos faltantes en el Model Registry activo.

    Args:
        tracking_uri: URI del servidor de MLflow; por defecto ``MLFLOW_TRACKING_URI``.
        alias: Alias a verificar o asignar (por defecto ``champion``).
        cache_dir: Carpeta donde buscar copias reales descargadas (``models/registry/``).
        questions: Subconjunto de preguntas a asegurar.
        wait_timeout: Segundos máximos para esperar al servidor HTTP de MLflow.

    Returns:
        Diccionario ``{pregunta: origen}`` donde ``origen`` es ``"existing"``,
        ``"cache"`` o ``"synthetic"``.
    """
    effective_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI
    clean_alias = alias.strip().lstrip("@")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    wait_for_mlflow_server(effective_uri, timeout_seconds=wait_timeout)
    mlflow.set_tracking_uri(effective_uri)
    client = mlflow.MlflowClient()

    outcomes: dict[str, str] = {}
    for question in questions:
        model_name = REGISTERED_MODELS[question]
        if _alias_exists(client, model_name, clean_alias):
            logger.info(
                f"[{question.upper()}] '{model_name}@{clean_alias}' ya existe en {effective_uri}"
            )
            outcomes[question] = "existing"
            continue

        cached_dir = Path(cache_dir) / question
        if (cached_dir / "MLmodel").exists():
            estimator = mlflow.sklearn.load_model(str(cached_dir))
            sample_df, _ = _build_synthetic_training_pair(question)
            source_kind = "cache"
        else:
            estimator, sample_df = _fit_synthetic_estimator(question)
            source_kind = "synthetic"

        mlflow.set_experiment(EXPERIMENTS[question])
        run_name = f"{question.upper()}-bootstrap-{source_kind}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "equipo": "equipo10",
                    "pregunta": question.upper(),
                    "etapa": "bootstrap",
                    "origen_bootstrap": source_kind,
                }
            )
            model_info = mlflow.sklearn.log_model(
                sk_model=estimator,
                artifact_path="modelo",
                serialization_format="cloudpickle",
                input_example=sample_df.head(2),
            )
            mv = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
            client.set_registered_model_alias(model_name, clean_alias, str(mv.version))

        logger.success(
            f"[{question.upper()}] Registrado '{model_name}@{clean_alias}' "
            f"(versión {mv.version}, origen={source_kind})"
        )
        outcomes[question] = source_kind

    return outcomes


@app.command()
def main(
    alias: Annotated[
        str, typer.Option(help="Alias que debe quedar disponible en el Registry.")
    ] = DEFAULT_MODEL_ALIAS,
    timeout: Annotated[
        float, typer.Option(help="Segundos máximos de espera por el servidor MLflow.")
    ] = 60.0,
) -> None:
    """Asegura que los tres modelos estén disponibles en MLflow antes de iniciar la API."""
    ensure_registered_models(alias=alias, wait_timeout=timeout)


if __name__ == "__main__":
    app()
