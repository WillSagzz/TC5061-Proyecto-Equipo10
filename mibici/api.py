"""API REST con FastAPI para los tres modelos de MiBici (P1, P2 y P3).

Al iniciar, carga una sola vez los tres modelos desde el Model Registry de MLflow
(``mibici-equipo10-p1-demanda``, ``-p2-riesgo`` y ``-p3-grupos``, con el alias ``champion``;
el alias se cambia con ``MIBICI_MODEL_ALIAS``) y los deja en memoria. Cada petición se valida
con los esquemas de ``schemas.py`` antes de llegar al modelo, y la respuesta también se valida
antes de salir.

Uso desde la raíz del repositorio:

    uvicorn mibici.api:app --reload

Después se abre http://127.0.0.1:8000/docs para probar los endpoints desde el navegador.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, ConfigDict

from mibici.config import REGISTERED_MODELS
from mibici.modeling.registry import LoadedModel, get_registered_model
from mibici.schemas import (
    P1FeaturesInput,
    P1PredictionOutput,
    P2FeaturesInput,
    P2PredictionOutput,
    P3FeaturesInput,
    P3PredictionOutput,
    build_p1_output,
    build_p2_output,
    build_p3_output,
)


class ModelInfo(BaseModel):
    """Datos de un modelo cargado: de dónde vino y de qué tipo es."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    question: str
    model_name: str
    model_uri: str
    alias: str | None
    version: str | None
    source: str
    estimator_type: str


class HealthOutput(BaseModel):
    """Respuesta de ``/health``: el servicio está activo y estos son los modelos que responden."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    models: list[ModelInfo]


def _model_info(loaded: LoadedModel) -> ModelInfo:
    """Resume un modelo cargado para la respuesta de ``/health``.

    Args:
        loaded: Modelo recuperado con ``get_registered_model``.

    Returns:
        Pregunta, nombre, URI, alias, versión, origen y tipo del modelo.
    """
    return ModelInfo(
        question=loaded.question,
        model_name=loaded.model_name,
        model_uri=loaded.model_uri,
        alias=loaded.alias,
        version=loaded.version,
        source=loaded.source,
        estimator_type=loaded.estimator_type,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Carga los tres modelos al iniciar el servicio, antes de aceptar peticiones.

    Si alguno no se puede recuperar, el servicio no arranca y el error queda en el log, en
    lugar de fallar más tarde con la primera petición.

    Args:
        app: Aplicación de FastAPI que se está iniciando.

    Yields:
        Nada; el servicio atiende peticiones mientras dura el ``yield``.
    """
    for question in REGISTERED_MODELS:
        loaded = get_registered_model(question)
        logger.info(
            f"[{question.upper()}] {loaded.model_name} ({loaded.model_uri}, "
            f"origen: {loaded.source}) -> {loaded.estimator_type}"
        )
    logger.success("API lista con los modelos de P1, P2 y P3")
    yield


app = FastAPI(
    title="API MiBici · demanda, riesgo y grupos de estaciones",
    description=(
        "Sirve los tres modelos de MiBici registrados en el Model Registry de MLflow: P1 "
        "pronostica cuántos viajes iniciarán en una estación y hora del día siguiente; P2, si "
        "una estación tiende a vaciarse o a llenarse en el siguiente intervalo de 3 horas; y "
        "P3, a qué grupo de patrones de uso pertenece una estación."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOutput, tags=["servicio"])
def health() -> HealthOutput:
    """Confirma que el servicio está activo e indica qué modelos están respondiendo.

    Returns:
        Estado del servicio y los datos de los modelos de P1, P2 y P3.
    """
    return HealthOutput(
        models=[_model_info(get_registered_model(question)) for question in REGISTERED_MODELS]
    )


@app.post("/predict/p1", response_model=P1PredictionOutput, tags=["predicción"])
def predict_p1(features: P1FeaturesInput) -> P1PredictionOutput:
    """P1 · Pronostica las salidas de una estación en una hora del día siguiente.

    FastAPI valida la entrada con ``P1FeaturesInput`` antes de llamar a esta función: si falta
    una variable, sobra una, tiene otro tipo o está fuera de rango (por ejemplo, una hora sin
    servicio), responde con el código 422 y el motivo, sin llegar al modelo.

    Args:
        features: Las 10 variables de P1 de una estación y una hora.

    Returns:
        Viajes pronosticados (nunca negativos) y el modelo que hizo la predicción.
    """
    loaded = get_registered_model("p1")
    prediction = loaded.model.predict(features.to_dataframe())[0]
    logger.info(
        f"P1 · estación {features.station_id}, hora {features.hour_of_day}: "
        f"{prediction:.2f} viajes"
    )
    return build_p1_output(
        features,
        raw_prediction=prediction,
        model_name=loaded.model_name,
        model_uri=loaded.model_uri,
        estimator_type=loaded.estimator_type,
    )


@app.post("/predict/p2", response_model=P2PredictionOutput, tags=["predicción"])
def predict_p2(features: P2FeaturesInput) -> P2PredictionOutput:
    """P2 · Predice si una estación se vaciará o se llenará en el siguiente intervalo de 3 h.

    La clase usa los umbrales de alerta elegidos con validación (``thresholds.py``): solo se
    alerta ``vaciarse`` o ``llenarse`` si su probabilidad supera su umbral. Por eso la clase
    no siempre es la de mayor probabilidad. Las probabilidades se devuelven sin umbral.

    Args:
        features: Las 13 variables de P2 de una estación y un intervalo.

    Returns:
        Clase de riesgo, probabilidad de cada clase y el modelo que hizo la predicción.
    """
    loaded = get_registered_model("p2")
    table = features.to_dataframe()
    label = loaded.model.predict(table)[0]
    probabilities = loaded.model.predict_proba(table)[0]
    logger.info(
        f"P2 · estación {features.station_id}, intervalo {features.interval_of_day}: {label}"
    )
    return build_p2_output(
        features,
        raw_label=label,
        raw_probabilities=probabilities,
        classes=loaded.model.classes_,
        model_name=loaded.model_name,
        model_uri=loaded.model_uri,
        estimator_type=loaded.estimator_type,
    )


@app.post("/predict/p3", response_model=P3PredictionOutput, tags=["predicción"])
def predict_p3(features: P3FeaturesInput) -> P3PredictionOutput:
    """P3 · Asigna una estación al grupo con el que comparte patrones de uso.

    Args:
        features: Las 8 variables del perfil de uso de una estación.

    Returns:
        Número del grupo asignado y el modelo que hizo la asignación.
    """
    loaded = get_registered_model("p3")
    group = loaded.model.predict(features.to_dataframe())[0]
    logger.info(f"P3 · grupo {group}")
    return build_p3_output(
        raw_group=group,
        model_name=loaded.model_name,
        model_uri=loaded.model_uri,
        estimator_type=loaded.estimator_type,
    )
