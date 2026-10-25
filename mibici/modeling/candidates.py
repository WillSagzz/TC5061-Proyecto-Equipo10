"""Catálogo de modelos de P1, P2 y P3: el baseline y los candidatos de cada pregunta.

``train.py`` elige un modelo por su nombre (``-P model=...`` en ``mlflow run``) y lo construye
con sus hiperparámetros: los de ``MODEL_DEFAULTS`` en ``config.py``, salvo los que lleguen
como parámetros del MLflow Project.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici.config import MODEL_DEFAULTS, RANDOM_SEED
from mibici.features import P1_FEATURES, P2_FEATURES
from mibici.modeling.baselines import (
    build_p1_baseline,
    build_p2_baseline,
    build_p3_baseline,
)


@dataclass(frozen=True)
class ModelSpec:
    """Un modelo del catálogo.

    Args:
        family: Familia del modelo, para la etiqueta ``familia_modelo`` de MLflow.
        build: Función que recibe los hiperparámetros y devuelve el modelo sin entrenar.
        defaults: Hiperparámetros por defecto (de ``MODEL_DEFAULTS``).
    """

    family: str
    build: Callable[[dict], object]
    defaults: dict = field(default_factory=dict)


def _linear_preprocessor(features: list[str], categorical: list[str]) -> ColumnTransformer:
    """Prepara los datos para los modelos lineales (Ridge y LogisticRegression).

    Estos modelos multiplican cada columna por un número y suman, así que necesitan tres
    arreglos:

    - La hora (o el intervalo) y el día de la semana se convierten en columnas de sí o no
      (una por cada valor), porque no son cantidades: las 8:00 no valen el doble que las 4:00.
    - Las celdas vacías (los días de "ayer" o "la semana pasada" que una estación todavía no
      tenía porque acababa de abrir) se rellenan con la mediana.
    - Las demás columnas numéricas se llevan a la misma escala, para que ninguna pese más
      solo por tener números grandes.

    ``station_id`` no se usa: es una etiqueta, no una cantidad. La mediana y la escala se
    calculan solo con el conjunto de entrenamiento.

    Args:
        features: Columnas de entrada de la pregunta.
        categorical: Columnas que se convierten en columnas de sí o no.

    Returns:
        El preprocesamiento, todavía sin ajustar.
    """
    numeric = [column for column in features if column not in categorical + ["station_id"]]
    return ColumnTransformer(
        [
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "numericas",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
        ]
    )


# -----------------------------------------------------------------------------
# P1: regresión de viajes por estación y hora
# -----------------------------------------------------------------------------
def _p1_ridge(hp: dict) -> Pipeline:
    preprocessor = _linear_preprocessor(P1_FEATURES, ["hour_of_day", "day_of_week"])
    return Pipeline([("preprocesamiento", preprocessor), ("estimador", Ridge(alpha=hp["alpha"]))])


def _p1_hgb(hp: dict, loss: str = "squared_error") -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=hp["learning_rate"],
        max_iter=hp["max_iter"],
        max_leaf_nodes=hp["max_leaf_nodes"],
        random_state=RANDOM_SEED,
    )


# -----------------------------------------------------------------------------
# P2: clasificación del riesgo por estación e intervalo de 3 horas
# -----------------------------------------------------------------------------
def _p2_logreg(hp: dict) -> Pipeline:
    preprocessor = _linear_preprocessor(P2_FEATURES, ["interval_of_day", "day_of_week"])
    estimator = LogisticRegression(
        C=hp["c"],
        max_iter=hp["max_iter"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return Pipeline([("preprocesamiento", preprocessor), ("estimador", estimator)])


def _p2_hgb(hp: dict, class_weight: str | None = "balanced") -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=hp["learning_rate"],
        max_iter=hp["max_iter"],
        max_leaf_nodes=hp["max_leaf_nodes"],
        class_weight=class_weight,
        random_state=RANDOM_SEED,
    )


# -----------------------------------------------------------------------------
# P3: agrupación de estaciones (todas las variables se estandarizan)
# -----------------------------------------------------------------------------
def _p3_kmeans(hp: dict) -> Pipeline:
    return build_p3_baseline(hp["n_clusters"])


def _p3_agglomerative(hp: dict) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        AgglomerativeClustering(n_clusters=hp["n_clusters"], linkage="ward"),
    )


def _p3_gmm(hp: dict) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        GaussianMixture(
            n_components=hp["n_clusters"],
            covariance_type="full",
            n_init=3,
            random_state=RANDOM_SEED,
        ),
    )


CATALOG: dict[str, dict[str, ModelSpec]] = {
    "p1": {
        "baseline": ModelSpec("dummy", lambda hp: build_p1_baseline()),
        "ridge": ModelSpec("lineal", _p1_ridge, MODEL_DEFAULTS["ridge"]),
        "hgb": ModelSpec("boosting", _p1_hgb, MODEL_DEFAULTS["hgb"]),
        "hgb_poisson": ModelSpec(
            "boosting", lambda hp: _p1_hgb(hp, loss="poisson"), MODEL_DEFAULTS["hgb"]
        ),
    },
    "p2": {
        "baseline": ModelSpec("dummy", lambda hp: build_p2_baseline()),
        "logreg": ModelSpec("lineal", _p2_logreg, MODEL_DEFAULTS["logreg"]),
        "hgb": ModelSpec("boosting", _p2_hgb, MODEL_DEFAULTS["hgb"]),
        "hgb_unweighted": ModelSpec(
            "boosting", lambda hp: _p2_hgb(hp, class_weight=None), MODEL_DEFAULTS["hgb"]
        ),
    },
    "p3": {
        "baseline": ModelSpec("kmeans", _p3_kmeans, MODEL_DEFAULTS["kmeans"]),
        "kmeans": ModelSpec("kmeans", _p3_kmeans, MODEL_DEFAULTS["kmeans"]),
        "agglomerative": ModelSpec(
            "jerarquico", _p3_agglomerative, MODEL_DEFAULTS["agglomerative"]
        ),
        "gmm": ModelSpec("mezcla_gaussiana", _p3_gmm, MODEL_DEFAULTS["gmm"]),
    },
}


def get_spec(question: str, model: str) -> ModelSpec:
    """Busca un modelo en el catálogo.

    Args:
        question: ``"p1"``, ``"p2"`` o ``"p3"``.
        model: Nombre del modelo, por ejemplo ``"ridge"``.

    Returns:
        La especificación del modelo.

    Raises:
        ValueError: Si la pregunta o el modelo no existen, con la lista de opciones válidas.
    """
    if question not in CATALOG:
        raise ValueError(f"Pregunta '{question}' desconocida. Opciones: {list(CATALOG)}")
    if model not in CATALOG[question]:
        raise ValueError(
            f"Modelo '{model}' desconocido para {question}. Opciones: {list(CATALOG[question])}"
        )
    return CATALOG[question][model]


def resolve_hyperparams(spec: ModelSpec, overrides: dict[str, str]) -> dict:
    """Combina los hiperparámetros por defecto con los recibidos desde ``mlflow run``.

    Un valor vacío significa "usar el de ``config.py``". En Windows, ``mlflow run`` entrega los
    valores vacíos como ``""`` (dos comillas), así que primero se quitan las comillas. Los
    valores recibidos se convierten al tipo del valor por defecto (entero o decimal). Si llega
    un hiperparámetro que el modelo no usa, se avisa y se ignora.

    Args:
        spec: Modelo del catálogo.
        overrides: Hiperparámetros recibidos como texto, por ejemplo ``{"alpha": "10"}``.

    Returns:
        Los hiperparámetros finales del modelo.
    """
    hyperparams = dict(spec.defaults)
    for name, value in overrides.items():
        text = str(value or "").strip().strip("\"'")
        if not text:
            continue
        if name not in hyperparams:
            logger.warning(f"El modelo no usa el hiperparámetro '{name}': se ignora")
            continue
        hyperparams[name] = type(hyperparams[name])(float(text))
    return hyperparams
