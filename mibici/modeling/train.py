"""Entrena un modelo de una pregunta y lo registra en MLflow Tracking.

Se ejecuta desde el MLflow Project de la raíz (``MLproject``), que registra el commit, el
punto de entrada y los parámetros del run, o directo para depurar, con la misma interfaz:

    mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300
    python -m mibici.modeling.train p1 --model rf --n-estimators 300

``model`` es un estimador del catálogo (``CATALOGUE``): ``baseline`` es la configuración fija
de la propuesta y no admite hiperparámetros; los demás son candidatos y aceptan solo los
suyos. Un run es un modelo entrenado: estimador, hiperparámetros, variables, versión de los
datos y commit. Experimento, nombre y etiquetas los pone ``experiments.py``; fechas, semilla y
valores por defecto salen de ``config.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import joblib
from loguru import logger
import mlflow
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    silhouette_score,
)
import typer

from mibici.config import (
    DATA_END,
    DATAPACKAGE_PATH,
    DAY_PERIODS,
    FIGURES_DIR,
    MIBICI_DATA_PAGE,
    MODELS_DIR,
    P3_K_VALUES,
    P3_N_CLUSTERS,
    RANDOM_SEED,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.experiments import (
    ExperimentMismatchError,
    open_run,
    run_name,
    run_note,
    standard_tags,
)
from mibici.features import (
    P1_FEATURES,
    P1_TARGET,
    P2_FEATURES,
    P2_TARGET,
    P3_FEATURES,
    RISK_CLASSES,
    RISK_EMPTY,
    RISK_FULL,
    build_p1_table,
    build_p2_table,
    build_p3_table,
    get_data_version,
    load_data,
    split_by_date,
)
from mibici.modeling.baselines import (
    build_p1_baseline,
    build_p2_baseline,
    build_p3_baseline,
)
from mibici.plots import (
    plot_p1_actual_vs_predicted,
    plot_p2_confusion_matrix,
    plot_p3_cluster_map,
)

app = typer.Typer(help="Entrena un modelo de P1, P2 o P3 y registra el run en MLflow.")


@app.callback()
def main() -> None:
    """Entrena un modelo y lo registra en MLflow. Elige la pregunta: p1, p2 o p3."""


# -----------------------------------------------------------------------------
# Catálogo de estimadores
# -----------------------------------------------------------------------------
BASELINE = "baseline"
HYPERPARAMETER_TYPES: dict[str, Callable[[str], Any]] = {
    "n_estimators": int,
    "max_iter": int,
    "max_depth": int,
    "learning_rate": float,
    "n_clusters": int,
}


@dataclass(frozen=True)
class Estimator:
    """Entrada del catálogo: nombre que lleva el run, constructor e hiperparámetros que acepta."""

    label: str
    build: Callable[..., Any]
    hyperparameters: tuple[str, ...] = ()


CATALOGUE: dict[str, dict[str, Estimator]] = {
    "p1": {
        BASELINE: Estimator("dummy-estacion-hora", build_p1_baseline),
        "rf": Estimator(
            "rf",
            lambda **hp: RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **hp),
            ("n_estimators", "max_depth"),
        ),
        "hgb": Estimator(
            "hgb",
            lambda **hp: HistGradientBoostingRegressor(random_state=RANDOM_SEED, **hp),
            ("max_iter", "max_depth", "learning_rate"),
        ),
    },
    "p2": {
        BASELINE: Estimator("dummy-estacion-intervalo-tipo-dia", build_p2_baseline),
        "rf": Estimator(
            "rf",
            lambda **hp: RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **hp),
            ("n_estimators", "max_depth"),
        ),
        "hgb": Estimator(
            "hgb",
            lambda **hp: HistGradientBoostingClassifier(random_state=RANDOM_SEED, **hp),
            ("max_iter", "max_depth", "learning_rate"),
        ),
    },
    "p3": {
        BASELINE: Estimator(f"kmeans-k{P3_N_CLUSTERS}", lambda: build_p3_baseline(P3_N_CLUSTERS)),
        "kmeans": Estimator(
            "kmeans",
            lambda n_clusters=P3_N_CLUSTERS: build_p3_baseline(n_clusters),
            ("n_clusters",),
        ),
    },
}


def parse_hyperparameters(**raw: str) -> dict[str, Any]:
    """Convierte los hiperparámetros recibidos como texto; vacío (o solo comillas) es "sin valor".

    Raises:
        typer.BadParameter: Si un valor no tiene el tipo esperado.
    """
    values = {}
    for name, text in raw.items():
        text = (text or "").strip().strip("\"'").strip()
        if text == "":
            continue
        try:
            values[name] = HYPERPARAMETER_TYPES[name](text)
        except ValueError as error:
            expected = HYPERPARAMETER_TYPES[name].__name__
            raise typer.BadParameter(f"{name}={text!r} no es un {expected}") from error
    return values


def build_model(
    question: str, name: str, hyperparameters: dict[str, Any]
) -> tuple[Any, Estimator]:
    """Construye el modelo pedido; valida el estimador y sus hiperparámetros contra el catálogo.

    Raises:
        typer.BadParameter: Si el estimador no existe para la pregunta, si ``baseline`` recibe
            hiperparámetros o si alguno no aplica al estimador.
    """
    options = CATALOGUE[question]
    if name not in options:
        raise typer.BadParameter(
            f"--model {name!r} no existe para {question}. Opciones: {', '.join(options)}."
        )
    estimator = options[name]
    if name == BASELINE and hyperparameters:
        raise typer.BadParameter(
            "baseline es la configuración fija de la propuesta y no admite hiperparámetros; "
            f"usa --model {', '.join(k for k in options if k != BASELINE)}."
        )
    unknown = sorted(set(hyperparameters) - set(estimator.hyperparameters))
    if unknown:
        accepted = ", ".join(estimator.hyperparameters) or "ninguno"
        raise typer.BadParameter(
            f"--model {name} no acepta {', '.join(unknown)}. Hiperparámetros válidos: {accepted}."
        )
    return estimator.build(**hyperparameters), estimator


# -----------------------------------------------------------------------------
# Utilidades de MLflow, compartidas por P1, P2 y P3
# -----------------------------------------------------------------------------
@contextmanager
def _start_run(question: str, name: str, estimator: Estimator, hyperparameters: dict[str, Any]):
    """Abre el run del modelo con los parámetros comunes, el conjunto de datos y el descriptor."""
    stage = BASELINE if name == BASELINE else "candidato"
    data_version = get_data_version()
    with ExitStack() as stack:
        try:
            run = stack.enter_context(
                open_run(
                    question,
                    run_name(question, stage, estimator.label),
                    standard_tags(question, estimator=name, stage=stage),
                    run_note(question, data_version, name, **hyperparameters),
                )
            )
        except ExperimentMismatchError as error:
            logger.error(str(error))
            raise typer.Exit(code=1) from error
        mlflow.log_params(
            {
                "inicio_entrenamiento": TRAIN_START,
                "inicio_validacion": VALIDATION_START,
                "inicio_prueba": TEST_START,
                "fin_datos": DATA_END,
                "version_datos": data_version,
                "semilla": RANDOM_SEED,
            }
        )
        mlflow.log_input(
            MetaDataset(
                source=HTTPDatasetSource(MIBICI_DATA_PAGE),
                name=DATAPACKAGE_PATH.stem,
                digest=data_version,
            ),
            context="entrenamiento",
        )
        mlflow.log_artifact(str(DATAPACKAGE_PATH), artifact_path="datos")
        yield run


def _model_params(model) -> dict[str, str]:
    """Hiperparámetros efectivos con prefijo ``modelo_`` (los pedidos ya los registra ``mlflow run``)."""
    params = {"modelo": type(model).__name__}
    for name, value in model.get_params(deep=True).items():
        if isinstance(value, (str, int, float, bool, type(None), tuple, list)):
            params[f"modelo_{name}"] = str(value)
        elif hasattr(value, "get_params"):
            params[f"modelo_{name}"] = type(value).__name__
    return params


def _log_model(model, question: str, name: str, input_example: pd.DataFrame) -> None:
    """Guarda el modelo en ``models/`` para ``predict.py`` y lo registra con la *flavor* de scikit-learn."""
    path = MODELS_DIR / f"{question}_{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    mlflow.sklearn.log_model(
        model,
        name="modelo",
        input_example=input_example,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )


def _regression_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    """MAE y RMSE con el prefijo del conjunto (``train``, ``val`` o ``test``)."""
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def _classification_metrics(y_true, y_pred, probabilities, classes, prefix: str) -> dict:
    """Métricas de P2 con el prefijo del conjunto (``train``, ``val`` o ``test``).

    Recall, precisión y precisión promedio (área bajo la curva precisión-recall) de cada
    clase de riesgo, y F1 macro de las tres clases. No se usa accuracy: con ~95 % de
    intervalos sin riesgo, un modelo que nunca alerta tendría 95 % de accuracy.
    """
    metrics = {
        f"{prefix}_f1_macro": f1_score(
            y_true, y_pred, labels=RISK_CLASSES, average="macro", zero_division=0
        )
    }
    for risk_class in (RISK_EMPTY, RISK_FULL):
        is_class = np.asarray(y_true) == risk_class
        metrics[f"{prefix}_recall_{risk_class}"] = recall_score(
            y_true, y_pred, labels=[risk_class], average="micro", zero_division=0
        )
        metrics[f"{prefix}_precision_{risk_class}"] = precision_score(
            y_true, y_pred, labels=[risk_class], average="micro", zero_division=0
        )
        position = list(classes).index(risk_class)
        metrics[f"{prefix}_avg_precision_{risk_class}"] = average_precision_score(
            is_class, probabilities[:, position]
        )
    return {name: float(value) for name, value in metrics.items()}


# -----------------------------------------------------------------------------
# Entrenamiento de cada pregunta
# -----------------------------------------------------------------------------
def _train_p1(name: str, hyperparameters: dict[str, Any]) -> None:
    model, estimator = build_model("p1", name, hyperparameters)
    with _start_run("p1", name, estimator, hyperparameters):
        # 1. Datos de la ingesta, tabla estación × hora y partición temporal
        trips, stations = load_data()
        splits = split_by_date(build_p1_table(trips, stations), "period_start")

        # 2. Entrenar solo con el conjunto de entrenamiento
        model.fit(splits["train"][P1_FEATURES], splits["train"][P1_TARGET])

        # 3. Registrar parámetros, métricas y artefactos
        mlflow.log_params(_model_params(model))
        mlflow.log_params(
            {f"filas_{split_name}": len(split) for split_name, split in splits.items()}
            | {"variables": ", ".join(P1_FEATURES)}
            | ({"n_grupos": model.n_groups_} if hasattr(model, "n_groups_") else {})
        )
        for split_name, split in splits.items():
            mlflow.log_metrics(
                _regression_metrics(
                    split[P1_TARGET], model.predict(split[P1_FEATURES]), prefix=split_name
                )
            )
        figure = plot_p1_actual_vs_predicted(
            splits["test"],
            model.predict(splits["test"][P1_FEATURES]),
            FIGURES_DIR / f"p1_real_vs_{name}.png",
        )
        mlflow.log_artifact(str(figure), artifact_path="graficas")
        _log_model(model, "p1", name, splits["train"][P1_FEATURES].head())


def _train_p2(name: str, hyperparameters: dict[str, Any]) -> None:
    model, estimator = build_model("p2", name, hyperparameters)
    with _start_run("p2", name, estimator, hyperparameters):
        # 1. Datos de la ingesta, tabla estación × intervalo con la clase de riesgo y partición
        trips, stations = load_data()
        splits = split_by_date(build_p2_table(trips, stations), "period_start")

        # 2. Entrenar solo con el conjunto de entrenamiento
        model.fit(splits["train"][P2_FEATURES], splits["train"][P2_TARGET])

        # 3. Registrar parámetros, métricas y artefactos
        mlflow.log_params(_model_params(model))
        mlflow.log_params(
            {f"filas_{split_name}": len(split) for split_name, split in splits.items()}
            | {"variables": ", ".join(P2_FEATURES)}
            | ({"n_grupos": model.n_groups_} if hasattr(model, "n_groups_") else {})
        )
        for split_name, split in splits.items():
            mlflow.log_metrics(
                _classification_metrics(
                    split[P2_TARGET],
                    model.predict(split[P2_FEATURES]),
                    model.predict_proba(split[P2_FEATURES]),
                    model.classes_,
                    split_name,
                )
            )
        figure = plot_p2_confusion_matrix(
            splits["test"][P2_TARGET],
            model.predict(splits["test"][P2_FEATURES]),
            RISK_CLASSES,
            FIGURES_DIR / f"p2_matriz_confusion_{name}.png",
        )
        mlflow.log_artifact(str(figure), artifact_path="graficas")
        _log_model(model, "p2", name, splits["train"][P2_FEATURES].head())


def _train_p3(name: str, hyperparameters: dict[str, Any]) -> None:
    model, estimator = build_model("p3", name, hyperparameters)
    with _start_run("p3", name, estimator, hyperparameters):
        # 1. Datos de la ingesta y perfil de cada estación (solo entrenamiento)
        trips, stations = load_data()
        table = build_p3_table(trips, stations)
        features = table[P3_FEATURES]

        # 2. Entrenar: estandarizar y agrupar
        model.fit(features)
        labels = model.predict(features)
        scaled = model[:-1].transform(features)

        # 3. Registrar parámetros, métricas y artefactos
        mlflow.log_params(_model_params(model[-1]))
        mlflow.log_params(
            {
                "conjunto_usado": "train",
                "variables": ", ".join(P3_FEATURES),
                "periodos_del_dia": str(DAY_PERIODS),
                "n_estaciones": len(table),
            }
        )
        mlflow.log_metrics(
            {
                "train_silhouette": float(silhouette_score(scaled, labels)),
                "train_davies_bouldin": float(davies_bouldin_score(scaled, labels)),
            }
        )
        # Curvas para elegir k: una métrica por valor de k (step = k)
        for k in P3_K_VALUES:
            k_labels = build_p3_baseline(k).fit(features).predict(features)
            mlflow.log_metric("train_silhouette_por_k", silhouette_score(scaled, k_labels), step=k)
            mlflow.log_metric(
                "train_davies_bouldin_por_k", davies_bouldin_score(scaled, k_labels), step=k
            )

        # Artefactos: mapa, perfil promedio de cada grupo y grupo de cada estación
        figure = plot_p3_cluster_map(table, labels, FIGURES_DIR / f"p3_mapa_grupos_{name}.png")
        mlflow.log_artifact(str(figure), artifact_path="graficas")
        grouped = table.assign(group=labels)
        profile = grouped.groupby("group")[P3_FEATURES].mean().round(3)
        profile.insert(0, "n_estaciones", grouped.groupby("group").size())
        mlflow.log_text(profile.to_csv(), "grupos/p3_perfil_grupos.csv")
        columns = ["station_id", "group", "latitude", "longitude"]
        mlflow.log_text(grouped[columns].to_csv(index=False), "grupos/p3_estaciones_por_grupo.csv")
        _log_model(model, "p3", name, features.head())


# -----------------------------------------------------------------------------
# Comandos p1, p2 y p3: la misma interfaz, definida una sola vez
# -----------------------------------------------------------------------------
TRAINERS = {"p1": _train_p1, "p2": _train_p2, "p3": _train_p3}
HELP = {
    "p1": "P1 · Regresión: viajes iniciados por estación y hora. Estimadores: baseline, rf, hgb.",
    "p2": "P2 · Clasificación: riesgo de vaciarse o llenarse. Estimadores: baseline, rf, hgb.",
    "p3": "P3 · Clustering: perfiles de uso de las estaciones. Estimadores: baseline, kmeans.",
}
ModelOption = Annotated[
    str,
    typer.Option(
        "--model",
        help="Estimador del catálogo: baseline (configuración fija de la propuesta) o un "
        "candidato (rf, hgb para P1 y P2; kmeans para P3).",
    ),
]
HyperOption = Annotated[
    str, typer.Option(help="Hiperparámetro del estimador; vacío = valor por defecto.")
]


def _make_command(question: str) -> Callable[..., None]:
    def command(
        model: ModelOption = BASELINE,
        n_estimators: HyperOption = "",
        max_iter: HyperOption = "",
        max_depth: HyperOption = "",
        learning_rate: HyperOption = "",
        n_clusters: HyperOption = "",
    ) -> None:
        TRAINERS[question](
            model,
            parse_hyperparameters(
                n_estimators=n_estimators,
                max_iter=max_iter,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_clusters=n_clusters,
            ),
        )

    command.__doc__ = HELP[question]
    return command


for _question in TRAINERS:
    app.command(_question)(_make_command(_question))


if __name__ == "__main__":
    app()
