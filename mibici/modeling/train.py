"""Entrena un modelo de una pregunta y lo registra en MLflow Tracking.

La forma oficial de correrlo es el MLflow Project de la raíz (``MLproject``), que fija el
entorno y registra el commit, el punto de entrada y los parámetros del run:

    mlflow run . -e train -P question=p1                      # baseline de P1
    mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300
    mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=6

El experimento (uno por pregunta, ``EXPERIMENTS`` en ``config.py``) lo elige el proyecto a
partir de la pregunta; no hay que indicarlo. Para depurar también puede llamarse directo, con
la misma interfaz:

    python -m mibici.modeling.train p1 --model rf --n-estimators 300

``model`` es el estimador del catálogo (``CATALOGUE``): ``baseline`` reproduce la
configuración fija de la propuesta y no admite hiperparámetros; los demás son candidatos y
aceptan solo los hiperparámetros que les corresponden. Un run es un modelo entrenado: el
estimador con sus hiperparámetros, las variables, la versión de los datos y el commit. Las
fechas de la partición, la semilla y los valores por defecto salen de ``config.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
import os
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
    EXPERIMENTS,
    FIGURES_DIR,
    MIBICI_DATA_PAGE,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    P3_K_VALUES,
    P3_N_CLUSTERS,
    RANDOM_SEED,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.experiments import (
    NOTE_TAG,
    ensure_experiment,
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
# Catálogo de estimadores: qué puede pedirse con --model y con qué hiperparámetros
# -----------------------------------------------------------------------------
BASELINE = "baseline"
# Tipo de cada hiperparámetro que acepta la línea de comandos (llegan como texto)
HYPERPARAMETER_TYPES: dict[str, Callable[[str], Any]] = {
    "n_estimators": int,
    "max_iter": int,
    "max_depth": int,
    "learning_rate": float,
    "n_clusters": int,
}


@dataclass(frozen=True)
class Estimator:
    """Una entrada del catálogo.

    Args:
        label: Nombre que lleva el run (``P1-candidato-{label}``).
        build: Construye el modelo sin entrenar con los hiperparámetros ya convertidos.
        hyperparameters: Hiperparámetros que acepta; los demás se rechazan.
    """

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
    """Convierte los hiperparámetros de la línea de comandos; el texto vacío es "sin valor".

    El MLproject sustituye una cadena vacía cuando un parámetro no se pasa, así que todos
    llegan como texto y aquí se descartan los vacíos y se convierten los demás con
    ``HYPERPARAMETER_TYPES``. En Windows, ``mlflow run`` entrega el vacío como dos comillas
    literales (``""``), por eso se quitan las comillas antes de decidir si hay valor.

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
    """Construye el modelo pedido y valida el estimador y sus hiperparámetros.

    Args:
        question: ``p1``, ``p2`` o ``p3``.
        name: Estimador del catálogo (``baseline``, ``rf``, ``hgb``, ``kmeans``).
        hyperparameters: Hiperparámetros ya convertidos (``parse_hyperparameters``).

    Returns:
        El modelo sin entrenar y su entrada del catálogo.

    Raises:
        typer.BadParameter: Si el estimador no existe para la pregunta, si ``baseline``
            recibe hiperparámetros o si algún hiperparámetro no aplica al estimador.
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
def _resolve_experiment(question: str) -> None:
    """Deja MLflow apuntando al experimento de la pregunta, o comprueba que ya lo esté.

    Bajo ``mlflow run`` el run ya existe (``MLFLOW_RUN_ID``) en el experimento que eligió la
    línea de comandos; si no es el de la pregunta, se detiene con el comando correcto en vez
    de registrar en el lugar equivocado. Sin ``mlflow run``, crea o selecciona el experimento.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    expected = EXPERIMENTS[question]
    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        ensure_experiment(question)
        mlflow.set_experiment(expected)
        return
    client = mlflow.MlflowClient()
    actual = client.get_experiment(client.get_run(run_id).info.experiment_id).name
    if actual != expected:
        logger.error(
            f"El run de mlflow run está en el experimento '{actual}' y {question} pertenece a "
            f"'{expected}'. Quita --experiment-name (el proyecto lo elige solo) o usa "
            f"--experiment-name {expected}."
        )
        raise typer.Exit(code=1)
    ensure_experiment(question)  # completa descripción y etiquetas si faltan


@contextmanager
def _start_run(question: str, name: str, estimator: Estimator, hyperparameters: dict[str, Any]):
    """Abre el run de un modelo con el nombre, las etiquetas y los parámetros comunes.

    Nombre ``P{n}-{etapa}-{estimador}`` y etiquetas de ``experiments.py``; parámetros: fechas
    de la partición, versión de los datos y semilla; el conjunto de datos como entrada del run
    (pestaña *Datasets*) con la versión del descriptor como huella; ``datapackage.json`` como
    artefacto; y una nota de una línea para la interfaz. Bajo ``mlflow run`` reanuda el run que
    creó la línea de comandos (y le pone el nombre); si no, abre uno nuevo.
    """
    _resolve_experiment(question)
    stage = BASELINE if name == BASELINE else "candidato"
    tags = standard_tags(question, estimator=name, stage=stage)
    if tags["git_cambios_sin_commit"] == "True":
        logger.warning("Hay cambios sin commit: haz commit antes del run definitivo")
    data_version = get_data_version()
    label = run_name(question, stage, estimator.label)
    with mlflow.start_run(run_name=label) as run:
        if run.info.run_name != label:  # run reanudado cuyo nombre no se actualizó
            mlflow.set_tag("mlflow.runName", label)
        mlflow.set_tags(
            tags | {NOTE_TAG: run_note(question, data_version, name, **hyperparameters)}
        )
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
        logger.success(f"Run '{label}' registrado en '{EXPERIMENTS[question]}'")


def _model_params(model) -> dict[str, str]:
    """Hiperparámetros efectivos del modelo, con el prefijo ``modelo_``.

    El prefijo evita chocar con los parámetros que ``mlflow run`` registra tal como se
    pidieron (``n_estimators`` o la cadena vacía si no se pasó): MLflow no permite cambiar el
    valor de un parámetro dentro del run. Así la interfaz muestra lo pedido y lo usado.
    """
    params = {"modelo": type(model).__name__}
    for name, value in model.get_params(deep=True).items():
        if isinstance(value, (str, int, float, bool, type(None), tuple, list)):
            params[f"modelo_{name}"] = str(value)
        elif hasattr(value, "get_params"):
            params[f"modelo_{name}"] = type(value).__name__
    return params


def _log_model(model, question: str, name: str, input_example: pd.DataFrame) -> None:
    """Guarda el modelo en ``models/`` para ``predict.py`` y lo registra como modelo de MLflow.

    El modelo registrado lleva la *flavor* de scikit-learn (firma y ejemplo de entrada), que
    es lo que el Model Registry puede versionar; se serializa con cloudpickle para que
    ``GroupedDummy`` viaje con su clase.
    """
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
# Comandos: la misma interfaz para las tres preguntas
# -----------------------------------------------------------------------------
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


@app.command()
def p1(
    model: ModelOption = BASELINE,
    n_estimators: HyperOption = "",
    max_iter: HyperOption = "",
    max_depth: HyperOption = "",
    learning_rate: HyperOption = "",
    n_clusters: HyperOption = "",
) -> None:
    """P1 · Regresión: viajes iniciados por estación y hora. Estimadores: baseline, rf, hgb."""
    _train_p1(
        model,
        parse_hyperparameters(
            n_estimators=n_estimators,
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_clusters=n_clusters,
        ),
    )


@app.command()
def p2(
    model: ModelOption = BASELINE,
    n_estimators: HyperOption = "",
    max_iter: HyperOption = "",
    max_depth: HyperOption = "",
    learning_rate: HyperOption = "",
    n_clusters: HyperOption = "",
) -> None:
    """P2 · Clasificación: riesgo de vaciarse o llenarse. Estimadores: baseline, rf, hgb."""
    _train_p2(
        model,
        parse_hyperparameters(
            n_estimators=n_estimators,
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_clusters=n_clusters,
        ),
    )


@app.command()
def p3(
    model: ModelOption = BASELINE,
    n_estimators: HyperOption = "",
    max_iter: HyperOption = "",
    max_depth: HyperOption = "",
    learning_rate: HyperOption = "",
    n_clusters: HyperOption = "",
) -> None:
    """P3 · Clustering: perfiles de uso de las estaciones. Estimadores: baseline, kmeans."""
    _train_p3(
        model,
        parse_hyperparameters(
            n_estimators=n_estimators,
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_clusters=n_clusters,
        ),
    )


if __name__ == "__main__":
    app()
