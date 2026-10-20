"""Experimento base: entrena los baselines y los registra en MLflow Tracking.

Uso desde la raíz del repositorio, después de la ingesta:

    python -m mibici.modeling.train p1
    python -m mibici.modeling.train p2
    python -m mibici.modeling.train p3
    python -m mibici.modeling.train all   # las tres preguntas

Las fechas de la partición, la semilla y el experimento de MLflow salen de ``config.py``.
"""

from contextlib import contextmanager
import subprocess

import joblib
from loguru import logger
import mlflow
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
import numpy as np
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
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    P3_K_VALUES,
    P3_N_CLUSTERS,
    PROJ_ROOT,
    RANDOM_SEED,
    TEAM,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.features import (
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

app = typer.Typer(help="Experimento base de MiBici: baselines registrados en MLflow.")


@app.callback()
def main() -> None:
    """Entrena un baseline y lo registra en MLflow. Elige la pregunta: p1, p2, p3 o all."""


# -----------------------------------------------------------------------------
# Utilidades de MLflow, compartidas por P1, P2 y P3
# -----------------------------------------------------------------------------
def _get_git_commit() -> tuple[str, bool]:
    """Devuelve el commit actual y si hay cambios sin commit (un run así no es reproducible)."""
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
        logger.warning("No se encontró un repositorio Git: el run quedará sin commit")
        return "sin_git", True


def _setup_mlflow() -> None:
    """Conecta con MLflow y crea el experimento del equipo si todavía no existe."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        location = (
            MLFLOW_ARTIFACTS_DIR.as_uri() if MLFLOW_TRACKING_URI.startswith("sqlite") else None
        )
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=location)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@contextmanager
def _start_run(question: str, run_name: str, model_type: str = "baseline"):
    """Abre un run con las etiquetas y los parámetros comunes a todas las preguntas.

    Etiquetas: equipo, pregunta y commit de Git. Parámetros: fechas de la partición,
    versión de los datos y semilla. Además registra el conjunto de datos como entrada del
    run (pestaña *datasets* de MLflow), con la versión del descriptor como huella y el portal
    de MiBici como origen, y adjunta ``datapackage.json`` como artefacto.

    Args:
        question: ``"P1"``, ``"P2"`` o ``"P3"``.
        run_name: Nombre descriptivo del run.
        model_type: ``"baseline"`` o ``"candidato"``.
    """
    _setup_mlflow()
    commit, uncommitted_changes = _get_git_commit()
    if uncommitted_changes:
        logger.warning("Hay cambios sin commit: haz commit antes del run definitivo")
    data_version = get_data_version()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "equipo": TEAM,
                "pregunta": question,
                "git_commit": commit,
                "git_cambios_sin_commit": str(uncommitted_changes),
                "tipo_modelo": model_type,
            }
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
        logger.success(f"Run '{run_name}' registrado en '{MLFLOW_EXPERIMENT_NAME}'")


def _model_params(model) -> dict[str, str]:
    """Hiperparámetros simples del modelo (texto, números, tuplas), para MLflow."""
    params = {"modelo": type(model).__name__}
    for name, value in model.get_params(deep=True).items():
        if isinstance(value, (str, int, float, bool, type(None), tuple, list)):
            params[name] = str(value)
        elif hasattr(value, "get_params"):
            params[name] = type(value).__name__
    return params


def _log_model(model, name: str) -> None:
    """Guarda el modelo en models/ (excluido de Git) y lo adjunta al run."""
    path = MODELS_DIR / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    mlflow.log_artifact(str(path), artifact_path="modelo")


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
# Comandos
# -----------------------------------------------------------------------------
@app.command()
def p1() -> None:
    """P1 · Regresión: DummyRegressor por estación y hora del día.

    Entrena con los 18 meses de entrenamiento y evalúa con validación y con prueba.
    """
    # 1. Datos de la ingesta y tabla estación × hora
    trips, stations = load_data()
    table = build_p1_table(trips, stations)

    # 2. Partición temporal
    splits = split_by_date(table, "period_start")

    # 3. Entrenar solo con el conjunto de entrenamiento
    model = build_p1_baseline().fit(splits["train"], splits["train"]["departures"])

    # 4. Registrar en MLflow
    with _start_run("P1", "P1-baseline-dummy-estacion-hora"):
        mlflow.log_params(_model_params(model))
        mlflow.log_params(
            {f"filas_{name}": len(split) for name, split in splits.items()}
            | {"n_grupos": model.n_groups_}
        )
        for name, split in splits.items():
            mlflow.log_metrics(
                _regression_metrics(split["departures"], model.predict(split), prefix=name)
            )

        figure = plot_p1_actual_vs_predicted(
            splits["test"],
            model.predict(splits["test"]),
            FIGURES_DIR / "p1_real_vs_baseline.png",
        )
        mlflow.log_artifact(str(figure), artifact_path="graficas")
        _log_model(model, "p1_baseline")


@app.command()
def p2() -> None:
    """P2 · Clasificación: DummyClassifier por estación, intervalo y tipo de día.

    Entrena con los 18 meses de entrenamiento y evalúa con validación y con prueba.
    """
    # 1. Datos de la ingesta y tabla estación × intervalo de 3 h con la clase de riesgo
    trips, stations = load_data()
    table = build_p2_table(trips, stations)

    # 2. Partición temporal
    splits = split_by_date(table, "period_start")

    # 3. Entrenar solo con el conjunto de entrenamiento
    model = build_p2_baseline().fit(splits["train"], splits["train"]["risk"])

    # 4. Registrar en MLflow
    with _start_run("P2", "P2-baseline-dummy-estacion-intervalo-tipo-dia"):
        mlflow.log_params(_model_params(model))
        mlflow.log_params(
            {f"filas_{name}": len(split) for name, split in splits.items()}
            | {"n_grupos": model.n_groups_}
        )
        for name, split in splits.items():
            metrics = _classification_metrics(
                split["risk"],
                model.predict(split),
                model.predict_proba(split),
                model.classes_,
                name,
            )
            mlflow.log_metrics(metrics)

        figure = plot_p2_confusion_matrix(
            splits["test"]["risk"],
            model.predict(splits["test"]),
            RISK_CLASSES,
            FIGURES_DIR / "p2_matriz_confusion_baseline.png",
        )
        mlflow.log_artifact(str(figure), artifact_path="graficas")
        _log_model(model, "p2_baseline")


@app.command()
def p3() -> None:
    """P3 · Clustering: K-Means sobre el perfil de uso de cada estación.

    Usa solo el periodo de entrenamiento. Registra el modelo con ``P3_N_CLUSTERS`` grupos y,
    para elegir k, la silueta y el índice de Davies-Bouldin de cada valor de ``P3_K_VALUES``.
    """
    # 1. Datos de la ingesta y perfil de cada estación (solo entrenamiento)
    trips, stations = load_data()
    table = build_p3_table(trips, stations)
    features = table[P3_FEATURES]

    # 2. Entrenar el baseline: estandarizar y agrupar
    model = build_p3_baseline(P3_N_CLUSTERS).fit(features)
    labels = model.predict(features)
    scaled = model[:-1].transform(features)

    # 3. Registrar en MLflow
    with _start_run("P3", f"P3-baseline-kmeans-k{P3_N_CLUSTERS}"):
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
        figure = plot_p3_cluster_map(table, labels, FIGURES_DIR / "p3_mapa_grupos_baseline.png")
        mlflow.log_artifact(str(figure), artifact_path="graficas")
        grouped = table.assign(group=labels)
        profile = grouped.groupby("group")[P3_FEATURES].mean().round(3)
        profile.insert(0, "n_estaciones", grouped.groupby("group").size())
        mlflow.log_text(profile.to_csv(), "grupos/p3_perfil_grupos.csv")
        columns = ["station_id", "group", "latitude", "longitude"]
        mlflow.log_text(grouped[columns].to_csv(index=False), "grupos/p3_estaciones_por_grupo.csv")
        _log_model(model, "p3_baseline")


@app.command("all")
def run_all() -> None:
    """Corre los baselines de P1, P2 y P3, uno después del otro."""
    p1()
    p2()
    p3()


if __name__ == "__main__":
    app()
