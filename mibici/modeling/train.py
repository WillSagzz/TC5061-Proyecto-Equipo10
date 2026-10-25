"""Entrena los baselines y los candidatos de P1, P2 y P3 y los registra en MLflow Tracking.

Se ejecuta como MLflow Project (``MLproject`` en la raíz del repositorio). El experimento (uno
por pregunta, ``EXPERIMENTS`` en ``config.py``) lo elige el propio proyecto a partir de
``question``, y el ``.env`` lo carga el plugin de ``experiments.py``; no hay que indicarlos:

    mlflow run . -e train --env-manager local -P question=p1 -P model=ridge -P alpha=10
    mlflow run . -e search --env-manager local -P question=p1 -P model=hgb

Para depurar, también se puede llamar sin MLflow Project:

    python -m mibici.modeling.train run --question p1 --model ridge
    python -m mibici.modeling.train all          # los tres baselines

Los modelos disponibles están en ``modeling/candidates.py``; sus hiperparámetros por defecto
y las rejillas de búsqueda, en ``config.py``.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
from typing import Annotated

import joblib
from loguru import logger
import mlflow
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
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
    PROJ_ROOT,
    RANDOM_SEED,
    SEARCH_GRIDS,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.experiments import (
    NOTE_TAG,
    ExperimentMismatchError,
    resolve_experiment,
    run_note,
    run_tags,
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
from mibici.modeling.baselines import build_p3_baseline
from mibici.modeling.candidates import get_spec, resolve_hyperparams
from mibici.plots import (
    plot_p1_actual_vs_predicted,
    plot_p2_confusion_matrix,
    plot_p2_precision_recall,
    plot_p3_cluster_map,
)

app = typer.Typer(help="Entrenamiento de MiBici: baselines y candidatos registrados en MLflow.")

QUESTIONS = ("p1", "p2", "p3")

# Nombres de los runs de los baselines: los mismos de la entrega anterior
BASELINE_RUN_NAMES = {
    "p1": "P1-baseline-dummy-estacion-hora",
    "p2": "P2-baseline-dummy-estacion-intervalo-tipo-dia",
    "p3": "P3-baseline-kmeans-k{n_clusters}",
}

# Métrica con la que se elige la mejor combinación de una búsqueda: (nombre, mayor es mejor)
SELECTION_METRIC = {
    "p1": ("val_mae", False),
    "p2": ("val_f1_macro", True),
    "p3": ("train_silhouette", True),
}


@app.callback()
def main() -> None:
    """Entrena un modelo y lo registra en MLflow. Comandos: run, search, p1, p2, p3 y all."""


# -----------------------------------------------------------------------------
# Datos de cada pregunta
# -----------------------------------------------------------------------------
@dataclass
class QuestionData:
    """Tabla de una pregunta lista para entrenar.

    Args:
        question: ``"p1"``, ``"p2"`` o ``"p3"``.
        table: Tabla completa de la pregunta.
        features: Variables de entrada que recibe el modelo.
        target: Variable objetivo (``None`` en P3, que no tiene).
        splits: Entrenamiento, validación y prueba (``None`` en P3, que solo usa
            entrenamiento).
    """

    question: str
    table: pd.DataFrame
    features: list[str]
    target: str | None
    splits: dict[str, pd.DataFrame] | None


def _load_question(question: str) -> QuestionData:
    """Carga los datos de la ingesta y construye la tabla de la pregunta."""
    trips, stations = load_data()
    if question == "p1":
        table = build_p1_table(trips, stations)
        return QuestionData(
            question, table, P1_FEATURES, P1_TARGET, split_by_date(table, "period_start")
        )
    if question == "p2":
        table = build_p2_table(trips, stations)
        return QuestionData(
            question, table, P2_FEATURES, P2_TARGET, split_by_date(table, "period_start")
        )
    return QuestionData(question, build_p3_table(trips, stations), P3_FEATURES, None, None)


# -----------------------------------------------------------------------------
# Utilidades de MLflow
# -----------------------------------------------------------------------------
@contextmanager
def _start_run(
    question: str,
    run_name: str,
    family: str,
    stage: str,
    nested: bool = False,
    note: str | None = None,
):
    """Abre un run con las etiquetas y los parámetros comunes a todas las preguntas.

    Etiquetas: equipo, usuario, pregunta, commit de Git, familia del modelo y etapa del
    trabajo (``baseline``, ``candidato`` o ``busqueda``). Parámetros: fechas de la partición,
    versión de los datos y semilla. Los runs principales registran además el conjunto de
    datos como entrada (pestaña *Datasets*) y ``datapackage.json`` como artefacto. Si el
    programa se lanzó con ``mlflow run``, continúa la corrida que este creó, en el experimento
    de la pregunta (``experiments.py``).

    Args:
        question: ``"p1"``, ``"p2"`` o ``"p3"``.
        run_name: Nombre descriptivo del run.
        family: Familia del modelo (``dummy``, ``lineal``, ``boosting``…).
        stage: ``"baseline"``, ``"candidato"`` o ``"busqueda"``.
        nested: ``True`` para una corrida hija dentro de una búsqueda.
        note: Una línea para la nota del run en la interfaz.
    """
    if not nested:
        try:
            experiment = resolve_experiment(question)
        except ExperimentMismatchError as error:
            logger.error(str(error))
            raise typer.Exit(code=1) from error
    tags = run_tags(question, stage, family)
    if tags["git_cambios_sin_commit"] == "True" and not nested:
        logger.warning("Hay cambios sin commit: haz commit antes del run definitivo")
    data_version = get_data_version()
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        # mlflow run le pone un nombre aleatorio a la corrida: se reemplaza por el descriptivo
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tags(tags | ({NOTE_TAG: note} if note else {}))
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
        if not nested:
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
        if not nested:
            logger.success(f"Run '{run_name}' registrado en '{experiment}'")


def _log_run_config(
    question: str, model_name: str, hyperparams: dict, features: list[str]
) -> None:
    """Adjunta los archivos de configuración con que se ejecutó la corrida (carpeta ``config``).

    Incluye ``MLproject``, ``python_env.yaml``, ``config.py`` y un resumen ``corrida.json``
    con la pregunta, el modelo, sus hiperparámetros, las variables y las fechas.
    """
    for path in (
        PROJ_ROOT / "MLproject",
        PROJ_ROOT / "python_env.yaml",
        PROJ_ROOT / "mibici" / "config.py",
    ):
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="config")
    summary = {
        "pregunta": question,
        "modelo": model_name,
        "hiperparametros": hyperparams,
        "variables": features,
        "particion": {
            "inicio_entrenamiento": TRAIN_START,
            "inicio_validacion": VALIDATION_START,
            "inicio_prueba": TEST_START,
            "fin_datos": DATA_END,
        },
    }
    mlflow.log_dict(summary, "config/corrida.json")


def _log_model(model, example: pd.DataFrame, output: np.ndarray, file_name: str) -> None:
    """Registra el modelo en MLflow con su firma y un ejemplo de entrada.

    También lo guarda en ``models/`` (excluido de Git) para ``predict.py``.

    Args:
        model: Modelo entrenado.
        example: Algunas filas de entrada, sin valores faltantes.
        output: Lo que el modelo devuelve para ``example`` (para la firma).
        file_name: Nombre del archivo en ``models/``.
    """
    path = MODELS_DIR / f"{file_name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    signature = infer_signature(example, output)
    mlflow.sklearn.log_model(model, name="modelo", signature=signature, input_example=example)


def _example_rows(X: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """Primeras filas sin valores faltantes, como ejemplo de entrada del modelo."""
    complete = X.dropna()
    return (complete if len(complete) else X).head(n_rows)


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
# Entrenamiento y registro de un modelo
# -----------------------------------------------------------------------------
def _fit_and_log(data: QuestionData, model_name: str, hyperparams: dict, full: bool) -> dict:
    """Entrena un modelo dentro del run activo y registra parámetros, métricas y artefactos.

    El modelo se entrena solo con el conjunto de entrenamiento. En P1 y P2 se registran las
    métricas de entrenamiento, validación y prueba; en P3, la silueta y el índice de
    Davies-Bouldin. Con ``full=True`` (baselines y candidatos) se registran además las
    gráficas, la configuración y el modelo con su firma; las corridas hijas de una búsqueda
    solo registran parámetros y métricas.

    Args:
        data: Tabla de la pregunta.
        model_name: Nombre del modelo en el catálogo.
        hyperparams: Hiperparámetros finales del modelo.
        full: Si se registran los artefactos completos.

    Returns:
        Las métricas registradas.
    """
    question = data.question
    model = get_spec(question, model_name).build(hyperparams)
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    mlflow.log_params(
        {"estimador": type(estimator).__name__, "variables": ", ".join(data.features)}
        | {f"hp_{name}": value for name, value in hyperparams.items()}
    )

    if question == "p3":
        X = data.table[data.features]
        labels = model.fit_predict(X)
        scaled = model[:-1].transform(X)
        metrics = {
            "train_silhouette": float(silhouette_score(scaled, labels)),
            "train_davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        }
        mlflow.log_params({"n_estaciones": len(X), "periodos_del_dia": str(DAY_PERIODS)})
        mlflow.log_metrics(metrics)
        if full:
            _log_p3_artifacts(data, model_name, labels)
            _log_run_config(question, model_name, hyperparams, data.features)
            # AgglomerativeClustering no tiene predict: se carga con mlflow.sklearn.load_model
            example = _example_rows(X)
            _log_model(model, example, labels[: len(example)], f"p3_{model_name}")
        return metrics

    splits = data.splits
    X_train, y_train = splits["train"][data.features], splits["train"][data.target]
    model.fit(X_train, y_train)
    mlflow.log_params({f"filas_{name}": len(split) for name, split in splits.items()})

    metrics = {}
    for name, split in splits.items():
        X, y = split[data.features], split[data.target]
        if question == "p1":
            metrics |= _regression_metrics(y, model.predict(X), prefix=name)
        else:
            metrics |= _classification_metrics(
                y, model.predict(X), model.predict_proba(X), model.classes_, name
            )
    mlflow.log_metrics(metrics)

    if full:
        test = splits["test"]
        if question == "p1":
            figure = plot_p1_actual_vs_predicted(
                test,
                model.predict(test[data.features]),
                FIGURES_DIR / f"p1_real_vs_{model_name}.png",
            )
            mlflow.log_artifact(str(figure), artifact_path="graficas")
        else:
            X_test, y_test = test[data.features], test[data.target]
            for figure in (
                plot_p2_confusion_matrix(
                    y_test,
                    model.predict(X_test),
                    RISK_CLASSES,
                    FIGURES_DIR / f"p2_matriz_confusion_{model_name}.png",
                ),
                plot_p2_precision_recall(
                    y_test,
                    model.predict_proba(X_test),
                    model.classes_,
                    [RISK_EMPTY, RISK_FULL],
                    FIGURES_DIR / f"p2_precision_recall_{model_name}.png",
                ),
            ):
                mlflow.log_artifact(str(figure), artifact_path="graficas")
        _log_run_config(question, model_name, hyperparams, data.features)
        example = _example_rows(X_train)
        _log_model(model, example, model.predict(example), f"{question}_{model_name}")
    return metrics


def _log_p3_artifacts(data: QuestionData, model_name: str, labels: np.ndarray) -> None:
    """Mapa de los grupos, perfil promedio de cada grupo y grupo de cada estación (P3)."""
    table = data.table
    figure = plot_p3_cluster_map(table, labels, FIGURES_DIR / f"p3_mapa_grupos_{model_name}.png")
    mlflow.log_artifact(str(figure), artifact_path="graficas")
    grouped = table.assign(group=labels)
    profile = grouped.groupby("group")[data.features].mean().round(3)
    profile.insert(0, "n_estaciones", grouped.groupby("group").size())
    mlflow.log_text(profile.to_csv(), "grupos/p3_perfil_grupos.csv")
    columns = ["station_id", "group", "latitude", "longitude"]
    mlflow.log_text(grouped[columns].to_csv(index=False), "grupos/p3_estaciones_por_grupo.csv")


def _log_k_curves(data: QuestionData) -> None:
    """Silueta y Davies-Bouldin de K-Means para cada k de ``P3_K_VALUES`` (step = k)."""
    X = data.table[data.features]
    for k in P3_K_VALUES:
        model = build_p3_baseline(k)
        labels = model.fit_predict(X)
        scaled = model[:-1].transform(X)
        mlflow.log_metric("train_silhouette_por_k", silhouette_score(scaled, labels), step=k)
        mlflow.log_metric(
            "train_davies_bouldin_por_k", davies_bouldin_score(scaled, labels), step=k
        )


def _hyperparams_label(hyperparams: dict) -> str:
    """Texto corto con los hiperparámetros, para el nombre del run (``alpha=10``)."""
    return "-".join(f"{name}={value}" for name, value in hyperparams.items())


def train_model(question: str, model_name: str, overrides: dict[str, str] | None = None) -> None:
    """Entrena un modelo del catálogo y lo registra como un run de MLflow.

    Args:
        question: ``"p1"``, ``"p2"`` o ``"p3"``.
        model_name: ``"baseline"`` o el nombre de un candidato de ``candidates.py``.
        overrides: Hiperparámetros recibidos desde ``mlflow run`` (texto; vacío = por defecto).
    """
    question = question.lower()
    spec = get_spec(question, model_name)
    hyperparams = resolve_hyperparams(spec, overrides or {})
    data = _load_question(question)
    if model_name == "baseline":
        stage = "baseline"
        run_name = BASELINE_RUN_NAMES[question].format(**hyperparams)
    else:
        stage = "candidato"
        label = _hyperparams_label(hyperparams)
        run_name = f"{question.upper()}-candidato-{model_name}" + (f"-{label}" if label else "")
    note = run_note(question, get_data_version(), model_name, **hyperparams)
    with _start_run(question, run_name, spec.family, stage, note=note):
        _fit_and_log(data, model_name, hyperparams, full=True)
        if question == "p3" and model_name == "baseline":
            _log_k_curves(data)


def search_model(question: str, model_name: str) -> None:
    """Búsqueda de hiperparámetros con corridas anidadas (nested runs).

    Abre una corrida padre y, dentro, una corrida hija por cada combinación de la rejilla de
    ``SEARCH_GRIDS``. Cada hija registra sus parámetros y métricas; la padre registra la
    mejor combinación según la métrica de validación de ``SELECTION_METRIC`` y una tabla con
    todos los resultados. El modelo elegido se entrena después como candidato, con todos sus
    artefactos, usando el entry point ``train``.

    Args:
        question: ``"p1"``, ``"p2"`` o ``"p3"``.
        model_name: Modelo con rejilla definida en ``SEARCH_GRIDS``.

    Raises:
        ValueError: Si el modelo no tiene rejilla para esa pregunta.
    """
    question = question.lower()
    spec = get_spec(question, model_name)
    grid = SEARCH_GRIDS.get(question, {}).get(model_name)
    if not grid:
        raise ValueError(
            f"No hay rejilla para {model_name} en {question}. "
            f"Rejillas disponibles: {SEARCH_GRIDS.get(question, {})}"
        )
    metric_name, higher_is_better = SELECTION_METRIC[question]
    combinations = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    data = _load_question(question)

    results = []
    with _start_run(
        question, f"{question.upper()}-busqueda-{model_name}", spec.family, "busqueda"
    ):
        mlflow.log_params(
            {
                "modelo_buscado": model_name,
                "rejilla": json.dumps(grid),
                "n_combinaciones": len(combinations),
                "metrica_seleccion": metric_name,
            }
        )
        for combination in combinations:
            hyperparams = spec.defaults | combination
            child_name = (
                f"{question.upper()}-busqueda-{model_name}-{_hyperparams_label(combination)}"
            )
            with _start_run(question, child_name, spec.family, "busqueda", nested=True) as child:
                metrics = _fit_and_log(data, model_name, hyperparams, full=False)
            results.append({"run_id": child.info.run_id, **combination, **metrics})
            logger.info(f"{child_name}: {metric_name} = {metrics[metric_name]:.4f}")

        summary = pd.DataFrame(results).sort_values(metric_name, ascending=not higher_is_better)
        best = summary.iloc[0]
        mlflow.log_params({f"mejor_{name}": best[name] for name in grid})
        mlflow.log_metric(f"mejor_{metric_name}", float(best[metric_name]))
        mlflow.set_tag("mejor_corrida_hija", best["run_id"])
        mlflow.log_text(summary.to_csv(index=False), "busqueda/resultados.csv")
        logger.success(
            f"Mejor combinación: {_hyperparams_label({n: best[n] for n in grid})} "
            f"({metric_name} = {best[metric_name]:.4f})"
        )


# -----------------------------------------------------------------------------
# Comandos
# -----------------------------------------------------------------------------
@app.command("run")
def run_command(
    question: Annotated[str, typer.Option(help="Pregunta: p1, p2 o p3")],
    model: Annotated[str, typer.Option(help="baseline o un candidato de candidates.py")] = (
        "baseline"
    ),
    alpha: Annotated[str, typer.Option(help="ridge (P1)")] = "",
    c: Annotated[str, typer.Option(help="logreg (P2)")] = "",
    learning_rate: Annotated[str, typer.Option(help="hgb, hgb_poisson, hgb_unweighted")] = "",
    max_iter: Annotated[str, typer.Option(help="logreg y los hgb")] = "",
    max_leaf_nodes: Annotated[str, typer.Option(help="Los hgb")] = "",
    n_clusters: Annotated[str, typer.Option(help="kmeans, agglomerative, gmm (P3)")] = "",
) -> None:
    """Entrena un modelo (entry point train del MLproject).

    Los hiperparámetros que no se pasan toman su valor de MODEL_DEFAULTS en config.py.
    """
    overrides = {
        "alpha": alpha,
        "c": c,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "n_clusters": n_clusters,
    }
    train_model(question, model, overrides)


@app.command("search")
def search_command(
    question: Annotated[str, typer.Option(help="Pregunta: p1, p2 o p3")],
    model: Annotated[str, typer.Option(help="Modelo con rejilla en SEARCH_GRIDS")],
) -> None:
    """Búsqueda de hiperparámetros con nested runs (entry point search del MLproject)."""
    search_model(question, model)


@app.command()
def p1() -> None:
    """P1 · Baseline: DummyRegressor por estación y hora del día."""
    train_model("p1", "baseline")


@app.command()
def p2() -> None:
    """P2 · Baseline: DummyClassifier por estación, intervalo y tipo de día."""
    train_model("p2", "baseline")


@app.command()
def p3() -> None:
    """P3 · Baseline: K-Means sobre el perfil de uso de cada estación."""
    train_model("p3", "baseline")


@app.command("all")
def run_all() -> None:
    """Corre los baselines de P1, P2 y P3, uno después del otro."""
    for question in QUESTIONS:
        train_model(question, "baseline")


if __name__ == "__main__":
    app()
