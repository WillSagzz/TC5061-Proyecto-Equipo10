"""Versión alternativa del modelo de P1 y A/B testing contra la versión vigente.

- ``retrain``: toma el modelo ``@champion`` de P1 (versión A), lo reentrena con los mismos
  hiperparámetros y los datos de septiembre de 2024 a agosto de 2026, lo registra como una
  versión nueva de ``mibici-equipo10-p1-demanda`` y le pone el alias ``challenger`` (versión B).
- ``compare``: evalúa A y B sobre el mismo periodo actual (septiembre de 2026, que ninguna vio
  al entrenar), en total y separando las estaciones que A conoció de las nuevas; estima con un
  bootstrap por día si la diferencia de MAE es real y recomienda qué versión debe quedar
  vigente. Con ``--promote``, si gana B, le mueve el alias ``champion``.

Uso desde la raíz del repositorio (con el ``.env`` del servidor de MLflow), con el MLflow
Project, igual que ``make train`` (puntos de entrada ``retrain`` y ``ab_test`` del MLproject):

    mlflow run . -e retrain --env-manager local      # make retrain
    mlflow run . -e ab_test --env-manager local      # make ab-test

También se pueden correr directo con Python (``compare --promote`` solo existe así):

    python -m mibici.modeling.ab_test retrain
    python -m mibici.modeling.ab_test compare
    python -m mibici.modeling.ab_test compare --promote
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import typer

from mibici.config import (
    AB_TEST_BOOTSTRAP_SAMPLES,
    AB_TEST_CONFIDENCE,
    AB_TEST_REPORTS_DIR,
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    CURRENT_END,
    CURRENT_START,
    DATAPACKAGE_PATH,
    MIBICI_DATA_PAGE,
    RANDOM_SEED,
    REGISTERED_MODELS,
    RETRAIN_END,
    RETRAIN_START,
    VALIDATION_START,
)
from mibici.experiments import (
    NOTE_TAG,
    ExperimentMismatchError,
    resolve_experiment,
    run_note,
    run_tags,
)
from mibici.features import P1_FEATURES, P1_TARGET, get_data_version
from mibici.modeling.registry import LoadedModel, load_registered_model
from mibici.monitoring import (
    MonitoringDatasets,
    load_or_ingest_monitoring_inputs,
    prepare_monitoring_data,
)

app = typer.Typer(help="Versión alternativa de P1 y A/B testing contra la versión vigente.")

QUESTION = "p1"
SEGMENTS = ("todas", "conocidas", "nuevas")
# Hiperparámetros del HistGradientBoostingRegressor que se registran si no se pueden copiar
# del run de @champion (por ejemplo, si se registró sin parámetros)
LOGGED_HYPERPARAMS = ("loss", "learning_rate", "max_iter", "max_leaf_nodes", "random_state")
# Parámetros del run de @champion que no se copian: describen sus fechas y el tamaño de su
# partición, no el modelo
PARTITION_PARAMS = (
    "inicio_entrenamiento",
    "inicio_validacion",
    "inicio_prueba",
    "fin_datos",
    "version_datos",
    "semilla",
    "filas_train",
    "filas_val",
    "filas_test",
)


@dataclass(frozen=True)
class ABTestResult:
    """Resultado del A/B testing entre la versión vigente (A) y la alternativa (B).

    Attributes:
        table: Una fila por versión y segmento con ``filas``, ``mae`` y ``rmse``.
        mae_difference: MAE de B menos MAE de A en todo el periodo actual (negativo = B mejor).
        ci_low: Límite inferior del intervalo de confianza de la diferencia.
        ci_high: Límite superior del intervalo de confianza de la diferencia.
        winner: Alias de la versión que debe quedar vigente (``champion`` o ``challenger``).
    """

    table: pd.DataFrame
    mae_difference: float
    ci_low: float
    ci_high: float
    winner: str


def build_ab_datasets(
    trips: pd.DataFrame, production_trips: pd.DataFrame, stations: pd.DataFrame
) -> MonitoringDatasets:
    """Tabla de P1 para reentrenar (sep 2024 a ago 2026) y para evaluar (sep 2026).

    Reusa ``prepare_monitoring_data`` con la ventana de referencia extendida hasta
    ``RETRAIN_END`` y sin muestreo: la evaluación usa todas las filas del periodo actual.

    Args:
        trips: Viajes del periodo base (``load_data``).
        production_trips: Viajes del periodo actual.
        stations: Catálogo de estaciones.

    Returns:
        ``reference`` para entrenar la versión B y ``current`` para evaluar A y B.
    """
    return prepare_monitoring_data(
        question=QUESTION,
        trips=trips,
        production_trips=production_trips,
        stations=stations,
        reference_start=RETRAIN_START,
        reference_end=RETRAIN_END,
        current_start=CURRENT_START,
        current_end=CURRENT_END,
    )


def retrain_challenger(champion: LoadedModel, training: pd.DataFrame):
    """Reentrena la versión vigente con los mismos hiperparámetros y más datos.

    ``clone`` copia la configuración del modelo (hiperparámetros y semilla) sin lo aprendido,
    así que lo único que cambia entre A y B son los datos de entrenamiento.

    Args:
        champion: Versión vigente cargada del Model Registry.
        training: Tabla de P1 de ``RETRAIN_START`` a ``RETRAIN_END``.

    Returns:
        El modelo B entrenado.
    """
    return clone(champion.model).fit(training[P1_FEATURES], training[P1_TARGET])


def _start_run(run_name: str) -> mlflow.ActiveRun:
    """Abre (o continúa, bajo ``mlflow run``) un run en el experimento de P1.

    Con ``mlflow run`` el run ya existe y MLflow le pone un nombre aleatorio; por eso el nombre
    descriptivo se escribe en la etiqueta ``mlflow.runName``, igual que en ``train.py``.

    Args:
        run_name: Nombre descriptivo del run.

    Returns:
        El run activo, para usarlo con ``with``.

    Raises:
        typer.Exit: Si ``mlflow run`` creó el run en un experimento que no es el de P1.
    """
    try:
        resolve_experiment(QUESTION)
    except ExperimentMismatchError as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tag("mlflow.runName", run_name)
    return run


def champion_params(champion: LoadedModel) -> dict[str, str]:
    """Parámetros del modelo con que se entrenó @champion, para repetirlos en la versión B.

    Copia del run que registró la versión A todo lo que describe el modelo (``model``,
    ``estimador``, ``variables``, ``hp_*``…) y deja fuera sus fechas y sus filas
    (``PARTITION_PARAMS``).
    Así, en la vista *Compare* de MLflow las dos versiones muestran los mismos parámetros y
    solo cambian las fechas. Si el run no tiene parámetros, usa los del propio modelo.

    Args:
        champion: Versión vigente cargada del Model Registry.

    Returns:
        Nombre y valor de cada parámetro.
    """
    client = mlflow.MlflowClient()
    try:
        run_id = client.get_model_version(champion.model_name, champion.version).run_id
        copied = client.get_run(run_id).data.params if run_id else {}
    except MlflowException as error:
        logger.warning(f"[P1] No se pudieron leer los parámetros de @champion: {error}")
        copied = {}
    copied = {name: value for name, value in copied.items() if name not in PARTITION_PARAMS}

    fallback = {"estimador": champion.estimator_type, "variables": ", ".join(P1_FEATURES)}
    if not any(name.startswith("hp_") for name in copied):
        params = champion.model.get_params()
        fallback |= {f"hp_{name}": params[name] for name in LOGGED_HYPERPARAMS if name in params}
    return fallback | copied


def register_challenger(model, champion: LoadedModel, training: pd.DataFrame) -> str:
    """Registra la versión B en MLflow y le asigna el alias ``challenger``.

    Abre un run en el experimento de P1 (o continúa el de ``mlflow run``) con las etiquetas del
    proyecto (equipo, pregunta, commit…), las fechas del reentrenamiento, la versión de los
    datos, la semilla, los mismos parámetros del modelo que el run de @champion y el MAE de
    entrenamiento, y registra el modelo como una versión nueva del mismo nombre del Model
    Registry.

    Args:
        model: Versión B entrenada.
        champion: Versión vigente (A), para anotar de qué versión se partió.
        training: Tabla con la que se entrenó B.

    Returns:
        Número de la versión registrada.
    """
    model_name = REGISTERED_MODELS[QUESTION]
    data_version = get_data_version()
    tags = run_tags(QUESTION, stage="reentrenamiento", family="boosting")
    if tags["git_cambios_sin_commit"] == "True":
        logger.warning("Hay cambios sin commit: haz commit antes del run definitivo")
    note = run_note(
        QUESTION,
        data_version,
        champion.estimator_type,
        base=f"v{champion.version}",
        hasta=RETRAIN_END,
    )
    X, y = training[P1_FEATURES], training[P1_TARGET]
    example = X.dropna().head(5)
    model_params = champion_params(champion)

    with _start_run(f"P1-reentrenado-hasta-{RETRAIN_END}"):
        mlflow.set_tags(tags | {NOTE_TAG: note})
        mlflow.log_params(
            {
                "inicio_entrenamiento": RETRAIN_START,
                "fin_entrenamiento": RETRAIN_END,
                "inicio_prueba": CURRENT_START,
                "fin_datos": CURRENT_END,
                "version_datos": data_version,
                "semilla": RANDOM_SEED,
                "version_base": champion.version,
                "filas_train": len(training),
            }
            | model_params
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
        predictions = model.predict(X)
        mlflow.log_metrics(
            {
                "train_mae": float(mean_absolute_error(y, predictions)),
                "train_rmse": float(root_mean_squared_error(y, predictions)),
            }
        )
        info = mlflow.sklearn.log_model(
            model,
            name="modelo",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=model_name,
        )

    version = str(info.registered_model_version)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(model_name, CHALLENGER_ALIAS, version)
    client.set_model_version_tag(model_name, version, "git_commit", tags["git_commit"])
    client.set_model_version_tag(model_name, version, "version_datos", data_version)
    logger.success(f"[P1] Versión {version} de {model_name} registrada con @{CHALLENGER_ALIAS}")
    return version


def evaluate_versions(
    models: dict[str, object], current: pd.DataFrame, known_stations: set[int]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """MAE y RMSE de cada versión en el periodo actual, en total y por tipo de estación.

    Args:
        models: Modelos por alias (``champion`` y ``challenger``).
        current: Tabla de P1 del periodo actual.
        known_stations: Estaciones que la versión A tuvo en su entrenamiento.

    Returns:
        La tabla (versión, segmento, filas, MAE, RMSE) y el error absoluto de cada versión
        en cada fila, para el bootstrap.
    """
    X, y = current[P1_FEATURES], current[P1_TARGET].to_numpy()
    is_known = current["station_id"].isin(known_stations).to_numpy()
    masks = {
        "todas": np.ones(len(current), dtype=bool),
        "conocidas": is_known,
        "nuevas": ~is_known,
    }

    rows, errors = [], {}
    for alias, model in models.items():
        predictions = model.predict(X)
        errors[alias] = np.abs(predictions - y)
        for segment in SEGMENTS:
            mask = masks[segment]
            if not mask.any():
                continue
            rows.append(
                {
                    "version": alias,
                    "segmento": segment,
                    "filas": int(mask.sum()),
                    "mae": float(mean_absolute_error(y[mask], predictions[mask])),
                    "rmse": float(root_mean_squared_error(y[mask], predictions[mask])),
                }
            )
    return pd.DataFrame(rows), errors


def paired_bootstrap(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    days: pd.Series,
    n_samples: int = AB_TEST_BOOTSTRAP_SAMPLES,
    confidence: float = AB_TEST_CONFIDENCE,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """Diferencia de MAE (B menos A) con su intervalo de confianza por bootstrap de días.

    Las dos versiones se evalúan sobre las mismas filas (comparación pareada). Se remuestrean
    días completos y no filas sueltas, porque las horas de un mismo día no son independientes:
    un día de lluvia o un puente afecta a todas a la vez.

    Args:
        errors_a: Error absoluto de la versión A en cada fila.
        errors_b: Error absoluto de la versión B en las mismas filas.
        days: Día de cada fila.
        n_samples: Número de remuestreos.
        confidence: Nivel del intervalo (0.95 = 95 %).
        seed: Semilla, para que el intervalo sea reproducible.

    Returns:
        Diferencia observada, límite inferior y límite superior del intervalo.
    """
    by_day = (
        pd.DataFrame({"day": np.asarray(days), "a": errors_a, "b": errors_b})
        .groupby("day")
        .agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    )
    sums_a, sums_b, counts = (by_day[col].to_numpy() for col in ("a", "b", "n"))
    observed = (sums_b.sum() - sums_a.sum()) / counts.sum()

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(by_day), size=(n_samples, len(by_day)))
    differences = (sums_b[picks].sum(axis=1) - sums_a[picks].sum(axis=1)) / counts[picks].sum(
        axis=1
    )
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(differences, [tail, 100 - tail])
    return float(observed), float(low), float(high)


def decide_winner(ci_high: float) -> str:
    """Elige la versión vigente: B solo gana si es mejor con la confianza pedida.

    Si todo el intervalo de la diferencia de MAE (B menos A) es negativo, B tiene menos error
    y no es por casualidad. Si el intervalo incluye el 0, la evidencia no alcanza y se
    conserva la versión vigente.

    Args:
        ci_high: Límite superior del intervalo de la diferencia de MAE.

    Returns:
        ``challenger`` si gana B; ``champion`` si se conserva A.
    """
    return CHALLENGER_ALIAS if ci_high < 0 else CHAMPION_ALIAS


def run_ab_test(datasets: MonitoringDatasets, champion: LoadedModel, challenger: LoadedModel):
    """Compara A y B en el periodo actual y decide cuál debe quedar vigente.

    Args:
        datasets: Salida de ``build_ab_datasets``.
        champion: Versión A.
        challenger: Versión B.

    Returns:
        ``ABTestResult`` con la tabla comparativa, la diferencia de MAE y la decisión.
    """
    reference = datasets.reference
    trained_by_a = reference["period_start"] < pd.Timestamp(VALIDATION_START)
    known_stations = set(reference.loc[trained_by_a, "station_id"])
    table, errors = evaluate_versions(
        {CHAMPION_ALIAS: champion.model, CHALLENGER_ALIAS: challenger.model},
        datasets.current,
        known_stations,
    )
    difference, low, high = paired_bootstrap(
        errors[CHAMPION_ALIAS],
        errors[CHALLENGER_ALIAS],
        datasets.current["period_start"].dt.date,
    )
    return ABTestResult(table, difference, low, high, decide_winner(high))


def ab_test_markdown(result: ABTestResult, champion: LoadedModel, challenger: LoadedModel) -> str:
    """Tabla comparativa en Markdown, lista para pegar en el reporte.

    Args:
        result: Salida de ``run_ab_test``.
        champion: Versión A.
        challenger: Versión B.

    Returns:
        Tabla de MAE y RMSE por versión y segmento, la diferencia de MAE con su intervalo y la
        versión que debe quedar vigente.
    """
    versions = {CHAMPION_ALIAS: champion.version, CHALLENGER_ALIAS: challenger.version}
    lines = [
        f"A/B testing de P1 en {CURRENT_START} a {CURRENT_END} (fin excluido)",
        "",
        "| Versión | Estaciones | Filas | MAE | RMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result.table.itertuples():
        lines.append(
            f"| `@{row.version}` (v{versions[row.version]}) | {row.segmento} | {row.filas:,} "
            f"| {row.mae:.4f} | {row.rmse:.4f} |"
        )
    winner = versions[result.winner]
    lines += [
        "",
        (
            f"MAE de `@{CHALLENGER_ALIAS}` menos MAE de `@{CHAMPION_ALIAS}`: "
            f"{result.mae_difference:+.4f} (IC {AB_TEST_CONFIDENCE:.0%} por bootstrap de días: "
            f"{result.ci_low:+.4f} a {result.ci_high:+.4f})."
        ),
        "",
        f"Versión que debe quedar vigente: `@{result.winner}` (v{winner}).",
    ]
    return "\n".join(lines) + "\n"


def log_ab_test(result: ABTestResult, champion: LoadedModel, challenger: LoadedModel) -> Path:
    """Guarda la tabla comparativa y registra el A/B testing como un run de P1.

    Args:
        result: Salida de ``run_ab_test``.
        champion: Versión A.
        challenger: Versión B.

    Returns:
        Ruta de la tabla comparativa en Markdown (``p1_ab_test.md``); junto a ella queda la
        misma tabla en CSV (``p1_ab_test.csv``).
    """
    AB_TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = AB_TEST_REPORTS_DIR / "p1_ab_test.csv"
    md_path = AB_TEST_REPORTS_DIR / "p1_ab_test.md"
    result.table.to_csv(csv_path, index=False)
    md_path.write_text(ab_test_markdown(result, champion, challenger), encoding="utf-8")

    data_version = get_data_version()
    tags = run_tags(QUESTION, stage="ab_test")
    metrics = {}
    for row in result.table.itertuples():
        suffix = "" if row.segmento == "todas" else f"_{row.segmento}"
        metrics[f"test_mae_{row.version}{suffix}"] = row.mae
        metrics[f"test_rmse_{row.version}{suffix}"] = row.rmse
    metrics |= {
        "test_mae_diferencia": result.mae_difference,
        "test_mae_diferencia_ic_inf": result.ci_low,
        "test_mae_diferencia_ic_sup": result.ci_high,
    }
    with _start_run(f"P1-ab-test-v{champion.version}-vs-v{challenger.version}"):
        mlflow.set_tags(
            tags
            | {
                NOTE_TAG: run_note(QUESTION, data_version, ganador=result.winner),
                "ganador": result.winner,
            }
        )
        mlflow.log_params(
            {
                "version_champion": champion.version,
                "version_challenger": challenger.version,
                "periodo_evaluacion": f"{CURRENT_START} a {CURRENT_END}",
                "version_datos": data_version,
                "semilla": RANDOM_SEED,
                "remuestreos_bootstrap": AB_TEST_BOOTSTRAP_SAMPLES,
                "nivel_confianza": AB_TEST_CONFIDENCE,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(csv_path), artifact_path="ab_test")
        mlflow.log_artifact(str(md_path), artifact_path="ab_test")
    return md_path


def promote(version: str) -> None:
    """Mueve el alias ``champion`` a ``version``: la API la servirá al reiniciar.

    Args:
        version: Versión que pasa a ser la vigente.
    """
    model_name = REGISTERED_MODELS[QUESTION]
    mlflow.MlflowClient().set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    logger.success(f"[P1] @{CHAMPION_ALIAS} ahora apunta a la versión {version} de {model_name}")


def _load(alias: str) -> LoadedModel:
    """Carga una versión de P1 por alias, sin usar copias locales: debe venir del registry."""
    return load_registered_model(QUESTION, alias=alias, allow_cache_fallback=False)


@app.command("retrain")
def retrain_command() -> None:
    """Paso 4: reentrena @champion con datos hasta agosto de 2026 y lo registra como @challenger."""
    champion = _load(CHAMPION_ALIAS)
    datasets = build_ab_datasets(*load_or_ingest_monitoring_inputs())
    model = retrain_challenger(champion, datasets.reference)
    register_challenger(model, champion, datasets.reference)


@app.command("compare")
def compare_command(
    promote_winner: Annotated[
        bool,
        typer.Option(
            "--promote", help="Si gana @challenger, moverle el alias @champion en el registry."
        ),
    ] = False,
) -> None:
    """Paso 5: A/B testing de @champion contra @challenger en septiembre de 2026."""
    champion, challenger = _load(CHAMPION_ALIAS), _load(CHALLENGER_ALIAS)
    datasets = build_ab_datasets(*load_or_ingest_monitoring_inputs())
    result = run_ab_test(datasets, champion, challenger)
    md_path = log_ab_test(result, champion, challenger)

    logger.info(f"Tabla comparativa ({md_path}):\n{result.table.to_string(index=False)}")
    logger.info(
        f"MAE de B menos MAE de A: {result.mae_difference:+.4f} "
        f"(IC {AB_TEST_CONFIDENCE:.0%}: {result.ci_low:+.4f} a {result.ci_high:+.4f})"
    )
    if result.winner == CHALLENGER_ALIAS:
        logger.success(
            f"Gana @challenger (v{challenger.version}): tiene menos error en septiembre"
        )
        if promote_winner:
            promote(challenger.version)
        else:
            logger.info(
                "Para dejarla vigente: python -m mibici.modeling.ab_test compare --promote"
            )
    else:
        logger.success(f"Se conserva @champion (v{champion.version}): B no es mejor con evidencia")


if __name__ == "__main__":
    app()
