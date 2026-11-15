"""Reporte de drift de datos con Evidently (Entrega 05, paso 2).

Compara, para cada pregunta, el conjunto de referencia (periodo de entrenamiento) con el
conjunto actual (septiembre de 2026) que prepara ``monitoring.py`` y escribe en
``reports/evidently/``:

- ``<pregunta>_drift.html``: reporte interactivo de Evidently con el drift por variable
  (``DataDriftPreset``) y la calidad de los datos (``DataSummaryPreset``).
- ``<pregunta>_drift_resumen.csv``: una fila por variable con el método estadístico, el
  puntaje, el umbral y si hay drift.

Uso desde la raíz del repositorio:

    python -m mibici.drift --question all
    python -m mibici.drift --question p1 --synthetic
"""

from pathlib import Path
from typing import Annotated

from evidently import DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import pandas as pd
import typer

from mibici.config import DRIFT_SHARE_THRESHOLD, EVIDENTLY_REPORTS_DIR, EVIDENTLY_SAMPLE_SIZE
from mibici.monitoring import (
    VALID_QUESTIONS,
    MonitoringDatasets,
    load_or_ingest_monitoring_inputs,
    prepare_monitoring_data,
)

app = typer.Typer(help="Reporte de drift de datos con Evidently (referencia vs. periodo actual).")

SUMMARY_COLUMNS = ["column", "method", "score", "threshold", "drift"]


def _monitored_columns(datasets: MonitoringDatasets) -> list[str]:
    """Variables del modelo y, si la pregunta tiene, la variable objetivo."""
    return datasets.features + ([datasets.target] if datasets.target else [])


def _to_evidently(datasets: MonitoringDatasets) -> tuple[Dataset, Dataset]:
    """Convierte las tablas de referencia y actual al formato de Evidently.

    Las variables categóricas se convierten a ``int64``: las tablas las guardan como ``int8``
    para ahorrar memoria, y Evidently 0.7 rechaza ``numpy.int8`` como etiqueta de categoría
    al calcular la calidad de los datos. El objetivo de P1 (``departures``) es numérico y el
    de P2 (``risk``), categórico.

    Args:
        datasets: Conjuntos de referencia y actual de ``prepare_monitoring_data``.

    Returns:
        Tupla ``(reference, current)`` de ``evidently.Dataset``.
    """
    columns = _monitored_columns(datasets)
    numerical = list(datasets.numerical_features)
    categorical = list(datasets.categorical_features)
    if datasets.target:
        if pd.api.types.is_numeric_dtype(datasets.reference[datasets.target]):
            numerical.append(datasets.target)
        else:
            categorical.append(datasets.target)

    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    casts = {column: "int64" for column in datasets.categorical_features}
    reference = datasets.reference[columns].astype(casts)
    current = datasets.current[columns].astype(casts)
    return (
        Dataset.from_pandas(reference, data_definition=definition),
        Dataset.from_pandas(current, data_definition=definition),
    )


def build_drift_report(
    datasets: MonitoringDatasets, drift_share: float = DRIFT_SHARE_THRESHOLD
) -> Snapshot:
    """Corre el reporte de Evidently: drift por variable y calidad de los datos.

    Evidently elige la prueba de cada variable según su tipo y el tamaño de la muestra: con
    más de 1,000 filas usa distancias (Wasserstein normalizada en numéricas y Jensen-Shannon
    en categóricas, con umbral 0.1); con menos, pruebas estadísticas (K-S en numéricas y
    chi cuadrada en categóricas, con p-value de 0.05).

    Args:
        datasets: Conjuntos de referencia y actual de ``prepare_monitoring_data``.
        drift_share: Fracción de variables con drift a partir de la cual se considera que el
            conjunto completo tiene drift (``DRIFT_SHARE_THRESHOLD`` en ``config.py``).

    Returns:
        Resultado del reporte (``Snapshot``), que se puede guardar como HTML.
    """
    reference, current = _to_evidently(datasets)
    report = Report([DataDriftPreset(drift_share=drift_share), DataSummaryPreset()])
    return report.run(current_data=current, reference_data=reference)


def summarize_drift(snapshot: Snapshot) -> pd.DataFrame:
    """Extrae del reporte una tabla con el drift de cada variable.

    Con las pruebas estadísticas el puntaje es un p-value: hay drift si es **menor** que el
    umbral. Con las distancias hay drift si el puntaje es **mayor o igual** que el umbral.

    Args:
        snapshot: Resultado de ``build_drift_report``.

    Returns:
        Una fila por variable con ``column``, ``method``, ``score``, ``threshold`` y
        ``drift``; primero las variables con drift.
    """
    rows = []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if not config["type"].endswith(":ValueDrift"):
            continue
        method, score, threshold = config["method"], float(metric["value"]), config["threshold"]
        is_p_value = "p_value" in method
        drift = score < threshold if is_p_value else score >= threshold
        rows.append(
            {
                "column": config["column"],
                "method": method,
                "score": score,
                "threshold": threshold,
                "drift": drift,
            }
        )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary.sort_values("drift", ascending=False, kind="stable").reset_index(drop=True)


def save_drift_report(
    datasets: MonitoringDatasets,
    output_dir: Path = EVIDENTLY_REPORTS_DIR,
    drift_share: float = DRIFT_SHARE_THRESHOLD,
) -> tuple[Path, Path, pd.DataFrame]:
    """Genera el reporte de una pregunta y lo guarda como HTML y como tabla CSV.

    Args:
        datasets: Conjuntos de referencia y actual de ``prepare_monitoring_data``.
        output_dir: Carpeta de salida (``reports/evidently`` por defecto).
        drift_share: Fracción de variables con drift para declarar drift del conjunto.

    Returns:
        Ruta del HTML, ruta del CSV y la tabla de drift por variable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = build_drift_report(datasets, drift_share=drift_share)
    summary = summarize_drift(snapshot)

    html_path = output_dir / f"{datasets.question}_drift.html"
    csv_path = output_dir / f"{datasets.question}_drift_resumen.csv"
    snapshot.save_html(str(html_path))
    summary.to_csv(csv_path, index=False)

    n_drift, n_columns = int(summary["drift"].sum()), len(summary)
    share = n_drift / n_columns if n_columns else 0.0
    verdict = "hay drift del conjunto" if share >= drift_share else "sin drift del conjunto"
    drifted = ", ".join(summary.loc[summary["drift"], "column"]) or "ninguna"
    logger.success(
        f"[{datasets.question.upper()}] {n_drift} de {n_columns} variables con drift "
        f"({share:.0%}; umbral {drift_share:.0%}): {verdict}. Con drift: {drifted}"
    )
    logger.info(f"[{datasets.question.upper()}] Reporte: {html_path} | Resumen: {csv_path}")
    return html_path, csv_path, summary


@app.command()
def main(
    question: Annotated[str, typer.Option(help="Pregunta: p1, p2, p3 o all.")] = "all",
    sample_size: Annotated[
        int, typer.Option(help="Muestra determinista máxima por conjunto (0 = sin límite).")
    ] = EVIDENTLY_SAMPLE_SIZE,
    synthetic: Annotated[
        bool, typer.Option(help="Usar datos sintéticos en memoria sin requerir Parquet locales.")
    ] = False,
) -> None:
    """Genera los reportes de drift de Evidently de una pregunta o de las tres."""
    normalized = question.strip().lower()
    questions = list(VALID_QUESTIONS) if normalized == "all" else [normalized]
    trips, production_trips, stations = load_or_ingest_monitoring_inputs(synthetic=synthetic)
    for current_question in questions:
        datasets = prepare_monitoring_data(
            question=current_question,
            trips=trips,
            production_trips=production_trips,
            stations=stations,
            sample_size=sample_size if sample_size > 0 else None,
        )
        save_drift_report(datasets)


if __name__ == "__main__":
    app()
