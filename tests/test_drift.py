"""Pruebas del reporte de drift con Evidently (``drift.py``) con datos sintéticos."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mibici.drift import SUMMARY_COLUMNS, build_drift_report, save_drift_report, summarize_drift
from mibici.monitoring import (
    MonitoringDatasets,
    build_synthetic_monitoring_data,
    prepare_monitoring_data,
)

# Muestra pequeña para que las pruebas tarden segundos
SAMPLE_SIZE = 2_000


def _datasets(question: str) -> MonitoringDatasets:
    """Conjuntos de referencia y actual de una pregunta, con los datos sintéticos de Alan.

    En los datos sintéticos, septiembre de 2026 tiene el doble de viajes y otras estaciones
    de origen y destino que el periodo de referencia: un drift hecho a propósito.

    Args:
        question: ``p1``, ``p2`` o ``p3``.

    Returns:
        Conjuntos de ``prepare_monitoring_data``.
    """
    trips, production_trips, stations = build_synthetic_monitoring_data()
    return prepare_monitoring_data(
        question, trips, production_trips, stations, sample_size=SAMPLE_SIZE
    )


def _evidently_drift_count(datasets: MonitoringDatasets) -> int:
    """Número de variables con drift según el propio Evidently (``DriftedColumnsCount``)."""
    for metric in build_drift_report(datasets).dict()["metrics"]:
        if metric["config"]["type"].endswith(":DriftedColumnsCount"):
            return int(metric["value"]["count"])
    raise AssertionError("El reporte no trae DriftedColumnsCount")


@pytest.mark.parametrize("question", ["p1", "p2", "p3"])
def test_save_drift_report_writes_html_and_summary(question: str, tmp_path: Path) -> None:
    """Caso normal: escribe el HTML y una fila por variable (y por objetivo) en el CSV."""
    datasets = _datasets(question)
    expected_columns = datasets.features + ([datasets.target] if datasets.target else [])

    html_path, csv_path, summary = save_drift_report(datasets, output_dir=tmp_path)

    assert html_path == tmp_path / f"{question}_drift.html"
    assert html_path.stat().st_size > 0
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert sorted(summary["column"]) == sorted(expected_columns)
    assert pd.read_csv(csv_path).shape == summary.shape


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("p1", id="p1_distances_many_rows"),
        pytest.param("p3", id="p3_p_values_few_stations"),
    ],
)
def test_summarize_drift_matches_evidently_count(question: str) -> None:
    """Caso normal: la tabla marca con drift las mismas variables que cuenta Evidently.

    P1 tiene miles de filas y Evidently usa distancias (drift si el puntaje es mayor o igual
    al umbral); P3 tiene pocas estaciones y usa p-values (drift si es menor al umbral).
    """
    datasets = _datasets(question)

    summary = summarize_drift(build_drift_report(datasets))

    assert int(summary["drift"].sum()) == _evidently_drift_count(datasets)


def test_drift_is_detected_in_the_synthetic_demand_increase() -> None:
    """Caso normal: el aumento de viajes de septiembre aparece como drift del objetivo de P1."""
    summary = summarize_drift(build_drift_report(_datasets("p1"))).set_index("column")

    assert summary.loc["departures", "drift"]
    assert not summary.loc["station_id", "drift"]  # las mismas 6 estaciones en ambos periodos


def test_identical_periods_show_no_drift() -> None:
    """Caso límite: si el periodo actual es igual al de referencia, ninguna variable cambia."""
    datasets = _datasets("p1")
    same_data = replace(datasets, current=datasets.reference.copy())

    summary = summarize_drift(build_drift_report(same_data))

    assert not summary["drift"].any()


def test_report_accepts_int8_categorical_columns() -> None:
    """Caso límite: las tablas guardan las categóricas como int8, que Evidently no acepta solo.

    ``drift.py`` las convierte a int64 antes del reporte; sin esa conversión, Evidently falla
    con ``numpy.int8 not supported as Label``.
    """
    datasets = _datasets("p1")
    assert datasets.reference["hour_of_day"].dtype == "int8"

    summary = summarize_drift(build_drift_report(datasets))

    assert "hour_of_day" in set(summary["column"])
