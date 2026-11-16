"""Pruebas del reporte de drift con Evidently (``drift.py``) con datos sintéticos."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici import drift
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


@pytest.mark.parametrize("question", ["p1", "p3"])
def test_known_stations_only_drops_the_new_stations(question: str) -> None:
    """Opción ``--known-stations-only``: el conjunto actual queda sin estaciones nuevas.

    Se quitan del histórico los viajes de una estación para que solo exista en septiembre,
    como las estaciones que abrieron después del periodo de referencia.
    """
    trips, production_trips, stations = build_synthetic_monitoring_data()
    new_station = int(stations["station_id"].max())
    is_old = (trips["origin_station_id"] != new_station) & (
        trips["destination_station_id"] != new_station
    )
    inputs = (trips.loc[is_old].reset_index(drop=True), production_trips, stations)

    base = prepare_monitoring_data(question, *inputs)
    known = prepare_monitoring_data(question, *inputs, known_stations_only=True)

    assert new_station in set(base.current["station_id"])
    assert new_station not in set(known.current["station_id"])
    assert set(known.current["station_id"]) <= set(known.reference["station_id"])
    assert len(known.reference) == len(base.reference)


def test_suffix_writes_new_files_next_to_the_base_report(tmp_path: Path) -> None:
    """Opción ``--suffix``: los archivos llevan el sufijo y los del reporte base no cambian."""
    datasets = _datasets("p3")
    _, base_csv, _ = save_drift_report(datasets, output_dir=tmp_path)
    base_content = base_csv.read_bytes()

    html_path, csv_path, _ = save_drift_report(datasets, output_dir=tmp_path, suffix="conocidas")

    assert html_path == tmp_path / "p3_drift_conocidas.html"
    assert csv_path == tmp_path / "p3_drift_conocidas_resumen.csv"
    assert base_csv.read_bytes() == base_content
    assert len(list(tmp_path.iterdir())) == 4


def test_command_options_reach_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opciones ``--reference-start`` y ``--reference-end``, junto con las otras dos.

    El comando corre con los datos sintéticos y escribe en una carpeta temporal en lugar de
    ``reports/evidently/``.
    """
    received: list[MonitoringDatasets] = []

    def _save_to_tmp(datasets: MonitoringDatasets, suffix: str = "") -> None:
        received.append(datasets)
        save_drift_report(datasets, output_dir=tmp_path, suffix=suffix)

    monkeypatch.setattr(drift, "save_drift_report", _save_to_tmp)
    arguments = ["--question", "p3", "--synthetic", "--known-stations-only"]
    arguments += ["--reference-start", "2025-09-01", "--reference-end", "2025-10-01"]
    arguments += ["--suffix", "mismo_mes"]

    result = CliRunner().invoke(drift.app, arguments)

    assert result.exit_code == 0, result.output
    assert (received[0].reference_start, received[0].reference_end) == ("2025-09-01", "2025-10-01")
    assert (tmp_path / "p3_drift_mismo_mes.html").exists()
    assert (tmp_path / "p3_drift_mismo_mes_resumen.csv").exists()
