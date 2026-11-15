"""Pruebas unitarias para la ingesta de producción y preparación de conjuntos (monitoring.py).

Verifica que se construyan los pares (reference, current) para P1, P2 y P3 con sus columnas
exactas, que los rezagos del arranque de septiembre de 2026 se alimenten del cierre de agosto
sin generar NaN artificiales, que P3 recalcule perfiles en ventanas arbitrarias y que se
rechacen preguntas o ventanas vacías sin requerir descarga de datos reales.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mibici.config import CURRENT_END, CURRENT_START, REFERENCE_END, REFERENCE_START
from mibici.features import (
    P1_FEATURES,
    P1_TARGET,
    P2_FEATURES,
    P2_TARGET,
    P3_FEATURES,
    build_p3_table,
)
from mibici.monitoring import (
    build_synthetic_monitoring_data,
    ingest_production_data,
    prepare_monitoring_data,
)


@pytest.fixture(scope="module")
def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera una sola vez los viajes históricos, de producción y estaciones sintéticas."""
    return build_synthetic_monitoring_data()


@pytest.mark.parametrize(
    ("question", "expected_features", "expected_target"),
    [
        ("p1", P1_FEATURES, P1_TARGET),
        ("p2", P2_FEATURES, P2_TARGET),
        ("p3", P3_FEATURES, None),
    ],
)
def test_prepare_monitoring_data_builds_reference_and_current_for_all_questions(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    question: str,
    expected_features: list[str],
    expected_target: str | None,
) -> None:
    """Caso normal: construye reference y current con las columnas del modelo para P1, P2 y P3."""
    trips, production_trips, stations = synthetic_inputs
    datasets = prepare_monitoring_data(
        question=question,
        trips=trips,
        production_trips=production_trips,
        stations=stations,
    )
    assert datasets.question == question
    assert datasets.features == expected_features
    assert datasets.target == expected_target
    assert datasets.reference_start == REFERENCE_START
    assert datasets.reference_end == REFERENCE_END
    assert datasets.current_start == CURRENT_START
    assert datasets.current_end == CURRENT_END
    assert not datasets.reference.empty
    assert not datasets.current.empty

    for col in expected_features:
        assert col in datasets.reference.columns
        assert col in datasets.current.columns
    if expected_target is not None:
        assert expected_target in datasets.reference.columns
        assert expected_target in datasets.current.columns


def test_p1_and_p2_current_sets_keep_continuous_lags_from_august(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Verifica que el 1 de septiembre de 2026 no tenga NaN en rezagos gracias a agosto."""
    trips, production_trips, stations = synthetic_inputs
    ds_p1 = prepare_monitoring_data("p1", trips, production_trips, stations)
    ds_p2 = prepare_monitoring_data("p2", trips, production_trips, stations)

    first_day_p1 = ds_p1.current.loc[ds_p1.current["period_start"] < pd.Timestamp("2026-09-02")]
    assert not first_day_p1.empty
    assert first_day_p1[P1_FEATURES].isna().sum().sum() == 0

    first_day_p2 = ds_p2.current.loc[ds_p2.current["period_start"] < pd.Timestamp("2026-09-02")]
    assert not first_day_p2.empty
    assert first_day_p2[P2_FEATURES].isna().sum().sum() == 0


def test_prepare_monitoring_data_respects_sample_size(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Aplica muestreo determinista cuando las tablas superan sample_size."""
    trips, production_trips, stations = synthetic_inputs
    ds = prepare_monitoring_data("p1", trips, production_trips, stations, sample_size=50)
    assert len(ds.reference) == 50
    assert len(ds.current) == 50


def test_prepare_monitoring_data_rejects_unknown_question_or_empty_window(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Caso límite: rechaza preguntas desconocidas o ventanas de fechas sin registros."""
    trips, production_trips, stations = synthetic_inputs
    with pytest.raises(ValueError, match="desconocida"):
        prepare_monitoring_data("p4", trips, production_trips, stations)

    with pytest.raises(ValueError, match="quedó vacío"):
        prepare_monitoring_data(
            "p1",
            trips,
            production_trips,
            stations,
            current_start="2027-01-01",
            current_end="2027-02-01",
        )


def test_build_p3_table_rejects_invalid_or_empty_date_range(
    synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Caso límite P3: rechaza ventanas invertidas o sin viajes."""
    trips, _, stations = synthetic_inputs
    with pytest.raises(ValueError, match="al menos un día"):
        build_p3_table(trips, stations, start_date="2026-09-01", end_date="2026-09-01")

    with pytest.raises(ValueError, match="No hay movimientos"):
        build_p3_table(trips, stations, start_date="2028-01-01", end_date="2028-02-01")


def test_ingest_production_data_cleans_local_csv_and_writes_parquet(
    tmp_path: Path,
) -> None:
    """Verifica que ingest_production_data limpie el CSV del mes y guarde el Parquet separado."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    stations_path = tmp_path / "nomenclatura.parquet"
    output_path = tmp_path / "meses_produccion.parquet"

    pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["GDL-001", "GDL-002"],
            "latitude": [20.67, 20.68],
            "longitude": [-103.35, -103.36],
            "dpcapacity": [15, 20],
        }
    ).to_parquet(stations_path, index=False)

    csv_path = raw_dir / "datos_abiertos_2026_09.csv"
    pd.DataFrame(
        {
            "Viaje_Id": [101, 102, 103],
            "Usuario_Id": [999, 998, 997],
            "Genero": ["M", "F", "M"],
            "Año_de_nacimiento": [1990, 1995, 1988],
            "Inicio_del_viaje": [
                "2026-09-01 08:10:00",
                "2026-09-01 09:00:00",
                "2026-09-01 02:30:00",  # hora sin servicio (se descarta)
            ],
            "Fin_del_viaje": [
                "2026-09-01 08:25:00",
                "2026-09-01 09:20:00",
                "2026-09-01 02:45:00",
            ],
            "Origen_Id": [1, 2, 1],
            "Destino_Id": [2, 1, 2],
        }
    ).to_csv(csv_path, index=False)

    result_path = ingest_production_data(
        month_date="2026-09-01",
        raw_dir=raw_dir,
        stations_path=stations_path,
        output_path=output_path,
        overwrite=False,
    )
    assert result_path == output_path
    assert output_path.exists()
    df_saved = pd.read_parquet(output_path)
    assert len(df_saved) == 2
    assert "usuario_id" not in df_saved.columns
