"""Pruebas de las tablas de modelado: ``build_p1_table``, ``build_p2_table`` y ``build_p3_table``."""

import numpy as np
import pandas as pd
import pytest

from mibici.config import (
    DAY_PERIODS,
    NO_SERVICE_HOURS,
    P1_HORIZON_HOURS,
    P2_INTERVAL_HOURS,
    ROLLING_WINDOW,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.features import (
    P1_FEATURES,
    P1_TARGET,
    P2_FEATURES,
    P2_TARGET,
    P3_FEATURES,
    RISK_CLASSES,
    build_p1_table,
    build_p2_table,
    build_p3_table,
    label_risk,
)
from mibici.validation import build_synthetic_raw_data


@pytest.fixture(scope="module")
def synthetic_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Viajes y estaciones sintéticos de ``validation.py``, construidos una sola vez por módulo."""
    return build_synthetic_raw_data()


@pytest.fixture(scope="module")
def p1_table(synthetic_raw: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Tabla de P1 con los datos sintéticos."""
    return build_p1_table(*synthetic_raw)


@pytest.fixture(scope="module")
def p2_table(synthetic_raw: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Tabla de P2 con los datos sintéticos."""
    return build_p2_table(*synthetic_raw)


@pytest.fixture(scope="module")
def p3_table(synthetic_raw: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Tabla de P3 con los datos sintéticos."""
    return build_p3_table(*synthetic_raw)


# -----------------------------------------------------------------------------
# P1
# -----------------------------------------------------------------------------
def test_build_p1_table_has_the_model_columns_and_a_complete_target(
    p1_table: pd.DataFrame,
) -> None:
    """Caso normal: trae todas las ``P1_FEATURES`` y el objetivo sin faltantes, una fila por estación y hora."""
    assert set(P1_FEATURES) <= set(p1_table.columns)
    assert p1_table[P1_TARGET].notna().all()
    assert (p1_table[P1_TARGET] >= 0).all()
    assert not p1_table.duplicated(["station_id", "period_start"]).any()
    assert "active" not in p1_table.columns


def test_build_p1_table_excludes_hours_without_service_but_keeps_them_as_history(
    p1_table: pd.DataFrame,
) -> None:
    """Caso límite: de 1:00 a 4:59 no se predice, pero esas horas sí cuentan en el promedio móvil.

    El promedio de las 5:00 cubre las 3:00, 4:00 y 5:00 del día anterior; si las horas sin
    servicio se hubieran quitado antes de calcularlo, ese promedio no existiría.
    """
    hours = p1_table["period_start"].dt.hour
    assert not hours.isin(NO_SERVICE_HOURS).any()

    first_week = p1_table["period_start"].min() + pd.Timedelta(days=7)
    at_5 = p1_table.loc[(hours == 5) & (p1_table["period_start"] > first_week)]
    assert at_5[f"departures_mean_{ROLLING_WINDOW}_lag_{P1_HORIZON_HOURS}"].notna().all()


def test_build_p1_table_first_hour_of_each_station_has_no_history(
    p1_table: pd.DataFrame,
) -> None:
    """Caso límite: la primera hora de cada estación no tiene "ayer": sus rezagos son NaN, no 0."""
    first_rows = p1_table.sort_values("period_start").groupby("station_id").head(1)

    assert first_rows["departures_lag_24"].isna().all()
    assert first_rows["net_flow_lag_24"].isna().all()


# -----------------------------------------------------------------------------
# P2
# -----------------------------------------------------------------------------
def test_build_p2_table_labels_risk_from_the_interval_flow(p2_table: pd.DataFrame) -> None:
    """Caso normal: trae ``P2_FEATURES``, ocho intervalos por día y las tres clases, etiquetadas con el flujo neto."""
    assert set(P2_FEATURES) <= set(p2_table.columns)
    assert set(p2_table[P2_TARGET].unique()) == set(RISK_CLASSES)
    assert p2_table["interval_of_day"].between(0, 24 // P2_INTERVAL_HOURS - 1).all()
    assert not p2_table.duplicated(["station_id", "period_start"]).any()

    expected = label_risk(p2_table["net_flow"], p2_table["capacity"])
    assert (p2_table[P2_TARGET] == expected).all()


def test_build_p2_table_lags_count_in_intervals_not_hours(p2_table: pd.DataFrame) -> None:
    """Caso límite (horizonte de 1 intervalo): el rezago 1 es el intervalo anterior y el 8, el día anterior."""
    assert f"departures_mean_{ROLLING_WINDOW}" in P2_FEATURES  # sin sufijo: horizonte 1
    station = p2_table.loc[p2_table["station_id"] == p2_table["station_id"].iloc[0]]
    station = station.sort_values("period_start").reset_index(drop=True)
    step = pd.Timedelta(hours=P2_INTERVAL_HOURS)

    assert (station["period_start"].diff().dropna() == step).all()
    assert np.isnan(station.loc[0, "departures_lag_1"])
    np.testing.assert_array_equal(
        station["departures_lag_1"].iloc[1:], station["departures"].shift(1).iloc[1:]
    )
    np.testing.assert_array_equal(
        station["departures_lag_8"].iloc[8:], station["departures"].shift(8).iloc[8:]
    )


def test_stations_without_capacity_are_excluded_from_p2_and_p3(
    synthetic_raw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Caso límite: sin anclajes no hay umbral de riesgo ni presión por anclaje: la estación sale de P2 y P3."""
    trips, stations = synthetic_raw
    stations = stations.copy()
    no_docks = stations["station_id"].iloc[-1]
    stations.loc[stations["station_id"] == no_docks, "capacity"] = 0

    assert no_docks not in build_p2_table(trips, stations)["station_id"].to_numpy()
    assert no_docks not in build_p3_table(trips, stations)["station_id"].to_numpy()


# -----------------------------------------------------------------------------
# P3
# -----------------------------------------------------------------------------
def test_build_p3_table_profiles_each_station_once(
    p3_table: pd.DataFrame, synthetic_raw: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Caso normal: una fila por estación, sin faltantes, periodos del día que suman 100 y balances en [−1, 1]."""
    _, stations = synthetic_raw

    assert sorted(p3_table["station_id"]) == sorted(stations["station_id"])
    assert p3_table[P3_FEATURES].notna().all().all()
    period_pct = p3_table[[f"{period}_pct" for period in DAY_PERIODS]].sum(axis=1)
    assert np.allclose(period_pct, 100)
    assert p3_table["weekdays_pct"].between(0, 100).all()
    assert p3_table[["balance_morning", "balance_afternoon"]].abs().le(1).all().all()
    assert (p3_table["log_daily_use_per_dock"] > 0).all()


def test_build_p3_table_uses_only_training_movements_and_signs_the_balance() -> None:
    """Caso límite: una estación con viajes solo en validación no entra; si solo salen bicicletas, el balance es −1."""
    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "latitude": [20.66, 20.67, 20.68],
            "longitude": [-103.38, -103.37, -103.36],
            "capacity": [10, 10, 10],
        }
    )
    in_train = pd.Timestamp(TRAIN_START) + pd.Timedelta(days=8, hours=8)  # 8:00: mañana
    in_val = pd.Timestamp(VALIDATION_START) + pd.Timedelta(days=8, hours=8)
    trips = pd.DataFrame(
        {
            "trip_id": [1, 2, 3],
            "start_time": [in_train, in_train, in_val],
            "end_time": [t + pd.Timedelta(minutes=15) for t in (in_train, in_train, in_val)],
            "origin_station_id": [1, 1, 3],
            "destination_station_id": [2, 2, 3],
        }
    )

    table = build_p3_table(trips, stations).set_index("station_id")

    assert table.index.tolist() == [1, 2]  # la 3 solo se movió en validación
    assert table.loc[1, "morning_pct"] == 100
    assert table.loc[1, "balance_morning"] == -1  # solo salidas
    assert table.loc[2, "balance_morning"] == 1  # solo llegadas
    assert table.loc[1, "balance_afternoon"] == 0  # sin movimientos en la tarde
    assert table.loc[1, "weekdays_pct"] == (100 if in_train.dayofweek < 5 else 0)
    n_days = (pd.Timestamp(VALIDATION_START) - pd.Timestamp(TRAIN_START)).days
    assert table.loc[1, "log_daily_use_per_dock"] == pytest.approx(np.log1p(2 / n_days / 10))
