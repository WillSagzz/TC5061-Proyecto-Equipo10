"""Pruebas de ``features.py``: conteo por estación y periodo (P1 y P2) y etiqueta de riesgo."""

import pandas as pd
import pytest

from mibici.config import P2_INTERVAL_HOURS
from mibici.features import NO_RISK, RISK_EMPTY, RISK_FULL, build_station_period_table, label_risk

# Estaciones con los nombres de ``load_data``; ``build_station_period_table`` usa ``station_id``
STATIONS = pd.DataFrame({"station_id": [1, 2, 3], "capacity": [10, 10, 10]})


def _example_trips() -> pd.DataFrame:
    """Cuatro viajes en dos días, con los nombres de columna de ``load_data``.

    Returns:
        Tres viajes de la estación 1 a la 2 el lunes en la mañana y uno de la 3 a la 1 el
        martes: la estación 3 abre el segundo día.
    """
    trips = pd.DataFrame(
        [
            ("2025-03-03 08:05", "2025-03-03 08:20", 1, 2),
            ("2025-03-03 08:30", "2025-03-03 08:45", 1, 2),
            ("2025-03-03 08:59", "2025-03-03 09:10", 1, 2),
            ("2025-03-04 10:15", "2025-03-04 10:40", 3, 1),
        ],
        columns=["start_time", "end_time", "origin_station_id", "destination_station_id"],
    )
    trips["start_time"] = pd.to_datetime(trips["start_time"])
    trips["end_time"] = pd.to_datetime(trips["end_time"])
    return trips


def _station_table(freq: str = "h") -> pd.DataFrame:
    """Tabla de ``build_station_period_table`` con índice estación y periodo para buscar filas.

    Args:
        freq: Duración del periodo: ``"h"`` como en P1 o ``"3h"`` como en P2.

    Returns:
        Tabla de los viajes de ``_example_trips`` indexada por ``station_id`` y
        ``period_start``.
    """
    table = build_station_period_table(_example_trips(), STATIONS, freq=freq)
    return table.set_index(["station_id", "period_start"])


def test_build_station_period_table_counts_trips_per_station_and_hour() -> None:
    """Caso normal: cada viaje cuenta como salida en su hora de inicio y llegada en la de fin.

    Las horas se redondean hacia abajo: 08:05, 08:30 y 08:59 cuentan en las 08:00.
    """
    table = _station_table()
    at_8 = pd.Timestamp("2025-03-03 08:00")
    at_9 = pd.Timestamp("2025-03-03 09:00")

    assert table.loc[(1, at_8), "departures"] == 3
    assert table.loc[(2, at_8), "arrivals"] == 2
    assert table.loc[(2, at_9), "arrivals"] == 1
    assert table.loc[(1, at_8), "net_flow"] == -3  # llegadas − salidas


def test_build_station_period_table_counts_trips_per_interval_as_in_p2() -> None:
    """Caso límite: con intervalos de 3 horas (P2), los viajes de 06:00 a 08:59 cuentan juntos.

    Cada día tiene 8 intervalos: 00:00, 03:00, 06:00, …, 21:00.
    """
    table = _station_table(freq=f"{P2_INTERVAL_HOURS}h")
    at_6 = pd.Timestamp("2025-03-03 06:00")
    at_9 = pd.Timestamp("2025-03-03 09:00")

    assert len(table) == 3 * 16  # 3 estaciones × 16 intervalos (dos días completos)
    assert table.loc[(1, at_6), "departures"] == 3
    assert table.loc[(2, at_6), "arrivals"] == 2
    assert table.loc[(2, at_9), "arrivals"] == 1


def test_build_station_period_table_fills_hours_without_trips_with_zero() -> None:
    """Caso límite: una hora sin viajes vale 0 y no queda vacía; la malla no tiene huecos."""
    table = _station_table()

    assert len(table) == 3 * 48  # 3 estaciones × 48 horas (dos días completos)
    assert table.loc[(1, pd.Timestamp("2025-03-03 12:00")), "departures"] == 0
    assert table[["departures", "arrivals"]].notna().all().all()


def test_build_station_period_table_marks_new_stations_inactive_before_opening() -> None:
    """Caso límite: la estación 3 abrió el segundo día; antes de eso sus ceros no son demanda."""
    station_3 = _station_table().loc[3]
    first_day = station_3.index < pd.Timestamp("2025-03-04")

    assert not station_3.loc[first_day, "active"].any()
    assert station_3.loc[~first_day, "active"].all()


def test_label_risk_labels_empty_full_and_no_risk() -> None:
    """Caso normal: con capacidad 10, el umbral es 0.4 × 10 = 4 bicicletas."""
    risk = label_risk(net_flow=pd.Series([-6, 0, 6]), capacity=pd.Series([10, 10, 10]))

    assert risk.tolist() == [RISK_EMPTY, NO_RISK, RISK_FULL]


@pytest.mark.parametrize(
    ("net_flow", "capacity", "expected"),
    [
        pytest.param(-4, 10, RISK_EMPTY, id="exactly_minus_threshold"),
        pytest.param(4, 10, RISK_FULL, id="exactly_plus_threshold"),
        pytest.param(-3, 10, NO_RISK, id="just_above_minus_threshold"),
        pytest.param(3, 10, NO_RISK, id="just_below_plus_threshold"),
        pytest.param(-4, 20, NO_RISK, id="same_flow_bigger_station"),
    ],
)
def test_label_risk_at_the_threshold(net_flow: int, capacity: int, expected: str) -> None:
    """Caso límite: justo en el umbral sí hay riesgo, y el umbral depende de la capacidad."""
    risk = label_risk(net_flow=pd.Series([net_flow]), capacity=pd.Series([capacity]))

    assert risk.iloc[0] == expected
