"""Pruebas de ``add_lag_features`` (P1 y P2): rezagos, promedio móvil y regla del horizonte."""

import numpy as np
import pandas as pd
import pytest

from mibici.config import LAG_HOURS, P1_HORIZON_HOURS, P1_LAG_HOURS
from mibici.features import add_lag_features

HOURS = 72  # tres días completos por estación
FIRST_HOUR = pd.Timestamp("2025-03-03")  # lunes a las 0:00


def _hourly_table(stations: tuple[int, ...] = (1, 2), hours: int = HOURS) -> pd.DataFrame:
    """Malla regular estación × hora como la que entrega ``build_station_period_table``.

    Las salidas valen ``estación × 1000 + número de hora`` (0, 1, 2, …) y las llegadas lo
    mismo más 500, así que cada celda es única y se sabe exactamente de qué estación y de qué
    hora salió cada rezago.

    Args:
        stations: Identificadores de las estaciones de la malla.
        hours: Horas de la malla, a partir de ``FIRST_HOUR``.

    Returns:
        Tabla con ``station_id``, ``period_start``, ``departures``, ``arrivals`` y ``active``
        (siempre ``True``).
    """
    periods = pd.date_range(FIRST_HOUR, periods=hours, freq="h")
    frames = [
        pd.DataFrame(
            {
                "station_id": station,
                "period_start": periods,
                "departures": station * 1000 + np.arange(hours),
                "arrivals": station * 1000 + 500 + np.arange(hours),
                "active": True,
            }
        )
        for station in stations
    ]
    return pd.concat(frames, ignore_index=True)


def _indexed(table: pd.DataFrame) -> pd.DataFrame:
    """Indexa por estación y periodo para leer celdas con ``.loc[(estación, hora)]``."""
    return table.set_index(["station_id", "period_start"])


def _hour(index: int) -> pd.Timestamp:
    """Hora número ``index`` de la malla (0 = ``FIRST_HOUR``)."""
    return FIRST_HOUR + pd.Timedelta(hours=index)


def test_lag_24_is_the_value_of_the_same_station_24_hours_before() -> None:
    """Caso normal (P1): ``departures_lag_24`` en t es ``departures`` de la misma estación en t − 24 h."""
    result = _indexed(
        add_lag_features(_hourly_table(), {"departures": [24]}, horizon=P1_HORIZON_HOURS)
    )

    assert result.loc[(1, _hour(30)), "departures_lag_24"] == 1000 + 6
    assert result.loc[(2, _hour(30)), "departures_lag_24"] == 2000 + 6
    assert result.loc[(2, _hour(71)), "departures_lag_24"] == 2000 + 47
    assert result["departures_lag_24"].dtype == "float32"


def test_every_lag_matches_the_own_station_history_across_the_grid() -> None:
    """Caso normal: en toda la malla, cada rezago es el valor de la propia estación n periodos antes.

    Con horizonte 1 (P2) el rezago de 1 periodo está permitido.
    """
    result = add_lag_features(
        _hourly_table(), {"departures": [1, 24], "arrivals": [24]}, horizon=1
    )

    grouped = result.groupby("station_id")
    for column, step in [("departures", 1), ("departures", 24), ("arrivals", 24)]:
        expected = grouped[column].shift(step).astype("float32")
        pd.testing.assert_series_equal(result[f"{column}_lag_{step}"], expected, check_names=False)


def test_lags_do_not_change_when_the_future_changes() -> None:
    """Caso normal: reescribir el tercer día no altera ningún rezago: la hora t solo mira antes de t − 24 h."""
    table = _hourly_table()
    altered = table.copy()
    altered.loc[altered["period_start"] >= _hour(48), ["departures", "arrivals"]] = 9999
    lags = {"departures": [24], "arrivals": [24]}

    original = add_lag_features(table, lags, horizon=P1_HORIZON_HOURS)
    changed = add_lag_features(altered, lags, horizon=P1_HORIZON_HOURS)

    new_columns = ["departures_lag_24", "arrivals_lag_24", "departures_mean_3_lag_24"]
    pd.testing.assert_frame_equal(original[new_columns], changed[new_columns])


def test_first_hours_without_history_are_nan_not_zero() -> None:
    """Caso límite: las primeras 24 h de cada estación no tienen "ayer": quedan NaN, no 0.

    El promedio móvil necesita las 3 horas cerradas que terminan 24 h antes (t − 26, t − 25 y
    t − 24), así que arranca dos horas después que el rezago.
    """
    result = add_lag_features(_hourly_table(), {"departures": [24]}, window=3, horizon=24)

    for _, rows in result.groupby("station_id"):
        assert rows["departures_lag_24"].iloc[:24].isna().all()
        assert rows["departures_lag_24"].iloc[24:].notna().all()
        assert rows["departures_mean_3_lag_24"].iloc[:26].isna().all()
        assert rows["departures_mean_3_lag_24"].iloc[26:].notna().all()


def test_rolling_mean_covers_the_window_closed_one_horizon_before() -> None:
    """Caso normal: con horizonte 24 y ventana 3, el promedio en t cubre t − 26, t − 25 y t − 24.

    Con horizonte 1 (P2) cubre t − 3, t − 2 y t − 1 y la columna no lleva sufijo de rezago.
    """
    p1 = _indexed(add_lag_features(_hourly_table(), {"departures": [24]}, window=3, horizon=24))
    p2 = _indexed(add_lag_features(_hourly_table(), {"departures": [1]}, window=3, horizon=1))

    assert p1.loc[(1, _hour(30)), "departures_mean_3_lag_24"] == pytest.approx(1000 + 5)
    assert "departures_mean_3" in p2.columns
    assert "departures_mean_3_lag_1" not in p2.columns
    assert p2.loc[(1, _hour(30)), "departures_mean_3"] == pytest.approx(1000 + 28)


@pytest.mark.parametrize(
    "lags",
    [
        pytest.param([1], id="one_hour"),
        pytest.param([23], id="one_hour_short_of_the_horizon"),
        pytest.param([1, 24], id="mixed_with_a_valid_lag"),
    ],
)
def test_lag_shorter_than_the_horizon_is_rejected(lags: list[int]) -> None:
    """Caso límite (P1): un rezago menor a 24 h usaría horas del día que se pronostica: error, no silencio."""
    with pytest.raises(ValueError, match="menores que el horizonte"):
        add_lag_features(_hourly_table(), {"departures": lags}, horizon=P1_HORIZON_HOURS)


def test_lag_equal_to_the_horizon_is_accepted() -> None:
    """Caso límite: el rezago igual al horizonte es el más reciente permitido."""
    result = add_lag_features(
        _hourly_table(), {"departures": [P1_HORIZON_HOURS]}, horizon=P1_HORIZON_HOURS
    )

    assert f"departures_lag_{P1_HORIZON_HOURS}" in result.columns


def test_p1_config_only_keeps_lags_of_at_least_one_day() -> None:
    """Caso límite: ``LAG_HOURS`` (con el de 1 h) se rechaza con el horizonte de P1; ``P1_LAG_HOURS`` pasa."""
    assert min(P1_LAG_HOURS) >= P1_HORIZON_HOURS
    with pytest.raises(ValueError, match="departures_lag_1"):
        add_lag_features(_hourly_table(), {"departures": LAG_HOURS}, horizon=P1_HORIZON_HOURS)

    result = add_lag_features(
        _hourly_table(hours=24 * 8), {"departures": P1_LAG_HOURS}, horizon=P1_HORIZON_HOURS
    )

    assert all(f"departures_lag_{hours}" in result.columns for hours in P1_LAG_HOURS)
    assert result[f"departures_lag_{max(P1_LAG_HOURS)}"].notna().any()


def test_irregular_grid_is_rejected() -> None:
    """Caso límite: si a una estación le falta una hora, un rezago de n filas ya no son n horas: error."""
    table = _hourly_table().drop(index=5)

    with pytest.raises(ValueError, match="no es regular"):
        add_lag_features(table, {"departures": [1]})


def test_inactive_periods_count_as_unknown_history() -> None:
    """Caso límite: la estación 2 abre el segundo día; los rezagos que caen antes quedan NaN, no 0."""
    table = _hourly_table()
    opens = _hour(24)
    table.loc[(table["station_id"] == 2) & (table["period_start"] < opens), "active"] = False

    result = _indexed(add_lag_features(table, {"departures": [24]}, horizon=24))

    assert np.isnan(result.loc[(2, _hour(47)), "departures_lag_24"])  # apunta al día 1, inactivo
    assert result.loc[(2, _hour(48)), "departures_lag_24"] == 2000 + 24  # primera hora activa
    assert result.loc[(1, _hour(47)), "departures_lag_24"] == 1000 + 23  # la 1 sí estaba activa
    assert result.loc[(2, _hour(10)), "departures"] == 2000 + 10  # la columna original no cambia


def test_result_is_sorted_and_independent_of_input_order() -> None:
    """Caso límite: la tabla desordenada da los mismos rezagos, ordenados por estación y periodo."""
    table = _hourly_table()
    shuffled = table.sample(frac=1, random_state=0)

    expected = add_lag_features(table, {"departures": [24]}, horizon=24)
    result = add_lag_features(shuffled, {"departures": [24]}, horizon=24)

    pd.testing.assert_frame_equal(result, expected)
    assert result["station_id"].is_monotonic_increasing
