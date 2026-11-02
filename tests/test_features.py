"""Pruebas de ``features.py``: funciones compartidas (partición, contrato, versión y calendario),
conteo por estación y periodo (P1 y P2) y etiqueta de riesgo (P2).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from mibici.config import (
    DATA_END,
    DATAPACKAGE_PATH,
    P2_INTERVAL_HOURS,
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
)
from mibici.features import (
    NO_RISK,
    RISK_EMPTY,
    RISK_FULL,
    add_calendar_features,
    build_station_period_table,
    get_data_version,
    label_risk,
    split_by_date,
    validate_contract,
)

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


# -----------------------------------------------------------------------------
# split_by_date
# -----------------------------------------------------------------------------
def _midpoint(start: str, end: str) -> pd.Timestamp:
    """Fecha a la mitad de un conjunto de la partición, para no depender de los cortes exactos.

    Args:
        start: Fecha de inicio del conjunto (incluida).
        end: Fecha de inicio del conjunto siguiente (excluida).

    Returns:
        El instante a la mitad del intervalo.
    """
    return pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2


# Una fecha claramente dentro de cada conjunto de ``config.py``
IN_TRAIN = _midpoint(TRAIN_START, VALIDATION_START)
IN_VAL = _midpoint(VALIDATION_START, TEST_START)
IN_TEST = _midpoint(TEST_START, DATA_END)


def _dated_table(dates: list) -> pd.DataFrame:
    """Tabla mínima para ``split_by_date``: una fila por fecha, numerada en ``row``.

    Args:
        dates: Fechas de ``period_start``, en el orden en que se numeran.

    Returns:
        Tabla con ``period_start`` y ``row`` (posición de la fecha en ``dates``).
    """
    return pd.DataFrame({"period_start": pd.to_datetime(dates), "row": range(len(dates))})


def test_split_by_date_assigns_each_row_to_its_period() -> None:
    """Caso normal: cada fila cae en el conjunto de su fecha y los tres van en orden cronológico."""
    table = _dated_table([IN_TRAIN, IN_TRAIN + pd.Timedelta(days=1), IN_VAL, IN_TEST])

    splits = split_by_date(table, "period_start")

    assert list(splits) == ["train", "val", "test"]
    assert splits["train"]["row"].tolist() == [0, 1]
    assert splits["val"]["row"].tolist() == [2]
    assert splits["test"]["row"].tolist() == [3]
    assert splits["train"]["period_start"].max() < splits["val"]["period_start"].min()
    assert splits["val"]["period_start"].max() < splits["test"]["period_start"].min()


@pytest.mark.parametrize(
    ("date", "expected_split"),
    [
        pytest.param(pd.Timestamp(TRAIN_START), "train", id="train_start_is_train"),
        pytest.param(
            pd.Timestamp(TRAIN_START) - pd.Timedelta(hours=1), None, id="before_train_dropped"
        ),
        pytest.param(
            pd.Timestamp(VALIDATION_START) - pd.Timedelta(hours=1),
            "train",
            id="hour_before_validation_is_train",
        ),
        pytest.param(pd.Timestamp(VALIDATION_START), "val", id="validation_start_is_val"),
        pytest.param(
            pd.Timestamp(TEST_START) - pd.Timedelta(hours=1), "val", id="hour_before_test_is_val"
        ),
        pytest.param(pd.Timestamp(TEST_START), "test", id="test_start_is_test"),
        pytest.param(pd.Timestamp(DATA_END), None, id="data_end_dropped"),
    ],
)
def test_split_by_date_at_the_cut_dates(date: pd.Timestamp, expected_split: str | None) -> None:
    """Caso límite: cada corte incluye su fecha de inicio y excluye la de fin; fuera, se descarta."""
    table = _dated_table([IN_TRAIN, IN_VAL, IN_TEST, date])

    splits = split_by_date(table, "period_start")

    containing = [name for name, split in splits.items() if 3 in split["row"].tolist()]
    assert containing == ([expected_split] if expected_split else [])


@pytest.mark.parametrize("missing", ["train", "val", "test"])
def test_split_by_date_fails_when_a_period_is_empty(missing: str) -> None:
    """Caso límite: si un conjunto queda vacío, avisa con su nombre en vez de seguir sin datos."""
    dates = {"train": IN_TRAIN, "val": IN_VAL, "test": IN_TEST}
    table = _dated_table([date for name, date in dates.items() if name != missing])

    with pytest.raises(ValueError, match=f"El conjunto '{missing}'"):
        split_by_date(table, "period_start")


# -----------------------------------------------------------------------------
# validate_contract
# -----------------------------------------------------------------------------
def _contract_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Viajes y estaciones que cumplen el contrato de ``config.py``, con los nombres de la ingesta.

    Returns:
        ``trips`` con las columnas de ``TRIP_COLUMNS`` y ``stations`` con las de
        ``STATION_COLUMNS``, con los tipos que exige el contrato.
    """
    start = pd.Timestamp("2025-03-03 08:00")
    trips = pd.DataFrame(
        {
            "viaje_id": pd.Series([1, 2], dtype="int64"),
            "inicio_del_viaje": [start, start + pd.Timedelta(hours=1)],
            "fin_del_viaje": [start + pd.Timedelta(minutes=15), start + pd.Timedelta(minutes=75)],
            "origen_id": pd.Series([1, 2], dtype="int64"),
            "destino_id": pd.Series([2, 1], dtype="int64"),
        }
    )
    stations = pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="int64"),
            "name": ["GDL-001", "GDL-002"],
            "latitude": [20.66, 20.67],
            "longitude": [-103.38, -103.37],
            "dpcapacity": pd.Series([10, 12], dtype="int64"),
        }
    )
    return trips, stations


def test_validate_contract_accepts_the_contract_types() -> None:
    """Caso normal: con los tipos del contrato no lanza ningún error."""
    trips, stations = _contract_data()

    assert validate_contract(trips, stations) is None


@pytest.mark.parametrize(
    ("table", "column", "bad_values", "expected_message"),
    [
        pytest.param(
            "trips",
            "inicio_del_viaje",
            ["2025-03-03 08:00", "2025-03-03 09:00"],
            "viajes.inicio_del_viaje debe ser datetime",
            id="dates_as_text",
        ),
        pytest.param(
            "trips", "origen_id", [1.0, 2.0], "viajes.origen_id debe ser integer", id="id_as_float"
        ),
        pytest.param(
            "stations",
            "dpcapacity",
            ["10", "12"],
            "estaciones.dpcapacity debe ser integer",
            id="capacity_as_text",
        ),
        pytest.param(
            "stations",
            "latitude",
            ["20.66", "20.67"],
            "estaciones.latitude debe ser float",
            id="latitude_as_text",
        ),
    ],
)
def test_validate_contract_rejects_a_wrong_type(
    table: str, column: str, bad_values: list, expected_message: str
) -> None:
    """Caso límite: una sola columna con otro tipo basta para rechazar, y el mensaje la nombra."""
    data = dict(zip(("trips", "stations"), _contract_data()))
    data[table][column] = bad_values

    with pytest.raises(TypeError, match=expected_message):
        validate_contract(data["trips"], data["stations"])


def test_validate_contract_reports_every_failing_column() -> None:
    """Caso límite: si fallan varias columnas, el mensaje las lista todas, no solo la primera."""
    trips, stations = _contract_data()
    trips["inicio_del_viaje"] = trips["inicio_del_viaje"].astype(str)
    stations["id"] = stations["id"].astype(float)

    with pytest.raises(TypeError) as error:
        validate_contract(trips, stations)

    assert "viajes.inicio_del_viaje debe ser datetime" in str(error.value)
    assert "estaciones.id debe ser integer" in str(error.value)


# -----------------------------------------------------------------------------
# get_data_version
# -----------------------------------------------------------------------------
def test_get_data_version_reads_the_versioned_descriptor() -> None:
    """Caso normal: devuelve la versión declarada en el ``datapackage.json`` del repositorio."""
    declared = json.loads(Path(DATAPACKAGE_PATH).read_text(encoding="utf-8"))["version"]

    version = get_data_version()

    assert version == str(declared)
    assert version  # nunca vacía: identifica los datos con que se entrena


def test_get_data_version_fails_without_version(tmp_path: Path) -> None:
    """Caso límite: un descriptor sin ``version`` no sirve para identificar los datos."""
    descriptor = tmp_path / "datapackage.json"
    descriptor.write_text(json.dumps({"name": "mibici"}), encoding="utf-8")

    with pytest.raises(ValueError, match="no declara `version`"):
        get_data_version(descriptor)


def test_get_data_version_fails_without_descriptor(tmp_path: Path) -> None:
    """Caso límite: sin descriptor, pide generarlo con ``make snapshot``."""
    with pytest.raises(FileNotFoundError, match="make snapshot"):
        get_data_version(tmp_path / "datapackage.json")


# -----------------------------------------------------------------------------
# add_calendar_features
# -----------------------------------------------------------------------------
def test_add_calendar_features_derives_hour_weekday_and_weekend() -> None:
    """Caso normal: lunes 08:00 es hora 8, día 0 y entre semana; sábado 14:00 es 14, 5 y fin de semana."""
    table = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2025-03-03 08:00", "2025-03-08 14:00"]),
            "departures": [3, 1],
        }
    )

    result = add_calendar_features(table)

    assert result["hour_of_day"].tolist() == [8, 14]
    assert result["day_of_week"].tolist() == [0, 5]
    assert result["is_weekend"].tolist() == [0, 1]
    assert result["departures"].tolist() == [3, 1]  # conserva las columnas originales
    for column in ["hour_of_day", "day_of_week", "is_weekend"]:
        assert result[column].dtype == "int8"
    assert "hour_of_day" not in table.columns  # no modifica la tabla de entrada


@pytest.mark.parametrize(
    ("date", "day_of_week", "is_weekend"),
    [
        pytest.param("2025-03-07 23:00", 4, 0, id="friday_night_is_weekday"),
        pytest.param("2025-03-08 00:00", 5, 1, id="saturday_midnight_is_weekend"),
        pytest.param("2025-03-09 23:00", 6, 1, id="sunday_night_is_weekend"),
        pytest.param("2025-03-10 00:00", 0, 0, id="monday_midnight_is_weekday"),
    ],
)
def test_add_calendar_features_at_the_weekend_boundaries(
    date: str, day_of_week: int, is_weekend: int
) -> None:
    """Caso límite: el fin de semana empieza el sábado a las 0:00 y termina el domingo a las 23:59."""
    table = pd.DataFrame({"start_time": pd.to_datetime([date])})

    result = add_calendar_features(table, time_column="start_time")

    assert result.loc[0, "hour_of_day"] == pd.Timestamp(date).hour
    assert result.loc[0, "day_of_week"] == day_of_week
    assert result.loc[0, "is_weekend"] == is_weekend
