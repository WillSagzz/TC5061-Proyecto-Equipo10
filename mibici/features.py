"""Funciones de feature engineering para P1, P2 y P3.

Todas parten de los dos Parquet limpios que entrega la ingesta (``dataset.py``), con el
contrato de datos definido en ``config.py``. Ninguna usa datos personales.

Funciones compartidas por las tres preguntas:

- ``load_data``: carga los viajes y las estaciones y valida el contrato.
- ``get_data_version``: versión de los datos, para registrarla en MLflow.
- ``split_by_date``: partición temporal en entrenamiento, validación y prueba.
- ``build_station_period_table``: salidas y llegadas por estación y periodo (P1 y P2).
- ``add_calendar_features``: hora del día, día de la semana y fin de semana (P1 y P2).

Funciones de cada pregunta: ``build_p1_table``, ``build_p2_table`` y ``build_p3_table``.
"""

import hashlib
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from mibici.config import (
    DATA_END,
    DATA_MANIFEST_PATH,
    DAY_PERIODS,
    NO_SERVICE_HOURS,
    P2_INTERVAL_HOURS,
    RISK_THRESHOLD,
    STATION_COLUMNS,
    STATIONS_PATH,
    TEST_START,
    TRAIN_START,
    TRIP_COLUMNS,
    TRIPS_PATH,
    VALIDATION_START,
)

# Nombres de dataset.py -> nombres que usan los modelos
RENAME_COLUMNS = {
    "viaje_id": "trip_id",
    "inicio_del_viaje": "start_time",
    "fin_del_viaje": "end_time",
    "origen_id": "origin_station_id",
    "destino_id": "destination_station_id",
    "id": "station_id",
    "dpcapacity": "capacity",
}

# Clases de P2 (valores de la variable objetivo ``risk``)
RISK_EMPTY = "vaciarse"
RISK_FULL = "llenarse"
NO_RISK = "sin_riesgo"
RISK_CLASSES = [RISK_EMPTY, RISK_FULL, NO_RISK]

# Variables con las que se agrupan las estaciones en P3 (latitud y longitud no entran)
P3_FEATURES = [
    "share_morning",
    "share_midday",
    "share_afternoon",
    "share_night",
    "share_weekdays",
    "log_daily_use_per_dock",
    "balance_morning",
    "balance_afternoon",
]

_TYPE_CHECKS = {
    "integer": pd.api.types.is_integer_dtype,
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "float": pd.api.types.is_numeric_dtype,
    "text": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
}


# -----------------------------------------------------------------------------
# Carga, contrato y versión de los datos
# -----------------------------------------------------------------------------
def _read_parquet(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    """Lee solo las columnas del contrato y avisa con claridad si falta el archivo o alguna columna."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero la ingesta: `python -m mibici.dataset`."
        )
    available = pa_dataset.dataset(path).schema.names
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(
            f"{Path(path).name} no cumple el contrato de datos. Faltan columnas: {missing}."
        )
    return pd.read_parquet(path, columns=list(columns))


def validate_contract(trips: pd.DataFrame, stations: pd.DataFrame) -> None:
    """Verifica que los tipos de datos coincidan con el contrato de ``config.py``.

    Atiende P1, P2 y P3: si la ingesta cambia un tipo (por ejemplo, fechas guardadas como
    texto), el error aparece aquí con un mensaje claro y no como un resultado incorrecto.

    Args:
        trips: Viajes con las columnas de ``TRIP_COLUMNS``.
        stations: Estaciones con las columnas de ``STATION_COLUMNS``.

    Raises:
        TypeError: Si alguna columna no tiene el tipo esperado.
    """
    problems = []
    for name, table, contract in [
        ("viajes", trips, TRIP_COLUMNS),
        ("estaciones", stations, STATION_COLUMNS),
    ]:
        for column, expected_type in contract.items():
            if not _TYPE_CHECKS[expected_type](table[column]):
                problems.append(
                    f"{name}.{column} debe ser {expected_type}, es {table[column].dtype}"
                )
    if problems:
        raise TypeError("El contrato de datos no se cumple:\n  " + "\n  ".join(problems))


def load_data(
    trips_path: Path = TRIPS_PATH, stations_path: Path = STATIONS_PATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los viajes y las estaciones que entrega la ingesta y valida el contrato.

    Atiende P1, P2 y P3: es el único punto de entrada de los datos a los modelos.

    Args:
        trips_path: Parquet de viajes generado por ``dataset.py``.
        stations_path: Parquet de la nomenclatura generado por ``dataset.py``.

    Returns:
        ``trips`` (trip_id, start_time, end_time, origin_station_id,
        destination_station_id) y ``stations`` (station_id, name, latitude, longitude,
        capacity).

    Raises:
        FileNotFoundError: Si no se ha corrido la ingesta.
        ValueError: Si falta alguna columna del contrato.
        TypeError: Si alguna columna no tiene el tipo esperado.
    """
    trips = _read_parquet(trips_path, TRIP_COLUMNS)
    stations = _read_parquet(stations_path, STATION_COLUMNS)
    validate_contract(trips, stations)
    logger.info(f"Datos cargados: {len(trips):,} viajes y {len(stations):,} estaciones")
    return trips.rename(columns=RENAME_COLUMNS), stations.rename(columns=RENAME_COLUMNS)


def get_data_version(
    manifest_path: Path = DATA_MANIFEST_PATH,
    trips_path: Path = TRIPS_PATH,
    stations_path: Path = STATIONS_PATH,
) -> str:
    """Devuelve la versión de los datos, para registrarla como parámetro en MLflow.

    Usa el campo ``version`` del manifiesto que genera ``dataset.py``. Si no existe, calcula
    el hash SHA-256 de los Parquet: el mismo contenido da siempre la misma versión.

    Args:
        manifest_path: Manifiesto de la ingesta.
        trips_path: Parquet de viajes.
        stations_path: Parquet de estaciones.

    Returns:
        Identificador de 12 caracteres.
    """
    if Path(manifest_path).exists():
        version = json.loads(Path(manifest_path).read_text()).get("version")
        if version:
            return str(version)
    digest = hashlib.sha256()
    for path in (trips_path, stations_path):
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()[:12]


# -----------------------------------------------------------------------------
# Partición temporal
# -----------------------------------------------------------------------------
def split_by_date(table: pd.DataFrame, time_column: str) -> dict[str, pd.DataFrame]:
    """Separa una tabla en entrenamiento, validación y prueba según las fechas de ``config.py``.

    Atiende P1 y P2 (y P3, que solo usa el entrenamiento): la partición es por fechas,
    nunca aleatoria, para que el modelo siempre se evalúe con datos posteriores a los que
    vio al entrenar. Entrenamiento: ``TRAIN_START`` a ``VALIDATION_START``; validación:
    ``VALIDATION_START`` a ``TEST_START``; prueba: ``TEST_START`` a ``DATA_END``.

    Args:
        table: Tabla con una columna de fechas.
        time_column: Nombre de la columna de fechas.

    Returns:
        Diccionario con las llaves ``train``, ``val`` y ``test``.

    Raises:
        ValueError: Si algún conjunto queda vacío (por ejemplo, si falta correr la
            ingesta completa).
    """
    cuts = {
        "train": (TRAIN_START, VALIDATION_START),
        "val": (VALIDATION_START, TEST_START),
        "test": (TEST_START, DATA_END),
    }
    times = table[time_column]
    splits = {}
    for name, (start, end) in cuts.items():
        splits[name] = table.loc[(times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))]
        if splits[name].empty:
            raise ValueError(
                f"El conjunto '{name}' ({start} a {end}) quedó vacío. Los datos van de "
                f"{times.min()} a {times.max()}: corre la ingesta completa."
            )
        logger.info(f"Conjunto {name}: {start} a {end}, {len(splits[name]):,} filas")
    return splits


# -----------------------------------------------------------------------------
# Tablas por estación y periodo (P1 y P2)
# -----------------------------------------------------------------------------
def build_station_period_table(
    trips: pd.DataFrame, stations: pd.DataFrame, freq: str = "h"
) -> pd.DataFrame:
    """Cuenta las salidas y llegadas de cada estación en cada periodo.

    Atiende P1 y P2 (Etapa 2 del pipeline). Una salida se asigna a la estación de origen y
    al periodo de inicio del viaje; una llegada, a la estación de destino y al periodo de
    fin. Las horas se redondean hacia abajo (08:47 pasa a 08:00). Se genera la malla
    completa de estaciones por periodos y las combinaciones sin viajes quedan en 0, para
    que las series sean continuas. P1 usa ``freq="h"``; P2 puede usar ``freq="3h"``.

    Args:
        trips: Viajes de ``load_data``.
        stations: Estaciones de ``load_data``; solo se cuentan las de la nomenclatura.
        freq: Duración del periodo, en formato de pandas.

    Returns:
        Una fila por estación y periodo, con ``station_id``, ``period_start``,
        ``departures`` y ``arrivals``.
    """
    known = stations["station_id"]
    departures = (
        trips.loc[trips["origin_station_id"].isin(known)]
        .groupby(["origin_station_id", trips["start_time"].dt.floor(freq)])
        .size()
        .rename_axis(["station_id", "period_start"])
    )
    arrivals = (
        trips.loc[trips["destination_station_id"].isin(known)]
        .groupby(["destination_station_id", trips["end_time"].dt.floor(freq)])
        .size()
        .rename_axis(["station_id", "period_start"])
    )

    active_stations = np.union1d(
        departures.index.get_level_values("station_id").unique(),
        arrivals.index.get_level_values("station_id").unique(),
    )
    first_day = trips["start_time"].min().normalize()
    last_day = trips["start_time"].max().normalize()
    periods = pd.date_range(
        first_day, last_day + pd.Timedelta(days=1), freq=freq, inclusive="left"
    )
    grid = pd.MultiIndex.from_product(
        [active_stations, periods], names=["station_id", "period_start"]
    )

    table = pd.DataFrame(
        {
            "departures": departures.reindex(grid, fill_value=0).astype("int32"),
            "arrivals": arrivals.reindex(grid, fill_value=0).astype("int32"),
        }
    ).reset_index()
    logger.info(
        f"Tabla estación × periodo ({freq}): {len(table):,} filas, {len(active_stations)} estaciones"
    )
    return table


def add_calendar_features(table: pd.DataFrame, time_column: str = "period_start") -> pd.DataFrame:
    """Agrega la hora del día, el día de la semana y el indicador de fin de semana.

    Atiende P1 y P2: la demanda sigue ciclos diarios (horas pico) y semanales (días
    laborales frente a fin de semana). Solo usa la fecha del propio periodo, que se conoce
    de antemano, así que no hay fuga de información del futuro.

    Args:
        table: Tabla con una columna de fechas.
        time_column: Nombre de la columna de fechas.

    Returns:
        La misma tabla con ``hour_of_day`` (0 a 23), ``day_of_week`` (0 = lunes) e
        ``is_weekend`` (1 = sábado o domingo).
    """
    times = table[time_column]
    return table.assign(
        hour_of_day=times.dt.hour.astype("int8"),
        day_of_week=times.dt.dayofweek.astype("int8"),
        is_weekend=(times.dt.dayofweek >= 5).astype("int8"),
    )


def build_p1_table(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de P1: viajes iniciados por estación y hora.

    Atiende P1: la variable objetivo ``departures`` es el número de viajes que salen de la
    estación en cada hora, la demanda que el operador necesita anticipar para planear la
    flota. Se excluyen las horas de ``NO_SERVICE_HOURS`` (de 1:00 a 4:59), en las que el
    sistema no permite retirar bicicletas, porque ahí la demanda es 0 por regla y no por
    comportamiento de los usuarios.

    Args:
        trips: Viajes de ``load_data``.
        stations: Estaciones de ``load_data``.

    Returns:
        Una fila por estación y hora, con ``departures`` y las variables de calendario.
    """
    table = build_station_period_table(trips, stations, freq="h")
    table = table.loc[~table["period_start"].dt.hour.isin(NO_SERVICE_HOURS)]
    return add_calendar_features(table.reset_index(drop=True))


# -----------------------------------------------------------------------------
# P2: riesgo de vaciarse o llenarse por estación e intervalo de 3 horas
# -----------------------------------------------------------------------------
def label_risk(net_flow: pd.Series, capacity: pd.Series) -> pd.Series:
    """Etiqueta el riesgo de cada estación e intervalo a partir de su flujo neto.

    Atiende P2: sin inventario en tiempo real, el riesgo se aproxima con el flujo neto
    (llegadas − salidas) comparado con la capacidad de la estación. Si el flujo es menor o
    igual a −``RISK_THRESHOLD`` × capacidad, la estación tiende a vaciarse; si es mayor o
    igual a +``RISK_THRESHOLD`` × capacidad, a llenarse; en otro caso, no hay riesgo.

    Args:
        net_flow: Llegadas menos salidas en el intervalo.
        capacity: Número de anclajes de la estación (mayor a 0).

    Returns:
        ``"vaciarse"``, ``"llenarse"`` o ``"sin_riesgo"`` por fila.
    """
    threshold = RISK_THRESHOLD * capacity
    risk = np.select(
        [net_flow <= -threshold, net_flow >= threshold], [RISK_EMPTY, RISK_FULL], NO_RISK
    )
    return pd.Series(risk, index=net_flow.index, name="risk")


def build_p2_table(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de P2: salidas, llegadas y riesgo por estación e intervalo de 3 h.

    Atiende P2: el día se divide en 8 intervalos de ``P2_INTERVAL_HOURS`` horas
    (``interval_of_day`` de 0 a 7). Una salida cuenta en el intervalo de inicio del viaje y
    una llegada en el intervalo de fin. Se excluyen las estaciones con capacidad 0, porque
    el umbral de riesgo depende de la capacidad. La variable objetivo ``risk`` solo usa los
    movimientos del propio intervalo; las variables de los candidatos deberán usar solo
    intervalos anteriores.

    Args:
        trips: Viajes de ``load_data``.
        stations: Estaciones de ``load_data``.

    Returns:
        Una fila por estación e intervalo, con ``departures``, ``arrivals``, ``net_flow``,
        ``capacity``, ``risk`` y las variables de calendario.
    """
    with_capacity = stations.loc[stations["capacity"] > 0, ["station_id", "capacity"]]
    table = build_station_period_table(trips, with_capacity, freq=f"{P2_INTERVAL_HOURS}h")
    table = add_calendar_features(table).merge(with_capacity, on="station_id")
    table["interval_of_day"] = (table["hour_of_day"] // P2_INTERVAL_HOURS).astype("int8")
    table["net_flow"] = table["arrivals"] - table["departures"]
    table["risk"] = label_risk(table["net_flow"], table["capacity"])
    shares = table["risk"].value_counts(normalize=True).round(4).to_dict()
    logger.info(f"Tabla P2: {len(table):,} filas; clases: {shares}")
    return table


# -----------------------------------------------------------------------------
# P3: perfil de uso de cada estación
# -----------------------------------------------------------------------------
def _day_period(hours: pd.Series) -> pd.Series:
    """Asigna cada hora a su periodo del día según ``DAY_PERIODS``."""
    conditions, names = [], []
    for name, (start, end) in DAY_PERIODS.items():
        in_period = (
            (hours >= start) & (hours < end) if start < end else (hours >= start) | (hours < end)
        )
        conditions.append(in_period)
        names.append(name)
    return pd.Series(np.select(conditions, names, default=""), index=hours.index)


def _train_movements(trips: pd.DataFrame, station_ids: pd.Series) -> pd.DataFrame:
    """Separa cada viaje en dos movimientos y conserva solo los del periodo de entrenamiento.

    Una salida ocurre en la estación de origen a la hora de inicio; una llegada, en la
    estación de destino a la hora de fin.
    """
    movements = pd.concat(
        [
            pd.DataFrame(
                {
                    "station_id": trips["origin_station_id"],
                    "time": trips["start_time"],
                    "is_arrival": False,
                }
            ),
            pd.DataFrame(
                {
                    "station_id": trips["destination_station_id"],
                    "time": trips["end_time"],
                    "is_arrival": True,
                }
            ),
        ],
        ignore_index=True,
    )
    in_train = (movements["time"] >= pd.Timestamp(TRAIN_START)) & (
        movements["time"] < pd.Timestamp(VALIDATION_START)
    )
    return movements.loc[in_train & movements["station_id"].isin(station_ids)]


def _balance(arrivals: pd.Series, departures: pd.Series) -> pd.Series:
    """(llegadas − salidas) ÷ (llegadas + salidas); 0 si la estación no tuvo movimientos."""
    total = arrivals + departures
    return ((arrivals - departures) / total.where(total > 0)).fillna(0.0)


def build_p3_table(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de P3: una fila por estación con su perfil de uso.

    Atiende P3: describe cómo se usa cada estación para agruparlas por patrón de uso y no
    por volumen. Solo usa los movimientos del periodo de entrenamiento (``TRAIN_START`` a
    ``VALIDATION_START``) y excluye las estaciones con capacidad 0. Variables
    (``P3_FEATURES``):

    - ``share_<periodo>``: proporción de movimientos en cada periodo de ``DAY_PERIODS``.
    - ``share_weekdays``: proporción de movimientos de lunes a viernes.
    - ``log_daily_use_per_dock``: log(1 + movimientos por día ÷ capacidad), la presión
      sobre los anclajes; el logaritmo evita que unas pocas estaciones extremas dominen.
    - ``balance_morning`` y ``balance_afternoon``: (llegadas − salidas) ÷ (llegadas +
      salidas) en la mañana y en la tarde; −1 si solo salen bicicletas, +1 si solo llegan.

    La latitud y la longitud se conservan solo para el mapa; no entran al modelo.

    Args:
        trips: Viajes de ``load_data``.
        stations: Estaciones de ``load_data``.

    Returns:
        Una fila por estación con ``P3_FEATURES``, ``capacity``, ``latitude`` y ``longitude``.
    """
    with_capacity = stations.loc[stations["capacity"] > 0]
    movements = _train_movements(trips, with_capacity["station_id"])
    movements = movements.assign(period=_day_period(movements["time"].dt.hour))

    # Movimientos por estación, periodo y tipo (salida o llegada); 0 si no hubo
    all_columns = pd.MultiIndex.from_product([list(DAY_PERIODS), [False, True]])
    counts = pd.crosstab(
        movements["station_id"], [movements["period"], movements["is_arrival"]]
    ).reindex(columns=all_columns, fill_value=0)
    total = movements.groupby("station_id").size()
    table = pd.DataFrame(index=total.index)
    for period in DAY_PERIODS:
        table[f"share_{period}"] = counts[period].sum(axis=1) / total
    is_weekday = movements["time"].dt.dayofweek < 5
    table["share_weekdays"] = is_weekday.groupby(movements["station_id"]).mean()

    capacity = with_capacity.set_index("station_id")["capacity"].reindex(table.index)
    n_days = (pd.Timestamp(VALIDATION_START) - pd.Timestamp(TRAIN_START)).days
    table["log_daily_use_per_dock"] = np.log1p(total / n_days / capacity)
    for period in ("morning", "afternoon"):
        table[f"balance_{period}"] = _balance(counts[(period, True)], counts[(period, False)])

    location = with_capacity.set_index("station_id")[["capacity", "latitude", "longitude"]]
    table = table.join(location).reset_index()
    logger.info(f"Tabla P3: {len(table):,} estaciones con movimientos en entrenamiento")
    return table
