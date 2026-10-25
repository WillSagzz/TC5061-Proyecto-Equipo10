"""Funciones de feature engineering para P1, P2 y P3.

Implementa las Etapas 2 y 3 del pipeline de la propuesta (entrega 01, secciones 4 y 5) con
el procedimiento del notebook de exploración (``notebooks/exploration.ipynb``): contar las
salidas y las llegadas de cada estación en cada periodo, completar la malla estación × periodo
con ceros y, a partir de ahí, construir la variable objetivo y las variables predictoras de
cada pregunta.

Todas las funciones parten de los dos Parquet limpios que entrega la ingesta (``dataset.py``),
con el contrato de datos definido en ``config.py``. Ninguna usa datos personales.

Funciones compartidas por las tres preguntas:

- ``load_data``: carga los viajes y las estaciones y valida el contrato.
- ``get_data_version``: versión de los datos (``datapackage.json``), para registrarla en MLflow.
- ``split_by_date``: partición temporal en entrenamiento, validación y prueba.
- ``build_station_period_table``: salidas, llegadas y flujo neto por estación y periodo,
  con la ventana en que cada estación estuvo activa (P1 y P2).
- ``add_calendar_features``: hora del día, día de la semana y fin de semana (P1 y P2).
- ``add_lag_features``: rezagos y promedio móvil calculados solo con periodos cerrados antes
  del horizonte de la predicción (P1 y P2).

Funciones de cada pregunta: ``build_p1_table``, ``build_p2_table`` y ``build_p3_table``. Los
modelos de P1 y P2 solo reciben las columnas de ``P1_FEATURES`` y ``P2_FEATURES``.
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset

from mibici.config import (
    DATA_END,
    DATAPACKAGE_PATH,
    DAY_PERIODS,
    FLOW_LAG_HOURS,
    LAG_HOURS,
    NO_SERVICE_HOURS,
    P1_FLOW_LAG_HOURS,
    P1_HORIZON_HOURS,
    P1_LAG_HOURS,
    P2_INTERVAL_HOURS,
    RISK_THRESHOLD,
    ROLLING_WINDOW,
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

# Variables de entrada de P1: solo lo que se conoce al cierre del día anterior. Quedan fuera
# la variable objetivo y los movimientos de la propia hora (``departures``, ``arrivals`` y
# ``net_flow``): las salidas son exactamente llegadas − flujo neto, así que cualquiera de las
# dos revelaría la respuesta. ``station_id`` y ``hour_of_day`` son los grupos del baseline.
P1_TARGET = "departures"
P1_FEATURES = [
    "station_id",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "capacity",
    *[f"departures_lag_{hours}" for hours in P1_LAG_HOURS],
    *[f"arrivals_lag_{hours}" for hours in P1_FLOW_LAG_HOURS],
    *[f"net_flow_lag_{hours}" for hours in P1_FLOW_LAG_HOURS],
    f"departures_mean_{ROLLING_WINDOW}_lag_{P1_HORIZON_HOURS}",
]

# Variables con las que se agrupan las estaciones en P3 (latitud y longitud no entran).
# Son las cinco de la Tabla 4 de la propuesta; la primera ocupa cuatro columnas.
P3_FEATURES = [
    "morning_pct",
    "midday_pct",
    "afternoon_pct",
    "night_pct",
    "weekdays_pct",
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
    texto, como venían en los CSV originales), el error aparece aquí con un mensaje claro y
    no como un resultado incorrecto.

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

    Atiende P1, P2 y P3: es el único punto de entrada de los datos a los modelos. Los viajes
    ya vienen limpios de ``dataset.py`` (fechas convertidas, sin horas sin servicio y solo
    con estaciones de la nomenclatura), así que aquí no se repite esa limpieza.

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


def get_data_version(datapackage_path: Path = DATAPACKAGE_PATH) -> str:
    """Devuelve la versión de los datos, para registrarla como parámetro en MLflow.

    Es el campo ``version`` del descriptor ``datapackage.json`` que escribe ``make snapshot``
    (por ejemplo ``2026-10``). ``make data`` solo produce Parquet que coinciden con ese
    descriptor, así que la versión identifica exactamente los datos con que se entrena.

    Args:
        datapackage_path: Descriptor Frictionless Data Package versionado en Git.

    Returns:
        Versión declarada en el descriptor.

    Raises:
        FileNotFoundError: Si el descriptor no existe.
        ValueError: Si el descriptor no declara una versión.
    """
    path = Path(datapackage_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Genera el descriptor con `make snapshot`.")
    version = json.loads(path.read_text(encoding="utf-8")).get("version")
    if not version:
        raise ValueError(
            f"{path.name} no declara `version`. Regenera el descriptor con `make snapshot`."
        )
    return str(version)


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
# Etapa 2: tabla base por estación y periodo (P1 y P2)
# -----------------------------------------------------------------------------
def build_station_period_table(
    trips: pd.DataFrame, stations: pd.DataFrame, freq: str = "h"
) -> pd.DataFrame:
    """Cuenta las salidas y las llegadas de cada estación en cada periodo y su flujo neto.

    Atiende P1 y P2 (Etapa 2 del pipeline). Sigue el procedimiento del notebook de
    exploración: cada viaje se asigna dos veces, como salida de su estación de origen en el
    periodo en que inicia y como llegada a su estación de destino en el periodo en que
    termina. Las horas se redondean hacia abajo (08:47 pasa a 08:00). Se genera la malla
    completa de estaciones activas por periodos, de las 0:00 del primer día a las 24:00 del
    último, y las combinaciones sin viajes quedan en 0, para que las series sean continuas y
    sin huecos. P1 usa ``freq="h"``; P2 usa intervalos de ``P2_INTERVAL_HOURS`` horas.

    La malla abarca las estaciones con viajes en cualquier momento de la ventana, pero 110
    abrieron durante ella (87 en el periodo de prueba): antes de abrir, sus ceros no son
    demanda y, si se quedaran, el entrenamiento incluiría estaciones que solo existen en el
    futuro. Por eso la columna ``active`` marca la ventana de cada estación, del día de su
    primer movimiento al día siguiente del último; ``build_p1_table`` y ``build_p2_table``
    descartan las filas fuera de ella después de calcular los rezagos.

    Args:
        trips: Viajes de ``load_data``.
        stations: Estaciones de ``load_data``; solo se cuentan las de la nomenclatura.
        freq: Duración del periodo, en formato de pandas.

    Returns:
        Una fila por estación y periodo, ordenada por estación y periodo, con ``station_id``,
        ``period_start``, ``departures``, ``arrivals``, ``net_flow`` (llegadas − salidas) y
        ``active`` (``True`` si la estación ya había abierto y no había cerrado).
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
    table["net_flow"] = (table["arrivals"] - table["departures"]).astype("int32")

    # Ventana activa: del día del primer movimiento de la estación al día siguiente del último
    movements = pd.concat([departures, arrivals]).index.to_frame(index=False)
    days = movements.groupby("station_id")["period_start"].agg(["min", "max"])
    opens = table["station_id"].map(days["min"].dt.floor("D"))
    closes = table["station_id"].map(days["max"].dt.floor("D") + pd.Timedelta(days=1))
    table["active"] = (table["period_start"] >= opens) & (table["period_start"] < closes)
    logger.info(
        f"Tabla estación × periodo ({freq}): {len(table):,} filas, {len(active_stations)} "
        f"estaciones, {table['active'].mean():.1%} de las filas con la estación activa"
    )
    return table


# -----------------------------------------------------------------------------
# Etapa 3: variables de calendario e históricas (P1 y P2)
# -----------------------------------------------------------------------------
def add_calendar_features(table: pd.DataFrame, time_column: str = "period_start") -> pd.DataFrame:
    """Agrega la hora del día, el día de la semana y el indicador de fin de semana.

    Atiende P1 y P2 (Tabla 5 de la propuesta): la demanda sigue ciclos diarios (horas pico)
    y semanales (días laborales frente a fin de semana). Solo usa la fecha del propio
    periodo, que se conoce de antemano, así que no hay fuga de información del futuro.

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


def add_lag_features(
    table: pd.DataFrame,
    lags: dict[str, list[int]],
    window: int = ROLLING_WINDOW,
    window_column: str = "departures",
    horizon: int = 1,
) -> pd.DataFrame:
    """Agrega rezagos y un promedio móvil calculados solo con periodos cerrados al predecir.

    Atiende P1 y P2 (Etapa 3 del pipeline, Tabla 5 de la propuesta). Para cada columna de
    ``lags`` crea ``<columna>_lag_<n>``: el valor de la misma estación ``n`` periodos antes
    (en P1, horas; en P2, intervalos de 3 horas). Para ``window_column`` crea el promedio de
    los ``window`` periodos que terminan ``horizon`` periodos antes, de
    t − horizon − window + 1 a t − horizon: ``<columna>_mean_<window>`` si ``horizon`` es 1
    y ``<columna>_mean_<window>_lag_<horizon>`` si es mayor. Los rezagos capturan la inercia
    inmediata de la estación y su hábito diario y semanal; el promedio móvil suaviza una
    sola hora atípica y muestra si la demanda se acelera o se frena.

    No hay fuga de información del futuro: ``horizon`` es la distancia, en periodos, entre
    el último periodo cerrado al emitir la predicción y el periodo que se predice (1 en P2,
    que pronostica el intervalo siguiente; 24 en P1, que al cierre de cada día pronostica
    todas las horas del siguiente), y ningún rezago ni ventana puede ser más reciente. En
    P1, un rezago de 1 h usaría para las 18:00 las 17:00 del mismo día que se pronostica.
    Además, los periodos fuera de la ventana activa de la estación (``active``) cuentan como
    desconocidos: los rezagos que caen antes de su apertura y las primeras filas de cada
    estación, sin historial suficiente, quedan como NaN y no como cero.

    Requiere la malla completa y regular de ``build_station_period_table``: así un rezago de
    ``n`` filas dentro de cada estación equivale exactamente a ``n`` periodos.

    Args:
        table: Tabla de ``build_station_period_table`` (puede llevar más columnas).
        lags: Columnas a rezagar y, para cada una, los rezagos en número de periodos.
        window: Número de periodos cerrados del promedio móvil.
        window_column: Columna a la que se aplica el promedio móvil.
        horizon: Periodos entre el último periodo cerrado al predecir y el que se predice;
            es el rezago mínimo permitido.

    Returns:
        La misma tabla, ordenada por estación y periodo, con las columnas nuevas
        (``float32``).

    Raises:
        ValueError: Si algún rezago es menor que ``horizon`` o si alguna estación no tiene
            todos los periodos de la malla.
    """
    too_recent = [
        f"{column}_lag_{step}"
        for column, steps in lags.items()
        for step in steps
        if step < horizon
    ]
    if too_recent:
        raise ValueError(
            f"Rezagos menores que el horizonte de {horizon} periodos: {too_recent}. Usarían "
            "periodos que todavía no ocurren cuando se emite la predicción."
        )
    n_periods = table["period_start"].nunique()
    irregular = table.groupby("station_id").size() != n_periods
    if irregular.any():
        raise ValueError(
            f"La malla estación × periodo no es regular ({irregular.sum()} estaciones con "
            "periodos faltantes). Calcula los rezagos sobre la tabla completa de "
            "build_station_period_table, antes de filtrar filas."
        )
    table = table.sort_values(["station_id", "period_start"]).reset_index(drop=True)
    # Los periodos con la estación inactiva son desconocidos (NaN), no demanda cero
    history = table[list(dict.fromkeys([*lags, window_column]))].astype("float32")
    history.loc[~table["active"]] = np.nan
    grouped = history.groupby(table["station_id"], sort=False)
    new_columns = {}
    for column, steps in lags.items():
        for step in steps:
            new_columns[f"{column}_lag_{step}"] = grouped[column].shift(step).astype("float32")
    suffix = f"_lag_{horizon}" if horizon > 1 else ""
    new_columns[f"{window_column}_mean_{window}{suffix}"] = (
        grouped[window_column]
        .transform(lambda values: values.shift(horizon).rolling(window).mean())
        .astype("float32")
    )
    logger.info(f"Variables históricas: {', '.join(new_columns)}")
    return table.assign(**new_columns)


# -----------------------------------------------------------------------------
# P1: viajes iniciados por estación y hora
# -----------------------------------------------------------------------------
def build_p1_table(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de P1: viajes iniciados por estación y hora, con sus predictores.

    Atiende P1 (Tablas 4 y 5 de la propuesta): la variable objetivo ``departures`` es el
    número de viajes que salen de la estación en cada hora, la demanda que el operador
    necesita anticipar para planear la flota. Al cierre de cada día se pronostican todas las
    horas del siguiente (``P1_HORIZON_HOURS`` = 24), así que las variables históricas solo
    usan horas cerradas al menos 24 h antes: salidas 24 h y 7 días antes
    (``P1_LAG_HOURS``), llegadas y flujo neto 24 h antes (``P1_FLOW_LAG_HOURS``) y el
    promedio de salidas de las ``ROLLING_WINDOW`` horas que terminan a la misma hora del día
    anterior. Lleva además la capacidad de la estación y las variables de calendario.

    La tabla conserva las llegadas y el flujo neto de la propia hora para análisis, pero no
    son variables de entrada: el modelo solo recibe ``P1_FEATURES``.

    Los rezagos se calculan sobre la malla horaria completa y solo después se excluyen las
    horas de ``NO_SERVICE_HOURS`` (de 1:00 a 4:59), en las que el sistema no presta
    bicicletas (ahí la demanda es 0 por regla y no por comportamiento de los usuarios, pero
    siguen contando como historial), y las horas fuera de la ventana activa de cada
    estación, antes de que abriera o después de que cerrara.

    Args:
        trips: Viajes de ``load_data``.
        stations: Estaciones de ``load_data``.

    Returns:
        Una fila por estación activa y hora de servicio, con ``departures``, ``arrivals``,
        ``net_flow``, ``capacity``, las variables de calendario y las históricas.
    """
    table = build_station_period_table(trips, stations, freq="h")
    table = add_calendar_features(table)
    table = table.merge(stations[["station_id", "capacity"]], on="station_id", how="left")
    table = add_lag_features(
        table,
        {
            "departures": P1_LAG_HOURS,
            "arrivals": P1_FLOW_LAG_HOURS,
            "net_flow": P1_FLOW_LAG_HOURS,
        },
        window=ROLLING_WINDOW,
        horizon=P1_HORIZON_HOURS,
    )
    in_service = ~table["period_start"].dt.hour.isin(NO_SERVICE_HOURS)
    table = table.loc[table["active"] & in_service].drop(columns="active")
    logger.info(f"Tabla P1: {len(table):,} filas de estación activa × hora de servicio")
    return table.reset_index(drop=True)


# -----------------------------------------------------------------------------
# P2: riesgo de vaciarse o llenarse por estación e intervalo de 3 horas
# -----------------------------------------------------------------------------
def label_risk(net_flow: pd.Series, capacity: pd.Series) -> pd.Series:
    """Etiqueta el riesgo de cada estación e intervalo a partir de su flujo neto.

    Atiende P2 (Tabla 4 de la propuesta): sin inventario en tiempo real, el riesgo se
    aproxima con el flujo neto (llegadas − salidas) comparado con la capacidad de la
    estación. Si el flujo es menor o igual a −``RISK_THRESHOLD`` × capacidad, la estación
    tiende a vaciarse; si es mayor o igual a +``RISK_THRESHOLD`` × capacidad, a llenarse; en
    otro caso, no hay riesgo. El umbral sí se aplica: un flujo distinto de cero pero menor al
    40 % de la capacidad es ``"sin_riesgo"``.

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


def _hours_to_intervals(hours: int) -> int:
    """Convierte un rezago en horas a intervalos de P2; nunca menos de un intervalo."""
    return max(1, hours // P2_INTERVAL_HOURS)


# Rezagos de P2 en intervalos: salidas 1, 8 y 56 intervalos antes; llegadas y flujo neto 1 y 8
P2_DEPARTURE_LAGS = [_hours_to_intervals(hours) for hours in LAG_HOURS]
P2_FLOW_LAGS = [_hours_to_intervals(hours) for hours in FLOW_LAG_HOURS]

# Variables de entrada de P2: solo lo que se conoce al cierre del intervalo anterior. Quedan
# fuera la variable objetivo y los movimientos del propio intervalo (``risk``, ``departures``,
# ``arrivals`` y ``net_flow``): la clase de riesgo se calcula con el flujo neto del intervalo.
# ``station_id``, ``interval_of_day`` e ``is_weekend`` son los grupos del baseline.
P2_TARGET = "risk"
P2_FEATURES = [
    "station_id",
    "interval_of_day",
    "day_of_week",
    "is_weekend",
    "capacity",
    *[f"departures_lag_{step}" for step in P2_DEPARTURE_LAGS],
    *[f"arrivals_lag_{step}" for step in P2_FLOW_LAGS],
    *[f"net_flow_lag_{step}" for step in P2_FLOW_LAGS],
    f"departures_mean_{ROLLING_WINDOW}",
]


def build_p2_table(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de P2: salidas, llegadas y riesgo por estación e intervalo de 3 h.

    Atiende P2 (Tablas 4 y 5 de la propuesta): el día se divide en 8 intervalos de
    ``P2_INTERVAL_HOURS`` horas (``interval_of_day`` de 0 a 7). Una salida cuenta en el
    intervalo de inicio del viaje y una llegada en el intervalo de fin. Se excluyen las
    estaciones con capacidad 0, porque el umbral de riesgo depende de la capacidad. La
    variable objetivo ``risk`` solo usa los movimientos del propio intervalo.

    Las variables históricas son las de P1 expresadas en intervalos: salidas 1, 8 (24 h) y
    56 (7 días) intervalos antes, llegadas y flujo neto 1 y 8 intervalos antes, y el promedio
    móvil de salidas de los ``ROLLING_WINDOW`` intervalos cerrados previos. Todas usan solo
    intervalos anteriores al que se predice, como exige el horizonte de P2 (el intervalo
    siguiente, con los viajes ocurridos hasta el cierre del actual). El modelo solo recibe
    ``P2_FEATURES``. Después de calcular los rezagos se excluyen los intervalos fuera de la
    ventana activa de cada estación, cuyos ceros inflarían la clase ``"sin_riesgo"``.

    Args:
        trips: Viajes de ``load_data``.
        stations: Estaciones de ``load_data``.

    Returns:
        Una fila por estación activa e intervalo, con ``departures``, ``arrivals``,
        ``net_flow``, ``capacity``, ``risk``, las variables de calendario y las históricas.
    """
    with_capacity = stations.loc[stations["capacity"] > 0, ["station_id", "capacity"]]
    table = build_station_period_table(trips, with_capacity, freq=f"{P2_INTERVAL_HOURS}h")
    table = add_calendar_features(table).merge(with_capacity, on="station_id")
    table["interval_of_day"] = (table["hour_of_day"] // P2_INTERVAL_HOURS).astype("int8")
    table["risk"] = label_risk(table["net_flow"], table["capacity"])
    table = add_lag_features(
        table,
        {"departures": P2_DEPARTURE_LAGS, "arrivals": P2_FLOW_LAGS, "net_flow": P2_FLOW_LAGS},
        window=ROLLING_WINDOW,
    )
    table = table.loc[table["active"]].drop(columns="active").reset_index(drop=True)
    shares = table["risk"].value_counts(normalize=True).round(4).to_dict()
    logger.info(f"Tabla P2: {len(table):,} filas; clases: {shares}")
    return table


# -----------------------------------------------------------------------------
# P3: perfil de uso de cada estación
# -----------------------------------------------------------------------------
def _day_period(hours: pd.Series) -> pd.Series:
    """Asigna cada hora a su periodo del día según ``DAY_PERIODS``.

    La noche cruza la medianoche (20:00 a 4:59), por eso ese periodo se evalúa como
    ``hora >= inicio`` o ``hora < fin`` y no como un rango continuo.
    """
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

    Atiende P3 (Tablas 4 y 6 de la propuesta): describe cómo se usa cada estación para
    agruparlas por patrón de uso y no por volumen. Un movimiento es cada salida o llegada,
    así que cada viaje aporta dos. Solo usa los movimientos del periodo de entrenamiento
    (``TRAIN_START`` a ``VALIDATION_START``) y excluye las estaciones con capacidad 0.
    Variables (``P3_FEATURES``):

    - ``<periodo>_pct``: porcentaje (0 a 100) de los movimientos en cada periodo de
      ``DAY_PERIODS``; las cuatro columnas suman 100.
    - ``weekdays_pct``: porcentaje de los movimientos de lunes a viernes.
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
        table[f"{period}_pct"] = counts[period].sum(axis=1) / total * 100
    is_weekday = movements["time"].dt.dayofweek < 5
    table["weekdays_pct"] = is_weekday.groupby(movements["station_id"]).mean() * 100

    capacity = with_capacity.set_index("station_id")["capacity"].reindex(table.index)
    n_days = (pd.Timestamp(VALIDATION_START) - pd.Timestamp(TRAIN_START)).days
    table["log_daily_use_per_dock"] = np.log1p(total / n_days / capacity)
    for period in ("morning", "afternoon"):
        table[f"balance_{period}"] = _balance(counts[(period, True)], counts[(period, False)])

    location = with_capacity.set_index("station_id")[["capacity", "latitude", "longitude"]]
    table = table.join(location).reset_index()
    logger.info(f"Tabla P3: {len(table):,} estaciones con movimientos en entrenamiento")
    return table
