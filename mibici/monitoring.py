"""Preparación de conjuntos de referencia y producción para monitoreo de drift.

Construye los pares de tablas ``(reference, current)`` para las tres preguntas del proyecto
(P1 regresión de demanda, P2 clasificación de riesgo y P3 clustering de perfiles de uso),
comparando el periodo de entrenamiento (``REFERENCE_START`` a ``REFERENCE_END``) contra el mes
posterior reservado para producción (septiembre de 2026, ``CURRENT_START`` a ``CURRENT_END``).

Uso desde la raíz del repositorio:

    python -m mibici.monitoring ingest
    python -m mibici.monitoring prepare --question all
    python -m mibici.monitoring prepare --question p1 --synthetic
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    CURRENT_END,
    CURRENT_MONTH_DATE,
    CURRENT_START,
    EVIDENTLY_SAMPLE_SIZE,
    MIBICI_BASE_URL,
    MIBICI_DATA_PAGE,
    P1_CATEGORICAL_FEATURES,
    P2_CATEGORICAL_FEATURES,
    P3_CATEGORICAL_FEATURES,
    PRODUCTION_TRIPS_PATH,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REFERENCE_END,
    REFERENCE_START,
    STATION_COLUMNS,
    STATIONS_PATH,
    TRAIN_START,
    TRIPS_PATH,
)
from mibici.dataset import (
    clean_trips,
    download_files,
    filter_paths,
    get_paths,
    get_url,
    norm_cols,
    read_raw_data,
)
from mibici.features import (
    P1_FEATURES,
    P1_TARGET,
    P2_FEATURES,
    P2_TARGET,
    P3_FEATURES,
    build_p1_table,
    build_p2_table,
    build_p3_table,
    load_data,
    load_production_trips,
)

app = typer.Typer(
    help="Ingesta del periodo actual y preparación de conjuntos para monitoreo de drift."
)

VALID_QUESTIONS = ("p1", "p2", "p3")

QUESTION_METADATA: dict[str, dict[str, list[str] | str | None]] = {
    "p1": {
        "features": P1_FEATURES,
        "categorical": P1_CATEGORICAL_FEATURES,
        "target": P1_TARGET,
    },
    "p2": {
        "features": P2_FEATURES,
        "categorical": P2_CATEGORICAL_FEATURES,
        "target": P2_TARGET,
    },
    "p3": {
        "features": P3_FEATURES,
        "categorical": P3_CATEGORICAL_FEATURES,
        "target": None,
    },
}


@dataclass(frozen=True)
class MonitoringDatasets:
    """Conjuntos de referencia y producción listos para evaluación de drift y A/B testing.

    Attributes:
        question: Pregunta del proyecto (``p1``, ``p2`` o ``p3``).
        reference: Tabla del periodo de referencia (entrenamiento).
        current: Tabla del periodo actual en producción (septiembre de 2026).
        features: Lista ordenada de variables de entrada del modelo.
        categorical_features: Subconjunto de variables categóricas.
        numerical_features: Subconjunto de variables numéricas continuas o discretas.
        target: Nombre de la variable objetivo (``None`` en P3).
        reference_start: Fecha de inicio (incluida) del conjunto de referencia.
        reference_end: Fecha de fin (excluida) del conjunto de referencia.
        current_start: Fecha de inicio (incluida) del conjunto actual.
        current_end: Fecha de fin (excluida) del conjunto actual.
    """

    question: str
    reference: pd.DataFrame
    current: pd.DataFrame
    features: list[str]
    categorical_features: list[str]
    numerical_features: list[str]
    target: str | None
    reference_start: str
    reference_end: str
    current_start: str
    current_end: str


def _expected_month_filename(month_date: str) -> str:
    """Devuelve el nombre de archivo CSV esperado para ``month_date`` (AAAA-MM-DD)."""
    parsed = date.fromisoformat(month_date)
    return f"datos_abiertos_{parsed.year}_{parsed.month:02d}.csv"


def _load_stations_catalog(
    stations_path: Path = STATIONS_PATH, raw_dir: Path = RAW_DATA_DIR
) -> pd.DataFrame:
    """Carga el catálogo de estaciones desde Parquet procesado o desde el CSV crudo."""
    if Path(stations_path).exists():
        return pd.read_parquet(stations_path)
    df_nom = read_raw_data(input_path=raw_dir, file_type="nom")
    if df_nom.empty:
        raise FileNotFoundError(
            f"No existe {stations_path} ni un archivo de nomenclatura en {raw_dir}. "
            "Ejecuta primero `make data`."
        )
    return norm_cols(df_nom)[list(STATION_COLUMNS)]


def ingest_production_data(
    month_date: str = CURRENT_MONTH_DATE,
    raw_dir: Path = RAW_DATA_DIR,
    stations_path: Path = STATIONS_PATH,
    output_path: Path = PRODUCTION_TRIPS_PATH,
    overwrite: bool = False,
) -> Path:
    """Descarga, limpia y guarda en Parquet los viajes del mes de producción.

    Escribe el resultado en ``PRODUCTION_TRIPS_PATH`` (``data/processed/meses_produccion.parquet``)
    sin modificar ``meses.parquet`` ni ``datapackage.json``.

    Args:
        month_date: Primer día del mes a ingerir en formato ``AAAA-MM-DD``.
        raw_dir: Carpeta donde se almacenan los CSV crudos.
        stations_path: Ruta del Parquet de nomenclatura procesada.
        output_path: Ruta del Parquet de salida para los viajes de producción.
        overwrite: Si es ``True``, vuelve a descargar el CSV aunque ya exista en disco.

    Returns:
        Ruta del archivo Parquet generado.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo del mes ni en disco ni en el portal.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = _expected_month_filename(month_date)
    local_csv = raw_dir / filename

    if not local_csv.exists() or overwrite:
        logger.info(f"Buscando '{filename}' en {MIBICI_DATA_PAGE}...")
        response = get_url(MIBICI_DATA_PAGE)
        paths = get_paths(response.text)
        matched = filter_paths(paths["Months"], month_date, month_date)
        if not matched:
            raise FileNotFoundError(
                f"No se encontró el archivo para {month_date} ({filename}) en {MIBICI_DATA_PAGE}."
            )
        download_files(MIBICI_BASE_URL, matched, raw_dir, overwrite=overwrite)

    if not local_csv.exists():
        raise FileNotFoundError(f"No existe el archivo crudo esperado: {local_csv}")

    df_nom = _load_stations_catalog(stations_path=stations_path, raw_dir=raw_dir)
    df_months = read_raw_data(
        input_path=raw_dir, file_type="month", allowed_files=[local_csv.name]
    )
    df_months = norm_cols(df_months)
    df_clean = clean_trips(df_months, df_nom)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    logger.success(
        f"Periodo actual ({month_date}) en {output_path}: {len(df_clean):,} viajes limpios"
    )
    return output_path


def _slice_period(table: pd.DataFrame, start_date: str, end_date: str, label: str) -> pd.DataFrame:
    """Filtra ``table`` en ``[start_date, end_date)`` y verifica que no quede vacía."""
    times = table["period_start"]
    mask = (times >= pd.Timestamp(start_date)) & (times < pd.Timestamp(end_date))
    subset = table.loc[mask].reset_index(drop=True)
    if subset.empty:
        raise ValueError(
            f"El conjunto '{label}' ({start_date} a {end_date}) quedó vacío. "
            f"Rango disponible en la tabla: {times.min()} a {times.max()}."
        )
    return subset


def prepare_monitoring_data(
    question: str,
    trips: pd.DataFrame,
    production_trips: pd.DataFrame,
    stations: pd.DataFrame,
    reference_start: str = REFERENCE_START,
    reference_end: str = REFERENCE_END,
    current_start: str = CURRENT_START,
    current_end: str = CURRENT_END,
    sample_size: int | None = None,
    seed: int = RANDOM_SEED,
) -> MonitoringDatasets:
    """Construye las tablas de referencia y periodo actual para ``p1``, ``p2`` o ``p3``.

    En P1 y P2 concatena los viajes históricos (``trips``) con los viajes del periodo actual
    (``production_trips``) antes de calcular la malla horaria y los rezagos. De este modo,
    los rezagos de 24 h y 7 días de la primera semana de septiembre de 2026 se alimentan con
    el cierre de agosto de 2026 sin generar valores ``NaN`` artificiales. En P3 calcula el
    perfil de uso de cada estación en la ventana de referencia y lo recalcula en la ventana
    del periodo actual.

    Args:
        question: Pregunta del proyecto (``p1``, ``p2`` o ``p3``).
        trips: Viajes limpios del periodo base (``load_data``).
        production_trips: Viajes limpios del periodo actual en producción.
        stations: Catálogo de estaciones (``load_data``).
        reference_start: Inicio (incluido) de la ventana de referencia.
        reference_end: Fin (excluido) de la ventana de referencia.
        current_start: Inicio (incluido) de la ventana actual.
        current_end: Fin (excluido) de la ventana actual.
        sample_size: Si se indica, toma una muestra determinista máxima de ``sample_size`` filas.
        seed: Semilla aleatoria para muestreo reproducible.

    Returns:
        Instancia de ``MonitoringDatasets`` con las tablas ``reference`` y ``current``.

    Raises:
        ValueError: Si la pregunta no es válida o si alguna ventana temporal queda vacía.
    """
    normalized = question.strip().lower()
    if normalized not in VALID_QUESTIONS:
        raise ValueError(
            f"Pregunta '{question}' desconocida. Opciones válidas: {VALID_QUESTIONS}."
        )

    combined_trips = (
        pd.concat([trips, production_trips], ignore_index=True)
        .drop_duplicates("trip_id")
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    meta = QUESTION_METADATA[normalized]
    features = list(meta["features"])  # type: ignore[arg-type]
    cat_features = list(meta["categorical"])  # type: ignore[arg-type]
    num_features = [col for col in features if col not in cat_features]
    target = meta["target"] if isinstance(meta["target"], str) else None

    if normalized == "p1":
        full_table = build_p1_table(combined_trips, stations)
        reference = _slice_period(full_table, reference_start, reference_end, "reference")
        current = _slice_period(full_table, current_start, current_end, "current")
    elif normalized == "p2":
        full_table = build_p2_table(combined_trips, stations)
        reference = _slice_period(full_table, reference_start, reference_end, "reference")
        current = _slice_period(full_table, current_start, current_end, "current")
    else:
        reference = build_p3_table(
            combined_trips, stations, start_date=reference_start, end_date=reference_end
        )
        current = build_p3_table(
            combined_trips, stations, start_date=current_start, end_date=current_end
        )

    if sample_size is not None:
        if len(reference) > sample_size:
            reference = (
                reference.sample(n=sample_size, random_state=seed)
                .sort_index()
                .reset_index(drop=True)
            )
        if len(current) > sample_size:
            current = (
                current.sample(n=sample_size, random_state=seed)
                .sort_index()
                .reset_index(drop=True)
            )

    logger.info(
        f"[{normalized.upper()}] Conjuntos preparados · referencia ({reference_start} a "
        f"{reference_end}): {len(reference):,} filas | actual ({current_start} a "
        f"{current_end}): {len(current):,} filas"
    )
    return MonitoringDatasets(
        question=normalized,
        reference=reference,
        current=current,
        features=features,
        categorical_features=cat_features,
        numerical_features=num_features,
        target=target,
        reference_start=reference_start,
        reference_end=reference_end,
        current_start=current_start,
        current_end=current_end,
    )


def build_synthetic_monitoring_data(
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera viajes históricos, viajes de producción con drift y estaciones sintéticas.

    Permite validar el flujo de monitoreo en pruebas automatizadas y ejecuciones con
    ``--synthetic`` sin descargar los archivos reales del portal. Introduce un incremento
    de demanda y cambio de distribución horaria en septiembre de 2026 para simular drift.

    Args:
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Tupla ``(trips, production_trips, stations)`` con el contrato de ``load_data``.
    """
    rng = np.random.default_rng(seed)
    stations = pd.DataFrame(
        {
            "station_id": np.arange(1, 7, dtype="int64"),
            "name": [f"GDL-00{i}" for i in range(1, 7)],
            "latitude": np.linspace(20.66, 20.70, 6),
            "longitude": np.linspace(-103.38, -103.34, 6),
            "capacity": np.array([10, 12, 15, 18, 20, 24], dtype="int64"),
        }
    )

    hist_dates = pd.date_range(TRAIN_START, "2026-08-31 22:00:00", freq="3h")
    hist_dates = hist_dates[~hist_dates.hour.isin([1, 2, 3, 4])]
    n_hist = len(hist_dates) * 3
    hist_starts = np.repeat(hist_dates.values, 3) + pd.to_timedelta(
        rng.integers(0, 40, size=n_hist), unit="m"
    )
    hist_durations = pd.to_timedelta(rng.integers(6, 30, size=n_hist), unit="m")
    origin_probs_ref = [0.28, 0.22, 0.18, 0.14, 0.10, 0.08]
    dest_probs_ref = [0.10, 0.12, 0.14, 0.18, 0.22, 0.24]

    trips = pd.DataFrame(
        {
            "trip_id": np.arange(1, n_hist + 1, dtype="int64"),
            "start_time": pd.to_datetime(hist_starts),
            "end_time": pd.to_datetime(hist_starts + hist_durations),
            "origin_station_id": rng.choice(
                stations["station_id"].to_numpy(), size=n_hist, p=origin_probs_ref
            ).astype("int64"),
            "destination_station_id": rng.choice(
                stations["station_id"].to_numpy(), size=n_hist, p=dest_probs_ref
            ).astype("int64"),
        }
    )

    # Septiembre 2026: mayor frecuencia de viajes (6 por bloque) y cambio en estaciones/horarios
    prod_dates = pd.date_range(CURRENT_START, "2026-09-30 22:00:00", freq="2h")
    prod_dates = prod_dates[~prod_dates.hour.isin([1, 2, 3, 4])]
    n_prod = len(prod_dates) * 6
    prod_starts = np.repeat(prod_dates.values, 6) + pd.to_timedelta(
        rng.integers(0, 40, size=n_prod), unit="m"
    )
    prod_durations = pd.to_timedelta(rng.integers(8, 35, size=n_prod), unit="m")
    origin_probs_cur = [0.08, 0.10, 0.12, 0.18, 0.24, 0.28]
    dest_probs_cur = [0.30, 0.24, 0.16, 0.12, 0.10, 0.08]

    production_trips = pd.DataFrame(
        {
            "trip_id": np.arange(n_hist + 1, n_hist + n_prod + 1, dtype="int64"),
            "start_time": pd.to_datetime(prod_starts),
            "end_time": pd.to_datetime(prod_starts + prod_durations),
            "origin_station_id": rng.choice(
                stations["station_id"].to_numpy(), size=n_prod, p=origin_probs_cur
            ).astype("int64"),
            "destination_station_id": rng.choice(
                stations["station_id"].to_numpy(), size=n_prod, p=dest_probs_cur
            ).astype("int64"),
        }
    )
    return trips, production_trips, stations


def load_or_ingest_monitoring_inputs(
    synthetic: bool = False,
    trips_path: Path = TRIPS_PATH,
    stations_path: Path = STATIONS_PATH,
    production_trips_path: Path = PRODUCTION_TRIPS_PATH,
    month_date: str = CURRENT_MONTH_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los viajes base, los de producción y las estaciones (o los genera en modo sintético).

    Si ``production_trips_path`` aún no existe en disco y ``synthetic`` es ``False``,
    ejecuta automáticamente ``ingest_production_data`` para construirlo.
    """
    if synthetic:
        return build_synthetic_monitoring_data()

    trips, stations = load_data(trips_path=trips_path, stations_path=stations_path)
    if not Path(production_trips_path).exists():
        ingest_production_data(
            month_date=month_date,
            stations_path=stations_path,
            output_path=production_trips_path,
        )
    production_trips = load_production_trips(
        trips_path=production_trips_path, stations_path=stations_path
    )
    return trips, production_trips, stations


@app.command("ingest")
def ingest_command(
    month_date: Annotated[
        str, typer.Option(help="Primer día del mes de producción (AAAA-MM-DD).")
    ] = CURRENT_MONTH_DATE,
    output_path: Annotated[
        Path, typer.Option(help="Ruta del Parquet de salida para los viajes de producción.")
    ] = PRODUCTION_TRIPS_PATH,
    overwrite: Annotated[
        bool, typer.Option(help="Forzar descarga del CSV aunque ya exista en data/raw/.")
    ] = False,
) -> None:
    """Descarga y limpia los viajes del mes de producción (por defecto septiembre de 2026)."""
    ingest_production_data(month_date=month_date, output_path=output_path, overwrite=overwrite)


@app.command("prepare")
def prepare_command(
    question: Annotated[str, typer.Option(help="Pregunta a preparar: p1, p2, p3 o all.")] = "all",
    sample_size: Annotated[
        int, typer.Option(help="Muestra determinista máxima por conjunto (0 = sin límite).")
    ] = EVIDENTLY_SAMPLE_SIZE,
    synthetic: Annotated[
        bool, typer.Option(help="Usar datos sintéticos en memoria sin requerir Parquet locales.")
    ] = False,
) -> None:
    """Construye y verifica los conjuntos de referencia y producción para monitoreo."""
    normalized = question.strip().lower()
    questions = list(VALID_QUESTIONS) if normalized == "all" else [normalized]
    trips, production_trips, stations = load_or_ingest_monitoring_inputs(synthetic=synthetic)
    effective_sample = sample_size if sample_size > 0 else None

    for q in questions:
        ds = prepare_monitoring_data(
            question=q,
            trips=trips,
            production_trips=production_trips,
            stations=stations,
            sample_size=effective_sample,
        )
        logger.success(
            f"[{ds.question.upper()}] OK · referencia={len(ds.reference):,} filas "
            f"({ds.reference_start} -> {ds.reference_end}) | actual={len(ds.current):,} filas "
            f"({ds.current_start} -> {ds.current_end}) · {len(ds.features)} variables"
        )


if __name__ == "__main__":
    app()
