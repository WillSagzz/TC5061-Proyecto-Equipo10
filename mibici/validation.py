"""Validación de calidad, integridad y drift de datos con DeepChecks.

Implementa las validaciones tabulares sobre las tablas de modelado de
P1, P2 y P3 generadas por ``mibici/features.py``. Complementa las validaciones físicas,
de hash SHA-256 y de esquema que ``frictionless`` realiza en ``dataset.py`` y
``datapackage.py`` sobre los CSV crudos y los Parquet base.

Uso desde la raíz del repositorio:

    python -m mibici.validation
    python -m mibici.validation --question p1 --sample-size 50000
    python -m mibici.validation --synthetic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    DEEPCHECKS_MAX_DRIFT_EMD,
    DEEPCHECKS_MAX_DRIFT_PSI,
    DEEPCHECKS_MAX_LABEL_DRIFT,
    DEEPCHECKS_MAX_NULL_RATIO,
    DEEPCHECKS_REPORTS_DIR,
    DEEPCHECKS_SAMPLE_SIZE,
    P1_CATEGORICAL_FEATURES,
    P2_CATEGORICAL_FEATURES,
    P3_CATEGORICAL_FEATURES,
    RANDOM_SEED,
    STATIONS_PATH,
    TEST_START,
    TRAIN_START,
    TRIPS_PATH,
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
    load_data,
    split_by_date,
)

app = typer.Typer(help="Validación de integridad y drift de datos con DeepChecks.")


def _ensure_numpy_compat() -> None:
    """Restaura los alias de NumPy 1.x requeridos por ``deepchecks==0.19.1`` en NumPy 2.x."""
    aliases = {
        "Inf": np.inf,
        "NaN": np.nan,
        "NAN": np.nan,
        "infty": np.inf,
        "float_": np.float64,
        "int_": np.int64,
        "unicode_": np.str_,
        "bool8": np.bool_,
    }
    for alias, target in aliases.items():
        if not hasattr(np, alias):
            setattr(np, alias, target)


_ensure_numpy_compat()


@dataclass(frozen=True)
class ValidationReport:
    """Resultado consolidado de una suite de validación de DeepChecks.

    Attributes:
        question: Pregunta evaluada (``p1``, ``p2`` o ``p3``).
        suite_name: Nombre descriptivo de la suite ejecutada.
        passed: ``True`` si todas las condiciones de la suite aprobaron.
        n_checks: Número total de chequeos ejecutados en la suite.
        n_not_ran: Número de chequeos que no se ejecutaron.
        report_path: Ruta del archivo HTML exportado, o ``None`` si no se guardó.
    """

    question: str
    suite_name: str
    passed: bool
    n_checks: int
    n_not_ran: int
    report_path: Path | None = None


def _sample_table(
    table: pd.DataFrame, sample_size: int | None, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Toma una muestra determinista ordenada si la tabla supera ``sample_size``.

    Conserva el orden por fecha o estación para no alterar las comprobaciones temporales
    de DeepChecks.

    Args:
        table: Tabla de características a muestrear.
        sample_size: Límite máximo de filas, o ``None`` para conservar todas.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        DataFrame muestreado (o el original si tiene menos filas que ``sample_size``).
    """
    if sample_size is None or len(table) <= sample_size:
        return table.reset_index(drop=True)
    sampled = table.sample(n=sample_size, random_state=seed)
    return sampled.sort_index().reset_index(drop=True)


def build_tabular_dataset(
    table: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    label: str | None = None,
    datetime_name: str | None = "period_start",
    index_name: str | None = None,
    sample_size: int | None = DEEPCHECKS_SAMPLE_SIZE,
) -> Any:
    """Construye un ``deepchecks.tabular.Dataset`` restringido al contrato del modelo.

    Selecciona únicamente las columnas predictoras (``features``), la etiqueta objetivo
    (``label``) y la referencia temporal (``datetime_name``), excluyendo variables
    auxiliares de la tabla (como ``arrivals`` o ``net_flow`` de la misma hora) que no
    entran a los modelos.

    Args:
        table: Tabla producida por ``build_p1_table``, ``build_p2_table`` o ``build_p3_table``.
        features: Lista de columnas de entrada del modelo.
        cat_features: Subconjunto de ``features`` que representan variables categóricas.
        label: Nombre de la columna objetivo, o ``None`` para clustering (P3).
        datetime_name: Nombre de la columna de fecha/hora, o ``None`` si no aplica.
        index_name: Nombre de la columna índice, o ``None``.
        sample_size: Máximo de filas a evaluar de forma determinista.

    Returns:
        Instancia de ``deepchecks.tabular.Dataset`` lista para validación.

    Raises:
        ValueError: Si falta alguna columna requerida en ``table``.
    """
    from deepchecks.tabular import Dataset

    selected_cols: list[str] = []
    if index_name is not None:
        selected_cols.append(index_name)
    if datetime_name is not None:
        selected_cols.append(datetime_name)
    selected_cols.extend(features)
    if label is not None:
        selected_cols.append(label)

    missing = [col for col in selected_cols if col not in table.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para DeepChecks: {missing}")

    subset = _sample_table(table[selected_cols], sample_size=sample_size)
    return Dataset(
        subset,
        label=label,
        features=features,
        cat_features=cat_features,
        datetime_name=datetime_name,
        index_name=index_name,
    )


def build_integrity_suite(
    name: str,
    max_null_ratio: float = DEEPCHECKS_MAX_NULL_RATIO,
    max_duplicate_ratio: float = 0.60,
    include_label_checks: bool = True,
) -> Any:
    """Crea la suite de integridad de datos de DeepChecks con condiciones explícitas.

    Evalúa columnas constantes, porcentaje de valores nulos, filas duplicadas, tipos
    mezclados, correlaciones entre predictores y ausencia de fugas hacia la etiqueta.

    Args:
        name: Título de la suite.
        max_null_ratio: Proporción máxima permitida de ``NaN`` por columna (0.0 para P3;
            ``DEEPCHECKS_MAX_NULL_RATIO`` para P1 y P2 por el arranque de los rezagos).
        max_duplicate_ratio: Proporción máxima de vectores de features idénticos (0.0 en
            P3; mayor en P1 y P2 porque horas valle de distintas semanas comparten
            exactamente los mismos rezagos en cero).
        include_label_checks: ``True`` cuando el dataset tiene variable objetivo (P1 y P2).

    Returns:
        Instancia de ``deepchecks.tabular.Suite`` configurada.
    """
    from deepchecks.tabular import Suite
    from deepchecks.tabular.checks import (
        DataDuplicates,
        FeatureFeatureCorrelation,
        FeatureLabelCorrelation,
        IsSingleValue,
        MixedDataTypes,
        MixedNulls,
        PercentOfNulls,
    )

    checks = [
        IsSingleValue().add_condition_not_single_value(),
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(max_null_ratio),
        DataDuplicates().add_condition_ratio_less_or_equal(max_duplicate_ratio),
        MixedNulls().add_condition_different_nulls_less_equal_to(1),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range(),
        FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
            threshold=0.95, n_pairs=2
        ),
    ]
    if include_label_checks:
        checks.append(FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.90))
    return Suite(name, *checks)


def build_train_test_suite(name: str) -> Any:
    """Crea la suite de validación y drift entre entrenamiento y prueba para P1 y P2.

    Verifica formalmente la ausencia de fugas temporales (solapamiento o fechas duplicadas
    entre ``train`` y ``test``) y evalúa el desplazamiento de distribución en las
    variables predictoras (``FeatureDrift``) y en la etiqueta objetivo (``LabelDrift``).

    Args:
        name: Título de la suite.

    Returns:
        Instancia de ``deepchecks.tabular.Suite`` configurada.
    """
    from deepchecks.tabular import Suite
    from deepchecks.tabular.checks import (
        DateTrainTestLeakageDuplicates,
        DateTrainTestLeakageOverlap,
        FeatureDrift,
        LabelDrift,
    )

    return Suite(
        name,
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0.0),
        DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_less_or_equal(0.0),
        FeatureDrift().add_condition_drift_score_less_than(
            max_allowed_categorical_score=DEEPCHECKS_MAX_DRIFT_PSI,
            max_allowed_numeric_score=DEEPCHECKS_MAX_DRIFT_EMD,
            allowed_num_features_exceeding_threshold=1,
        ),
        LabelDrift().add_condition_drift_score_less_than(
            max_allowed_drift_score=DEEPCHECKS_MAX_LABEL_DRIFT
        ),
    )


def _run_and_export(
    question: str,
    suite: Any,
    train_dataset: Any,
    test_dataset: Any | None = None,
    output_dir: Path | None = DEEPCHECKS_REPORTS_DIR,
    filename: str | None = None,
    model_classes: list[str] | None = None,
) -> ValidationReport:
    """Ejecuta una suite de DeepChecks y opcionalmente exporta el reporte HTML."""
    classes = sorted(model_classes) if model_classes is not None else None
    result = (
        suite.run(train_dataset, model_classes=classes)
        if test_dataset is None
        else suite.run(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_classes=classes,
        )
    )
    passed = bool(result.passed(fail_if_warning=False, fail_if_check_not_run=True))
    n_not_ran = len(result.get_not_ran_checks())
    n_checks = len(result.results)

    report_path: Path | None = None
    if output_dir is not None and filename is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / filename
        if target.exists():
            target.unlink()
        saved = result.save_as_html(str(target), as_widget=False)
        report_path = Path(saved)
        logger.info(f"Reporte DeepChecks guardado en: {report_path}")

    status = "PASS" if passed else "FAIL"
    logger.info(
        f"[{question.upper()}] {suite.name}: {status} "
        f"(ejecutados={n_checks}, no_ejecutados={n_not_ran})"
    )
    return ValidationReport(
        question=question,
        suite_name=suite.name,
        passed=passed,
        n_checks=n_checks,
        n_not_ran=n_not_ran,
        report_path=report_path,
    )


def validate_p1(
    p1_table: pd.DataFrame,
    sample_size: int | None = DEEPCHECKS_SAMPLE_SIZE,
    output_dir: Path | None = DEEPCHECKS_REPORTS_DIR,
) -> list[ValidationReport]:
    """Ejecuta las suites de integridad y de train-test drift para la tabla de P1.

    Args:
        p1_table: Tabla generada por ``build_p1_table``.
        sample_size: Máximo de filas por partición a evaluar.
        output_dir: Carpeta donde exportar los reportes HTML, o ``None``.

    Returns:
        Lista con los reportes de integridad y de validación train-test.
    """
    splits = split_by_date(p1_table, "period_start")
    ds_train = build_tabular_dataset(
        splits["train"],
        features=P1_FEATURES,
        cat_features=P1_CATEGORICAL_FEATURES,
        label=P1_TARGET,
        datetime_name="period_start",
        sample_size=sample_size,
    )
    ds_test = build_tabular_dataset(
        splits["test"],
        features=P1_FEATURES,
        cat_features=P1_CATEGORICAL_FEATURES,
        label=P1_TARGET,
        datetime_name="period_start",
        sample_size=sample_size,
    )
    integrity_report = _run_and_export(
        question="p1",
        suite=build_integrity_suite("P1 · Integridad de Datos (Train)"),
        train_dataset=ds_train,
        output_dir=output_dir,
        filename="p1_integrity.html",
    )
    drift_report = _run_and_export(
        question="p1",
        suite=build_train_test_suite("P1 · Validación Train-Test y Drift"),
        train_dataset=ds_train,
        test_dataset=ds_test,
        output_dir=output_dir,
        filename="p1_train_test_drift.html",
    )
    return [integrity_report, drift_report]


def validate_p2(
    p2_table: pd.DataFrame,
    sample_size: int | None = DEEPCHECKS_SAMPLE_SIZE,
    output_dir: Path | None = DEEPCHECKS_REPORTS_DIR,
) -> list[ValidationReport]:
    """Ejecuta las suites de integridad y de train-test drift para la tabla de P2.

    Args:
        p2_table: Tabla generada por ``build_p2_table``.
        sample_size: Máximo de filas por partición a evaluar.
        output_dir: Carpeta donde exportar los reportes HTML, o ``None``.

    Returns:
        Lista con los reportes de integridad y de validación train-test.
    """
    splits = split_by_date(p2_table, "period_start")
    ds_train = build_tabular_dataset(
        splits["train"],
        features=P2_FEATURES,
        cat_features=P2_CATEGORICAL_FEATURES,
        label=P2_TARGET,
        datetime_name="period_start",
        sample_size=sample_size,
    )
    ds_test = build_tabular_dataset(
        splits["test"],
        features=P2_FEATURES,
        cat_features=P2_CATEGORICAL_FEATURES,
        label=P2_TARGET,
        datetime_name="period_start",
        sample_size=sample_size,
    )
    integrity_report = _run_and_export(
        question="p2",
        suite=build_integrity_suite("P2 · Integridad de Datos (Train)"),
        train_dataset=ds_train,
        output_dir=output_dir,
        filename="p2_integrity.html",
        model_classes=RISK_CLASSES,
    )
    drift_report = _run_and_export(
        question="p2",
        suite=build_train_test_suite("P2 · Validación Train-Test y Drift"),
        train_dataset=ds_train,
        test_dataset=ds_test,
        output_dir=output_dir,
        filename="p2_train_test_drift.html",
        model_classes=RISK_CLASSES,
    )
    return [integrity_report, drift_report]


def validate_p3(
    p3_table: pd.DataFrame,
    output_dir: Path | None = DEEPCHECKS_REPORTS_DIR,
) -> list[ValidationReport]:
    """Ejecuta la suite de integridad sobre la tabla de perfiles de estaciones de P3.

    En P3 no existe variable objetivo ni partición train-test (se agrupa sobre los
    perfiles construidos con el periodo de entrenamiento), y no se permiten valores nulos
    en ninguna de las variables de ``P3_FEATURES``.

    Args:
        p3_table: Tabla generada por ``build_p3_table``.
        output_dir: Carpeta donde exportar el reporte HTML, o ``None``.

    Returns:
        Lista con el reporte de integridad de P3.
    """
    ds_train = build_tabular_dataset(
        p3_table,
        features=P3_FEATURES,
        cat_features=P3_CATEGORICAL_FEATURES,
        label=None,
        datetime_name=None,
        index_name="station_id",
        sample_size=None,
    )
    integrity_report = _run_and_export(
        question="p3",
        suite=build_integrity_suite(
            "P3 · Integridad de Perfiles de Estaciones",
            max_null_ratio=0.0,
            max_duplicate_ratio=0.0,
            include_label_checks=False,
        ),
        train_dataset=ds_train,
        output_dir=output_dir,
        filename="p3_integrity.html",
    )
    return [integrity_report]


def build_synthetic_raw_data(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera viajes y estaciones sintéticos que cumplen el contrato de ``load_data``.

    Permite probar el pipeline completo de ``features.py`` y ``validation.py`` de forma
    instantánea en entornos donde aún no se han descargado los 758 MB de CSV crudos.
    Asigna probabilidades asimétricas de salida y llegada para que en P2 aparezcan las
    tres clases de riesgo (``vaciarse``, ``llenarse`` y ``sin_riesgo``).

    Args:
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Par ``(trips, stations)`` con los nombres y tipos que devuelve ``load_data``.
    """
    rng = np.random.default_rng(seed)
    stations = pd.DataFrame(
        {
            "station_id": np.arange(1, 7, dtype="int64"),
            "name": [f"GDL-00{i}" for i in range(1, 7)],
            "latitude": np.linspace(20.66, 20.70, 6),
            "longitude": np.linspace(-103.38, -103.34, 6),
            "capacity": np.array([4, 5, 6, 7, 5, 6], dtype="int64"),
        }
    )
    dates = pd.date_range(TRAIN_START, "2026-08-25 22:00:00", freq="2h")
    dates = dates[~dates.hour.isin([1, 2, 3, 4])]
    n_trips = len(dates) * 4
    starts = np.repeat(dates.values, 4) + pd.to_timedelta(
        rng.integers(0, 45, size=n_trips), unit="m"
    )
    durations = pd.to_timedelta(rng.integers(5, 35, size=n_trips), unit="m")
    origin_probs = [0.30, 0.24, 0.16, 0.12, 0.10, 0.08]
    dest_probs = [0.08, 0.10, 0.12, 0.16, 0.24, 0.30]
    origins = rng.choice(stations["station_id"].to_numpy(), size=n_trips, p=origin_probs)
    destinations = rng.choice(stations["station_id"].to_numpy(), size=n_trips, p=dest_probs)

    trips = pd.DataFrame(
        {
            "trip_id": np.arange(1, n_trips + 1, dtype="int64"),
            "start_time": pd.to_datetime(starts),
            "end_time": pd.to_datetime(starts + durations),
            "origin_station_id": origins.astype("int64"),
            "destination_station_id": destinations.astype("int64"),
        }
    )
    logger.info(
        f"Datos sintéticos generados ({TRAIN_START} a {VALIDATION_START} / {TEST_START}): "
        f"{len(trips):,} viajes y {len(stations)} estaciones"
    )
    return trips, stations


@app.command()
def main(
    question: str = "all",
    sample_size: int = DEEPCHECKS_SAMPLE_SIZE,
    output_dir: Path = DEEPCHECKS_REPORTS_DIR,
    synthetic: bool = False,
) -> None:
    """Ejecuta las validaciones de DeepChecks sobre las tablas de P1, P2 y P3.

    Args:
        question: Pregunta a validar ('p1', 'p2', 'p3' o 'all' para las tres).
        sample_size: Número máximo de filas por partición a evaluar en P1 y P2.
        output_dir: Carpeta donde se guardan los reportes HTML de DeepChecks.
        synthetic: Si es True, usa datos sintéticos en memoria cuando no existen los Parquet.
    """
    valid_questions = ("p1", "p2", "p3", "all")
    if question not in valid_questions:
        logger.error(f"Pregunta '{question}' no válida. Usa una de: {valid_questions}")
        raise typer.Exit(code=1)

    if synthetic:
        trips, stations = build_synthetic_raw_data()
    else:
        if not TRIPS_PATH.exists() or not STATIONS_PATH.exists():
            logger.error(
                f"No existen {TRIPS_PATH.name} o {STATIONS_PATH.name}. "
                "Ejecuta primero `python -m mibici.dataset` (o usa `--synthetic`)."
            )
            raise typer.Exit(code=1)
        trips, stations = load_data(TRIPS_PATH, STATIONS_PATH)

    reports: list[ValidationReport] = []
    if question in ("p1", "all"):
        reports.extend(validate_p1(build_p1_table(trips, stations), sample_size, output_dir))
    if question in ("p2", "all"):
        reports.extend(validate_p2(build_p2_table(trips, stations), sample_size, output_dir))
    if question in ("p3", "all"):
        reports.extend(validate_p3(build_p3_table(trips, stations), output_dir))

    failed = [r for r in reports if not r.passed]
    if failed:
        names = ", ".join(f"{r.question}:{r.suite_name}" for r in failed)
        logger.warning(f"Suites con condiciones en alerta o fallo: {names}")
    else:
        logger.success(f"Todas las suites de DeepChecks ({len(reports)}) aprobaron.")


if __name__ == "__main__":
    app()
