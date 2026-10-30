"""Pruebas automatizadas de validación de datos con DeepChecks y PyTest.

Verifica que las tablas de modelado de P1, P2 y P3 cumplan las suites de integridad y de
validación train-test (casos normales), y que DeepChecks detecte activamente anomalías de
calidad o fugas temporales cuando los datos se corrompen (casos límite).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mibici.config import P1_CATEGORICAL_FEATURES
from mibici.features import (
    P1_FEATURES,
    P1_TARGET,
    build_p1_table,
    build_p2_table,
    build_p3_table,
    split_by_date,
)
from mibici.validation import (
    build_integrity_suite,
    build_synthetic_raw_data,
    build_tabular_dataset,
    build_train_test_suite,
    validate_p1,
    validate_p2,
    validate_p3,
)


@pytest.fixture(scope="module")
def synthetic_tables() -> dict[str, pd.DataFrame]:
    """Construye una sola vez las tablas sintéticas de P1, P2 y P3 para las pruebas."""
    trips, stations = build_synthetic_raw_data()
    return {
        "p1": build_p1_table(trips, stations),
        "p2": build_p2_table(trips, stations),
        "p3": build_p3_table(trips, stations),
    }


def test_validate_p1_passes_and_exports_html(
    synthetic_tables: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    """Verifica que P1 apruebe integridad y train-test drift y exporte los reportes HTML."""
    reports = validate_p1(synthetic_tables["p1"], sample_size=600, output_dir=tmp_path)
    assert len(reports) == 2
    for report in reports:
        assert report.question == "p1"
        assert report.passed is True
        assert report.n_not_ran == 0
        assert report.report_path is not None
        assert report.report_path.exists()
        assert report.report_path.stat().st_size > 0


def test_validate_p2_passes(synthetic_tables: dict[str, pd.DataFrame]) -> None:
    """Verifica que P2 apruebe las suites de integridad y de train-test drift."""
    reports = validate_p2(synthetic_tables["p2"], sample_size=600, output_dir=None)
    assert len(reports) == 2
    assert all(report.passed for report in reports)
    assert all(report.n_not_ran == 0 for report in reports)


def test_validate_p3_passes(synthetic_tables: dict[str, pd.DataFrame]) -> None:
    """Verifica que los perfiles de estaciones de P3 aprueben sin nulos ni duplicados."""
    reports = validate_p3(synthetic_tables["p3"], output_dir=None)
    assert len(reports) == 1
    assert reports[0].passed is True
    assert reports[0].n_not_ran == 0


def test_build_tabular_dataset_missing_columns_raises(
    synthetic_tables: dict[str, pd.DataFrame],
) -> None:
    """Caso límite: lanza ValueError si falta una columna del contrato del modelo."""
    incomplete = synthetic_tables["p1"].drop(columns=["capacity"])
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        build_tabular_dataset(
            incomplete,
            features=P1_FEATURES,
            cat_features=P1_CATEGORICAL_FEATURES,
            label=P1_TARGET,
            sample_size=100,
        )


def test_integrity_suite_detects_corrupted_features(
    synthetic_tables: dict[str, pd.DataFrame],
) -> None:
    """Caso límite: la suite de integridad falla si hay columna constante o exceso de NaN."""
    corrupted = split_by_date(synthetic_tables["p1"], "period_start")["train"].head(400).copy()
    corrupted["capacity"] = 15  # Columna constante (viola IsSingleValue)
    corrupted.loc[corrupted.index[:100], "departures_lag_24"] = np.nan  # 25% de nulos (> 10%)

    dataset = build_tabular_dataset(
        corrupted,
        features=P1_FEATURES,
        cat_features=P1_CATEGORICAL_FEATURES,
        label=P1_TARGET,
        sample_size=None,
    )
    suite = build_integrity_suite("Suite Integridad Corrupta")
    result = suite.run(dataset)
    assert bool(result.passed()) is False


def test_train_test_suite_detects_temporal_leakage(
    synthetic_tables: dict[str, pd.DataFrame],
) -> None:
    """Caso límite: la suite train-test falla si hay solapamiento de fechas entre train y test."""
    splits = split_by_date(synthetic_tables["p1"], "period_start")
    train_subset = splits["train"].tail(300).copy()
    # Fuga temporal intencional: el conjunto de prueba contiene fechas del entrenamiento
    leaked_test = pd.concat(
        [splits["train"].tail(50), splits["test"].head(250)], ignore_index=True
    )

    ds_train = build_tabular_dataset(
        train_subset,
        features=P1_FEATURES,
        cat_features=P1_CATEGORICAL_FEATURES,
        label=P1_TARGET,
        sample_size=None,
    )
    ds_test = build_tabular_dataset(
        leaked_test,
        features=P1_FEATURES,
        cat_features=P1_CATEGORICAL_FEATURES,
        label=P1_TARGET,
        sample_size=None,
    )
    suite = build_train_test_suite("Suite Fuga Temporal")
    result = suite.run(train_dataset=ds_train, test_dataset=ds_test)
    assert bool(result.passed()) is False
