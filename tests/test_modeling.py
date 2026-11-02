"""Pruebas de reproducibilidad del entrenamiento de P1, P2 y P3: misma semilla, mismos resultados."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, mean_absolute_error, silhouette_score

from mibici.config import RANDOM_SEED
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
    split_by_date,
)
from mibici.modeling.candidates import get_spec, resolve_hyperparams
from mibici.validation import build_synthetic_raw_data

# Solo acelera las pruebas; la semilla sigue siendo la de ``config.py``, como en MLflow
FAST = {"max_iter": "30"}


@pytest.fixture(scope="module")
def synthetic_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Viajes y estaciones sintéticos de ``validation.py``, con ``RANDOM_SEED``, una vez por módulo."""
    return build_synthetic_raw_data(seed=RANDOM_SEED)


def test_training_is_reproducible_with_the_same_seed(
    synthetic_raw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reproducibilidad (P1): entrenar dos veces el modelo elegido da las mismas predicciones y el mismo MAE.

    Usa los datos sintéticos de ``validation.py`` y la semilla de ``config.py``, igual que las
    corridas de MLflow.
    """
    splits = split_by_date(build_p1_table(*synthetic_raw), "period_start")
    X_train, y_train = splits["train"][P1_FEATURES], splits["train"][P1_TARGET]
    X_val, y_val = splits["val"][P1_FEATURES], splits["val"][P1_TARGET]
    spec = get_spec("p1", "hgb_poisson")
    hyperparams = resolve_hyperparams(spec, FAST)

    first = spec.build(hyperparams).fit(X_train, y_train).predict(X_val)
    second = spec.build(hyperparams).fit(X_train, y_train).predict(X_val)

    np.testing.assert_array_equal(first, second)
    assert mean_absolute_error(y_val, first) == mean_absolute_error(y_val, second)


def test_p2_threshold_tuning_is_reproducible_with_the_same_seed(
    synthetic_raw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reproducibilidad (P2): ``hgb_threshold`` entrenado dos veces elige los mismos umbrales y da el mismo F1 macro.

    Los umbrales se eligen con validación y el F1 macro se mide en prueba, como en
    ``train.py``.
    """
    splits = split_by_date(build_p2_table(*synthetic_raw), "period_start")
    X_train, y_train = splits["train"][P2_FEATURES], splits["train"][P2_TARGET]
    X_val, y_val = splits["val"][P2_FEATURES], splits["val"][P2_TARGET]
    X_test, y_test = splits["test"][P2_FEATURES], splits["test"][P2_TARGET]
    spec = get_spec("p2", "hgb_threshold")
    hyperparams = resolve_hyperparams(spec, FAST)

    models = []
    for _ in range(2):
        model = spec.build(hyperparams).fit(X_train, y_train)
        model.tune_thresholds(X_val, y_val)
        models.append(model)
    predictions = [model.predict(X_test) for model in models]
    f1_scores = [
        f1_score(y_test, predicted, labels=RISK_CLASSES, average="macro", zero_division=0)
        for predicted in predictions
    ]

    assert models[0].thresholds_ == models[1].thresholds_
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert f1_scores[0] == f1_scores[1]


def test_p3_clustering_is_reproducible_with_the_same_seed(
    synthetic_raw: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Reproducibilidad (P3): el K-Means del baseline da las mismas etiquetas y la misma silueta dos veces."""
    X = build_p3_table(*synthetic_raw)[P3_FEATURES]
    spec = get_spec("p3", "kmeans")
    hyperparams = resolve_hyperparams(spec, {})

    models = [spec.build(hyperparams) for _ in range(2)]
    labels = [model.fit_predict(X) for model in models]
    silhouettes = [
        silhouette_score(model[:-1].transform(X), assigned)
        for model, assigned in zip(models, labels)
    ]

    np.testing.assert_array_equal(labels[0], labels[1])
    assert silhouettes[0] == silhouettes[1]
