"""Pruebas de ``modeling/baselines.py``: Dummy por grupo (P1 y P2) y K-Means estandarizado (P3)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import StandardScaler

from mibici.config import RANDOM_SEED
from mibici.features import NO_RISK, RISK_EMPTY, RISK_FULL
from mibici.modeling.baselines import (
    GroupedDummy,
    build_p1_baseline,
    build_p2_baseline,
    build_p3_baseline,
)


def _p1_history() -> tuple[pd.DataFrame, pd.Series]:
    """Historial de P1: dos estaciones y dos horas con salidas conocidas por grupo.

    Returns:
        ``X`` con ``station_id`` y ``hour_of_day`` e ``y`` con las salidas. Promedios por
        grupo: (1, 8) = 3, (1, 9) = 10, (2, 8) = 25, (2, 9) = 0; promedio general = 11.
    """
    X = pd.DataFrame({"station_id": [1, 1, 1, 2, 2, 2], "hour_of_day": [8, 8, 9, 8, 8, 9]})
    y = pd.Series([2, 4, 10, 20, 30, 0])
    return X, y


def _p2_history() -> tuple[pd.DataFrame, pd.Series]:
    """Historial de P2: dos estaciones en el mismo intervalo entre semana.

    Returns:
        ``X`` con los grupos del baseline de P2 e ``y`` con el riesgo. Estación 1: vaciarse,
        vaciarse, sin_riesgo; estación 2: llenarse, sin_riesgo, sin_riesgo.
    """
    X = pd.DataFrame(
        {"station_id": [1, 1, 1, 2, 2, 2], "interval_of_day": [2] * 6, "is_weekend": [0] * 6}
    )
    y = pd.Series([RISK_EMPTY, RISK_EMPTY, NO_RISK, RISK_FULL, NO_RISK, NO_RISK])
    return X, y


def _probabilities(model: GroupedDummy, X: pd.DataFrame) -> pd.DataFrame:
    """Probabilidades por clase con el nombre de cada clase como columna."""
    return pd.DataFrame(model.predict_proba(X), columns=model.classes_)


def test_p1_baseline_predicts_the_mean_of_each_station_and_hour() -> None:
    """Caso normal: cada combinación de estación y hora recibe el promedio de su propio historial."""
    model = build_p1_baseline().fit(*_p1_history())
    X_new = pd.DataFrame({"station_id": [1, 1, 2, 2], "hour_of_day": [8, 9, 8, 9]})

    assert model.predict(X_new).tolist() == [3.0, 10.0, 25.0, 0.0]
    assert model.n_groups_ == 4
    assert model.is_classifier_ is False


def test_grouped_dummy_falls_back_to_the_global_mean_for_unseen_groups() -> None:
    """Caso límite: una estación nueva o una hora sin historial recibe el promedio general."""
    model = build_p1_baseline().fit(*_p1_history())
    X_new = pd.DataFrame({"station_id": [3, 1, 1], "hour_of_day": [8, 12, 8]})

    assert model.predict(X_new).tolist() == [11.0, 11.0, 3.0]


def test_grouped_dummy_defaults_to_a_regressor_without_probabilities() -> None:
    """Caso límite: sin estimador es un ``DummyRegressor`` por grupo y no tiene ``predict_proba``."""
    X = pd.DataFrame({"station_id": [1, 1, 2]})
    model = GroupedDummy(group_columns=("station_id",)).fit(X, [1, 3, 10])

    assert model.predict(pd.DataFrame({"station_id": [1, 2]})).tolist() == [2.0, 10.0]
    with pytest.raises(AttributeError, match="solo está disponible para clasificadores"):
        model.predict_proba(pd.DataFrame({"station_id": [1]}))


def test_p2_baseline_predicts_the_most_frequent_class_of_each_group() -> None:
    """Caso normal: predice la clase más frecuente del grupo y, como probabilidad, la proporción de cada clase."""
    model = build_p2_baseline().fit(*_p2_history())
    X_new = pd.DataFrame({"station_id": [1, 2], "interval_of_day": [2, 2], "is_weekend": [0, 0]})

    probabilities = _probabilities(model, X_new)

    assert model.is_classifier_ is True
    assert model.predict(X_new).tolist() == [RISK_EMPTY, NO_RISK]
    assert probabilities[RISK_EMPTY].tolist() == pytest.approx([2 / 3, 0.0])
    assert probabilities[NO_RISK].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert probabilities[RISK_FULL].tolist() == pytest.approx([0.0, 1 / 3])
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_p2_baseline_uses_global_frequencies_for_unseen_groups() -> None:
    """Caso límite: un grupo sin historial (fin de semana) recibe las frecuencias de todo el entrenamiento."""
    model = build_p2_baseline().fit(*_p2_history())
    X_new = pd.DataFrame({"station_id": [1], "interval_of_day": [2], "is_weekend": [1]})

    probabilities = _probabilities(model, X_new)

    assert model.predict(X_new).tolist() == [NO_RISK]
    assert probabilities.loc[0, RISK_EMPTY] == pytest.approx(2 / 6)
    assert probabilities.loc[0, RISK_FULL] == pytest.approx(1 / 6)
    assert probabilities.loc[0, NO_RISK] == pytest.approx(3 / 6)


def test_baselines_group_by_the_columns_of_each_question() -> None:
    """Caso normal: P1 agrupa por estación y hora con la media; P2, por estación, intervalo y fin de semana."""
    p1, p2 = build_p1_baseline(), build_p2_baseline()

    assert isinstance(p1.estimator, DummyRegressor) and p1.estimator.strategy == "mean"
    assert p1.group_columns == ("station_id", "hour_of_day")
    assert isinstance(p2.estimator, DummyClassifier) and p2.estimator.strategy == "prior"
    assert p2.group_columns == ("station_id", "interval_of_day", "is_weekend")


def test_p3_baseline_standardizes_and_clusters_with_the_project_seed() -> None:
    """Caso normal: ``StandardScaler`` seguido de ``KMeans`` con ``n_clusters`` grupos y la semilla de ``config.py``."""
    model = build_p3_baseline(3)
    scaler, kmeans = model.steps[0][1], model.steps[-1][1]
    # Tres grupos evidentes; la segunda columna tiene otra escala
    X = pd.DataFrame(
        {"a": [0.0, 0.1, 10.0, 10.1, 100.0, 100.1], "b": [0, 0, 500, 500, 1000, 1000]}
    )

    labels = model.fit_predict(X)

    assert isinstance(scaler, StandardScaler) and isinstance(kmeans, KMeans)
    assert kmeans.n_clusters == 3 and kmeans.random_state == RANDOM_SEED
    assert len(set(labels)) == 3
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[4] == labels[5]


def test_p3_baseline_scaling_lets_small_scale_features_matter() -> None:
    """Caso límite: sin estandarizar dominaría la columna de miles; con ella, una de 0 a 1 también separa grupos."""
    X = pd.DataFrame({"volume": [1000.0, 1001.0, 1000.0, 1001.0], "share": [0.0, 0.0, 1.0, 1.0]})

    labels = build_p3_baseline(2).fit_predict(X)

    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert labels[0] != labels[2]
