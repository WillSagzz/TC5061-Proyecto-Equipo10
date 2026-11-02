"""Pruebas de ``modeling/thresholds.py``: umbral de alerta de P2 y F1 macro."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

from mibici.config import P2_THRESHOLD_GRID
from mibici.features import NO_RISK, RISK_CLASSES, RISK_EMPTY, RISK_FULL
from mibici.modeling.thresholds import ThresholdClassifier, _macro_f1

# Orden de ``classes_`` del clasificador interno: ``np.unique`` ordena alfabéticamente
CLASS_ORDER = sorted(RISK_CLASSES)


class _FixedProbabilities(ClassifierMixin, BaseEstimator):
    """Clasificador de juguete: aprende las clases en ``fit`` y devuelve probabilidades fijas.

    Args:
        probabilities: Una fila de probabilidades por fila de ``X``, en el orden de
            ``CLASS_ORDER``.
    """

    def __init__(self, probabilities=None):
        self.probabilities = probabilities

    def fit(self, X: pd.DataFrame, y) -> "_FixedProbabilities":
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.probabilities, dtype=float)[: len(X)]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _row(empty: float = 0.0, full: float = 0.0) -> list[float]:
    """Fila de probabilidades en el orden de ``CLASS_ORDER``; lo que falta va a ``sin_riesgo``.

    Args:
        empty: Probabilidad de ``vaciarse``.
        full: Probabilidad de ``llenarse``.

    Returns:
        Las tres probabilidades, que suman 1.
    """
    probabilities = {RISK_EMPTY: empty, RISK_FULL: full, NO_RISK: 1 - empty - full}
    return [probabilities[risk_class] for risk_class in CLASS_ORDER]


def _X(n_rows: int) -> pd.DataFrame:
    """Entradas de relleno: el clasificador de juguete no las mira."""
    return pd.DataFrame({"x": range(n_rows)})


def _fitted(rows: list[list[float]], grid: list[float] | None = None) -> ThresholdClassifier:
    """``ThresholdClassifier`` entrenado con las tres clases presentes y probabilidades fijas.

    Args:
        rows: Probabilidades que devolverá el clasificador interno (al menos tres filas).
        grid: Rejilla de umbrales; ``None`` usa ``P2_THRESHOLD_GRID``.

    Returns:
        El modelo tras ``fit``, todavía sin umbrales.
    """
    y = pd.Series([RISK_CLASSES[i % 3] for i in range(len(rows))])
    return ThresholdClassifier(_FixedProbabilities(rows), grid=grid).fit(_X(len(rows)), y)


def test_without_thresholds_predicts_like_the_inner_classifier() -> None:
    """Caso normal: tras ``fit`` no hay umbrales y gana la clase más probable, como en cualquier clasificador."""
    rows = [_row(empty=0.5, full=0.1), _row(empty=0.1, full=0.6), _row(empty=0.3, full=0.3)]

    model = _fitted(rows)

    assert model.thresholds_ is None
    assert list(model.classes_) == CLASS_ORDER
    assert model.predict(_X(3)).tolist() == [RISK_EMPTY, RISK_FULL, NO_RISK]
    np.testing.assert_array_equal(model.predict_proba(_X(3)), np.array(rows))


def test_with_thresholds_only_alerts_when_the_probability_reaches_them() -> None:
    """Caso normal: con umbral 0.5 alerta si el riesgo llega a 0.5; si no, ``sin_riesgo`` aunque sea la más probable."""
    rows = [_row(empty=0.6), _row(full=0.7), _row(empty=0.49, full=0.10)]
    model = _fitted(rows)
    model.thresholds_ = {RISK_EMPTY: 0.5, RISK_FULL: 0.5}

    assert model.predict(_X(3)).tolist() == [RISK_EMPTY, RISK_FULL, NO_RISK]
    np.testing.assert_array_equal(model.predict_proba(_X(3)), np.array(rows))  # no cambia


@pytest.mark.parametrize(
    ("empty", "full", "expected"),
    [
        pytest.param(0.4, 0.0, RISK_EMPTY, id="empty_exactly_at_threshold_alerts"),
        pytest.param(0.3999, 0.0, NO_RISK, id="empty_just_below_no_alert"),
        pytest.param(0.0, 0.5, RISK_FULL, id="full_exactly_at_threshold_alerts"),
        pytest.param(0.0, 0.4999, NO_RISK, id="full_just_below_no_alert"),
        pytest.param(0.45, 0.55, RISK_EMPTY, id="both_reach_most_relative_wins"),  # 1.125 > 1.1
        pytest.param(0.4, 0.5, RISK_EMPTY, id="tie_in_ratio_keeps_the_first_class"),  # 1.0 = 1.0
    ],
)
def test_predict_at_and_around_the_thresholds(empty: float, full: float, expected: str) -> None:
    """Caso límite: umbrales 0.4 (vaciarse) y 0.5 (llenarse); llegar basta y, si ambos llegan, gana el que más lo supera en proporción."""
    model = _fitted([_row(empty, full), _row(), _row()])
    model.thresholds_ = {RISK_EMPTY: 0.4, RISK_FULL: 0.5}

    assert model.predict(_X(1))[0] == expected


def test_tune_thresholds_picks_the_pair_with_the_best_macro_f1() -> None:
    """Caso normal: con la rejilla [0.3, 0.6], el 0.6 evita las falsas alarmas del 0.4 y logra F1 macro = 1."""
    rows = [_row(empty=0.7), _row(empty=0.4), _row(full=0.7), _row(full=0.4)]
    y = pd.Series([RISK_EMPTY, NO_RISK, RISK_FULL, NO_RISK])
    model = ThresholdClassifier(_FixedProbabilities(rows), grid=[0.3, 0.6]).fit(_X(4), y)

    score = model.tune_thresholds(_X(4), y)

    assert score == 1.0
    assert model.thresholds_ == {RISK_EMPTY: 0.6, RISK_FULL: 0.6}
    assert model.predict(_X(4)).tolist() == y.tolist()


def test_tune_thresholds_uses_the_config_grid_by_default() -> None:
    """Caso límite: sin rejilla propia, los umbrales elegidos salen de ``P2_THRESHOLD_GRID``."""
    rows = [_row(empty=0.7), _row(empty=0.4), _row(full=0.7), _row(full=0.4)]
    y = pd.Series([RISK_EMPTY, NO_RISK, RISK_FULL, NO_RISK])
    model = ThresholdClassifier(_FixedProbabilities(rows)).fit(_X(4), y)

    score = model.tune_thresholds(_X(4), y)

    assert model.grid is None
    assert set(model.thresholds_) == {RISK_EMPTY, RISK_FULL}
    assert set(model.thresholds_.values()) <= set(P2_THRESHOLD_GRID)
    assert score == 1.0


def test_macro_f1_matches_scikit_learn() -> None:
    """Caso normal: coincide con ``f1_score(average="macro")`` de scikit-learn sobre las tres clases."""
    y_true = np.array([RISK_EMPTY, RISK_EMPTY, NO_RISK, RISK_FULL, NO_RISK, NO_RISK])
    y_pred = np.array([RISK_EMPTY, NO_RISK, NO_RISK, RISK_FULL, RISK_FULL, NO_RISK])

    expected = f1_score(y_true, y_pred, labels=RISK_CLASSES, average="macro", zero_division=0)

    assert _macro_f1(y_true, y_pred) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        pytest.param(
            [RISK_EMPTY, RISK_FULL, NO_RISK], [RISK_EMPTY, RISK_FULL, NO_RISK], 1.0, id="perfect"
        ),
        pytest.param(
            [RISK_EMPTY, RISK_FULL, NO_RISK], [NO_RISK, NO_RISK, NO_RISK], 1 / 6, id="never_alerts"
        ),
        pytest.param(
            [RISK_EMPTY, NO_RISK], [RISK_EMPTY, NO_RISK], 2 / 3, id="absent_class_scores_0"
        ),
    ],
)
def test_macro_f1_at_the_extremes(y_true: list[str], y_pred: list[str], expected: float) -> None:
    """Caso límite: perfecto vale 1; nunca alertar solo acierta ``sin_riesgo``; una clase ausente aporta 0."""
    assert _macro_f1(np.array(y_true), np.array(y_pred)) == pytest.approx(expected)
