"""Umbral de alerta de P2: a partir de qué probabilidad se avisa que una estación está en riesgo.

Un clasificador elige normalmente la clase con la probabilidad más alta. Con
``class_weight="balanced"`` eso hace que alerte demasiado (muchas falsas alarmas). Aquí se
exige una probabilidad mínima para cada clase de riesgo y esos mínimos se eligen con el
conjunto de validación, buscando el mejor F1 macro. El conjunto de prueba no se usa.
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from mibici.config import P2_THRESHOLD_GRID
from mibici.features import NO_RISK, RISK_CLASSES, RISK_EMPTY, RISK_FULL


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador de P2 que alerta solo si la probabilidad del riesgo supera su umbral.

    Se entrena en dos pasos: ``fit`` entrena el clasificador interno con el conjunto de
    entrenamiento y ``tune_thresholds`` elige los umbrales con el conjunto de validación. Los
    umbrales quedan guardados dentro del modelo, así que el modelo registrado en MLflow ya
    alerta con ellos.

    Args:
        estimator: Clasificador que da probabilidades (por ejemplo, HistGradientBoosting).
        grid: Umbrales que se prueban para cada clase de riesgo.
    """

    def __init__(self, estimator, grid: list[float] | None = None):
        self.estimator = estimator
        self.grid = grid

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ThresholdClassifier":
        """Entrena el clasificador interno. Sin umbrales, predice como el clasificador interno.

        Args:
            X: Variables de entrada del conjunto de entrenamiento.
            y: Clase real de riesgo.

        Returns:
            El propio modelo, entrenado.
        """
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        self.thresholds_ = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad de cada clase, en el orden de ``classes_`` (no cambia con el umbral)."""
        return self.estimator_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Clase de riesgo de cada fila, aplicando los umbrales.

        Returns:
            ``vaciarse``, ``llenarse`` o ``sin_riesgo`` para cada fila.
        """
        if self.thresholds_ is None:
            return self.estimator_.predict(X)
        return self._predict_from_proba(self.predict_proba(X), self.thresholds_)

    def _predict_from_proba(self, probabilities: np.ndarray, thresholds: dict) -> np.ndarray:
        """Aplica los umbrales a las probabilidades.

        Una fila recibe una clase de riesgo si su probabilidad llega al umbral de esa clase.
        Si llega a los dos, gana la que más lo supera (probabilidad ÷ umbral). Si no llega a
        ninguno, queda ``sin_riesgo``.
        """
        classes = list(self.classes_)
        prediction = np.full(len(probabilities), NO_RISK, dtype=object)
        best_ratio = np.zeros(len(probabilities))
        for risk_class, threshold in thresholds.items():
            probability = probabilities[:, classes.index(risk_class)]
            ratio = probability / max(threshold, 1e-9)
            passes = (probability >= threshold) & (ratio > best_ratio)
            prediction[passes] = risk_class
            best_ratio = np.where(passes, ratio, best_ratio)
        return prediction

    def tune_thresholds(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Elige con validación el umbral de ``vaciarse`` y el de ``llenarse``.

        Prueba todas las parejas de umbrales de ``grid`` (por defecto ``P2_THRESHOLD_GRID``) y
        se queda con la de mejor F1 macro de las tres clases.

        Args:
            X: Variables de entrada del conjunto de validación.
            y: Clase real de riesgo del conjunto de validación.

        Returns:
            El F1 macro de validación con los umbrales elegidos.
        """
        grid = self.grid if self.grid is not None else P2_THRESHOLD_GRID
        probabilities = self.predict_proba(X)
        y_true = np.asarray(y)
        best_score, best_thresholds = -1.0, self.thresholds_
        for empty, full in itertools.product(grid, grid):
            thresholds = {RISK_EMPTY: empty, RISK_FULL: full}
            score = _macro_f1(y_true, self._predict_from_proba(probabilities, thresholds))
            if score > best_score:
                best_score, best_thresholds = score, thresholds
        self.thresholds_ = best_thresholds
        return best_score


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 macro de las tres clases de riesgo (más rápido que scikit-learn dentro del ciclo)."""
    scores = []
    for risk_class in RISK_CLASSES:
        is_true, is_pred = y_true == risk_class, y_pred == risk_class
        hits = np.sum(is_true & is_pred)
        total = is_true.sum() + is_pred.sum()
        scores.append(2 * hits / total if total else 0.0)
    return float(np.mean(scores))
