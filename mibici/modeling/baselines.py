"""Modelos baseline con la interfaz de scikit-learn (fit / predict)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import RANDOM_SEED


class GroupedDummy(BaseEstimator):
    """Entrena un Dummy de scikit-learn distinto para cada grupo.

    ``DummyRegressor`` y ``DummyClassifier`` ignoran las variables de entrada y predicen lo
    mismo para todas las filas. Esta clase entrena uno por cada combinación de
    ``group_columns`` (por ejemplo, estación y hora del día), así cada grupo recibe la
    predicción de su propio historial. Las combinaciones que no aparecen en entrenamiento
    usan un Dummy general entrenado con todos los datos. Sirve para P1 (regresión) y P2
    (clasificación).

    Args:
        estimator: ``DummyRegressor`` o ``DummyClassifier`` que se entrena en cada grupo.
        group_columns: Columnas que definen los grupos.
    """

    def __init__(self, estimator=None, group_columns: tuple[str, ...] = ("station_id",)):
        self.estimator = estimator
        self.group_columns = group_columns

    def fit(self, X: pd.DataFrame, y) -> "GroupedDummy":
        """Entrena un Dummy por grupo y uno general de respaldo.

        Args:
            X: Tabla con las columnas de ``group_columns``.
            y: Variable objetivo.

        Returns:
            El propio modelo entrenado.
        """
        columns = list(self.group_columns)
        base = self.estimator if self.estimator is not None else DummyRegressor()
        self.is_classifier_ = is_classifier(base)
        group_values = X[columns].to_numpy()
        y = np.asarray(y)

        # Dummy general: respaldo para grupos que no aparecen en entrenamiento
        self.fallback_ = clone(base).fit(group_values, y)
        if self.is_classifier_:
            self.classes_ = self.fallback_.classes_

        # Un Dummy por grupo; se guarda lo que predice cada uno
        keys, outputs = [], []
        for key, positions in X.groupby(columns, sort=False, observed=True).indices.items():
            model = clone(base).fit(group_values[positions], y[positions])
            keys.append(key if isinstance(key, tuple) else (key,))
            outputs.append(self._output(model, group_values[positions[:1]]))

        index = pd.MultiIndex.from_tuples(keys, names=columns)
        if self.is_classifier_:
            self.lookup_ = pd.DataFrame(np.vstack(outputs), index=index, columns=self.classes_)
        else:
            self.lookup_ = pd.Series(np.concatenate(outputs), index=index, dtype=float)
        self.n_groups_ = len(keys)
        return self

    def _output(self, model, row: np.ndarray) -> np.ndarray:
        """Lo que predice un Dummy entrenado: un valor (regresión) o probabilidades (clasificación)."""
        if not self.is_classifier_:
            return model.predict(row)[:1].astype(float)
        probabilities = np.zeros(len(self.classes_))
        positions = np.searchsorted(self.classes_, model.classes_)
        probabilities[positions] = model.predict_proba(row)[0]
        return probabilities

    def _lookup(self, X: pd.DataFrame):
        keys = pd.MultiIndex.from_frame(X[list(self.group_columns)])
        return self.lookup_.reindex(keys)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice con el Dummy del grupo de cada fila (o con el general si el grupo es nuevo).

        Args:
            X: Tabla con las columnas de ``group_columns``.

        Returns:
            Un valor (regresión) o una clase (clasificación) por fila.
        """
        if self.is_classifier_:
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        values = self._lookup(X).to_numpy(dtype=float)
        missing = np.isnan(values)
        if missing.any():
            row = X[list(self.group_columns)].iloc[:1].to_numpy()
            values[missing] = self.fallback_.predict(row)[0]
        return values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidad de cada clase por fila (solo clasificación).

        Args:
            X: Tabla con las columnas de ``group_columns``.

        Returns:
            Matriz de filas × clases, en el orden de ``classes_``.

        Raises:
            AttributeError: Si el modelo es de regresión.
        """
        if not self.is_classifier_:
            raise AttributeError("predict_proba solo está disponible para clasificadores")
        probabilities = self._lookup(X).to_numpy(dtype=float)
        missing = np.isnan(probabilities).any(axis=1)
        if missing.any():
            row = X[list(self.group_columns)].iloc[:1].to_numpy()
            probabilities[missing] = self.fallback_.predict_proba(row)[0]
        return probabilities


def build_p1_baseline() -> GroupedDummy:
    """Baseline de P1: promedio de viajes de cada estación a cada hora del día.

    Returns:
        Un ``DummyRegressor(strategy="mean")`` por cada combinación de estación y hora.
    """
    return GroupedDummy(
        DummyRegressor(strategy="mean"), group_columns=("station_id", "hour_of_day")
    )


def build_p2_baseline() -> GroupedDummy:
    """Baseline de P2: frecuencia de cada clase de riesgo por estación, intervalo y tipo de día.

    Returns:
        Un ``DummyClassifier(strategy="prior")`` por cada combinación de estación,
        intervalo del día y fin de semana. Predice la clase más frecuente del grupo y, como
        probabilidad, la proporción de cada clase en el grupo.
    """
    return GroupedDummy(
        DummyClassifier(strategy="prior"),
        group_columns=("station_id", "interval_of_day", "is_weekend"),
    )


def build_p3_baseline(n_clusters: int) -> Pipeline:
    """Baseline de P3: estandarización y K-Means.

    Las variables se estandarizan para que ninguna domine la distancia solo por su escala.

    Args:
        n_clusters: Número de grupos.

    Returns:
        ``StandardScaler`` seguido de ``KMeans`` con la semilla de ``config.py``.
    """
    return make_pipeline(
        StandardScaler(), KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED)
    )
