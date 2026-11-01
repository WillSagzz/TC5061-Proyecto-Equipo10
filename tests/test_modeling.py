"""Prueba de reproducibilidad del entrenamiento."""

import numpy as np

from mibici.features import P1_FEATURES, P1_TARGET, build_p1_table, split_by_date
from mibici.modeling.candidates import get_spec, resolve_hyperparams
from mibici.validation import build_synthetic_raw_data


def test_training_is_reproducible_with_the_same_seed() -> None:
    """Reproducibilidad: entrenar dos veces el modelo elegido de P1 da las mismas predicciones.

    Usa los datos sintéticos de ``validation.py`` y la semilla de ``config.py``, igual que las
    corridas de MLflow. ``max_iter=30`` solo hace la prueba más rápida.
    """
    splits = split_by_date(build_p1_table(*build_synthetic_raw_data()), "period_start")
    X_train, y_train = splits["train"][P1_FEATURES], splits["train"][P1_TARGET]
    X_val = splits["val"][P1_FEATURES]
    spec = get_spec("p1", "hgb_poisson")
    hyperparams = resolve_hyperparams(spec, {"max_iter": "30"})

    first = spec.build(hyperparams).fit(X_train, y_train).predict(X_val)
    second = spec.build(hyperparams).fit(X_train, y_train).predict(X_val)

    np.testing.assert_array_equal(first, second)
