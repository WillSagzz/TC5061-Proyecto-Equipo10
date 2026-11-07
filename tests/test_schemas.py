"""Pruebas unitarias para los esquemas Pydantic de entrada y salida (schemas.py).

Verifica que los contratos de P1, P2 y P3 conviertan las entradas exactamente a las columnas
de ``P1_FEATURES``, ``P2_FEATURES`` y ``P3_FEATURES``, rechacen valores fuera de dominio
(horas sin servicio, incoherencia de fin de semana, porcentajes que no suman 100 % o campos
extra) y estructuren las salidas de predicción sin errores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from mibici.features import P1_FEATURES, P2_FEATURES, P3_FEATURES, RISK_CLASSES
from mibici.modeling.candidates import get_spec
from mibici.schemas import (
    P1FeaturesInput,
    P2FeaturesInput,
    P2PredictionOutput,
    P3FeaturesInput,
    build_p1_output,
    build_p2_output,
    build_p3_output,
)


def _valid_p1_payload() -> dict[str, int | float]:
    return {
        "station_id": 50,
        "hour_of_day": 8,
        "day_of_week": 1,
        "is_weekend": 0,
        "capacity": 20,
        "departures_lag_24": 14.0,
        "departures_lag_168": 12.0,
        "arrivals_lag_24": 10.0,
        "net_flow_lag_24": -4.0,
        "departures_mean_3_lag_24": 11.33,
    }


def _valid_p2_payload() -> dict[str, int | float]:
    return {
        "station_id": 50,
        "interval_of_day": 2,
        "day_of_week": 5,
        "is_weekend": 1,
        "capacity": 20,
        "departures_lag_1": 8.0,
        "departures_lag_8": 25.0,
        "departures_lag_56": 22.0,
        "arrivals_lag_1": 5.0,
        "arrivals_lag_8": 14.0,
        "net_flow_lag_1": -3.0,
        "net_flow_lag_8": -11.0,
        "departures_mean_3": 9.67,
    }


def _valid_p3_payload() -> dict[str, float]:
    return {
        "morning_pct": 35.0,
        "midday_pct": 25.0,
        "afternoon_pct": 28.0,
        "night_pct": 12.0,
        "weekdays_pct": 78.5,
        "log_daily_use_per_dock": 1.42,
        "balance_morning": -0.25,
        "balance_afternoon": 0.22,
    }


def test_p1_input_to_dataframe_matches_p1_features_and_runs_model() -> None:
    """Caso normal P1: convierte al orden exacto de P1_FEATURES y alimenta al modelo."""
    schema = P1FeaturesInput(**_valid_p1_payload())
    df = schema.to_dataframe()
    assert list(df.columns) == P1_FEATURES
    assert len(df) == 1

    spec = get_spec("p1", "hgb_poisson")
    model = spec.build(spec.defaults)
    X_train = pd.concat([df] * 10, ignore_index=True)
    y_train = np.linspace(1.0, 10.0, 10)
    model.fit(X_train, y_train)

    pred = float(model.predict(df)[0])
    output = build_p1_output(
        schema,
        pred,
        model_name="mibici-equipo10-p1-demanda",
        model_uri="models:/mibici-equipo10-p1-demanda@champion",
        estimator_type="HistGradientBoostingRegressor",
    )
    assert output.question == "p1"
    assert output.station_id == 50
    assert output.departures_pred >= 0.0


@pytest.mark.parametrize("invalid_hour", [1, 2, 3, 4, -1, 24])
def test_p1_input_rejects_no_service_and_out_of_range_hours(invalid_hour: int) -> None:
    """Caso límite P1: rechaza las horas sin servicio (1 a 4) o fuera de 0..23."""
    payload = _valid_p1_payload() | {"hour_of_day": invalid_hour}
    with pytest.raises(ValidationError):
        P1FeaturesInput(**payload)


def test_p1_input_rejects_weekend_inconsistency_and_extra_fields() -> None:
    """Caso límite P1: rechaza contradicción entre day_of_week e is_weekend o campos extra."""
    inconsistent = _valid_p1_payload() | {"day_of_week": 6, "is_weekend": 0}
    with pytest.raises(ValidationError, match="Inconsistencia de calendario"):
        P1FeaturesInput(**inconsistent)

    with_extra = _valid_p1_payload() | {"departures": 99}
    with pytest.raises(ValidationError):
        P1FeaturesInput(**with_extra)


def test_p2_input_to_dataframe_matches_p2_features_and_builds_output() -> None:
    """Caso normal P2: convierte al orden exacto de P2_FEATURES y valida probabilidades."""
    schema = P2FeaturesInput(**_valid_p2_payload())
    df = schema.to_dataframe()
    assert list(df.columns) == P2_FEATURES
    assert len(df) == 1

    output = build_p2_output(
        schema,
        raw_label="sin_riesgo",
        raw_probabilities=[0.05, 0.10, 0.85],
        classes=RISK_CLASSES,
        model_name="mibici-equipo10-p2-riesgo",
        model_uri="models:/mibici-equipo10-p2-riesgo@champion",
        estimator_type="ThresholdClassifier(HistGradientBoostingClassifier)",
    )
    assert output.question == "p2"
    assert output.risk_pred == "sin_riesgo"
    assert set(output.probabilities) == set(RISK_CLASSES)


def test_p2_rejects_invalid_interval_and_invalid_probabilities() -> None:
    """Caso límite P2: rechaza intervalos fuera de 0..7 y probabilidades fuera de [0, 1]."""
    bad_interval = _valid_p2_payload() | {"interval_of_day": 8}
    with pytest.raises(ValidationError):
        P2FeaturesInput(**bad_interval)

    with pytest.raises(ValidationError, match="Faltan probabilidades"):
        P2PredictionOutput(
            station_id=50,
            interval_of_day=2,
            risk_pred="vaciarse",
            probabilities={"vaciarse": 0.9},
            model_name="m",
            model_uri="u",
            estimator_type="e",
        )


def test_p3_input_to_dataframe_matches_p3_features_and_builds_output() -> None:
    """Caso normal P3: convierte al orden exacto de P3_FEATURES y construye la salida."""
    schema = P3FeaturesInput(**_valid_p3_payload())
    df = schema.to_dataframe()
    assert list(df.columns) == P3_FEATURES
    assert len(df) == 1

    output = build_p3_output(
        raw_group=2,
        model_name="mibici-equipo10-p3-grupos",
        model_uri="models:/mibici-equipo10-p3-grupos@champion",
        estimator_type="Pipeline",
    )
    assert output.question == "p3"
    assert output.group == 2


def test_p3_input_rejects_percentages_not_summing_100_or_invalid_balance() -> None:
    """Caso límite P3: rechaza porcentajes que no suman 100 % o balances fuera de [-1, 1]."""
    bad_sum = _valid_p3_payload() | {"night_pct": 30.0}
    with pytest.raises(ValidationError, match="sumar 100 %"):
        P3FeaturesInput(**bad_sum)

    bad_balance = _valid_p3_payload() | {"balance_morning": -1.5}
    with pytest.raises(ValidationError):
        P3FeaturesInput(**bad_balance)
