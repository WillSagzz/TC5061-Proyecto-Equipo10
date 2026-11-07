"""Esquemas Pydantic de validación de entradas y salidas para la API REST.

Define los contratos de datos de entrada y respuesta para los tres modelos registrados en el
Model Registry de MLflow (P1 regresión de demanda, P2 clasificación de riesgo y P3 clustering
de perfiles de estaciones), garantizando que las variables, tipos y restricciones de dominio
coincidan exactamente con ``mibici/features.py`` y ``mibici/config.py``.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mibici.config import NO_SERVICE_HOURS, P2_INTERVAL_HOURS
from mibici.features import (
    P1_FEATURES,
    P2_FEATURES,
    P3_FEATURES,
    RISK_CLASSES,
)

QuestionLiteral = Literal["p1", "p2", "p3"]
RiskClassLiteral = Literal["vaciarse", "llenarse", "sin_riesgo"]

MAX_P2_INTERVAL = (24 // P2_INTERVAL_HOURS) - 1
P3_PCT_TOLERANCE = 0.5


def _validate_weekend_coherence(day_of_week: int, is_weekend: int) -> None:
    """Verifica que ``is_weekend`` coincida con ``day_of_week`` (0=lunes ... 6=domingo)."""
    expected = 1 if day_of_week >= 5 else 0
    if is_weekend != expected:
        raise ValueError(
            f"Inconsistencia de calendario: para day_of_week={day_of_week}, "
            f"is_weekend debe ser {expected} (1 en sábado/domingo, 0 de lunes a viernes)."
        )


class P1FeaturesInput(BaseModel):
    """Variables predictoras de entrada para P1 (demanda por estación y hora).

    Corresponde exactamente a las 10 columnas de ``P1_FEATURES`` en ``mibici/features.py``.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )

    station_id: int = Field(
        ..., ge=1, description="Identificador de la estación en el catálogo de MiBici."
    )
    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hora del día a pronosticar (0 a 23; sin servicio de 1:00 a 4:59).",
    )
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Día de la semana (0 = lunes, 6 = domingo)."
    )
    is_weekend: int = Field(
        ..., ge=0, le=1, description="Indicador de fin de semana (1 = sábado o domingo)."
    )
    capacity: int = Field(
        ..., ge=1, description="Capacidad de anclajes de la estación (dpcapacity > 0)."
    )
    departures_lag_24: float = Field(
        ...,
        ge=0.0,
        allow_inf_nan=False,
        description="Salidas registradas 24 horas antes en la misma estación.",
    )
    departures_lag_168: float = Field(
        ...,
        ge=0.0,
        allow_inf_nan=False,
        description="Salidas registradas 168 horas (7 días) antes en la misma estación.",
    )
    arrivals_lag_24: float = Field(
        ...,
        ge=0.0,
        allow_inf_nan=False,
        description="Llegadas registradas 24 horas antes en la misma estación.",
    )
    net_flow_lag_24: float = Field(
        ...,
        allow_inf_nan=False,
        description="Flujo neto (llegadas − salidas) registrado 24 horas antes.",
    )
    departures_mean_3_lag_24: float = Field(
        ...,
        ge=0.0,
        allow_inf_nan=False,
        description="Promedio móvil de salidas de las 3 horas cerradas 24 horas antes.",
    )

    @field_validator("hour_of_day")
    @classmethod
    def _reject_no_service_hours(cls, value: int) -> int:
        """Rechaza las horas de 1:00 a 4:59 en que MiBici no presta servicio."""
        if value in NO_SERVICE_HOURS:
            raise ValueError(
                f"La hora {value}:00 pertenece al horario sin servicio ({NO_SERVICE_HOURS}); "
                "P1 solo pronostica horas de servicio activas."
            )
        return value

    @model_validator(mode="after")
    def _check_calendar_consistency(self) -> P1FeaturesInput:
        """Valida que el indicador de fin de semana corresponda al día de la semana."""
        _validate_weekend_coherence(self.day_of_week, self.is_weekend)
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte el registro validado en un DataFrame con el orden de ``P1_FEATURES``."""
        payload = self.model_dump()
        return pd.DataFrame([[payload[col] for col in P1_FEATURES]], columns=P1_FEATURES)


class P2FeaturesInput(BaseModel):
    """Variables predictoras de entrada para P2 (riesgo por estación e intervalo de 3 h).

    Corresponde exactamente a las 13 columnas de ``P2_FEATURES`` en ``mibici/features.py``.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "station_id": 50,
                    "interval_of_day": 2,
                    "day_of_week": 1,
                    "is_weekend": 0,
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
            ]
        },
    )

    station_id: int = Field(
        ..., ge=1, description="Identificador de la estación en el catálogo de MiBici."
    )
    interval_of_day: int = Field(
        ...,
        ge=0,
        le=MAX_P2_INTERVAL,
        description="Bloque de 3 horas del día (0 = 00:00-02:59 ... 7 = 21:00-23:59).",
    )
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Día de la semana (0 = lunes, 6 = domingo)."
    )
    is_weekend: int = Field(
        ..., ge=0, le=1, description="Indicador de fin de semana (1 = sábado o domingo)."
    )
    capacity: int = Field(
        ..., ge=1, description="Capacidad de anclajes de la estación (mayor a 0)."
    )
    departures_lag_1: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Salidas 1 intervalo (3 h) antes."
    )
    departures_lag_8: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Salidas 8 intervalos (24 h) antes."
    )
    departures_lag_56: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Salidas 56 intervalos (7 días) antes."
    )
    arrivals_lag_1: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Llegadas 1 intervalo (3 h) antes."
    )
    arrivals_lag_8: float = Field(
        ..., ge=0.0, allow_inf_nan=False, description="Llegadas 8 intervalos (24 h) antes."
    )
    net_flow_lag_1: float = Field(
        ..., allow_inf_nan=False, description="Flujo neto 1 intervalo (3 h) antes."
    )
    net_flow_lag_8: float = Field(
        ..., allow_inf_nan=False, description="Flujo neto 8 intervalos (24 h) antes."
    )
    departures_mean_3: float = Field(
        ...,
        ge=0.0,
        allow_inf_nan=False,
        description="Promedio móvil de salidas de los 3 intervalos cerrados previos.",
    )

    @model_validator(mode="after")
    def _check_calendar_consistency(self) -> P2FeaturesInput:
        """Valida que el indicador de fin de semana corresponda al día de la semana."""
        _validate_weekend_coherence(self.day_of_week, self.is_weekend)
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte el registro validado en un DataFrame con el orden de ``P2_FEATURES``."""
        payload = self.model_dump()
        return pd.DataFrame([[payload[col] for col in P2_FEATURES]], columns=P2_FEATURES)


class P3FeaturesInput(BaseModel):
    """Variables de perfil funcional de estación para P3 (clustering de estaciones).

    Corresponde exactamente a las 8 columnas de ``P3_FEATURES`` en ``mibici/features.py``.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "morning_pct": 35.0,
                    "midday_pct": 25.0,
                    "afternoon_pct": 28.0,
                    "night_pct": 12.0,
                    "weekdays_pct": 78.5,
                    "log_daily_use_per_dock": 1.42,
                    "balance_morning": -0.25,
                    "balance_afternoon": 0.22,
                }
            ]
        },
    )

    morning_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Porcentaje de movimientos en la mañana (5:00 a 11:59).",
    )
    midday_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Porcentaje de movimientos al mediodía (12:00 a 15:59).",
    )
    afternoon_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Porcentaje de movimientos en la tarde (16:00 a 19:59).",
    )
    night_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Porcentaje de movimientos en la noche (20:00 a 4:59).",
    )
    weekdays_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Porcentaje de movimientos en días laborales (lunes a viernes).",
    )
    log_daily_use_per_dock: float = Field(
        ...,
        ge=0.0,
        allow_inf_nan=False,
        description="Rotación diaria por anclaje: log(1 + uso_diario / capacidad).",
    )
    balance_morning: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
        description="Balance relativo (llegadas − salidas) / total en la mañana (-1 a 1).",
    )
    balance_afternoon: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
        description="Balance relativo (llegadas − salidas) / total en la tarde (-1 a 1).",
    )

    @model_validator(mode="after")
    def _check_period_percentages_sum_100(self) -> P3FeaturesInput:
        """Verifica que los cuatro periodos del día sumen 100 % (±0.5 % por redondeo)."""
        total = self.morning_pct + self.midday_pct + self.afternoon_pct + self.night_pct
        if abs(total - 100.0) > P3_PCT_TOLERANCE:
            raise ValueError(
                f"Los porcentajes por periodo del día deben sumar 100 % (±{P3_PCT_TOLERANCE} %), "
                f"pero suman {total:.2f} %."
            )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte el perfil validado en un DataFrame con el orden de ``P3_FEATURES``."""
        payload = self.model_dump()
        return pd.DataFrame([[payload[col] for col in P3_FEATURES]], columns=P3_FEATURES)


class P1PredictionOutput(BaseModel):
    """Esquema de respuesta JSON para una predicción de demanda de P1."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    question: Literal["p1"] = "p1"
    station_id: int = Field(..., ge=1)
    hour_of_day: int = Field(..., ge=0, le=23)
    departures_pred: float = Field(
        ...,
        ge=0.0,
        allow_inf_nan=False,
        description="Viajes iniciados (salidas) pronosticados para la estación y hora.",
    )
    model_name: str
    model_uri: str
    estimator_type: str


class P2PredictionOutput(BaseModel):
    """Esquema de respuesta JSON para una predicción de clase de riesgo de P2."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    question: Literal["p2"] = "p2"
    station_id: int = Field(..., ge=1)
    interval_of_day: int = Field(..., ge=0, le=MAX_P2_INTERVAL)
    risk_pred: RiskClassLiteral = Field(
        ..., description="Clase de riesgo prevista: vaciarse, llenarse o sin_riesgo."
    )
    probabilities: dict[str, float] = Field(
        ..., description="Probabilidad estimada por el clasificador para cada clase de riesgo."
    )
    model_name: str
    model_uri: str
    estimator_type: str

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        """Comprueba que estén presentes las tres clases de ``RISK_CLASSES`` en [0, 1]."""
        missing = [c for c in RISK_CLASSES if c not in value]
        if missing:
            raise ValueError(f"Faltan probabilidades para las clases: {missing}.")
        for cls_name, prob in value.items():
            if prob < 0.0 or prob > 1.0:
                raise ValueError(
                    f"La probabilidad de '{cls_name}' debe estar entre 0 y 1 (recibido: {prob})."
                )
        return value


class P3PredictionOutput(BaseModel):
    """Esquema de respuesta JSON para la asignación de grupo funcional de P3."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    question: Literal["p3"] = "p3"
    group: int = Field(
        ..., ge=0, description="Identificador del grupo o perfil funcional asignado."
    )
    model_name: str
    model_uri: str
    estimator_type: str


def build_p1_output(
    features: P1FeaturesInput,
    raw_prediction: float,
    model_name: str,
    model_uri: str,
    estimator_type: str,
) -> P1PredictionOutput:
    """Construye y valida la respuesta de P1 asegurando conteos no negativos."""
    clipped = max(0.0, float(raw_prediction))
    return P1PredictionOutput(
        station_id=features.station_id,
        hour_of_day=features.hour_of_day,
        departures_pred=round(clipped, 4),
        model_name=model_name,
        model_uri=model_uri,
        estimator_type=estimator_type,
    )


def build_p2_output(
    features: P2FeaturesInput,
    raw_label: Any,
    raw_probabilities: Any,
    classes: Any,
    model_name: str,
    model_uri: str,
    estimator_type: str,
) -> P2PredictionOutput:
    """Construye y valida la respuesta de P2 con sus probabilidades por clase."""
    probs_dict = {
        str(cls_name): round(float(prob), 6)
        for cls_name, prob in zip(classes, raw_probabilities, strict=True)
    }
    return P2PredictionOutput(
        station_id=features.station_id,
        interval_of_day=features.interval_of_day,
        risk_pred=str(raw_label),
        probabilities=probs_dict,
        model_name=model_name,
        model_uri=model_uri,
        estimator_type=estimator_type,
    )


def build_p3_output(
    raw_group: Any,
    model_name: str,
    model_uri: str,
    estimator_type: str,
) -> P3PredictionOutput:
    """Construye y valida la respuesta de P3 con el identificador entero del grupo."""
    return P3PredictionOutput(
        group=int(raw_group),
        model_name=model_name,
        model_uri=model_uri,
        estimator_type=estimator_type,
    )
