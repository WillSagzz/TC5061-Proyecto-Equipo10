"""Pruebas de la API (``api.py``) con modelos falsos, sin servidor de MLflow."""

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici import api
from mibici.config import REGISTERED_MODELS
from mibici.features import P1_FEATURES, P2_FEATURES, P3_FEATURES, RISK_CLASSES
from mibici.modeling.registry import LoadedModel

# Una petición válida por pregunta
VALID_REQUESTS = {
    # P1: estación 50, martes a las 8:00
    "p1": {
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
    },
    # P2: estación 50, martes en el intervalo de 6:00 a 8:59
    "p2": {
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
    },
    # P3: perfil de una estación con más uso en la mañana
    "p3": {
        "morning_pct": 35.0,
        "midday_pct": 25.0,
        "afternoon_pct": 28.0,
        "night_pct": 12.0,
        "weekdays_pct": 78.5,
        "log_daily_use_per_dock": 1.42,
        "balance_morning": -0.25,
        "balance_afternoon": 0.22,
    },
}


class _FakeModel:
    """Modelo falso: siempre predice lo mismo y recuerda las columnas que recibió."""

    def __init__(self, prediction: object, probabilities: list[float] | None = None) -> None:
        self.prediction = prediction
        self.probabilities = probabilities
        self.classes_ = np.array(RISK_CLASSES)  # orden de las probabilidades (solo P2)
        self.received_columns: list[str] = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.received_columns = list(X.columns)
        return np.array([self.prediction] * len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.probabilities] * len(X))


def _client(
    monkeypatch: pytest.MonkeyPatch, p1_prediction: float = 7.5
) -> tuple[TestClient, dict[str, _FakeModel]]:
    """Crea un cliente de prueba de la API que usa modelos falsos en lugar de los de MLflow.

    Los modelos falsos predicen: P1, ``p1_prediction`` viajes; P2, ``vaciarse`` con
    probabilidades 0.7, 0.1 y 0.2 (vaciarse, llenarse, sin_riesgo); P3, el grupo 2.

    Args:
        monkeypatch: Herramienta de PyTest para reemplazar funciones durante una prueba.
        p1_prediction: Número que devolverá el modelo falso de P1.

    Returns:
        El cliente de prueba y los modelos falsos por pregunta, para revisar qué recibieron.
    """
    fake_models = {
        "p1": _FakeModel(p1_prediction),
        "p2": _FakeModel("vaciarse", probabilities=[0.7, 0.1, 0.2]),
        "p3": _FakeModel(2),
    }
    loaded_models = {
        question: LoadedModel(
            question=question,
            model_name=REGISTERED_MODELS[question],
            model_uri=f"models:/{REGISTERED_MODELS[question]}@champion",
            alias="champion",
            version=None,
            source="registry",
            estimator_type="FakeModel",
            features=[],
            model=fake_model,
        )
        for question, fake_model in fake_models.items()
    }
    monkeypatch.setattr(api, "get_registered_model", lambda question: loaded_models[question])
    return TestClient(api.app), fake_models


def test_health_reports_the_three_loaded_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso normal: ``/health`` responde 200 e indica los modelos de P1, P2 y P3."""
    client, _ = _client(monkeypatch)

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    models = response.json()["models"]
    assert [model["question"] for model in models] == ["p1", "p2", "p3"]
    assert models[0]["model_name"] == "mibici-equipo10-p1-demanda"
    assert models[0]["alias"] == "champion"


def test_predict_p1_returns_the_model_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso normal: P1 responde 200 con los viajes pronosticados, la estación y la hora.

    También revisa que el modelo reciba las 10 variables en el orden de ``P1_FEATURES``, el
    mismo con el que se entrenó.
    """
    client, fake_models = _client(monkeypatch, p1_prediction=7.5)

    with client:
        response = client.post("/predict/p1", json=VALID_REQUESTS["p1"])

    assert response.status_code == 200
    body = response.json()
    assert body["departures_pred"] == 7.5
    assert body["station_id"] == 50
    assert body["hour_of_day"] == 8
    assert body["model_name"] == "mibici-equipo10-p1-demanda"
    assert fake_models["p1"].received_columns == P1_FEATURES


def test_predict_p2_returns_the_risk_and_its_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso normal: P2 responde 200 con la clase de riesgo y la probabilidad de cada clase."""
    client, fake_models = _client(monkeypatch)

    with client:
        response = client.post("/predict/p2", json=VALID_REQUESTS["p2"])

    assert response.status_code == 200
    body = response.json()
    assert body["risk_pred"] == "vaciarse"
    assert body["probabilities"] == {"vaciarse": 0.7, "llenarse": 0.1, "sin_riesgo": 0.2}
    assert body["interval_of_day"] == 2
    assert fake_models["p2"].received_columns == P2_FEATURES


def test_predict_p3_returns_the_station_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso normal: P3 responde 200 con el número de grupo de la estación."""
    client, fake_models = _client(monkeypatch)

    with client:
        response = client.post("/predict/p3", json=VALID_REQUESTS["p3"])

    assert response.status_code == 200
    assert response.json()["group"] == 2
    assert fake_models["p3"].received_columns == P3_FEATURES


@pytest.mark.parametrize(
    ("question", "changes"),
    [
        pytest.param("p1", {"hour_of_day": 3}, id="p1_no_service_hour"),
        pytest.param("p1", {"hour_of_day": 24}, id="p1_hour_out_of_range"),
        pytest.param("p1", {"is_weekend": 1}, id="p1_weekend_does_not_match_day"),
        pytest.param("p1", {"departures_lag_24": -3}, id="p1_negative_trips"),
        pytest.param("p1", {"capacity": "veinte"}, id="p1_text_instead_of_number"),
        pytest.param("p1", {"capacity": None}, id="p1_missing_variable"),
        pytest.param("p1", {"temperature": 25}, id="p1_extra_variable"),
        pytest.param("p2", {"interval_of_day": 8}, id="p2_interval_out_of_range"),
        pytest.param("p2", {"is_weekend": 1}, id="p2_weekend_does_not_match_day"),
        pytest.param("p2", {"capacity": 0}, id="p2_station_without_docks"),
        pytest.param("p3", {"night_pct": 40.0}, id="p3_percentages_do_not_sum_100"),
        pytest.param("p3", {"balance_morning": 1.5}, id="p3_balance_out_of_range"),
        pytest.param("p3", {"latitude": 20.67}, id="p3_extra_variable"),
    ],
)
def test_predict_rejects_invalid_requests(
    monkeypatch: pytest.MonkeyPatch, question: str, changes: dict
) -> None:
    """Caso límite: una petición inválida responde 422 y nunca llega al modelo.

    Cada caso copia la petición válida de la pregunta y le aplica un solo cambio; un valor
    ``None`` quita esa variable de la petición.
    """
    client, fake_models = _client(monkeypatch)
    bad_request = {**VALID_REQUESTS[question], **changes}
    bad_request = {key: value for key, value in bad_request.items() if value is not None}

    with client:
        response = client.post(f"/predict/{question}", json=bad_request)

    assert response.status_code == 422
    assert fake_models[question].received_columns == []


def test_predict_p1_never_returns_negative_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso límite: si el modelo de P1 predice un número negativo, la API responde 0 viajes."""
    client, _ = _client(monkeypatch, p1_prediction=-0.4)

    with client:
        response = client.post("/predict/p1", json=VALID_REQUESTS["p1"])

    assert response.status_code == 200
    assert response.json()["departures_pred"] == 0.0
