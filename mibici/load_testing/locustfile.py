"""Escenario de la prueba de carga de la API REST con Locust (Entrega 06).

Simula operadores del sistema MiBici que piden predicciones a la API levantada con
``make compose-up``. Cada usuario simulado elige un endpoint según los pesos de
``LOAD_TEST_TASK_WEIGHTS`` (sobre todo P1, la demanda por estación y hora), manda una petición
válida y distinta cada vez, y espera entre ``LOAD_TEST_WAIT_SECONDS`` antes de la siguiente.

La carga sube por escalones (``LOAD_TEST_STAGES``: 10, 25, 50, 100, 150 y 250 usuarios, un
minuto cada uno) para ver en qué punto la latencia empieza a crecer y aparecen errores.

Lo corre ``make load-test`` (``python -m mibici.load_testing.run``). Para verlo en vivo en el
navegador (http://localhost:8089):

    locust -f mibici/load_testing/locustfile.py --host http://127.0.0.1:8000
"""

from __future__ import annotations

from itertools import count

from locust import HttpUser, LoadTestShape, between, task
import numpy as np

from mibici.config import (
    LOAD_TEST_SPAWN_RATE,
    LOAD_TEST_STAGES,
    LOAD_TEST_TASK_WEIGHTS,
    LOAD_TEST_WAIT_SECONDS,
    RANDOM_SEED,
)
from mibici.load_testing.payloads import p1_payload, p2_payload, p3_payload, users_for

# Cada usuario recibe su propia semilla (42, 43, 44…): la prueba es variada pero repetible
_user_seeds = count(RANDOM_SEED)

# Campo que debe traer la respuesta de cada endpoint para contarla como exitosa
EXPECTED_FIELD = {"p1": "departures_pred", "p2": "risk_pred", "p3": "group"}


class StationOperator(HttpUser):
    """Operador que consulta las predicciones de la API una tras otra."""

    wait_time = between(*LOAD_TEST_WAIT_SECONDS)

    def on_start(self) -> None:
        """Crea el generador aleatorio propio de este usuario."""
        self.rng = np.random.default_rng(next(_user_seeds))

    def _predict(self, question: str, payload: dict) -> None:
        """Manda una predicción y la marca como fallida si la respuesta no es la esperada.

        Locust ya cuenta como error cualquier código distinto de 200; además se revisa que la
        respuesta traiga la predicción, para no contar como éxito una respuesta vacía.
        """
        with self.client.post(
            f"/predict/{question}", json=payload, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            elif EXPECTED_FIELD[question] not in response.json():
                response.failure(f"La respuesta no trae '{EXPECTED_FIELD[question]}'")

    @task(LOAD_TEST_TASK_WEIGHTS["p1"])
    def predict_p1(self) -> None:
        """Demanda de una estación en una hora (P1)."""
        self._predict("p1", p1_payload(self.rng))

    @task(LOAD_TEST_TASK_WEIGHTS["p2"])
    def predict_p2(self) -> None:
        """Riesgo de que una estación se vacíe o se llene (P2)."""
        self._predict("p2", p2_payload(self.rng))

    @task(LOAD_TEST_TASK_WEIGHTS["p3"])
    def predict_p3(self) -> None:
        """Grupo de uso de una estación (P3)."""
        self._predict("p3", p3_payload(self.rng))

    @task(LOAD_TEST_TASK_WEIGHTS["health"])
    def health(self) -> None:
        """Revisión de salud, como la que haría un monitor del servicio."""
        self.client.get("/health")


class StepLoad(LoadTestShape):
    """Carga por escalones: Locust la usa en lugar de ``--users`` y ``--run-time``."""

    def tick(self) -> tuple[int, float] | None:
        """Usuarios y velocidad de arranque del escalón en curso; ``None`` termina la prueba."""
        users = users_for(self.get_run_time(), LOAD_TEST_STAGES)
        return None if users is None else (users, LOAD_TEST_SPAWN_RATE)
