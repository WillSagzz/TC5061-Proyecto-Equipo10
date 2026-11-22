"""Peticiones y escalones de carga de la prueba con Locust.

Este módulo no importa Locust a propósito: Locust modifica la biblioteca estándar al importarse
(``gevent.monkey``), y eso afectaría a las demás pruebas de PyTest. Aquí vive la lógica que se
puede probar sola, y ``locustfile.py`` solo la usa.

Cada petición es distinta (estación, hora, rezagos…) para que la API no responda siempre lo
mismo, pero todas cumplen las reglas de ``mibici/schemas.py``: si una petición fallara por datos
inválidos, la prueba de carga mediría errores de validación en lugar del desempeño del sistema.
"""

from __future__ import annotations

import numpy as np

from mibici.config import NO_SERVICE_HOURS, P2_INTERVAL_HOURS

# Rangos plausibles de MiBici (no salen de los datos: solo dan variedad a las peticiones)
STATION_IDS = (1, 520)  # identificadores de estación, ambos incluidos
CAPACITIES = (12, 15, 16, 19, 20, 24, 30)  # anclajes por estación
SERVICE_HOURS = tuple(hour for hour in range(24) if hour not in NO_SERVICE_HOURS)
P2_INTERVALS = 24 // P2_INTERVAL_HOURS  # 8 intervalos de 3 horas por día
WEEKEND_FIRST_DAY = 5  # 0 = lunes ... 5 = sábado, 6 = domingo


def _calendar(rng: np.random.Generator) -> dict[str, int]:
    """Día de la semana y su indicador de fin de semana, siempre coherentes entre sí."""
    day = int(rng.integers(0, 7))
    return {"day_of_week": day, "is_weekend": int(day >= WEEKEND_FIRST_DAY)}


def _station(rng: np.random.Generator) -> dict[str, int]:
    """Estación y capacidad al azar dentro de los rangos de MiBici."""
    return {
        "station_id": int(rng.integers(STATION_IDS[0], STATION_IDS[1] + 1)),
        "capacity": int(rng.choice(CAPACITIES)),
    }


def _count(rng: np.random.Generator, mean: float) -> float:
    """Conteo de viajes (salidas o llegadas) con distribución de Poisson."""
    return float(rng.poisson(mean))


def p1_payload(rng: np.random.Generator) -> dict[str, int | float]:
    """Petición válida para ``POST /predict/p1`` (salidas de una estación en una hora).

    Args:
        rng: Generador aleatorio del usuario simulado.

    Returns:
        Cuerpo JSON con las 10 variables de ``P1FeaturesInput``.
    """
    departures_24, arrivals_24 = _count(rng, 4), _count(rng, 4)
    return {
        **_station(rng),
        "hour_of_day": int(rng.choice(SERVICE_HOURS)),
        **_calendar(rng),
        "departures_lag_24": departures_24,
        "departures_lag_168": _count(rng, 4),
        "arrivals_lag_24": arrivals_24,
        "net_flow_lag_24": arrivals_24 - departures_24,
        "departures_mean_3_lag_24": round(float(rng.gamma(2.0, 2.0)), 2),
    }


def p2_payload(rng: np.random.Generator) -> dict[str, int | float]:
    """Petición válida para ``POST /predict/p2`` (riesgo de una estación en un intervalo).

    Args:
        rng: Generador aleatorio del usuario simulado.

    Returns:
        Cuerpo JSON con las 13 variables de ``P2FeaturesInput``.
    """
    departures_1, arrivals_1 = _count(rng, 10), _count(rng, 10)
    departures_8, arrivals_8 = _count(rng, 10), _count(rng, 10)
    return {
        **_station(rng),
        "interval_of_day": int(rng.integers(0, P2_INTERVALS)),
        **_calendar(rng),
        "departures_lag_1": departures_1,
        "departures_lag_8": departures_8,
        "departures_lag_56": _count(rng, 10),
        "arrivals_lag_1": arrivals_1,
        "arrivals_lag_8": arrivals_8,
        "net_flow_lag_1": arrivals_1 - departures_1,
        "net_flow_lag_8": arrivals_8 - departures_8,
        "departures_mean_3": round(float(rng.gamma(4.0, 2.5)), 2),
    }


def p3_payload(rng: np.random.Generator) -> dict[str, float]:
    """Petición válida para ``POST /predict/p3`` (perfil de uso de una estación).

    Los cuatro porcentajes del día se reparten con una distribución de Dirichlet para que sumen
    100 %, como exige ``P3FeaturesInput``.

    Args:
        rng: Generador aleatorio del usuario simulado.

    Returns:
        Cuerpo JSON con las 8 variables de ``P3FeaturesInput``.
    """
    shares = rng.dirichlet([3.0, 2.0, 2.5, 1.5]) * 100
    morning, midday, afternoon = (round(float(share), 2) for share in shares[:3])
    return {
        "morning_pct": morning,
        "midday_pct": midday,
        "afternoon_pct": afternoon,
        "night_pct": max(0.0, round(100 - morning - midday - afternoon, 2)),
        "weekdays_pct": round(float(rng.uniform(55, 90)), 2),
        "log_daily_use_per_dock": round(float(rng.uniform(0.1, 2.5)), 3),
        "balance_morning": round(float(rng.uniform(-0.6, 0.6)), 3),
        "balance_afternoon": round(float(rng.uniform(-0.6, 0.6)), 3),
    }


def users_for(run_time: float, stages: tuple[tuple[int, int], ...]) -> int | None:
    """Usuarios simultáneos que debe haber en cada momento de la prueba (carga escalonada).

    Args:
        run_time: Segundos transcurridos desde el inicio de la prueba.
        stages: Escalones ``(segundo en que termina, usuarios)`` en orden.

    Returns:
        Usuarios del escalón en curso, o ``None`` cuando ya terminó el último (fin de la prueba).
    """
    for end, users in stages:
        if run_time < end:
            return users
    return None
