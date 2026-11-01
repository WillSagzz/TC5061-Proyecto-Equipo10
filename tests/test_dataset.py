"""Pruebas de ``dataset.py``: limpieza de los viajes (Etapa 1 del pipeline)."""

import pandas as pd
import pytest

from mibici.dataset import clean_trips

# Nomenclatura mínima: ``clean_trips`` solo usa la columna ``id``
STATIONS = pd.DataFrame({"id": [1, 2]})


def _trip(
    trip_id: int, start: str | None, minutes: float, origin: int = 1, destination: int = 2
) -> dict:
    """Arma un viaje con las columnas de ``TRIP_COLUMNS``.

    Args:
        trip_id: Identificador del viaje.
        start: Fecha y hora de salida; ``None`` simula una fecha que no se pudo leer.
        minutes: Duración del viaje en minutos.
        origin: Estación de salida.
        destination: Estación de llegada.

    Returns:
        Diccionario con una fila de viajes.
    """
    start_time = pd.Timestamp(start)
    return {
        "viaje_id": trip_id,
        "inicio_del_viaje": start_time,
        "fin_del_viaje": start_time + pd.Timedelta(minutes=minutes),
        "origen_id": origin,
        "destino_id": destination,
    }


def test_clean_trips_keeps_valid_trips() -> None:
    """Caso normal: los viajes válidos se conservan, incluso los que están justo en el borde.

    - Salida a las 5:00: es la primera hora de servicio.
    - Duración de 24 h exactas: la limpieza solo quita los viajes que superan
      ``MAX_TRIP_MINUTES`` (24 h, criterio del equipo; MiBici no fija ese límite).
    - Viaje de 30 segundos entre estaciones distintas: solo se quitan los viajes cortos que
      regresan a la misma estación.
    """
    trips = pd.DataFrame(
        [
            _trip(1, "2025-03-03 08:00", minutes=15),
            _trip(2, "2025-03-03 05:00", minutes=15),
            _trip(3, "2025-03-03 08:00", minutes=24 * 60),
            _trip(4, "2025-03-03 08:00", minutes=0.5),
        ]
    )

    clean = clean_trips(trips, STATIONS)

    assert clean["viaje_id"].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "bad_trip",
    [
        pytest.param(_trip(2, None, minutes=15), id="invalid_date"),
        pytest.param(_trip(2, "2025-03-03 08:00", minutes=-5), id="negative_duration"),
        pytest.param(_trip(2, "2025-03-03 08:00", minutes=24 * 60 + 1), id="longer_than_24h"),
        pytest.param(
            _trip(2, "2025-03-03 08:00", minutes=0.5, destination=1),
            id="under_1_min_same_station",
        ),
        pytest.param(_trip(2, "2025-03-03 01:00", minutes=15), id="start_at_1_00"),
        pytest.param(_trip(2, "2025-03-03 04:59", minutes=15), id="start_at_4_59"),
        pytest.param(
            _trip(2, "2025-03-03 08:00", minutes=15, origin=405),
            id="origin_not_in_nomenclature",
        ),
        pytest.param(
            _trip(2, "2025-03-03 08:00", minutes=15, destination=405),
            id="destination_not_in_nomenclature",
        ),
        pytest.param(_trip(1, "2025-03-03 09:00", minutes=15), id="duplicated_trip_id"),
    ],
)
def test_clean_trips_removes_invalid_trips(bad_trip: dict) -> None:
    """Caso límite: de un viaje válido y uno inválido, solo queda el válido.

    Se corre una vez por cada regla de limpieza. El duplicado repite el ``viaje_id`` 1.
    """
    trips = pd.DataFrame([_trip(1, "2025-03-03 08:00", minutes=15), bad_trip])

    clean = clean_trips(trips, STATIONS)

    assert clean["viaje_id"].tolist() == [1]
