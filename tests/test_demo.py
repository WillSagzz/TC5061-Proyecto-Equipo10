"""Pruebas de la demo del video (``mibici/demo.py``).

No necesitan la API ni Docker: revisan que las peticiones de la demo sean las que la API acepta,
que la petición inválida sea inválida de verdad y que el comando ``curl`` que se muestra en
pantalla reproduzca la petición.
"""

import json

from pydantic import ValidationError
import pytest

from mibici import demo
from mibici.config import NO_SERVICE_HOURS
from mibici.schemas import P1FeaturesInput, P2FeaturesInput, P3FeaturesInput


@pytest.mark.parametrize("schema", [P1FeaturesInput, P2FeaturesInput, P3FeaturesInput])
def test_example_payload_is_accepted_by_its_schema(schema) -> None:
    """Caso normal: el ejemplo de cada esquema (el de /docs) pasa su propia validación."""
    schema(**demo.example_payload(schema))


def test_example_payload_is_a_copy() -> None:
    """Caso límite: modificar el ejemplo devuelto no altera el que publica el esquema."""
    payload = demo.example_payload(P1FeaturesInput)
    payload["station_id"] = 999
    assert demo.example_payload(P1FeaturesInput)["station_id"] != 999


def test_invalid_payload_is_rejected_by_the_schema() -> None:
    """La petición inválida de la demo usa una hora sin servicio y el esquema la rechaza."""
    payload = demo.invalid_p1_payload()
    assert payload["hour_of_day"] in NO_SERVICE_HOURS
    with pytest.raises(ValidationError):
        P1FeaturesInput(**payload)


def test_curl_command_reproduces_the_request() -> None:
    """El comando curl lleva el método, la URL, el encabezado JSON y el mismo cuerpo."""
    payload = demo.example_payload(P2FeaturesInput)
    command = demo.curl_command("POST", "http://127.0.0.1:8000/predict/p2", payload)
    assert command.startswith("curl -X POST http://127.0.0.1:8000/predict/p2 ")
    assert '"Content-Type: application/json"' in command
    body = command.split("-d '", 1)[1].rstrip("'")
    assert json.loads(body) == payload
    assert demo.curl_command("GET", "http://127.0.0.1:8000/health") == (
        "curl http://127.0.0.1:8000/health"
    )


def test_compose_status_without_compose(monkeypatch) -> None:
    """Caso límite: sin Docker ni Podman la demo avisa en lugar de fallar."""

    def missing(*args, **kwargs):
        raise OSError("no existe")

    monkeypatch.setattr(demo.subprocess, "run", missing)
    assert demo.compose_status().startswith("No se pudo consultar Compose")
