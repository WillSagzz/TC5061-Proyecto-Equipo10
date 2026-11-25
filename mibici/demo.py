"""Demo del sistema para el video de la entrega final: la API respondiendo a peticiones reales.

Con el sistema levantado (``make compose-up``), desde la raíz del repositorio:

    python -m mibici.demo             # make demo: pausa entre pasos (Enter para seguir)
    python -m mibici.demo --no-pause  # sin pausas, para comprobar que todo responde

Recorre, en orden: el estado de los contenedores de Compose, ``/health``, una predicción de P1,
P2 y P3 con los ejemplos de ``mibici/schemas.py`` (los mismos que muestra ``/docs``), una
petición inválida que la API rechaza con 422 y los modelos registrados en el MLflow de Compose
con sus alias. De cada petición imprime el comando ``curl`` equivalente y la respuesta, para que
quien ve el video pueda repetirla.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
import json
import subprocess  # nosec B404
import sys
from typing import Annotated

import mlflow
from mlflow.exceptions import MlflowException
from pydantic import BaseModel
import requests
import typer

from mibici.config import (
    DEMO_TIMEOUT_SECONDS,
    LOAD_TEST_HOST,
    LOAD_TEST_MLFLOW_URI,
    NO_SERVICE_HOURS,
    PROJ_ROOT,
    REGISTERED_MODELS,
)
from mibici.schemas import P1FeaturesInput, P2FeaturesInput, P3FeaturesInput

app = typer.Typer(help="Demo de la API y el Model Registry levantados con Compose.")

COMPOSE_COMMANDS = (["docker", "compose", "ps"], ["podman-compose", "ps"])


# ---------------------------------------------------------------------------------------------
# Peticiones
# ---------------------------------------------------------------------------------------------
def example_payload(schema: type[BaseModel]) -> dict:
    """Devuelve el ejemplo que el esquema publica en ``/docs``.

    Args:
        schema: Esquema de entrada de ``mibici/schemas.py`` con ``json_schema_extra``.

    Returns:
        Copia del primer ejemplo, lista para enviarse como JSON.
    """
    return copy.deepcopy(schema.model_config["json_schema_extra"]["examples"][0])


def invalid_p1_payload() -> dict:
    """El ejemplo de P1 con una hora sin servicio, que la API debe rechazar con 422."""
    payload = example_payload(P1FeaturesInput)
    payload["hour_of_day"] = NO_SERVICE_HOURS[-1]
    return payload


def curl_command(method: str, url: str, payload: dict | None = None) -> str:
    """Escribe el comando ``curl`` equivalente a una petición, para mostrarlo en pantalla.

    Args:
        method: ``GET`` o ``POST``.
        url: URL completa del endpoint.
        payload: Cuerpo JSON de la petición, si la tiene.

    Returns:
        Comando de una sola línea, válido en bash y Git Bash.
    """
    if payload is None:
        return f"curl {url}"
    body = json.dumps(payload, ensure_ascii=False)
    return f"curl -X {method} {url} -H \"Content-Type: application/json\" -d '{body}'"


def request(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    """Manda la petición a la API y devuelve el código HTTP y el cuerpo de la respuesta.

    Args:
        method: ``GET`` o ``POST``.
        url: URL completa del endpoint.
        payload: Cuerpo JSON de la petición, si la tiene.

    Returns:
        Código de estado y JSON de la respuesta.

    Raises:
        typer.Exit: Si la API no responde.
    """
    try:
        response = requests.request(method, url, json=payload, timeout=DEMO_TIMEOUT_SECONDS)
    except requests.RequestException as error:
        typer.secho(
            f"La API no responde en {url}: {error}. Levántala con `make compose-up` y espera a "
            "que `make compose-ps` muestre los dos servicios en healthy.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1) from error
    return response.status_code, response.json()


def compose_status() -> str:
    """Devuelve la salida de ``docker compose ps`` (o ``podman-compose ps``).

    Returns:
        La tabla de contenedores, o un aviso si ningún Compose está disponible.
    """
    for command in COMPOSE_COMMANDS:
        try:
            result = subprocess.run(  # nosec B603
                command,
                cwd=PROJ_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.rstrip()
    return "No se pudo consultar Compose: ¿está levantado Docker o Podman?"


def registry_summary(tracking_uri: str) -> list[tuple[str, str, str]]:
    """Lista las versiones de los modelos del proyecto en el MLflow de Compose.

    Args:
        tracking_uri: URL del servidor de MLflow levantado por ``docker-compose.yml``.

    Returns:
        Una tupla ``(modelo, versión, alias)`` por versión registrada, en orden de versión.
    """
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    rows: list[tuple[str, str, str]] = []
    for name in REGISTERED_MODELS.values():
        # Los alias viven en el modelo registrado (alias -> versión); search_model_versions
        # devuelve las versiones sin ellos.
        by_version: dict[str, list[str]] = {}
        for alias, version in client.get_registered_model(name).aliases.items():
            by_version.setdefault(version, []).append(f"@{alias}")
        versions = client.search_model_versions(f"name='{name}'")
        for version in sorted(versions, key=lambda v: int(v.version)):
            aliases = ", ".join(sorted(by_version.get(version.version, []))) or "sin alias"
            rows.append((name, version.version, aliases))
    return rows


# ---------------------------------------------------------------------------------------------
# Pasos de la demo
# ---------------------------------------------------------------------------------------------
def _show(title: str, command: str, output: str, step: int) -> None:
    typer.secho(f"\nPaso {step}. {title}", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"$ {command}", fg=typer.colors.CYAN)
    typer.echo(output)


def _pretty(status: int, body: dict) -> str:
    return f"HTTP {status}\n" + json.dumps(body, indent=2, ensure_ascii=False)


def _wait(pause: bool) -> None:
    if pause and sys.stdin.isatty():
        typer.pause("Enter para continuar...")


@app.command()
def run(
    host: Annotated[
        str, typer.Option(help="URL base de la API levantada con Compose.")
    ] = LOAD_TEST_HOST,
    mlflow_uri: Annotated[
        str, typer.Option(help="URL del MLflow levantado con Compose.")
    ] = LOAD_TEST_MLFLOW_URI,
    pause: Annotated[
        bool, typer.Option(help="Esperar Enter entre pasos (para grabar el video).")
    ] = True,
) -> None:
    """Recorre la demo: estado de Compose, /health, P1, P2, P3, una petición inválida y el Registry."""
    steps: list[tuple[str, str, Callable[[], str]]] = [
        ("Estado de los servicios de Compose", "docker compose ps", compose_status),
        (
            "Qué modelo responde en cada pregunta",
            curl_command("GET", f"{host}/health"),
            lambda: _pretty(*request("GET", f"{host}/health")),
        ),
    ]
    for question, schema, what in (
        ("p1", P1FeaturesInput, "Pronóstico de P1: salidas de la estación 50 a las 8:00"),
        ("p2", P2FeaturesInput, "Alerta de P2: riesgo en el siguiente intervalo de 3 horas"),
        ("p3", P3FeaturesInput, "Grupo de P3 para el perfil de uso de una estación"),
    ):
        url = f"{host}/predict/{question}"
        payload = example_payload(schema)
        steps.append(
            (
                what,
                curl_command("POST", url, payload),
                lambda url=url, payload=payload: _pretty(*request("POST", url, payload)),
            )
        )
    invalid_url = f"{host}/predict/p1"
    invalid = invalid_p1_payload()
    steps.append(
        (
            f"Petición inválida: P1 a las {invalid['hour_of_day']}:00, hora sin servicio (422)",
            curl_command("POST", invalid_url, invalid),
            lambda: _pretty(*request("POST", invalid_url, invalid)),
        )
    )

    def registry() -> str:
        try:
            rows = registry_summary(mlflow_uri)
        except MlflowException as error:
            return f"No se pudo leer el Model Registry en {mlflow_uri}: {error}"
        lines = [f"{'Modelo':<28} {'Versión':>7}  Alias"]
        lines += [f"{name:<28} {version:>7}  {aliases}" for name, version, aliases in rows]
        return "\n".join(lines)

    steps.append(("Model Registry de Compose", f"mlflow (MlflowClient) -> {mlflow_uri}", registry))

    typer.secho(
        f"Demo de MiBici - API en {host} - MLflow en {mlflow_uri}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    for number, (title, command, action) in enumerate(steps, start=1):
        _wait(pause)
        _show(title, command, action(), number)
    typer.secho(
        f"\nPara verlo en el navegador: {host}/docs (Try it out) y {mlflow_uri} (Model registry).",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
