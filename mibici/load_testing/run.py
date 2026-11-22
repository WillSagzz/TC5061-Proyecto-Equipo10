"""Corre la prueba de carga de la API con Locust y mide los contenedores (Entrega 06).

Antes hay que levantar el sistema con los modelos reales:

    python -m mibici.modeling.registry download --question all   # modelos reales (usa .env)
    make compose-up                                              # MLflow + API en Docker

Después, desde la raíz del repositorio:

    python -m mibici.load_testing.run run        # make load-test: ~6 minutos
    python -m mibici.load_testing.run summarize  # vuelve a hacer el resumen y las figuras

Los resultados quedan en ``reports/locust/`` (ignorada por Git): el reporte HTML de Locust, los
CSV de Locust y de ``docker stats``, el resumen por escalón y por endpoint, y las dos figuras.
"""

from __future__ import annotations

from pathlib import Path
import subprocess  # nosec B404
import sys
from typing import Annotated

from loguru import logger
import mlflow
from mlflow.exceptions import MlflowException
import requests
import typer

from mibici.config import (
    DEFAULT_MODEL_ALIAS,
    LOAD_TEST_HOST,
    LOAD_TEST_MLFLOW_URI,
    LOAD_TEST_REPORTS_DIR,
    LOAD_TEST_STAGES,
    PROJ_ROOT,
    REGISTERED_MODELS,
)
from mibici.load_testing.analysis import (
    DOCKER_STATS_FILE,
    LOCUST_PREFIX,
    DockerStatsRecorder,
    write_summary,
)

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
HTML_REPORT = "locust_reporte.html"

app = typer.Typer(help="Prueba de carga de la API con Locust y docker stats (Entrega 06).")


def check_api(host: str) -> None:
    """Confirma que la API responde antes de lanzar la carga.

    Args:
        host: URL base de la API.

    Raises:
        typer.Exit: Si la API no responde con ``/health``.
    """
    try:
        response = requests.get(f"{host}/health", timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        logger.error(f"La API no responde en {host}/health ({error}). ¿Corriste make compose-up?")
        raise typer.Exit(code=1) from error
    models = ", ".join(f"{m['question']} v{m['version']}" for m in response.json()["models"])
    logger.info(f"API lista en {host}: {models}")


def synthetic_models(tracking_uri: str = LOAD_TEST_MLFLOW_URI) -> list[str]:
    """Preguntas cuyo modelo ``@champion`` en el MLflow de Compose es sintético.

    ``mibici.modeling.bootstrap`` registra un modelo sintético (entrenado con datos inventados)
    cuando no encuentra los modelos reales en ``models/registry/``. Esos modelos son más ligeros
    que los reales, así que la prueba de carga mediría latencias y costos más bajos de lo real.

    Args:
        tracking_uri: MLflow levantado por ``docker-compose.yml``.

    Returns:
        Preguntas con modelo sintético; vacía si todos son reales o si no se pudo consultar.
    """
    try:  # sin esta revisión, MLflow reintenta varios minutos si el servidor no responde
        requests.get(f"{tracking_uri}/health", timeout=5).raise_for_status()
    except requests.RequestException as error:
        logger.warning(f"No se pudo revisar el origen de los modelos en {tracking_uri}: {error}")
        return []
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    synthetic = []
    for question, model_name in REGISTERED_MODELS.items():
        try:
            version = client.get_model_version_by_alias(model_name, DEFAULT_MODEL_ALIAS)
            tags = client.get_run(version.run_id).data.tags if version.run_id else {}
        except (MlflowException, requests.RequestException) as error:
            logger.warning(f"No se pudo revisar el origen del modelo de {question}: {error}")
            continue
        if tags.get("origen_bootstrap") == "synthetic":
            synthetic.append(question)
    return synthetic


def run_locust(host: str, output_dir: Path) -> int:
    """Corre Locust sin interfaz con la carga escalonada de ``locustfile.py``.

    Args:
        host: URL base de la API.
        output_dir: Carpeta donde Locust deja su reporte HTML y sus CSV.

    Returns:
        Código de salida de Locust (1 si hubo peticiones fallidas).
    """
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        host,
        "--csv",
        str(output_dir / LOCUST_PREFIX),
        "--csv-full-history",
        "--html",
        str(output_dir / HTML_REPORT),
        "--only-summary",
    ]
    duration = LOAD_TEST_STAGES[-1][0]
    users = " → ".join(str(stage_users) for _, stage_users in LOAD_TEST_STAGES)
    logger.info(f"Locust: {users} usuarios en {duration} s contra {host}")
    return subprocess.run(command, cwd=PROJ_ROOT, check=False).returncode  # nosec B603


@app.command("run")
def run_command(
    host: Annotated[str, typer.Option(help="URL de la API.")] = LOAD_TEST_HOST,
    allow_synthetic: Annotated[
        bool, typer.Option(help="Correr aunque algún modelo sea sintético.")
    ] = False,
) -> None:
    """Corre Locust y docker stats al mismo tiempo y escribe el resumen y las figuras."""
    check_api(host)
    synthetic = synthetic_models()
    if synthetic and not allow_synthetic:
        logger.error(
            f"El MLflow de Compose tiene modelos sintéticos para {', '.join(synthetic)}: la "
            "prueba no mediría los modelos reales. Corre `python -m mibici.modeling.registry "
            "download --question all`, luego `docker compose down -v` (borra el MLflow local con "
            "los modelos sintéticos) y `make compose-up`; o usa --allow-synthetic para probar "
            "solo la herramienta."
        )
        raise typer.Exit(code=1)

    output_dir = LOAD_TEST_REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    with DockerStatsRecorder(output_dir / DOCKER_STATS_FILE):
        exit_code = run_locust(host, output_dir)
    if exit_code != 0:
        logger.warning(f"Locust terminó con código {exit_code}: hubo peticiones fallidas")

    summary = write_summary(output_dir)
    logger.success(f"Prueba terminada. Resumen: {summary}")
    logger.info(f"Reporte interactivo de Locust: {output_dir / HTML_REPORT}")


@app.command("summarize")
def summarize_command() -> None:
    """Vuelve a escribir el resumen y las figuras de la última prueba."""
    logger.success(f"Resumen: {write_summary(LOAD_TEST_REPORTS_DIR)}")


if __name__ == "__main__":
    app()
