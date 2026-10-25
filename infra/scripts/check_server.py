"""Prueba de humo del servidor MLflow compartido.

Con las credenciales del ``.env`` crea un run en el experimento ``smoke-test`` con un
parámetro, una métrica y un artefacto de texto; después lo vuelve a leer y descarga el
artefacto. Si todo pasa, funcionan la autenticación, la base PostgreSQL y el proxy de
artefactos hacia Blob Storage. Con ``--expect-read-only`` (para el usuario de solo lectura)
comprueba en cambio que el servidor rechace registrar un run en el experimento del equipo
(MLflow sin *workspaces* deja crear experimentos propios a cualquier usuario autenticado,
así que la prueba se hace sobre el experimento que ya existe).

Lo ejecuta ``make infra-check``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
import mlflow
from mlflow.exceptions import MlflowException

EXPERIMENT = "smoke-test"
# Un experimento del equipo que ya existe, para la prueba de solo lectura
try:
    from mibici.config import EXPERIMENTS

    TEAM_EXPERIMENT = EXPERIMENTS["p1"]
except ImportError:  # pragma: no cover
    TEAM_EXPERIMENT = os.getenv("MLFLOW_TEAM_EXPERIMENT", "mibici-p1-demanda")


def main() -> None:
    # Windows abre la consola en cp1252 y MLflow imprime emojis y acentos: forzar UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--expect-read-only", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    user = os.getenv("MLFLOW_TRACKING_USERNAME")
    if not uri.startswith("http") or not user or not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        sys.exit("Define MLFLOW_TRACKING_URI (https://...), USERNAME y PASSWORD en tu .env")

    mlflow.set_tracking_uri(uri)
    print(f"Servidor: {uri}  usuario: {user}  cliente mlflow {mlflow.__version__}")

    experiments = mlflow.search_experiments()
    print(f"Lectura OK: {len(experiments)} experimento(s) visibles")

    if args.expect_read_only:
        experiment = mlflow.get_experiment_by_name(TEAM_EXPERIMENT)
        if experiment is None:
            sys.exit(f"No existe el experimento {TEAM_EXPERIMENT}: corre antes make infra-users")
        try:
            with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="no-deberia"):
                pass
        except MlflowException as error:
            print(f"Escritura rechazada como se esperaba: {str(error)[:120]}")
            return
        sys.exit("ERROR: el usuario de solo lectura pudo escribir un run")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"smoke-{user}") as run:
        mlflow.log_param("usuario", user)
        mlflow.log_metric("ok", 1.0)
        mlflow.log_text("hola desde el equipo 10\n", "saludo.txt")
        run_id = run.info.run_id
    print(f"Run creado: {run_id}")

    client = mlflow.MlflowClient()
    stored = client.get_run(run_id)
    assert stored.data.params["usuario"] == user
    assert stored.data.metrics["ok"] == 1.0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(client.download_artifacts(run_id, "saludo.txt", tmp))
        assert path.read_text(encoding="utf-8").startswith("hola")
    print("Artefacto descargado desde Blob a través del servidor. Todo OK.")


if __name__ == "__main__":
    main()
