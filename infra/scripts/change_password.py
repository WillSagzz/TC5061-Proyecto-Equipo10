"""Cambia tu propia contraseña en el servidor MLflow del equipo.

Lee ``MLFLOW_TRACKING_URI``, ``MLFLOW_TRACKING_USERNAME`` y ``MLFLOW_TRACKING_PASSWORD`` del
``.env`` (la contraseña actual), pide la nueva en la terminal y la actualiza en el servidor.
No necesita permisos de administrador ni acceso a Azure. Después, actualiza tu ``.env``.

Lo ejecuta ``make mlflow-password``.
"""

from __future__ import annotations

from getpass import getpass
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auth_api import AuthApi

MIN_LENGTH = 12


def main() -> None:
    # Windows abre la consola en cp1252 y MLflow imprime emojis y acentos: forzar UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    url = os.getenv("MLFLOW_TRACKING_URI", "")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    current = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not url.startswith("http") or not username or not current:
        sys.exit("Define MLFLOW_TRACKING_URI (https://...), USERNAME y PASSWORD en tu .env")

    new = getpass(f"Nueva contraseña para {username} (mínimo {MIN_LENGTH} caracteres): ")
    if len(new) < MIN_LENGTH:
        sys.exit("Demasiado corta")
    if getpass("Repite la nueva contraseña: ") != new:
        sys.exit("No coinciden")

    AuthApi(url, username, current).update_password(username, new, current_password=current)
    print(
        "Contraseña actualizada en el servidor. Ahora cambia MLFLOW_TRACKING_PASSWORD en tu .env."
    )


if __name__ == "__main__":
    main()
