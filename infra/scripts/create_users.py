"""Crea los usuarios del equipo en el servidor MLflow y reparte permisos.

Lee ``infra/users.txt`` (archivo local, fuera de Git; formato en ``users.example.txt``: una línea
por persona, ``usuario,rol``; rol ``editor`` o ``lector``),
crea los usuarios que falten con una contraseña generada, otorga ``EDIT`` a los editores sobre
cada experimento existente (los lectores se quedan con el permiso por defecto, ``READ``) y
escribe las contraseñas nuevas en ``infra/.users.env`` (fuera de Git) para entregarlas en
privado. Es idempotente: se puede volver a correr cuando aparezca un experimento o una
persona nueva. Con ``--prune`` borra los usuarios que ya no estén en la lista y con
``--reset USUARIO`` le asigna una contraseña nueva a quien olvidó la suya.

Lo ejecuta ``make infra-users`` con las credenciales de ``admin``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import string
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auth_api import AuthApi

INFRA_DIR = Path(__file__).resolve().parents[1]
# El nombre del experimento del equipo vive en mibici/config.py; si el paquete no está
# instalado en este entorno se toma de la variable de entorno del mismo nombre.
try:
    from mibici.config import MLFLOW_EXPERIMENT_NAME
except ImportError:  # pragma: no cover
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "")
ROLES = {"editor": "EDIT", "lector": None}
PROTECTED = {"admin"}


def generate_password(length: int = 20) -> str:
    """Contraseña aleatoria solo con letras y dígitos, fácil de pegar en un .env."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def read_users(path: Path) -> dict[str, str]:
    users: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        username, _, role = (part.strip() for part in line.partition(","))
        role = role or "editor"
        if role not in ROLES:
            sys.exit(f"Rol desconocido '{role}' para {username}; usa editor o lector")
        users[username] = role
    return users


def main() -> None:
    # Windows abre la consola en cp1252 y MLflow imprime emojis y acentos: forzar UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.getenv("MLFLOW_URL"), help="URL del servidor")
    parser.add_argument("--users-file", type=Path, default=INFRA_DIR / "users.txt")
    parser.add_argument("--out", type=Path, default=INFRA_DIR / ".users.env")
    parser.add_argument("--prune", action="store_true", help="borra usuarios fuera de la lista")
    parser.add_argument(
        "--reset", action="append", default=[], metavar="USUARIO", help="contraseña nueva"
    )
    args = parser.parse_args()

    admin_password = os.getenv("MLFLOW_ADMIN_PASSWORD")
    if not args.url or not admin_password:
        sys.exit("Faltan MLFLOW_URL o MLFLOW_ADMIN_PASSWORD (los exporta `make infra-users`)")

    if not args.users_file.exists():
        sys.exit(
            f"Falta {args.users_file}: cópialo de users.example.txt con los usuarios del equipo "
            "(ese archivo no se versiona)"
        )
    api = AuthApi(args.url, "admin", admin_password)
    wanted = read_users(args.users_file)
    created: dict[str, str] = {}

    for username in wanted:
        if api.user_exists(username):
            print(f"  = {username}: ya existe")
            continue
        password = generate_password()
        api.create_user(username, password)
        created[username] = password
        print(f"  + {username}: creado")

    # El experimento del equipo se crea aquí (y no en el primer `make train`) para que todos
    # los editores reciban EDIT desde el principio; quien crea un experimento es MANAGE y el
    # resto solo READ hasta que se vuelva a correr este script.
    if MLFLOW_EXPERIMENT_NAME and api.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        api.create_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"  + experimento '{MLFLOW_EXPERIMENT_NAME}': creado")
    for username in args.reset:
        if not api.user_exists(username):
            sys.exit(f"No existe el usuario {username}")
        password = generate_password()
        api.update_password(username, password, current_password=None)  # admin: sin la actual
        created[username] = password
        print(f"  * {username}: contraseña restablecida")

    experiments = api.list_experiments()
    for username, role in wanted.items():
        permission = ROLES[role]
        if permission is None:
            continue
        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            if api.get_experiment_permission(username, experiment_id) == permission:
                continue
            api.grant_experiment_permission(username, experiment_id, permission)
            print(f"  ~ {username}: {permission} en '{experiment['name']}'")

    if args.prune:
        for username in api.list_users():
            if username not in wanted and username not in PROTECTED:
                api.delete_user(username)
                print(f"  - {username}: eliminado")

    if created:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(f"# Contraseñas iniciales generadas el {stamp}. NO entra a Git.\n")
            for username, password in created.items():
                handle.write(f"{username}={password}\n")
        print(f"\nContraseñas nuevas en {args.out}; entrégalas por un canal privado.")
    print(f"\nListo: {len(wanted)} usuarios en {args.url}")


if __name__ == "__main__":
    main()
