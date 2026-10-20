"""Cliente mínimo de la API de autenticación básica de MLflow.

Se usa ``requests`` directamente, en lugar de ``mlflow.server.get_app_client``, porque la
clase de MLflow cambia entre versiones (el cliente del repo es 3.4 y el servidor 3.17) y
estos *endpoints* REST existen en ambas. Referencia: ``mlflow/server/auth/routes.py``.
"""

from __future__ import annotations

import requests


class AuthApi:
    """Llamadas autenticadas a un servidor MLflow con ``--app-name basic-auth``."""

    def __init__(self, url: str, username: str, password: str, timeout: int = 30) -> None:
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.timeout = timeout

    def _call(self, method: str, path: str, **kwargs) -> requests.Response:
        return self.session.request(method, f"{self.url}{path}", timeout=self.timeout, **kwargs)

    def _ok(self, method: str, path: str, **kwargs) -> dict:
        response = self._call(method, path, **kwargs)
        if not response.ok:
            raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text[:300]}")
        return response.json() if response.content else {}

    # Usuarios ----------------------------------------------------------------------------
    def user_exists(self, username: str) -> bool:
        response = self._call("GET", "/api/2.0/mlflow/users/get", params={"username": username})
        if response.status_code == 200:
            return True
        if response.status_code in (400, 404):
            return False
        raise RuntimeError(f"users/get -> {response.status_code}: {response.text[:300]}")

    def list_users(self) -> list[str]:
        data = self._ok("GET", "/api/2.0/mlflow/users/list")
        return [user["username"] for user in data.get("users", [])]

    def create_user(self, username: str, password: str) -> None:
        self._ok(
            "POST",
            "/api/2.0/mlflow/users/create",
            json={"username": username, "password": password},
        )

    def delete_user(self, username: str) -> None:
        self._ok("DELETE", "/api/2.0/mlflow/users/delete", json={"username": username})

    def update_password(self, username: str, password: str, current_password: str | None) -> None:
        body = {"username": username, "password": password}
        if current_password is not None:
            # Obligatorio cuando una persona cambia su propia contraseña (servidores >= 3.15)
            body["current_password"] = current_password
        self._ok("PATCH", "/api/2.0/mlflow/users/update-password", json=body)

    # Experimentos ------------------------------------------------------------------------
    def get_experiment_by_name(self, name: str) -> dict | None:
        response = self._call(
            "GET", "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name}
        )
        return response.json().get("experiment") if response.status_code == 200 else None

    def create_experiment(self, name: str) -> str:
        return self._ok("POST", "/api/2.0/mlflow/experiments/create", json={"name": name})[
            "experiment_id"
        ]

    # Roles (servidores >= 3.16): permisos por patrón de recurso, p. ej. EDIT en experiment:*
    def list_roles(self, workspace: str) -> list[dict]:
        data = self._ok("GET", "/api/3.0/mlflow/roles/list", params={"workspace": workspace})
        return data.get("roles", [])

    def create_role(self, workspace: str, name: str, description: str) -> dict:
        body = {"workspace": workspace, "name": name, "description": description}
        return self._ok("POST", "/api/3.0/mlflow/roles/create", json=body)["role"]

    def get_role(self, role_id: int) -> dict:
        return self._ok("GET", "/api/3.0/mlflow/roles/get", params={"role_id": str(role_id)})[
            "role"
        ]

    def add_role_permission(
        self, role_id: int, resource_type: str, resource_pattern: str, permission: str
    ) -> None:
        body = {
            "role_id": role_id,
            "resource_type": resource_type,
            "resource_pattern": resource_pattern,
            "permission": permission,
        }
        self._ok("POST", "/api/3.0/mlflow/roles/permissions/add", json=body)

    def list_user_roles(self, username: str) -> list[dict]:
        data = self._ok("GET", "/api/3.0/mlflow/users/roles/list", params={"username": username})
        return data.get("roles", [])

    def assign_role(self, username: str, role_id: int) -> None:
        body = {"username": username, "role_id": role_id}
        self._ok("POST", "/api/3.0/mlflow/roles/assign", json=body)

    def unassign_role(self, username: str, role_id: int) -> None:
        body = {"username": username, "role_id": role_id}
        self._ok("DELETE", "/api/3.0/mlflow/roles/unassign", json=body)
