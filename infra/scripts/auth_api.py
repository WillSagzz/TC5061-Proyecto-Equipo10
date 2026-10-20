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
        response = self.session.request(
            method, f"{self.url}{path}", timeout=self.timeout, **kwargs
        )
        return response

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

    # Experimentos y permisos -------------------------------------------------------------
    def list_experiments(self) -> list[dict]:
        data = self._ok(
            "POST",
            "/api/2.0/mlflow/experiments/search",
            json={"max_results": 1000, "view_type": "ACTIVE_ONLY"},
        )
        return data.get("experiments", [])

    def get_experiment_by_name(self, name: str) -> dict | None:
        response = self._call(
            "GET", "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name}
        )
        return response.json().get("experiment") if response.status_code == 200 else None

    def create_experiment(self, name: str) -> str:
        return self._ok("POST", "/api/2.0/mlflow/experiments/create", json={"name": name})[
            "experiment_id"
        ]

    def get_experiment_permission(self, username: str, experiment_id: str) -> str | None:
        response = self._call(
            "GET",
            "/api/3.0/mlflow/users/permissions/get",
            params={
                "username": username,
                "resource_type": "experiment",
                "resource_id": experiment_id,
            },
        )
        if response.status_code == 200:
            return response.json().get("permission")
        return None

    def grant_experiment_permission(
        self, username: str, experiment_id: str, permission: str
    ) -> None:
        body = {
            "username": username,
            "resource_type": "experiment",
            "resource_id": experiment_id,
            "permission": permission,
        }
        response = self._call("POST", "/api/3.0/mlflow/users/permissions/grant", json=body)
        if response.ok:
            return
        # Si ya había una concesión distinta, se reemplaza
        self._ok(
            "POST",
            "/api/3.0/mlflow/users/permissions/revoke",
            json={k: v for k, v in body.items() if k != "permission"},
        )
        self._ok("POST", "/api/3.0/mlflow/users/permissions/grant", json=body)
