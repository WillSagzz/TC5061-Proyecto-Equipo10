# Orquestación con Docker Compose

`docker-compose.yml` levanta el sistema completo con un comando: `docker compose up -d --build` (o `make compose-up`; también funciona con `podman-compose`). Son dos servicios que se construyen de la misma imagen (`Containerfile`).

**Tabla 1.** Servicios de `docker-compose.yml`.

| Servicio | Qué hace | Puerto | Límites |
|---|---|---|---|
| `mlflow` | Servidor de MLflow con el Model Registry. Guarda su base de datos y sus artefactos en dos volúmenes | 5001 | 2 CPU y 3 GB |
| `api` | API REST de FastAPI: `/health`, `/predict/p1`, `/predict/p2` y `/predict/p3`. Carga los modelos `@champion` desde `mlflow` | 8000 | 2 CPU y 2 GB |

El arranque no tiene pasos manuales. `api` espera a que `mlflow` pase su *healthcheck* (`depends_on` con `service_healthy`). Antes de iniciar la API, `mibici/modeling/bootstrap.py` revisa el Model Registry: si faltan modelos, registra los que estén descargados en `models/registry/` y, si no hay ninguno, entrena unos sintéticos. Por eso el sistema también levanta en un clon limpio y en GitHub Actions, sin credenciales.

La configuración incluye:

- Una red propia (`mibici-net`), por la que la API llega a MLflow por su nombre (`http://mlflow:5000`).
- Dos volúmenes con nombre, que conservan los modelos registrados entre reinicios.
- *Healthchecks* en los dos servicios y límites de CPU y memoria.
- Variables de entorno con valor por defecto para los puertos, el alias del modelo, los *workers* de la API y los límites. Se cambian al invocar, sin editar el archivo: `API_WORKERS=2 docker compose up -d`.
