# Imagen OCI de la API REST de MiBici (mibici/api.py), pensada para Podman:
#
#   podman build -t mibici-api .                                   # o: make image
#   podman run --rm --env-file .env -p 8000:8000 mibici-api        # o: make container
#
# Al arrancar, la API carga los tres modelos @champion desde el servidor MLflow del equipo.
# Las credenciales (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD)
# no se copian a la imagen: .dockerignore excluye .env y se pasan con --env-file al ejecutar.

# Misma versión de Python que exige pyproject.toml y que usa GitHub Actions.
FROM python:3.12-slim

# Sin archivos .pyc dentro de la imagen y con los logs sin búfer para verlos al momento
# en `podman logs`. pip no busca versiones nuevas de sí mismo al instalar.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Todo el proyecto vive en /app: mibici/config.py calcula PROJ_ROOT como el directorio padre
# del paquete, así que dentro del contenedor PROJ_ROOT es /app y las rutas derivadas
# (models/registry, mlflow.db, data/) quedan bajo este directorio.
WORKDIR /app

# requirements.txt instala el paquete en modo editable (-e .), que flit resuelve con
# pyproject.toml y README.md; por eso ambos se copian junto con el código antes de instalar.
COPY requirements.txt pyproject.toml README.md ./
COPY mibici/ ./mibici/
RUN pip install --no-cache-dir -r requirements.txt

# Usuario sin privilegios para ejecutar el servicio. Su HOME y models/registry (donde escribe
# `python -m mibici.modeling.registry download`) le pertenecen; los artefactos que MLflow
# descarga al cargar cada modelo van al directorio temporal del sistema, que ya puede escribir.
RUN useradd --system --create-home --shell /usr/sbin/nologin mibici \
    && mkdir -p /app/models/registry \
    && chown -R mibici:mibici /app/models
USER mibici

# Puerto en el que escucha uvicorn; se publica al ejecutar con -p 8000:8000.
EXPOSE 8000

# Escucha en todas las interfaces del contenedor para que el puerto publicado llegue al host.
CMD ["uvicorn", "mibici.api:app", "--host", "0.0.0.0", "--port", "8000"]
