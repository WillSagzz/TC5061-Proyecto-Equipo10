# Repositorio y reproducción

Repositorio: <https://github.com/WillSagzz/TC5061-Proyecto-Equipo10>. Los resultados de este reporte se obtuvieron con la versión de datos `2026-10` y los *commits* `5a95a6f5` y `7a74fff3`; el *commit* exacto de cada corrida queda registrado en su etiqueta `git_commit`.

El registro de experimentos se hizo en el servidor de *tracking* compartido del equipo, desplegado en **Azure Container Apps** (`rg-mibici-equipo10`, región `mexicocentral`), con los metadatos en **Azure Database for PostgreSQL Flexible Server** y los artefactos en **Azure Blob Storage**. Al estar respaldado por una base de datos, el servidor admite el Model Registry (MLflow, s.f.-b, s.f.-e). Sin archivo `.env`, el mismo código registra en una base local SQLite (`sqlite:///mlflow.db`).

En la raíz del repositorio, el archivo `MLproject` define el proyecto (MLflow, s.f.-d): declara el entorno en `python_env.yaml` (Python 3.10 y las versiones fijas de `requirements.txt`) y tres puntos de entrada:

- `ingest` descarga los datos, los verifica contra `datapackage.json` y escribe los Parquet limpios (`mibici/dataset.py`).
- `train` entrena y registra un modelo; recibe `question`, `model` y los hiperparámetros `alpha`, `c`, `learning_rate`, `max_iter`, `max_leaf_nodes` y `n_clusters`, cuyos valores por defecto salen de `mibici/config.py`.
- `search` ejecuta una búsqueda de hiperparámetros con corridas anidadas.

Al instalar el paquete (`requirements.txt` incluye `-e .`), el *plugin* de `mibici/experiments.py` asigna a cada corrida el experimento de su pregunta: `mibici-ingesta`, `mibici-p1-demanda`, `mibici-p2-riesgo` o `mibici-p3-perfiles`. Cada corrida registra además el conjunto de datos como entrada, con `version_datos` como huella (MLflow, s.f.-a). Todas las corridas de esta entrega se lanzaron con `mlflow run`, lo que se verifica en sus etiquetas `mlflow.source.type = PROJECT` y `mlflow.project.entryPoint` (`ingest`, `train` o `search`).

Desde un clon limpio:

```bash
# 1. Entorno virtual, dependencias con versión fija y paquete del proyecto
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# 2. Servidor de tracking: copiar .env.example a .env y llenar la URI, el usuario y la
#    contraseña del servidor en Azure. Para un servidor local con base de datos:
#    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
#    y MLFLOW_TRACKING_URI=http://127.0.0.1:5000 en el .env
cp .env.example .env

# 3. Ingesta reproducible (verifica hashes y esquema contra datapackage.json, versión 2026-10)
mlflow run . -e ingest --env-manager local

# 4. Modelo elegido de cada pregunta (el registrado como @champion)
mlflow run . -e train --env-manager local -P question=p1 -P model=hgb_poisson -P learning_rate=0.1 -P max_leaf_nodes=127
mlflow run . -e train --env-manager local -P question=p2 -P model=hgb_threshold -P learning_rate=0.05 -P max_leaf_nodes=63
mlflow run . -e train --env-manager local -P question=p3

# 5. (Opcional) Búsquedas de hiperparámetros con corridas anidadas
mlflow run . -e search --env-manager local -P question=p1 -P model=hgb
mlflow run . -e search --env-manager local -P question=p2 -P model=hgb
mlflow run . -e search --env-manager local -P question=p3 -P model=kmeans
```

`make data` y `make train` ejecutan los pasos 3 y 4. La opción `--env-manager local` usa el entorno ya activado; sin ella, `mlflow run` construye el entorno de `python_env.yaml` en un *virtualenv* propio. Al cerrar la entrega, la ingesta se volvió a ejecutar con el comando del paso 3 sobre el servidor compartido (corrida `ingesta-2026-10`).
