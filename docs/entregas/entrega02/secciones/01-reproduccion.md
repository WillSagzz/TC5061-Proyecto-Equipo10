# Repositorio y reproducción

Repositorio: <https://github.com/WillSagzz/TC5061-Proyecto-Equipo10>. Resultados obtenidos con los *commits* `5a95a6f5` y `7a74fff3` de `main` y la versión de datos `2026-10`.

El registro de experimentos se realizó en el servidor de *tracking* compartido del equipo desplegado en **Azure Container Apps** (`rg-mibici-equipo10`, región `mexicocentral`), con metadatos almacenados en **Azure Database for PostgreSQL Flexible Server** y artefactos en **Azure Blob Storage** (o bien, en modo local sin archivo `.env`, sobre una base de datos SQLite en `sqlite:///mlflow.db`). En la raíz del repositorio, el archivo `MLproject` declara el entorno reproducible en `python_env.yaml` (con las versiones fijas de `requirements.txt`) y define tres puntos de entrada: `ingest` (que descarga, valida contra `datapackage.json` y preprocesa los datos en `mibici/dataset.py`), `train` (que entrena y registra modelos recibiendo los parámetros `question`, `model`, `alpha`, `c`, `learning_rate`, `max_iter`, `max_leaf_nodes` y `n_clusters` con valores por defecto tomados de `mibici/config.py`) y `search` (que ejecuta la búsqueda de hiperparámetros con corridas anidadas). Asimismo, al instalar el paquete en modo editable (`pip install -e .`), el proveedor `mibici/experiments.py` asigna automáticamente el experimento correspondiente (`mibici-ingesta`, `mibici-p1-demanda`, `mibici-p2-riesgo` o `mibici-p3-perfiles`). Todas las corridas reportadas en esta entrega se ejecutaron mediante `mlflow run`, lo cual es verificable en las etiquetas `mlflow.source.type = PROJECT` y `mlflow.project.entryPoint` (`ingest`, `train` o `search`) registradas en cada corrida.

Desde un clon limpio:

```bash
# 1. Crear entorno virtual, instalar dependencias fijas y registrar el plugin de experimentos
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -U pip && python -m pip install -r requirements.txt
python -m pip install -e .

# 2. Configurar el servidor de tracking (Opción A: servidor en Azure con .env; Opción B: local con SQLite)
cp .env.example .env   # Opcional: definir MLFLOW_TRACKING_URI, usuario y contraseña de Azure
# Si se ejecuta en modo 100% local sin .env:
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
export MLFLOW_TRACKING_URI=[http://127.0.0.1:5000](http://127.0.0.1:5000)

# 3. Ejecutar la ingesta reproducible (valida hashes y esquemas de datapackage.json, versión 2026-10)
mlflow run . -e ingest --env-manager local

# 4. Reproducir los modelos en producción (@champion) de P1, P2 y P3
mlflow run . -e train -P question=p1 -P model=hgb_poisson -P learning_rate=0.1 -P max_leaf_nodes=127 --env-manager local     # modelo en producción de P1
mlflow run . -e train -P question=p2 -P model=hgb_threshold -P learning_rate=0.05 -P max_leaf_nodes=63 --env-manager local   # modelo en producción de P2
mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=4 --env-manager local                                     # modelo en producción de P3

# 5. (Opcional) Reproducir las búsquedas de hiperparámetros con corridas anidadas (nested runs)
mlflow run . -e search -P question=p1 -P model=hgb --env-manager local
mlflow run . -e search -P question=p2 -P model=hgb --env-manager local
mlflow run . -e search -P question=p3 -P model=kmeans --env-manager local