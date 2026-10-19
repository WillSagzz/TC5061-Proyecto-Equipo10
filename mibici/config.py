from datetime import date
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

START_DATE = "2024-09-01"  # primer mes incluido (datos_abiertos_2024_09)
END_DATE = "2026-08-01"  # último mes incluido (datos_abiertos_2026_08); septiembre 2026 reservado

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


# =============================================================================
# Ingesta (dataset.py)
# =============================================================================
MIBICI_BASE_URL = "https://mibici.net"
MIBICI_DATA_PAGE = f"{MIBICI_BASE_URL}/es/datos-abiertos/"

# Contrato de datos: lo que entrega la ingesta y leen los modelos (features.py).
# Nombres de columnas tal como los normaliza dataset.py; no hay datos personales.
TRIPS_PATH = PROCESSED_DATA_DIR / "meses.parquet"
STATIONS_PATH = PROCESSED_DATA_DIR / "nomenclatura.parquet"
TRIP_COLUMNS = {
    "viaje_id": "integer",
    "inicio_del_viaje": "datetime",
    "fin_del_viaje": "datetime",
    "origen_id": "integer",  # estación de donde sale la bicicleta
    "destino_id": "integer",  # estación a donde llega la bicicleta
}
STATION_COLUMNS = {
    "id": "integer",
    "name": "text",
    "latitude": "float",
    "longitude": "float",
    "dpcapacity": "integer",  # número de anclajes
}

# Manifiesto con la versión de los datos: lo genera dataset.py y se versiona en Git
DATA_MANIFEST_PATH = PROJ_ROOT / "data_manifest.json"

# Limpieza de viajes
MIN_TRIP_MINUTES = 1  # retiros y devoluciones en la misma estación por debajo de esto
MAX_TRIP_MINUTES = 24 * 60
NO_SERVICE_HOURS = [1, 2, 3, 4]  # de 1:00 a 4:59 no se pueden retirar bicicletas


# =============================================================================
# Modelado y experimentos (features.py y modeling/)
# =============================================================================
RANDOM_SEED = 42

# Partición temporal, compartida por P1, P2 y P3. Cada conjunto va de su fecha de inicio
# (incluida) a la fecha de inicio del siguiente (excluida).
TRAIN_START = START_DATE  # entrenamiento: 18 meses, septiembre 2024 a febrero 2026
VALIDATION_START = "2026-03-01"  # validación: 3 meses, marzo a mayo 2026
TEST_START = "2026-06-01"  # prueba: 3 meses, junio a agosto 2026; se usa una sola vez
# Fin de los datos (excluido): primer día del mes siguiente a END_DATE, 2026-09-01
_last_month = date.fromisoformat(END_DATE)
DATA_END = date(
    _last_month.year + _last_month.month // 12, _last_month.month % 12 + 1, 1
).isoformat()

# MLflow Tracking
TEAM = "equipo10"
MLFLOW_EXPERIMENT_NAME = f"mibici-{TEAM}"
# Por defecto, una base local en la raíz del proyecto (excluida de Git). Para usar un
# servidor de MLflow, definir MLFLOW_TRACKING_URI en el archivo .env.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"

# P2: riesgo de que una estación se vacíe o se llene
P2_INTERVAL_HOURS = 3  # 8 intervalos por día
RISK_THRESHOLD = 0.4  # fracción de la capacidad que debe alcanzar el flujo neto

# P3: perfiles de uso de las estaciones (solo con el periodo de entrenamiento)
# Periodos del día: hora de inicio (incluida) y de fin (excluida); la noche cruza la medianoche
DAY_PERIODS = {
    "morning": (5, 12),
    "midday": (12, 16),
    "afternoon": (16, 20),
    "night": (20, 5),
}
P3_N_CLUSTERS = 4  # número de grupos del baseline
P3_K_VALUES = list(range(2, 9))  # valores de k que se comparan: de 2 a 8
