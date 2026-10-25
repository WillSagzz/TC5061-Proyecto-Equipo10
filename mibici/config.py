from datetime import date
import getpass
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

# Descriptor Frictionless Data Package con la versión de los datos: lo escribe `make snapshot`
# (datapackage.py), se versiona en Git y `make data` (dataset.py) verifica cada archivo contra
# él antes de procesarlo. Las rutas de los recursos son relativas a la raíz del repositorio.
DATAPACKAGE_PATH = PROJ_ROOT / "datapackage.json"
RAW_VALIDATION_ROWS = 10_000  # filas de cada CSV que `make data` valida contra el esquema

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
# Un experimento por pregunta y otro para la ingesta. Cada pregunta tiene su propia tabla,
# variable objetivo y métricas, así que sus runs solo se comparan entre sí; una versión nueva
# de los datos no abre un experimento nuevo (el run registra `version_datos`). El experimento
# lo elige el propio proyecto a partir de la pregunta (experiments.py), de modo que
# `mlflow run` no necesita --experiment-name.
EXPERIMENTS = {
    "p1": "mibici-p1-demanda",
    "p2": "mibici-p2-riesgo",
    "p3": "mibici-p3-perfiles",
    "ingesta": "mibici-ingesta",
}
EXPERIMENT_TASKS = {
    "p1": "regresion",
    "p2": "clasificacion",
    "p3": "clustering",
    "ingesta": "ingesta",
}
EXPERIMENT_DESCRIPTIONS = {
    "p1": "P1 · Regresión: viajes iniciados por estación y hora (horizonte de 24 h).",
    "p2": "P2 · Clasificación: riesgo de que una estación se vacíe o se llene en el siguiente "
    "intervalo de 3 h.",
    "p3": "P3 · Clustering: perfiles de uso de las estaciones (solo periodo de entrenamiento).",
    "ingesta": "Ingesta reproducible: un run por versión de datos reconstruida desde "
    "datapackage.json.",
}
# Nombres reservados para el Model Registry (un modelo registrado por pregunta, con versiones
# y alias `baseline` y `elegido`); todavía no se usan.
REGISTERED_MODELS = {key: name for key, name in EXPERIMENTS.items() if key != "ingesta"}
# Por defecto, una base local en la raíz del proyecto (excluida de Git). Para usar un
# servidor de MLflow, definir MLFLOW_TRACKING_URI en el archivo .env.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"
# Quién ejecutó el run: el usuario del servidor compartido (MLFLOW_TRACKING_USERNAME en .env) o,
# en modo local, el usuario del sistema operativo. experiments.py lo escribe en la etiqueta
# `mlflow.user`, que la interfaz muestra en la columna "User".
try:
    MLFLOW_RUN_USER = os.getenv("MLFLOW_TRACKING_USERNAME") or getpass.getuser()
except (KeyError, OSError, ImportError):  # contenedores o CI sin usuario conocido
    MLFLOW_RUN_USER = "desconocido"

# P1 y P2: variables históricas de la Tabla 5 de la propuesta (features.py). Se expresan en
# horas; P2 las convierte a intervalos (1 intervalo, 24 h = 8 intervalos y 7 días = 56) y las
# usa todas, porque su horizonte es de un intervalo. P1 solo usa las de P1_LAG_HOURS.
LAG_HOURS = [1, 24, 24 * 7]  # salidas registradas 1 h, 24 h y 7 días antes
FLOW_LAG_HOURS = [1, 24]  # llegadas y flujo neto 1 h y 24 h antes
ROLLING_WINDOW = 3  # promedio móvil de salidas: periodos cerrados previos al que se predice

# P1: horizonte de 24 h (al cierre de cada día se pronostica hora por hora el día siguiente).
# Solo sirven los rezagos de al menos 24 h: el de 1 h de las 18:00 serían las 17:00 del mismo
# día que se pronostica, que aún no ocurren. El promedio móvil se desplaza el mismo horizonte.
P1_HORIZON_HOURS = 24
P1_LAG_HOURS = [h for h in LAG_HOURS if h >= P1_HORIZON_HOURS]  # 24 h y 7 días
P1_FLOW_LAG_HOURS = [h for h in FLOW_LAG_HOURS if h >= P1_HORIZON_HOURS]  # 24 h

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
