"""Ingesta reproducible de los datos abiertos de MiBici.

Uso desde la raíz del repositorio, como punto de entrada del MLflow Project (``make data``):

    mlflow run . -e ingest

o directo, para depurar: ``python -m mibici.dataset``. Bajo ``mlflow run`` la ingesta queda
registrada como un run del experimento ``mibici-ingesta`` (nombre ``ingesta-{version}``, la
versión como parámetro y ``datapackage.json`` como artefacto); directo no registra nada.

Reconstruye la versión de datos descrita en ``datapackage.json`` (descriptor Frictionless Data
Package que escribe ``make snapshot``, ver ``datapackage.py``): descarga a ``data/raw/`` los
archivos que falten, verifica el tamaño y el hash SHA-256 de cada uno contra el descriptor y
termina con error si alguno difiere, valida una muestra de filas contra el esquema declarado,
normaliza y limpia los viajes, escribe los Parquet de ``data/processed/`` y comprueba que
coinciden con los que describe el descriptor. Rutas, fechas y reglas de limpieza salen de
``config.py``.
"""

from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import time
import unicodedata

from bs4 import BeautifulSoup
from frictionless import FrictionlessException, Package, Resource
from loguru import logger
import pandas
import pyarrow.parquet as pq
import requests
from tqdm import tqdm
import typer

from mibici.config import (
    DATAPACKAGE_PATH,
    END_DATE,
    MAX_TRIP_MINUTES,
    MIN_TRIP_MINUTES,
    NO_SERVICE_HOURS,
    RAW_DATA_DIR,
    RAW_VALIDATION_ROWS,
    START_DATE,
    STATION_COLUMNS,
    STATIONS_PATH,
    TRIP_COLUMNS,
    TRIPS_PATH,
)

app = typer.Typer()

# Nombre del recurso del catálogo de estaciones en datapackage.json; los meses se llaman
# datos-abiertos-AAAA-MM y los Parquet, como sus archivos (meses y nomenclatura-procesada)
NOMENCLATURA_RESOURCE = "nomenclatura"
TRIPS_RESOURCE = "meses"
STATIONS_RESOURCE = "nomenclatura-procesada"


class DataVersionError(RuntimeError):
    """Los datos locales o el portal no coinciden con la versión descrita en datapackage.json."""


# MARK: Fetch URL
def get_url(url: str) -> requests.Response:
    """Descarga una URL del portal (página o archivo) con un agente de navegador.

    Args:
        url: Dirección completa de la página o del archivo.

    Returns:
        Respuesta HTTP con el contenido.

    Raises:
        requests.HTTPError: Si el portal responde con error; así no se guarda una página de
            error como si fuera un CSV.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    return r


# MARK: File paths
def get_paths(site_html: str) -> dict[str, list[str]]:
    """Extrae de la página de datos abiertos los enlaces a los CSV mensuales y al catálogo.

    Los enlaces llevan un identificador numérico que asigna el portal, por eso se leen de la
    página en lugar de mantener una lista a mano. Solo ``make snapshot`` los necesita:
    ``make data`` descarga las URL que ya están en el descriptor.

    Args:
        site_html: HTML de la página de datos abiertos.

    Returns:
        Diccionario con las rutas relativas de los archivos mensuales (``"Months"``) y de la
        nomenclatura (``"Nom"``).
    """
    logger.info("Parsing HTML for file links...")
    soup = BeautifulSoup(site_html, "html.parser")
    links = soup.find_all("a")

    parent_dir = "/site/assets/files"
    file_paths = {"Months": [], "Nom": []}
    for link in links:
        # str() solo para evitar los avisos de tipo del editor; el valor ya es texto
        link_href = str(link.get("href"))
        if not link_href.startswith(parent_dir):
            continue
        if "datos_abiertos" in link_href:
            file_paths["Months"].append(link_href)
        elif "nomenclatura" in link_href:
            file_paths["Nom"].append(link_href)
        else:
            continue

    return file_paths


# MARK: Filter by date
def filter_paths(paths: list, start_date: str, end_date: str) -> list[str]:
    """Conserva solo las rutas de los meses dentro del periodo del proyecto.

    El mes se lee del nombre del archivo (``datos_abiertos_AAAA_MM.csv``).

    Args:
        paths: Rutas de los CSV mensuales.
        start_date: Primer mes incluido, en formato ``AAAA-MM-DD``.
        end_date: Último mes incluido, en formato ``AAAA-MM-DD``.

    Returns:
        Rutas de los meses entre ``start_date`` y ``end_date``, ambos incluidos.
    """
    logger.info("Filtering by date range...")
    logger.debug(f"Start date: {start_date}")
    logger.debug(f"End date: {end_date}")

    # Tabla ruta-fecha para filtrar con pandas
    df_paths = pandas.DataFrame(paths, columns=["path"])
    df_dates = df_paths.path.str.extract(r"(\d{4}_\d{2})").rename(columns={0: "date"})
    df_paths["date"] = pandas.to_datetime(df_dates["date"], format="%Y_%m")

    # Filtro por fecha
    df_filtered = df_paths.query("(date >= @start_date) and (date <= @end_date)")
    paths_list = df_filtered.path.to_list()

    return paths_list


# MARK: Download files
def download_file(url: str, destination: Path) -> None:
    """Descarga un archivo del portal a ``destination`` y hace una pausa para no saturarlo.

    Args:
        url: Dirección completa del archivo.
        destination: Ruta local; la carpeta se crea si no existe (``data/`` no está en Git).
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    r = get_url(url)
    with open(destination, "wb") as f:
        f.write(r.content)
    time.sleep(1)  # pausa entre descargas para no saturar el portal de MiBici


def download_files(
    base_url: str, path_list: list, input_path: Path, overwrite: bool = False
) -> None:
    """Descarga a ``input_path`` los archivos de ``path_list`` (lo usa ``make snapshot``).

    Args:
        base_url: Dominio del portal.
        path_list: Rutas relativas de los archivos (las que devuelve ``get_paths``).
        input_path: Carpeta de destino; se crea si no existe.
        overwrite: Si es falso, los archivos ya presentes no se vuelven a pedir.
    """
    input_path = Path(input_path)
    input_path.mkdir(parents=True, exist_ok=True)
    for filepath in tqdm(path_list, desc="Downloading files", unit=" files"):
        destination = input_path / filepath.split("/")[-1]
        if destination.exists() and not overwrite:
            continue
        download_file(base_url + filepath, destination)


# MARK: Read files
def detect_encoding(filepath: Path) -> str:
    """Detecta la codificación de un CSV de MiBici.

    Los archivos mensuales no usan todos la misma: 16 de los 24 meses están en UTF-8 y
    8 en Latin-1. Leer un archivo UTF-8 como Latin-1 cambia los acentos por otros caracteres.

    Args:
        filepath: Ruta del CSV.

    Returns:
        ``"utf-8-sig"`` si el archivo es UTF-8 válido; si no, ``"latin-1"``.
    """
    with open(filepath, "rb") as f:
        content = f.read()
    try:
        content.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


def detect_date_format(sample: str) -> str:
    """Devuelve el patrón ``strptime`` con que se escribió una fecha de MiBici.

    Septiembre y octubre de 2024 usan día/mes/año sin segundos (``01/09/2024 08:15``); el
    resto, ``AAAA-MM-DD HH:MM:SS``. El patrón se guarda en el esquema de cada recurso del
    descriptor, así un tercer formato en un mes futuro falla en la validación y no se lee mal.

    Args:
        sample: Una fecha tal como aparece en el CSV.

    Returns:
        ``"%d/%m/%Y %H:%M"`` o ``"%Y-%m-%d %H:%M:%S"``.
    """
    if "/" in sample:
        return "%d/%m/%Y %H:%M"
    return "%Y-%m-%d %H:%M:%S"


def parse_dates(col: pandas.Series) -> pandas.Series:
    """Convierte una columna de texto a fecha y hora.

    Septiembre y octubre de 2024 usan día/mes/año (``01/09/2024 08:15``); el resto,
    año-mes-día. Sin ``dayfirst``, el 1 de septiembre se leería como el 9 de enero.

    Args:
        col: Columna con las fechas como texto.

    Returns:
        Columna de tipo fecha; los valores que no se pueden leer quedan como ``NaT``.
    """
    sample = str(col.dropna().iloc[0])
    if "/" in sample:
        return pandas.to_datetime(col, dayfirst=True, errors="coerce")
    return pandas.to_datetime(col, format="ISO8601", errors="coerce")


def read_raw_data(
    input_path: Path, file_type: str = "month", allowed_files: list | None = None
) -> pandas.DataFrame:
    """Lee los CSV crudos de ``input_path`` y los concatena en una sola tabla.

    Los archivos se leen en orden alfabético para que el resultado (y por tanto el hash del
    Parquet) sea el mismo en cualquier sistema operativo. De los archivos mensuales se
    conservan solo las columnas del contrato (``TRIP_COLUMNS``), así que los datos personales
    nunca pasan de aquí.

    Args:
        input_path: Carpeta con los CSV descargados.
        file_type: ``"month"`` para los viajes mensuales o ``"nom"`` para la nomenclatura.
        allowed_files: Si se indica, solo se leen los archivos con estos nombres (los del
            rango de fechas).

    Returns:
        Viajes de todos los meses concatenados, o la nomenclatura. Tabla vacía si
        ``file_type`` no es válido o no hay archivos que leer.
    """
    input_path = Path(input_path)
    filenames = sorted(path.name for path in input_path.iterdir())
    if allowed_files is not None:  # solo los archivos del rango de fechas
        filenames = [f for f in filenames if f in allowed_files]
    if file_type == "month":
        filenames = [f for f in filenames if "datos_abiertos" in f]
    elif file_type == "nom":
        filenames = [f for f in filenames if "nomenclatura" in f][:1]
    else:
        logger.error('File type must be "month" or "nom".')
        return pandas.DataFrame()

    df_list = []
    for filename in filenames:
        filepath = input_path / filename
        df = pandas.read_csv(filepath, encoding=detect_encoding(filepath))
        if file_type == "month":
            # Solo las columnas del contrato: los datos personales nunca se guardan
            df = norm_cols(df)[list(TRIP_COLUMNS)]
            df["inicio_del_viaje"] = parse_dates(df["inicio_del_viaje"])
            df["fin_del_viaje"] = parse_dates(df["fin_del_viaje"])
        df_list.append(df)

    if not df_list:
        logger.warning(f"No hay archivos de tipo '{file_type}' en {input_path}")
        return pandas.DataFrame()
    if file_type == "nom":
        return df_list[0]
    return pandas.concat(df_list, ignore_index=True)


# MARK: Normalize column names
def normalize_text(col: str) -> str:
    """Pasa un nombre de columna a minúsculas y sin acentos (``Año_de_nacimiento`` → ``ano_de_nacimiento``).

    Args:
        col: Nombre original de la columna.

    Returns:
        Nombre en minúsculas y solo con caracteres ASCII.
    """
    col_lower = col.lower()
    col_norm = unicodedata.normalize("NFKD", col_lower)
    col_ascii = col_norm.encode("ascii", "ignore").decode("ascii")
    return col_ascii


def norm_cols(df: pandas.DataFrame) -> pandas.DataFrame:
    """Normaliza los nombres de todas las columnas con ``normalize_text``.

    Los CSV de MiBici cambian mayúsculas y acentos entre meses; con nombres uniformes se
    pueden seleccionar las columnas del contrato de ``config.py``.

    Args:
        df: Tabla recién leída de un CSV.

    Returns:
        La misma tabla con los nombres de columna normalizados.
    """
    cols = list(df.columns)
    cols_norm = [normalize_text(col) for col in cols]
    df.columns = cols_norm
    return df


# MARK: Clean trips
def clean_trips(df_months: pandas.DataFrame, df_nom: pandas.DataFrame) -> pandas.DataFrame:
    """Elimina los viajes que no representan traslados reales (Etapa 1 del pipeline).

    Quita los viajes con fechas que no se pudieron leer, con duración negativa o mayor
    a ``MAX_TRIP_MINUTES``, los retiros y devoluciones en la misma estación en menos de
    ``MIN_TRIP_MINUTES``, las salidas en ``NO_SERVICE_HOURS`` (el sistema no presta
    bicicletas) y los viajes con estaciones que no están en la nomenclatura. Al final,
    elimina los duplicados por ``viaje_id``.

    Args:
        df_months: Viajes con las columnas de ``TRIP_COLUMNS`` y fechas ya convertidas.
        df_nom: Nomenclatura con la columna ``id``.

    Returns:
        Viajes limpios, con los identificadores como enteros.
    """
    duration = (df_months["fin_del_viaje"] - df_months["inicio_del_viaje"]).dt.total_seconds() / 60
    invalid = (
        df_months["inicio_del_viaje"].isna()
        | df_months["fin_del_viaje"].isna()
        | (duration < 0)
        | (duration > MAX_TRIP_MINUTES)
        | ((df_months["origen_id"] == df_months["destino_id"]) & (duration < MIN_TRIP_MINUTES))
        | df_months["inicio_del_viaje"].dt.hour.isin(NO_SERVICE_HOURS)
        # Estaciones que no están en la nomenclatura (por ejemplo, 405 y 407)
        | ~df_months["origen_id"].isin(df_nom["id"])
        | ~df_months["destino_id"].isin(df_nom["id"])
    )
    df_clean = df_months.loc[~invalid].drop_duplicates("viaje_id")
    df_clean = df_clean.astype({"viaje_id": "int64", "origen_id": "int64", "destino_id": "int64"})
    logger.info(f"Cleaning: {len(df_clean):,} trips kept out of {len(df_months):,}")
    return df_clean


# MARK: Process and write
def process_raw(
    input_path: Path, filenames: list[str]
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Lee, normaliza y limpia los CSV crudos (Etapas de normalización y limpieza).

    Lo comparten ``make data`` y ``make snapshot`` para que los Parquet que describe el
    descriptor sean exactamente los que produce la ingesta.

    Args:
        input_path: Carpeta con los CSV descargados.
        filenames: Nombres de los archivos del periodo (meses y nomenclatura).

    Returns:
        Viajes limpios con las columnas de ``TRIP_COLUMNS`` y nomenclatura con las de
        ``STATION_COLUMNS``.
    """
    logger.info("Processing dataset...")
    df_months = read_raw_data(input_path=input_path, file_type="month", allowed_files=filenames)
    df_nom = read_raw_data(input_path=input_path, file_type="nom", allowed_files=filenames)

    # Nombres de columna uniformes y solo las columnas del contrato
    df_months = norm_cols(df_months)
    df_nom = norm_cols(df_nom)[list(STATION_COLUMNS)]

    df_months = clean_trips(df_months, df_nom)
    return df_months, df_nom


def write_processed(df_months: pandas.DataFrame, df_nom: pandas.DataFrame) -> None:
    """Escribe los dos Parquet de ``data/processed/`` (``TRIPS_PATH`` y ``STATIONS_PATH``)."""
    logger.info("Saving processed dataset...")
    TRIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_months.to_parquet(TRIPS_PATH, index=False)
    df_nom.to_parquet(STATIONS_PATH, index=False)
    logger.success("Processing dataset complete.")


# MARK: Data version
def file_hash(filepath: Path) -> str:
    """Calcula el hash SHA-256 de un archivo: el mismo contenido da siempre el mismo hash.

    Args:
        filepath: Ruta del archivo.

    Returns:
        Hash en hexadecimal.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_package(datapackage_path: Path = DATAPACKAGE_PATH) -> Package:
    """Carga el descriptor ``datapackage.json`` y comprueba que es un Data Package válido.

    Args:
        datapackage_path: Ruta del descriptor; las rutas de sus recursos son relativas a la
            raíz del repositorio.

    Returns:
        Paquete con un recurso por CSV crudo y uno por Parquet procesado.

    Raises:
        DataVersionError: Si el descriptor no existe o no cumple el estándar.
    """
    if not Path(datapackage_path).exists():
        raise DataVersionError(
            f"No existe {datapackage_path}. Genera la versión de datos con `make snapshot`."
        )
    try:
        # Solo comprueba la estructura del descriptor; los archivos se verifican después
        package = Package(str(datapackage_path))
    except FrictionlessException as error:
        raise DataVersionError(
            f"{Path(datapackage_path).name} no es un Data Package válido: {error.error.message}"
        ) from error
    window = package.custom.get("mibici", {})
    if (window.get("start_date"), window.get("end_date")) != (START_DATE, END_DATE):
        raise DataVersionError(
            f"El descriptor cubre {window.get('start_date')} a {window.get('end_date')} y config.py "
            f"pide {START_DATE} a {END_DATE}. Regenera la versión con `make snapshot` o corrige config.py."
        )
    return package


def raw_resources(package: Package) -> list[Resource]:
    """Recursos de los CSV crudos (meses y nomenclatura), en orden alfabético de archivo."""
    resources = [r for r in package.resources if r.name not in (TRIPS_RESOURCE, STATIONS_RESOURCE)]
    return sorted(resources, key=lambda r: Path(r.path).name)


def expected_hash(resource: Resource) -> str:
    """Hash SHA-256 (hexadecimal) declarado en el recurso; el descriptor lo guarda como ``sha256:<hex>``."""
    declared = resource.hash or ""
    algorithm, _, digest = declared.partition(":")
    if algorithm != "sha256" or not digest:
        raise DataVersionError(
            f"El recurso '{resource.name}' no declara un hash sha256 (tiene '{declared}')."
        )
    return digest


def verify_file(resource: Resource) -> None:
    """Comprueba que el archivo local de un recurso tiene el tamaño y el hash del descriptor.

    Args:
        resource: Recurso del descriptor con ``path`` local, ``bytes`` y ``hash``.

    Raises:
        DataVersionError: Si el archivo no existe o su tamaño o hash difieren.
    """
    path = Path(resource.normpath)
    if not path.exists():
        raise DataVersionError(f"Falta el archivo {path}.")
    size = path.stat().st_size
    if size != resource.bytes:
        raise DataVersionError(
            f"{path.name}: el descriptor espera {resource.bytes:,} bytes y el archivo tiene {size:,}."
        )
    digest = file_hash(path)
    if digest != expected_hash(resource):
        raise DataVersionError(
            f"{path.name}: el descriptor espera sha256 {expected_hash(resource)} y el archivo tiene {digest}."
        )


def ensure_raw_files(package: Package) -> list[str]:
    """Descarga lo que falte y verifica cada CSV crudo contra el descriptor antes de procesarlo.

    Un archivo presente con tamaño y hash correctos no se vuelve a descargar. Uno presente
    pero distinto (descarga interrumpida, edición local) se descarga de nuevo una sola vez.
    Si tras descargarlo sigue sin coincidir, el portal lo republicó con otro contenido y la
    ingesta termina con error: no hay opción para omitir la verificación, el remedio es una
    versión nueva de los datos (``make snapshot``) revisada por *Pull Request*.

    Args:
        package: Descriptor cargado con ``load_package``.

    Returns:
        Nombres de los archivos crudos verificados.

    Raises:
        DataVersionError: Si algún archivo no coincide con el descriptor tras descargarlo.
    """
    filenames = []
    for resource in tqdm(raw_resources(package), desc="Verifying raw files", unit=" files"):
        path = Path(resource.normpath)
        url = resource.sources[0]["path"]
        if not path.exists():
            logger.info(f"Descargando {path.name}...")
            download_file(url, path)
        try:
            verify_file(resource)
        except DataVersionError as first_error:
            logger.warning(f"{first_error} Se descarga de nuevo desde {url}")
            download_file(url, path)
            try:
                verify_file(resource)
            except DataVersionError as second_error:
                raise DataVersionError(
                    f"{second_error}\n  El portal republicó el archivo o el descriptor está desactualizado. "
                    "Genera una versión nueva de los datos con `make snapshot` y revísala por Pull Request."
                ) from second_error
        filenames.append(path.name)
    logger.success(f"{len(filenames)} archivos crudos coinciden con {DATAPACKAGE_PATH.name}")
    return filenames


def validate_raw_sample(package: Package, rows: int = RAW_VALIDATION_ROWS) -> None:
    """Valida con frictionless las primeras ``rows`` filas de cada CSV contra su esquema.

    Detecta un cambio de formato (columnas renombradas, un tercer formato de fecha, otra
    codificación) antes de limpiar, con un mensaje claro en lugar de fechas vacías que la
    limpieza descartaría en silencio. La validación completa de todas las filas la hace
    ``make validate`` o ``make snapshot``.

    Args:
        package: Descriptor cargado con ``load_package``.
        rows: Filas de cada archivo que se validan.

    Raises:
        DataVersionError: Si alguna fila de la muestra no cumple el esquema.
    """
    problems = []
    for resource in tqdm(raw_resources(package), desc="Validating schemas", unit=" files"):
        # Con una muestra frictionless no compara hash ni bytes; eso ya lo hizo verify_file
        report = resource.validate(limit_rows=rows)
        if not report.valid:
            for error in report.flatten(["rowNumber", "fieldName", "type", "message"])[:5]:
                problems.append(f"{resource.name}: {error[3]}")
    if problems:
        raise DataVersionError(
            "Los archivos crudos no cumplen el esquema del descriptor:\n  " + "\n  ".join(problems)
        )
    logger.success("Las muestras de los archivos crudos cumplen el esquema del descriptor")


def verify_processed(package: Package) -> None:
    """Comprueba que los Parquet recién escritos coinciden con los que describe el descriptor.

    Si los CSV crudos coincidían y los Parquet no, cambiaron las reglas de limpieza o el
    entorno: la versión de datos debe regenerarse con ``make snapshot`` para que el cambio
    quede en el descriptor y se revise.

    Args:
        package: Descriptor cargado con ``load_package``.

    Raises:
        DataVersionError: Si el número de filas, el tamaño o el hash de un Parquet difieren.
    """
    for name in (TRIPS_RESOURCE, STATIONS_RESOURCE):
        resource = package.get_resource(name)
        path = Path(resource.normpath)
        rows = pq.read_metadata(path).num_rows
        if rows != resource.rows:
            raise DataVersionError(
                f"{path.name}: el descriptor espera {resource.rows:,} filas y el Parquet tiene {rows:,}. "
                "Cambiaron las reglas de limpieza o el entorno: regenera la versión con `make snapshot`."
            )
        try:
            verify_file(resource)
        except DataVersionError as error:
            raise DataVersionError(
                f"{error}\n  Cambiaron las reglas de limpieza o el entorno: regenera la versión con `make snapshot`."
            ) from error
    logger.success(f"Los Parquet procesados coinciden con {DATAPACKAGE_PATH.name}")


# MARK: MLflow
@contextmanager
def _ingest_run(version: str, datapackage_path: Path):
    """Registra la ingesta como run de MLflow solo bajo ``mlflow run`` (``MLFLOW_RUN_ID``).

    Reanuda el run que creó la línea de comandos con el nombre, las etiquetas y la nota de
    ``experiments.py``, la versión de datos como parámetro y el descriptor como artefacto.
    """
    if not os.getenv("MLFLOW_RUN_ID"):
        yield
        return
    import mlflow

    from mibici.experiments import (
        ExperimentMismatchError,
        open_run,
        run_name,
        run_note,
        standard_tags,
    )

    with ExitStack() as stack:
        try:
            stack.enter_context(
                open_run(
                    "ingesta",
                    run_name("ingesta", label=version),
                    standard_tags("ingesta"),
                    run_note("ingesta", version),
                )
            )
        except ExperimentMismatchError as error:
            logger.error(str(error))
            raise typer.Exit(code=1) from error
        mlflow.log_param("version_datos", version)
        mlflow.log_artifact(str(datapackage_path), artifact_path="datos")
        yield


# MARK: Main fn
@app.command()
def main(datapackage_path: Path = DATAPACKAGE_PATH) -> None:
    """Reconstruye la versión de datos de datapackage.json: descarga, verifica, limpia y guarda.

    Las rutas de los recursos son relativas a la raíz del repositorio y las reglas de
    limpieza salen de ``config.py``.
    """
    try:
        package = load_package(datapackage_path)
        logger.info(
            f"Versión de datos {package.version} ({package.name}) de {Path(datapackage_path).name}"
        )
        with _ingest_run(str(package.version), Path(datapackage_path)):
            # 1. Descarga lo que falte y verifica tamaño y hash antes de procesar
            filenames = ensure_raw_files(package)
            # 2. Valida una muestra de cada archivo contra el esquema declarado
            validate_raw_sample(package)
            # 3. Normaliza, limpia y escribe los Parquet
            df_months, df_nom = process_raw(RAW_DATA_DIR, filenames)
            write_processed(df_months, df_nom)
            # 4. Comprueba que los Parquet son los que describe el descriptor
            verify_processed(package)
    except DataVersionError as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error
    logger.success(f"Datos reconstruidos: versión {package.version}")


if __name__ == "__main__":
    app()
