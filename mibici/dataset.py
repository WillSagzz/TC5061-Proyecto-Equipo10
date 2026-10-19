"""Ingesta reproducible de los datos abiertos de MiBici.

Uso desde la raíz del repositorio (``make data``):

    python -m mibici.dataset

Descubre los enlaces del periodo en el portal, descarga a ``data/raw/`` lo que falte, normaliza
y limpia los viajes, escribe los Parquet de ``data/processed/`` y regenera ``data_manifest.json``
con el hash SHA-256 de cada archivo y la versión de datos. Rutas, fechas y reglas de limpieza
salen de ``config.py``.
"""

import datetime
import hashlib
import json
from pathlib import Path
import time
import unicodedata

from bs4 import BeautifulSoup
from loguru import logger
import pandas
import requests
from tqdm import tqdm
import typer

from mibici.config import (
    DATA_DIR,
    DATA_MANIFEST_PATH,
    END_DATE,
    MAX_TRIP_MINUTES,
    MIBICI_BASE_URL,
    MIBICI_DATA_PAGE,
    MIN_TRIP_MINUTES,
    NO_SERVICE_HOURS,
    RAW_DATA_DIR,
    START_DATE,
    STATION_COLUMNS,
    STATIONS_PATH,
    TRIP_COLUMNS,
    TRIPS_PATH,
)

app = typer.Typer()


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
    página en lugar de mantener una lista a mano.

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
def download_files(base_url: str, path_list: list, input_path: Path) -> None:
    """Descarga a ``input_path`` los archivos que todavía no estén ahí.

    Los archivos ya descargados no se vuelven a pedir, así la ingesta se puede repetir sin
    costo y ``data/raw/`` funciona como copia inmutable de la que se calculan los hashes.

    Args:
        base_url: Dominio del portal.
        path_list: Rutas relativas de los archivos (las que devuelve ``get_paths``).
        input_path: Carpeta de destino; se crea si no existe (``data/`` no está en Git).
    """
    input_path = Path(input_path)
    input_path.mkdir(parents=True, exist_ok=True)
    existing_filenames = {path.name for path in input_path.iterdir()}

    for filepath in tqdm(path_list, desc="Downloading files", unit=" files"):
        filename = filepath.split("/")[-1]
        input_filepath = input_path / filename

        # Si el archivo ya se descargó, no hace falta repetirlo
        if filename in existing_filenames:
            continue

        file_url = base_url + filepath
        r = get_url(file_url)

        with open(input_filepath, "wb") as f:
            f.write(r.content)
        time.sleep(1)  # pausa entre descargas para no saturar el portal de MiBici


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
    Parquet y la versión de datos) sea el mismo en cualquier sistema operativo. De los
    archivos mensuales se conservan solo las columnas del contrato (``TRIP_COLUMNS``), así
    que los datos personales nunca pasan de aquí.

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


# MARK: Main fn
@app.command()
def main(
    data_path: Path = DATA_DIR,
    input_path: Path = RAW_DATA_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> None:
    """Ejecuta la ingesta completa: descubre, descarga, limpia, guarda en Parquet y versiona.

    Las opciones toman por defecto las rutas y fechas de ``config.py``.
    """
    logger.info("Fetching dataset...")
    logger.info(f"Input path: {input_path}")

    # Página de datos abiertos
    logger.info("Fetching website...")
    r = get_url(MIBICI_DATA_PAGE)

    # Enlaces a los archivos
    file_paths = get_paths(r.text)

    # Solo los meses del periodo, más el catálogo
    paths_list = filter_paths(file_paths["Months"], start_date, end_date)
    paths_list.append(file_paths["Nom"][0])

    # Descarga de lo que falte
    download_files(MIBICI_BASE_URL, paths_list, input_path)

    logger.info("Processing dataset...")

    # Lectura (solo los archivos del rango de fechas)
    filenames = [p.split("/")[-1] for p in paths_list]
    df_months = read_raw_data(input_path=input_path, file_type="month", allowed_files=filenames)
    df_nom = read_raw_data(input_path=input_path, file_type="nom", allowed_files=filenames)

    # Nombres de columna uniformes y solo las columnas del contrato
    df_months = norm_cols(df_months)
    df_nom = norm_cols(df_nom)[list(STATION_COLUMNS)]

    # Limpieza de viajes
    df_months = clean_trips(df_months, df_nom)

    # Escritura en Parquet
    logger.info("Saving processed dataset...")
    TRIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_months.to_parquet(TRIPS_PATH, index=False)
    df_nom.to_parquet(STATIONS_PATH, index=False)

    logger.success("Processing dataset complete.")

    logger.info("Recording data version...")
    raw_hashes = {f: file_hash(Path(input_path) / f) for f in sorted(filenames)}
    processed_hashes = {path.name: file_hash(path) for path in (TRIPS_PATH, STATIONS_PATH)}
    # La versión depende solo del contenido: mismos datos, misma versión
    version = hashlib.sha256(json.dumps(processed_hashes, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    manifest_dict = {
        "version": version,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%d/%m/%Y, %H:%M:%S UTC"
        ),
        "date_range": {"start_date": start_date, "end_date": end_date},
        "record_count_months": df_months.shape[0],
        "record_count_nom": df_nom.shape[0],
        "raw_files_sha256": raw_hashes,
        "processed_files_sha256": processed_hashes,
    }
    # Fuera de data/ para poder versionarlo en Git
    with open(DATA_MANIFEST_PATH, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    logger.success(f"Data version {version} saved in {DATA_MANIFEST_PATH.name}")


if __name__ == "__main__":
    app()
