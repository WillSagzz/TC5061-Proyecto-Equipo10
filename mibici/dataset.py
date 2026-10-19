import datetime
import hashlib
import json
import os
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
# (Website or File)
def get_url(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()  # si el portal responde con error, no se guarda una página de error como CSV
    return r


# MARK: File paths
def get_paths(site_html: str):
    logger.info("Parsing HTML for file links...")
    soup = BeautifulSoup(site_html, "html.parser")
    links = soup.find_all("a")

    parent_dir = "/site/assets/files"
    file_paths = {"Months": [], "Nom": []}
    for link in links:
        link_href = str(
            link.get("href")
        )  # Can work without the str(). Just added it to remove warnings in VS Code.
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
def filter_paths(paths: list, start_date: str, end_date: str):
    logger.info("Filtering by date range...")
    logger.debug(f"Start date: {start_date}")
    logger.debug(f"End date: {end_date}")

    # Path & date dataframe
    df_paths = pandas.DataFrame(paths, columns=["path"])
    df_dates = df_paths.path.str.extract(r"(\d{4}_\d{2})").rename(columns={0: "date"})
    df_paths["date"] = pandas.to_datetime(df_dates["date"], format="%Y_%m")

    # Filter by date
    df_filtered = df_paths.query("(date >= @start_date) and (date <= @end_date)")
    paths_list = df_filtered.path.to_list()

    return paths_list


# MARK: Download files
def download_files(base_url: str, path_list: list, input_path: Path):
    os.makedirs(input_path, exist_ok=True)  # data/ no está en Git: en un clon limpio no existe
    existing_filenames = os.listdir(input_path)

    for filepath in tqdm(path_list, desc="Downloading files", unit=" files"):
        filename = filepath.split("/")[-1]
        input_filepath = f"{input_path}/{filename}"

        # If the file has already been downloaded, no need to do so again
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


def read_raw_data(input_path: Path, file_type: str = "month", allowed_files: list | None = None):
    filenames = os.listdir(input_path)
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
        filepath = f"{input_path}/{filename}"
        df = pandas.read_csv(filepath, encoding=detect_encoding(filepath))
        if file_type == "month":
            # Solo las columnas del contrato: los datos personales nunca se guardan
            df = norm_cols(df)[list(TRIP_COLUMNS)]
            df["inicio_del_viaje"] = parse_dates(df["inicio_del_viaje"])
            df["fin_del_viaje"] = parse_dates(df["fin_del_viaje"])
        df_list.append(df)

    if file_type == "nom":
        return df
    return pandas.concat(df_list, ignore_index=True)


# MARK: Normalize column names
def normalize_text(col: str):
    col_lower = col.lower()
    col_norm = unicodedata.normalize("NFKD", col_lower)
    col_ascii = col_norm.encode("ascii", "ignore").decode("ascii")
    return col_ascii


def norm_cols(df):
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
):
    logger.info("Fetching dataset...")
    logger.info(f"Input path: {input_path}")

    # Fetch website
    logger.info("Fetching website...")
    r = get_url(MIBICI_DATA_PAGE)

    # Parse html
    file_paths = get_paths(r.text)

    # Filter by date
    paths_list = filter_paths(file_paths["Months"], start_date, end_date)
    paths_list.append(file_paths["Nom"][0])

    # Download files
    download_files(MIBICI_BASE_URL, paths_list, input_path)

    logger.info("Processing dataset...")

    # Read files (only the ones in the date range)
    filenames = [p.split("/")[-1] for p in paths_list]
    df_months = read_raw_data(input_path=input_path, file_type="month", allowed_files=filenames)
    df_nom = read_raw_data(input_path=input_path, file_type="nom", allowed_files=filenames)

    # Standarize column names
    df_months = norm_cols(df_months)
    df_nom = norm_cols(df_nom)[list(STATION_COLUMNS)]

    # Clean trips
    df_months = clean_trips(df_months, df_nom)

    # Write to parquet
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
