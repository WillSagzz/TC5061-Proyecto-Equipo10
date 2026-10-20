"""Versión de los datos como descriptor Frictionless Data Package (``datapackage.json``).

Uso desde la raíz del repositorio:

    python -m mibici.datapackage snapshot --version 2026-10   # make snapshot VERSION=2026-10
    python -m mibici.datapackage validate                     # make validate

``snapshot`` congela una versión nueva de los datos: descubre en el portal los enlaces del
periodo, descarga los archivos, describe cada uno (tamaño, hash SHA-256, codificación,
dialecto y esquema de columnas), corre la limpieza de ``dataset.py``, describe los Parquet
resultantes y escribe el descriptor. El descriptor se versiona en Git y cada cambio entra a
``main`` por *Pull Request*; ``make data`` (``dataset.py``) reconstruye y verifica esa versión.

``validate`` comprueba los archivos locales contra el descriptor: tamaño y hash de cada uno,
todas las filas de los CSV contra su esquema con frictionless y los Parquet procesados.
"""

import datetime
import json
from pathlib import Path

from frictionless import Package, Resource, Schema
from loguru import logger
import pandas
import pyarrow.parquet as pq
from tqdm import tqdm
import typer

from mibici.config import (
    DATAPACKAGE_PATH,
    END_DATE,
    MIBICI_BASE_URL,
    MIBICI_DATA_PAGE,
    PROJ_ROOT,
    RAW_DATA_DIR,
    START_DATE,
    STATION_COLUMNS,
    STATIONS_PATH,
    TEAM,
    TRIP_COLUMNS,
    TRIPS_PATH,
)
from mibici.dataset import (
    NOMENCLATURA_RESOURCE,
    STATIONS_RESOURCE,
    TRIPS_RESOURCE,
    DataVersionError,
    detect_date_format,
    detect_encoding,
    download_files,
    file_hash,
    filter_paths,
    get_paths,
    get_url,
    load_package,
    normalize_text,
    process_raw,
    raw_resources,
    verify_file,
    verify_processed,
    write_processed,
)

app = typer.Typer(help="Versión de los datos de MiBici como Frictionless Data Package.")

# Metadatos del paquete (ficha del conjunto de datos). El portal no publica una licencia
# explícita: la AMIM gestiona la información como datos abiertos y el proyecto la usa solo
# con fines académicos, sin redistribuirla (sección 4 de la propuesta).
PACKAGE_NAME = "mibici-viajes"
PACKAGE_TITLE = "Viajes y estaciones de MiBici (Área Metropolitana de Guadalajara)"
PACKAGE_LICENSES = [
    {
        "name": "other-open",
        "title": "Datos abiertos de la AMIM, sin licencia explícita; uso académico sin redistribución",
        "path": "https://mibici.net/es/datos-abiertos/",
    }
]
PACKAGE_SOURCES = [{"title": "Portal de datos abiertos de MiBici", "path": MIBICI_DATA_PAGE}]
PACKAGE_CONTRIBUTORS = [{"title": f"TC5061 {TEAM}", "role": "author"}]

# Columnas de fecha de los CSV crudos (nombres normalizados): llevan el formato en el esquema
DATE_COLUMNS = {"inicio_del_viaje", "fin_del_viaje"}
# Marcas de valor faltante en los CSV crudos (Table Schema solo asume la celda vacía)
MISSING_VALUES = ["", "NA"]
# Tipos del contrato de datos de config.py -> tipos de Table Schema
_SCHEMA_TYPES = {"integer": "integer", "datetime": "datetime", "float": "number", "text": "string"}


@app.callback()
def main() -> None:
    """Versión de los datos de MiBici: `snapshot` la congela y `validate` la comprueba."""


# MARK: Describe
def resource_name(filename: str) -> str:
    """Nombre del recurso a partir del archivo: ``datos_abiertos_2024_09.csv`` → ``datos-abiertos-2024-09``."""
    if "nomenclatura" in filename:
        return NOMENCLATURA_RESOURCE
    return Path(filename).stem.replace("_", "-")


def describe_raw(path: Path, url: str) -> Resource:
    """Describe un CSV crudo: codificación, dialecto, esquema, tamaño, hash y URL de origen.

    frictionless infiere la codificación, el dialecto y las columnas de una muestra; las
    dos columnas de fecha se declaran ``datetime`` con el patrón del archivo
    (``detect_date_format``), porque cambia entre meses y una inferencia las dejaría como
    texto. Tamaño y hash se calculan sobre el archivo completo.

    Args:
        path: CSV descargado en ``data/raw/``.
        url: Dirección del portal de donde se descargó.

    Returns:
        Recurso con ``path`` relativo a la raíz del repositorio.
    """
    # La codificación se fija con detect_encoding (UTF-8 o Latin-1) en lugar de inferirla:
    # la inferencia estadística puede confundir Latin-1 con otra página de códigos
    encoding = detect_encoding(path)
    resource = Resource(
        path=path.relative_to(PROJ_ROOT).as_posix(),
        basepath=str(PROJ_ROOT),
        encoding="utf-8" if encoding == "utf-8-sig" else "iso8859-1",
    )
    resource.infer()
    resource.name = resource_name(path.name)
    resource.mediatype = "text/csv"
    # Los CSV marcan los valores faltantes (año de nacimiento) con "NA", además de la celda vacía
    resource.schema.missing_values = MISSING_VALUES
    sample = pandas.read_csv(path, nrows=1, encoding=encoding)
    for field in resource.schema.fields:
        if normalize_text(field.name) in DATE_COLUMNS:
            value = str(sample[field.name].iloc[0])
            resource.schema.set_field_type(field.name, "datetime")
            resource.schema.get_field(field.name).format = detect_date_format(value)
    resource.bytes = path.stat().st_size
    resource.hash = f"sha256:{file_hash(path)}"
    resource.sources = [{"title": path.name, "path": url}]
    return resource


def describe_parquet(path: Path, name: str, title: str, columns: dict[str, str]) -> Resource:
    """Describe un Parquet procesado con el contrato de datos de ``config.py`` como esquema.

    Args:
        path: Parquet escrito por la ingesta en ``data/processed/``.
        name: Nombre del recurso.
        title: Descripción corta.
        columns: Contrato de columnas y tipos (``TRIP_COLUMNS`` o ``STATION_COLUMNS``).

    Returns:
        Recurso con esquema, número de filas, tamaño y hash.
    """
    schema = Schema.from_descriptor(
        {
            "fields": [
                {"name": column, "type": _SCHEMA_TYPES[kind]} for column, kind in columns.items()
            ]
        }
    )
    resource = Resource(
        name=name,
        title=title,
        path=path.relative_to(PROJ_ROOT).as_posix(),
        basepath=str(PROJ_ROOT),
        format="parquet",
        schema=schema,
    )
    resource.rows = pq.read_metadata(path).num_rows
    resource.bytes = path.stat().st_size
    resource.hash = f"sha256:{file_hash(path)}"
    return resource


def build_package(
    raw_files: list[tuple[Path, str]], version: str, start_date: str, end_date: str
) -> Package:
    """Arma el descriptor con los recursos crudos y los Parquet procesados.

    Args:
        raw_files: Pares (archivo descargado, URL del portal), meses y nomenclatura.
        version: Versión de datos (el mes en que se congeló, por ejemplo ``2026-10``).
        start_date: Primer mes incluido, ``AAAA-MM-DD``.
        end_date: Último mes incluido, ``AAAA-MM-DD``.

    Returns:
        Paquete listo para escribirse con ``to_json``.
    """
    months = [path.name for path, _ in raw_files if "datos_abiertos" in path.name]
    resources = [describe_raw(path, url) for path, url in raw_files]
    resources.append(
        describe_parquet(
            TRIPS_PATH,
            TRIPS_RESOURCE,
            "Viajes limpios del periodo, sin datos personales (lo escribe make data)",
            TRIP_COLUMNS,
        )
    )
    resources.append(
        describe_parquet(
            STATIONS_PATH,
            STATIONS_RESOURCE,
            "Catálogo de estaciones con capacidad de anclajes (lo escribe make data)",
            STATION_COLUMNS,
        )
    )
    package = Package(
        name=PACKAGE_NAME,
        title=PACKAGE_TITLE,
        description=(
            f"{len(months)} archivos mensuales de viajes ({start_date[:7]} a {end_date[:7]}) y el "
            "catálogo de estaciones del portal de datos abiertos de MiBici, con el hash SHA-256, "
            "el tamaño, la codificación y el esquema de cada archivo, más los dos Parquet que "
            "produce la ingesta (make data). Las columnas de usuario, género y año de "
            "nacimiento existen en los CSV crudos y se descartan al procesar."
        ),
        version=version,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        homepage=MIBICI_DATA_PAGE,
        licenses=PACKAGE_LICENSES,
        sources=PACKAGE_SOURCES,
        contributors=PACKAGE_CONTRIBUTORS,
        resources=resources,
    )
    package.custom["mibici"] = {"start_date": start_date, "end_date": end_date}
    return package


# MARK: Validate
def validate_raw_full(package: Package) -> None:
    """Valida con frictionless todas las filas de cada CSV crudo contra su esquema, hash y tamaño.

    Args:
        package: Descriptor cargado con ``load_package``.

    Raises:
        DataVersionError: Si algún archivo no cumple el descriptor.
    """
    problems = []
    for resource in tqdm(raw_resources(package), desc="Validating raw files", unit=" files"):
        report = resource.validate()
        if not report.valid:
            for error in report.flatten(["rowNumber", "fieldName", "type", "message"])[:5]:
                problems.append(f"{resource.name}: {error[3]}")
    if problems:
        raise DataVersionError(
            "Los archivos crudos no cumplen el descriptor:\n  " + "\n  ".join(problems)
        )
    logger.success("Todas las filas de los archivos crudos cumplen el descriptor")


# MARK: Commands
@app.command()
def snapshot(
    version: str = typer.Option(
        None, help="Versión de datos; por defecto el mes actual, por ejemplo 2026-10."
    ),
    start: str = typer.Option(START_DATE, help="Primer mes incluido, AAAA-MM-DD."),
    end: str = typer.Option(END_DATE, help="Último mes incluido, AAAA-MM-DD."),
    reuse_raw: bool = typer.Option(
        False,
        help="No volver a descargar los archivos que ya estén en data/raw/ (describe lo que hay en disco).",
    ),
    datapackage_path: Path = DATAPACKAGE_PATH,
) -> None:
    """Congela una versión nueva de los datos y escribe datapackage.json.

    Descubre los enlaces del periodo en el portal, descarga los archivos, los describe, corre
    la limpieza y describe los Parquet. El descriptor resultante entra a main por Pull Request.
    """
    version = version or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
    logger.info(f"Congelando la versión de datos {version} ({start[:7]} a {end[:7]})")

    # 1. Enlaces del periodo y del catálogo, descubiertos en el portal
    file_paths = get_paths(get_url(MIBICI_DATA_PAGE).text)
    paths_list = filter_paths(file_paths["Months"], start, end)
    paths_list.append(file_paths["Nom"][0])

    # 2. Descarga (completa, salvo --reuse-raw) a data/raw/
    download_files(MIBICI_BASE_URL, paths_list, RAW_DATA_DIR, overwrite=not reuse_raw)
    raw_files = sorted(
        ((RAW_DATA_DIR / p.split("/")[-1], MIBICI_BASE_URL + p) for p in paths_list),
        key=lambda pair: pair[0].name,
    )

    # 3. Limpieza y Parquet, con las mismas funciones que make data
    df_months, df_nom = process_raw(RAW_DATA_DIR, [path.name for path, _ in raw_files])
    write_processed(df_months, df_nom)

    # 4. Descriptor
    logger.info("Describiendo los archivos...")
    package = build_package(raw_files, version, start, end)
    with open(datapackage_path, "w", encoding="utf-8") as file:
        # ensure_ascii=False para que los nombres con acento (Año_de_nacimiento) se lean en Git
        json.dump(package.to_descriptor(), file, indent=2, ensure_ascii=False)
        file.write("\n")
    logger.success(f"Versión de datos {version} escrita en {datapackage_path.name}")

    # 5. Validación completa de lo que se acaba de escribir
    validate(datapackage_path)


@app.command()
def validate(datapackage_path: Path = DATAPACKAGE_PATH) -> None:
    """Comprueba los archivos locales contra datapackage.json: hashes, tamaños y esquemas."""
    try:
        package = load_package(datapackage_path)
        for resource in raw_resources(package):
            verify_file(resource)
        validate_raw_full(package)
        verify_processed(package)
    except DataVersionError as error:
        logger.error(str(error))
        raise typer.Exit(code=1) from error
    logger.success(f"La versión de datos {package.version} es válida")


if __name__ == "__main__":
    app()
