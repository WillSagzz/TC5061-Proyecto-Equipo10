from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from feature_engineering.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, DATA_DIR, START_DATE, END_DATE

import requests
from bs4 import BeautifulSoup
import pandas
import os
import unicodedata
import datetime
import json

app = typer.Typer()

# -------------------------------------------------------------------------------------------------
# Download data
# -------------------------------------------------------------------------------------------------

# MARK: Fetch URL
# (Website or File)
def get_url(url:str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
    }
    r = requests.get(url, headers=headers)
    return r

# MARK: File paths
def get_paths(site_html:str):
    logger.info('Parsing HTML for file links...')
    soup = BeautifulSoup(site_html, 'html.parser')
    links = soup.find_all('a')

    parent_dir = '/site/assets/files'
    file_paths = {'Months':[], 'Nom':[]}
    for link in links:
        link_href = str( link.get('href') ) #Can work without the str(). Just added it to remove warnings in VS Code.
        if not link_href.startswith(parent_dir): continue
        if 'datos_abiertos' in link_href:
            file_paths['Months'].append(link_href)
        elif 'nomenclatura' in link_href:
            file_paths['Nom'].append(link_href)
        else:
            continue

    return file_paths

# MARK: Filter by date
def filter_paths(paths:list, start_date:str, end_date:str):
    logger.info('Filtering by date range...')
    logger.debug(f'Start date: {start_date}')
    logger.debug(f'End date: {end_date}')

    # Path & date dataframe
    df_paths = pandas.DataFrame(paths, columns=['path'])
    df_dates = ( df_paths.path
        .str.extract(r'(\d{4}_\d{2})')
        .rename(columns={0:'date'})
    )
    df_paths['date'] = pandas.to_datetime(df_dates['date'], format='%Y_%m')

    # Filter by date
    df_filtered = df_paths.query(
        '(date >= @start_date) and (date <= @end_date)'
    )
    paths_list = df_filtered.path.to_list()

    return paths_list

# MARK: Download files
def download_files(base_url:str, path_list:list, input_path:Path):
    existing_filenames = os.listdir(input_path)

    for filepath in tqdm(path_list, desc='Downloading files', unit=' files'):
        filename = filepath.split('/')[-1]
        input_filepath = f'{input_path}/{filename}'

        # If the file has already been downloaded, no need to do so again
        if filename in existing_filenames: continue

        file_url = base_url + filepath
        r = get_url(file_url)

        with open(input_filepath, 'wb') as f:
            f.write(r.content)
    return

# -------------------------------------------------------------------------------------------------
# Process data
# -------------------------------------------------------------------------------------------------

# MARK: Read files
def read_raw_data(input_path:Path, file_type:str='month'):
    filenames = os.listdir(input_path)
    if file_type=='month':
        filenames = [f for f in filenames if 'datos_abiertos' in f]
        encoding = 'latin-1'
    elif file_type=='nom':
        filenames = [f for f in filenames if 'nomenclatura' in f][:1]
        encoding = 'utf-8'
    else:
        print('File type must be "month" or "nom".')
        return pandas.DataFrame()

    df_list = []
    for filename in filenames:
        filepath = f'{input_path}/{filename}'
        df = pandas.read_csv(filepath, encoding=encoding)
        df = df.rename(columns={'AÃ±o_de_nacimiento':'Año_de_nacimiento'})
        df_list.append(df)

    if file_type=='nom': return df
    return pandas.concat(df_list, ignore_index=True)

# MARK: Normalize column names
def normalize_text(col:str):
    col_lower = col.lower()
    col_norm = unicodedata.normalize('NFKD', col_lower)
    col_ascii = col_norm.encode('ascii', 'ignore').decode('ascii')
    col_ascii = 'ano_de_nacimiento' if col_ascii=='aao_de_nacimiento' else col_ascii
    return col_ascii

def norm_cols(df:pandas.DataFrame):
    cols = list(df.columns)
    cols_norm = [normalize_text(col) for col in cols]
    logger.debug(f'Normalized column names: {cols_norm}')

    rename_dict = dict(zip(cols, cols_norm))
    logger.debug(f'Rename dict: {rename_dict}')
    df_renamed = df.rename(columns=rename_dict)
    return df_renamed

def set_col_types(df:pandas.DataFrame, filetype:str):
    logger.info(f'Setting column types for {filetype} data...')
    datatypes_dicts = {
        'months': {
            'viaje_id':'int64',
            'usuario_id':'int64',
            'genero':'str',
            'ano_de_nacimiento':'float16',
            'inicio_del_viaje':'str',
            'fin_del_viaje':'str',
            'origen_id':'int64',
            'destino_id':'int64'
        },
        'nom': {
            'id':'int64',
            'name':'str',
            'latitude':'float64',
            'longitude':'float64',
            'dpcapacity':'int64',
        }
    }

    if not filetype in ['months', 'nom']:
        logger.error(f'The only valid filetype values are "months" and "nom". But the value received was: {filetype}')

    df_new = df.copy()
    datatypes_dict = datatypes_dicts[filetype]
    for k in datatypes_dict:
        datatype = datatypes_dict[k]
        df_new[k] = df_new[k].astype( datatype )

    return df_new

# -------------------------------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------------------------------

# MARK: Main fn
@app.command()
def main(
    data_path: Path = DATA_DIR,
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE
):
    logger.info("Fetching dataset...")
    logger.info(f'Input path: {input_path}')

    base_url = 'https://mibici.net'

    # Fetch website
    logger.info('Fetching website...')
    source_url = f'{base_url}/es/datos-abiertos/'
    r = get_url(source_url)

    # Parse html
    file_paths = get_paths(r.text)

    # Filter by date
    paths_list = filter_paths(file_paths['Months'], start_date, end_date)
    paths_list.append( file_paths['Nom'][0] )

    # Download files
    download_files(base_url, paths_list, input_path)

    logger.info("Processing dataset...")

    # Read files
    df_months = read_raw_data(input_path=input_path, file_type='month')
    df_nom = read_raw_data(input_path=input_path, file_type='nom')

    # Standarize column names
    df_months = norm_cols(df_months)
    df_nom = norm_cols(df_nom)

    # Set column types
    df_months = set_col_types(df_months, 'months')
    df_nom = set_col_types(df_nom, 'nom')

    # Write to parquet
    logger.info("Saving processed dataset...")
    filepath_months = f'{output_path}/meses.parquet'
    df_months.to_parquet(filepath_months)

    filepath_nom = f'{output_path}/nomenclatura.parquet'
    df_nom.to_parquet(filepath_nom)

    logger.success("Processing dataset complete.")

    logger.info("Recording data version...")
    manifest_dict = {
        'timestamp': datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S'),
        'date_range': {'start_date':start_date, 'end_date':end_date},
        'record_count_months': df_months.shape[0],
        'record_count_nom': df_nom.shape[0]
    }
    manifest_path = f'{output_path}/data_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest_dict, f)


if __name__ == "__main__":
    app()
