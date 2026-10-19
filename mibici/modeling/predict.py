"""Inferencia con los modelos entrenados por ``train.py``.

Uso desde la raíz del repositorio, después de ``make data`` y ``make train``:

    python -m mibici.modeling.predict p1
    python -m mibici.modeling.predict p2
    python -m mibici.modeling.predict p3

Cada comando carga el modelo serializado de ``models/``, reconstruye la tabla de la pregunta
con ``features.py`` (misma partición temporal que el entrenamiento) y escribe las predicciones
en ``data/processed/``, fuera de Git.
"""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from mibici.config import MODELS_DIR, PROCESSED_DATA_DIR
from mibici.features import (
    P3_FEATURES,
    build_p1_table,
    build_p2_table,
    build_p3_table,
    load_data,
    split_by_date,
)

app = typer.Typer(help="Inferencia de MiBici con los modelos entrenados por train.py.")


@app.callback()
def main() -> None:
    """Predice con un modelo entrenado. Elige la pregunta: p1, p2 o p3."""


def _load_model(name: str):
    """Carga ``models/<name>.joblib``, que escribe ``train.py``."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero `make train`.")
    return joblib.load(path)


def _save_predictions(table: pd.DataFrame, name: str) -> Path:
    """Escribe las predicciones en ``data/processed/<name>.csv`` y devuelve la ruta."""
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.success(f"{len(table):,} predicciones guardadas en {path}")
    return path


@app.command()
def p1(model_name: str = "p1_baseline") -> None:
    """P1 · Regresión: salidas previstas por estación y hora en el conjunto de prueba."""
    trips, stations = load_data()
    test = split_by_date(build_p1_table(trips, stations), "period_start")["test"]
    model = _load_model(model_name)
    predictions = test[["station_id", "period_start", "departures"]].assign(
        departures_pred=model.predict(test)
    )
    _save_predictions(predictions, f"{model_name}_predicciones")


@app.command()
def p2(model_name: str = "p2_baseline") -> None:
    """P2 · Clasificación: clase de riesgo prevista por estación e intervalo en prueba."""
    trips, stations = load_data()
    test = split_by_date(build_p2_table(trips, stations), "period_start")["test"]
    model = _load_model(model_name)
    probabilities = pd.DataFrame(
        model.predict_proba(test), index=test.index, columns=[f"prob_{c}" for c in model.classes_]
    )
    predictions = test[["station_id", "period_start", "risk"]].assign(
        risk_pred=model.predict(test)
    )
    _save_predictions(predictions.join(probabilities), f"{model_name}_predicciones")


@app.command()
def p3(model_name: str = "p3_baseline") -> None:
    """P3 · Clustering: grupo asignado a cada estación según su perfil de uso."""
    trips, stations = load_data()
    table = build_p3_table(trips, stations)
    model = _load_model(model_name)
    predictions = table[["station_id", "latitude", "longitude"]].assign(
        group=model.predict(table[P3_FEATURES])
    )
    _save_predictions(predictions, f"{model_name}_grupos")


if __name__ == "__main__":
    app()
