"""Pruebas de la versión alternativa y el A/B testing de P1 (``ab_test.py``).

Usan los datos sintéticos de ``monitoring.py`` y un MLflow local en una carpeta temporal:
nunca tocan el servidor del equipo.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici import config
from mibici.config import REGISTERED_MODELS, VALIDATION_START
from mibici.experiments import project_run_key, resolve_experiment
from mibici.features import P1_FEATURES, P1_TARGET
from mibici.modeling import ab_test
from mibici.modeling.candidates import get_spec, resolve_hyperparams
from mibici.modeling.registry import LoadedModel, load_registered_model
from mibici.monitoring import MonitoringDatasets, build_synthetic_monitoring_data

MODEL_NAME = REGISTERED_MODELS["p1"]


def _local_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Apunta MLflow a una base SQLite temporal en lugar del servidor del equipo.

    ``resolve_experiment`` crea el experimento de P1 en esa base y lo deja activo; sin eso,
    MLflow recordaría el experimento de la prueba anterior, que vivía en otra base.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "MLFLOW_ARTIFACTS_DIR", tmp_path / "mlartifacts")
    monkeypatch.setattr(ab_test, "AB_TEST_REPORTS_DIR", tmp_path / "ab_test")
    mlflow.set_tracking_uri(uri)
    resolve_experiment("p1")


def _datasets() -> MonitoringDatasets:
    """Tablas de P1 para reentrenar (hasta agosto de 2026) y evaluar (septiembre de 2026)."""
    return ab_test.build_ab_datasets(*build_synthetic_monitoring_data())


# Parámetros como los que deja `make train` en el run de @champion
CHAMPION_RUN_PARAMS = {
    "question": "p1",
    "model": "hgb_poisson",
    "estimador": "HistGradientBoostingRegressor",
    "hp_max_iter": "20",
    "inicio_entrenamiento": config.TRAIN_START,
    "inicio_validacion": VALIDATION_START,
    "inicio_prueba": config.TEST_START,
    "fin_datos": config.DATA_END,
}


def _register_champion(datasets: MonitoringDatasets) -> LoadedModel:
    """Entrena una versión A pequeña con el periodo de entrenamiento y la marca @champion.

    Su run lleva parámetros como los de ``make train``, para revisar cuáles se copian a B.
    """
    training = datasets.reference.loc[
        datasets.reference["period_start"] < pd.Timestamp(VALIDATION_START)
    ]
    spec = get_spec("p1", "hgb_poisson")
    model = spec.build(resolve_hyperparams(spec, {"max_iter": "20"}))
    model.fit(training[P1_FEATURES], training[P1_TARGET])
    with mlflow.start_run():
        mlflow.log_params(CHAMPION_RUN_PARAMS)
        info = mlflow.sklearn.log_model(
            model,
            name="modelo",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=MODEL_NAME,
        )
    mlflow.MlflowClient().set_registered_model_alias(
        MODEL_NAME, "champion", info.registered_model_version
    )
    return load_registered_model("p1", alias="champion", allow_cache_fallback=False)


def _register_both(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Registra A (@champion) y B (@challenger) en el MLflow temporal."""
    _local_registry(monkeypatch, tmp_path)
    datasets = _datasets()
    champion = _register_champion(datasets)
    model = ab_test.retrain_challenger(champion, datasets.reference)
    ab_test.register_challenger(model, champion, datasets.reference)
    challenger = load_registered_model("p1", alias="challenger", allow_cache_fallback=False)
    return datasets, champion, challenger


def test_retrain_registers_a_new_version_of_the_same_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso normal: B es la versión 2 del mismo modelo, con @challenger y los mismos ajustes.

    Solo cambian los datos: B se entrena con más meses que A, pero con los mismos
    hiperparámetros y la misma semilla.
    """
    datasets, champion, challenger = _register_both(monkeypatch, tmp_path)

    assert champion.version == "1"
    assert challenger.version == "2"
    assert challenger.model_name == champion.model_name == MODEL_NAME
    assert challenger.model.get_params() == champion.model.get_params()
    assert datasets.reference["period_start"].max() < pd.Timestamp(config.CURRENT_START)


def _challenger_run(challenger: LoadedModel) -> mlflow.entities.Run:
    """Run que registró la versión B."""
    client = mlflow.MlflowClient()
    return client.get_run(client.get_model_version(MODEL_NAME, challenger.version).run_id)


def test_retrain_logs_the_same_model_params_as_the_champion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso normal: el run de B repite los parámetros del modelo de A, con sus propias fechas.

    Así la vista *Compare* de MLflow muestra los mismos parámetros en las dos versiones.
    """
    _, _, challenger = _register_both(monkeypatch, tmp_path)

    run = _challenger_run(challenger)
    params = run.data.params

    for name in ("question", "model", "estimador", "hp_max_iter"):
        assert params[name] == CHAMPION_RUN_PARAMS[name]
    assert params["inicio_entrenamiento"] == config.RETRAIN_START
    assert params["fin_entrenamiento"] == config.RETRAIN_END
    assert params["inicio_prueba"] == config.CURRENT_START
    assert "inicio_validacion" not in params
    assert run.data.tags["mlflow.runName"] == f"P1-reentrenado-hasta-{config.RETRAIN_END}"
    assert run.inputs.dataset_inputs  # pestaña Datasets, como los runs de make train


def test_retrain_continues_the_run_created_by_mlflow_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso normal: con ``mlflow run`` (``MLFLOW_RUN_ID``), B se registra en ese mismo run."""
    _local_registry(monkeypatch, tmp_path)
    datasets = _datasets()
    champion = _register_champion(datasets)
    experiment = mlflow.get_experiment_by_name(config.EXPERIMENTS["p1"])
    project_run = mlflow.MlflowClient().create_run(experiment.experiment_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", project_run.info.run_id)

    model = ab_test.retrain_challenger(champion, datasets.reference)
    version = ab_test.register_challenger(model, champion, datasets.reference)

    registered = mlflow.MlflowClient().get_model_version(MODEL_NAME, version)
    assert registered.run_id == project_run.info.run_id


def test_mlproject_entry_points_use_the_p1_experiment() -> None:
    """Caso límite: ``mlflow run -e retrain`` y ``-e ab_test`` van al experimento de P1."""
    for entry_point in ("retrain", "ab_test"):
        assert project_run_key(["mlflow", "run", str(config.PROJ_ROOT), "-e", entry_point]) == "p1"


def test_ab_test_compares_both_versions_on_the_same_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso normal: A y B se evalúan en las mismas filas de septiembre y se registra el run.

    También revisa que se escriban la tabla en Markdown (la que va en el reporte) y en CSV.
    """
    datasets, champion, challenger = _register_both(monkeypatch, tmp_path)

    result = ab_test.run_ab_test(datasets, champion, challenger)
    md_path = ab_test.log_ab_test(result, champion, challenger)

    rows = result.table.pivot(index="segmento", columns="version", values="filas")
    assert set(result.table["version"]) == {"champion", "challenger"}
    assert (rows["champion"] == rows["challenger"]).all()
    assert rows.loc["todas", "champion"] == len(datasets.current)
    assert result.ci_low <= result.mae_difference <= result.ci_high
    assert result.winner in {"champion", "challenger"}
    assert md_path.name == "p1_ab_test.md"
    assert "| Versión | Estaciones | Filas | MAE | RMSE |" in md_path.read_text(encoding="utf-8")
    assert (md_path.parent / "p1_ab_test.csv").exists()
    run = mlflow.search_runs(search_all_experiments=True, filter_string="tags.etapa = 'ab_test'")
    assert len(run) == 1
    assert "metrics.test_mae_champion" in run.columns


def test_evaluate_versions_separates_known_and_new_stations() -> None:
    """Caso límite: las filas de estaciones nuevas y conocidas suman las de todo el periodo."""
    current = _datasets().current

    class _Constant:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), 1.0)

    table, _ = ab_test.evaluate_versions({"champion": _Constant()}, current, known_stations={1, 2})
    rows = table.set_index("segmento")["filas"]

    assert rows["conocidas"] + rows["nuevas"] == rows["todas"]
    assert rows["nuevas"] == (~current["station_id"].isin({1, 2})).sum()


def _days(n_days: int = 30, rows_per_day: int = 50) -> pd.Series:
    """Un día por fila: ``n_days`` días con ``rows_per_day`` filas cada uno."""
    return pd.Series(np.repeat(pd.date_range("2026-09-01", periods=n_days).date, rows_per_day))


def test_bootstrap_gives_the_win_to_a_clearly_better_version() -> None:
    """Caso normal: si B tiene la mitad del error todos los días, el intervalo queda bajo 0."""
    days = _days()
    rng = np.random.default_rng(0)
    errors_a = rng.uniform(1.5, 2.5, size=len(days))

    difference, _low, high = ab_test.paired_bootstrap(errors_a, errors_a / 2, days)

    assert difference < 0 and high < 0
    assert ab_test.decide_winner(high) == "challenger"


def test_bootstrap_keeps_the_champion_when_versions_tie() -> None:
    """Caso límite: con errores idénticos la diferencia es 0 y se conserva la versión vigente."""
    days = _days()
    errors = np.random.default_rng(0).uniform(1, 3, size=len(days))

    difference, low, high = ab_test.paired_bootstrap(errors, errors.copy(), days)

    assert difference == low == high == 0
    assert ab_test.decide_winner(high) == "champion"


def test_bootstrap_is_reproducible_with_the_same_seed() -> None:
    """Caso límite: con la misma semilla, el intervalo es exactamente el mismo."""
    days = _days()
    rng = np.random.default_rng(1)
    errors_a, errors_b = rng.uniform(1, 3, size=len(days)), rng.uniform(1, 3, size=len(days))

    first = ab_test.paired_bootstrap(errors_a, errors_b, days)
    second = ab_test.paired_bootstrap(errors_a, errors_b, days)

    assert first == second
