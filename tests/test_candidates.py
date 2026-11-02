"""Pruebas de ``modeling/candidates.py``: catálogo de modelos y resolución de hiperparámetros."""

import pytest

from mibici.config import MODEL_DEFAULTS
from mibici.modeling.candidates import CATALOG, ModelSpec, get_spec, resolve_hyperparams

ALL_MODELS = [
    pytest.param(question, model, id=f"{question}-{model}")
    for question, models in CATALOG.items()
    for model in models
]


def test_get_spec_returns_the_catalog_entry() -> None:
    """Caso normal: devuelve la especificación con su familia y los valores por defecto de ``config.py``."""
    spec = get_spec("p1", "hgb_poisson")

    assert isinstance(spec, ModelSpec)
    assert spec.family == "boosting"
    assert spec.defaults == MODEL_DEFAULTS["hgb"]


@pytest.mark.parametrize(("question", "model"), ALL_MODELS)
def test_every_catalog_model_builds_with_its_defaults(question: str, model: str) -> None:
    """Caso normal: cada modelo del catálogo se construye con sus valores por defecto y sabe entrenar."""
    spec = get_spec(question, model)

    estimator = spec.build(resolve_hyperparams(spec, {}))

    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "predict") or hasattr(estimator, "fit_predict")


@pytest.mark.parametrize(
    ("question", "model", "message"),
    [
        pytest.param("p4", "ridge", "Pregunta 'p4' desconocida", id="unknown_question"),
        pytest.param("p1", "svm", "Modelo 'svm' desconocido para p1", id="unknown_model"),
        pytest.param(
            "p1", "kmeans", "Modelo 'kmeans' desconocido para p1", id="model_of_another_question"
        ),
    ],
)
def test_get_spec_rejects_unknown_question_or_model(
    question: str, model: str, message: str
) -> None:
    """Caso límite: un nombre desconocido lanza ``ValueError`` y lista las opciones válidas."""
    with pytest.raises(ValueError, match=message) as error:
        get_spec(question, model)

    assert "Opciones" in str(error.value)


def test_resolve_hyperparams_casts_cli_text_to_the_default_type() -> None:
    """Caso normal: el texto de ``mlflow run -P`` toma el tipo del valor por defecto; lo demás se conserva."""
    spec = get_spec("p1", "hgb")

    hyperparams = resolve_hyperparams(spec, {"max_iter": "30", "learning_rate": "0.05"})

    assert hyperparams == {**MODEL_DEFAULTS["hgb"], "max_iter": 30, "learning_rate": 0.05}
    assert isinstance(hyperparams["max_iter"], int)
    assert isinstance(hyperparams["learning_rate"], float)
    assert spec.defaults == MODEL_DEFAULTS["hgb"]  # los valores por defecto no se modifican


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="no_overrides"),
        pytest.param({"alpha": ""}, id="empty_string"),
        pytest.param({"alpha": '""'}, id="windows_empty_quotes"),
        pytest.param({"alpha": "   "}, id="blank"),
        pytest.param({"alpha": None}, id="none"),
    ],
)
def test_resolve_hyperparams_empty_value_means_the_default(overrides: dict) -> None:
    """Caso límite: vacío (incluidas las dos comillas de ``mlflow run`` en Windows) es el valor de ``config.py``."""
    spec = get_spec("p1", "ridge")

    assert resolve_hyperparams(spec, overrides) == MODEL_DEFAULTS["ridge"]


def test_resolve_hyperparams_strips_quotes_and_spaces() -> None:
    """Caso límite: comillas y espacios alrededor del valor se descartan: ``" '10' "`` vale 10.0."""
    spec = get_spec("p1", "ridge")

    assert resolve_hyperparams(spec, {"alpha": " '10' "}) == {"alpha": 10.0}


def test_resolve_hyperparams_ignores_hyperparameters_the_model_does_not_use() -> None:
    """Caso límite: un hiperparámetro ajeno al modelo se avisa y se ignora; no detiene la corrida."""
    spec = get_spec("p1", "ridge")

    assert resolve_hyperparams(spec, {"max_iter": "50"}) == MODEL_DEFAULTS["ridge"]


def test_resolve_hyperparams_truncates_decimals_for_integer_hyperparameters() -> None:
    """Caso límite: un decimal para un entero se trunca (``"3.9"`` pasa a 3): manda el tipo por defecto."""
    spec = get_spec("p3", "kmeans")

    assert resolve_hyperparams(spec, {"n_clusters": "3.9"}) == {"n_clusters": 3}
