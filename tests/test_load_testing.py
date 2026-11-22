"""Pruebas de la prueba de carga con Locust y del resumen con docker stats (Entrega 06).

No importan ``locustfile.py`` (Locust modifica la biblioteca estándar al importarse) ni
necesitan Docker: revisan las peticiones, los escalones y el resumen con CSV sintéticos con el
mismo formato que escriben Locust y ``docker stats``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mibici.config import LOAD_TEST_CONTAINERS, LOAD_TEST_STAGES, NO_SERVICE_HOURS
from mibici.load_testing import analysis
from mibici.load_testing.payloads import p1_payload, p2_payload, p3_payload, users_for
from mibici.schemas import P1FeaturesInput, P2FeaturesInput, P3FeaturesInput

API, MLFLOW = LOAD_TEST_CONTAINERS


# ---------------------------------------------------------------------------------------------
# Peticiones y escalones
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("build", "schema"),
    [(p1_payload, P1FeaturesInput), (p2_payload, P2FeaturesInput), (p3_payload, P3FeaturesInput)],
)
def test_every_payload_is_accepted_by_the_api_schema(build, schema) -> None:
    """Caso normal: 500 peticiones al azar pasan la validación de la API sin excepción.

    Si alguna fallara, la prueba de carga contaría errores de validación (HTTP 422) en lugar de
    medir el desempeño del sistema.
    """
    rng = np.random.default_rng(0)
    for _ in range(500):
        schema(**build(rng))


def test_payloads_vary_and_respect_the_service_hours() -> None:
    """Caso límite: las peticiones cambian entre sí y P1 nunca pide horas sin servicio."""
    rng = np.random.default_rng(1)
    payloads = [p1_payload(rng) for _ in range(300)]

    assert len({p["station_id"] for p in payloads}) > 100
    assert not {p["hour_of_day"] for p in payloads} & set(NO_SERVICE_HOURS)
    assert {p["is_weekend"] for p in payloads} == {0, 1}


def test_payloads_are_reproducible_with_the_same_seed() -> None:
    """Caso límite: la misma semilla produce exactamente las mismas peticiones (y son JSON)."""
    first = [p2_payload(np.random.default_rng(7)) for _ in range(3)]
    second = [p2_payload(np.random.default_rng(7)) for _ in range(3)]

    assert first == second
    json.dumps(first)  # tipos nativos de Python, no de NumPy


def test_users_follow_the_stages_and_the_test_ends() -> None:
    """Caso normal y límite: cada escalón tiene sus usuarios y después del último termina."""
    stages = ((60, 10), (120, 25))

    assert users_for(0, stages) == 10
    assert users_for(59.9, stages) == 10
    assert users_for(60, stages) == 25
    assert users_for(120, stages) is None


# ---------------------------------------------------------------------------------------------
# docker stats
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "mib"),
    [("512MiB", 512.0), ("2GiB", 2048.0), ("1024KiB", 1.0), ("1.5GB", 1.5e9 / 1024**2)],
)
def test_memory_units_are_converted_to_mib(text: str, mib: float) -> None:
    """Caso normal: las unidades que usa Docker se convierten a MiB."""
    assert analysis.parse_memory(text) == pytest.approx(mib)


def test_a_docker_stats_line_becomes_a_row() -> None:
    """Caso normal: una línea JSON de docker stats da CPU en % y memoria en MiB."""
    line = json.dumps({"Name": API, "CPUPerc": "85.31%", "MemUsage": "412.5MiB / 2GiB"})

    row = analysis.parse_docker_stats_line(line, timestamp=100.0)

    assert row == {
        "timestamp": 100.0,
        "container": API,
        "cpu_pct": 85.31,
        "mem_mib": 412.5,
        "mem_limit_mib": 2048.0,
    }


def test_unknown_memory_unit_is_rejected() -> None:
    """Caso límite: una unidad desconocida no se convierte en silencio a un número falso."""
    with pytest.raises(ValueError, match="no reconocida"):
        analysis.parse_memory("12 bananas")


def test_recorder_without_docker_leaves_an_empty_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso límite: sin Docker la prueba no se cae; deja el CSV vacío con sus columnas."""

    def _no_docker(*_args, **_kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(analysis.subprocess, "run", _no_docker)
    path = tmp_path / "docker_stats.csv"

    with analysis.DockerStatsRecorder(path, interval=0.01):
        pass

    assert list(pd.read_csv(path).columns) == analysis.DOCKER_STATS_COLUMNS


# ---------------------------------------------------------------------------------------------
# Resumen y figuras con resultados sintéticos
# ---------------------------------------------------------------------------------------------
START = 1_800_000_000  # timestamp del segundo 0 de la prueba sintética


def _write_results(output_dir: Path, with_resources: bool = True) -> None:
    """Escribe CSV con el formato de Locust y de docker stats para una prueba de 300 s.

    En cada escalón la API atiende 0.8 peticiones por segundo por usuario; en el último (150
    usuarios) falla 1 de cada 10 peticiones y la latencia se dispara.
    """
    rows, total, failed = [], 0.0, 0.0
    for second in range(LOAD_TEST_STAGES[-1][0]):
        users = users_for(second, LOAD_TEST_STAGES)
        rps = 0.8 * users
        fail_rate = 0.1 if users == 150 else 0.0
        total, failed = total + rps, failed + rps * fail_rate
        rows.append(
            {
                "Timestamp": START + second,
                "User Count": users,
                "Type": "",
                "Name": "Aggregated",
                "Requests/s": rps,
                "Failures/s": rps * fail_rate,
                "50%": 20 if users < 150 else 900,
                "95%": 50 if users < 150 else 2500,
                "99%": 80 if users < 150 else 4000,
                "Total Request Count": total,
                "Total Failure Count": failed,
            }
        )
    history = pd.DataFrame(rows)
    # Locust también escribe filas por endpoint en el historial; el resumen debe ignorarlas
    endpoint_rows = history.assign(Name="/predict/p1", Type="POST")
    pd.concat([history, endpoint_rows]).to_csv(
        output_dir / "locust_stats_history.csv", index=False
    )

    pd.DataFrame(
        {
            "Type": ["POST", ""],
            "Name": ["/predict/p1", "Aggregated"],
            "Request Count": [100, 120],
            "Failure Count": [0, 2],
            "50%": [20, 20],
            "95%": [50, 60],
            "99%": [80, 90],
            "Requests/s": [30.5, 36.2],
        }
    ).to_csv(output_dir / "locust_stats.csv", index=False)

    stats = pd.DataFrame(
        [
            {
                "timestamp": START + second,
                "container": container,
                "cpu_pct": (users_for(second, LOAD_TEST_STAGES) if container == API else 1.0),
                "mem_mib": 400.0 + second if container == API else 300.0,
                "mem_limit_mib": 2048.0,
            }
            for second in range(LOAD_TEST_STAGES[-1][0])
            for container in LOAD_TEST_CONTAINERS
        ]
    )
    if not with_resources:
        stats = stats.iloc[0:0]
    stats.to_csv(output_dir / "docker_stats.csv", index=False)


def test_stage_summary_measures_each_load_level(tmp_path: Path) -> None:
    """Caso normal: el resumen recupera las peticiones, errores, latencia y CPU de cada escalón."""
    _write_results(tmp_path)

    summary = analysis.summarize_stages(analysis.load_results(tmp_path)).set_index("usuarios")

    assert list(summary.index) == [users for _, users in LOAD_TEST_STAGES]
    assert summary.loc[50, "peticiones_por_segundo"] == pytest.approx(40.0)
    assert summary.loc[50, "errores_pct"] == 0
    assert summary.loc[150, "errores_pct"] == pytest.approx(10.0)
    assert summary.loc[150, "latencia_p95_ms"] == 2500
    assert summary.loc[100, f"cpu_pct_{API}"] == pytest.approx(100.0)
    assert summary.loc[100, f"cpu_pct_{MLFLOW}"] == pytest.approx(1.0)


def test_write_summary_writes_tables_and_figures(tmp_path: Path) -> None:
    """Caso normal: quedan el resumen en CSV y Markdown y las dos figuras del reporte."""
    _write_results(tmp_path)

    markdown = analysis.write_summary(tmp_path).read_text(encoding="utf-8")

    assert "| Usuarios | Peticiones/s | Errores (%)" in markdown
    assert "| Total |" in markdown
    for name in (*analysis.FIGURES, analysis.SUMMARY_FILE, analysis.ENDPOINTS_FILE):
        assert (tmp_path / name).exists()


def test_write_summary_without_docker_stats_skips_the_resources_figure(tmp_path: Path) -> None:
    """Caso límite: sin lecturas de docker stats queda el resumen de desempeño y la CPU vacía."""
    _write_results(tmp_path, with_resources=False)

    analysis.write_summary(tmp_path)
    summary = pd.read_csv(tmp_path / analysis.SUMMARY_FILE)

    assert (tmp_path / analysis.FIGURES[0]).exists()
    assert not (tmp_path / analysis.FIGURES[1]).exists()
    assert summary[f"cpu_pct_{API}"].isna().all()
