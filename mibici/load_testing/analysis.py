"""Uso de recursos con ``docker stats`` y resumen de la prueba de carga (Entrega 06).

Junta las dos fuentes de la prueba:

- **Locust** escribe ``locust_stats_history.csv`` (una fila por segundo con usuarios, peticiones
  por segundo, errores y percentiles de latencia) y ``locust_stats.csv`` (totales por endpoint).
- **docker stats** se lee cada segundo mientras corre Locust y se guarda en
  ``docker_stats.csv`` (CPU y memoria de cada contenedor).

Con eso arma un resumen por escalón de carga (los números que usa el análisis FinOps) y las dos
figuras del reporte.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess  # nosec B404
import threading
import time
from typing import Self

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from mibici.config import (
    LOAD_TEST_CONTAINERS,
    LOAD_TEST_STAGES,
    LOAD_TEST_STATS_INTERVAL_SECONDS,
    LOAD_TEST_WARMUP_SECONDS,
)

LOCUST_PREFIX = "locust"  # Locust agrega _stats.csv, _stats_history.csv, etc.
DOCKER_STATS_FILE = "docker_stats.csv"
SUMMARY_FILE = "resumen_por_escalon.csv"
ENDPOINTS_FILE = "resumen_por_endpoint.csv"
MARKDOWN_FILE = "resumen.md"
FIGURES = ("locust_01_desempeno.png", "locust_02_recursos.png")
DOCKER_STATS_COLUMNS = ["timestamp", "container", "cpu_pct", "mem_mib", "mem_limit_mib"]

# Unidades de memoria de `docker stats` convertidas a MiB
_MEMORY_UNITS = {
    "B": 1 / 1024**2,
    "KIB": 1 / 1024,
    "MIB": 1.0,
    "GIB": 1024.0,
    "TIB": 1024.0**2,
    "KB": 1e3 / 1024**2,
    "MB": 1e6 / 1024**2,
    "GB": 1e9 / 1024**2,
}


# ---------------------------------------------------------------------------------------------
# docker stats
# ---------------------------------------------------------------------------------------------
def parse_memory(text: str) -> float:
    """Convierte una cantidad de memoria de ``docker stats`` (``"350.2MiB"``) a MiB.

    Args:
        text: Cantidad con unidad, como la escribe Docker.

    Returns:
        Cantidad en MiB.

    Raises:
        ValueError: Si el texto no tiene el formato ``<número><unidad>``.
    """
    match = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]+)\s*", text)
    if match is None or match.group(2).upper() not in _MEMORY_UNITS:
        raise ValueError(f"Cantidad de memoria no reconocida: '{text}'")
    return float(match.group(1)) * _MEMORY_UNITS[match.group(2).upper()]


def parse_docker_stats_line(line: str, timestamp: float) -> dict[str, float | str]:
    """Convierte una línea de ``docker stats --format "{{json .}}"`` en una fila de la tabla.

    Args:
        line: Línea JSON con los campos ``Name``, ``CPUPerc`` y ``MemUsage``.
        timestamp: Momento de la lectura (segundos desde 1970, como los de Locust).

    Returns:
        Fila con ``timestamp``, ``container``, ``cpu_pct``, ``mem_mib`` y ``mem_limit_mib``.
    """
    stats = json.loads(line)
    used, limit = stats["MemUsage"].split("/")
    return {
        "timestamp": timestamp,
        "container": stats["Name"],
        "cpu_pct": float(stats["CPUPerc"].strip().rstrip("%")),
        "mem_mib": parse_memory(used),
        "mem_limit_mib": parse_memory(limit),
    }


class DockerStatsRecorder:
    """Lee ``docker stats`` en segundo plano mientras corre la prueba de carga.

    Se usa con ``with``: empieza a leer al entrar y deja ``docker_stats.csv`` al salir. Si Docker
    no está disponible o los contenedores no existen, avisa una vez y la prueba sigue sin datos de
    recursos.
    """

    def __init__(
        self,
        output_path: Path,
        containers: tuple[str, ...] = LOAD_TEST_CONTAINERS,
        interval: float = LOAD_TEST_STATS_INTERVAL_SECONDS,
    ) -> None:
        """Prepara el lector.

        Args:
            output_path: CSV donde se guardan las lecturas.
            containers: Nombres de los contenedores que se leen.
            interval: Pausa entre lecturas, en segundos.
        """
        self.output_path = output_path
        self.containers = containers
        self.interval = interval
        self.rows: list[dict[str, float | str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def read_once(self) -> list[dict[str, float | str]]:
        """Una lectura de ``docker stats`` de todos los contenedores.

        Returns:
            Una fila por contenedor.

        Raises:
            RuntimeError: Si Docker no responde o falta algún contenedor.
        """
        timestamp = time.time()
        try:
            result = subprocess.run(  # nosec B603 B607
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", *self.containers],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"No se pudo ejecutar docker stats: {error}") from error
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "docker stats terminó con error")
        return [
            parse_docker_stats_line(line, timestamp)
            for line in result.stdout.splitlines()
            if line.strip()
        ]

    def _loop(self) -> None:
        """Lee hasta que se pida detenerse; ante un error avisa y deja de leer."""
        while not self._stop.is_set():
            try:
                self.rows.extend(self.read_once())
            except RuntimeError as error:
                logger.warning(f"Sin datos de recursos (docker stats): {error}")
                return
            self._stop.wait(self.interval)

    def __enter__(self) -> Self:
        """Empieza a leer en segundo plano."""
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Detiene la lectura y guarda el CSV."""
        self._stop.set()
        self._thread.join(timeout=60)
        pd.DataFrame(self.rows, columns=DOCKER_STATS_COLUMNS).to_csv(self.output_path, index=False)
        logger.info(f"{len(self.rows)} lecturas de docker stats en {self.output_path}")


# ---------------------------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------------------------
@dataclass
class LoadTestResults:
    """Resultados leídos de la carpeta de la prueba."""

    history: pd.DataFrame  # una fila por segundo (Locust, fila "Aggregated")
    endpoints: pd.DataFrame  # totales por endpoint (Locust)
    resources: pd.DataFrame  # lecturas de docker stats (puede estar vacía)
    start: float  # primer timestamp de Locust: el segundo 0 de la prueba


def load_results(output_dir: Path) -> LoadTestResults:
    """Lee los CSV de Locust y de ``docker stats`` de una prueba.

    Args:
        output_dir: Carpeta de la prueba (``reports/locust``).

    Returns:
        Las tres tablas y el inicio de la prueba.
    """
    history = pd.read_csv(output_dir / f"{LOCUST_PREFIX}_stats_history.csv")
    history = history.loc[history["Name"] == "Aggregated"].copy()
    numeric = [column for column in history.columns if column not in ("Type", "Name")]
    history[numeric] = history[numeric].apply(pd.to_numeric, errors="coerce")
    start = float(history["Timestamp"].min())
    history["elapsed"] = history["Timestamp"] - start

    endpoints = pd.read_csv(output_dir / f"{LOCUST_PREFIX}_stats.csv")
    stats_path = output_dir / DOCKER_STATS_FILE
    resources = (
        pd.read_csv(stats_path)
        if stats_path.exists() and stats_path.stat().st_size > 0
        else pd.DataFrame(columns=DOCKER_STATS_COLUMNS)
    )
    resources["elapsed"] = resources["timestamp"] - start
    return LoadTestResults(history, endpoints, resources, start)


def stage_windows(
    stages: tuple[tuple[int, int], ...] = LOAD_TEST_STAGES,
    warmup: float = LOAD_TEST_WARMUP_SECONDS,
) -> list[tuple[int, float, float]]:
    """Ventana de cada escalón que entra al resumen, sin sus primeros segundos de arranque.

    Args:
        stages: Escalones ``(segundo en que termina, usuarios)``.
        warmup: Segundos al inicio de cada escalón que se descartan mientras suben los usuarios.

    Returns:
        ``(usuarios, segundo inicial, segundo final)`` de cada escalón; el segundo final ya es
        del escalón siguiente y no entra.
    """
    windows, previous_end = [], 0
    for end, users in stages:
        windows.append((users, previous_end + warmup, float(end)))
        previous_end = end
    return windows


def summarize_stages(
    results: LoadTestResults, stages: tuple[tuple[int, int], ...] = LOAD_TEST_STAGES
) -> pd.DataFrame:
    """Desempeño y recursos promedio de cada escalón de carga.

    Las peticiones y los errores salen de la diferencia de los totales acumulados de Locust al
    principio y al final de la ventana; la latencia es la mediana de los percentiles que Locust
    reporta cada segundo; la CPU es el promedio y la memoria el máximo de cada contenedor.

    Args:
        results: Resultados de ``load_results``.
        stages: Escalones de la prueba.

    Returns:
        Una fila por escalón.
    """
    rows = []
    for users, start, end in stage_windows(stages):
        window = results.history.loc[results.history["elapsed"].between(start, end, "left")]
        if len(window) < 2:
            continue
        seconds = window["elapsed"].iloc[-1] - window["elapsed"].iloc[0]
        requests = window["Total Request Count"].iloc[-1] - window["Total Request Count"].iloc[0]
        failures = window["Total Failure Count"].iloc[-1] - window["Total Failure Count"].iloc[0]
        row = {
            "usuarios": users,
            "peticiones_por_segundo": requests / seconds if seconds else float("nan"),
            "errores_pct": 100 * failures / requests if requests else 0.0,
            "latencia_p50_ms": window["50%"].median(),
            "latencia_p95_ms": window["95%"].median(),
            "latencia_p99_ms": window["99%"].median(),
        }
        resources = results.resources.loc[results.resources["elapsed"].between(start, end, "left")]
        for container in LOAD_TEST_CONTAINERS:
            readings = resources.loc[resources["container"] == container]
            row[f"cpu_pct_{container}"] = readings["cpu_pct"].mean()
            row[f"mem_mib_{container}"] = readings["mem_mib"].max()
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def summarize_endpoints(results: LoadTestResults) -> pd.DataFrame:
    """Totales de toda la prueba por endpoint (y el agregado), con nombres en español.

    Args:
        results: Resultados de ``load_results``.

    Returns:
        Una fila por endpoint.
    """
    columns = {
        "Name": "endpoint",
        "Request Count": "peticiones",
        "Failure Count": "errores",
        "50%": "latencia_p50_ms",
        "95%": "latencia_p95_ms",
        "99%": "latencia_p99_ms",
        "Requests/s": "peticiones_por_segundo",
    }
    table = results.endpoints[list(columns)].rename(columns=columns)
    table["endpoint"] = table["endpoint"].replace({"Aggregated": "Total"})
    return table.round(2)


def markdown_table(table: pd.DataFrame) -> str:
    """Tabla de pandas en formato Markdown (sin depender de ``tabulate``).

    Args:
        table: Tabla con encabezados.

    Returns:
        Tabla en Markdown; los vacíos (``NaN``) se escriben como ``–``.
    """
    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    separator = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = [
        "| "
        + " | ".join(
            "–" if pd.isna(value) else f"{value:g}" if isinstance(value, float) else str(value)
            for value in row
        )
        + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def summary_markdown(stages: pd.DataFrame, endpoints: pd.DataFrame) -> str:
    """Las dos tablas del resumen en Markdown, listas para el reporte.

    Args:
        stages: Resultado de ``summarize_stages``.
        endpoints: Resultado de ``summarize_endpoints``.

    Returns:
        Texto en Markdown.
    """
    api, mlflow = LOAD_TEST_CONTAINERS
    stage_table = stages.rename(
        columns={
            "usuarios": "Usuarios",
            "peticiones_por_segundo": "Peticiones/s",
            "errores_pct": "Errores (%)",
            "latencia_p50_ms": "p50 (ms)",
            "latencia_p95_ms": "p95 (ms)",
            "latencia_p99_ms": "p99 (ms)",
            f"cpu_pct_{api}": "CPU API (%)",
            f"mem_mib_{api}": "Memoria API (MiB)",
            f"cpu_pct_{mlflow}": "CPU MLflow (%)",
            f"mem_mib_{mlflow}": "Memoria MLflow (MiB)",
        }
    )
    return (
        "Resumen por escalón de carga (sin los primeros segundos de cada escalón)\n\n"
        f"{markdown_table(stage_table)}\n\n"
        "Totales por endpoint\n\n"
        f"{markdown_table(endpoints)}\n"
    )


# ---------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------
def plot_performance(results: LoadTestResults, path: Path) -> None:
    """Figura de desempeño: usuarios, peticiones por segundo y errores, y latencia.

    Args:
        results: Resultados de ``load_results``.
        path: Archivo PNG de salida.
    """
    history = results.history
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].step(history["elapsed"], history["User Count"], where="post", color="#4C72B0")
    axes[0].set_ylabel("Usuarios")
    axes[0].set_title("Carga y desempeño de la API")

    axes[1].plot(history["elapsed"], history["Requests/s"], label="Peticiones/s", color="#55A868")
    axes[1].plot(history["elapsed"], history["Failures/s"], label="Errores/s", color="#C44E52")
    axes[1].set_ylabel("Por segundo")
    axes[1].legend(loc="upper left")

    for column, color in (("50%", "#4C72B0"), ("95%", "#DD8452"), ("99%", "#C44E52")):
        axes[2].plot(history["elapsed"], history[column], label=f"p{column[:-1]}", color=color)
    axes[2].set_ylabel("Latencia (ms)")
    axes[2].set_xlabel("Segundos desde el inicio de la prueba")
    axes[2].legend(loc="upper left")

    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_resources(results: LoadTestResults, path: Path) -> None:
    """Figura de recursos: CPU y memoria de cada contenedor durante la prueba.

    Args:
        results: Resultados de ``load_results``.
        path: Archivo PNG de salida.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for container, readings in results.resources.groupby("container"):
        axes[0].plot(readings["elapsed"], readings["cpu_pct"], label=container)
        axes[1].plot(readings["elapsed"], readings["mem_mib"], label=container)
    axes[0].set_ylabel("CPU (%)")
    axes[0].set_title("Uso de recursos de los contenedores (docker stats)")
    axes[1].set_ylabel("Memoria (MiB)")
    axes[1].set_xlabel("Segundos desde el inicio de la prueba")
    for axis in axes:
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(output_dir: Path) -> Path:
    """Escribe el resumen (CSV y Markdown) y las figuras de una prueba ya corrida.

    Args:
        output_dir: Carpeta de la prueba (``reports/locust``).

    Returns:
        Ruta del resumen en Markdown.
    """
    results = load_results(output_dir)
    stages = summarize_stages(results)
    endpoints = summarize_endpoints(results)
    stages.to_csv(output_dir / SUMMARY_FILE, index=False)
    endpoints.to_csv(output_dir / ENDPOINTS_FILE, index=False)
    markdown_path = output_dir / MARKDOWN_FILE
    markdown_path.write_text(summary_markdown(stages, endpoints), encoding="utf-8")

    plot_performance(results, output_dir / FIGURES[0])
    if results.resources.empty:
        logger.warning("No hay lecturas de docker stats: se omite la figura de recursos")
    else:
        plot_resources(results, output_dir / FIGURES[1])
    return markdown_path
