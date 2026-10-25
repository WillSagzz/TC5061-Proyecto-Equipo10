"""Gráficas del proyecto; se guardan en reports/figures/ y se registran en MLflow."""

from pathlib import Path

import matplotlib

# Permite guardar figuras sin abrir ventanas (por ejemplo, al correr desde la terminal)
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

INK = "#262624"
INK_SECONDARY = "#5e5d59"
GRID = "#e6e5e0"
BACKGROUND = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CONTEXT_GRAY = "#d3d1c7"  # estaciones de referencia en los mapas
# Escala secuencial de un solo tono (azul), de claro a oscuro
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


def _apply_style(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(BACKGROUND)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)
    return path


def plot_p1_actual_vs_predicted(
    table: pd.DataFrame, predictions: np.ndarray, path: Path, days: int = 7
) -> Path:
    """Grafica los viajes por hora de todo el sistema: reales contra predichos.

    Suma todas las estaciones para que se vea el patrón diario y muestra solo los primeros
    días del conjunto para que las horas se distingan.

    Args:
        table: Tabla de P1 con ``period_start`` y ``departures``.
        predictions: Predicciones del modelo, en el mismo orden que ``table``.
        path: Ruta del PNG.
        days: Número de días que se muestran.

    Returns:
        La ruta del PNG guardado.
    """
    data = table.assign(predicted=predictions)
    limit = data["period_start"].min() + pd.Timedelta(days=days)
    totals = (
        data.loc[data["period_start"] < limit]
        .groupby("period_start")[["departures", "predicted"]]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(totals.index, totals["departures"], color=INK, linewidth=1.6, label="Real")
    ax.plot(totals.index, totals["predicted"], color=BLUE, linewidth=2, label="Predicción")
    ax.set_title(
        f"P1 · Viajes por hora en el sistema, primeros {days} días",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.set_ylabel("Viajes por hora", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _apply_style(ax)
    return _save(fig, path)


def plot_p2_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, labels: list[str], path: Path
) -> Path:
    """Grafica la matriz de confusión de P2, normalizada por fila.

    Cada fila es la clase real y suma 100 %: la diagonal es el recall de cada clase. Cada
    celda muestra el porcentaje y, debajo, el número de intervalos.

    Args:
        y_true: Clases reales.
        y_pred: Clases predichas.
        labels: Orden de las clases en filas y columnas.
        path: Ruta del PNG.

    Returns:
        La ruta del PNG guardado.
    """
    counts = confusion_matrix(y_true, y_pred, labels=labels)
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
    for row in range(len(labels)):
        for col in range(len(labels)):
            color = "white" if shares[row, col] >= 0.5 else INK
            ax.text(
                col,
                row,
                f"{shares[row, col]:.0%}\n{counts[row, col]:,}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Clase predicha", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Clase real", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        "P2 · Matriz de confusión (% por clase real)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    # Separación de 2 px entre celdas, del color del fondo
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, path)


def plot_p3_cluster_map(table: pd.DataFrame, labels: np.ndarray, path: Path) -> Path:
    """Muestra en el mapa dónde están las estaciones de cada grupo de P3.

    Un panel por grupo: las estaciones del grupo en azul y el resto en gris como
    referencia. Así cada grupo se distingue por su panel y no por el color, sin importar
    cuántos grupos haya.

    Args:
        table: Tabla de P3 con ``latitude`` y ``longitude``.
        labels: Grupo asignado a cada estación, en el mismo orden que ``table``.
        path: Ruta del PNG.

    Returns:
        La ruta del PNG guardado.
    """
    groups = np.sort(np.unique(labels))
    n_cols = min(4, len(groups))
    n_rows = int(np.ceil(len(groups) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.6 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, group in zip(axes.flat, groups):
        in_group = labels == group
        ax.scatter(table["longitude"], table["latitude"], s=8, color=CONTEXT_GRAY, linewidths=0)
        ax.scatter(
            table.loc[in_group, "longitude"],
            table.loc[in_group, "latitude"],
            s=16,
            color=BLUE,
            edgecolors=BACKGROUND,
            linewidths=0.8,
        )
        ax.set_title(
            f"Grupo {group} · {in_group.sum()} estaciones",
            loc="left",
            fontsize=10,
            color=INK,
        )
        ax.set_aspect("equal")
        _apply_style(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(groups) :]:
        ax.set_visible(False)
    fig.suptitle(
        "P3 · Ubicación de las estaciones de cada grupo",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    return _save(fig, path)


def plot_p2_precision_recall(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: list[str],
    risk_classes: list[str],
    path: Path,
) -> Path:
    """Grafica la curva precisión-recall de cada clase de riesgo de P2.

    Muestra, para cada umbral de probabilidad, qué parte de los intervalos en riesgo detecta
    el modelo (recall) y qué parte de sus alertas son correctas (precisión). La línea
    punteada de cada color es la proporción de esa clase: un modelo sin información queda
    a esa altura.

    Args:
        y_true: Clases reales.
        probabilities: Probabilidad de cada clase, en el orden de ``classes``.
        classes: Orden de las columnas de ``probabilities``.
        risk_classes: Clases de riesgo que se grafican.
        path: Ruta del PNG.

    Returns:
        La ruta del PNG guardado.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for risk_class, color in zip(risk_classes, [BLUE, ORANGE]):
        is_class = np.asarray(y_true) == risk_class
        position = list(classes).index(risk_class)
        precision, recall, _ = precision_recall_curve(is_class, probabilities[:, position])
        # El último punto (recall 0, precisión 1) lo agrega scikit-learn por convención y no
        # corresponde a ningún umbral: se omite para no dibujar una raya vertical en el eje
        ax.plot(recall[:-1], precision[:-1], color=color, linewidth=2, label=risk_class)
        ax.axhline(is_class.mean(), color=color, linewidth=1, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (intervalos en riesgo detectados)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Precisión (alertas correctas)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(
        "P2 · Curva precisión-recall por clase de riesgo",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _apply_style(ax)
    return _save(fig, path)
