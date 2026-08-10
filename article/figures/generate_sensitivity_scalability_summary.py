"""Regenerate Figure 3 from the values reported in Supplementary Table S3."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class SensitivityRow(NamedTuple):
    clients: int
    malicious_ratio: int
    accuracy_mean: float
    accuracy_sd: float
    f1_mean: float
    f1_sd: float
    mean_weight_mean: float
    mean_weight_sd: float
    final_weight_mean: float
    final_weight_sd: float
    suppression_mean: float
    suppression_sd: float


def sensitivity_rows() -> tuple[SensitivityRow, ...]:
    """Return the eight mean/SD configurations printed in Supplementary Table S3."""
    return (
        SensitivityRow(5, 20, 0.893, 0.098, 0.890, 0.102, 0.0000, 0.0000, 0.0000, 0.0000, 1.000, 0.000),
        SensitivityRow(5, 40, 0.591, 0.109, 0.497, 0.122, 0.2404, 0.0343, 0.2401, 0.0210, 0.044, 0.133),
        SensitivityRow(10, 10, 0.868, 0.116, 0.842, 0.155, 0.0002, 0.0004, 0.0005, 0.0010, 0.867, 0.224),
        SensitivityRow(10, 20, 0.859, 0.100, 0.837, 0.136, 0.0182, 0.0250, 0.0384, 0.0757, 0.378, 0.441),
        SensitivityRow(10, 30, 0.855, 0.094, 0.826, 0.133, 0.0361, 0.0327, 0.0632, 0.0726, 0.178, 0.338),
        SensitivityRow(20, 10, 0.915, 0.058, 0.912, 0.062, 0.0087, 0.0132, 0.0121, 0.0177, 0.378, 0.441),
        SensitivityRow(20, 20, 0.883, 0.125, 0.875, 0.142, 0.0148, 0.0190, 0.0282, 0.0480, 0.156, 0.343),
        SensitivityRow(20, 30, 0.769, 0.208, 0.714, 0.283, 0.0499, 0.0509, 0.0965, 0.0989, 0.000, 0.000),
    )


COLORS = {5: "#0072B2", 10: "#E69F00", 20: "#009E73"}
MARKERS = {5: "o", 10: "s", 20: "^"}


def _series(rows: tuple[SensitivityRow, ...], clients: int, field: str) -> list[float]:
    return [getattr(row, field) for row in rows if row.clients == clients]


def _ratios(rows: tuple[SensitivityRow, ...], clients: int) -> list[int]:
    return [row.malicious_ratio for row in rows if row.clients == clients]


def _errorbar(ax, x, y, yerr, clients: int, *, linestyle: str, filled: bool = True) -> None:
    color = COLORS[clients]
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=color,
        linestyle=linestyle,
        linewidth=0.9,
        marker=MARKERS[clients],
        markersize=3.2,
        markerfacecolor=color if filled else "white",
        markeredgewidth=0.7,
        elinewidth=0.65,
        capsize=1.4,
        capthick=0.65,
        zorder=3,
    )


def _format_axis(ax, *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=6.2, labelpad=1.2)
    ax.set_ylabel(ylabel, fontsize=6.2, labelpad=1.8)
    ax.tick_params(axis="both", which="major", labelsize=5.7, length=2.2, pad=1.2)
    ax.grid(True, linewidth=0.35, alpha=0.28)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def render(*, png_path: Path, pdf_path: Path):
    """Render the single-column three-panel summary as raster and vector assets."""
    rows = sensitivity_rows()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(3.5, 2.63))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.145,
        right=0.985,
        bottom=0.145,
        top=0.865,
        hspace=0.72,
        wspace=0.50,
    )
    utility_ax = figure.add_subplot(grid[0, :])
    weight_ax = figure.add_subplot(grid[1, 0])
    suppression_ax = figure.add_subplot(grid[1, 1])

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[clients],
            marker=MARKERS[clients],
            linewidth=0.9,
            markersize=3.2,
            label=f"N={clients}",
        )
        for clients in (5, 10, 20)
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.988),
        ncol=3,
        frameon=False,
        fontsize=5.8,
        handlelength=1.5,
        columnspacing=1.2,
    )

    for clients in (5, 10, 20):
        ratios = _ratios(rows, clients)
        _errorbar(
            utility_ax,
            ratios,
            _series(rows, clients, "accuracy_mean"),
            _series(rows, clients, "accuracy_sd"),
            clients,
            linestyle="-",
        )
        _errorbar(
            utility_ax,
            ratios,
            _series(rows, clients, "f1_mean"),
            _series(rows, clients, "f1_sd"),
            clients,
            linestyle="--",
            filled=False,
        )

        mean_x = [ratio - 0.35 for ratio in ratios]
        final_x = [ratio + 0.35 for ratio in ratios]
        _errorbar(
            weight_ax,
            mean_x,
            _series(rows, clients, "mean_weight_mean"),
            _series(rows, clients, "mean_weight_sd"),
            clients,
            linestyle="-",
        )
        _errorbar(
            weight_ax,
            final_x,
            _series(rows, clients, "final_weight_mean"),
            _series(rows, clients, "final_weight_sd"),
            clients,
            linestyle="--",
            filled=False,
        )
        _errorbar(
            suppression_ax,
            ratios,
            _series(rows, clients, "suppression_mean"),
            _series(rows, clients, "suppression_sd"),
            clients,
            linestyle="-",
        )

    utility_ax.set_title("(a) Global utility", fontsize=7.2, pad=1.5)
    utility_ax.text(
        0.02,
        0.04,
        "solid: accuracy; dashed/open: macro-F1",
        transform=utility_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.8,
    )
    utility_ax.set_xlim(8.5, 41.5)
    utility_ax.set_ylim(0.32, 1.07)
    utility_ax.set_xticks((10, 20, 30, 40))
    _format_axis(utility_ax, xlabel="Realized malicious-client ratio (%)", ylabel="Score")

    weight_ax.set_title("(b) Malicious influence", fontsize=7.0, pad=1.5)
    weight_ax.text(
        0.02,
        0.96,
        "solid/filled: mean\ndashed/open: final",
        transform=weight_ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.5,
    )
    weight_ax.set_xlim(8.0, 42.0)
    weight_ax.set_ylim(-0.012, 0.285)
    weight_ax.set_xticks((10, 20, 30, 40))
    _format_axis(weight_ax, xlabel="Malicious ratio (%)", ylabel="Total malicious weight")

    suppression_ax.set_title("(c) Suppression rate", fontsize=7.0, pad=1.5)
    suppression_ax.set_xlim(8.0, 42.0)
    suppression_ax.set_ylim(-0.08, 1.13)
    suppression_ax.set_xticks((10, 20, 30, 40))
    _format_axis(
        suppression_ax,
        xlabel="Malicious ratio (%)",
        ylabel="Suppressed-round fraction",
    )

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "TWA sensitivity to federation size and malicious-client ratio"}
    figure.savefig(png_path, dpi=600, facecolor="white", metadata=metadata)
    figure.savefig(pdf_path, facecolor="white", metadata=metadata)
    return figure


def main() -> None:
    directory = Path(__file__).resolve().parent
    figure = render(
        png_path=directory / "sensitivity_scalability_summary.png",
        pdf_path=directory / "sensitivity_scalability_summary.pdf",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
