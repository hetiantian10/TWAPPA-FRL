from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Run from the repository root with:
# python article/figures/generate_roundwise_comm_overhead.py
out_dir = Path(__file__).resolve().parent

# Source-scoped ratios reported in Table VI. A status of "na" means that the
# source does not report a like-for-like plaintext/FedAvg expansion ratio.
SOURCE_SCOPED_EXPANSION = [
    {"work": "Bonawitz", "label": "[18]", "status": "ratio", "low": 1.73, "high": 1.98, "scope": "clear/client"},
    {"work": "BatchCrypt", "label": "[19]", "status": "ratio", "low": 1.60, "high": 2.50, "scope": "PT/worker/iteration"},
    {"work": "Manh", "label": "[6]", "status": "na"},
    {"work": "Hercules", "label": "[20]", "status": "na"},
    {"work": "FLTrust", "label": "[21]", "status": "na"},
    {"work": "FoolsGold", "label": "[22]", "status": "na"},
    {"work": "FLDetector", "label": "[23]", "status": "na"},
    {"work": "FLAME", "label": "[24]", "status": "na"},
    {"work": "WW-FL", "label": "[25]", "status": "na"},
    {"work": "RFA", "label": "[26]", "status": "ratio", "low": 1.00, "high": 3.00, "scope": "FedAvg/RA"},
    {"work": "Ours: TWA-PPA", "label": "TWAPPA-FRL", "status": "ratio", "low": 13.35, "high": 13.46, "scope": "PT/client/round"},
]


def na():
    return {"status": "na"}


def absolute_annotation_layout(work: str, field: str) -> tuple[tuple[int, int], str, str]:
    """Return panel (c) label offsets as (points, vertical, horizontal)."""
    if field == "system_ft" and work in {"BatchCrypt", "Hercules"}:
        return (0, -3), "top", "center"
    if (work, field) == ("BatchCrypt", "iteration"):
        return (2, 2), "bottom", "center"
    if (work, field) == ("WW-FL", "iteration"):
        return (0, 4), "bottom", "center"
    if (work, field) == ("Ours: TWA-PPA", "iteration"):
        return (3, 4), "bottom", "left"
    if (work, field) == ("Ours: TWA-PPA", "participant_ft"):
        return (3, 2), "bottom", "left"
    if (work, field) == ("Ours: TWA-PPA", "system_ft"):
        return (3, 0), "center", "left"
    return (0, 2), "bottom", "center"


# Absolute values are retained in their source scopes and normalized to MiB
# only for a common logarithmic plotting coordinate.
ABSOLUTE_COMMUNICATION_SCHEMA = [
    {"work": "Bonawitz", "label": "[18]", "iteration": na(), "participant_ft": na(), "system_ft": na()},
    {"work": "BatchCrypt", "label": "[19]", "iteration": {"status": "range", "low_mib": 2 / 1.048576, "high_mib": 49 / 1.048576, "scope": "worker/iteration", "display": "1.91–46.73 MiB"}, "participant_ft": {"status": "range", "low_mib": 58.7 * 1024 / 9, "high_mib": 227.8 * 1024 / 9, "scope": "worker/FT (derived)", "display": "6.52–25.31 GB"}, "system_ft": {"status": "range", "low_mib": 58.7 * 1024, "high_mib": 227.8 * 1024, "scope": "projected system/FT", "display": "58.7–227.8 GB"}},
    {"work": "Manh", "label": "[6]", "iteration": na(), "participant_ft": na(), "system_ft": na()},
    {"work": "Hercules", "label": "[20]", "iteration": {"status": "range", "low_mib": 6.00064, "high_mib": 155.99995, "scope": "user/global iteration (derived)", "display": "6–156 MiB"}, "participant_ft": {"status": "range", "low_mib": 0.59 * 1024, "high_mib": 8226.56 * 1024, "scope": "user/FT", "display": "0.59–8226.56 GB"}, "system_ft": {"status": "range", "low_mib": 5.9 * 1024, "high_mib": 411328 * 1024, "scope": "system/FT (derived)", "display": "5.9–411,328 GB"}},
    {"work": "FLTrust", "label": "[21]", "iteration": na(), "participant_ft": na(), "system_ft": na()},
    {"work": "FoolsGold", "label": "[22]", "iteration": na(), "participant_ft": na(), "system_ft": na()},
    {"work": "FLDetector", "label": "[23]", "iteration": na(), "participant_ft": na(), "system_ft": na()},
    {"work": "FLAME", "label": "[24]", "iteration": na(), "participant_ft": na(), "system_ft": na()},
    {"work": "WW-FL", "label": "[25]", "iteration": {"status": "point", "low_mib": 105.8 / 1.048576, "high_mib": 105.8 / 1.048576, "scope": "selected client/iteration (derived)", "display": "100.90 MiB"}, "participant_ft": {"status": "point", "low_mib": 26.45 * 1024, "high_mib": 26.45 * 1024, "scope": "enrolled client/FT (derived)", "display": "26.45 GB"}, "system_ft": {"status": "point", "low_mib": 26.45 * 1024 * 1024, "high_mib": 26.45 * 1024 * 1024, "scope": "system/FT", "display": "26.45 TB"}},
    {"work": "RFA", "label": "[26]", "iteration": na(), "participant_ft": na(), "system_ft": na()},
    {"work": "Ours: TWA-PPA", "label": "TWAPPA-FRL", "iteration": {"status": "point", "low_mib": 2.745, "high_mib": 2.745, "scope": "client/round", "display": "2.745 MiB"}, "participant_ft": {"status": "point", "low_mib": 82.338, "high_mib": 82.338, "scope": "client/FT", "display": "82.338 MiB"}, "system_ft": {"status": "point", "low_mib": 411.69, "high_mib": 411.69, "scope": "system/FT", "display": "411.69 MiB"}},
]


def plot_source_scoped_expansion(ax: plt.Axes) -> None:
    y = np.arange(len(SOURCE_SCOPED_EXPANSION))
    numeric_color = "#1f77b4"
    na_color = "#8c8c8c"

    for position, item in zip(y, SOURCE_SCOPED_EXPANSION):
        if item["status"] == "na":
            ax.scatter(30, position, marker="s", s=18, facecolors="none", edgecolors=na_color, linewidths=0.9, zorder=3)
            continue

        low, high = item["low"], item["high"]
        ax.hlines(position, low, high, color=numeric_color, linewidth=1.4, zorder=2)
        ax.scatter([low, high], [position, position], color=numeric_color, s=15, zorder=3)
        if item["work"] in {"Bonawitz", "BatchCrypt"}:
            ax.annotate(f"{low:.2f}–{high:.2f}×", (high, position), xytext=(3, 0), textcoords="offset points", va="center", ha="left", fontsize=5.4)
        else:
            ax.annotate(f"{low:.2f}–{high:.2f}×", ((low * high) ** 0.5, position), xytext=(0, 2), textcoords="offset points", va="bottom", ha="center", fontsize=5.4)

    ax.axvline(24, color="#b0b0b0", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 42)
    ax.set_xticks([1, 2, 3, 10, 20])
    ax.set_xticklabels(["1×", "2×", "3×", "10×", "20×"])
    ax.set_yticks(y)
    ax.set_yticklabels([item["label"] for item in SOURCE_SCOPED_EXPANSION], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_title("(a) Source-scoped reported expansion", fontsize=9)
    ax.set_xlabel("Reported expansion factor (log scale)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax.legend(
        handles=[
            Line2D([0], [0], color=numeric_color, marker="o", markersize=3, linewidth=1, label="reported ratio"),
            Line2D([0], [0], color=na_color, marker="s", markerfacecolor="none", markersize=3, linewidth=0, label="N/A"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=6.5,
    )


def plot_source_scoped_absolute_communication(ax: plt.Axes) -> None:
    y = np.arange(len(ABSOLUTE_COMMUNICATION_SCHEMA))
    fields = [
        ("iteration", -0.22, "#1b9e77", "participant / iteration"),
        ("participant_ft", 0.00, "#377eb8", "participant / FT"),
        ("system_ft", 0.22, "#d95f02", "system / FT"),
    ]
    na_color = "#8c8c8c"

    for position, item in zip(y, ABSOLUTE_COMMUNICATION_SCHEMA):
        for field, offset, color, _ in fields:
            value = item[field]
            lane = position + offset
            if value["status"] == "na":
                ax.scatter(2e8, lane, marker="s", s=12, facecolors="none", edgecolors=na_color, linewidths=0.7, zorder=3)
                continue

            low, high = value["low_mib"], value["high_mib"]
            if value["status"] == "range":
                ax.hlines(lane, low, high, color=color, linewidth=1.0, zorder=2)
                ax.scatter([low, high], [lane, lane], color=color, s=10, zorder=3)
            else:
                ax.scatter(low, lane, color=color, s=13, zorder=3)
            xytext, va, ha = absolute_annotation_layout(item["work"], field)
            anchor_x = low if (item["work"], field) in {("Ours: TWA-PPA", "iteration"), ("Ours: TWA-PPA", "participant_ft"), ("Ours: TWA-PPA", "system_ft")} else (low * high) ** 0.5
            ax.annotate(value["display"], (anchor_x, lane), xytext=xytext, textcoords="offset points", va=va, ha=ha, fontsize=4.2, color=color)

    ax.set_xscale("log")
    ax.set_xlim(1, 1e9)
    ax.set_xticks([10, 1024, 1024 * 100, 1024 * 1024 * 10])
    ax.set_xticklabels(["10 MiB", "1 GiB", "100 GiB", "10 TiB"])
    ax.set_yticks(y)
    ax.set_yticklabels([item["label"] for item in ABSOLUTE_COMMUNICATION_SCHEMA], fontsize=5.4)
    ax.invert_yaxis()
    ax.set_title("(b) Source-scoped reported absolute communication", fontsize=8.5, pad=4)
    ax.set_xlabel("Reported communication volume (log scale)", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    ax.legend(
        handles=[Line2D([0], [0], color=color, marker="o", markersize=3, linewidth=1, label=label) for _, _, color, label in fields] + [Line2D([0], [0], color=na_color, marker="s", markerfacecolor="none", markersize=3, linewidth=0, label="N/A")],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=6.5,
    )


def main() -> None:
    fig, (expansion_axis, absolute_axis) = plt.subplots(
        2,
        1,
        figsize=(3.45, 5.55),
        gridspec_kw={"height_ratios": [1.15, 1.65]},
    )
    plot_source_scoped_expansion(expansion_axis)
    plot_source_scoped_absolute_communication(absolute_axis)
    fig.subplots_adjust(bottom=0.15, top=0.96, hspace=0.50)
    absolute_axis.set_position(absolute_axis.get_position().translated(0, 0.015))

    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"roundwise_comm_overhead_combined_mb.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
