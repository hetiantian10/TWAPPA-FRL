"""Generate the matched-baseline clean-accuracy comparison used in Figure 2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).with_name("accuracy_loss_comparison.pdf")


def main() -> None:
    # Citation numbers follow the current main-paper bibliography order:
    # FLDetector [23], FLAME [24], FLTrust [21], and FoolsGold [22].
    labels = ["[23]", "[24]", "Ours", "[21]", "[22]"]
    y = np.arange(len(labels))[::-1]

    # Values are percentage-point deviations from each work's matched baseline.
    representative = np.array([0.23, 0.30, 0.06, 0.83, 1.00])
    ranges = [(0.00, 1.50), (0.20, 0.40), None, (0.00, 2.00), (0.00, 2.00)]

    fig, ax = plt.subplots(figsize=(7.1033, 3.70))
    gray = "#949ba1"
    dark_gray = "#747d84"
    blue = "#1885bf"

    for idx, bounds in enumerate(ranges):
        if bounds is None:
            continue
        linestyle = "--" if idx == 4 else "-"
        ax.hlines(
            y[idx],
            bounds[0],
            bounds[1],
            color=gray,
            linewidth=3.0,
            linestyle=linestyle,
            zorder=1,
        )

    ax.scatter(
        representative[[0, 1, 3]],
        y[[0, 1, 3]],
        s=82,
        color=dark_gray,
        zorder=3,
    )
    ax.scatter(
        representative[2],
        y[2],
        s=92,
        color=blue,
        zorder=4,
    )
    ax.scatter(
        representative[4],
        y[4],
        s=76,
        facecolors="white",
        edgecolors=dark_gray,
        linewidths=2.0,
        zorder=4,
    )

    value_labels = ["0.23 pp", "0.30 pp", "0.06 pp", "0.83 pp", "≈1.00 pp"]
    offsets = [(0, -14), (0, -14), (16, 0), (0, -14), (0, -14)]
    alignments = [("center", "center"), ("center", "center"), ("left", "center"),
                  ("center", "center"), ("center", "center")]
    for idx, text in enumerate(value_labels):
        ax.annotate(
            text,
            (representative[idx], y[idx]),
            xytext=offsets[idx],
            textcoords="offset points",
            ha=alignments[idx][0],
            va=alignments[idx][1],
            fontsize=10,
            fontweight="bold" if idx == 2 else "normal",
        )

    ax.set_yticks(y, labels)
    ax.set_xlim(0.00, 2.15)
    ax.set_ylim(-0.65, 4.65)
    ticks = np.arange(0.00, 2.01, 0.25)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
    ax.set_xlabel("Clean-accuracy deviation from matched baseline (pp)", fontsize=11)

    ax.grid(axis="x", color="#dddddd", linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=11, length=0, pad=14)

    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.23, top=0.97)
    fig.savefig(OUTPUT, format="pdf", metadata={"Creator": "Matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    main()
