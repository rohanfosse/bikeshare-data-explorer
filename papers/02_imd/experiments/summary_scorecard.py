"""Generate a one-glance scorecard of the twelve executed experiments.

Each experiment is summarised by:
  - its risk axis (calibration, construct, geometry, equity, panel)
  - a headline statistic with an acceptance threshold
  - a pass/fail/qualified verdict

Outputs:
    outputs/summary_scorecard.pdf
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

EXPERIMENTS = [
    # (id, risk axis, short, stat label, value, threshold, verdict_color)
    # Verdicts:
    #   clean       -- test ran as pre-specified, passes its threshold
    #   qualified   -- test passes but with a documented caveat
    #   substitute  -- pre-registered protocol deferred, proxy reported
    #                  in its place (E4 city-level Bayes vs station kriging,
    #                  E5 station bootstrap vs full re-enrichment,
    #                  E11 parametric sweep vs full re-enrichment)
    ("E1", "Calibration",
     "Leave-one-city-out CV",
     r"$\rho_{\mathrm{LOO}}$", 0.52, 0.30, "clean"),
    ("E2", "Calibration",
     "Weight-space LOO bootstrap",
     "Top-10 retention", 0.90, 0.50, "clean"),
    ("E3", "Construct",
     "Behavioural vs eco-counters",
     r"Partial $R^2$ net size", 0.24, 0.10, "clean"),
    ("E4", "Construct",
     "Exposure-adjusted safety (city-level proxy)",
     r"$\rho_S$ shift (raw $\to$ adj.)", 0.92, 0.50, "substitute"),
    ("E5", "Geometry",
     "Within-city station bootstrap (radius proxy)",
     r"Kendall $\tau$ (median)", 0.85, 0.70, "substitute"),
    ("E6", "Equity",
     "IES specification sweep",
     r"Robust deserts P$\geq$0.75", 0.83, 0.50, "clean"),
    ("E7", "Geometry",
     "Sobol/Jansen variance decomposition",
     r"$\sum_k S_k$ (score)", 0.98, 0.80, "clean"),
    ("E8", "Panel",
     "Spatial autocorrelation",
     r"$|$Moran $I|$", 0.025, 0.10, "clean"),
    ("E9", "Equity",
     "Bayesian IES (with prior sensitivity)",
     r"P-invariant deserts at $\tau\in\{0.1,1,10\}$", 4, 3, "qualified"),
    ("E10", "Panel",
     "Per-city IMD bootstrap CI",
     "Top-3 CI-distinct", 1.00, 0.80, "clean"),
    ("E11", "Geometry",
     "Parametric buffer-radius sweep (lit. elasticities)",
     r"min Kendall $\tau$ (150-500m)", 0.92, 0.70, "substitute"),
    ("E12", "Panel",
     "Cook-style leverage (Lyon-removed)",
     r"$\tau$(ref, $-$Lyon)", 0.83, 0.70, "qualified"),
    ("E13", "Panel",
     "k-means typology",
     r"Silhouette ($k=4$)", 0.30, 0.20, "clean"),
]

COLOR = {
    "clean": "#5B7E4F",
    "qualified": "#D08020",
    "substitute": "#7095C8",
    "fail": "#A8201A",
}


def main() -> None:
    n = len(EXPERIMENTS)
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * n + 0.8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    # Soft alternating row backgrounds
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F4F4F4", zorder=0)

    for i, (xid, axis, name, stat, value, threshold, verdict) in enumerate(EXPERIMENTS):
        color = COLOR[verdict]
        ax.text(-0.02, i, xid, ha="right", va="center",
                fontsize=9.5, color="#404040", fontweight="bold")
        ax.text(0.02, i - 0.15, name,
                ha="left", va="center", fontsize=9, color="#202020")
        ax.text(0.02, i + 0.18, f"[{axis}]",
                ha="left", va="center", fontsize=7.5, color="#7A7A7A",
                style="italic")
        ax.text(0.62, i - 0.15, stat,
                ha="right", va="center", fontsize=9, color="#202020")
        ax.text(0.62, i + 0.18,
                f"value = {value:.2f}   threshold = {threshold:.2f}",
                ha="right", va="center", fontsize=7.5, color="#7A7A7A")

        # Indicator chip
        chip_x, chip_w = 0.65, 0.35
        ax.add_patch(plt.Rectangle(
            (chip_x, i - 0.30), chip_w, 0.60,
            facecolor="white", edgecolor="#D0D0D0",
            linewidth=0.4, zorder=1,
        ))
        # Filled portion = min(value/threshold, 1) but capped at chip width
        # For E8 and E12 the metric is "smaller is better": flip
        if xid in ("E8", "E12"):
            ratio = 1.0 - min(value / max(threshold, 1e-6), 1.0)
        else:
            ratio = min(value / max(threshold, 1e-6) / 1.5, 1.0)
        ax.add_patch(plt.Rectangle(
            (chip_x + 0.01, i - 0.20),
            (chip_w - 0.02) * ratio, 0.40,
            facecolor=color, edgecolor="none", zorder=2, alpha=0.85,
        ))
        ax.text(chip_x + chip_w / 2 + 0.005, i,
                verdict.upper(),
                ha="center", va="center", fontsize=8,
                color="white" if ratio > 0.3 else "#404040",
                fontweight="bold", zorder=3)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Validation scorecard: twelve executed experiments",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_scorecard.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", OUT_DIR / "summary_scorecard.pdf")


if __name__ == "__main__":
    main()
