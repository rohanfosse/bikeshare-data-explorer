"""B2 -- European panel under the Bayesian IMD-3 weights.

Builds on B1: takes the posterior on the three IMD weights
calibrated on the French panel and applies it to the ten
European systems with successful OSM Overpass multimodality
counts from the previous paper's E26 experiment. Cycling-
infrastructure and topography are unavailable on the European
panel under our pipeline, so the European IMD-lite reported here
uses only multimodality (which carries 82% of posterior weight)
and treats $C_I$ and $C_T$ as fixed at the French panel mean.

This is a deliberately limited exercise: it tells us how the
European cities would rank \emph{if their cycling infrastructure
and topography were average for the French panel}, isolating
the multimodality dimension. The resulting EU IMD-3 is therefore
a multimodality-driven lower bound (for cities with average or
worse I, T) or upper bound (for cities with better I, T).

Reuses the E26 OSM-based multimodality numbers and applies the
B1 weight posterior.

Outputs:
    outputs/b2_european_results.json
    outputs/b2_european_ranking.pdf
"""
from __future__ import annotations

import json
import logging
import sys
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [HERE, *HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "papers" / "02_imd" / "experiments"))

from _common import load_panel  # noqa: E402
from utils.data_loader import load_stations  # noqa: E402

B1_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b1_bayesian_imd.py"
spec = importlib.util.spec_from_file_location("b1", B1_PATH)
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
E26_PATH = ROOT / "papers" / "02_imd" / "experiments" / "outputs" / "e26_results.json"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(2026)
N_DRAWS = 500


def main() -> None:
    if not E26_PATH.exists():
        log.error("E26 results not found at %s; run E26 first", E26_PATH)
        return
    e26 = json.loads(E26_PATH.read_text(encoding="utf-8"))
    eu_M_raw = e26["eu_M"]
    log.info("Loaded %d European M values from E26", len(eu_M_raw))

    # Run B1 to obtain the weight posterior
    log.info("Loading panel and stations + running B1 MH sampler...")
    panel = load_panel()
    stations = load_stations()
    dock = b1.normalise_components(stations)
    city_means = dock.groupby("city")[["M_norm", "I_norm", "T_norm"]].mean()
    cmm = city_means.reindex(panel.cities).fillna(city_means.median())
    component_city_means = cmm.to_numpy()
    fub = b1.standardise(panel.fub)
    emp = b1.standardise(np.log1p(panel.emp))
    chain = b1.mh_sample(component_city_means, fub, emp,
                          n_burn=b1.N_BURN, n_keep=b1.N_KEEP)
    z_samples = chain["z"]

    # Need the same min/max for raw M as the French panel for fair scaling.
    # Use the panel-station raw M to compute the global min/max.
    raw_M_panel_station = dock["gtfs_heavy_stops_300m"].fillna(
        dock["gtfs_heavy_stops_300m"].median()
    ).to_numpy()
    M_min, M_max = raw_M_panel_station.min(), raw_M_panel_station.max()
    log.info("French panel raw M: min=%.2f, max=%.2f", M_min, M_max)

    # Normalise EU raw M to the same Min-Max scale
    eu_M_norm = {
        city: (val - M_min) / (M_max - M_min) for city, val in eu_M_raw.items()
    }
    eu_M_norm = {k: min(max(v, 0.0), 1.0) for k, v in eu_M_norm.items()}
    log.info("EU M normalised to FR panel scale:")
    for c, v in sorted(eu_M_norm.items(), key=lambda x: -x[1]):
        log.info("  %-15s  raw = %.2f  normalised = %.3f",
                 c, eu_M_raw[c], v)

    # French panel I and T means (used as defaults for European cities)
    I_mean_FR = float(dock["I_norm"].mean())
    T_mean_FR = float(dock["T_norm"].mean())
    log.info("French panel means: I_norm = %.3f, T_norm = %.3f",
             I_mean_FR, T_mean_FR)

    # Draw weight samples
    idx = RNG.choice(len(z_samples), size=N_DRAWS, replace=False)
    w_draws = np.array([b1.softmax_with_floor(z_samples[i]) for i in idx])

    # Compute European city IMD posterior
    results = {}
    for city, M in eu_M_norm.items():
        comps = np.array([M, I_mean_FR, T_mean_FR])
        imd_draws = (w_draws @ comps) * 100.0
        results[city] = {
            "imd_median": float(np.median(imd_draws)),
            "imd_q025": float(np.percentile(imd_draws, 2.5)),
            "imd_q975": float(np.percentile(imd_draws, 97.5)),
            "imd_sd": float(np.std(imd_draws)),
            "M_norm": float(M),
            "M_raw": float(eu_M_raw[city]),
        }

    # French panel city-level posterior under the same scale for comparison
    fr_results = {}
    for i, city in enumerate(panel.cities):
        comps = cmm.iloc[i].to_numpy()
        imd_draws = (w_draws @ comps) * 100.0
        fr_results[city] = {
            "imd_median": float(np.median(imd_draws)),
            "imd_q025": float(np.percentile(imd_draws, 2.5)),
            "imd_q975": float(np.percentile(imd_draws, 97.5)),
        }

    # Comparison summary
    eu_medians = np.array([r["imd_median"] for r in results.values()])
    fr_medians = np.array([r["imd_median"] for r in fr_results.values()])
    log.info("\nIMD-3 medians (under FR weight posterior):")
    log.info("  EU (n=%d):  median = %.1f, max = %.1f, min = %.1f",
             len(eu_medians), np.median(eu_medians),
             eu_medians.max(), eu_medians.min())
    log.info("  FR (n=%d):  median = %.1f, max = %.1f, min = %.1f",
             len(fr_medians), np.median(fr_medians),
             fr_medians.max(), fr_medians.min())

    # Combined ranking
    combined = []
    for c, r in results.items():
        combined.append({"city": c, "country": "EU", **r})
    for c, r in fr_results.items():
        combined.append({"city": c, "country": "FR",
                          "imd_median": r["imd_median"],
                          "imd_q025": r["imd_q025"],
                          "imd_q975": r["imd_q975"],
                          "imd_sd": 0.0, "M_norm": None, "M_raw": None})
    combined.sort(key=lambda r: -r["imd_median"])

    log.info("Combined Top-15:")
    for r in combined[:15]:
        log.info("  %s | %-20s | IMD = %.1f [%.1f, %.1f]",
                 r["country"], r["city"], r["imd_median"],
                 r["imd_q025"], r["imd_q975"])

    out = {
        "eu_panel": results,
        "fr_panel": fr_results,
        "summary": {
            "eu_median_imd": float(np.median(eu_medians)),
            "fr_median_imd": float(np.median(fr_medians)),
            "ratio_eu_to_fr": float(np.median(eu_medians) / np.median(fr_medians)),
            "n_eu": int(len(eu_medians)),
            "n_fr": int(len(fr_medians)),
        },
        "top15_combined": combined[:15],
    }
    out_json = OUT_DIR / "b2_european_results.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: combined ranking Top-20
    top20 = combined[:20][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    y_pos = np.arange(len(top20))
    colors = ["#D08020" if r["country"] == "EU" else "#1F3A6B" for r in top20]
    err = np.array([[r["imd_median"] - r["imd_q025"] for r in top20],
                    [r["imd_q975"] - r["imd_median"] for r in top20]])
    ax.errorbar([r["imd_median"] for r in top20], y_pos, xerr=err,
                fmt="o", capsize=3, markersize=5,
                elinewidth=0.9, capthick=0.9,
                color="#404040", ecolor="#404040")
    for j, r in enumerate(top20):
        ax.scatter(r["imd_median"], j, color=colors[j], s=70, zorder=3,
                    edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['country']} | {r['city']}" for r in top20],
                       fontsize=8)
    ax.set_xlabel(r"Bayesian IMD-3 (95\% CrI under FR-calibrated weights)")
    ax.set_title("B2: combined Top-20 (FR + EU non-FR)\n"
                 "EU cities use observed M; I and T set to FR panel means",
                 fontsize=10)
    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#1F3A6B", markersize=8, label="France"),
        Line2D([0], [0], marker="o", color="w",
                markerfacecolor="#D08020", markersize=8, label="Europe (non-FR)"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b2_european_ranking.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b2_european_ranking.pdf")


if __name__ == "__main__":
    main()
