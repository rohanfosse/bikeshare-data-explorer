"""E12 -- Leave-one-out leverage diagnostic on the calibration panel.

For composite indicators calibrated against external behavioural
references on a finite panel (here 32 FUB + 44 EMP cities), individual
cities can exert disproportionate influence on the supervised weight
vector. We quantify this leverage by Cook's-style distance:
for each city i, we re-calibrate the weights on the panel without i
and measure
    D_i = || w(-i) - w* ||_2 / || w* ||_2
plus the change in the panel-mean Spearman correlation
    delta_rho_i = bar_rho(-i) - bar_rho*

A city is "high leverage" if D_i exceeds the 90th percentile of the
panel. The diagnostic identifies which cities, if removed from the
calibration, would most reshape the weight vector. The reading is
*compositional* (sensitivity to panel composition) and complements
the cross-validation diagnostic of E1 (which measures generalisation).

Outputs:
    outputs/e12_results.json
    outputs/e12_leverage.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from _common import COMPONENTS, calibrate_weights, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    n = panel.n
    log.info("  n=%d cities", n)

    # Reference weights (in-sample)
    log.info("Reference calibration...")
    w_ref, rho_ref = calibrate_weights(
        panel.components, panel.fub, panel.emp, seed=42, maxiter=200,
    )
    log.info("  w* = %s   bar_rho = %.3f",
             dict(zip(COMPONENTS, [round(float(x), 3) for x in w_ref])),
             rho_ref)

    leverage = []
    for i in range(n):
        # Only include cities that contribute to the calibration objective
        if not (np.isfinite(panel.fub[i]) or np.isfinite(panel.emp[i])):
            continue
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        try:
            w_i, rho_i = calibrate_weights(
                panel.components[keep],
                panel.fub[keep],
                panel.emp[keep],
                seed=42 + i,
                maxiter=120,
            )
        except Exception as exc:
            log.warning("LOO fold %s failed: %s", panel.cities[i], exc)
            continue

        d = float(np.linalg.norm(w_i - w_ref) / max(np.linalg.norm(w_ref), 1e-9))
        # delta_rho on the held-out panel (exclude city i)
        delta_rho = float(rho_i - rho_ref)
        leverage.append({
            "city": panel.cities[i],
            "d_cook_like": d,
            "delta_rho": delta_rho,
            "w_minus_i": dict(zip(COMPONENTS, [float(x) for x in w_i])),
        })

    leverage.sort(key=lambda r: -r["d_cook_like"])
    d_values = np.array([r["d_cook_like"] for r in leverage])
    delta_rho_values = np.array([r["delta_rho"] for r in leverage])
    p90 = float(np.percentile(d_values, 90))
    p50 = float(np.percentile(d_values, 50))

    high_lev_cities = [r["city"] for r in leverage if r["d_cook_like"] >= p90]
    log.info("High-leverage cities (D_i >= 90th pct = %.4f):", p90)
    for r in leverage[: max(10, len(high_lev_cities))]:
        log.info(
            "  %-22s  D = %.4f   delta_rho = %+.4f",
            r["city"], r["d_cook_like"], r["delta_rho"],
        )

    results = {
        "n_loo_folds": int(len(leverage)),
        "w_reference": dict(zip(COMPONENTS, [float(x) for x in w_ref])),
        "rho_reference": float(rho_ref),
        "median_d_cook_like": p50,
        "p90_d_cook_like": p90,
        "high_leverage_cities": high_lev_cities,
        "median_abs_delta_rho": float(np.median(np.abs(delta_rho_values))),
        "max_abs_delta_rho": float(np.max(np.abs(delta_rho_values))),
        "per_city": leverage,
    }
    out_json = OUT_DIR / "e12_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)
    log.info("Median |delta_rho| = %.4f, max |delta_rho| = %.4f",
             np.median(np.abs(delta_rho_values)),
             np.max(np.abs(delta_rho_values)))

    # Figure: D_i bar chart for top-15 highest-leverage cities
    top_n = 15
    top = leverage[:top_n]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                              gridspec_kw={"width_ratios": [1, 1.05]})
    cities_top = [r["city"] for r in top]
    d_top = np.array([r["d_cook_like"] for r in top])
    delta_top = np.array([r["delta_rho"] for r in top])

    colors = ["#A8201A" if d >= p90 else "#1F3A6B" for d in d_top]
    axes[0].barh(cities_top[::-1], d_top[::-1],
                 color=colors[::-1], edgecolor="white", linewidth=0.4)
    axes[0].axvline(p90, color="#A8201A", linewidth=0.7,
                    linestyle=":", alpha=0.7, label=f"90th pct = {p90:.3f}")
    axes[0].set_xlabel(r"$D_i = \|w_{-i} - w^\ast\|_2 / \|w^\ast\|_2$")
    axes[0].set_title("Per-city influence on the calibration", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    bar_colors = ["#A8201A" if dr < 0 else "#5B7E4F" for dr in delta_top]
    axes[1].barh(cities_top[::-1], delta_top[::-1],
                 color=bar_colors[::-1], edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="#404040", linewidth=0.6, alpha=0.7)
    axes[1].set_xlabel(
        r"$\Delta \bar\rho = \bar\rho_{-i} - \bar\rho^\ast$",
    )
    axes[1].set_title("Change in calibration fit on removal",
                      fontsize=10)
    axes[1].tick_params(axis="y", labelsize=8)
    axes[1].grid(True, axis="x", color="#E5E5E5", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "e12_leverage.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e12_leverage.pdf")


if __name__ == "__main__":
    main()
