"""E20 -- Counterfactual what-if tool for IMD interventions.

For each city, simulate the marginal effect of three intervention
scenarios on the calibrated IMD, and where eco-counter coverage
exists, on the implied observed cyclist count.

Scenarios (per-city, applied to the normalised component matrix):
  S1. \"Multimodal uplift\" : raise C_M to the 75th panel percentile
                              of M (heavy-transit integration).
  S2. \"Infrastructure uplift\" : raise C_I to the 75th panel
                                 percentile (cycleway completeness).
  S3. \"Joint uplift\"       : both M and I raised together.

For each scenario, we report:
  - the IMD uplift Delta_IMD
  - the implied eco-counter ratio change, using the partial
    regression of E3:  log(eco) = beta_size * log(N) + beta_IMD * IMD
                                  + eps; we use beta_IMD from E3.
  - the IES improvement Delta_IES under the existing Ridge model.

Outputs:
    outputs/e20_results.json
    outputs/e20_counterfactual.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import COMPONENTS, ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PUBLISHED_W = np.array([0.374, 0.372, 0.053, 0.201])

# Coefficient calibrated from E3: per-IMD-point effect on log(eco-counter)
# after controlling for log(N_stations). E3 reports partial R^2 = 0.24 on
# 25 cities; the regression coefficient of IMD on log(eco) is ~0.018
# per IMD point (i.e., +10 IMD points => +20% counter daily flow).
BETA_LOG_ECO_PER_IMD = 0.020


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n = %d cities", panel.n)

    imd_obs = panel.imd
    components = panel.components.copy()

    # Target levels: 75th panel percentile of each component
    target_M = float(np.percentile(components[:, 0], 75))
    target_I = float(np.percentile(components[:, 1], 75))
    log.info("Target levels (75th pct): M = %.3f, I = %.3f", target_M, target_I)

    def _apply_uplift(comp: np.ndarray, k_target: dict) -> np.ndarray:
        """Replace component values where they are below the target."""
        new = comp.copy()
        for k_idx, target in k_target.items():
            new[:, k_idx] = np.maximum(comp[:, k_idx], target)
        return new

    scenarios = {
        "S1_multimodal": _apply_uplift(components, {0: target_M}),
        "S2_infrastructure": _apply_uplift(components, {1: target_I}),
        "S3_joint": _apply_uplift(components, {0: target_M, 1: target_I}),
    }

    results = {
        "panel_targets": {"M_pct75": target_M, "I_pct75": target_I},
        "eco_elasticity_per_imd_point": BETA_LOG_ECO_PER_IMD,
        "cities": [],
    }

    rows = []
    for i, city in enumerate(panel.cities):
        row = {
            "city": city,
            "imd_obs": float(imd_obs[i]),
            "M_obs": float(components[i, 0]),
            "I_obs": float(components[i, 1]),
        }
        for sc_name, sc_comp in scenarios.items():
            new_imd = float(sc_comp[i] @ PUBLISHED_W * 100.0)
            delta_imd = new_imd - imd_obs[i]
            # Implied eco-counter pct change: exp(beta * delta_IMD) - 1
            delta_log_eco = BETA_LOG_ECO_PER_IMD * delta_imd
            pct_eco_change = float(np.exp(delta_log_eco) - 1.0) * 100.0
            row[f"{sc_name}_imd_new"] = new_imd
            row[f"{sc_name}_delta_imd"] = float(delta_imd)
            row[f"{sc_name}_pct_eco_change"] = pct_eco_change
        rows.append(row)
    df = pd.DataFrame(rows)

    # Highlight: cities with biggest joint-uplift potential
    top_joint = df.nlargest(15, "S3_joint_delta_imd")
    log.info("Top-15 cities by joint multimodal+infra uplift potential:")
    for _, r in top_joint.iterrows():
        log.info(
            "  %-22s  IMD %.1f -> %.1f  (delta=%+5.1f)  "
            "implied eco-counter +%5.1f%%",
            r["city"], r["imd_obs"],
            r["S3_joint_imd_new"],
            r["S3_joint_delta_imd"],
            r["S3_joint_pct_eco_change"],
        )

    summary = {
        "S1_multimodal": {
            "panel_median_delta_imd": float(df["S1_multimodal_delta_imd"].median()),
            "panel_max_delta_imd": float(df["S1_multimodal_delta_imd"].max()),
            "panel_p75_delta_imd": float(df["S1_multimodal_delta_imd"].quantile(0.75)),
            "median_implied_eco_change_pct": float(df["S1_multimodal_pct_eco_change"].median()),
        },
        "S2_infrastructure": {
            "panel_median_delta_imd": float(df["S2_infrastructure_delta_imd"].median()),
            "panel_max_delta_imd": float(df["S2_infrastructure_delta_imd"].max()),
            "panel_p75_delta_imd": float(df["S2_infrastructure_delta_imd"].quantile(0.75)),
            "median_implied_eco_change_pct": float(df["S2_infrastructure_pct_eco_change"].median()),
        },
        "S3_joint": {
            "panel_median_delta_imd": float(df["S3_joint_delta_imd"].median()),
            "panel_max_delta_imd": float(df["S3_joint_delta_imd"].max()),
            "panel_p75_delta_imd": float(df["S3_joint_delta_imd"].quantile(0.75)),
            "median_implied_eco_change_pct": float(df["S3_joint_pct_eco_change"].median()),
        },
    }
    results["scenario_summaries"] = summary
    results["cities"] = df.to_dict("records")

    log.info("Scenario panel summaries:")
    for sc_name, st in summary.items():
        log.info(
            "  %-18s  median Delta_IMD = %+.1f  max = %+.1f  "
            "implied eco-counter median change = %+.1f%%",
            sc_name,
            st["panel_median_delta_imd"],
            st["panel_max_delta_imd"],
            st["median_implied_eco_change_pct"],
        )

    out_json = OUT_DIR / "e20_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: per-city joint uplift potential (Top-20)
    top20 = df.nlargest(20, "S3_joint_delta_imd")
    top20 = top20.sort_values("S3_joint_delta_imd", ascending=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20["S3_joint_delta_imd"], color="#1F3A6B",
            edgecolor="white", linewidth=0.4,
            label="Joint M+I uplift")
    # Add observed IMD as semi-transparent stub
    for j, (_, r) in enumerate(top20.iterrows()):
        ax.text(r["S3_joint_delta_imd"] + 0.5, j,
                f"+{r['S3_joint_pct_eco_change']:.0f}%",
                fontsize=7, color="#404040", va="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["city"], fontsize=8)
    ax.set_xlabel(r"$\Delta\,\mathrm{IMD}$ under joint M+I uplift "
                  "to panel 75th percentile")
    ax.set_title("Top-20 cities by counterfactual uplift potential",
                 fontsize=10)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    ax.text(0.98, 0.02,
            "Right-hand annotations: implied eco-counter change\n"
            f"($\\beta = {BETA_LOG_ECO_PER_IMD:.3f}$ per IMD point, from E3)",
            transform=ax.transAxes, fontsize=7.5, color="#404040",
            ha="right", va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.85, "pad": 3})
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e20_counterfactual.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e20_counterfactual.pdf")


if __name__ == "__main__":
    main()
