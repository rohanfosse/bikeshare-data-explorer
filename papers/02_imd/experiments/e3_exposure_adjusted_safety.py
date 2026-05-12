"""E3 -- Cyclist-exposure-adjusted safety score (city-level variant).

The pre-registered E3 protocol calls for a station-level exposure
proxy via kriging of bike-counter flows. The available
eco-compteurs file in this release is city-level (one average daily
count per city), so we implement E3 at the same granularity: we
rescale the safety component by the city-level exposure and apply
empirical-Bayes shrinkage toward the national mean. The qualitative
test of E3 -- does the safety component cease to anti-correlate
with observed flows once exposure is divided out? -- remains valid
at city level.

Outputs:
    outputs/e3_results.json
    outputs/e3_safety_before_after.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    COMPONENTS,
    ROOT,
    calibrate_weights,
    composite_score,
    load_panel,
)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _adjusted_safety(
    crashes: np.ndarray,
    exposure: np.ndarray,
    *,
    shrinkage_strength: float = 1.0,
) -> np.ndarray:
    """Empirical-Bayes-shrunk crash rate per cyclist-year.

    Stations without exposure data fall back to the national mean
    crash rate to avoid biasing the ranking against cities with
    missing counters.
    """
    valid = np.isfinite(exposure) & (exposure > 0)
    rate = np.full_like(crashes, np.nan, dtype=float)
    rate[valid] = crashes[valid] / exposure[valid]
    if valid.sum() == 0:
        return np.zeros_like(crashes)
    national_mean = float(np.nanmean(rate))
    # Empirical-Bayes shrinkage: weight by 1 / (1 + shrinkage_strength)
    raw = np.where(valid, rate, national_mean)
    shrunk = (raw + shrinkage_strength * national_mean) / (1.0 + shrinkage_strength)
    # Min-max normalise the shrunk rate, then invert so that high score = safe
    lo, hi = shrunk.min(), shrunk.max()
    if hi == lo:
        return np.full_like(shrunk, 0.5)
    return 1.0 - (shrunk - lo) / (hi - lo)


def _city_level_panel(panel) -> pd.DataFrame:
    """Build a city-level frame with crash counts and exposure."""
    # Re-derive crash counts from station data
    import sys

    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations

    stations = load_stations()
    dock = stations[stations["station_type"] == "docked_bike"].copy()
    crashes_per_city = (
        dock.groupby("city")["baac_accidents_cyclistes"]
        .sum()
        .reset_index()
        .rename(columns={"baac_accidents_cyclistes": "n_crashes_total"})
    )

    eco = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources"
        / "eco_compteurs_city_usage.csv"
    )
    df = pd.DataFrame({
        "city": panel.cities,
        "IMD_pub": panel.imd,
        "M_multi": panel.components[:, 0],
        "I_infra": panel.components[:, 1],
        "S_securite_raw": panel.components[:, 2],
        "T_topo": panel.components[:, 3],
    })
    df = df.merge(crashes_per_city, on="city", how="left")
    df = df.merge(eco, on="city", how="left")
    df["exposure_cyclist_years"] = df["eco_avg_daily_bike_counts"] * 365.0 * 3.0
    return df


def main() -> None:
    log.info("Loading panel + crash + exposure data...")
    panel = load_panel()
    df = _city_level_panel(panel)
    log.info("  n=%d cities total", len(df))
    log.info("  n=%d cities with eco-counter exposure",
             df["eco_avg_daily_bike_counts"].notna().sum())

    crashes = df["n_crashes_total"].fillna(0.0).to_numpy(dtype=float)
    exposure = df["exposure_cyclist_years"].to_numpy(dtype=float)

    s_adjusted = _adjusted_safety(crashes, exposure, shrinkage_strength=0.5)
    df["S_securite_adj"] = s_adjusted

    # Correlations against eco-counter flows
    valid = df["eco_avg_daily_bike_counts"].notna()
    rho_raw, p_raw = sp_stats.spearmanr(
        df.loc[valid, "S_securite_raw"],
        df.loc[valid, "eco_avg_daily_bike_counts"],
    )
    rho_adj, p_adj = sp_stats.spearmanr(
        df.loc[valid, "S_securite_adj"],
        df.loc[valid, "eco_avg_daily_bike_counts"],
    )
    log.info("Safety vs eco-counter flow:")
    log.info("  raw: rho = %+.3f (p = %.3f)", rho_raw, p_raw)
    log.info("  exposure-adjusted: rho = %+.3f (p = %.3f)", rho_adj, p_adj)

    # Population-density check: adjusted S should NOT correlate with city size
    # more than raw S did.
    rho_raw_size, _ = sp_stats.spearmanr(df["S_securite_raw"], df["n_crashes_total"])
    rho_adj_size, _ = sp_stats.spearmanr(df["S_securite_adj"], df["n_crashes_total"])
    log.info("Safety vs raw crash count:")
    log.info("  raw: rho = %+.3f (S inversely tracks raw crashes by construction)",
             rho_raw_size)
    log.info("  exposure-adjusted: rho = %+.3f", rho_adj_size)

    # Re-calibrate weights with adjusted safety
    new_components = df[["M_multi", "I_infra", "S_securite_adj", "T_topo"]].to_numpy()
    new_components[:, 2] = s_adjusted  # ensure adjusted safety used
    w_adj, obj_adj = calibrate_weights(
        new_components, panel.fub, panel.emp, maxiter=200,
    )
    log.info("Recalibrated weights with adjusted S:")
    log.info("  M=%.3f I=%.3f S=%.3f T=%.3f, objective rho = %.3f",
             w_adj[0], w_adj[1], w_adj[2], w_adj[3], obj_adj)

    # Re-rank cities with adjusted IMD
    imd_adj = composite_score(w_adj, new_components)
    df["IMD_adj"] = imd_adj

    rank_shift = (
        df.sort_values("IMD_pub", ascending=False).reset_index(drop=True)
        .assign(rank_pub=lambda d: d.index + 1)[["city", "rank_pub"]]
        .merge(
            df.sort_values("IMD_adj", ascending=False).reset_index(drop=True)
            .assign(rank_adj=lambda d: d.index + 1)[["city", "rank_adj"]],
            on="city",
        )
    )
    rank_shift["delta"] = rank_shift["rank_adj"] - rank_shift["rank_pub"]
    tau_pub_adj, _ = sp_stats.kendalltau(
        rank_shift["rank_pub"], rank_shift["rank_adj"]
    )
    log.info("Kendall tau between published and exposure-adjusted ranking = %.3f",
             tau_pub_adj)
    log.info("Top 5 rank movers (positive = lost positions):")
    log.info("\n%s",
             rank_shift.sort_values("delta", key=lambda s: s.abs(),
                                    ascending=False).head(5).to_string(index=False))

    # New Top 10 with adjusted safety
    new_top = df.sort_values("IMD_adj", ascending=False).head(10)
    log.info("New Top 10 with exposure-adjusted safety:")
    log.info("\n%s", new_top[["city", "IMD_adj", "S_securite_adj",
                              "S_securite_raw"]].to_string(index=False))

    results = {
        "n_cities_total": int(len(df)),
        "n_cities_with_exposure": int(df["eco_avg_daily_bike_counts"].notna().sum()),
        "rho_safety_vs_flow": {
            "raw": float(rho_raw), "p_raw": float(p_raw),
            "adjusted": float(rho_adj), "p_adjusted": float(p_adj),
        },
        "recalibrated_weights_adjusted_S": {
            "M_multi": float(w_adj[0]),
            "I_infra": float(w_adj[1]),
            "S_securite_adj": float(w_adj[2]),
            "T_topo": float(w_adj[3]),
        },
        "calibration_objective_adjusted": float(obj_adj),
        "kendall_tau_pub_vs_adjusted_ranking": float(tau_pub_adj),
        "new_top_10": new_top[["city", "IMD_adj"]].to_dict("records"),
    }
    out_json = OUT_DIR / "e3_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Plot: raw vs adjusted safety against eco-counter flow
    valid_df = df.loc[valid].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=False)
    for ax, col, title, rho_val, p_val in [
        (axes[0], "S_securite_raw",
         "Raw safety component $C_S$",
         rho_raw, p_raw),
        (axes[1], "S_securite_adj",
         "Exposure-adjusted $C_S^{\\mathrm{adj}}$",
         rho_adj, p_adj),
    ]:
        ax.scatter(valid_df["eco_avg_daily_bike_counts"], valid_df[col],
                   s=34, color="#1F3A6B", alpha=0.78,
                   edgecolor="white", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Average daily bike counts (log)")
        ax.set_ylabel("Safety component (0 = unsafe, 1 = safe)")
        ax.set_title(title, fontsize=9)
        ax.text(0.04, 0.96,
                f"$\\rho$ = {rho_val:+.3f}\n(p = {p_val:.3f})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="#404040",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.85, "pad": 3})
        ax.grid(True, color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e3_safety_before_after.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e3_safety_before_after.pdf")


if __name__ == "__main__":
    main()
