"""E6 -- Out-of-sample validation against eco-counter flow data.

The IMD claims to capture cycling-environment quality. If true, it
should correlate with observed cyclist flows (eco-counters) even
though counters did not enter the calibration.

Outputs:
    outputs/e6_results.json
    outputs/e6_imd_vs_counter.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from _common import COMPONENTS, ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Loading panel + eco-counter data...")
    panel = load_panel()
    eco = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources"
        / "eco_compteurs_city_usage.csv"
    )
    imd_df = pd.DataFrame({
        "city": panel.cities,
        "IMD": panel.imd,
        "M_multi": panel.components[:, 0],
        "I_infra": panel.components[:, 1],
        "S_securite": panel.components[:, 2],
        "T_topo": panel.components[:, 3],
        "n_stations": [
            int((stations := pd.DataFrame())) if False else 0
            for _ in panel.cities
        ],
    })
    # Replace n_stations with the actual values from stations table
    from utils.data_loader import load_stations as _ls
    stations = _ls()
    dock = stations[stations["station_type"] == "docked_bike"]
    n_per_city = dock.groupby("city").size().rename("n_stations")
    imd_df = imd_df.drop(columns=["n_stations"]).merge(
        n_per_city.reset_index(), on="city", how="left"
    )

    df = imd_df.merge(eco, on="city", how="inner")
    df = df.dropna(subset=["eco_avg_daily_bike_counts"]).reset_index(drop=True)
    log.info("  n=%d cities with eco-counter coverage", len(df))

    # 1) Bivariate Spearman (IMD vs daily counts)
    rho, pval = sp_stats.spearmanr(df["IMD"], df["eco_avg_daily_bike_counts"])
    rho_log, _ = sp_stats.spearmanr(
        df["IMD"], np.log(df["eco_avg_daily_bike_counts"])
    )

    # 2) Partial R^2 of IMD net of city size (n_stations as proxy)
    df["log_counts"] = np.log(df["eco_avg_daily_bike_counts"])
    df["log_size"] = np.log(df["n_stations"].clip(lower=1))

    scaler = StandardScaler()
    x_full = scaler.fit_transform(df[["IMD", "log_size"]])
    x_size = scaler.fit_transform(df[["log_size"]])
    y = df["log_counts"].to_numpy()

    full_model = LinearRegression().fit(x_full, y)
    size_model = LinearRegression().fit(x_size, y)
    r2_full = full_model.score(x_full, y)
    r2_size = size_model.score(x_size, y)
    partial_r2_imd = max(0.0, r2_full - r2_size)

    # 3) Partial Spearman of each component against counts (controlling size)
    component_partials: dict[str, dict] = {}
    for comp in COMPONENTS:
        x_comp = scaler.fit_transform(df[[comp, "log_size"]])
        m_full = LinearRegression().fit(x_comp, y)
        partial = max(0.0, m_full.score(x_comp, y) - r2_size)
        component_partials[comp] = {
            "spearman": float(sp_stats.spearmanr(df[comp], df["log_counts"]).statistic),
            "partial_r2_net_size": float(partial),
        }

    results = {
        "n_cities": int(len(df)),
        "spearman_imd_vs_counts": float(rho),
        "p_imd_vs_counts": float(pval),
        "spearman_imd_vs_log_counts": float(rho_log),
        "r2_imd_and_size": float(r2_full),
        "r2_size_only": float(r2_size),
        "partial_r2_imd_net_size": float(partial_r2_imd),
        "components": component_partials,
        "cities_used": df["city"].tolist(),
    }

    out_json = OUT_DIR / "e6_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)
    log.info("Spearman(IMD, daily counts) = %.3f (p = %.3f)", rho, pval)
    log.info("Partial R^2 of IMD net of city size = %.3f", partial_r2_imd)
    log.info("Component partial R^2 net of size:")
    for c, vals in component_partials.items():
        log.info("  %s: partial_r2 = %.3f, spearman = %+.3f",
                 c, vals["partial_r2_net_size"], vals["spearman"])

    # Figure: IMD vs log(daily counts)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(df["IMD"], df["eco_avg_daily_bike_counts"],
               s=36, color="#1F3A6B", alpha=0.78,
               edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("IMD (0--100)")
    ax.set_ylabel("Average daily bike counts (log scale)")
    ax.text(
        0.04, 0.96,
        f"$\\rho$ = {rho:+.3f} (p = {pval:.3f}, n = {len(df)})\n"
        f"partial $R^2$ net of city size = {partial_r2_imd:.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="#404040",
        bbox={"facecolor": "white", "edgecolor": "none",
              "alpha": 0.85, "pad": 3},
    )
    # Annotate top counter cities
    top = df.nlargest(5, "eco_avg_daily_bike_counts")
    for _, row in top.iterrows():
        ax.annotate(
            row["city"],
            (row["IMD"], row["eco_avg_daily_bike_counts"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7.5, color="#404040",
        )
    ax.grid(True, axis="both", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e6_imd_vs_counter.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e6_imd_vs_counter.pdf")


if __name__ == "__main__":
    main()
