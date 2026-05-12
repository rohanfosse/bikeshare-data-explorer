"""E18 -- Station-level IMD and within-city variance decomposition.

The headline IMD aggregates four components to one number per city.
At the panel scale this is informative, but it discards two equally
important pieces of information: (i) the variance of IMD inside a
city (which can be as large as the variance between cities), and
(ii) the identification of within-city outliers -- the "bad stations
in good cities" and vice versa.

We compute a per-station IMD using the published supervised weights
and the station-level normalisation of each component, then decompose
the panel-wide variance using a one-way ANOVA into between-city and
within-city contributions:

    Var(IMD_s) = Var_between + Var_within
                = Σ_c n_c (mu_c - mu_grand)² / N
                  + Σ_c Σ_{s ∈ c} (IMD_s - mu_c)² / N

We also report the Theil index decomposition for robustness to
heavy-tailed distributions.

Outputs:
    outputs/e18_results.json
    outputs/e18_within_vs_between.pdf
    outputs/e18_top_anomalies.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import COMPONENTS, ROOT

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PUBLISHED_W = np.array([0.374, 0.372, 0.053, 0.201])


def _station_imd(stations: pd.DataFrame) -> pd.DataFrame:
    """Compute station-level IMD on the dock-based panel.

    Each of the four raw components is min-max normalised across
    the full set of dock-based stations (not aggregated per city),
    with the sign convention preserved for safety and topography.
    """
    dock = stations[stations["station_type"] == "docked_bike"].copy()

    def _normalise(col: str, invert: bool = False) -> np.ndarray:
        s = dock[col].astype(float).fillna(dock[col].median())
        lo, hi = s.min(), s.max()
        if hi == lo:
            norm = np.full(len(s), 0.5)
        else:
            norm = (s - lo) / (hi - lo)
        return 1.0 - norm if invert else norm

    dock["M_norm"] = _normalise("gtfs_heavy_stops_300m")
    dock["I_norm"] = _normalise("infra_cyclable_pct")
    dock["S_norm"] = _normalise("baac_accidents_cyclistes", invert=True)
    dock["T_norm"] = _normalise("topography_roughness_index", invert=True)
    components = dock[["M_norm", "I_norm", "S_norm", "T_norm"]].to_numpy()
    dock["IMD_station"] = (components @ PUBLISHED_W) * 100.0
    return dock


def _variance_decomposition(df: pd.DataFrame) -> dict:
    """One-way ANOVA on station IMDs grouped by city."""
    grouped = df.groupby("city")["IMD_station"]
    grand_mean = float(df["IMD_station"].mean())
    total_var = float(df["IMD_station"].var(ddof=0))
    total_n = len(df)

    between_ss = 0.0
    within_ss = 0.0
    for city, vals in grouped:
        n_c = len(vals)
        mu_c = vals.mean()
        between_ss += n_c * (mu_c - grand_mean) ** 2
        within_ss += float(((vals - mu_c) ** 2).sum())

    var_between = between_ss / total_n
    var_within = within_ss / total_n
    return {
        "n_stations": int(total_n),
        "n_cities": int(df["city"].nunique()),
        "grand_mean_imd": grand_mean,
        "total_variance": total_var,
        "var_between": float(var_between),
        "var_within": float(var_within),
        "share_between": float(var_between / total_var),
        "share_within": float(var_within / total_var),
        # Intra-class correlation coefficient (ICC1)
        "icc1": float(var_between / (var_between + var_within)),
    }


def _theil_decomposition(df: pd.DataFrame) -> dict:
    """Theil's T index decomposition (between/within cities).

    Theil-T is well-defined for non-negative outcomes; the IMD is in
    [0, 100] so we shift by a small epsilon to handle stations at 0.
    """
    y = df["IMD_station"].to_numpy(dtype=float) + 1e-9
    mu = y.mean()
    theil_total = float(np.mean((y / mu) * np.log(y / mu)))

    between = 0.0
    within = 0.0
    for city, sub in df.groupby("city"):
        y_c = sub["IMD_station"].to_numpy(dtype=float) + 1e-9
        n_c = len(y_c)
        mu_c = y_c.mean()
        share_n = n_c / len(y)
        share_y = y_c.sum() / y.sum()
        if mu_c > 0:
            between += share_y * np.log(mu_c / mu)
            within += share_y * float(np.mean((y_c / mu_c) * np.log(y_c / mu_c)))
    return {
        "theil_total": theil_total,
        "theil_between": float(between),
        "theil_within": float(within),
        "share_within_theil": float(within / theil_total) if theil_total > 0 else float("nan"),
        "share_between_theil": float(between / theil_total) if theil_total > 0 else float("nan"),
    }


def _per_city_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-city station-IMD summary stats."""
    g = df.groupby("city")["IMD_station"]
    out = pd.DataFrame({
        "n_stations": g.size(),
        "imd_mean": g.mean(),
        "imd_median": g.median(),
        "imd_sd": g.std(ddof=1),
        "imd_min": g.min(),
        "imd_max": g.max(),
        "imd_p10": g.quantile(0.10),
        "imd_p90": g.quantile(0.90),
        "imd_iqr": g.quantile(0.75) - g.quantile(0.25),
    }).reset_index()
    out["coef_var"] = out["imd_sd"] / out["imd_mean"]
    out["spread"] = out["imd_max"] - out["imd_min"]
    return out


def _detect_anomalies(df: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """Identify outlier stations within each city: z-score >= 2 from city mean."""
    df = df.merge(summary[["city", "imd_mean", "imd_sd"]], on="city")
    df["z_within_city"] = (df["IMD_station"] - df["imd_mean"]) / df["imd_sd"].replace(0, np.nan)
    high_low = df.dropna(subset=["z_within_city"])
    # "Bad stations in good cities": z <= -2 in a top-quartile city
    panel_median_mean = summary["imd_mean"].median()
    good_cities = set(summary[summary["imd_mean"] >= summary["imd_mean"].quantile(0.75)]["city"])
    bad_cities = set(summary[summary["imd_mean"] <= summary["imd_mean"].quantile(0.25)]["city"])

    bad_in_good = high_low[
        (high_low["z_within_city"] <= -2) & (high_low["city"].isin(good_cities))
    ].sort_values("z_within_city").head(15)
    good_in_bad = high_low[
        (high_low["z_within_city"] >= 2) & (high_low["city"].isin(bad_cities))
    ].sort_values("z_within_city", ascending=False).head(15)
    return {
        "n_bad_in_good": int(len(bad_in_good)),
        "n_good_in_bad": int(len(good_in_bad)),
        "panel_median_city_imd": float(panel_median_mean),
        "bad_in_good_examples": bad_in_good[[
            "city", "station_name", "IMD_station", "z_within_city",
        ]].to_dict("records"),
        "good_in_bad_examples": good_in_bad[[
            "city", "station_name", "IMD_station", "z_within_city",
        ]].to_dict("records"),
    }


def main() -> None:
    log.info("Loading stations...")
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    stations = load_stations()
    log.info("  total = %d", len(stations))

    df = _station_imd(stations)
    log.info("  dock-based stations: %d in %d cities",
             len(df), df["city"].nunique())

    log.info("Variance decomposition (ANOVA)...")
    anova = _variance_decomposition(df)
    log.info("  IMD station-level mean = %.2f", anova["grand_mean_imd"])
    log.info("  total variance = %.2f", anova["total_variance"])
    log.info("  between cities = %.2f (%.1f%%)",
             anova["var_between"], 100 * anova["share_between"])
    log.info("  within cities  = %.2f (%.1f%%)",
             anova["var_within"], 100 * anova["share_within"])
    log.info("  ICC1 = %.3f", anova["icc1"])

    log.info("Theil index decomposition...")
    theil = _theil_decomposition(df)
    log.info("  Theil total = %.4f", theil["theil_total"])
    log.info("  between = %.4f (%.1f%%)   within = %.4f (%.1f%%)",
             theil["theil_between"], 100 * theil["share_between_theil"],
             theil["theil_within"], 100 * theil["share_within_theil"])

    summary = _per_city_summary(df)
    log.info("Per-city spread (top-5 most heterogeneous cities):")
    top_spread = summary.nlargest(5, "spread")
    for _, row in top_spread.iterrows():
        log.info(
            "  %-20s  n = %3d   mean = %.1f   sd = %.1f   min = %.1f   max = %.1f",
            row["city"], int(row["n_stations"]),
            row["imd_mean"], row["imd_sd"], row["imd_min"], row["imd_max"],
        )

    log.info("Per-city spread (top-5 most homogeneous cities):")
    top_homog = summary[summary["n_stations"] >= 10].nsmallest(5, "imd_sd")
    for _, row in top_homog.iterrows():
        log.info(
            "  %-20s  n = %3d   mean = %.1f   sd = %.1f   min = %.1f   max = %.1f",
            row["city"], int(row["n_stations"]),
            row["imd_mean"], row["imd_sd"], row["imd_min"], row["imd_max"],
        )

    anomalies = _detect_anomalies(df, summary)
    log.info("Anomalies: %d bad stations in good cities, %d good stations in bad cities",
             anomalies["n_bad_in_good"], anomalies["n_good_in_bad"])

    results = {
        "anova_decomposition": anova,
        "theil_decomposition": theil,
        "per_city_summary": summary.to_dict("records"),
        "anomalies": anomalies,
    }
    out_json = OUT_DIR / "e18_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # ---- Figure 1: stacked bar of variance shares ----
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    labels = ["ANOVA\n(homoscedastic)", "Theil index\n(scale-free)"]
    between = [100 * anova["share_between"], 100 * theil["share_between_theil"]]
    within = [100 * anova["share_within"], 100 * theil["share_within_theil"]]
    x = np.arange(2)
    ax.bar(x, between, color="#1F3A6B", label="Between cities",
           edgecolor="white", linewidth=0.5)
    ax.bar(x, within, bottom=between, color="#7095C8",
           label="Within cities", edgecolor="white", linewidth=0.5)
    for i, (b, w) in enumerate(zip(between, within)):
        ax.text(i, b / 2, f"{b:.0f}%", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(i, b + w / 2, f"{w:.0f}%", ha="center", va="center",
                fontsize=10, color="#202020", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Share of IMD variance (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_title("Within-city vs.\\ between-city IMD variance "
                 f"(N = {anova['n_stations']:,} stations)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e18_within_vs_between.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e18_within_vs_between.pdf")

    # ---- Figure 2: per-city IMD distribution (top-15 by n_stations) ----
    top_cities = summary.nlargest(15, "n_stations")["city"].tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    box_data = [
        df[df["city"] == c]["IMD_station"].to_numpy()
        for c in top_cities
    ]
    bp = ax.boxplot(
        box_data, vert=False,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
        medianprops={"color": "#A8201A"},
        whiskerprops={"color": "#404040", "linewidth": 0.6},
        capprops={"color": "#404040", "linewidth": 0.6},
        boxprops={"facecolor": "#1F3A6B", "edgecolor": "white",
                  "linewidth": 0.4, "alpha": 0.85},
    )
    ax.set_yticklabels([
        f"{c} (n={int(summary[summary['city']==c]['n_stations'].iloc[0])})"
        for c in top_cities
    ], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(anova["grand_mean_imd"], color="#404040",
               linewidth=0.6, linestyle=":", alpha=0.7,
               label=f"panel mean = {anova['grand_mean_imd']:.1f}")
    ax.set_xlabel("Station-level IMD")
    ax.set_title("Within-city dispersion of station-level IMD"
                 " (15 largest networks)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e18_top_anomalies.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e18_top_anomalies.pdf")


if __name__ == "__main__":
    main()
