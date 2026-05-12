"""E10 -- Per-city IMD bootstrap confidence intervals.

E4 reports panel-level rank stability under within-city station
bootstrap. This experiment extends the same procedure to the score
side: it produces a 95% percentile bootstrap CI on the IMD value of
each city in the panel. The two-way diagnostic (rank CI in E4, score
CI here) lets readers compare cities not on point estimates but on
overlapping vs separated intervals.

The bootstrap is paired and joint: at each replicate we resample
stations within every city, recompute the panel-wide Min-Max
normalisation, and aggregate to the city-level IMD with the published
weights. This jointly captures within-city sampling noise and the
finite-panel renormalisation noise.

Outputs:
    outputs/e10_results.json
    outputs/e10_imd_ci.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import COMPONENTS, ROOT, composite_score

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PUBLISHED_W = np.array([0.374, 0.372, 0.053, 0.201])
N_REPS = 2000


def _per_station(stations_df: pd.DataFrame) -> pd.DataFrame:
    dock = stations_df[stations_df["station_type"] == "docked_bike"].copy()
    keep = [
        "city",
        "gtfs_heavy_stops_300m",
        "infra_cyclable_pct",
        "baac_accidents_cyclistes",
        "topography_roughness_index",
    ]
    return dock[keep].copy()


def _aggregate_panel(
    per_station: pd.DataFrame,
    rng: np.random.Generator | None = None,
    bootstrap: bool = False,
) -> pd.DataFrame:
    """Aggregate to city level with optional within-city bootstrap."""
    rows = []
    for city, g in per_station.groupby("city"):
        if bootstrap and rng is not None and len(g) > 1:
            sampled = g.sample(n=len(g), replace=True, random_state=rng)
        else:
            sampled = g
        rows.append({
            "city": city,
            "n_stations": len(g),
            "gtfs_heavy_stops_300m": sampled["gtfs_heavy_stops_300m"].mean(),
            "infra_cyclable_pct": sampled["infra_cyclable_pct"].mean(),
            "baac_accidents_cyclistes":
                sampled["baac_accidents_cyclistes"].mean(),
            "topography_roughness_index":
                sampled["topography_roughness_index"].mean(),
        })
    df = pd.DataFrame(rows)
    df = df[df["n_stations"] >= 5].copy()
    for col in ["gtfs_heavy_stops_300m", "infra_cyclable_pct",
                "baac_accidents_cyclistes", "topography_roughness_index"]:
        s = df[col].fillna(df[col].median())
        lo, hi = s.min(), s.max()
        df[col + "_norm"] = (
            np.full(len(s), 0.5) if hi == lo else (s - lo) / (hi - lo)
        )
    df["M_multi"] = df["gtfs_heavy_stops_300m_norm"]
    df["I_infra"] = df["infra_cyclable_pct_norm"]
    df["S_securite"] = 1.0 - df["baac_accidents_cyclistes_norm"]
    df["T_topo"] = 1.0 - df["topography_roughness_index_norm"]
    return df


def main() -> None:
    log.info("Loading stations table...")
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    stations = load_stations()
    per_station = _per_station(stations)
    log.info("  %d stations / %d cities", len(per_station),
             per_station["city"].nunique())

    # Reference
    ref = _aggregate_panel(per_station, bootstrap=False)
    ref["IMD"] = composite_score(PUBLISHED_W, ref[list(COMPONENTS)].to_numpy())
    ref = ref.sort_values("IMD", ascending=False).reset_index(drop=True)
    ref["rank"] = ref.index + 1
    log.info("Reference Top-10: %s", ref.head(10)["city"].tolist())

    cities = ref["city"].tolist()
    n_cities = len(cities)
    score_matrix = np.full((N_REPS, n_cities), np.nan)
    rng = np.random.default_rng(2026)
    for rep in range(N_REPS):
        boot = _aggregate_panel(per_station, rng=rng, bootstrap=True)
        boot_imd = composite_score(
            PUBLISHED_W, boot[list(COMPONENTS)].to_numpy(),
        )
        lookup = dict(zip(boot["city"], boot_imd))
        for idx, c in enumerate(cities):
            score_matrix[rep, idx] = lookup.get(c, np.nan)
    log.info("Bootstrap complete (%d replicates)", N_REPS)

    mean_imd = np.nanmean(score_matrix, axis=0)
    q025 = np.nanpercentile(score_matrix, 2.5, axis=0)
    q975 = np.nanpercentile(score_matrix, 97.5, axis=0)
    sd_imd = np.nanstd(score_matrix, axis=0)
    width = q975 - q025

    # CI-based "indistinguishability" graph: pairs of cities whose
    # 95% CIs overlap. We count for each city how many higher-ranked
    # cities have overlapping CIs.
    overlap_count = np.zeros(n_cities, dtype=int)
    for i in range(n_cities):
        for j in range(i):
            if not (q025[i] > q975[j] or q975[i] < q025[j]):
                overlap_count[i] += 1

    results = {
        "n_replicates": int(N_REPS),
        "median_ci_width": float(np.median(width)),
        "p25_ci_width": float(np.percentile(width, 25)),
        "p75_ci_width": float(np.percentile(width, 75)),
        "median_sd_imd": float(np.median(sd_imd)),
        "per_city": [
            {
                "city": cities[i],
                "rank_ref": int(ref["rank"].iloc[i]),
                "imd_ref": float(ref["IMD"].iloc[i]),
                "imd_mean_boot": float(mean_imd[i]),
                "imd_q025": float(q025[i]),
                "imd_q975": float(q975[i]),
                "imd_sd_boot": float(sd_imd[i]),
                "ci_width": float(width[i]),
                "n_higher_ranked_overlapping": int(overlap_count[i]),
            } for i in range(n_cities)
        ],
        "top10_max_ci_width": float(width[:10].max()),
        "top10_min_ci_width": float(width[:10].min()),
    }
    out_json = OUT_DIR / "e10_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)
    log.info("Median 95%% CI width over panel = %.2f IMD points",
             np.median(width))

    # Figure: top-20 IMD ranking with 95% CIs
    top_n = 20
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    y_pos = np.arange(top_n)
    err = np.vstack([
        ref["IMD"].iloc[:top_n].to_numpy() - q025[:top_n],
        q975[:top_n] - ref["IMD"].iloc[:top_n].to_numpy(),
    ])
    ax.errorbar(
        ref["IMD"].iloc[:top_n].to_numpy(),
        y_pos,
        xerr=err,
        fmt="o", color="#1F3A6B", ecolor="#404040",
        capsize=3, markersize=5,
        elinewidth=0.9, capthick=0.9,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ref["city"].iloc[:top_n], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("IMD score")
    ax.set_title(
        f"Top-{top_n} IMD with bootstrap 95% CI (N={N_REPS})",
        fontsize=10,
    )
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    median_w = float(np.median(width[:top_n]))
    ax.text(
        0.98, 0.02,
        f"median CI width = {median_w:.1f} pts",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#404040",
        bbox={"facecolor": "white", "edgecolor": "none",
              "alpha": 0.85, "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e10_imd_ci.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e10_imd_ci.pdf")


if __name__ == "__main__":
    main()
