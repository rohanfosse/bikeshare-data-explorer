"""E4 -- Component-bootstrap proxy for buffer-radius sensitivity.

The pre-registered E4 protocol calls for a full re-enrichment of
the GBFS panel at six buffer radii. That re-enrichment requires
re-querying OSM Overpass, BAAC and GTFS for ~46,000 stations and
is not available within this revision. As a proxy, we bootstrap
stations within each city: for each city we resample its stations
with replacement, recompute the mean component values, and assemble
a bootstrap distribution of IMD scores. The IMD-rank Kendall tau
between the original ranking and each bootstrap replicate
quantifies the sensitivity to within-city sampling variability,
which is what a buffer change would, in part, induce.

Outputs:
    outputs/e4_results.json
    outputs/e4_rank_stability.pdf
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
    composite_score,
    load_panel,
)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PUBLISHED_W = np.array([0.578, 0.184, 0.142, 0.096])  # M, I, S, T


def _per_station_features(stations_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the per-station inputs used by the city aggregator."""
    dock = stations_df[stations_df["station_type"] == "docked_bike"].copy()
    keep_cols = [
        "city", "uid",
        "gtfs_heavy_stops_300m",
        "infra_cyclable_pct",
        "baac_accidents_cyclistes",
        "topography_roughness_index",
    ]
    return dock[keep_cols].copy()


def _city_components_from_stations(
    per_station: pd.DataFrame,
    rng: np.random.Generator | None = None,
    bootstrap: bool = False,
) -> pd.DataFrame:
    """Aggregate station-level features to city-level components.

    When bootstrap is True, resamples stations within each city
    with replacement before averaging. The Min-Max normalisation
    is recomputed on the bootstrap panel for each replicate.
    """
    grouped = per_station.groupby("city")
    rows = []
    for city, g in grouped:
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
    city_df = pd.DataFrame(rows)
    city_df = city_df[city_df["n_stations"] >= 5].copy()

    for col in ["gtfs_heavy_stops_300m", "infra_cyclable_pct",
                "baac_accidents_cyclistes", "topography_roughness_index"]:
        s = city_df[col].fillna(city_df[col].median())
        lo, hi = s.min(), s.max()
        city_df[col + "_norm"] = (
            np.full(len(s), 0.5) if hi == lo
            else (s - lo) / (hi - lo)
        )

    city_df["M_multi"] = city_df["gtfs_heavy_stops_300m_norm"]
    city_df["I_infra"] = city_df["infra_cyclable_pct_norm"]
    city_df["S_securite"] = 1.0 - city_df["baac_accidents_cyclistes_norm"]
    city_df["T_topo"] = 1.0 - city_df["topography_roughness_index_norm"]
    return city_df


def main() -> None:
    log.info("Loading panel and stations table...")
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    stations = load_stations()
    per_station = _per_station_features(stations)
    log.info("  %d dock-based stations across %d cities",
             len(per_station), per_station["city"].nunique())

    # Reference (point-estimate) ranking
    ref_city = _city_components_from_stations(per_station, bootstrap=False)
    ref_components = ref_city[list(COMPONENTS)].to_numpy()
    ref_score = composite_score(PUBLISHED_W, ref_components)
    ref_city = ref_city.assign(IMD=ref_score)
    ref_ranked = (
        ref_city.sort_values("IMD", ascending=False).reset_index(drop=True)
        .assign(rank=lambda d: d.index + 1)
    )
    log.info("Reference Top-10: %s",
             ref_ranked.head(10)["city"].tolist())

    # Bootstrap replicates
    rng = np.random.default_rng(42)
    n_reps = 1_000
    rank_matrix = np.full((n_reps, len(ref_ranked)), np.nan)
    score_matrix = np.full((n_reps, len(ref_ranked)), np.nan)
    cities_index = {c: i for i, c in enumerate(ref_ranked["city"])}
    kendall_taus: list[float] = []
    top10_membership: dict[str, int] = {c: 0 for c in ref_ranked["city"]}

    for rep in range(n_reps):
        boot_city = _city_components_from_stations(
            per_station, rng=rng, bootstrap=True,
        )
        boot_components = boot_city[list(COMPONENTS)].to_numpy()
        boot_score = composite_score(PUBLISHED_W, boot_components)
        boot_df = boot_city.assign(IMD=boot_score)
        boot_ranked = (
            boot_df.sort_values("IMD", ascending=False).reset_index(drop=True)
            .assign(rank=lambda d: d.index + 1)
        )
        # Align with reference
        rank_lookup = dict(zip(boot_ranked["city"], boot_ranked["rank"]))
        score_lookup = dict(zip(boot_ranked["city"], boot_ranked["IMD"]))
        for c, idx in cities_index.items():
            rank_matrix[rep, idx] = rank_lookup.get(c, np.nan)
            score_matrix[rep, idx] = score_lookup.get(c, np.nan)
        for c in boot_ranked.head(10)["city"]:
            if c in top10_membership:
                top10_membership[c] += 1
        ref_ranks_for_tau = ref_ranked["rank"].to_numpy()
        boot_ranks_for_tau = np.array([
            rank_lookup.get(c, np.nan) for c in ref_ranked["city"]
        ])
        valid = np.isfinite(boot_ranks_for_tau)
        if valid.sum() >= 5:
            tau, _ = sp_stats.kendalltau(
                ref_ranks_for_tau[valid], boot_ranks_for_tau[valid],
            )
            kendall_taus.append(float(tau))

    kendall_taus_arr = np.array(kendall_taus)
    log.info("Kendall tau across %d bootstrap replicates:",
             len(kendall_taus_arr))
    log.info("  median = %.3f, p25 = %.3f, p975 = %.3f",
             float(np.median(kendall_taus_arr)),
             float(np.percentile(kendall_taus_arr, 2.5)),
             float(np.percentile(kendall_taus_arr, 97.5)))

    # Top-10 stability per published city
    ref_top10 = ref_ranked.head(10)["city"].tolist()
    log.info("Top-10 membership probability under station bootstrap:")
    for c in ref_top10:
        log.info("  %-20s : %5.1f %%", c, 100 * top10_membership[c] / n_reps)

    rank_sd = np.nanstd(rank_matrix, axis=0)
    rank_mean = np.nanmean(rank_matrix, axis=0)
    rank_stability = pd.DataFrame({
        "city": ref_ranked["city"],
        "rank_ref": ref_ranked["rank"],
        "rank_mean_boot": rank_mean,
        "rank_sd_boot": rank_sd,
    }).sort_values("rank_ref")

    results = {
        "n_replicates": n_reps,
        "median_kendall_tau": float(np.median(kendall_taus_arr)),
        "ci95_kendall_tau": [
            float(np.percentile(kendall_taus_arr, 2.5)),
            float(np.percentile(kendall_taus_arr, 97.5)),
        ],
        "top10_membership_pct": {
            c: round(100 * top10_membership[c] / n_reps, 2)
            for c in ref_top10
        },
        "rank_sd_by_city": rank_stability.head(15).to_dict("records"),
    }
    out_json = OUT_DIR / "e4_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Plot: rank SD bar chart for the Top-15 cities
    top15 = rank_stability.head(15)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.barh(top15["city"][::-1], top15["rank_sd_boot"][::-1],
            color="#1F3A6B", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Bootstrap rank standard deviation (positions)")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.text(0.98, 0.02,
            f"n = {n_reps} bootstrap replicates\n"
            f"median Kendall $\\tau$ = "
            f"{float(np.median(kendall_taus_arr)):.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#404040",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.85, "pad": 3})
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e4_rank_stability.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e4_rank_stability.pdf")


if __name__ == "__main__":
    main()
