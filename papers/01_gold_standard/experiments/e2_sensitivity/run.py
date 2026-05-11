"""E2 partial sweep: sigma_max sensitivity on the certified set.

Implements the single-dimension version of experiment E2 of the
Gold Standard validation roadmap. Sweeps the topological filter
threshold sigma_max over a 5-point grid and reports, for each
value:

  - the resulting certified-set size (per station_type)
  - the Jaccard similarity to the reference run (sigma_max = 3.0)
  - Kendall's tau of the IMD ranking against the reference
  - the Top-10 churn (number of cities entering/leaving the Top-10)

The four-dimensional grid of the full E2 (sigma_max x N_min x
BAAC buffer x BD TOPO buffer) would require re-running the whole
enrichment pipeline and is left to follow-up work.

Run from the repository root:

    python papers/01_gold_standard/experiments/e2_sensitivity/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir, repo_root  # noqa: E402

ROOT: Final[Path] = repo_root(__file__)
OUT_DIR: Final[Path] = outputs_dir(__file__)
sys.path.insert(0, str(ROOT))

from utils.data_loader import compute_imd_cities, load_stations  # noqa: E402

SIGMA_GRID: Final[tuple[float, ...]] = (2.0, 2.5, 3.0, 3.5, 4.0)
REFERENCE: Final[float] = 3.0


def _per_station_distance(df: pd.DataFrame) -> pd.Series:
    """Distance of each station from its system centroid, in degrees.

    Returns a Series aligned with df.index.
    """
    centroids = df.groupby("system_id")[["lat", "lon"]].transform("mean")
    d = np.sqrt(
        (df["lat"] - centroids["lat"]) ** 2
        + (df["lon"] - centroids["lon"]) ** 2
    )
    return d


def _system_sigma(df: pd.DataFrame, distances: pd.Series) -> pd.Series:
    """Per-system spatial standard deviation of distances."""
    sigma_per_sys = distances.groupby(df["system_id"]).transform("std")
    return sigma_per_sys.fillna(0)


def _apply_sigma_filter(df: pd.DataFrame, sigma_max: float) -> pd.DataFrame:
    """Apply the S5 topological filter at threshold sigma_max.

    The certified Gold Standard was already filtered at sigma_max = 3.
    For sigma_max < 3, this further restricts the set. For sigma_max > 3,
    no station can be added back (they were already filtered upstream),
    so the metric measures the *downward* sensitivity only.
    """
    if sigma_max >= REFERENCE:
        return df  # The certified set IS already at sigma_max = 3.

    distances = _per_station_distance(df)
    sigma = _system_sigma(df, distances)
    keep = distances <= sigma_max * sigma
    # Keep rows where sigma is 0 (single-station systems) to avoid divisions.
    keep = keep | (sigma == 0)
    return df[keep].copy()


def _jaccard(a: pd.Series, b: pd.Series) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _kendall_tau(rank_a: pd.Series, rank_b: pd.Series) -> float:
    """Kendall's tau on two ranking Series indexed by city name."""
    common = list(set(rank_a.index) & set(rank_b.index))
    if len(common) < 2:
        return float("nan")
    a = rank_a.reindex(common).values
    b = rank_b.reindex(common).values
    n = len(common)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (a[i] - a[j]) * (b[i] - b[j])
            if d > 0:
                concordant += 1
            elif d < 0:
                discordant += 1
    total = n * (n - 1) / 2
    if total == 0:
        return float("nan")
    return (concordant - discordant) / total


def _imd_ranking(df: pd.DataFrame) -> pd.Series:
    """Compute the IMD ranking for a given station set; returns city -> rank."""
    cities = compute_imd_cities(df)
    cities = cities.dropna(subset=["IMD"]).sort_values("IMD", ascending=False)
    return pd.Series(
        np.arange(1, len(cities) + 1),
        index=cities["city"].values,
        name="rank",
    )


def main() -> None:
    print("Loading certified Gold Standard corpus...")
    df = load_stations()
    print(f"  {len(df):,} stations, "
          f"{df['system_id'].nunique()} systems")

    # Reference run: certified Gold Standard at sigma_max = 3.
    ref_uid = df["uid"].copy() if "uid" in df.columns else df.index.to_series()
    ref_rank = _imd_ranking(df)
    print(f"  reference: {len(ref_rank)} ranked cities at sigma_max = 3.0")
    print()

    results = []
    print(f"  {'sigma_max':>10}{'kept':>10}{'jaccard':>10}"
          f"{'kendall':>10}{'top10churn':>12}")
    for sigma in SIGMA_GRID:
        sub = _apply_sigma_filter(df, sigma)
        sub_uid = sub["uid"].copy() if "uid" in sub.columns else sub.index.to_series()
        sub_rank = _imd_ranking(sub)
        jaccard = _jaccard(ref_uid, sub_uid)
        tau = _kendall_tau(ref_rank, sub_rank)
        ref_top10 = set(ref_rank[ref_rank <= 10].index)
        sub_top10 = set(sub_rank[sub_rank <= 10].index)
        top10_churn = len(ref_top10.symmetric_difference(sub_top10)) // 2
        results.append({
            "sigma_max": sigma,
            "kept_stations": int(len(sub)),
            "jaccard_vs_ref": round(jaccard, 4),
            "kendall_tau_vs_ref": round(tau, 4) if not np.isnan(tau) else None,
            "top10_churn": top10_churn,
        })
        print(f"  {sigma:>10.1f}{len(sub):>10,}{jaccard:>10.4f}"
              f"{tau:>10.4f}{top10_churn:>12}")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(
        json.dumps({"reference_sigma_max": REFERENCE,
                    "grid": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
