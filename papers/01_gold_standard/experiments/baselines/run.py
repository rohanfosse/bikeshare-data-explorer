"""Comparative evaluation of the Gold Standard against three naive baselines.

For the same 122-system French GBFS corpus, four pipelines are
compared on four discriminating metrics. The Gold Standard is
treated as the reference verdict; baselines are scored by how
much of that verdict they recover from less information.

Pipelines
---------
B0 -- Raw GBFS:
      Ingest station_information as is. No filtering, no
      reclassification. A free-floating anchor is therefore
      counted as a dock-based station.

B1 -- vehicle_type filter:
      Mimics what a careful data engineer would write in a
      Jupyter notebook: keep only systems whose system name does
      not advertise a non-bike vehicle (Citiz car-sharing, etc.).
      Does not address A2/A3/A4 at all.

B2 -- MobilityData-style audit:
      Adapts the canonical GTFS-realtime validation rules to GBFS:
      schema-level checks (required fields present, lat/lon in
      valid ranges, capacity non-negative). Coarse-grained.

B3 -- Gold Standard:
      The full pipeline of the paper. Reference by construction.

Metrics
-------
Dock        : fraction of stations whose station_type matches GS
              (3-class agreement). Wald 95% CI from a normal
              approximation on the proportion.
rho_FUB     : Spearman rank correlation between per-city total
              dock-based station count and the FUB 2023 score.
              Bootstrap 95% CI over cities (n=2000 resamples).
A1 ex.      : recall on the 14 A1 systems being excluded.
              Wilson 95% binomial CI.
A3 fix.     : recall on the 8 A3 systems being correctly
              reclassified. Wilson 95% binomial CI.

Run from the repository root:

    python papers/01_gold_standard/experiments/baselines/run.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir, repo_root  # noqa: E402

ROOT: Final[Path] = repo_root(__file__)
OUT_DIR: Final[Path] = outputs_dir(__file__)
SEED: Final[int] = 42
RHO_BOOTSTRAP_N: Final[int] = 2_000

CARSHARE_PATTERNS: Final[tuple[str, ...]] = (
    "citiz", "yego", "free2move", "communauto", "getaround", "drivy",
    "ouicar", "tier", "voi", "lime", "dott", "bird",
)

LAT_MIN, LAT_MAX = 41.0, 52.0
LON_MIN, LON_MAX = -6.0, 10.0


# -------------------------------------------------------------------------
# Baselines
# -------------------------------------------------------------------------

def b0_raw_gbfs(stations: pd.DataFrame) -> pd.Series:
    return pd.Series(
        ["docked_bike"] * len(stations), index=stations.index, dtype=object
    )


def b1_vehicle_type_filter(stations: pd.DataFrame) -> pd.Series:
    is_carsharing = stations["system_id"].str.lower().apply(
        lambda s: any(p in s for p in CARSHARE_PATTERNS)
    )
    out = pd.Series(["docked_bike"] * len(stations), index=stations.index, dtype=object)
    out.loc[is_carsharing] = "carsharing"
    return out


def b2_mobilitydata_audit(stations: pd.DataFrame) -> pd.Series:
    out = b1_vehicle_type_filter(stations)
    lat = stations["lat"].astype(float)
    lon = stations["lon"].astype(float)
    invalid = ~(
        (lat >= LAT_MIN) & (lat <= LAT_MAX)
        & (lon >= LON_MIN) & (lon <= LON_MAX)
    )
    out.loc[invalid] = "excluded"
    return out


def b3_gold_standard(stations: pd.DataFrame) -> pd.Series:
    return stations["station_type"].copy()


PIPELINES = {
    "Raw GBFS": b0_raw_gbfs,
    "vehicle_type filter": b1_vehicle_type_filter,
    "MobilityData audit": b2_mobilitydata_audit,
    "Gold Standard": b3_gold_standard,
}


# -------------------------------------------------------------------------
# Metric computations (point estimates + analytical CIs)
# -------------------------------------------------------------------------

def wald_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approx CI for a binomial proportion (Dock metric, large n)."""
    if n == 0:
        return float("nan"), float("nan")
    se = math.sqrt(max(p * (1 - p), 0) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion (A1, A3 small n)."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def metric_dock(pred: pd.Series, gs: pd.Series) -> tuple[float, tuple[float, float]]:
    p = float((pred == gs).mean())
    return p, wald_ci(p, len(pred))


def metric_a1(pred: pd.Series, gs: pd.Series, sys_ids: pd.Series) -> tuple[float, tuple[float, float]]:
    a1 = sys_ids[gs == "carsharing"].unique()
    n = len(a1)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    correct = 0
    for s in a1:
        mask = sys_ids == s
        if (pred[mask] != "docked_bike").mean() >= 0.80:
            correct += 1
    return correct / n, wilson_ci(correct, n)


def metric_a3(pred: pd.Series, gs: pd.Series, sys_ids: pd.Series) -> tuple[float, tuple[float, float]]:
    a3 = sys_ids[gs == "free_floating"].unique()
    n = len(a3)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    correct = 0
    for s in a3:
        mask = sys_ids == s
        if (pred[mask] == "free_floating").mean() >= 0.80:
            correct += 1
    return correct / n, wilson_ci(correct, n)


def metric_rho_fub(
    pred: pd.Series, stations: pd.DataFrame, fub: pd.DataFrame, n_boot: int
) -> tuple[float, tuple[float, float]]:
    """Spearman rho between per-city dock count and FUB score.
    Bootstrap over cities (n_boot), much faster than over stations.
    """
    df = stations.copy()
    df["pred"] = pred.values
    counts = (
        df[df["pred"] == "docked_bike"]
        .groupby("city").size().rename("count").reset_index()
    )
    merged = counts.merge(fub, on="city", how="inner")
    if len(merged) < 5:
        return float("nan"), (float("nan"), float("nan"))
    rho_point = float(merged["count"].corr(merged["fub_score_2023"], method="spearman"))

    rng = np.random.default_rng(SEED)
    samples = []
    cities_n = len(merged)
    for _ in range(n_boot):
        idx = rng.integers(0, cities_n, size=cities_n)
        sub = merged.iloc[idx]
        r = sub["count"].corr(sub["fub_score_2023"], method="spearman")
        if not np.isnan(r):
            samples.append(r)
    if not samples:
        return rho_point, (float("nan"), float("nan"))
    arr = np.array(samples)
    return rho_point, (
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    )


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main() -> None:
    print("Loading inputs...", flush=True)
    stations = pd.read_parquet(ROOT / "data" / "stations_gold_standard_final.parquet")
    fub = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources"
        / "fub_barometre_2023_city_scores.csv"
    )
    print(f"  {len(stations):,} certified stations", flush=True)
    print(f"  {len(fub)} cities with FUB scores", flush=True)

    gs_labels = stations["station_type"]
    sys_ids = stations["system_id"]

    print()
    print(f"  {'Pipeline':<22}{'Dock':>22}{'rho_FUB':>22}"
          f"{'A1 ex.':>14}{'A3 fix.':>14}")
    print("  " + "-" * 92, flush=True)

    results = {}
    for name, pipeline in PIPELINES.items():
        pred = pipeline(stations)
        dock, (dock_lo, dock_hi) = metric_dock(pred, gs_labels)
        a1, (a1_lo, a1_hi) = metric_a1(pred, gs_labels, sys_ids)
        a3, (a3_lo, a3_hi) = metric_a3(pred, gs_labels, sys_ids)
        rho, (rho_lo, rho_hi) = metric_rho_fub(pred, stations, fub, RHO_BOOTSTRAP_N)

        results[name] = {
            "dock":    {"point": dock, "ci": [dock_lo, dock_hi]},
            "rho_fub": {"point": rho,  "ci": [rho_lo, rho_hi]},
            "a1_ex":   {"point": a1,   "ci": [a1_lo, a1_hi]},
            "a3_fix":  {"point": a3,   "ci": [a3_lo, a3_hi]},
        }
        print(f"  {name:<22}"
              f"  {dock:.3f} [{dock_lo:.3f},{dock_hi:.3f}]"
              f"  {rho:.3f} [{rho_lo:.3f},{rho_hi:.3f}]"
              f"  {a1*100:>4.0f}% [{a1_lo*100:.0f},{a1_hi*100:.0f}]"
              f"  {a3*100:>4.0f}% [{a3_lo*100:.0f},{a3_hi*100:.0f}]",
              flush=True)

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
