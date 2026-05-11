"""Three independent FULL-TASK classifiers for experiment E1.

Complementary to ``auto_annotators.py`` (which uses three
orthogonal SIGNAL sources) this script implements three
independent full-task classifiers: each sees ALL the available
signals (name, capacity statistics, geospatial coordinates) but
applies them in a different decision-tree order. These three
classifiers are the algorithmic analogue of three human
annotators who all read the same data but weigh the rules
differently.

The agreement between them is interpretable: high pairwise Cohen
kappa indicates that the A1--A5 taxonomy is robust to the order
in which the rules are applied; disagreements localise the
ambiguity zones of the taxonomy.

Run from the repository root, after ``sample_stations.py``:

    python papers/01_gold_standard/experiments/e1_irr/full_task_classifiers.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir, repo_root  # noqa: E402

ROOT: Final[Path] = repo_root(__file__)
DIR: Final[Path] = outputs_dir(__file__)

CARSHARE_PATTERNS: Final[tuple[str, ...]] = (
    "citiz", "yego", "free2move", "communauto", "getaround", "drivy",
    "ouicar", "tier", "voi", "lime", "dott", "bird",
)
FLOATING_PATTERNS: Final[tuple[str, ...]] = (
    "pony", "cykleo", "donkey", "bolt",
)
LAT_MIN, LAT_MAX = 41.0, 52.0
LON_MIN, LON_MAX = -6.0, 10.0


def _is_carsharing_system(text: str) -> bool:
    t = str(text).lower()
    return any(p in t for p in CARSHARE_PATTERNS)


def _is_floating_system(text: str) -> bool:
    t = str(text).lower()
    return any(p in t for p in FLOATING_PATTERNS)


def _system_stats(sample: pd.DataFrame) -> pd.DataFrame:
    """Per-system aggregates used by the three classifiers."""
    g = sample.groupby("system_id").agg(
        cap_mean=("capacity", "mean"),
        cap_std=("capacity", "std"),
        cap_median=("capacity", "median"),
        n=("station_id", "size"),
        lat_min=("lat", "min"),
        lat_max=("lat", "max"),
        lon_min=("lon", "min"),
        lon_max=("lon", "max"),
        lat_mean=("lat", "mean"),
        lon_mean=("lon", "mean"),
        lat_std=("lat", "std"),
        lon_std=("lon", "std"),
    )
    g["spatial_extent_deg"] = np.sqrt(
        (g["lat_max"] - g["lat_min"]) ** 2
        + (g["lon_max"] - g["lon_min"]) ** 2
    )
    return g


# -------------------------------------------------------------------------
# Classifier 1 -- name-first decision tree
# -------------------------------------------------------------------------

def classifier_name_first(sample: pd.DataFrame, stats: pd.DataFrame) -> pd.Series:
    labels = pd.Series(["ok"] * len(sample), index=sample.index, dtype=object)
    for idx, row in sample.iterrows():
        sys_id = row["system_id"]
        s = stats.loc[sys_id] if sys_id in stats.index else None
        # 1. Name -> A1
        if _is_carsharing_system(f"{row.get('system_name', '')} {sys_id}"):
            labels.at[idx] = "A1"; continue
        # 2. Capacity zero variance -> A2
        if s is not None and s["n"] >= 3 and s["cap_std"] == 0 and s["cap_mean"] > 0:
            labels.at[idx] = "A2"; continue
        # 3. Floating name -> A3
        if _is_floating_system(f"{row.get('system_name', '')} {sys_id}"):
            labels.at[idx] = "A3"; continue
        # 4. Coordinates invalid -> A4
        lat, lon = row["lat"], row["lon"]
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            labels.at[idx] = "A4"; continue
        # 5. Macro-regional spatial extent -> A5
        if s is not None and s["spatial_extent_deg"] > 3.0:
            labels.at[idx] = "A5"; continue
    return labels


# -------------------------------------------------------------------------
# Classifier 2 -- statistical-first decision tree
# -------------------------------------------------------------------------

def classifier_stats_first(sample: pd.DataFrame, stats: pd.DataFrame) -> pd.Series:
    labels = pd.Series(["ok"] * len(sample), index=sample.index, dtype=object)
    for idx, row in sample.iterrows():
        sys_id = row["system_id"]
        s = stats.loc[sys_id] if sys_id in stats.index else None
        lat, lon = row["lat"], row["lon"]
        # 1. Out-of-perimeter coordinates -> A4
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            labels.at[idx] = "A4"; continue
        # 2. Placeholder capacity -> A2
        if s is not None and s["n"] >= 3 and s["cap_std"] == 0 and s["cap_mean"] > 0:
            labels.at[idx] = "A2"; continue
        # 3. Tiny capacity median -> A3 (free-floating signature)
        if s is not None and s["n"] >= 3 and s["cap_median"] <= 2:
            labels.at[idx] = "A3"; continue
        # 4. Carsharing name -> A1
        if _is_carsharing_system(f"{row.get('system_name', '')} {sys_id}"):
            labels.at[idx] = "A1"; continue
        # 5. Macro extent -> A5
        if s is not None and s["spatial_extent_deg"] > 3.0:
            labels.at[idx] = "A5"; continue
    return labels


# -------------------------------------------------------------------------
# Classifier 3 -- geo-first decision tree
# -------------------------------------------------------------------------

def classifier_geo_first(sample: pd.DataFrame, stats: pd.DataFrame) -> pd.Series:
    labels = pd.Series(["ok"] * len(sample), index=sample.index, dtype=object)
    for idx, row in sample.iterrows():
        sys_id = row["system_id"]
        s = stats.loc[sys_id] if sys_id in stats.index else None
        lat, lon = row["lat"], row["lon"]
        # 1. Macro spatial extent -> A5
        if s is not None and s["spatial_extent_deg"] > 3.0:
            labels.at[idx] = "A5"; continue
        # 2. Coordinates outside or transposed -> A4
        in_box = LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX
        in_swap = LON_MIN <= lat <= LON_MAX and LAT_MIN <= lon <= LAT_MAX
        if not in_box or in_swap:
            labels.at[idx] = "A4"; continue
        # 3. Carsharing name -> A1
        if _is_carsharing_system(f"{row.get('system_name', '')} {sys_id}"):
            labels.at[idx] = "A1"; continue
        # 4. Capacity zero variance -> A2
        if s is not None and s["n"] >= 3 and s["cap_std"] == 0 and s["cap_mean"] > 0:
            labels.at[idx] = "A2"; continue
        # 5. Floating name OR tiny capacity -> A3
        if (_is_floating_system(f"{row.get('system_name', '')} {sys_id}")
                or (s is not None and s["n"] >= 3 and s["cap_median"] <= 2)):
            labels.at[idx] = "A3"; continue
    return labels


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main() -> None:
    sample_path = DIR / "e1_sample_v1.csv"
    if not sample_path.exists():
        raise SystemExit(f"Run sample_stations.py first; {sample_path} missing")

    sample = pd.read_csv(sample_path, encoding="utf-8")
    full = pd.read_parquet(
        ROOT / "data" / "stations_gold_standard_final.parquet"
    )
    sample = sample.merge(
        full[["station_id", "system_id", "system_name"]].drop_duplicates(),
        on=["station_id", "system_id"], how="left",
    )

    print(f"Loaded {len(sample)} sample rows")
    stats = _system_stats(sample)
    print(f"  computed per-system statistics on {len(stats)} systems")

    l1 = classifier_name_first(sample, stats)
    l2 = classifier_stats_first(sample, stats)
    l3 = classifier_geo_first(sample, stats)

    pd.DataFrame({"row_id": sample["row_id"], "label": l1.values}).to_csv(
        DIR / "full_task_C1_name_first.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": sample["row_id"], "label": l2.values}).to_csv(
        DIR / "full_task_C2_stats_first.csv", index=False, encoding="utf-8")
    pd.DataFrame({"row_id": sample["row_id"], "label": l3.values}).to_csv(
        DIR / "full_task_C3_geo_first.csv", index=False, encoding="utf-8")

    for name, lab in [("C1 name-first", l1),
                      ("C2 stats-first", l2),
                      ("C3 geo-first", l3)]:
        counts = lab.value_counts().reindex(
            ["ok", "A1", "A2", "A3", "A4", "A5"], fill_value=0
        )
        print(f"  {name:<18}  " + "  ".join(f"{k}={v:>3}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
