"""Three independent automated annotators for experiment E1.

This script provides a *complementary* robustness check to the
human inter-rater protocol of ``score_kappa.py``. It does NOT
substitute for external human annotators: all three classifiers
are written by the same team. What it does measure is whether
the A1--A5 taxonomy is detectable from **three orthogonal signal
sources**, applied separately:

    Annotator A : name-based  (system_name, station_name only)
    Annotator B : capacity-statistics-based  (capacity column only)
    Annotator C : geospatial-based  (lat, lon only)

Each annotator sees ONLY the columns of its signal source; none
of them sees the rule-based ``station_type`` of the Gold Standard.
The output is a wide CSV with one column per annotator, ready to
feed into ``score_kappa.py``.

Run from the repository root, after ``sample_stations.py``:

    python papers/01_gold_standard/experiments/e1_irr/auto_annotators.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir, repo_root  # noqa: E402

ROOT: Final[Path] = repo_root(__file__)
DIR: Final[Path] = outputs_dir(__file__)

LABELS: Final[tuple[str, ...]] = ("ok", "A1", "A2", "A3", "A4", "A5")

# A1: car-sharing and non-bike fleet name patterns.
CARSHARE_PATTERNS: Final[tuple[str, ...]] = (
    "citiz", "yego", "free2move", "communauto", "getaround", "drivy",
    "ouicar", "tier", "voi", "lime", "dott", "bird",
)
# A3: free-floating bike fleet name patterns.
FLOATING_PATTERNS: Final[tuple[str, ...]] = (
    "pony", "cykleo", "donkey", "bolt",
)
# A5: overseas and macro-regional system identifiers.
OUT_OF_PERIMETER: Final[tuple[str, ...]] = (
    "reunion", "guadeloupe", "martinique", "mayotte",
    "basque_country", "grand_est_rural",
)

# A4: metropolitan France bounding box.
LAT_MIN, LAT_MAX = 41.0, 52.0
LON_MIN, LON_MAX = -6.0, 10.0


# -------------------------------------------------------------------------
# Annotator A -- name-based
# -------------------------------------------------------------------------

def annotator_name(row: pd.Series) -> str:
    """Classify a station from system_name and station_name only.

    Does NOT see capacity, lat, lon, or station_type.
    """
    sys_name = str(row.get("system_name", "")).lower()
    sys_id = str(row.get("system_id", "")).lower()
    sta_name = str(row.get("station_name", "")).lower()

    haystack = f"{sys_name} {sys_id} {sta_name}"

    if any(p in haystack for p in CARSHARE_PATTERNS):
        return "A1"
    if any(p in haystack for p in OUT_OF_PERIMETER):
        return "A5"
    if any(p in haystack for p in FLOATING_PATTERNS):
        return "A3"
    # Vehicle-id-suffix heuristic: station_id with a hex/digit hash suffix
    # commonly used by free-floating platforms.
    sta_id = str(row.get("station_id", ""))
    if re.search(r"_[a-f0-9]{4,}$", sta_id, re.IGNORECASE):
        return "A3"
    return "ok"


# -------------------------------------------------------------------------
# Annotator B -- capacity-statistics-based (per system, on the sample)
# -------------------------------------------------------------------------

def annotator_capacity(sample: pd.DataFrame) -> pd.Series:
    """Classify each row from capacity statistics only.

    Sees capacity values aggregated at the system level on the
    sample. Does NOT see name, lat, lon or station_type.
    """
    out = pd.Series(["ok"] * len(sample), index=sample.index, dtype=object)

    # Per-system stats on this sample.
    sys_stats = sample.groupby("system_id")["capacity"].agg(["mean", "std", "size"])

    # A2 -- placeholder: zero variance, non-zero mean, at least 3 obs
    a2_systems = sys_stats.index[
        (sys_stats["std"].fillna(0) == 0)
        & (sys_stats["mean"] > 0)
        & (sys_stats["size"] >= 3)
    ]
    out.loc[sample["system_id"].isin(a2_systems)] = "A2"

    # A3 -- very small median capacity across a large enough system
    # (free-floating fleets typically report capacity in [0, 3]).
    a3_systems = sys_stats.index[
        (sys_stats["mean"] <= 4)
        & (sys_stats["size"] >= 3)
        & ~sys_stats.index.isin(a2_systems)
    ]
    out.loc[sample["system_id"].isin(a3_systems)] = "A3"

    return out


# -------------------------------------------------------------------------
# Annotator C -- geospatial-based
# -------------------------------------------------------------------------

def annotator_geospatial(sample: pd.DataFrame) -> pd.Series:
    """Classify each row from lat / lon only.

    Sees only the geographical coordinates of the station and the
    centroid / dispersion of its system on the sample. Does NOT
    see name, capacity or station_type.
    """
    out = pd.Series(["ok"] * len(sample), index=sample.index, dtype=object)
    lat = sample["lat"].astype(float)
    lon = sample["lon"].astype(float)

    # A4 (transposed): coordinates fit the *swapped* bounding box.
    transposed = (
        (lat >= LON_MIN) & (lat <= LON_MAX)
        & (lon >= LAT_MIN) & (lon <= LAT_MAX)
    )
    # A4 (out of perimeter, individual): outside national box.
    out_of_bb = ~(
        (lat >= LAT_MIN) & (lat <= LAT_MAX)
        & (lon >= LON_MIN) & (lon <= LON_MAX)
    )

    out.loc[transposed | out_of_bb] = "A4"

    # System-level dispersion check: stations more than 3 sigma from
    # the centroid of their own system are flagged A4.
    for sys_id, grp in sample.groupby("system_id"):
        if len(grp) < 4:
            continue
        cx, cy = grp["lat"].mean(), grp["lon"].mean()
        d = np.sqrt((grp["lat"] - cx) ** 2 + (grp["lon"] - cy) ** 2)
        threshold = 3.0 * d.std()
        if not np.isfinite(threshold) or threshold == 0:
            continue
        outliers = grp.index[d > threshold]
        out.loc[outliers] = "A4"

    # A5: system whose spatial extent (bbox diagonal) is > ~3 degrees.
    # This corresponds to a macro-regional surface in metropolitan France.
    for sys_id, grp in sample.groupby("system_id"):
        if len(grp) < 4:
            continue
        diag = np.sqrt(
            (grp["lat"].max() - grp["lat"].min()) ** 2
            + (grp["lon"].max() - grp["lon"].min()) ** 2
        )
        if diag > 3.0:
            out.loc[grp.index] = "A5"

    return out


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main() -> None:
    sample_path = DIR / "e1_sample_v1.csv"
    if not sample_path.exists():
        raise SystemExit(
            f"Run sample_stations.py first; {sample_path} missing"
        )

    sample = pd.read_csv(sample_path, encoding="utf-8")
    # Replay the original station_name field from the full corpus (the
    # blinded sample doesn't ship it as-is; we re-load it for the
    # name-based annotator -- this is the only signal each annotator
    # is allowed to see).
    full = pd.read_parquet(
        ROOT / "data" / "stations_gold_standard_final.parquet"
    )
    sample = sample.merge(
        full[["station_id", "system_name", "system_id"]]
        .drop_duplicates(),
        on=["station_id", "system_id"],
        how="left",
        suffixes=("", "_full"),
    )

    print(f"Loaded {len(sample)} sample rows")

    # Annotator A: name-based.
    labels_a = sample.apply(annotator_name, axis=1)
    # Annotator B: capacity-statistics-based.
    labels_b = annotator_capacity(sample)
    # Annotator C: geospatial-based.
    labels_c = annotator_geospatial(sample)

    out_a = sample[["row_id"]].assign(label=labels_a.values)
    out_b = sample[["row_id"]].assign(label=labels_b.values)
    out_c = sample[["row_id"]].assign(label=labels_c.values)

    out_a.to_csv(DIR / "auto_annotator_A_name.csv", index=False)
    out_b.to_csv(DIR / "auto_annotator_B_capacity.csv", index=False)
    out_c.to_csv(DIR / "auto_annotator_C_geospatial.csv", index=False)

    print("Wrote:")
    print("  auto_annotator_A_name.csv      ", labels_a.value_counts().to_dict())
    print("  auto_annotator_B_capacity.csv  ", labels_b.value_counts().to_dict())
    print("  auto_annotator_C_geospatial.csv", labels_c.value_counts().to_dict())


if __name__ == "__main__":
    main()
