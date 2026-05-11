"""
E5 (European generalisation) --- third-system pilot.

System: Bergen Bysykkel (Norway), GBFS v2 feed by UrbanSharing.
Same protocol as Bicing and Oslo pilots.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any

import pandas as pd

HERE = Path(__file__).parent
TMP = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
ROOT_PATH     = TMP / "bergen_root.json"
STATIONS_PATH = TMP / "bergen_stations.json"
VTYPES_PATH   = TMP / "bergen_vtypes.json"

GEOFILTER_LAT = (57.5, 71.5)
GEOFILTER_LON = (4.0, 31.5)
SIGMA_MAX     = 3.0
N_MIN_DOCK    = 20


def audit() -> dict[str, Any]:
    root     = json.loads(ROOT_PATH.read_text(encoding="utf-8"))
    vtypes   = json.loads(VTYPES_PATH.read_text(encoding="utf-8"))
    stations = json.loads(STATIONS_PATH.read_text(encoding="utf-8"))

    feed_names = sorted(f["name"] for f in list(root["data"].values())[0]["feeds"])
    vtype_list = vtypes["data"].get("vehicle_types", [])
    vtype_forms = sorted({v.get("form_factor", "?") for v in vtype_list})
    a1_cars     = any("car" in f for f in vtype_forms)

    raw = stations["data"]["stations"]
    df = pd.DataFrame(raw)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    nonzero = df.loc[df["capacity"].fillna(0) > 0, "capacity"]
    cap_std = float(pstdev(nonzero)) if len(nonzero) > 1 else 0.0
    a2 = (nonzero.nunique() == 1) and (len(nonzero) > 0)

    mean_all    = float(df["capacity"].fillna(0).mean())
    mean_active = float(nonzero.mean()) if len(nonzero) else 0.0
    ratio = mean_active / mean_all if mean_all > 0 else float("nan")
    a3 = ratio > 5.0

    in_perim = df["lat"].between(*GEOFILTER_LAT) & df["lon"].between(*GEOFILTER_LON)
    df_in = df[in_perim].copy()
    mu_lat, mu_lon = df_in["lat"].mean(), df_in["lon"].mean()
    sl, sw = df_in["lat"].std(), df_in["lon"].std()
    df_in["d_sigma"] = (
        ((df_in["lat"] - mu_lat) / sl) ** 2 + ((df_in["lon"] - mu_lon) / sw) ** 2
    ) ** 0.5
    a4_outliers = int((df_in["d_sigma"] > SIGMA_MAX).sum())

    bbox_km2 = (df_in["lat"].max() - df_in["lat"].min()) * 111.0 * \
               (df_in["lon"].max() - df_in["lon"].min()) * 111.0

    verdict = {
        "system": "Bergen Bysykkel",
        "country": "NO",
        "feed_url": "https://gbfs.urbansharing.com/bergenbysykkel.no/gbfs.json",
        "gbfs_version_feeds": feed_names,
        "vehicle_form_factors": vtype_forms,
        "raw_station_count": int(len(df)),
        "capacity_unique_nonzero": int(nonzero.nunique()),
        "capacity_std_nonzero": round(cap_std, 3),
        "capacity_mean_all": round(mean_all, 3),
        "capacity_mean_active": round(mean_active, 3),
        "capacity_profile_ratio": round(ratio, 3),
        "anomalies_detected": {
            "A1_carsharing":         bool(a1_cars),
            "A2_placeholder":        bool(a2),
            "A3_overcapacity":       bool(a3),
            "A4_geospatial_out":     int((~in_perim).sum()),
            "A4_geospatial_outlier": a4_outliers,
            "A5_macro_region":       bool(bbox_km2 > 50_000.0),
        },
        "certified_dock_stations": int(in_perim.sum() - a4_outliers),
        "bbox_km2_proxy":          round(bbox_km2, 1),
    }
    (HERE / "bergen_audit_report.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return verdict


if __name__ == "__main__":
    if not STATIONS_PATH.exists():
        sys.exit(f"Missing input: {STATIONS_PATH}")
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
