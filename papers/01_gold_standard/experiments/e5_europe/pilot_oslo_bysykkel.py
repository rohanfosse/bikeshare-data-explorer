"""
E5 (European generalisation) — second-system pilot.

System under audit: Oslo Bysykkel (Norway), GBFS v2.3 feed served by
UrbanSharing. Auto-discovery:
    https://gbfs.urbansharing.com/oslobysykkel.no/gbfs.json

Goal: apply the same A1–A5 detection rules used on the French corpus
and on Bicing Barcelona, with country-appropriate perimeter only
(no other recalibration), and report whether the protocol holds.
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
ROOT_PATH     = TMP / "oslo_root.json"
STATIONS_PATH = TMP / "oslo_stations.json"
VTYPES_PATH   = TMP / "oslo_vtypes.json"

# Norway perimeter (broad, mainland Norway)
GEOFILTER_LAT   = (57.5, 71.5)
GEOFILTER_LON   = (4.0, 31.5)
SIGMA_MAX       = 3.0
N_MIN_DOCK      = 20


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    root     = load(ROOT_PATH)
    vtypes   = load(VTYPES_PATH)
    stations = load(STATIONS_PATH)

    feed_names = sorted(f["name"] for f in list(root["data"].values())[0]["feeds"])
    vtype_list = vtypes["data"].get("vehicle_types", [])
    vtype_forms = sorted({v.get("form_factor", "?") for v in vtype_list})
    a1_cars     = any("car" in f for f in vtype_forms)

    raw_stations = stations["data"]["stations"]
    n_raw        = len(raw_stations)
    df = pd.DataFrame(raw_stations)
    df["lat"]      = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]      = pd.to_numeric(df["lon"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    nonzero_caps = df.loc[df["capacity"].fillna(0) > 0, "capacity"]
    cap_unique   = nonzero_caps.nunique()
    cap_std      = float(pstdev(nonzero_caps)) if len(nonzero_caps) > 1 else 0.0
    a2_placeholder = (cap_unique == 1) and (len(nonzero_caps) > 0)

    mean_all    = float(df["capacity"].fillna(0).mean())
    mean_active = float(nonzero_caps.mean()) if len(nonzero_caps) else 0.0
    ratio       = mean_active / mean_all if mean_all > 0 else float("nan")
    a3_floating = ratio > 5.0

    in_perimeter = df["lat"].between(*GEOFILTER_LAT) & df["lon"].between(*GEOFILTER_LON)
    a4_out_perimeter = int((~in_perimeter).sum())
    n_in = int(in_perimeter.sum())

    df_in = df[in_perimeter].copy()
    mu_lat, mu_lon = df_in["lat"].mean(), df_in["lon"].mean()
    sigma_lat = df_in["lat"].std()
    sigma_lon = df_in["lon"].std()
    df_in["d_sigma"] = (
        ((df_in["lat"] - mu_lat) / sigma_lat) ** 2 +
        ((df_in["lon"] - mu_lon) / sigma_lon) ** 2
    ) ** 0.5
    a4_outliers = int((df_in["d_sigma"] > SIGMA_MAX).sum())

    bbox_km2 = (
        (df_in["lat"].max() - df_in["lat"].min()) * 111.0 *
        (df_in["lon"].max() - df_in["lon"].min()) * 111.0
    )
    a5_macro_region = bbox_km2 > 50_000.0

    n_certified_dock = n_in - a4_outliers
    verdict = {
        "system": "Oslo Bysykkel",
        "country": "NO",
        "feed_url": "https://gbfs.urbansharing.com/oslobysykkel.no/gbfs.json",
        "gbfs_version_feeds": feed_names,
        "vehicle_form_factors": vtype_forms,
        "raw_station_count": n_raw,
        "capacity_unique_nonzero": int(cap_unique),
        "capacity_std_nonzero": round(cap_std, 3),
        "capacity_mean_all": round(mean_all, 3),
        "capacity_mean_active": round(mean_active, 3),
        "capacity_profile_ratio": round(ratio, 3),
        "anomalies_detected": {
            "A1_carsharing":         bool(a1_cars),
            "A2_placeholder":        bool(a2_placeholder),
            "A3_overcapacity":       bool(a3_floating),
            "A4_geospatial_out":     int(a4_out_perimeter),
            "A4_geospatial_outlier": int(a4_outliers),
            "A5_macro_region":       bool(a5_macro_region),
        },
        "certified_dock_stations": int(n_certified_dock),
        "too_small_under_N_min":   bool(n_certified_dock < N_MIN_DOCK),
        "bbox_km2_proxy":          round(bbox_km2, 1),
        "thresholds_used": {
            "geofilter_lat": GEOFILTER_LAT,
            "geofilter_lon": GEOFILTER_LON,
            "sigma_max":     SIGMA_MAX,
            "N_min":         N_MIN_DOCK,
        },
    }
    (HERE / "oslo_audit_report.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return verdict


def main():
    if not STATIONS_PATH.exists():
        sys.exit(f"Missing input: {STATIONS_PATH}")
    print(json.dumps(audit(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
