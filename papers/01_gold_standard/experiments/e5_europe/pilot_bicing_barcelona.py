"""
E5 (European generalisation) — single-system pilot.

System under audit: Bicing Barcelona (Spain), GBFS v2 feed served by
PublicBikeSystem / Lyft. Auto-discovery:
    https://barcelona.publicbikesystem.net/customer/gbfs/v2/gbfs.json

Goal: apply the A1–A5 detection rules from the Gold Standard GBFS France
protocol to a non-French dock-based system, with no parameter
recalibration, and report whether the protocol holds.

Outputs:
    - bicing_audit_report.json   (per-class verdict + numbers)
    - bicing_stations.parquet    (typed audited stations)

Rule application is deliberately the same as for the French corpus so
that the result is a direct test of transferability, not a parameter-
tuning exercise.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import median, pstdev
from typing import Any

import pandas as pd

HERE = Path(__file__).parent

# ── Inputs (fetched via curl into the OS temp dir; see README) ────────
# Resolves to /tmp on POSIX and %TEMP% on Windows.
TMP = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
ROOT_PATH     = TMP / "bicing_root.json"
STATIONS_PATH = TMP / "bicing_stations.json"
VTYPES_PATH   = TMP / "bicing_vtypes.json"

# ── Detection thresholds (identical to French Gold Standard v1.0) ─────
GEOFILTER_LAT   = (35.0, 44.5)   # Spain perimeter (vs FR 41–52)
GEOFILTER_LON   = (-9.5, 4.5)    # Spain perimeter (vs FR -6–10)
SIGMA_MAX       = 3.0
N_MIN_DOCK      = 20
A2_PLACEHOLDER_THRESHOLD = 0.0   # zero capacity variance signals A2

FF_KEYWORDS     = {"bird", "dott", "pony", "voi", "tier", "lime", "bolt", "wind"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    root     = load(ROOT_PATH)
    vtypes   = load(VTYPES_PATH)
    stations = load(STATIONS_PATH)

    feeds    = list(root["data"].values())[0]["feeds"]
    feed_names = sorted(f["name"] for f in feeds)

    # ── Vehicle types: A1 detection (cars labelled as BSS?) ───────────
    vtype_list = vtypes["data"].get("vehicle_types", [])
    vtype_forms = sorted({v.get("form_factor", "?") for v in vtype_list})
    a1_cars     = any("car" in f for f in vtype_forms)

    raw_stations = stations["data"]["stations"]
    n_raw        = len(raw_stations)

    df = pd.DataFrame(raw_stations)
    # Bicing schema: lat, lon, capacity, station_id, name, ...
    df["lat"]      = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"]      = pd.to_numeric(df["lon"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    # ── A2 placeholder: zero variance on non-zero capacities ──────────
    nonzero_caps = df.loc[df["capacity"].fillna(0) > 0, "capacity"]
    cap_unique   = nonzero_caps.nunique()
    cap_std      = float(pstdev(nonzero_caps)) if len(nonzero_caps) > 1 else 0.0
    a2_placeholder = (cap_unique == 1) and (len(nonzero_caps) > 0)

    # ── A3 over-capacity: conditional-averaging signature ─────────────
    # Heuristic on French corpus: if free-floating, c_profile >> c_actual.
    # On a dock-based system, c_profile ≈ c_actual.
    mean_all    = float(df["capacity"].fillna(0).mean())
    mean_active = float(nonzero_caps.mean()) if len(nonzero_caps) else 0.0
    ratio       = mean_active / mean_all if mean_all > 0 else float("nan")
    a3_floating = ratio > 5.0   # same threshold used on the French corpus

    # ── A4 geospatial: out of perimeter or extreme outliers ───────────
    in_perimeter = (
        (df["lat"].between(*GEOFILTER_LAT)) &
        (df["lon"].between(*GEOFILTER_LON))
    )
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

    # ── A5 out-of-perimeter (system level) ────────────────────────────
    # System area proxy = bounding-box surface (deg² → ~km²)
    bbox_km2 = (
        (df_in["lat"].max() - df_in["lat"].min()) * 111.0 *
        (df_in["lon"].max() - df_in["lon"].min()) * 111.0 *
        abs((df_in["lat"].mean() * 3.14159 / 180.0).real)
    )
    a5_macro_region = bbox_km2 > 50_000.0

    # ── S6 size threshold ─────────────────────────────────────────────
    n_certified_dock = n_in - a4_outliers
    too_small        = n_certified_dock < N_MIN_DOCK

    verdict = {
        "system": "Bicing Barcelona",
        "country": "ES",
        "feed_url": "https://barcelona.publicbikesystem.net/customer/gbfs/v2/gbfs.json",
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
        "too_small_under_N_min":   bool(too_small),
        "bbox_km2_proxy":          round(bbox_km2, 1),
        "thresholds_used": {
            "geofilter_lat": GEOFILTER_LAT,
            "geofilter_lon": GEOFILTER_LON,
            "sigma_max":     SIGMA_MAX,
            "N_min":         N_MIN_DOCK,
        },
    }

    # ── Persist outputs ──────────────────────────────────────────────
    (HERE / "bicing_audit_report.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_df = df_in[df_in["d_sigma"] <= SIGMA_MAX].copy()
    out_df["station_type"] = "docked_bike"
    out_df["audit_status"] = "ok"
    out_df["system_id"]    = "bicing_barcelona"
    out_df["country"]      = "ES"
    out_cols = [
        "station_id", "name", "lat", "lon", "capacity",
        "station_type", "audit_status", "system_id", "country",
    ]
    out_df[out_cols].to_parquet(HERE / "bicing_stations.parquet", index=False)

    return verdict


def main():
    if not STATIONS_PATH.exists():
        sys.exit(
            f"Missing input: {STATIONS_PATH}. Fetch first with:\n"
            "  curl -o /tmp/bicing_root.json     <root_url>\n"
            "  curl -o /tmp/bicing_stations.json <station_info_url>\n"
            "  curl -o /tmp/bicing_vtypes.json   <vehicle_types_url>\n"
        )
    v = audit()
    print(json.dumps(v, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
