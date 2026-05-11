"""
Massive cross-country audit of GBFS systems from the MobilityData
canonical catalogue (https://github.com/MobilityData/gbfs).

Applies the A1-A5 rule set (plus the candidate A6 zero-capacity dock
discovered on Citi Bike NYC) to all reachable non-French GBFS feeds
in the catalogue. Reports a per-system verdict and aggregates by
country.

Inputs:
  $TEMP/gbfs_systems.csv  (fetched from raw.githubusercontent.com)

Outputs:
  massive_audit_results.csv   one row per audited system
  massive_audit_summary.json  country-level + anomaly-level aggregates
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = Path(__file__).parent
TMP = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")
CSV_IN = TMP / "gbfs_systems.csv"

# Country perimeters (broad bounding boxes for A4/A5 checks).
COUNTRY_BBOX: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    # Calibrated against the OpenStreetMap "place=country" bounding boxes
    # with a 0.5-degree buffer on each side to absorb GPS noise at coastal
    # stations. Where the country has remote territories (Spitsbergen for
    # Norway, Canary for Spain, Hawaii for US) the bbox is extended.
    "FR": ((41.0, 51.5),  (-6.0, 10.0)),
    "ES": ((27.5, 44.5),  (-19.0, 4.5)),
    "PT": ((32.0, 42.5),  (-32.0, -6.0)),
    "IT": ((35.5, 47.5),  (6.0, 19.0)),
    "DE": ((47.0, 55.5),  (5.5, 15.5)),
    "AT": ((46.0, 49.5),  (9.0, 17.5)),
    "CH": ((45.5, 48.0),  (5.5, 11.0)),
    "BE": ((49.0, 52.0),  (2.0, 7.0)),
    "NL": ((50.5, 54.0),  (3.0, 7.5)),
    "LU": ((49.4, 50.2),  (5.7, 6.6)),
    "GB": ((49.5, 61.5),  (-9.0, 2.5)),
    "IE": ((51.0, 55.5),  (-11.0, -5.0)),
    "DK": ((54.0, 58.0),  (7.5, 15.5)),
    "SE": ((54.5, 69.5),  (10.5, 24.5)),
    "NO": ((57.5, 71.5),  (4.0, 31.5)),
    "FI": ((59.5, 70.5),  (19.0, 32.0)),
    "PL": ((48.5, 55.5),  (13.5, 24.5)),
    "CZ": ((48.0, 51.5),  (12.0, 19.0)),
    "SK": ((47.5, 50.0),  (16.5, 23.0)),
    "HU": ((45.5, 49.0),  (16.0, 23.0)),
    "RO": ((43.0, 48.5),  (20.0, 30.0)),
    "HR": ((42.0, 47.0),  (13.0, 20.0)),
    "SI": ((45.0, 47.0),  (13.0, 17.0)),
    "BA": ((42.5, 45.5),  (15.5, 20.0)),
    "GR": ((34.5, 42.0),  (19.0, 28.5)),
    "BG": ((41.0, 44.5),  (22.0, 29.0)),
    "CY": ((34.5, 35.7),  (32.0, 35.0)),
    "LT": ((53.5, 56.5),  (20.5, 27.0)),
    "LV": ((55.5, 58.5),  (20.5, 28.5)),
    "EE": ((57.0, 60.0),  (21.5, 28.5)),
    "IS": ((63.0, 67.0),  (-25.0, -13.0)),
    "MC": ((43.7, 43.8),  (7.4,  7.5)),
    "TR": ((35.5, 42.5),  (25.5, 45.0)),
    "UA": ((44.0, 53.0),  (22.0, 41.0)),
    "XK": ((41.5, 43.5),  (20.0, 22.0)),
    "LI": ((47.0, 47.3),  (9.4,  9.7)),
    "US": ((18.0, 72.0),  (-180.0, -65.0)),  # incl. Alaska, Hawaii
    "CA": ((42.0, 84.0),  (-141.0, -52.0)),
    "MX": ((14.0, 33.0),  (-118.5, -86.0)),
    "BR": ((-34.0, 6.0),  (-74.0, -34.0)),
    "AR": ((-55.0, -22.0),(-74.0, -53.5)),
    "CL": ((-56.0, -17.0),(-76.0, -66.0)),
    "CO": ((-5.0, 14.0),  (-80.0, -66.0)),
    "AU": ((-44.0, -10.0),(112.0, 154.0)),
    "NZ": ((-47.5, -34.0),(166.0, 179.0)),
    "JP": ((20.0, 46.0),  (122.0, 154.0)),
    "KR": ((33.0, 39.0),  (124.0, 132.0)),
    "TW": ((21.5, 25.5),  (119.0, 122.5)),
    "AE": ((22.0, 26.5),  (51.0, 57.0)),
    "QA": ((24.0, 27.0),  (50.5, 52.0)),
    "SA": ((16.0, 33.0),  (34.0, 56.0)),
    "IL": ((29.0, 33.5),  (34.0, 36.0)),
    "MY": ((1.0, 7.5),    (99.5, 119.5)),
    "SG": ((1.1, 1.5),    (103.5, 104.1)),
}
SIGMA_MAX = 3.0
N_MIN_DOCK = 20
TIMEOUT = 8
# A4 is flagged at the system level only if >= 5% of stations OR
# >= 5 absolute stations fall outside the country perimeter, to avoid
# false positives from edge-of-country coastal stations.
A4_PERIM_MIN_PCT = 5.0
A4_PERIM_MIN_ABS = 5

# Reused HTTP session per thread
import threading
_local = threading.local()


def _convex_hull_area_km2(lats: list[float], lons: list[float]) -> float:
    """Return the convex hull area of a station cloud, in km^2.

    Projects each station to a local equirectangular frame centred on
    the cloud centroid, then computes the convex hull area via the
    shoelace formula. Returns 0.0 for degenerate clouds (< 3 stations
    or collinear).
    """
    n = len(lats)
    if n < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        # Fallback: return bbox-style area if scipy is unavailable.
        return (max(lats) - min(lats)) * 111.0 * \
               (max(lons) - min(lons)) * 111.0

    mean_lat = sum(lats) / n
    mean_lon = sum(lons) / n
    km_per_deg_lat = 111.0
    # Equirectangular x-scale depends on the latitude.
    import math
    km_per_deg_lon = 111.0 * math.cos(math.radians(mean_lat))

    pts = np.column_stack([
        (np.array(lons) - mean_lon) * km_per_deg_lon,
        (np.array(lats) - mean_lat) * km_per_deg_lat,
    ])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 0.0
    # ConvexHull.volume in 2D is the polygon area (in km^2 here).
    return float(hull.volume)


def get_session() -> requests.Session:
    if not hasattr(_local, "session"):
        s = requests.Session()
        retry = Retry(total=1, backoff_factor=0.2,
                      status_forcelist=(429, 500, 502, 503, 504))
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "gbfs-audit-catalogue/1.0 (research)"})
        _local.session = s
    return _local.session


def fetch_json(url: str) -> dict | None:
    try:
        r = get_session().get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def discover_endpoints(root: dict) -> dict[str, str]:
    """Extract feed-name -> url mapping from a GBFS root document."""
    data = root.get("data", {})
    # Pick first language entry
    feeds = []
    if isinstance(data, dict):
        for lang_block in data.values():
            if isinstance(lang_block, dict) and "feeds" in lang_block:
                feeds = lang_block["feeds"]
                break
        if not feeds:
            # GBFS v3 may put feeds at root data directly
            feeds = data.get("feeds", []) or []
    if isinstance(feeds, dict):
        feeds = feeds.get("en", []) if "en" in feeds else next(iter(feeds.values()), [])
    return {f["name"]: f["url"] for f in feeds if isinstance(f, dict)}


def audit_system(row: dict) -> dict[str, Any]:
    country  = (row.get("Country Code") or "").upper()
    name     = row.get("Name") or "?"
    sys_id   = row.get("System ID") or "?"
    root_url = (row.get("Auto-Discovery URL") or "").strip()

    out: dict[str, Any] = {
        "country": country, "name": name, "system_id": sys_id,
        "root_url": root_url, "reachable": False, "stations": 0,
        "vehicle_form_factors": "", "a1_cars": None, "a2_placeholder": None,
        "a3_overcap_ratio": None, "a3_overcap_flag": None,
        "a4_out_of_perim": None, "a4_outliers": None,
        "a5_macro_bbox_km2": None, "a5_macro_flag": None,
        "a6_zero_capacity_pct": None,
        "capacity_nan_pct": None,
        "centroid_lat": None, "centroid_lon": None,
        "any_anomaly": None,
    }
    if not root_url:
        return out

    root = fetch_json(root_url)
    if not root:
        return out
    out["reachable"] = True

    endpoints = discover_endpoints(root)
    si_url = endpoints.get("station_information")
    vt_url = endpoints.get("vehicle_types")
    if not si_url:
        return out

    si = fetch_json(si_url)
    if not si:
        return out
    stations = (si.get("data") or {}).get("stations") or []
    out["stations"] = len(stations)
    if not stations:
        return out

    # Vehicle form factors -> A1
    # A1 triggers when at least one vehicle_type has a true automobile
    # form factor. The match must be EXACT against the GBFS v3
    # vocabulary; substring matching incorrectly catches
    # "cargo_bicycle" (cargo bike, not car).
    if vt_url:
        vt = fetch_json(vt_url) or {}
        vtypes = (vt.get("data") or {}).get("vehicle_types") or []
        forms = sorted({(v.get("form_factor") or "?") for v in vtypes})
        out["vehicle_form_factors"] = ",".join(forms)
        # GBFS v3 reserves "car" for automobiles.
        out["a1_cars"] = "car" in forms

    # Capacity stats
    caps = []
    for s in stations:
        c = s.get("capacity")
        if isinstance(c, (int, float)):
            caps.append(float(c))
        else:
            caps.append(None)
    n = len(caps)
    n_nan = sum(1 for c in caps if c is None)
    n_zero = sum(1 for c in caps if c == 0)
    nonzero = [c for c in caps if c is not None and c > 0]
    out["capacity_nan_pct"] = round(100 * n_nan / n, 2) if n else 0.0
    out["a6_zero_capacity_pct"] = round(100 * n_zero / n, 2) if n else 0.0

    # A2: constant placeholder
    if nonzero:
        unique_nz = len(set(nonzero))
        out["a2_placeholder"] = (unique_nz == 1 and len(nonzero) > 1)
    else:
        out["a2_placeholder"] = False

    # A3: capacity-profile ratio
    if caps and nonzero:
        mean_all = sum(c if c is not None else 0 for c in caps) / n
        mean_active = sum(nonzero) / len(nonzero)
        ratio = mean_active / mean_all if mean_all > 0 else float("inf")
        out["a3_overcap_ratio"] = round(ratio, 3)
        out["a3_overcap_flag"] = ratio > 5.0
    else:
        out["a3_overcap_ratio"] = None
        out["a3_overcap_flag"] = False

    # A4 + A5: geospatial
    lats = [s.get("lat") for s in stations if isinstance(s.get("lat"), (int, float))]
    lons = [s.get("lon") for s in stations if isinstance(s.get("lon"), (int, float))]
    if not lats or not lons:
        return out
    bbox = COUNTRY_BBOX.get(country)
    if bbox:
        (la_min, la_max), (lo_min, lo_max) = bbox
        in_perim = [(la_min <= la <= la_max and lo_min <= lo <= lo_max)
                    for la, lo in zip(lats, lons)]
        out["a4_out_of_perim"] = sum(1 for ok in in_perim if not ok)
        lats_in = [la for la, ok in zip(lats, in_perim) if ok]
        lons_in = [lo for lo, ok in zip(lons, in_perim) if ok]
    else:
        out["a4_out_of_perim"] = 0
        lats_in, lons_in = lats, lons

    if len(lats_in) > 2:
        mlat = sum(lats_in) / len(lats_in)
        mlon = sum(lons_in) / len(lons_in)
        out["centroid_lat"] = round(mlat, 5)
        out["centroid_lon"] = round(mlon, 5)
        slat = (sum((la - mlat) ** 2 for la in lats_in) / max(1, len(lats_in) - 1)) ** 0.5
        slon = (sum((lo - mlon) ** 2 for lo in lons_in) / max(1, len(lons_in) - 1)) ** 0.5
        outliers = 0
        for la, lo in zip(lats_in, lons_in):
            if slat > 0 and slon > 0:
                d = (((la - mlat) / slat) ** 2 + ((lo - mlon) / slon) ** 2) ** 0.5
                if d > SIGMA_MAX:
                    outliers += 1
        out["a4_outliers"] = outliers

        # Bounding box (loose envelope, kept for backward comparison)
        bbox_km2 = (max(lats_in) - min(lats_in)) * 111.0 * \
                   (max(lons_in) - min(lons_in)) * 111.0
        out["a5_macro_bbox_km2"] = round(bbox_km2, 1)

        # Convex hull (tight envelope, used for A5 detection)
        hull_km2 = _convex_hull_area_km2(lats_in, lons_in)
        out["a5_macro_hull_km2"] = round(hull_km2, 1)
        out["a5_macro_flag"] = hull_km2 > 50_000.0
    else:
        out["a4_outliers"] = 0
        out["a5_macro_bbox_km2"] = 0.0
        out["a5_macro_hull_km2"] = 0.0
        out["a5_macro_flag"] = False

    # Any anomaly flag: A4 requires a non-trivial out-of-perimeter mass.
    n_stations = out["stations"] or 1
    a4_perim_pct = 100.0 * (out.get("a4_out_of_perim") or 0) / n_stations
    a4_perim_flag = (a4_perim_pct >= A4_PERIM_MIN_PCT
                     and (out.get("a4_out_of_perim") or 0) >= A4_PERIM_MIN_ABS)
    out["a4_perim_flag"] = a4_perim_flag
    out["a4_perim_pct"] = round(a4_perim_pct, 2)

    flags = [out["a1_cars"], out["a2_placeholder"], out["a3_overcap_flag"],
             a4_perim_flag, out["a5_macro_flag"]]
    out["any_anomaly"] = any(flags)
    return out


def main(max_systems: int | None = None, skip_fr: bool = True) -> None:
    with CSV_IN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if skip_fr:
        rows = [r for r in rows if (r.get("Country Code") or "").upper() != "FR"]
    if max_systems:
        rows = rows[:max_systems]
    print(f"Auditing {len(rows)} systems...", file=sys.stderr)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(audit_system, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result(timeout=30))
            except Exception:
                results.append({"reachable": False})
            if i % 50 == 0:
                print(f"  ... {i}/{len(rows)}", file=sys.stderr)

    # Write CSV
    keys = ["country", "name", "system_id", "stations", "reachable",
            "vehicle_form_factors", "a1_cars", "a2_placeholder",
            "a3_overcap_ratio", "a3_overcap_flag",
            "a4_out_of_perim", "a4_perim_pct", "a4_perim_flag",
            "a4_outliers",
            "a5_macro_bbox_km2", "a5_macro_hull_km2", "a5_macro_flag",
            "a6_zero_capacity_pct", "capacity_nan_pct",
            "centroid_lat", "centroid_lon", "any_anomaly",
            "root_url"]
    out_csv = HERE / "massive_audit_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})

    # Aggregates
    reachable = [r for r in results if r.get("reachable")]
    have_stations = [r for r in reachable if (r.get("stations") or 0) > 0]
    flagged = [r for r in have_stations if r.get("any_anomaly")]

    by_country = defaultdict(lambda: {"audited": 0, "reachable": 0, "flagged": 0,
                                       "anomalies": defaultdict(int)})
    for r in results:
        c = r.get("country", "?")
        by_country[c]["audited"] += 1
        if r.get("reachable"):
            by_country[c]["reachable"] += 1
        if r.get("any_anomaly"):
            by_country[c]["flagged"] += 1
            for k in ("a1_cars", "a2_placeholder", "a3_overcap_flag",
                      "a4_perim_flag", "a5_macro_flag"):
                if r.get(k):
                    by_country[c]["anomalies"][k] += 1
    by_country_sorted = sorted(by_country.items(), key=lambda kv: -kv[1]["audited"])

    # A6 candidate (c=0): top systems by zero-capacity rate
    a6_top = sorted(
        [r for r in have_stations if (r.get("a6_zero_capacity_pct") or 0) > 1.0],
        key=lambda r: -(r.get("a6_zero_capacity_pct") or 0),
    )[:25]

    # NaN top
    nan_top = sorted(
        [r for r in have_stations if (r.get("capacity_nan_pct") or 0) > 50.0],
        key=lambda r: -(r.get("capacity_nan_pct") or 0),
    )[:25]

    summary = {
        "total_audited": len(results),
        "reachable": len(reachable),
        "with_stations": len(have_stations),
        "flagged_anomalies": len(flagged),
        "by_country": {
            c: {"audited": v["audited"], "reachable": v["reachable"],
                "flagged": v["flagged"], "anomalies": dict(v["anomalies"])}
            for c, v in by_country_sorted
        },
        "a6_zero_capacity_top": [
            {"country": r["country"], "name": r["name"],
             "stations": r["stations"], "c0_pct": r["a6_zero_capacity_pct"]}
            for r in a6_top
        ],
        "high_nan_systems": [
            {"country": r["country"], "name": r["name"],
             "stations": r["stations"], "nan_pct": r["capacity_nan_pct"]}
            for r in nan_top
        ],
    }
    (HERE / "massive_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2)[:3000])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_systems=n)
