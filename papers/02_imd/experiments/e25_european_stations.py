"""E25 -- Live station-level European GBFS fetch.

Extends E24 from a country-level catalogue analysis to a
city-level station inventory by querying the live GBFS endpoints
of fifteen major European bike-sharing systems. For each system
we attempt to retrieve station_information.json, count active
docked stations, and compute the convex-hull area of the network
in km^2 as a coarse "served-area" proxy.

The resulting European panel is then put side-by-side with the
French panel computed by the Gold Standard pipeline. We cannot
compute the full IMD on European cities here -- that would
require running the OSM/BAAC/GTFS enrichment for each city --
but we can compare the most universal supply-side metric:
station density per km^2 of served area.

Cities targeted (auto-discovery URL from MobilityData catalogue):
    Amsterdam, Rotterdam, Utrecht, Madrid (BiciMAD),
    Barcelona (AMBici), Seville (Sevici), Vienna (WienMobil),
    Berlin (nextbike), Copenhagen (Donkey), Geneva (Donkey),
    Budapest (Donkey), Helsinki (Bird -- skipped if dockless),
    Warsaw (Dott -- skipped if dockless), Brussels (Bolt --
    skipped if dockless), Krakow (Dott).

Outputs:
    outputs/e25_results.json
    outputs/e25_european_stations.pdf
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CANDIDATE_SYSTEMS = [
    ("NL", "Amsterdam",
     "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_ams/gbfs.json",
     "Donkey Republic"),
    ("NL", "Rotterdam",
     "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_rtm/gbfs.json",
     "Donkey Republic"),
    ("NL", "Utrecht",
     "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_ut/gbfs.json",
     "Donkey Republic"),
    ("ES", "Madrid",
     "https://madrid.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json",
     "BiciMAD"),
    ("ES", "Barcelona",
     "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_bs/gbfs.json",
     "AMBici (nextbike)"),
    ("ES", "Seville",
     "https://api.cyclocity.fr/contracts/seville/gbfs/v3/gbfs.json",
     "Sevici (Cyclocity)"),
    ("AT", "Vienna",
     "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_wr/gbfs.json",
     "WienMobil (nextbike)"),
    ("DE", "Berlin",
     "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_cb/gbfs.json",
     "Campus Berlin-Buch"),
    ("DK", "Copenhagen",
     "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_copenhagen/gbfs.json",
     "Donkey Republic"),
    ("CH", "Geneva",
     "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_ge/gbfs.json",
     "Donkey Republic"),
    ("HU", "Budapest",
     "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_budapest/gbfs.json",
     "Donkey Republic"),
    ("CZ", "Prague",
     "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_cz/gbfs.json",
     "nextbike"),
    ("IT", "Milan",
     "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_mb/gbfs.json",
     "BikeMi (nextbike)"),
    ("FI", "Helsinki",
     "https://api.smoove.pro/api-public/gbfs/2/hki-cb/gbfs.json",
     "City Bikes"),
    ("PL", "Warsaw",
     "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_pl/gbfs.json",
     "Veturilo (nextbike)"),
]


def _fetch_json(url: str, timeout: float = 12.0) -> dict | None:
    try:
        req = Request(url, headers={"User-Agent": "bikeshare-research/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, json.JSONDecodeError, TimeoutError, ConnectionError) as exc:
        log.warning("  fetch failed for %s: %s", url, exc)
        return None


def _find_feed(gbfs_root: dict, target_name: str) -> str | None:
    """Find a feed URL by name across all language buckets."""
    if not isinstance(gbfs_root, dict):
        return None
    data = gbfs_root.get("data", gbfs_root)
    # GBFS v3
    if "feeds" in data:
        feeds = data["feeds"]
    elif isinstance(data, dict):
        # GBFS v2: {language_code: {feeds: [...]}}
        for k, v in data.items():
            if isinstance(v, dict) and "feeds" in v:
                feeds = v["feeds"]
                break
        else:
            feeds = []
    else:
        feeds = []
    for feed in feeds:
        if feed.get("name") == target_name:
            return feed.get("url")
    return None


def _hull_area_km2(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return 0.0
    arr = np.array(coords)
    # Convert lat/lon to approx local meters with equirectangular projection
    lat_mean = arr[:, 0].mean()
    lon0 = arr[:, 1].mean()
    R = 6371000.0  # earth radius m
    x = R * np.radians(arr[:, 1] - lon0) * np.cos(np.radians(lat_mean))
    y = R * np.radians(arr[:, 0] - lat_mean)
    pts = np.column_stack([x, y])
    try:
        hull = ConvexHull(pts)
        return float(hull.volume / 1e6)  # 2D volume = area; m^2 -> km^2
    except Exception:
        return 0.0


def _fetch_one(country: str, city: str, gbfs_url: str, operator: str) -> dict:
    log.info("  %s -- %s (%s) ...", country, city, operator)
    root = _fetch_json(gbfs_url)
    if root is None:
        return {"country": country, "city": city, "operator": operator,
                "status": "unreachable", "n_stations": 0,
                "area_km2": 0.0, "density_per_km2": 0.0}
    info_url = _find_feed(root, "station_information")
    if info_url is None:
        return {"country": country, "city": city, "operator": operator,
                "status": "no_station_information_feed",
                "n_stations": 0, "area_km2": 0.0, "density_per_km2": 0.0}
    info = _fetch_json(info_url)
    if info is None:
        return {"country": country, "city": city, "operator": operator,
                "status": "info_unreachable", "n_stations": 0,
                "area_km2": 0.0, "density_per_km2": 0.0}
    stations = info.get("data", {}).get("stations", [])
    if not stations and "stations" in info:
        stations = info["stations"]
    coords = []
    capacities = []
    for s in stations:
        lat = s.get("lat")
        lon = s.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lat), float(lon)))
        cap = s.get("capacity")
        if cap is not None:
            try:
                capacities.append(int(cap))
            except (TypeError, ValueError):
                pass
    if not coords:
        return {"country": country, "city": city, "operator": operator,
                "status": "no_coords", "n_stations": 0,
                "area_km2": 0.0, "density_per_km2": 0.0}
    area_km2 = _hull_area_km2(coords)
    return {
        "country": country,
        "city": city,
        "operator": operator,
        "status": "ok",
        "n_stations": len(coords),
        "n_with_capacity": len(capacities),
        "mean_capacity": float(np.mean(capacities)) if capacities else None,
        "area_km2": float(area_km2),
        "density_per_km2": float(len(coords) / area_km2) if area_km2 > 0 else 0.0,
        "centroid_lat": float(np.mean([c[0] for c in coords])),
        "centroid_lon": float(np.mean([c[1] for c in coords])),
    }


def _french_comparators() -> pd.DataFrame:
    """Build the French panel comparators using Gold Standard data."""
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    st = load_stations()
    dock = st[st["station_type"] == "docked_bike"]
    rows = []
    for city, sub in dock.groupby("city"):
        coords = list(zip(sub["lat"], sub["lon"]))
        area = _hull_area_km2(coords)
        rows.append({
            "country": "FR",
            "city": city,
            "operator": str(sub["system_name"].iloc[0]) if "system_name" in sub.columns else "",
            "status": "fr_gold_standard",
            "n_stations": int(len(sub)),
            "n_with_capacity": int(sub["capacity"].notna().sum()),
            "mean_capacity": float(sub["capacity"].mean()),
            "area_km2": float(area),
            "density_per_km2": float(len(sub) / area) if area > 0 else 0.0,
            "centroid_lat": float(sub["lat"].mean()),
            "centroid_lon": float(sub["lon"].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    log.info("Fetching European GBFS systems...")
    eu_rows = []
    for cc, city, url, operator in CANDIDATE_SYSTEMS:
        eu_rows.append(_fetch_one(cc, city, url, operator))
        time.sleep(0.5)  # polite delay
    eu_df = pd.DataFrame(eu_rows)

    log.info("\nEuropean fetch results:")
    for _, r in eu_df.iterrows():
        log.info("  %s | %-20s | n=%4d | %6.1f km^2 | %.1f stations/km^2 | %s",
                 r["country"], r["city"], r["n_stations"],
                 r["area_km2"], r["density_per_km2"], r["status"])

    ok_eu = eu_df[eu_df["status"] == "ok"].copy()
    log.info("\nSuccessfully fetched %d European systems", len(ok_eu))

    log.info("\nBuilding French comparator panel from Gold Standard...")
    fr_df = _french_comparators()
    fr_df = fr_df[fr_df["n_stations"] >= 5].copy()
    log.info("  French dock-based cities with >= 5 stations: %d", len(fr_df))

    # Combined panel for comparison
    combined = pd.concat([ok_eu, fr_df], ignore_index=True)

    # Summary statistics
    stats = {}
    for label, sub in [("Europe (EU non-FR)", ok_eu),
                       ("France (Gold Standard)", fr_df)]:
        if sub.empty:
            continue
        stats[label] = {
            "n_cities": int(len(sub)),
            "median_n_stations": float(sub["n_stations"].median()),
            "max_n_stations": int(sub["n_stations"].max()),
            "median_area_km2": float(sub["area_km2"].median()),
            "median_density_per_km2": float(sub["density_per_km2"].median()),
            "p75_density_per_km2": float(sub["density_per_km2"].quantile(0.75)),
        }
        log.info(
            "%-25s  n_cities=%3d  median_stations=%5.0f  "
            "median_area=%5.0f km^2  median_density=%4.2f stations/km^2",
            label, stats[label]["n_cities"],
            stats[label]["median_n_stations"],
            stats[label]["median_area_km2"],
            stats[label]["median_density_per_km2"],
        )

    results = {
        "european_systems": eu_df.to_dict("records"),
        "summary_stats": stats,
        "ok_european_cities": ok_eu["city"].tolist(),
        "n_attempted": int(len(eu_df)),
        "n_success": int(len(ok_eu)),
    }
    out_json = OUT_DIR / "e25_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: density distribution (Europe vs France) + scatter
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    ax = axes[0]
    # Box plot of density (EU non-FR vs FR)
    box_eu = ok_eu["density_per_km2"][ok_eu["density_per_km2"] > 0].to_numpy()
    box_fr = fr_df["density_per_km2"][fr_df["density_per_km2"] > 0].to_numpy()
    bp = ax.boxplot([box_fr, box_eu], vert=True, labels=["France", "Europe (non-FR)"],
                    patch_artist=True, showfliers=False,
                    medianprops={"color": "#A8201A"})
    for patch, col in zip(bp["boxes"], ["#1F3A6B", "#7095C8"]):
        patch.set_facecolor(col)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.4)
    ax.set_ylabel(r"Station density (stations / km$^{2}$ of served area)")
    ax.set_title(f"FR (n={len(box_fr)}) vs.\\ EU non-FR (n={len(box_eu)})",
                 fontsize=10)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)

    ax = axes[1]
    # Scatter: n_stations vs area_km2, log-log, France blue, EU orange
    if not fr_df.empty:
        ax.scatter(fr_df["area_km2"], fr_df["n_stations"],
                   s=24, color="#1F3A6B", alpha=0.6,
                   edgecolor="white", linewidth=0.4,
                   label=f"France n={len(fr_df)}")
    if not ok_eu.empty:
        ax.scatter(ok_eu["area_km2"], ok_eu["n_stations"],
                   s=48, color="#D08020", alpha=0.85,
                   edgecolor="white", linewidth=0.4,
                   label=f"EU non-FR n={len(ok_eu)}")
        for _, r in ok_eu.iterrows():
            ax.annotate(r["city"], (r["area_km2"], r["n_stations"]),
                        fontsize=7, color="#202020",
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Served-area convex hull (km$^{2}$, log scale)")
    ax.set_ylabel("Dock-based stations (log scale)")
    ax.set_title("Network footprint comparison", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", color="#E5E5E5", linewidth=0.5)

    fig.suptitle("European vs.\\ French bike-sharing footprints "
                 "(live GBFS, April 2026)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e25_european_stations.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e25_european_stations.pdf")


if __name__ == "__main__":
    main()
