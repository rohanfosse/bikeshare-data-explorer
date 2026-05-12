"""E26 -- IMD-lite for European cities via OSM Overpass.

E25 obtained station coordinates for 10 European bike-sharing
systems. We now compute the most universal IMD component --
multimodality (M, heavy-transit stops within 300 m) -- on those
European cities using a single OSM Overpass query per city,
and compare directly with the French Gold-Standard panel.

For each city:
    1. Read the GBFS station coordinates cached from E25.
    2. Build a bounding box around the station set, expanded by
       1 km on each side.
    3. Query OSM Overpass for "railway=tram_stop OR
       railway=station OR public_transport=stop_position with
       train/subway/light_rail/tram tag" inside the bbox.
    4. For each station, count heavy stops within 300 m via
       spatial join in local-metric coordinates.
    5. Aggregate to a city-level mean (= the M component before
       Min-Max normalisation in our IMD definition).

We then place the EU non-FR cities on the same axis as the
French panel, computing for each an "IMD-lite" = M-component
only (Min-Max normalised on the combined panel of FR + EU
non-FR cities).

This is not the full four-dimensional IMD: I would require
detailed OSM cycleway extraction per buffer, S would require a
European equivalent of BAAC (not available at city granularity
in this study), and T requires SRTM elevation. We commit to the
M-only IMD-lite because (i) E16 identified M as the single
component that credibly drives well-being, (ii) the Sobol
decomposition of E7 attributed 77 % of the IMD-rank variance to
M, and (iii) it is the only component that can be computed
cleanly on a continental panel without extending the enrichment
pipeline.

Outputs:
    outputs/e26_results.json
    outputs/e26_imd_lite_eu.pdf
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

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BUFFER_RADIUS_M = 300.0


def _e25_cities() -> list[dict]:
    """Reload the European cities that E25 successfully fetched."""
    e25_path = OUT_DIR / "e25_results.json"
    if not e25_path.exists():
        log.error("E25 results not found. Run E25 first.")
        return []
    data = json.loads(e25_path.read_text(encoding="utf-8"))
    return [s for s in data["european_systems"] if s["status"] == "ok"]


def _fetch_station_coords(gbfs_url: str) -> list[tuple[float, float]]:
    """Re-fetch station coordinates from the GBFS auto-discovery URL."""
    req = Request(gbfs_url, headers={"User-Agent": "bikeshare-research/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            root = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("  gbfs root fetch failed: %s", exc)
        return []
    info_url = None
    data = root.get("data", {})
    if "feeds" in data:
        feeds = data["feeds"]
    else:
        for v in data.values():
            if isinstance(v, dict) and "feeds" in v:
                feeds = v["feeds"]
                break
        else:
            feeds = []
    for feed in feeds:
        if feed.get("name") == "station_information":
            info_url = feed.get("url")
            break
    if info_url is None:
        return []
    try:
        with urlopen(Request(info_url, headers={"User-Agent": "bikeshare-research/1.0"}), timeout=15) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("  station_information fetch failed: %s", exc)
        return []
    sts = info.get("data", {}).get("stations", [])
    coords = []
    for s in sts:
        lat = s.get("lat")
        lon = s.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lat), float(lon)))
    return coords


# Hardcoded GBFS URLs (re-used from E25)
GBFS_URLS = {
    "Utrecht": "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_ut/gbfs.json",
    "Madrid": "https://madrid.publicbikesystem.net/customer/gbfs/v3.0/gbfs.json",
    "Barcelona": "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_bs/gbfs.json",
    "Seville": "https://api.cyclocity.fr/contracts/seville/gbfs/v3/gbfs.json",
    "Vienna": "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_wr/gbfs.json",
    "Berlin": "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_cb/gbfs.json",
    "Copenhagen": "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_copenhagen/gbfs.json",
    "Geneva": "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_ge/gbfs.json",
    "Budapest": "https://stables.donkey.bike/api/public/gbfs/3.0/donkey_budapest/gbfs.json",
    "Warsaw": "https://gbfs.api.ridedott.com/public/v2/warsaw/gbfs.json",
}


def _query_overpass_heavy_stops(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Query OSM Overpass for heavy-transit stops within a bbox.

    bbox = (min_lat, min_lon, max_lat, max_lon).
    Heavy = railway=station OR railway=tram_stop OR
            railway=subway_entrance OR public_transport=station.
    """
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:60];
    (
      node["railway"="station"]({south},{west},{north},{east});
      node["railway"="tram_stop"]({south},{west},{north},{east});
      node["railway"="subway_entrance"]({south},{west},{north},{east});
      node["station"="subway"]({south},{west},{north},{east});
      node["station"="light_rail"]({south},{west},{north},{east});
      node["public_transport"="station"]({south},{west},{north},{east});
    );
    out center;
    """
    data = ("data=" + query).encode("utf-8")
    req = Request(OVERPASS_URL, data=data,
                  headers={"User-Agent": "bikeshare-research/1.0"})
    try:
        with urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("  Overpass query failed: %s", exc)
        return []
    coords = []
    for el in payload.get("elements", []):
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None:  # ways: use center
            c = el.get("center", {})
            lat = c.get("lat")
            lon = c.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lat), float(lon)))
    return coords


def _haversine_m(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Pairwise haversine distances in meters; coords are (lat, lon) arrays."""
    lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def _heavy_stops_per_station(stations: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """For each station, count heavy stops within 300 m."""
    if len(stops) == 0 or len(stations) == 0:
        return np.zeros(len(stations))
    dist = _haversine_m(stations, stops)
    return (dist <= BUFFER_RADIUS_M).sum(axis=1)


def _french_panel_M() -> dict[str, float]:
    """Mean heavy-stops per station per French city from Gold Standard."""
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    st = load_stations()
    dock = st[st["station_type"] == "docked_bike"]
    return dock.groupby("city")["gtfs_heavy_stops_300m"].mean().to_dict()


def main() -> None:
    log.info("Loading E25 city list...")
    eu_cities = _e25_cities()
    log.info("  %d successful European cities from E25", len(eu_cities))

    eu_M = {}
    eu_n_stations = {}
    eu_failed = []
    for entry in eu_cities:
        city = entry["city"]
        gbfs_url = GBFS_URLS.get(city)
        if gbfs_url is None:
            log.warning("  %s: no GBFS URL recorded", city)
            eu_failed.append(city)
            continue
        log.info("  %s ...", city)
        coords = _fetch_station_coords(gbfs_url)
        if len(coords) < 5:
            log.warning("  %s: only %d coords retrieved, skipping",
                        city, len(coords))
            eu_failed.append(city)
            continue
        coords_arr = np.array(coords)
        # bbox: 1 km buffer around min/max lat-lon
        lat_buf = 1000.0 / 111000.0
        lon_buf = 1000.0 / (111000.0 * np.cos(np.radians(coords_arr[:, 0].mean())))
        bbox = (coords_arr[:, 0].min() - lat_buf,
                coords_arr[:, 1].min() - lon_buf,
                coords_arr[:, 0].max() + lat_buf,
                coords_arr[:, 1].max() + lon_buf)
        log.info("    bbox: lat=[%.3f, %.3f], lon=[%.3f, %.3f]",
                 bbox[0], bbox[2], bbox[1], bbox[3])
        # Polite delay between Overpass queries
        time.sleep(1.5)
        stops = _query_overpass_heavy_stops(bbox)
        if not stops:
            log.warning("    %s: zero heavy stops found (maybe rate-limited)", city)
        log.info("    %d heavy stops, %d stations -> computing per-station count",
                 len(stops), len(coords))
        counts = _heavy_stops_per_station(
            coords_arr,
            np.array(stops) if stops else np.zeros((0, 2)),
        )
        mean_M = float(counts.mean())
        eu_M[city] = mean_M
        eu_n_stations[city] = int(len(coords))
        log.info("    mean heavy stops / station = %.2f", mean_M)

    log.info("\nFrench Gold Standard reference M values:")
    fr_M = _french_panel_M()
    log.info("  %d French cities", len(fr_M))

    # Combined panel
    rows = []
    for city, M in eu_M.items():
        rows.append({"city": city, "country": "EU", "M": M,
                     "n_stations": eu_n_stations[city]})
    for city, M in fr_M.items():
        rows.append({"city": city, "country": "FR", "M": M,
                     "n_stations": int(0)})  # FR n_stations not strictly needed here
    df = pd.DataFrame(rows).dropna(subset=["M"])

    # Cross-panel Min-Max normalisation
    M_min, M_max = df["M"].min(), df["M"].max()
    df["M_norm"] = (df["M"] - M_min) / (M_max - M_min) if M_max > M_min else 0.5

    fr_M_dist = df[df["country"] == "FR"]["M"]
    eu_M_dist = df[df["country"] == "EU"]["M"]
    log.info("\nDistribution of raw M (heavy stops / station within 300 m):")
    log.info("  FR median = %.2f (p25=%.2f, p75=%.2f, max=%.2f, n=%d)",
             fr_M_dist.median(), fr_M_dist.quantile(0.25),
             fr_M_dist.quantile(0.75), fr_M_dist.max(), len(fr_M_dist))
    log.info("  EU median = %.2f (p25=%.2f, p75=%.2f, max=%.2f, n=%d)",
             eu_M_dist.median(), eu_M_dist.quantile(0.25),
             eu_M_dist.quantile(0.75), eu_M_dist.max(), len(eu_M_dist))

    log.info("\nIMD-lite (Min-Max on combined panel):")
    df_sorted = df.sort_values("M_norm", ascending=False).reset_index(drop=True)
    log.info("Top-15 cities by M_norm:")
    for _, r in df_sorted.head(15).iterrows():
        log.info("  %s | %-20s | M = %5.2f | M_norm = %.3f",
                 r["country"], r["city"], r["M"], r["M_norm"])

    results = {
        "eu_failed": eu_failed,
        "eu_M": eu_M,
        "eu_n_stations": eu_n_stations,
        "fr_panel_n": int(len(fr_M)),
        "M_min": float(M_min),
        "M_max": float(M_max),
        "fr_M_summary": {
            "median": float(fr_M_dist.median()),
            "p25": float(fr_M_dist.quantile(0.25)),
            "p75": float(fr_M_dist.quantile(0.75)),
            "max": float(fr_M_dist.max()),
            "n": int(len(fr_M_dist)),
        },
        "eu_M_summary": {
            "median": float(eu_M_dist.median()) if len(eu_M_dist) > 0 else None,
            "p25": float(eu_M_dist.quantile(0.25)) if len(eu_M_dist) > 0 else None,
            "p75": float(eu_M_dist.quantile(0.75)) if len(eu_M_dist) > 0 else None,
            "max": float(eu_M_dist.max()) if len(eu_M_dist) > 0 else None,
            "n": int(len(eu_M_dist)),
        },
        "imd_lite_ranking": df_sorted.to_dict("records"),
    }
    out_json = OUT_DIR / "e26_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: distribution comparison + top-20 ranking
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    bp = ax.boxplot(
        [fr_M_dist.dropna(), eu_M_dist.dropna()],
        vert=True, tick_labels=["France", "Europe (non-FR)"],
        patch_artist=True, showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.6},
        medianprops={"color": "#A8201A"},
    )
    for patch, col in zip(bp["boxes"], ["#1F3A6B", "#D08020"]):
        patch.set_facecolor(col)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)
    ax.set_ylabel("Heavy GTFS / OSM rail stops per station within 300 m")
    ax.set_title(f"Multimodality M: FR (n={len(fr_M_dist)}) vs.\\ "
                 f"EU non-FR (n={len(eu_M_dist)})", fontsize=10)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)

    ax = axes[1]
    top = df_sorted.head(20).iloc[::-1]
    colors = ["#D08020" if c == "EU" else "#1F3A6B"
              for c in top["country"]]
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["M"], color=colors,
            edgecolor="white", linewidth=0.4)
    for j, (_, r) in enumerate(top.iterrows()):
        ax.text(r["M"] + 0.05, j, f"{r['country']}",
                fontsize=7, color="#404040", va="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["city"], fontsize=8)
    ax.set_xlabel("Raw M (heavy stops / station within 300 m)")
    ax.set_title("Top-20 cities by multimodality (FR + EU non-FR)",
                 fontsize=10)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)

    fig.suptitle("IMD-lite multimodality component: France against Europe",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e26_imd_lite_eu.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e26_imd_lite_eu.pdf")


if __name__ == "__main__":
    main()
