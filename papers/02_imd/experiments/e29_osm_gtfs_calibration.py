"""E29 -- OSM-vs-GTFS calibration of the multimodality component.

E26 reported a 6.6x gap on multimodality between the French panel
(GTFS-based) and the European panel (OSM-based). Before accepting
this ratio as substantive, we test whether the OSM and GTFS
operationalisations of "heavy public-transport stop" agree on the
French panel.

For each of the ten largest dock-based French cities by station
count, we:

  1. Read the Gold Standard's per-station GTFS heavy-stop counts
     (gtfs_heavy_stops_300m) and aggregate to a city mean
     (= M-GTFS).
  2. Query OSM Overpass for the same tag set as E26
     (railway=station OR tram_stop OR subway_entrance OR
     public_transport=station) inside a bbox covering the city's
     stations.
  3. Count, per station, OSM heavy stops within 300 m and
     aggregate to a city mean (= M-OSM).
  4. Compute the ratio M-OSM / M-GTFS and the Spearman rank
     correlation across the panel.

Interpretation:
- If M-OSM ≈ M-GTFS, the EU/FR comparison of E26 stands as-is.
- If M-OSM systematically under-counts M-GTFS, the EU-vs-FR gap
  reported in E26 is even larger than 6.6x (because OSM
  under-counts in EU too).
- If M-OSM over-counts M-GTFS, the gap is smaller -- and we must
  apply a correction factor to E26.

Outputs:
    outputs/e29_results.json
    outputs/e29_osm_vs_gtfs.pdf
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import ROOT

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BUFFER_RADIUS_M = 300.0


def _haversine_m(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def _query_overpass_heavy_stops(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
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
        if lat is None:
            c = el.get("center", {})
            lat = c.get("lat")
            lon = c.get("lon")
        if lat is not None and lon is not None:
            coords.append((float(lat), float(lon)))
    return coords


def main() -> None:
    log.info("Loading Gold Standard stations...")
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    st = load_stations()
    dock = st[st["station_type"] == "docked_bike"].copy()
    log.info("  %d dock-based stations across %d cities",
             len(dock), dock["city"].nunique())

    # Select cities to query: 10 largest by station count (avoids tiny networks)
    by_city = dock.groupby("city").size().sort_values(ascending=False)
    # Skip non-city aggregations from data (Grand Est, FR, etc.)
    skip = {"France", "FR", "Grand Est", "Basque Country"}
    target_cities = [c for c in by_city.index if c not in skip][:10]
    log.info("Target cities (10 largest): %s", target_cities)

    rows = []
    for city in target_cities:
        sub = dock[dock["city"] == city]
        coords = sub[["lat", "lon"]].dropna().to_numpy()
        if len(coords) < 5:
            log.warning("  %s: only %d coords, skipping", city, len(coords))
            continue
        # GTFS reference
        gtfs_M = float(sub["gtfs_heavy_stops_300m"].mean())
        # OSM bbox + 1km buffer
        lat_buf = 1000.0 / 111000.0
        lon_buf = 1000.0 / (111000.0 * np.cos(np.radians(coords[:, 0].mean())))
        bbox = (coords[:, 0].min() - lat_buf,
                coords[:, 1].min() - lon_buf,
                coords[:, 0].max() + lat_buf,
                coords[:, 1].max() + lon_buf)
        log.info("  %s -- bbox = (%.3f, %.3f, %.3f, %.3f)",
                 city, bbox[0], bbox[1], bbox[2], bbox[3])
        time.sleep(1.5)
        stops = _query_overpass_heavy_stops(bbox)
        log.info("    %d OSM stops, %d stations", len(stops), len(coords))
        if stops:
            counts = (_haversine_m(coords, np.array(stops)) <= BUFFER_RADIUS_M).sum(axis=1)
            osm_M = float(counts.mean())
        else:
            osm_M = 0.0
        rows.append({
            "city": city,
            "n_stations": int(len(coords)),
            "M_gtfs": gtfs_M,
            "M_osm": osm_M,
            "ratio_osm_to_gtfs": (osm_M / gtfs_M) if gtfs_M > 0 else float("nan"),
            "n_osm_stops_in_bbox": int(len(stops)),
        })
        log.info(
            "    M_gtfs = %.2f, M_osm = %.2f, ratio = %s",
            gtfs_M, osm_M,
            f"{osm_M / gtfs_M:.2f}" if gtfs_M > 0 else "nan",
        )

    df = pd.DataFrame(rows)
    log.info("\nCalibration summary:")
    log.info("  panel-mean M_gtfs = %.2f", df["M_gtfs"].mean())
    log.info("  panel-mean M_osm  = %.2f", df["M_osm"].mean())
    log.info("  panel-median ratio (OSM / GTFS) = %.2f",
             df["ratio_osm_to_gtfs"].median())
    rho, p = sp_stats.spearmanr(df["M_gtfs"], df["M_osm"])
    log.info("  Spearman(M_gtfs, M_osm) = %+.3f  (p = %.3f)", rho, p)

    # Apply implied correction to E26 EU comparison
    eu_median_M_osm = 1.45  # from E26 results
    fr_median_M_gtfs = 0.22  # from E26 results
    panel_correction = df["ratio_osm_to_gtfs"].median()
    eu_corrected = (eu_median_M_osm / panel_correction
                    if panel_correction > 0 else eu_median_M_osm)
    ratio_corrected = eu_corrected / fr_median_M_gtfs
    log.info("\nE26 correction sweep:")
    log.info("  reported EU median M (OSM)       = %.2f", eu_median_M_osm)
    log.info("  reported FR median M (GTFS)      = %.2f", fr_median_M_gtfs)
    log.info("  reported EU / FR ratio (raw)     = %.2f", eu_median_M_osm / fr_median_M_gtfs)
    log.info("  median OSM/GTFS correction       = %.2f", panel_correction)
    log.info("  EU median M (corrected to GTFS)  = %.2f", eu_corrected)
    log.info("  EU / FR ratio after correction   = %.2f", ratio_corrected)

    results = {
        "per_city": df.to_dict("records"),
        "panel_summary": {
            "mean_M_gtfs": float(df["M_gtfs"].mean()),
            "mean_M_osm": float(df["M_osm"].mean()),
            "median_ratio_osm_gtfs": float(df["ratio_osm_to_gtfs"].median()),
            "p25_ratio": float(df["ratio_osm_to_gtfs"].quantile(0.25)),
            "p75_ratio": float(df["ratio_osm_to_gtfs"].quantile(0.75)),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
        },
        "e26_correction": {
            "eu_median_M_osm_raw": eu_median_M_osm,
            "fr_median_M_gtfs": fr_median_M_gtfs,
            "raw_ratio_eu_to_fr": eu_median_M_osm / fr_median_M_gtfs,
            "panel_correction_osm_to_gtfs": float(panel_correction),
            "eu_median_M_corrected": float(eu_corrected),
            "corrected_ratio_eu_to_fr": float(ratio_corrected),
        },
    }
    out_json = OUT_DIR / "e29_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: per-city scatter M_gtfs vs M_osm
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    ax = axes[0]
    ax.scatter(df["M_gtfs"], df["M_osm"], s=60, color="#1F3A6B",
               alpha=0.8, edgecolor="white", linewidth=0.5)
    for _, r in df.iterrows():
        ax.annotate(r["city"], (r["M_gtfs"], r["M_osm"]),
                    fontsize=8, color="#202020",
                    xytext=(4, 4), textcoords="offset points")
    # y = x line
    xmax = max(df["M_gtfs"].max(), df["M_osm"].max()) * 1.1
    ax.plot([0, xmax], [0, xmax], color="#A8201A",
            linestyle="--", linewidth=0.7, alpha=0.6, label="y = x")
    ax.text(0.02, 0.98,
            f"$\\rho_{{Sp}}$ = {rho:+.2f}\n"
            f"median ratio OSM/GTFS = {panel_correction:.2f}",
            transform=ax.transAxes, fontsize=8, color="#202020",
            ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                  "alpha": 0.9, "pad": 4})
    ax.set_xlabel("M (GTFS, Gold Standard)")
    ax.set_ylabel("M (OSM, same query as E26)")
    ax.set_title(f"OSM vs.\\ GTFS agreement, $n = {len(df)}$ FR cities",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)

    ax = axes[1]
    ax.barh(df.sort_values("ratio_osm_to_gtfs")["city"],
            df.sort_values("ratio_osm_to_gtfs")["ratio_osm_to_gtfs"],
            color="#1F3A6B", edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color="#404040", linewidth=0.6,
               linestyle="--", alpha=0.7)
    ax.axvline(panel_correction, color="#A8201A", linewidth=0.6,
               linestyle=":", alpha=0.7,
               label=f"panel median = {panel_correction:.2f}")
    ax.set_xlabel("Per-city ratio OSM / GTFS")
    ax.set_title("Calibration ratio per city", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)

    fig.suptitle("E29: OSM-vs-GTFS calibration on French panel "
                 "(robustness check for E26)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e29_osm_vs_gtfs.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e29_osm_vs_gtfs.pdf")


if __name__ == "__main__":
    main()
