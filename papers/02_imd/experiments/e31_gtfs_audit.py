"""E31 -- Systematic GTFS audit of the French panel.

E29 established the OSM-vs-GTFS calibration on 10 cities. E30
quantified the impact of the A6 patch on Lyon/Marseille/Lille.
We now extend the audit to the entire panel and propose a
five-class GTFS anomaly taxonomy G1-G5, parallel to the
station-level GBFS taxonomy A1-A5 of \\citet{Fosse2026gold}.

The approach is the Gold Standard six-step protocol applied
to the GTFS layer instead of the GBFS layer:
  1. Inventory: every panel city has a GTFS-derived
     gtfs_heavy_stops_300m mean from Module 4.
  2. Triangulation: re-query OSM heavy-transit stops in the
     city's station bounding box (same query as E26/E29).
  3. Computation: per-city ratio M_OSM / M_GTFS.
  4. Classification: assign each city to one of G1-G5.
  5. Audit log: per-city verdict + recommended action.
  6. Patch protocol: M = max(GTFS, OSM) for G1-G2 cities.

Taxonomy:
  G1  Severe GTFS under-coverage     (OSM/GTFS > 5 OR
                                      GTFS = 0 with OSM > 0)
  G2  Moderate GTFS under-coverage   (2 < OSM/GTFS <= 5)
  G3  Well-calibrated                (0.5 <= OSM/GTFS <= 2)
  G4  GTFS over-coverage             (OSM/GTFS < 0.5)
  G5  Empty network on both feeds    (GTFS = 0 AND OSM = 0)

Outputs:
    outputs/e31_results.json
    outputs/e31_audit_log.pdf
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

from _common import ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BUFFER_RADIUS_M = 300.0

G_CLASSES = {
    "G1": ("Severe GTFS under-coverage", "OSM/GTFS > 5 or GTFS = 0 with OSM > 0", "#A8201A"),
    "G2": ("Moderate GTFS under-coverage", "2 < OSM/GTFS <= 5", "#D08020"),
    "G3": ("Well-calibrated", "0.5 <= OSM/GTFS <= 2", "#5B7E4F"),
    "G4": ("GTFS over-coverage", "OSM/GTFS < 0.5", "#7095C8"),
    "G5": ("Empty on both feeds", "GTFS = 0 and OSM = 0", "#9A9A9A"),
}


def _haversine_m(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1[:, None]) * np.cos(lat2[None, :]) * np.sin(dlon / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def _query_overpass(bbox: tuple[float, float, float, float],
                    max_attempts: int = 2) -> list[tuple[float, float]]:
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
    for attempt in range(max_attempts):
        try:
            req = Request(OVERPASS_URL, data=data,
                          headers={"User-Agent": "bikeshare-research/1.0"})
            with urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            coords = []
            for el in payload.get("elements", []):
                lat = el.get("lat")
                lon = el.get("lon")
                if lat is None:
                    c = el.get("center", {})
                    lat = c.get("lat"); lon = c.get("lon")
                if lat is not None and lon is not None:
                    coords.append((float(lat), float(lon)))
            return coords
        except Exception as exc:
            log.warning("  Overpass attempt %d failed: %s", attempt + 1, exc)
            time.sleep(3.0 * (attempt + 1))
    return []


def _classify(m_gtfs: float, m_osm: float) -> str:
    if m_gtfs == 0 and m_osm == 0:
        return "G5"
    if m_gtfs == 0 and m_osm > 0:
        return "G1"
    ratio = m_osm / m_gtfs
    if ratio > 5.0:
        return "G1"
    if ratio > 2.0:
        return "G2"
    if ratio >= 0.5:
        return "G3"
    return "G4"


def _recommended_action(g_class: str) -> str:
    return {
        "G1": "Triangulate: take M = max(GTFS, OSM); flag operator for AOM contact",
        "G2": "Triangulate: take M = max(GTFS, OSM)",
        "G3": "No action needed",
        "G4": "Audit OSM tagging completeness; GTFS authoritative",
        "G5": "Verify that the city actually has heavy transit; possible peripheral position",
    }[g_class]


def main() -> None:
    log.info("Loading panel from Gold Standard...")
    panel = load_panel()
    log.info("  n = %d cities", panel.n)

    # Load station coordinates for bbox
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    st = load_stations()
    dock = st[st["station_type"] == "docked_bike"].copy()
    raw_M_gtfs = dock.groupby("city")["gtfs_heavy_stops_300m"].mean()

    log.info("Running Overpass query per panel city...")
    rows = []
    for i, city in enumerate(panel.cities, 1):
        sub = dock[dock["city"] == city]
        if len(sub) == 0:
            continue
        coords = sub[["lat", "lon"]].dropna().to_numpy()
        if len(coords) < 3:
            continue
        m_gtfs = float(raw_M_gtfs.get(city, 0.0))
        # bbox + 1km buffer
        lat_buf = 1000.0 / 111000.0
        lon_buf = 1000.0 / (111000.0 * np.cos(np.radians(coords[:, 0].mean())))
        bbox = (coords[:, 0].min() - lat_buf,
                coords[:, 1].min() - lon_buf,
                coords[:, 0].max() + lat_buf,
                coords[:, 1].max() + lon_buf)
        # bbox diagonal -- if absurd, the city has bad geocoding
        diag_km = ((bbox[2]-bbox[0])*111 + (bbox[3]-bbox[1])*78) / 2
        if diag_km > 200:
            log.warning("  %s (n=%d): bbox diagonal %.0f km > 200, skipping",
                        city, len(coords), diag_km)
            rows.append({
                "city": city, "n_stations": len(coords),
                "M_gtfs": m_gtfs, "M_osm": float("nan"),
                "ratio_osm_to_gtfs": float("nan"),
                "g_class": "BBOX_FAIL",
                "n_osm_stops_in_bbox": 0,
            })
            continue
        time.sleep(1.5)
        log.info("  [%2d/%d] %-20s  bbox_diag=%.0fkm  n=%4d", i,
                 panel.n, city, diag_km, len(coords))
        stops = _query_overpass(bbox)
        if stops:
            counts = (_haversine_m(coords, np.array(stops)) <= BUFFER_RADIUS_M).sum(axis=1)
            m_osm = float(counts.mean())
        else:
            m_osm = 0.0
        g_class = _classify(m_gtfs, m_osm)
        rows.append({
            "city": city,
            "n_stations": int(len(coords)),
            "M_gtfs": m_gtfs,
            "M_osm": m_osm,
            "ratio_osm_to_gtfs": (m_osm / m_gtfs) if m_gtfs > 0 else (float("inf") if m_osm > 0 else 0.0),
            "g_class": g_class,
            "n_osm_stops_in_bbox": int(len(stops)),
        })

    df = pd.DataFrame(rows)
    df = df[df["g_class"] != "BBOX_FAIL"].copy()
    log.info("\nClassification panel-wide:")
    counts = df["g_class"].value_counts().reindex(["G1", "G2", "G3", "G4", "G5"])
    for g, n in counts.items():
        n = 0 if pd.isna(n) else int(n)
        log.info("  %s (%s)  n = %2d", g, G_CLASSES[g][0], n)

    # Audit log: top G1 cities (most severe under-coverage)
    g1 = df[df["g_class"] == "G1"].sort_values("ratio_osm_to_gtfs", ascending=False)
    log.info("\nG1 incidence detail (severe under-coverage):")
    for _, r in g1.iterrows():
        log.info("  %-20s  M_gtfs=%.2f  M_osm=%.2f  ratio=%s",
                 r["city"], r["M_gtfs"], r["M_osm"],
                 f"{r['ratio_osm_to_gtfs']:.1f}" if np.isfinite(r["ratio_osm_to_gtfs"]) else "inf")

    g2 = df[df["g_class"] == "G2"]
    log.info("\nG2 incidence (moderate under-coverage), n=%d:", len(g2))
    for _, r in g2.iterrows():
        log.info("  %-20s  M_gtfs=%.2f  M_osm=%.2f  ratio=%.2f",
                 r["city"], r["M_gtfs"], r["M_osm"], r["ratio_osm_to_gtfs"])

    g4 = df[df["g_class"] == "G4"]
    log.info("\nG4 incidence (OSM under-counts GTFS), n=%d:", len(g4))
    for _, r in g4.iterrows():
        log.info("  %-20s  M_gtfs=%.2f  M_osm=%.2f  ratio=%.2f",
                 r["city"], r["M_gtfs"], r["M_osm"], r["ratio_osm_to_gtfs"])

    results = {
        "n_panel": int(panel.n),
        "n_audited": int(len(df)),
        "taxonomy": {g: {"description": G_CLASSES[g][0],
                         "signature": G_CLASSES[g][1]}
                     for g in ["G1", "G2", "G3", "G4", "G5"]},
        "class_counts": counts.fillna(0).astype(int).to_dict(),
        "audit_log": [
            {"city": r["city"],
             "n_stations": int(r["n_stations"]),
             "M_gtfs": r["M_gtfs"],
             "M_osm": r["M_osm"],
             "ratio": r["ratio_osm_to_gtfs"] if np.isfinite(r["ratio_osm_to_gtfs"]) else "inf",
             "g_class": r["g_class"],
             "action": _recommended_action(r["g_class"])}
            for _, r in df.sort_values("g_class").iterrows()
        ],
    }
    out_json = OUT_DIR / "e31_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: bar chart of class counts + per-city ratio scatter
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # Left: class counts
    ax = axes[0]
    classes_present = [g for g in ["G1", "G2", "G3", "G4", "G5"]
                       if counts.get(g, 0) > 0]
    counts_present = [int(counts[g]) for g in classes_present]
    colors = [G_CLASSES[g][2] for g in classes_present]
    bars = ax.bar(classes_present, counts_present, color=colors,
                  edgecolor="white", linewidth=0.4)
    for bar, n in zip(bars, counts_present):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"n={n}",
                ha="center", va="bottom", fontsize=9,
                color="#202020", fontweight="bold")
    ax.set_ylabel("Cities")
    ax.set_title(f"GTFS audit class distribution "
                 f"(n = {len(df)} cities)", fontsize=10)
    ax.set_ylim(0, max(counts_present) * 1.18)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)

    # Right: per-city scatter M_gtfs vs M_osm (log-log)
    ax = axes[1]
    for g in classes_present:
        sub = df[df["g_class"] == g]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["M_gtfs"].clip(lower=0.01),
            sub["M_osm"].clip(lower=0.01),
            s=40, color=G_CLASSES[g][2], alpha=0.85,
            edgecolor="white", linewidth=0.3,
            label=f"{g} ({len(sub)})",
        )
    # Label G1 & G2 cities
    for _, r in df[df["g_class"].isin(["G1", "G2"])].iterrows():
        ax.annotate(r["city"],
                    (max(r["M_gtfs"], 0.01), max(r["M_osm"], 0.01)),
                    fontsize=6.5, color="#202020",
                    xytext=(3, 3), textcoords="offset points")
    xmax = max(df["M_gtfs"].max(), df["M_osm"].max()) * 1.5
    ax.plot([0.01, xmax], [0.01, xmax], "k--",
            linewidth=0.6, alpha=0.5, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$ GTFS (Gold Standard module 4)")
    ax.set_ylabel(r"$M$ OSM (Overpass tag query)")
    ax.set_title("Per-city OSM vs.\\ GTFS multimodality", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, which="both", color="#E5E5E5", linewidth=0.5)

    fig.suptitle("E31: systematic GTFS audit of the IMD panel "
                 "(Gold Standard methodology applied to module 4)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e31_audit_log.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e31_audit_log.pdf")


if __name__ == "__main__":
    main()
