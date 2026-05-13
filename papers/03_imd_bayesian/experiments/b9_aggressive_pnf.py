"""B9 -- Aggressive PNF / eco-counter aggregation across French metropoles.

Probes ~20 metropolitan Opendatasoft portals + the federated
Opendatasoft network for "comptage velo" / "frequentation"
datasets, fetches them, aggregates to a per-city mean daily count
that is comparable to the existing eco_compteurs_city_usage.csv
scale (counter-mean, not city-total).

Aggregation rule (uniform across portals):
  For each counter (identifiant_compteur), compute the mean
  daily total (sum of hourly/15-min counts per day, then mean
  across days). Then take the mean across counters in the city.

This gives a "typical counter daily volume" interpretable in
the same units as the existing file.

Outputs:
    outputs/b9_aggregated_eco_counters.csv
    outputs/b9_results.json
"""
from __future__ import annotations

import json
import logging
import sys
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [HERE, *HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR = ROOT / "data" / "external" / "pnf_aggregate"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Federated Opendatasoft search
# ---------------------------------------------------------------------------

def federated_search(query: str, n_per_page: int = 50) -> list[dict]:
    """Query the Opendatasoft federated catalogue for datasets matching.

    Public endpoint: https://public.opendatasoft.com/api/explore/v2.1/
    """
    url = (
        f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets"
        f"?where=search%28dataset_id%2C%20'{query}'%29&limit={n_per_page}"
    )
    req = Request(url, headers={"User-Agent": "bikeshare-research/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", [])
    except Exception as exc:
        log.warning("  federated search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Per-portal fetch
# ---------------------------------------------------------------------------

PORTALS = [
    # (portal_host, dataset_id_pattern, city_label)
    ("opendata.paris.fr", "comptage-velo-donnees-compteurs", "Paris"),
    ("data.tours-metropole.fr", "comptage-velo-donnees-compteurs-syndicat-des-mobilites-de-touraine", "Tours"),
    ("data.saintnazaireagglo.fr", "ex-vue-personnalisee-comptage-velo-et-pedestre-donnees-compteurs", "Saint-Nazaire"),
    ("opendata.bordeaux-metropole.fr", "bm_compt_velo", "Bordeaux"),
    ("data.metropolegrenoble.fr", "comptages-velo-donnees-compteurs-grenoble-alpes-metropole", "Grenoble"),
    ("data.angersloiremetropole.fr", "compteur-velos-quotidien", "Angers"),
    ("data.brest-metropole.fr", "comptage-velo", "Brest"),
    ("opendata.lillemetropole.fr", "comptages-velos-permanents-de-la-mel", "Lille"),
    ("data.metropole-rouen-normandie.fr", "trafic-velo-mrn", "Rouen"),
    ("data.nantesmetropole.fr", "244400404_comptage-velo-nantes-metropole", "Nantes"),
    ("data.strasbourg.eu", "lignes-comptage-velo-cts-frequentation", "Strasbourg"),
    ("opendata.lyon.fr", "trafic-velo-comptages-permanents-grand-lyon", "Lyon"),
    ("data.toulouse-metropole.fr", "passages-velos-en-volume-mensuel", "Toulouse"),
    ("anglet-opendatapaysbasque.opendatasoft.com",
        "20231011-comptage_velo_donnees_compteurs_2023-21640024200014", "Anglet"),
    ("data.clermontmetropole.eu",
        "mesure-de-la-frequentation-quotidienne-velo-par-les-capteurs-zelt", "Clermont-Ferrand"),
    ("opendata.hauts-de-seine.fr", "fr-229200506-compteurs-velos", "Hauts-de-Seine"),
    ("data.grandchambery.fr", "compteurs-velo", "Chambéry"),
    ("opendata.caenlamer.fr", "compteurs-velo", "Caen"),
    ("data.rennesmetropole.fr", "compteurs-velo-permanents", "Rennes"),
    ("data.mulhouse-alsace.fr", "compteurs-velo", "Mulhouse"),
    ("data.bordeaux-metropole.fr", "comptages-velos", "Bordeaux"),
    ("data.bm.bordeaux.fr", "bm_compteur_velo", "Bordeaux"),
]


def fetch_with_cache(url: str, cache_name: str) -> bytes | None:
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 100:
        return cache_path.read_bytes()
    try:
        req = Request(url, headers={"User-Agent": "bikeshare-research/1.0"})
        with urlopen(req, timeout=120) as resp:
            raw = resp.read()
        cache_path.write_bytes(raw)
        return raw
    except Exception as exc:
        log.warning("    fetch failed: %s", exc)
        return None


def find_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Locate (date, count, counter_id) columns by name fuzzy match."""
    date_col = None; count_col = None; counter_col = None
    cols_lower = {c: c.lower() for c in df.columns}
    for c, cl in cols_lower.items():
        if date_col is None and ("date" in cl or "heure" in cl or "horodate" in cl or "timestamp" in cl):
            date_col = c
        if count_col is None and (
            "comptage" in cl or "counts" in cl or "passages" in cl
            or "nb_velo" in cl or "valeur" in cl or "volume" in cl
            or "frequent" in cl
        ):
            # Skip identifiant columns
            if "id" in cl or "identifiant" in cl:
                continue
            count_col = c
        if counter_col is None and ("id_compteur" in cl or "id_counter" in cl
                                     or "identifiant_compteur" in cl or "id du compteur" in cl
                                     or "site" in cl or "nom_compteur" in cl):
            counter_col = c
    return date_col, count_col, counter_col


def aggregate_panel(df: pd.DataFrame, label: str) -> float | None:
    """Aggregate to a counter-mean daily count.

    1. Find (date, count, counter_id) columns
    2. Parse date
    3. Sum counts per (counter, day)
    4. Mean daily count per counter
    5. Mean across counters
    """
    date_col, count_col, counter_col = find_columns(df)
    log.info("    columns: date=%s, count=%s, counter=%s",
             date_col, count_col, counter_col)
    if date_col is None or count_col is None:
        return None
    try:
        # Make a copy to avoid SettingWithCopy
        df = df[[c for c in [date_col, count_col, counter_col] if c is not None]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
        df = df.dropna(subset=[date_col, count_col])
        if len(df) < 100:
            log.warning("    only %d rows after cleaning", len(df))
            return None
        df["day"] = df[date_col].dt.date
        if counter_col is None:
            # Treat as single counter
            daily = df.groupby("day")[count_col].sum()
            return float(daily.mean())
        # Per-counter daily total, then mean across counters
        per_counter_day = df.groupby([counter_col, "day"])[count_col].sum()
        per_counter_mean_daily = per_counter_day.groupby(level=0).mean()
        return float(per_counter_mean_daily.mean())
    except Exception as exc:
        log.warning("    aggregation failed: %s", exc)
        return None


def main() -> None:
    results = []

    for host, slug, city in PORTALS:
        log.info("== %s (%s) ==", city, host)
        # Opendatasoft v2 CSV export
        url = (
            f"https://{host}/api/explore/v2.1/catalog/datasets/"
            f"{slug}/exports/csv?delimiter=%3B"
        )
        raw = fetch_with_cache(url, f"{host}__{slug}.csv")
        if raw is None or len(raw) < 200:
            continue
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")
        try:
            df = pd.read_csv(StringIO(text), sep=";", on_bad_lines="skip",
                              low_memory=False)
        except Exception:
            try:
                df = pd.read_csv(StringIO(text), sep=",", on_bad_lines="skip",
                                  low_memory=False)
            except Exception as exc:
                log.warning("  parse failed: %s", exc)
                continue
        log.info("  fetched: %d rows, %d cols", len(df), len(df.columns))
        if len(df) < 100:
            continue
        value = aggregate_panel(df, city)
        if value is None:
            continue
        if not (50 < value < 50000):
            log.warning("  aggregated value %.0f outside plausible range; skipping",
                         value)
            continue
        log.info("  %s -> counter-mean daily count = %.0f", city, value)
        results.append({
            "city": city,
            "eco_avg_daily_bike_counts": float(value),
            "source": f"{host} / {slug}",
            "n_rows": int(len(df)),
        })

    # Summary
    log.info("\n=== Summary ===")
    log.info("Successfully fetched and aggregated %d cities:", len(results))
    extended = pd.DataFrame(results)
    if not extended.empty:
        log.info("\n%s", extended[["city", "eco_avg_daily_bike_counts", "n_rows"]].to_string(index=False))

    # Merge with existing
    existing_path = ROOT / "data" / "external" / "mobility_sources" / "eco_compteurs_city_usage.csv"
    existing = pd.read_csv(existing_path)
    n_existing = existing["eco_avg_daily_bike_counts"].notna().sum()
    log.info("\nExisting eco-counter file: %d cities (%d with values)",
             len(existing), n_existing)

    merged = existing.copy()
    n_filled = 0
    n_added = 0
    for r in results:
        city = r["city"]
        v = r["eco_avg_daily_bike_counts"]
        existing_match = merged["city"] == city
        if existing_match.any():
            current = merged.loc[existing_match, "eco_avg_daily_bike_counts"].iloc[0]
            if pd.isna(current):
                merged.loc[existing_match, "eco_avg_daily_bike_counts"] = v
                n_filled += 1
                log.info("  filled %s = %.0f", city, v)
            else:
                ratio = v / current
                log.info("  %s already %.0f (PNF: %.0f, ratio %.2fx; kept existing)",
                         city, current, v, ratio)
        else:
            merged = pd.concat([
                merged,
                pd.DataFrame({"city": [city], "eco_avg_daily_bike_counts": [v]}),
            ], ignore_index=True)
            n_added += 1
            log.info("  added %s = %.0f", city, v)

    n_after = merged["eco_avg_daily_bike_counts"].notna().sum()
    log.info("\nFinal panel: %d rows, %d with values "
             "(was %d, +%d filled empty, +%d new)",
             len(merged), n_after, n_existing, n_filled, n_added)

    out_csv = OUT_DIR / "b9_aggregated_eco_counters.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # Save a manifest of what we tried
    out_json = OUT_DIR / "b9_results.json"
    out_json.write_text(json.dumps({
        "n_portals_attempted": len(PORTALS),
        "n_successful": len(results),
        "n_existing_before": int(n_existing),
        "n_filled_empty": int(n_filled),
        "n_new_added": int(n_added),
        "n_total_after": int(n_after),
        "results": results,
    }, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
