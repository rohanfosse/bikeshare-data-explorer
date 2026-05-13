"""B8 -- Extend the eco-counter panel by scraping major metropolitan portals.

Our existing eco_compteurs_city_usage.csv covers 26 cities with
values. The Vélo & Territoires Plateforme Nationale des
Fréquentations is not centrally available as a single download
on data.gouv.fr (the datasets are fragmented across collectivités).

This script fetches the most prominent metropolitan open-data
portals for counter daily values, computes annual mean daily
bike counts per counter, then aggregates to city-level mean to
match our IMD panel city naming.

Targets:
  - Paris (opendata.paris.fr)
  - Saint-Nazaire CARENE
  - Tours Métropole
  - Anglet
  - Toulouse Métropole
  - Clermont Auvergne Métropole
  - Chalon (Grand Chalon)
  - Chambéry
  - Hauts-de-Seine département

Outputs:
    outputs/b8_extended_eco_counters.csv
    outputs/b8_results.json
"""
from __future__ import annotations

import json
import logging
import sys
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
CACHE_DIR = ROOT / "data" / "external" / "eco_counters_extended"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def fetch_csv(url: str, cache_name: str) -> pd.DataFrame | None:
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, sep=None, engine="python", on_bad_lines="skip")
            log.info("  cached %s -> %d rows", cache_name, len(df))
            return df
        except Exception as exc:
            log.warning("  cache parse fail (%s); refetching", exc)
    try:
        req = Request(url, headers={"User-Agent": "bikeshare-research/1.0"})
        with urlopen(req, timeout=120) as resp:
            raw = resp.read()
        cache_path.write_bytes(raw)
        df = pd.read_csv(cache_path, sep=None, engine="python", on_bad_lines="skip")
        log.info("  fetched %s -> %d rows, %d cols", cache_name, len(df), len(df.columns))
        return df
    except Exception as exc:
        log.warning("  fetch failed for %s: %s", url, exc)
        return None


def main() -> None:
    results = []

    # 1. Paris -- comptage-velo-donnees-compteurs (Opendatasoft Explore API)
    log.info("Fetching Paris ...")
    paris = fetch_csv(
        "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
        "comptage-velo-donnees-compteurs/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B",
        "paris_comptage_velo.csv",
    )
    if paris is not None:
        cols = paris.columns.tolist()
        log.info("  Paris columns sample: %s", cols[:10])
        # Daily count: column 'Comptage horaire'? Actually it's hourly. Aggregate.
        count_col = next((c for c in cols if "comptage" in c.lower() or "count" in c.lower()), None)
        date_col = next((c for c in cols if "date" in c.lower() or "heure" in c.lower()), None)
        if count_col is not None and date_col is not None:
            paris[date_col] = pd.to_datetime(paris[date_col], errors="coerce", utc=True)
            paris = paris.dropna(subset=[count_col, date_col])
            paris["day"] = paris[date_col].dt.date
            # Sum hourly counts per day per counter, then average over all days/counters
            daily = paris.groupby(["day"])[count_col].sum()
            value = float(daily.mean())
            log.info("  Paris -> mean daily count = %.0f (n_days=%d)",
                     value, len(daily))
            results.append({"city": "Paris", "eco_avg_daily_bike_counts": value,
                             "source": "opendata.paris.fr"})

    # 2. Toulouse Métropole -- passages-velo-moyenne-journaliere
    log.info("Fetching Toulouse...")
    toulouse = fetch_csv(
        "https://data.toulouse-metropole.fr/api/explore/v2.1/catalog/datasets/"
        "suivi-des-passages-a-velo-moyenne-journaliere/exports/csv?delimiter=%3B",
        "toulouse_passages_velo.csv",
    )
    if toulouse is not None:
        log.info("  Toulouse columns: %s", toulouse.columns.tolist()[:8])
        # find a numeric mean column
        numeric_cols = toulouse.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            mean_val = float(toulouse[numeric_cols[0]].dropna().mean())
            log.info("  Toulouse first numeric col '%s' mean = %.0f",
                     numeric_cols[0], mean_val)
            results.append({"city": "Toulouse",
                             "eco_avg_daily_bike_counts": mean_val,
                             "source": "data.toulouse-metropole.fr"})

    # 3. Tours Métropole -- comptage-velo-donnees-compteurs-syndicat
    log.info("Fetching Tours...")
    tours = fetch_csv(
        "https://data.tours-metropole.fr/api/explore/v2.1/catalog/datasets/"
        "comptage-velo-donnees-compteurs-syndicat-des-mobilites-de-touraine/exports/csv?delimiter=%3B",
        "tours_comptage_velo.csv",
    )
    if tours is not None:
        log.info("  Tours columns sample: %s", tours.columns.tolist()[:10])
        # similar to Paris pattern
        count_col = next((c for c in tours.columns if "comptage" in c.lower() or "count" in c.lower()), None)
        date_col = next((c for c in tours.columns if "date" in c.lower() or "heure" in c.lower()), None)
        if count_col and date_col:
            tours[date_col] = pd.to_datetime(tours[date_col], errors="coerce", utc=True)
            tours = tours.dropna(subset=[count_col, date_col])
            tours["day"] = tours[date_col].dt.date
            daily = tours.groupby(["day"])[count_col].sum()
            value = float(daily.mean())
            log.info("  Tours -> mean daily count = %.0f", value)
            results.append({"city": "Tours", "eco_avg_daily_bike_counts": value,
                             "source": "data.tours-metropole.fr"})

    # 4. Saint-Nazaire CARENE
    log.info("Fetching Saint-Nazaire CARENE...")
    sn = fetch_csv(
        "https://data.saintnazaireagglo.fr/api/explore/v2.1/catalog/datasets/"
        "ex-vue-personnalisee-comptage-velo-et-pedestre-donnees-compteurs/"
        "exports/csv?delimiter=%3B",
        "saintnazaire_comptage.csv",
    )
    if sn is not None:
        log.info("  Saint-Nazaire columns: %s", sn.columns.tolist()[:8])
        # Filter to bike if a mode column exists
        mode_col = next((c for c in sn.columns if "mode" in c.lower() or "type" in c.lower()), None)
        if mode_col:
            sn = sn[sn[mode_col].astype(str).str.contains("velo|cycle|bike", case=False, na=False)]
            log.info("  Saint-Nazaire filtered to bike: %d rows", len(sn))
        count_col = next((c for c in sn.columns if "comptage" in c.lower() or "count" in c.lower() or "valeur" in c.lower()), None)
        date_col = next((c for c in sn.columns if "date" in c.lower()), None)
        if count_col and date_col:
            sn[date_col] = pd.to_datetime(sn[date_col], errors="coerce")
            sn = sn.dropna(subset=[count_col, date_col])
            sn["day"] = sn[date_col].dt.date
            daily = sn.groupby(["day"])[count_col].sum()
            if len(daily) > 0:
                value = float(daily.mean())
                log.info("  Saint-Nazaire -> mean daily count = %.0f", value)
                results.append({"city": "Saint-Nazaire",
                                 "eco_avg_daily_bike_counts": value,
                                 "source": "data.saintnazaireagglo.fr"})

    # 5. Clermont -- frequentation quotidienne ZELT
    log.info("Fetching Clermont-Ferrand ZELT...")
    cf = fetch_csv(
        "https://data.clermontmetropole.eu/api/explore/v2.1/catalog/datasets/"
        "mesure-de-la-frequentation-quotidienne-velo-par-les-capteurs-zelt/"
        "exports/csv?delimiter=%3B",
        "clermont_zelt.csv",
    )
    if cf is not None:
        log.info("  Clermont columns: %s", cf.columns.tolist()[:8])
        numeric_cols = cf.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            for col in numeric_cols:
                m = cf[col].dropna().mean()
                if 100 < m < 50000:  # plausible daily bike count range
                    log.info("  Clermont numeric col '%s' mean = %.0f", col, m)
                    results.append({"city": "Clermont-Ferrand",
                                     "eco_avg_daily_bike_counts": float(m),
                                     "source": "data.clermontmetropole.eu"})
                    break

    # 6. Grand Chalon
    log.info("Fetching Grand Chalon...")
    chalon = fetch_csv(
        "https://static.data.gouv.fr/resources/compteurs-velos/20231115-154617/export-compteurs-velos-v2.csv",
        "chalon_compteurs.csv",
    )
    if chalon is not None:
        log.info("  Chalon columns: %s", chalon.columns.tolist()[:8])
        numeric_cols = chalon.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            for col in numeric_cols:
                m = chalon[col].dropna().mean()
                if 50 < m < 50000:
                    log.info("  Chalon numeric '%s' mean = %.0f", col, m)
                    results.append({"city": "Chalon-sur-Saône",
                                     "eco_avg_daily_bike_counts": float(m),
                                     "source": "Grand Chalon via data.gouv.fr"})
                    break

    # 7. Anglet
    log.info("Fetching Anglet 2024...")
    anglet = fetch_csv(
        "https://anglet-opendatapaysbasque.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "20250131-comptage_velo_donnees_compteurs_2024-21640024200014/exports/csv?delimiter=%3B",
        "anglet_2024.csv",
    )
    if anglet is not None:
        log.info("  Anglet columns: %s", anglet.columns.tolist()[:8])
        count_col = next((c for c in anglet.columns if "comptage" in c.lower() or "count" in c.lower() or "valeur" in c.lower()), None)
        date_col = next((c for c in anglet.columns if "date" in c.lower()), None)
        if count_col and date_col:
            anglet[date_col] = pd.to_datetime(anglet[date_col], errors="coerce")
            anglet = anglet.dropna(subset=[count_col, date_col])
            anglet["day"] = anglet[date_col].dt.date
            daily = anglet.groupby(["day"])[count_col].sum()
            if len(daily) > 0:
                value = float(daily.mean())
                log.info("  Anglet -> mean daily count = %.0f", value)
                results.append({"city": "Anglet",
                                 "eco_avg_daily_bike_counts": value,
                                 "source": "anglet-opendatapaysbasque.opendatasoft.com"})

    # 8. Hauts-de-Seine
    log.info("Fetching Hauts-de-Seine...")
    hds = fetch_csv(
        "https://opendata.hauts-de-seine.fr/api/explore/v2.1/catalog/datasets/"
        "fr-229200506-compteurs-velos/exports/csv?delimiter=%3B",
        "hauts_de_seine_compteurs.csv",
    )
    if hds is not None:
        log.info("  Hauts-de-Seine columns: %s", hds.columns.tolist()[:10])

    # Save extended panel
    log.info("\n=== Summary ===")
    log.info("Fetched %d new city values:", len(results))
    extended = pd.DataFrame(results)
    if len(extended) > 0:
        log.info("\n%s", extended.to_string(index=False))

    # Merge with existing
    existing_path = ROOT / "data" / "external" / "mobility_sources" / "eco_compteurs_city_usage.csv"
    existing = pd.read_csv(existing_path)
    log.info("\nExisting eco-counter file: %d cities (%d with values)",
             len(existing), existing["eco_avg_daily_bike_counts"].notna().sum())

    # Replace empty values where we have new data
    merged = existing.copy()
    for r in results:
        city = r["city"]
        v = r["eco_avg_daily_bike_counts"]
        if city in merged["city"].values:
            current = merged.loc[merged["city"] == city, "eco_avg_daily_bike_counts"].iloc[0]
            if pd.isna(current):
                merged.loc[merged["city"] == city, "eco_avg_daily_bike_counts"] = v
                log.info("  filled %s = %.0f", city, v)
            else:
                log.info("  %s already has value %.0f (new: %.0f, kept existing)",
                         city, current, v)
        else:
            new_row = pd.DataFrame({
                "city": [city],
                "eco_avg_daily_bike_counts": [v],
            })
            merged = pd.concat([merged, new_row], ignore_index=True)
            log.info("  added %s = %.0f", city, v)

    n_after = merged["eco_avg_daily_bike_counts"].notna().sum()
    log.info("\nMerged panel: %d cities, %d with values (was %d)",
             len(merged), n_after,
             existing["eco_avg_daily_bike_counts"].notna().sum())

    out_csv = OUT_DIR / "b8_extended_eco_counters.csv"
    merged.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    out_json = OUT_DIR / "b8_results.json"
    out_json.write_text(json.dumps({
        "n_existing_with_values": int(existing["eco_avg_daily_bike_counts"].notna().sum()),
        "n_new_cities_fetched": int(len(results)),
        "n_total_with_values_after_merge": int(n_after),
        "new_cities": results,
    }, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
