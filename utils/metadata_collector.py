"""
metadata_collector.py — Collecteur GBFS des feeds de métadonnées.

Complète gbfs_collector.py (station_status, temps réel) et
vehicle_collector.py (free_bike_status, temps réel). Ici on capte les
feeds à cadence lente, utiles pour enrichir toutes les expériences :

  Tier 1 — statiques (fetch écrasé, dernière valeur) :
    system_information     opérateur, fuseau, langue, nom du réseau
    vehicle_types          méca / élec / cargo + autonomie (split OD)
    system_pricing_plans   tarifs (xps élasticité / demande)
    system_regions         découpage interne du réseau

  Tier 2 — semi-dynamiques (snapshots datés, 1×/jour) :
    geofencing_zones       zones d'usage + no-park (interpréter le free-floating)
    system_alerts          pannes / interruptions (horodate les trous de collecte)

Sortie : data/system_metadata/<system>/...
  - statiques : <feed>.parquet            (écrasé à chaque passage)
  - datés     : <feed>/<YYYY-MM-DD>.parquet

Conçu pour tourner 1×/jour (cron). Ré-écrit les statiques au passage
(coût négligeable) afin de capter les changements éventuels.

Usage CLI : scripts/collect_metadata.py
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.gbfs_collector import (
    PRIORITY_SYSTEMS,
    _ROOT,
    _TIMEOUT,
    _discover_feed_url,
    _get_session,
)

_META_DIR = _ROOT / "data" / "system_metadata"

log = logging.getLogger("metadata_collector")

# Feeds statiques (écrasés) et datés (snapshots horodatés).
STATIC_FEEDS = ("system_information", "vehicle_types",
                "system_pricing_plans", "system_regions")
DATED_FEEDS  = ("geofencing_zones", "system_alerts")
ALL_FEEDS    = STATIC_FEEDS + DATED_FEEDS

# Clé de la liste d'enregistrements sous data.* pour chaque feed.
# system_information est un objet (pas une liste) -> traité à part.
_LIST_KEY = {
    "vehicle_types":        "vehicle_types",
    "system_pricing_plans": "plans",
    "system_regions":       "regions",
    "system_alerts":        "alerts",
    "geofencing_zones":     "geofencing_zones",   # FeatureCollection -> .features
}


def _fetch_json(url: str, timeout: int = _TIMEOUT) -> dict | None:
    try:
        r = _get_session().get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("Erreur feed %s : %s", url, exc)
        return None


def _extract_records(doc: dict, feed_name: str) -> list[dict]:
    """Extrait la liste d'enregistrements d'un feed GBFS (v2 ou v3).

    Gère l'imbrication par langue (data.<lang>.<key>) des feeds v2.
    """
    data = doc.get("data", {})
    if not isinstance(data, dict):
        return []

    if feed_name == "system_information":
        if data.get("system_id") or data.get("name"):
            return [data]
        for v in data.values():
            if isinstance(v, dict) and (v.get("system_id") or v.get("name")):
                return [v]
        return [data] if data else []

    key = _LIST_KEY[feed_name]
    container = data.get(key)
    if container is None:
        for v in data.values():
            if isinstance(v, dict) and key in v:
                container = v[key]
                break
    if container is None:
        return []

    if feed_name == "geofencing_zones":
        # GeoJSON FeatureCollection -> liste de features.
        if isinstance(container, dict):
            return container.get("features", [])
        return container if isinstance(container, list) else []

    return container if isinstance(container, list) else []


def _to_frame(records: list[dict], fetched_at: datetime, system_id: str) -> pd.DataFrame:
    """Aplatit en DataFrame, en sérialisant les colonnes imbriquées en JSON.

    Parquet n'aime pas les colonnes contenant des dict/list hétérogènes
    (geometry, rules, times...) : on les passe en chaîne JSON pour ne
    rien perdre tout en restant colonne-stable.
    """
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
            )
    # Certains feeds embarquent déjà system_id / fetched_at : on les
    # renomme pour ne pas écraser nos colonnes d'indexation.
    df = df.rename(columns={
        "system_id":  "feed_system_id",
        "fetched_at": "feed_fetched_at",
    })
    df.insert(0, "system_id", system_id)
    df.insert(0, "fetched_at", fetched_at)
    return df


class MetadataCollector:
    """Collecteur des feeds de métadonnées (Tier 1 statiques + Tier 2 datés)."""

    def __init__(
        self,
        system_ids:  list[str] | None = None,
        timeout:     int = _TIMEOUT,
        out_dir:     Path | None = None,
        max_workers: int = 12,
    ) -> None:
        self.system_ids  = system_ids or PRIORITY_SYSTEMS
        self.timeout     = timeout
        self.out_dir     = Path(out_dir) if out_dir is not None else _META_DIR
        self.max_workers = max_workers
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._catalog = self._load_catalog()

    def _load_catalog(self) -> pd.DataFrame:
        cat = pd.read_csv(_ROOT / "data" / "gbfs_france" / "systems_catalog.csv")
        return cat[cat["status"] == "ok"].copy()

    def get_target_systems(self) -> pd.DataFrame:
        mask = self._catalog["system_id"].isin(self.system_ids)
        return self._catalog[mask].reset_index(drop=True)

    def _collect_one(self, row: pd.Series, today: str) -> tuple[str, dict[str, int]]:
        sid, gbfs = str(row["system_id"]), str(row["gbfs_url"])
        counts: dict[str, int] = {}
        now = datetime.now(timezone.utc)
        for feed in ALL_FEEDS:
            url = _discover_feed_url(gbfs, feed, self.timeout)
            if not url:
                continue
            doc = _fetch_json(url, self.timeout)
            if not doc:
                continue
            df = _to_frame(_extract_records(doc, feed), now, sid)
            if df.empty:
                continue
            if feed in STATIC_FEEDS:
                out = self.out_dir / sid / f"{feed}.parquet"
            else:
                out = self.out_dir / sid / feed / f"{today}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            counts[feed] = len(df)
        return sid, counts

    def collect_all(self) -> dict[str, dict[str, int]]:
        targets = self.get_target_systems()
        today = datetime.now(timezone.utc).date().isoformat()
        results: dict[str, dict[str, int]] = {}
        n_workers = min(self.max_workers, max(len(targets), 1))
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="meta") as pool:
            futures = {pool.submit(self._collect_one, row, today): str(row["system_id"])
                       for _, row in targets.iterrows()}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    sid, counts = future.result()
                    results[sid] = counts
                    if counts:
                        log.info("[%s] %s", sid,
                                 ", ".join(f"{k}={v}" for k, v in counts.items()))
                except Exception as exc:
                    log.warning("[%s] Erreur : %s", sid, exc)
                    results[sid] = {}
        return results
