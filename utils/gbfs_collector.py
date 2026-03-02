"""
gbfs_collector.py — Collecteur de données GBFS station_status pour les grandes villes françaises.

Principe :
  1. Lit le catalogue des systèmes (systems_catalog.csv).
  2. Pour chaque système, découvre dynamiquement l'URL station_status via gbfs.json.
  3. Prend des snapshots à intervalle régulier et les stocke en Parquet.
  4. Calcule les pseudo-flux (∆bikes) entre snapshots consécutifs.

Usage en module :
    from utils.gbfs_collector import GBFSCollector
    coll = GBFSCollector()
    coll.run(interval_sec=60, max_duration_sec=3600)

Usage en CLI via scripts/collect_status.py.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_ROOT       = Path(__file__).parent.parent
_CATALOG    = _ROOT / "data" / "gbfs_france" / "systems_catalog.csv"
_SNAP_DIR   = _ROOT / "data" / "status_snapshots"
_TIMEOUT    = 15       # secondes par requête HTTP
_FF_KEYWORDS = {"bird", "dott", "pony", "voi", "tier", "lime", "bolt", "wind"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gbfs_collector")


# ── Cibles prioritaires (dock-based, grandes villes) ──────────────────────────
PRIORITY_SYSTEMS: list[str] = [
    "Paris",              # Vélib' Metropole
    "lyon",               # Vélo'v
    "toulouse",           # VélôToulouse
    "velo-tbm-bordeaux",  # TBM Bordeaux
    "v_lille",            # V'Lille
    "levelo_inurba_marseille",  # Le Vélo Marseille
    "le_velo_star",       # LE vélo STAR Rennes
    "nantes",             # Bicloo Nantes
    "nextbike_af",        # Mulhouse
    "montpellier",        # Vélomagg
    "inurba-rouen",       # Lovelo Rouen
    "dijon",              # DiVia Dijon
    "nancy",              # Vélostan'lib Nancy
    "amiens",             # Vélo'ic Amiens
    "cergy",              # Cy'clic Cergy
    "velonecy60minutes_annecy",  # Vélo'necy
    "zebullo",            # Zebullo Reims
    "besancon",           # VéloCité Besançon
    "velozef",            # Vélo Brest
    "velivert_saint_etienne",    # Vélivert Saint-Étienne
]


# ── Parsers GBFS v2 et v3 ────────────────────────────────────────────────────

def _discover_feed_url(gbfs_url: str, feed_name: str, timeout: int = _TIMEOUT) -> str | None:
    """
    Interroge gbfs.json et retourne l'URL du feed demandé (ex: 'station_status').
    Compatible GBFS v2.x et v3.x.
    """
    try:
        r = requests.get(gbfs_url, timeout=timeout)
        r.raise_for_status()
        doc = r.json()
    except Exception as exc:
        log.warning("Impossible de charger %s : %s", gbfs_url, exc)
        return None

    # v3 : data.feeds est un tableau plat
    feeds_v3 = doc.get("data", {}).get("feeds", [])
    if feeds_v3:
        for feed in feeds_v3:
            if feed.get("name") == feed_name:
                return feed.get("url")

    # v2 : data.<lang>.feeds
    data = doc.get("data", {})
    for lang_key, lang_val in data.items():
        if isinstance(lang_val, dict):
            for feed in lang_val.get("feeds", []):
                if feed.get("name") == feed_name:
                    return feed.get("url")

    return None


def _fetch_station_status(status_url: str, timeout: int = _TIMEOUT) -> list[dict]:
    """
    Télécharge station_status.json et retourne la liste des stations.
    Compatible GBFS v2 et v3.
    """
    try:
        r = requests.get(status_url, timeout=timeout)
        r.raise_for_status()
        doc = r.json()
    except Exception as exc:
        log.warning("Erreur station_status %s : %s", status_url, exc)
        return []

    data = doc.get("data", {})

    # v3 : data.stations
    if "stations" in data:
        return data["stations"]

    # v2 : data.<lang>.stations ou data.stations
    for val in data.values():
        if isinstance(val, dict) and "stations" in val:
            return val["stations"]

    return []


def _fetch_station_info(info_url: str, timeout: int = _TIMEOUT) -> pd.DataFrame:
    """
    Télécharge station_information.json et retourne un DataFrame (id, name, lat, lon, capacity).
    """
    try:
        r = requests.get(info_url, timeout=timeout)
        r.raise_for_status()
        doc = r.json()
    except Exception as exc:
        log.warning("Erreur station_information %s : %s", info_url, exc)
        return pd.DataFrame()

    data = doc.get("data", {})
    stations = []
    if "stations" in data:
        stations = data["stations"]
    else:
        for val in data.values():
            if isinstance(val, dict) and "stations" in val:
                stations = val["stations"]
                break

    if not stations:
        return pd.DataFrame()

    rows = []
    for s in stations:
        rows.append({
            "station_id": str(s.get("station_id", "")),
            "name":       str(s.get("name", s.get("station_name", ""))),
            "lat":        float(s.get("lat", 0.0)),
            "lon":        float(s.get("lon", 0.0)),
            "capacity":   int(s.get("capacity", 0)),
        })
    return pd.DataFrame(rows)


def _parse_status_snapshot(
    stations_raw: list[dict],
    fetched_at: datetime,
    system_id: str,
) -> pd.DataFrame:
    """
    Transforme la liste brute de station_status en DataFrame normalisé.
    """
    rows = []
    for s in stations_raw:
        sid = str(s.get("station_id", ""))
        bikes = s.get("num_bikes_available", s.get("num_vehicles_available", 0))
        docks = s.get("num_docks_available", 0)
        is_renting   = bool(s.get("is_renting", True))
        is_returning = bool(s.get("is_returning", True))
        rows.append({
            "fetched_at":           fetched_at,
            "system_id":            system_id,
            "station_id":           sid,
            "num_bikes_available":  int(bikes) if bikes is not None else 0,
            "num_docks_available":  int(docks) if docks is not None else 0,
            "is_renting":           is_renting,
            "is_returning":         is_returning,
        })
    return pd.DataFrame(rows)


def compute_pseudo_flows(snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les pseudo-flux entre snapshots consécutifs.

    Logique :
      ∆bikes = bikes(t) - bikes(t-1)
      ∆bikes < 0 → |∆| départs nets estimés
      ∆bikes > 0 → ∆ arrivées nettes estimées

    Cette estimation est un minorant car elle ignore les mouvements simultanés
    d'arrivées et de départs dans le même intervalle.

    Paramètre
    ---------
    snapshots : DataFrame avec colonnes fetched_at, station_id, num_bikes_available, ...
                trié par station_id puis fetched_at

    Retourne
    --------
    DataFrame avec colonnes : fetched_at, station_id, delta_bikes, departures_est, arrivals_est
    """
    df = snapshots.sort_values(["station_id", "fetched_at"]).copy()
    df["bikes_prev"] = df.groupby("station_id")["num_bikes_available"].shift(1)
    df = df.dropna(subset=["bikes_prev"]).copy()
    df["delta_bikes"]    = df["num_bikes_available"] - df["bikes_prev"]
    df["departures_est"] = (-df["delta_bikes"]).clip(lower=0).astype(int)
    df["arrivals_est"]   = df["delta_bikes"].clip(lower=0).astype(int)
    df["net_flow_est"]   = df["arrivals_est"] - df["departures_est"]  # + = arrivées, - = départs
    return df[[
        "fetched_at", "system_id", "station_id",
        "delta_bikes", "departures_est", "arrivals_est", "net_flow_est",
        "num_bikes_available", "num_docks_available",
    ]]


# ── Classe principale ─────────────────────────────────────────────────────────

class GBFSCollector:
    """
    Collecteur de snapshots station_status pour les réseaux VLS français.

    Paramètres
    ----------
    system_ids   : liste de system_id à collecter (None = PRIORITY_SYSTEMS)
    min_stations : taille minimale du réseau (filtrage catalog)
    timeout      : timeout HTTP en secondes
    snap_dir     : répertoire de stockage des snapshots
    """

    def __init__(
        self,
        system_ids:   list[str] | None = None,
        min_stations: int               = 10,
        timeout:      int               = _TIMEOUT,
        snap_dir:     Path | None       = None,
    ) -> None:
        self.system_ids   = system_ids or PRIORITY_SYSTEMS
        self.min_stations = min_stations
        self.timeout      = timeout
        self.snap_dir     = snap_dir or _SNAP_DIR
        self.snap_dir.mkdir(parents=True, exist_ok=True)

        self._catalog     = self._load_catalog()
        self._feed_cache: dict[str, dict[str, str]] = {}   # system_id → {feed_name → url}
        self._info_cache: dict[str, pd.DataFrame]   = {}   # system_id → station_info DataFrame

    # ── Catalogue ────────────────────────────────────────────────────────────

    def _load_catalog(self) -> pd.DataFrame:
        cat = pd.read_csv(_CATALOG)
        # Filtrer les systèmes en status OK, non free-float, taille suffisante
        cat = cat[cat["status"] == "ok"].copy()
        cat = cat[cat["n_stations"] >= self.min_stations].copy()

        def _is_ff(url: str) -> bool:
            u = url.lower()
            return any(k in u for k in _FF_KEYWORDS)

        cat["is_freeflat"] = cat["gbfs_url"].apply(_is_ff)
        cat = cat[~cat["is_freeflat"]].copy()
        log.info("Catalogue : %d systèmes dock-based éligibles", len(cat))
        return cat

    def get_target_systems(self) -> pd.DataFrame:
        """Retourne le sous-ensemble du catalogue correspondant à self.system_ids."""
        mask = self._catalog["system_id"].isin(self.system_ids)
        found = self._catalog[mask]
        missing = set(self.system_ids) - set(found["system_id"])
        if missing:
            log.warning("system_id introuvables dans le catalogue : %s", missing)
        return found

    # ── Découverte des feeds ──────────────────────────────────────────────────

    def _get_feed_urls(self, system_id: str, gbfs_url: str) -> dict[str, str]:
        if system_id in self._feed_cache:
            return self._feed_cache[system_id]

        urls = {}
        for feed in ("station_status", "station_information"):
            url = _discover_feed_url(gbfs_url, feed, self.timeout)
            if url:
                urls[feed] = url

        self._feed_cache[system_id] = urls
        return urls

    # ── Info stations (statique) ──────────────────────────────────────────────

    def load_station_info(self, system_id: str, gbfs_url: str) -> pd.DataFrame:
        """
        Charge les informations statiques des stations (nom, coordonnées, capacité).
        Met en cache et persiste dans snap_dir/{system_id}/station_info.parquet.
        """
        if system_id in self._info_cache:
            return self._info_cache[system_id]

        cache_path = self.snap_dir / system_id / "station_info.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._info_cache[system_id] = df
            return df

        feeds = self._get_feed_urls(system_id, gbfs_url)
        info_url = feeds.get("station_information")
        if not info_url:
            log.warning("[%s] Pas d'URL station_information.", system_id)
            return pd.DataFrame()

        df = _fetch_station_info(info_url, self.timeout)
        if not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            self._info_cache[system_id] = df
            log.info("[%s] station_info chargée : %d stations", system_id, len(df))
        return df

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def take_snapshot(self, system_id: str, gbfs_url: str) -> pd.DataFrame:
        """
        Prend un snapshot du status actuel et le retourne sous forme de DataFrame.
        Ne persiste pas — appelé par collect_and_save().
        """
        feeds = self._get_feed_urls(system_id, gbfs_url)
        status_url = feeds.get("station_status")
        if not status_url:
            log.debug("[%s] Pas d'URL station_status.", system_id)
            return pd.DataFrame()

        now = datetime.now(timezone.utc)
        raw = _fetch_station_status(status_url, self.timeout)
        if not raw:
            return pd.DataFrame()

        return _parse_status_snapshot(raw, now, system_id)

    def collect_and_save(self, targets: pd.DataFrame) -> dict[str, int]:
        """
        Prend un snapshot pour chaque système dans targets et l'ajoute au fichier
        Parquet journalier (data/status_snapshots/{system_id}/{YYYY-MM-DD}.parquet).

        Retourne un dict {system_id: nb_stations_collectées}.
        """
        today = datetime.now(timezone.utc).date().isoformat()
        results: dict[str, int] = {}

        for _, row in targets.iterrows():
            sid      = str(row["system_id"])
            gbfs_url = str(row["gbfs_url"])

            snap = self.take_snapshot(sid, gbfs_url)
            if snap.empty:
                results[sid] = 0
                continue

            out_path = self.snap_dir / sid / f"{today}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                existing = pd.read_parquet(out_path)
                combined = pd.concat([existing, snap], ignore_index=True)
                combined.to_parquet(out_path, index=False)
            else:
                snap.to_parquet(out_path, index=False)

            results[sid] = len(snap)
            log.info("[%s] Snapshot : %d stations — sauvegardé dans %s",
                     sid, len(snap), out_path.name)

        return results

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(
        self,
        interval_sec:     int = 60,
        max_duration_sec: int | None = None,
    ) -> None:
        """
        Lance la collecte en boucle jusqu'à max_duration_sec secondes (None = infini).

        Paramètres
        ----------
        interval_sec     : intervalle entre deux séries de snapshots (secondes)
        max_duration_sec : durée totale max (None = boucle infinie)
        """
        targets = self.get_target_systems()
        log.info(
            "Démarrage collecte : %d systèmes | intervalle %ds | durée max %s",
            len(targets),
            interval_sec,
            f"{max_duration_sec}s" if max_duration_sec else "∞",
        )

        # Pré-charger les infos statiques
        for _, row in targets.iterrows():
            self.load_station_info(str(row["system_id"]), str(row["gbfs_url"]))

        start = time.monotonic()
        iteration = 0

        while True:
            iteration += 1
            t0 = time.monotonic()
            log.info("── Itération %d ──────────────────────────────────────────", iteration)

            results = self.collect_and_save(targets)
            n_ok = sum(1 for v in results.values() if v > 0)
            n_total = sum(results.values())
            log.info("Résumé itération %d : %d/%d systèmes OK, %d stations collectées",
                     iteration, n_ok, len(results), n_total)

            elapsed = time.monotonic() - start
            if max_duration_sec and elapsed >= max_duration_sec:
                log.info("Durée maximale atteinte (%.0f s). Arrêt.", elapsed)
                break

            # Attendre le prochain intervalle
            sleep_s = max(0, interval_sec - (time.monotonic() - t0))
            if sleep_s > 0:
                log.info("Prochaine collecte dans %.0f s...", sleep_s)
                time.sleep(sleep_s)

    # ── Lecture des données collectées ────────────────────────────────────────

    def load_snapshots(
        self,
        system_id: str,
        date_start: str | None = None,
        date_end:   str | None = None,
    ) -> pd.DataFrame:
        """
        Charge les snapshots stockés pour un système donné.

        Paramètres
        ----------
        system_id  : identifiant du système (ex: 'Paris', 'lyon')
        date_start : date ISO minimale (ex: '2026-03-01'), None = tout
        date_end   : date ISO maximale, None = tout

        Retourne
        --------
        DataFrame trié par fetched_at.
        """
        sys_dir = self.snap_dir / system_id
        if not sys_dir.exists():
            return pd.DataFrame()

        files = sorted(sys_dir.glob("????-??-??.parquet"))
        if date_start:
            files = [f for f in files if f.stem >= date_start]
        if date_end:
            files = [f for f in files if f.stem <= date_end]

        if not files:
            return pd.DataFrame()

        frames = [pd.read_parquet(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        if "fetched_at" in df.columns:
            df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
            df = df.sort_values(["station_id", "fetched_at"])
        return df

    def load_flows(
        self,
        system_id: str,
        date_start: str | None = None,
        date_end:   str | None = None,
    ) -> pd.DataFrame:
        """
        Charge les snapshots et calcule les pseudo-flux (∆bikes entre snapshots consécutifs).
        """
        snaps = self.load_snapshots(system_id, date_start, date_end)
        if snaps.empty:
            return pd.DataFrame()
        return compute_pseudo_flows(snaps)

    def list_available(self) -> pd.DataFrame:
        """
        Liste tous les systèmes pour lesquels des données ont été collectées,
        avec le nombre de fichiers et la plage de dates.
        """
        rows = []
        if not self.snap_dir.exists():
            return pd.DataFrame()
        for sys_dir in sorted(self.snap_dir.iterdir()):
            if not sys_dir.is_dir():
                continue
            files = sorted(sys_dir.glob("????-??-??.parquet"))
            if not files:
                continue
            dates = [f.stem for f in files]
            total_rows = sum(pd.read_parquet(f).shape[0] for f in files[:3])  # sample
            rows.append({
                "system_id":  sys_dir.name,
                "n_files":    len(files),
                "date_debut": dates[0],
                "date_fin":   dates[-1],
                "sample_rows": total_rows,
            })
        return pd.DataFrame(rows)

    # ── Diagnostic ────────────────────────────────────────────────────────────

    def probe_status_feeds(self, system_ids: list[str] | None = None) -> pd.DataFrame:
        """
        Teste quels systèmes exposent station_status et retourne un tableau de diagnostic.
        Utile avant de lancer une longue collecte.
        """
        targets = self.get_target_systems()
        if system_ids:
            targets = targets[targets["system_id"].isin(system_ids)]

        rows = []
        for _, row in targets.iterrows():
            sid      = str(row["system_id"])
            gbfs_url = str(row["gbfs_url"])
            city     = str(row.get("city", ""))

            feeds = self._get_feed_urls(sid, gbfs_url)
            has_status = "station_status" in feeds
            has_info   = "station_information" in feeds

            n_stations_live = 0
            if has_status:
                raw = _fetch_station_status(feeds["station_status"], self.timeout)
                n_stations_live = len(raw)

            rows.append({
                "system_id":       sid,
                "city":            city,
                "has_status_feed": has_status,
                "has_info_feed":   has_info,
                "n_stations_live": n_stations_live,
                "status_url":      feeds.get("station_status", ""),
            })
            status_str = f"OK ({n_stations_live} stations)" if has_status else "ABSENT"
            log.info("[%s] %s — station_status : %s", sid, city, status_str)

        return pd.DataFrame(rows)
