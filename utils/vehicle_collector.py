"""
vehicle_collector.py — Collecteur GBFS free_bike_status / vehicle_status.

Complément du collecteur station_status (gbfs_collector.py). Là où
station_status ne livre que des compteurs par station (impossible de
relier un départ à une arrivée), free_bike_status / vehicle_status
expose un identifiant *persistant* par vélo (vehicle_id / bike_id) et
sa position. En suivant la disparition/réapparition d'un identifiant,
on reconstruit des trajets réels — donc une matrice OD observée, et
non une pseudo-OD sous modèle gravitaire.

Principe :
  1. Réutilise le catalogue + la découverte de feeds de gbfs_collector.
  2. Détecte automatiquement les systèmes exposant free_bike_status
     (GBFS v2) ou vehicle_status (GBFS v3) ; ignore les autres.
  3. Prend des snapshots véhicule en parallèle et les stocke en Parquet,
     un fichier par système et par jour, sous data/vehicle_snapshots/.

Ce module n'altère **pas** la collecte station_status existante :
répertoire de sortie séparé, classe séparée, aucune écriture partagée.

Usage CLI : scripts/collect_vehicles.py
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gbfs_toolkit import to_canonical_vehicles
from utils.gbfs_collector import (
    PRIORITY_SYSTEMS,
    _ROOT,
    _TIMEOUT,
    _discover_feed_url,
    _get_session,
)

_VEHICLE_DIR = _ROOT / "data" / "vehicle_snapshots"

log = logging.getLogger("vehicle_collector")

# Feeds véhicule par ordre de préférence (v2 puis v3).
_VEHICLE_FEEDS = ("free_bike_status", "vehicle_status")


# ── Fetch + parse ─────────────────────────────────────────────────────────────

def _fetch_vehicles(url: str, timeout: int = _TIMEOUT) -> list[dict]:
    """Télécharge un feed free_bike_status / vehicle_status.

    Gère les deux formes : data.bikes (v2) et data.vehicles (v3), y
    compris quand elles sont imbriquées sous une clé de langue.
    """
    try:
        r = _get_session().get(url, timeout=timeout)
        r.raise_for_status()
        doc = r.json()
    except Exception as exc:
        log.warning("Erreur feed véhicule %s : %s", url, exc)
        return []

    data = doc.get("data", {})
    for key in ("bikes", "vehicles"):
        if key in data and isinstance(data[key], list):
            return data[key]
    # Imbrication par langue (rare sur ces feeds, mais on couvre).
    for val in data.values():
        if isinstance(val, dict):
            for key in ("bikes", "vehicles"):
                if key in val and isinstance(val[key], list):
                    return val[key]
    return []


_VEHICLE_COLUMNS = [
    "fetched_at", "system_id", "vehicle_id", "lat", "lon",
    "station_id", "is_disabled", "is_reserved", "vehicle_type_id",
]


def _parse_vehicle_snapshot(
    vehicles_raw: list[dict],
    fetched_at: datetime,
    system_id: str,
) -> pd.DataFrame:
    """Normalise un snapshot véhicule.

    Conserve l'identifiant persistant, la position, et les drapeaux qui distinguent un vélo
    disponible d'un vélo HS / réservé (utilisés pour filtrer le rééquilibrage lors de la
    reconstruction OD). Le parsing des champs (v2 ``bike_id`` / v3 ``vehicle_id``) est délégué
    à gbfs_toolkit.to_canonical_vehicles, puis ramené au schéma/dtypes historiques de l'app.
    """
    raw = [v for v in vehicles_raw if (v.get("vehicle_id") or v.get("bike_id")) is not None]
    if not raw:
        return pd.DataFrame(columns=_VEHICLE_COLUMNS)

    canon = to_canonical_vehicles(
        {"data": {"vehicles": raw}},
        system_id=system_id,
        fetched_at=pd.Timestamp(fetched_at),
    )

    def _opt_str(col: pd.Series) -> pd.Series:
        # nullable string -> object with None (preserve the app's historical None, not pd.NA)
        return col.astype(object).where(col.notna(), None)

    return pd.DataFrame({
        "fetched_at":      fetched_at,
        "system_id":       system_id,
        "vehicle_id":      canon["vehicle_id"].astype(str),
        "lat":             pd.to_numeric(canon["lat"], errors="coerce"),
        "lon":             pd.to_numeric(canon["lon"], errors="coerce"),
        "station_id":      _opt_str(canon["station_id"]),
        "is_disabled":     canon["is_disabled"].fillna(False).astype(bool),
        "is_reserved":     canon["is_reserved"].fillna(False).astype(bool),
        "vehicle_type_id": _opt_str(canon["vehicle_type_id"]),
    })


# ── Reconstruction OD (couche analyse, à lancer une fois l'historique accumulé) ──

def reconstruct_trips(
    snapshots: pd.DataFrame,
    min_move_m: float = 100.0,
    max_simultaneous: int = 5,
) -> pd.DataFrame:
    """Reconstruit des trajets observés à partir des snapshots véhicule.

    Un trajet est détecté quand un vehicle_id change de position de plus
    de ``min_move_m`` entre deux apparitions consécutives (ou disparaît
    puis réapparaît ailleurs). Le déplacement est l'origine→destination.

    Filtres :
      - ``min_move_m`` : seuil de distance pour ignorer le jitter GPS.
      - ``max_simultaneous`` : si plus de ``max_simultaneous`` véhicules
        bougent dans le même pas de temps depuis une même zone, c'est
        probablement un rééquilibrage opérateur (marqué ``rebalancing``).
      - les transitions vers/depuis ``is_disabled=True`` sont marquées
        ``rebalancing`` (vélo retiré/remis par l'opérateur).

    Renvoie un trajet par ligne : vehicle_id, t_start, t_end, o_lat,
    o_lon, d_lat, d_lon, dist_m, o_station, d_station, rebalancing.
    """
    import numpy as np

    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    df = snapshots.dropna(subset=["lat", "lon"]).copy()
    df = df.sort_values(["vehicle_id", "fetched_at"])
    g = df.groupby("vehicle_id", sort=False)
    df["plat"] = g["lat"].shift(1)
    df["plon"] = g["lon"].shift(1)
    df["pt"]   = g["fetched_at"].shift(1)
    df["pstation"]  = g["station_id"].shift(1)
    df["pdisabled"] = g["is_disabled"].shift(1)
    moves = df.dropna(subset=["plat", "plon"]).copy()
    moves["dist_m"] = _haversine_m(
        moves["plat"].to_numpy(), moves["plon"].to_numpy(),
        moves["lat"].to_numpy(),  moves["lon"].to_numpy(),
    )
    trips = moves[moves["dist_m"] >= min_move_m].copy()

    # Heuristique rééquilibrage : transition impliquant is_disabled.
    # .eq(True) évite le downcast object->bool de fillna (NaN -> False).
    pdis = trips["pdisabled"].eq(True)
    trips["rebalancing"] = trips["is_disabled"].astype(bool) | pdis
    # Heuristique rééquilibrage : grappes de départs simultanés.
    burst = (
        trips.groupby("t_start" if "t_start" in trips else "pt")["vehicle_id"]
        .transform("count")
    )
    trips["rebalancing"] = trips["rebalancing"] | (burst > max_simultaneous)

    out = trips.rename(columns={
        "pt": "t_start", "fetched_at": "t_end",
        "plat": "o_lat", "plon": "o_lon",
        "lat": "d_lat", "lon": "d_lon",
        "pstation": "o_station", "station_id": "d_station",
    })
    return out[[
        "vehicle_id", "system_id", "t_start", "t_end",
        "o_lat", "o_lon", "d_lat", "d_lon", "dist_m",
        "o_station", "d_station", "rebalancing",
    ]].reset_index(drop=True)


# ── Collecteur ──────────────────────────────────────────────────────────────────

class VehicleCollector:
    """Collecteur parallèle de snapshots free_bike_status / vehicle_status."""

    def __init__(
        self,
        system_ids:  list[str] | None = None,
        timeout:     int = _TIMEOUT,
        out_dir:     Path | None = None,
        max_workers: int = 12,
    ) -> None:
        self.system_ids  = system_ids or PRIORITY_SYSTEMS
        self.timeout     = timeout
        self.out_dir     = Path(out_dir) if out_dir is not None else _VEHICLE_DIR
        self.max_workers = max_workers
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._catalog = self._load_catalog()
        self._vehicle_url: dict[str, str] = {}   # system_id -> feed url (résolu une fois)
        self._write_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def _load_catalog(self) -> pd.DataFrame:
        cat = pd.read_csv(_ROOT / "data" / "gbfs_france" / "systems_catalog.csv")
        cat = cat[cat["status"] == "ok"].copy()
        return cat

    def _get_write_lock(self, system_id: str) -> threading.Lock:
        with self._locks_lock:
            if system_id not in self._write_locks:
                self._write_locks[system_id] = threading.Lock()
            return self._write_locks[system_id]

    def get_target_systems(self) -> pd.DataFrame:
        mask = self._catalog["system_id"].isin(self.system_ids)
        return self._catalog[mask].reset_index(drop=True)

    def discover_vehicle_feeds(self, targets: pd.DataFrame) -> dict[str, str]:
        """Résout, une fois, l'URL du feed véhicule de chaque système.

        Ne garde que les systèmes qui exposent free_bike_status ou
        vehicle_status — les autres (Vélib' Paris, Lyon, etc.) sont
        écartés silencieusement.
        """
        def _resolve(row: pd.Series) -> tuple[str, str | None]:
            sid, gbfs = str(row["system_id"]), str(row["gbfs_url"])
            for feed in _VEHICLE_FEEDS:
                url = _discover_feed_url(gbfs, feed, self.timeout)
                if url:
                    return sid, url
            return sid, None

        found: dict[str, str] = {}
        n_workers = min(self.max_workers, max(len(targets), 1))
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="vdisc") as pool:
            for sid, url in pool.map(_resolve, [r for _, r in targets.iterrows()]):
                if url:
                    found[sid] = url
        self._vehicle_url = found
        log.info("Feeds véhicule détectés : %d / %d systèmes", len(found), len(targets))
        return found

    def take_snapshot(self, system_id: str) -> pd.DataFrame:
        url = self._vehicle_url.get(system_id)
        if not url:
            return pd.DataFrame()
        now = datetime.now(timezone.utc)
        raw = _fetch_vehicles(url, self.timeout)
        if not raw:
            return pd.DataFrame()
        return _parse_vehicle_snapshot(raw, now, system_id)

    def _collect_one(self, system_id: str, today: str) -> tuple[str, int]:
        snap = self.take_snapshot(system_id)
        if snap.empty:
            return system_id, 0
        out_path = self.out_dir / system_id / f"{today}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_write_lock(system_id):
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                pd.concat([existing, snap], ignore_index=True).to_parquet(out_path, index=False)
            else:
                snap.to_parquet(out_path, index=False)
        return system_id, len(snap)

    def collect_and_save(self) -> dict[str, int]:
        today = datetime.now(timezone.utc).date().isoformat()
        results: dict[str, int] = {}
        sids = list(self._vehicle_url.keys())
        n_workers = min(self.max_workers, max(len(sids), 1))
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="veh") as pool:
            futures = {pool.submit(self._collect_one, sid, today): sid for sid in sids}
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    sid, n = future.result()
                    results[sid] = n
                    if n > 0:
                        log.info("[%s] %d véhicules collectés", sid, n)
                except Exception as exc:
                    log.warning("[%s] Erreur : %s", sid, exc)
                    results[sid] = 0
        return results

    def run(
        self,
        interval_sec: int = 120,
        max_duration_sec: int | None = None,
        max_iter: int | None = None,
    ) -> None:
        targets = self.get_target_systems()
        self.discover_vehicle_feeds(targets)
        if not self._vehicle_url:
            log.warning("Aucun feed véhicule détecté — rien à collecter.")
            return
        log.info(
            "Démarrage collecte véhicule : %d systèmes | %d workers | intervalle %ds | durée %s",
            len(self._vehicle_url), self.max_workers, interval_sec,
            f"{max_duration_sec}s" if max_duration_sec else "∞",
        )
        start = time.monotonic()
        iteration = 0
        while True:
            iteration += 1
            t0 = time.monotonic()
            results = self.collect_and_save()
            n_ok = sum(1 for v in results.values() if v > 0)
            n_tot = sum(results.values())
            t_iter = time.monotonic() - t0
            log.info(
                "Itération %d : %d/%d systèmes OK, %d véhicules — %.1f s",
                iteration, n_ok, len(results), n_tot, t_iter,
            )
            if max_iter and iteration >= max_iter:
                log.info("Nombre maximal d'itérations atteint (%d). Arrêt.", max_iter)
                break
            elapsed = time.monotonic() - start
            if max_duration_sec and elapsed >= max_duration_sec:
                log.info("Durée maximale atteinte (%.0f s). Arrêt.", elapsed)
                break
            sleep_s = max(0.0, interval_sec - t_iter)
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ── Lecture ────────────────────────────────────────────────────────────────

    def load_snapshots(
        self,
        system_id: str,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> pd.DataFrame:
        sys_dir = self.out_dir / system_id
        if not sys_dir.exists():
            return pd.DataFrame()
        files = sorted(sys_dir.glob("????-??-??.parquet"))
        if date_start:
            files = [f for f in files if f.stem >= date_start]
        if date_end:
            files = [f for f in files if f.stem <= date_end]
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
        return df.sort_values(["vehicle_id", "fetched_at"])
