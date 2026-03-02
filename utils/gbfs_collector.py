"""
gbfs_collector.py — Collecteur GBFS station_status, parallélisé, pour les villes françaises.

Principe :
  1. Lit le catalogue des systèmes (systems_catalog.csv).
  2. Pour chaque système, découvre dynamiquement l'URL station_status via gbfs.json.
  3. Prend des snapshots en parallèle à intervalle régulier et les stocke en Parquet.
  4. Calcule les pseudo-flux (∆bikes) entre snapshots consécutifs.

Usage en module :
    from utils.gbfs_collector import GBFSCollector
    coll = GBFSCollector(max_workers=16)
    coll.run(interval_sec=60, max_duration_sec=3600)

Usage en CLI via scripts/collect_status.py.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ROOT       = Path(__file__).parent.parent
_CATALOG    = _ROOT / "data" / "gbfs_france" / "systems_catalog.csv"
_SNAP_DIR   = _ROOT / "data" / "status_snapshots"
_TIMEOUT    = 12       # secondes par requête HTTP
_FF_KEYWORDS = {"bird", "dott", "pony", "voi", "tier", "lime", "bolt", "wind"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gbfs_collector")

# ── Thread-local sessions HTTP (connection pooling par thread) ─────────────────
_thread_local = threading.local()

def _get_session() -> requests.Session:
    """Retourne une session requests réutilisable par thread (pool de connexions)."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session


# ── Cibles prioritaires (dock-based, 42 villes françaises) ───────────────────
PRIORITY_SYSTEMS: list[str] = [
    # ── Grandes métropoles ──────────────────────────────────────
    "Paris",                      # Vélib' Métropole        (~1 500 stations)
    "lyon",                       # Vélo'v Lyon             (~450)
    "toulouse",                   # VélôToulouse            (~430)
    "citiz_alpes_loire",          # Grenoble Métropole      (~410)
    "citiz_grand_est",            # Vélo'hop Strasbourg     (~285)
    "v_lille",                    # V'Lille                 (~270)
    "velo-tbm-bordeaux",          # TBM Bordeaux            (~225)
    "levelo_inurba_marseille",    # Le Vélo Marseille       (~220)
    # ── Villes moyennes ─────────────────────────────────────────
    "vilvolt_epinal",             # Vilvolt Épinal          (~105)
    "inurba-rouen",               # Lovélo Rouen            (~130)
    "citiz_occitanie",            # Vélomagg Montpellier 2  (~128)
    "nantes",                     # Bicloo Nantes           (~124)
    "CVelo_FR_Clermont-Ferrand",  # C.Vélo Clermont         (~82)
    "calais-velos",               # Vélos Calais            (~80)
    "velonecy60minutes_annecy",   # Vélo'necy Annecy        (~80)
    "donkey_brest",               # Donkey Brest            (~75)
    "twisto_velolib_caen",        # Vélib Caen              (~75)
    "libelo",                     # Libélo Valence          (~68)
    "zebullo",                    # Zebullo Reims           (~63)
    "nextbike_af",                # Véloplus Mulhouse       (~60)
    "le_velo_star",               # LE vélo STAR Rennes     (~57)
    "auxrmlevelo",                # Vélo Auxerre            (~55)
    "citiz_rennes_metropole",     # Citiz Rennes            (~55)
    "montpellier",                # Vélomagg Montpellier    (~56)
    "citiz_bfc",                  # VéloCité Besançon       (~56)  (alias besancon)
    "Optymo_Belfort_ALS",         # Optymo Belfort          (~52)
    "tanlib",                     # Tan'Lib Niort           (~51)
    "capcotentin",                # Cap Cotentin Cherbourg  (~49)
    "donkey_valenciennes",        # Donkey Valenciennes     (~40)
    "idecycle_pau",               # Idécycle Pau            (~40)
    "beb",                        # Vélocité Bourg-en-Bresse (~45)
    "semo",                       # Semo Louviers           (~40)
    "velozef",                    # Vélo Brest (ZEF)        (~40)
    "dijon",                      # DiVia Dijon             (~40)
    "amiens",                     # Vélo'ic Amiens          (~45)
    "cergy",                      # Cy'clic Cergy           (~45)
    "nancy",                      # Vélostan'lib Nancy      (~37)
    "velivert_saint_etienne",     # Vélivert Saint-Étienne  (~97)
    "velopop",                    # Vélopop Avignon         (~29)
    "citiz_angers",               # Vélo Angers             (~26)
    "velibeo",                    # Vélibéo Brive           (~26)
    "citiz_la_rochelle",          # Yélo La Rochelle        (~24)
    "citiz_grand_poitiers",       # Vélo Poitiers           (~22)
]


# ── Parsers GBFS v2/v3 ────────────────────────────────────────────────────────

def _discover_feed_url(gbfs_url: str, feed_name: str, timeout: int = _TIMEOUT) -> str | None:
    """
    Interroge gbfs.json et retourne l'URL du feed demandé.
    Compatible GBFS v2.x et v3.x.
    """
    try:
        r = _get_session().get(gbfs_url, timeout=timeout)
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
    """
    try:
        r = _get_session().get(status_url, timeout=timeout)
        r.raise_for_status()
        doc = r.json()
    except Exception as exc:
        log.warning("Erreur station_status %s : %s", status_url, exc)
        return []

    data = doc.get("data", {})
    if "stations" in data:
        return data["stations"]
    for val in data.values():
        if isinstance(val, dict) and "stations" in val:
            return val["stations"]
    return []


def _fetch_station_info(info_url: str, timeout: int = _TIMEOUT) -> pd.DataFrame:
    """
    Télécharge station_information.json et retourne un DataFrame.
    """
    try:
        r = _get_session().get(info_url, timeout=timeout)
        r.raise_for_status()
        doc = r.json()
    except Exception as exc:
        log.warning("Erreur station_information %s : %s", info_url, exc)
        return pd.DataFrame()

    data = doc.get("data", {})
    stations: list[dict] = []
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
    rows = []
    for s in stations_raw:
        sid  = str(s.get("station_id", ""))
        bikes = s.get("num_bikes_available", s.get("num_vehicles_available", 0))
        docks = s.get("num_docks_available", 0)
        rows.append({
            "fetched_at":           fetched_at,
            "system_id":            system_id,
            "station_id":           sid,
            "num_bikes_available":  int(bikes) if bikes is not None else 0,
            "num_docks_available":  int(docks) if docks is not None else 0,
            "is_renting":           bool(s.get("is_renting", True)),
            "is_returning":         bool(s.get("is_returning", True)),
        })
    return pd.DataFrame(rows)


def compute_pseudo_flows(snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les pseudo-flux entre snapshots consécutifs.

    ∆bikes < 0 → |∆| départs nets estimés
    ∆bikes > 0 → ∆ arrivées nettes estimées
    """
    df = snapshots.sort_values(["station_id", "fetched_at"]).copy()
    df["bikes_prev"] = df.groupby("station_id")["num_bikes_available"].shift(1)
    df = df.dropna(subset=["bikes_prev"]).copy()
    df["delta_bikes"]    = df["num_bikes_available"] - df["bikes_prev"]
    df["departures_est"] = (-df["delta_bikes"]).clip(lower=0).astype(int)
    df["arrivals_est"]   = df["delta_bikes"].clip(lower=0).astype(int)
    df["net_flow_est"]   = df["arrivals_est"] - df["departures_est"]
    return df[[
        "fetched_at", "system_id", "station_id",
        "delta_bikes", "departures_est", "arrivals_est", "net_flow_est",
        "num_bikes_available", "num_docks_available",
    ]]


# ── Classe principale ─────────────────────────────────────────────────────────

class GBFSCollector:
    """
    Collecteur parallèle de snapshots station_status pour les réseaux VLS français.

    Paramètres
    ----------
    system_ids   : liste de system_id (None = PRIORITY_SYSTEMS)
    min_stations : taille minimale du réseau
    timeout      : timeout HTTP par requête (secondes)
    snap_dir     : répertoire de stockage des snapshots
    max_workers  : nombre de threads parallèles pour les requêtes HTTP
    """

    def __init__(
        self,
        system_ids:   list[str] | None = None,
        min_stations: int               = 10,
        timeout:      int               = _TIMEOUT,
        snap_dir:     Path | None       = None,
        max_workers:  int               = 12,
    ) -> None:
        self.system_ids   = system_ids or PRIORITY_SYSTEMS
        self.min_stations = min_stations
        self.timeout      = timeout
        self.snap_dir     = snap_dir or _SNAP_DIR
        self.max_workers  = max_workers
        self.snap_dir.mkdir(parents=True, exist_ok=True)

        self._catalog     = self._load_catalog()
        self._feed_cache: dict[str, dict[str, str]] = {}
        self._info_cache: dict[str, pd.DataFrame]   = {}
        # Un verrou par system_id pour les écritures Parquet concurrentes
        self._write_locks: dict[str, threading.Lock] = {}
        self._locks_lock  = threading.Lock()

    def _get_write_lock(self, system_id: str) -> threading.Lock:
        with self._locks_lock:
            if system_id not in self._write_locks:
                self._write_locks[system_id] = threading.Lock()
            return self._write_locks[system_id]

    # ── Catalogue ─────────────────────────────────────────────────────────────

    def _load_catalog(self) -> pd.DataFrame:
        cat = pd.read_csv(_CATALOG)
        cat = cat[cat["status"] == "ok"].copy()
        cat = cat[cat["n_stations"] >= self.min_stations].copy()

        def _is_ff(url: str) -> bool:
            u = str(url).lower()
            return any(k in u for k in _FF_KEYWORDS)

        cat = cat[~cat["gbfs_url"].apply(_is_ff)].copy()
        log.info("Catalogue : %d systèmes dock-based éligibles", len(cat))
        return cat

    def get_target_systems(self) -> pd.DataFrame:
        """Retourne le sous-ensemble du catalogue correspondant à self.system_ids."""
        mask    = self._catalog["system_id"].isin(self.system_ids)
        found   = self._catalog[mask]
        missing = set(self.system_ids) - set(found["system_id"])
        if missing:
            log.warning("system_id absents du catalogue (ignorés) : %s", sorted(missing))
        return found.reset_index(drop=True)

    # ── Découverte des feeds ──────────────────────────────────────────────────

    def _get_feed_urls(self, system_id: str, gbfs_url: str) -> dict[str, str]:
        if system_id in self._feed_cache:
            return self._feed_cache[system_id]

        urls: dict[str, str] = {}
        for feed in ("station_status", "station_information"):
            url = _discover_feed_url(gbfs_url, feed, self.timeout)
            if url:
                urls[feed] = url

        self._feed_cache[system_id] = urls
        return urls

    # ── Info stations (statique) ──────────────────────────────────────────────

    def load_station_info(self, system_id: str, gbfs_url: str) -> pd.DataFrame:
        if system_id in self._info_cache:
            return self._info_cache[system_id]

        cache_path = self.snap_dir / system_id / "station_info.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._info_cache[system_id] = df
            return df

        feeds   = self._get_feed_urls(system_id, gbfs_url)
        info_url = feeds.get("station_information")
        if not info_url:
            log.debug("[%s] Pas d'URL station_information.", system_id)
            return pd.DataFrame()

        df = _fetch_station_info(info_url, self.timeout)
        if not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            self._info_cache[system_id] = df
            log.info("[%s] station_info : %d stations", system_id, len(df))
        return df

    # ── Snapshot unitaire ─────────────────────────────────────────────────────

    def take_snapshot(self, system_id: str, gbfs_url: str) -> pd.DataFrame:
        feeds      = self._get_feed_urls(system_id, gbfs_url)
        status_url = feeds.get("station_status")
        if not status_url:
            log.debug("[%s] Pas d'URL station_status.", system_id)
            return pd.DataFrame()

        now = datetime.now(timezone.utc)
        raw = _fetch_station_status(status_url, self.timeout)
        if not raw:
            return pd.DataFrame()

        return _parse_status_snapshot(raw, now, system_id)

    # ── Collecte parallèle ────────────────────────────────────────────────────

    def _collect_one(self, row: pd.Series, today: str) -> tuple[str, int]:
        """
        Collecte et sauvegarde un snapshot pour un système.
        Peut s'exécuter en parallèle (thread-safe par verrou par fichier).
        """
        sid      = str(row["system_id"])
        gbfs_url = str(row["gbfs_url"])

        snap = self.take_snapshot(sid, gbfs_url)
        if snap.empty:
            return sid, 0

        out_path = self.snap_dir / sid / f"{today}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_write_lock(sid):
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                pd.concat([existing, snap], ignore_index=True).to_parquet(out_path, index=False)
            else:
                snap.to_parquet(out_path, index=False)

        return sid, len(snap)

    def collect_and_save(self, targets: pd.DataFrame) -> dict[str, int]:
        """
        Prend un snapshot pour chaque système en parallèle et sauvegarde.

        Retourne un dict {system_id: nb_stations_collectées}.
        """
        today    = datetime.now(timezone.utc).date().isoformat()
        results: dict[str, int] = {}
        n_workers = min(self.max_workers, max(len(targets), 1))

        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="gbfs") as pool:
            futures = {
                pool.submit(self._collect_one, row, today): str(row["system_id"])
                for _, row in targets.iterrows()
            }
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    sid, n = future.result()
                    results[sid] = n
                    if n > 0:
                        log.info("[%s] %d stations collectées", sid, n)
                    else:
                        log.debug("[%s] aucune donnée (feed absent ou vide)", sid)
                except Exception as exc:
                    log.warning("[%s] Erreur inattendue : %s", sid, exc)
                    results[sid] = 0

        return results

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(
        self,
        interval_sec:     int = 60,
        max_duration_sec: int | None = None,
    ) -> None:
        """
        Lance la collecte en boucle jusqu'à max_duration_sec secondes (None = infini).
        """
        targets = self.get_target_systems()
        log.info(
            "Démarrage collecte : %d systèmes | %d workers | intervalle %ds | durée %s",
            len(targets),
            self.max_workers,
            interval_sec,
            f"{max_duration_sec}s" if max_duration_sec else "∞",
        )

        # Pré-charger les infos statiques en parallèle
        log.info("Pré-chargement des station_information (parallèle)...")
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="info") as pool:
            list(pool.map(
                lambda row: self.load_station_info(str(row["system_id"]), str(row["gbfs_url"])),
                [row for _, row in targets.iterrows()],
            ))

        start    = time.monotonic()
        iteration = 0

        while True:
            iteration += 1
            t0 = time.monotonic()
            log.info("── Itération %d ──────────────────────────────────────────", iteration)

            results = self.collect_and_save(targets)
            n_ok    = sum(1 for v in results.values() if v > 0)
            n_total = sum(results.values())
            t_iter  = time.monotonic() - t0
            log.info(
                "Résumé itération %d : %d/%d systèmes OK, %d stations — %.1f s",
                iteration, n_ok, len(results), n_total, t_iter,
            )

            elapsed = time.monotonic() - start
            if max_duration_sec and elapsed >= max_duration_sec:
                log.info("Durée maximale atteinte (%.0f s). Arrêt.", elapsed)
                break

            sleep_s = max(0.0, interval_sec - t_iter)
            if sleep_s > 0:
                log.info("Prochaine collecte dans %.0f s...", sleep_s)
                time.sleep(sleep_s)

    # ── Lecture des données collectées ────────────────────────────────────────

    def load_snapshots(
        self,
        system_id:  str,
        date_start: str | None = None,
        date_end:   str | None = None,
    ) -> pd.DataFrame:
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

        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "fetched_at" in df.columns:
            df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
            df = df.sort_values(["station_id", "fetched_at"])
        return df

    def load_flows(
        self,
        system_id:  str,
        date_start: str | None = None,
        date_end:   str | None = None,
    ) -> pd.DataFrame:
        snaps = self.load_snapshots(system_id, date_start, date_end)
        if snaps.empty:
            return pd.DataFrame()
        return compute_pseudo_flows(snaps)

    def list_available(self) -> pd.DataFrame:
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
            rows.append({
                "system_id":  sys_dir.name,
                "n_files":    len(files),
                "date_debut": dates[0],
                "date_fin":   dates[-1],
            })
        return pd.DataFrame(rows)

    # ── Diagnostic ────────────────────────────────────────────────────────────

    def probe_status_feeds(self, system_ids: list[str] | None = None) -> pd.DataFrame:
        """
        Teste en parallèle quels systèmes exposent station_status.
        """
        targets = self.get_target_systems()
        if system_ids:
            targets = targets[targets["system_id"].isin(system_ids)]

        def _probe_one(row: pd.Series) -> dict:
            sid      = str(row["system_id"])
            gbfs_url = str(row["gbfs_url"])
            city     = str(row.get("city", ""))
            feeds    = self._get_feed_urls(sid, gbfs_url)
            has_status = "station_status" in feeds
            n_live     = 0
            if has_status:
                n_live = len(_fetch_station_status(feeds["station_status"], self.timeout))
            status_str = f"OK ({n_live} st.)" if has_status else "ABSENT"
            log.info("[%s] %s — %s", sid, city, status_str)
            return {
                "system_id":       sid,
                "city":            city,
                "has_status_feed": has_status,
                "has_info_feed":   "station_information" in feeds,
                "n_stations_live": n_live,
                "status_url":      feeds.get("station_status", ""),
            }

        n_workers = min(self.max_workers, max(len(targets), 1))
        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="probe") as pool:
            futures = [pool.submit(_probe_one, row) for _, row in targets.iterrows()]
            for f in as_completed(futures):
                try:
                    rows.append(f.result())
                except Exception as exc:
                    log.warning("Erreur sonde : %s", exc)

        return pd.DataFrame(rows)
