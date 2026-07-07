#!/usr/bin/env python3
"""collect_transit_alerts.py — snapshot des alertes transport (GTFS-RT service_alerts).

Pourquoi : attribuer les chocs de demande VLS (grèves, coupures tram/métro) à leur
cause. Les flux temps réel ne sont archivés nulle part ; ce collecteur fige, toutes
les 15 min (cron), l'état des alertes Tisséo (Toulouse) et TaM (Montpellier) dans un
parquet journalier par réseau : data/transit_alerts/<réseau>/YYYY-MM-DD.parquet.

Robustesse (mêmes conventions que collect_status.py) :
  - un tir par exécution, pas de process long ni d'état ;
  - chaque source sous try/except : une source en panne n'affecte pas les autres,
    le script sort TOUJOURS en code 0 (le cron ne spamme jamais) ;
  - timeouts réseau courts, retry unique ;
  - upsert idempotent par (réseau, id d'alerte, hash du contenu) avec first_seen /
    last_seen : le fichier du jour est réécrit atomiquement (tmp + rename) à chaque
    passage, donc sa fraîcheur (mtime) vaut heartbeat pour healthcheck_collectors ;
  - volumes minuscules (quelques dizaines d'alertes texte par réseau).

Usage : collect_transit_alerts.py [--root DATA_DIR]   (défaut : ./data)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import sys
import tempfile
import urllib.request

SOURCES = {
    # réseau -> liste d'URLs GTFS-RT contenant des service_alerts
    "tisseo": ["https://api.tisseo.fr/opendata/gtfsrt/GtfsRt.pb"],
    "tam": [
        "https://data.montpellier3m.fr/GTFS/Urbain/Alert.pb",
        "https://data.montpellier3m.fr/GTFS/Suburbain/Alert.pb",
    ],
}
TIMEOUT_S = 25
UA = {"User-Agent": "recherche-mobilite-alertes/1.0 (contact: rohan.fosse@gmail.com)"}

log = logging.getLogger("transit_alerts")


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    try:
        return urllib.request.urlopen(req, timeout=TIMEOUT_S).read()
    except Exception:  # un seul retry, puis on laisse l'appelant logger
        return urllib.request.urlopen(req, timeout=TIMEOUT_S).read()


def _txt(translated) -> str:
    """Premier texte d'un TranslatedString GTFS-RT (fr en pratique)."""
    return translated.translation[0].text if translated.translation else ""


def _alert_rows(network: str, raw: bytes, now_iso: str) -> list[dict]:
    from google.transit import gtfs_realtime_pb2 as rt

    msg = rt.FeedMessage()
    msg.ParseFromString(raw)
    rows = []
    for e in msg.entity:
        if not e.HasField("alert"):
            continue  # les flux mixtes (Tisséo) portent aussi des trip_updates
        a = e.alert
        routes = sorted({ie.route_id for ie in a.informed_entity if ie.route_id})
        stops = sorted({ie.stop_id for ie in a.informed_entity if ie.stop_id})
        periods = [
            [int(p.start) if p.HasField("start") else None,
             int(p.end) if p.HasField("end") else None]
            for p in a.active_period
        ]
        row = {
            "network": network,
            "alert_id": e.id,
            "header": _txt(a.header_text),
            "description": _txt(a.description_text),
            "cause": a.Cause.Name(a.cause),
            "effect": a.Effect.Name(a.effect),
            "routes": json.dumps(routes, ensure_ascii=False),
            "stops": json.dumps(stops, ensure_ascii=False),
            "active_periods": json.dumps(periods),
        }
        row["content_hash"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:16]
        row["first_seen"] = now_iso
        row["last_seen"] = now_iso
        rows.append(row)
    return rows


def _upsert_daily(out_dir: str, day: str, rows: list[dict]) -> int:
    """Fusionne les alertes du passage dans le parquet du jour (atomique).

    Clé = (alert_id, content_hash) : une alerte éditée devient une nouvelle ligne
    (l'historique des versions est conservé), une alerte inchangée ne fait
    qu'avancer son last_seen. Réécrit le fichier même sans nouveauté, pour que le
    mtime serve de heartbeat au healthcheck.
    """
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{day}.parquet")
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_parquet(path)
        # Un passage sans alerte peut avoir cree un parquet vide SANS colonnes ;
        # le merge sur alert_id leverait KeyError et perdrait les alertes du jour
        # (bug des 3-5 juillet 2026, 12 ecritures TaM perdues). Vide avec schema.
        if "alert_id" not in old.columns:
            old = pd.DataFrame(columns=new.columns)
        if len(new):
            key = ["alert_id", "content_hash"]
            merged = old.merge(new[key + ["last_seen"]], on=key, how="left",
                               suffixes=("", "_new"))
            old["last_seen"] = merged["last_seen_new"].fillna(merged["last_seen"]).to_numpy()
            fresh = new[~new.set_index(key).index.isin(old.set_index(key).index)]
            out = pd.concat([old, fresh], ignore_index=True)
        else:
            out = old
    else:
        out = new
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    os.close(fd)
    out.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # atomique : jamais de parquet tronqué
    return len(out)


def main() -> int:
    root = "data"
    for a in sys.argv[1:]:
        if a.startswith("--root"):
            root = a.split("=", 1)[1] if "=" in a else "data"
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    now = dt.datetime.now(dt.timezone.utc)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    day = now.astimezone(dt.timezone(dt.timedelta(hours=2))).strftime("%Y-%m-%d")
    summary = []
    for network, urls in SOURCES.items():
        rows: list[dict] = []
        errs = 0
        for url in urls:
            try:
                rows += _alert_rows(network, _fetch(url), now_iso)
            except Exception as exc:  # noqa: BLE001 - une source HS ne casse rien
                errs += 1
                log.info("%s: échec %s (%s)", network, url.rsplit("/", 2)[-2:], exc)
        try:
            total = _upsert_daily(os.path.join(root, "transit_alerts", network), day, rows)
            summary.append(f"{network}:{len(rows)}a/{total}j" + (f"/{errs}err" if errs else ""))
        except Exception as exc:  # noqa: BLE001
            summary.append(f"{network}:écriture-KO({exc})")
            # Filet de secours (leçon des 3-5 juillet 2026) : ne JAMAIS perdre
            # des alertes déjà parsées à cause d'un bug d'écriture. Dump JSON
            # atomique à côté du parquet ; volumes minuscules.
            if rows:
                try:
                    rdir = os.path.join(root, "transit_alerts", network)
                    os.makedirs(rdir, exist_ok=True)
                    rpath = os.path.join(
                        rdir, "RESCUE_%s_%s.json" % (day, now.strftime("%H%M%S")))
                    with open(rpath + ".tmp", "w", encoding="utf-8") as fh:
                        json.dump(rows, fh, ensure_ascii=False)
                    os.replace(rpath + ".tmp", rpath)
                    summary.append(f"{network}:secours={os.path.basename(rpath)}")
                except Exception:  # noqa: BLE001 - le secours ne casse jamais le tir
                    pass
    log.info("alertes %s", " ".join(summary))
    return 0  # toujours : le cron ne doit jamais s'alarmer, healthcheck s'en charge


if __name__ == "__main__":
    sys.exit(main())
