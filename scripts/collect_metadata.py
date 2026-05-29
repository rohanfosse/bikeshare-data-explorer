#!/usr/bin/env python
"""
collect_metadata.py — Collecte des feeds de métadonnées GBFS (1 passage).

Tier 1 (statiques, écrasés) : system_information, vehicle_types,
system_pricing_plans, system_regions.
Tier 2 (datés, snapshots)   : geofencing_zones, system_alerts.

Conçu pour tourner 1×/jour via cron. Un seul passage, pas de boucle.

Exemples :
  python scripts/collect_metadata.py
  python scripts/collect_metadata.py --systems montpellier velivert_saint_etienne
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.metadata_collector import MetadataCollector  # noqa: E402

log = logging.getLogger("collect_metadata")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collecteur des feeds de métadonnées GBFS (1 passage).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--systems", "-s", nargs="+", default=None, metavar="SYSTEM_ID",
                   help="system_id à collecter (défaut : villes prioritaires).")
    p.add_argument("--workers", "-w", type=int, default=12, metavar="N",
                   help="Threads HTTP parallèles (défaut : 12).")
    p.add_argument("--timeout", "-t", type=int, default=12, metavar="SECONDES",
                   help="Timeout HTTP par requête (défaut : 12 s).")
    p.add_argument("--out-dir", type=Path, default=None, metavar="CHEMIN",
                   help="Répertoire de sortie (défaut : data/system_metadata/).")
    p.add_argument("--verbose", "-v", action="store_true", help="Messages DEBUG.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    collector = MetadataCollector(
        system_ids=args.systems,
        timeout=args.timeout,
        out_dir=args.out_dir,
        max_workers=args.workers,
    )
    results = collector.collect_all()
    n_sys = sum(1 for c in results.values() if c)
    feeds = {}
    for c in results.values():
        for k, v in c.items():
            feeds[k] = feeds.get(k, 0) + 1
    log.info("Terminé : %d/%d systèmes avec au moins un feed.", n_sys, len(results))
    log.info("Couverture par feed : %s",
             ", ".join(f"{k}={v} sys." for k, v in sorted(feeds.items())))


if __name__ == "__main__":
    main()
