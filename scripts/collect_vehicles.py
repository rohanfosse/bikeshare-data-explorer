#!/usr/bin/env python
"""
collect_vehicles.py — Collecte continue des flux GBFS free_bike_status /
vehicle_status (identifiant véhicule persistant -> reconstruction OD réelle).

Complément de collect_status.py. Écrit dans data/vehicle_snapshots/,
n'altère pas la collecte station_status.

Exemples :

  # Collecte toutes les 120 s pendant 30 jours
  python scripts/collect_vehicles.py --interval 120 --duration 2592000

  # Test rapide : 2 snapshots, intervalle 30 s
  python scripts/collect_vehicles.py --interval 30 --max-iter 2

  # Sous-ensemble de systèmes
  python scripts/collect_vehicles.py --systems montpellier velivert_saint_etienne
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.vehicle_collector import VehicleCollector  # noqa: E402

log = logging.getLogger("collect_vehicles")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collecteur parallèle free_bike_status / vehicle_status.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--systems", "-s", nargs="+", default=None, metavar="SYSTEM_ID",
                   help="system_id à collecter (défaut : villes prioritaires "
                        "exposant un feed véhicule).")
    p.add_argument("--interval", "-i", type=int, default=120, metavar="SECONDES",
                   help="Intervalle entre snapshots (défaut : 120 s).")
    p.add_argument("--duration", "-d", type=int, default=None, metavar="SECONDES",
                   help="Durée totale (défaut : infini, Ctrl+C pour arrêter).")
    p.add_argument("--max-iter", "-n", type=int, default=None, metavar="N",
                   help="Nombre maximal d'itérations (prioritaire sur --duration).")
    p.add_argument("--workers", "-w", type=int, default=12, metavar="N",
                   help="Threads HTTP parallèles (défaut : 12).")
    p.add_argument("--timeout", "-t", type=int, default=12, metavar="SECONDES",
                   help="Timeout HTTP par requête (défaut : 12 s).")
    p.add_argument("--out-dir", type=Path, default=None, metavar="CHEMIN",
                   help="Répertoire de sortie (défaut : data/vehicle_snapshots/).")
    p.add_argument("--verbose", "-v", action="store_true", help="Messages DEBUG.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    collector = VehicleCollector(
        system_ids=args.systems,
        timeout=args.timeout,
        out_dir=args.out_dir,
        max_workers=args.workers,
    )

    try:
        collector.run(
            interval_sec=args.interval,
            max_duration_sec=args.duration,
            max_iter=args.max_iter,
        )
    except KeyboardInterrupt:
        log.info("Interruption clavier — arrêt propre.")


if __name__ == "__main__":
    main()
