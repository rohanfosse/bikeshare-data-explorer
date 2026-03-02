#!/usr/bin/env python
"""
collect_status.py — Script principal de collecte continue des flux VLS GBFS.

Exemples d'utilisation :

  # Collecte toutes les 60 s pendant 8 h (mode standard quotidien)
  python scripts/collect_status.py --interval 60 --duration 28800

  # Test rapide : 3 snapshots espacés de 30 s sur Paris + Lyon uniquement
  python scripts/collect_status.py --systems Paris lyon --interval 30 --max-iter 3

  # Collecte continue (Ctrl+C pour arrêter) sur toutes les villes prioritaires
  python scripts/collect_status.py

  # Collecte longue en tâche de fond (Windows)
  start /B python scripts/collect_status.py --interval 90 --duration 86400 > logs/collect.log 2>&1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Assurer l'import depuis la racine du projet
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.gbfs_collector import GBFSCollector, PRIORITY_SYSTEMS

log = logging.getLogger("collect_status")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collecteur de snapshots station_status GBFS pour les réseaux VLS français.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--systems", "-s",
        nargs="+",
        default=None,
        metavar="SYSTEM_ID",
        help=(
            "system_id à collecter (séparés par des espaces). "
            f"Défaut : les {len(PRIORITY_SYSTEMS)} villes prioritaires. "
            "Exemples : Paris lyon toulouse montpellier"
        ),
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        metavar="SECONDES",
        help="Intervalle entre deux séries de snapshots en secondes (défaut : 60).",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        metavar="SECONDES",
        help=(
            "Durée totale de la collecte en secondes. "
            "Si absent, la collecte tourne indéfiniment jusqu'à Ctrl+C."
        ),
    )
    parser.add_argument(
        "--max-iter", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Nombre maximal d'itérations (prioritaire sur --duration).",
    )
    parser.add_argument(
        "--min-stations",
        type=int,
        default=10,
        metavar="N",
        help="Taille minimale du réseau pour être éligible (défaut : 10).",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=15,
        metavar="SECONDES",
        help="Timeout HTTP par requête (défaut : 15 s).",
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=None,
        metavar="CHEMIN",
        help="Répertoire de stockage des snapshots (défaut : data/status_snapshots/).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher les messages DEBUG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    collector = GBFSCollector(
        system_ids=args.systems,
        min_stations=args.min_stations,
        timeout=args.timeout,
        snap_dir=args.snap_dir,
    )

    targets = collector.get_target_systems()
    if targets.empty:
        log.error("Aucun système cible trouvé. Vérifiez les system_id passés.")
        sys.exit(1)

    print(f"\nSystèmes ciblés ({len(targets)}) :")
    for _, row in targets.iterrows():
        print(f"  • {row['system_id']:<35} {row['city']:<30} ({row['n_stations']} stations)")
    print(f"\nIntervalle : {args.interval} s")
    print(f"Durée max  : {args.duration or '∞'} s")
    if args.max_iter:
        print(f"Max itérations : {args.max_iter}")
    print(f"Stockage   : {collector.snap_dir}\n")

    # Si --max-iter est fourni, on court-circuite la durée
    if args.max_iter is not None:
        import time
        # Pré-charger station_info
        for _, row in targets.iterrows():
            collector.load_station_info(str(row["system_id"]), str(row["gbfs_url"]))

        for i in range(args.max_iter):
            log.info("── Itération %d / %d ────────────────────────────────────", i + 1, args.max_iter)
            results = collector.collect_and_save(targets)
            n_ok    = sum(1 for v in results.values() if v > 0)
            n_total = sum(results.values())
            log.info("Résumé : %d/%d systèmes OK, %d stations", n_ok, len(results), n_total)
            if i < args.max_iter - 1:
                time.sleep(args.interval)
    else:
        try:
            collector.run(
                interval_sec=args.interval,
                max_duration_sec=args.duration,
            )
        except KeyboardInterrupt:
            log.info("Collecte interrompue par l'utilisateur (Ctrl+C).")

    # Résumé final
    avail = collector.list_available()
    if not avail.empty:
        print("\nFichiers collectés :")
        print(avail.to_string(index=False))
    else:
        print("\nAucun fichier collecté.")


if __name__ == "__main__":
    main()
