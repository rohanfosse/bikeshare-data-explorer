#!/usr/bin/env python
"""
collect_status.py — Collecte continue des flux VLS GBFS (parallélisé).

Exemples :

  # Collecte toutes les 60 s pendant 8 h, 12 workers
  python scripts/collect_status.py --interval 60 --duration 28800

  # Test rapide : 3 snapshots sur Paris + Lyon
  python scripts/collect_status.py --systems Paris lyon --interval 30 --max-iter 3

  # Toutes les villes prioritaires, sans mise en veille (Windows)
  python scripts/collect_status.py --keep-awake

  # Voir le script PowerShell pour le mode arrière-plan :
  powershell -File scripts/run_collect.ps1 -Interval 60 -Duration 28800
"""
from __future__ import annotations

import argparse
import ctypes
import logging
import signal
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.gbfs_collector import GBFSCollector, PRIORITY_SYSTEMS

log = logging.getLogger("collect_status")

# ── Veille Windows ─────────────────────────────────────────────────────────────
_ES_CONTINUOUS      = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _prevent_sleep() -> bool:
    """
    Empêche Windows de mettre le système en veille pendant la collecte.
    Appelle SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
    Retourne True si l'appel a réussi.
    """
    if sys.platform != "win32":
        log.debug("--keep-awake ignoré (non-Windows).")
        return False
    try:
        result = ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )
        if result:
            log.info("Mode veille désactivé (SetThreadExecutionState).")
            return True
        log.warning("SetThreadExecutionState a retourné 0 — veille non désactivée.")
        return False
    except Exception as exc:
        log.warning("Impossible de désactiver la veille : %s", exc)
        return False


def _allow_sleep() -> None:
    """Restaure le comportement de veille par défaut."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        log.info("Mode veille restauré.")
    except Exception:
        pass


# ── Arguments ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collecteur parallèle de snapshots station_status GBFS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--systems", "-s",
        nargs="+",
        default=None,
        metavar="SYSTEM_ID",
        help=(
            f"system_id à collecter. Défaut : {len(PRIORITY_SYSTEMS)} villes prioritaires."
        ),
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        metavar="SECONDES",
        help="Intervalle entre deux séries de snapshots (défaut : 60 s).",
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        metavar="SECONDES",
        help="Durée totale en secondes (défaut : infini, arrêt par Ctrl+C).",
    )
    parser.add_argument(
        "--max-iter", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Nombre maximal d'itérations (prioritaire sur --duration).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=12,
        metavar="N",
        help="Nombre de threads parallèles pour les requêtes HTTP (défaut : 12).",
    )
    parser.add_argument(
        "--min-stations",
        type=int,
        default=10,
        metavar="N",
        help="Taille minimale du réseau (défaut : 10 stations).",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=12,
        metavar="SECONDES",
        help="Timeout HTTP par requête (défaut : 12 s).",
    )
    parser.add_argument(
        "--snap-dir",
        type=Path,
        default=None,
        metavar="CHEMIN",
        help="Répertoire de stockage des snapshots (défaut : data/status_snapshots/).",
    )
    parser.add_argument(
        "--keep-awake",
        action="store_true",
        default=True,
        help="Empêche Windows de mettre l'ordinateur en veille pendant la collecte (défaut : activé).",
    )
    parser.add_argument(
        "--no-keep-awake",
        dest="keep_awake",
        action="store_false",
        help="Désactive la prévention de veille.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Afficher les messages DEBUG.",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Prévention de veille Windows
    _sleep_prevented = False
    if args.keep_awake:
        _sleep_prevented = _prevent_sleep()

    collector = GBFSCollector(
        system_ids=args.systems,
        min_stations=args.min_stations,
        timeout=args.timeout,
        snap_dir=args.snap_dir,
        max_workers=args.workers,
    )

    targets = collector.get_target_systems()
    if targets.empty:
        log.error("Aucun système cible trouvé. Vérifiez les system_id passés.")
        if _sleep_prevented:
            _allow_sleep()
        sys.exit(1)

    print(f"\nSystèmes ciblés ({len(targets)}) :")
    for _, row in targets.iterrows():
        print(f"  • {row['system_id']:<40} {row['city']:<30} ({row['n_stations']} stations)")
    print(f"\nIntervalle    : {args.interval} s")
    print(f"Workers HTTP  : {args.workers}")
    print(f"Durée max     : {args.duration or '∞'} s")
    print(f"Prév. veille  : {'oui' if _sleep_prevented else 'non (non-Windows ou désactivé)'}")
    if args.max_iter:
        print(f"Max itérations: {args.max_iter}")
    print(f"Stockage      : {collector.snap_dir}\n")

    # Gestionnaire de signal propre (Ctrl+C)
    _stop = threading.Event() if False else None  # placeholder

    try:
        if args.max_iter is not None:
            # Pré-charger station_info en parallèle
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="info") as pool:
                list(pool.map(
                    lambda row: collector.load_station_info(str(row["system_id"]), str(row["gbfs_url"])),
                    [row for _, row in targets.iterrows()],
                ))

            for i in range(args.max_iter):
                log.info("── Itération %d / %d ──────────────────────────────────", i + 1, args.max_iter)
                t0      = time.monotonic()
                results = collector.collect_and_save(targets)
                n_ok    = sum(1 for v in results.values() if v > 0)
                n_total = sum(results.values())
                t_iter  = time.monotonic() - t0
                log.info(
                    "Résumé : %d/%d systèmes OK, %d stations — %.1f s",
                    n_ok, len(results), n_total, t_iter,
                )
                if i < args.max_iter - 1:
                    sleep_s = max(0.0, args.interval - t_iter)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
        else:
            collector.run(
                interval_sec=args.interval,
                max_duration_sec=args.duration,
            )

    except KeyboardInterrupt:
        log.info("Collecte interrompue par l'utilisateur (Ctrl+C).")
    finally:
        if _sleep_prevented:
            _allow_sleep()

    # Résumé final
    avail = collector.list_available()
    if not avail.empty:
        print("\nFichiers collectés :")
        print(avail.to_string(index=False))
    else:
        print("\nAucun fichier collecté.")


if __name__ == "__main__":
    main()
