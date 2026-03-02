#!/usr/bin/env python
"""
test_status_feeds.py — Diagnostic rapide : quelles villes exposent station_status ?

Lance une sonde sur tous les systèmes prioritaires et affiche un tableau de compatibilité.
Utile avant de lancer collect_status.py pour vérifier les feeds disponibles.

Usage :
  python scripts/test_status_feeds.py
  python scripts/test_status_feeds.py --systems Paris lyon toulouse
  python scripts/test_status_feeds.py --all          # tester tous les systèmes dock-based
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils.gbfs_collector import GBFSCollector, PRIORITY_SYSTEMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic des feeds station_status GBFS disponibles.",
    )
    parser.add_argument(
        "--systems", "-s",
        nargs="+",
        default=None,
        metavar="SYSTEM_ID",
        help="system_id à tester (défaut : villes prioritaires).",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Tester tous les systèmes dock-based du catalogue (lent).",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=12,
        help="Timeout HTTP par requête (défaut : 12 s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    collector = GBFSCollector(
        system_ids=args.systems if not args.all else None,
        min_stations=5 if args.all else 10,
        timeout=args.timeout,
    )

    if args.all:
        # Remplacer par tous les systèmes du catalogue filtré
        collector.system_ids = collector._catalog["system_id"].tolist()

    print(f"\nSonde station_status sur {len(collector.system_ids)} systèmes...\n")

    df = collector.probe_status_feeds()

    ok    = df[df["has_status_feed"] & (df["n_stations_live"] > 0)]
    nok   = df[~df["has_status_feed"]]
    empty = df[df["has_status_feed"] & (df["n_stations_live"] == 0)]

    print("=" * 70)
    print(f"RÉSULTATS : {len(ok)} OK | {len(empty)} vide | {len(nok)} absent")
    print("=" * 70)

    if not ok.empty:
        print(f"\n✔  Systèmes avec station_status actif ({len(ok)}) :")
        for _, row in ok.sort_values("n_stations_live", ascending=False).iterrows():
            print(f"   {row['system_id']:<40} {row['city']:<25} {row['n_stations_live']} stations")

    if not empty.empty:
        print(f"\n⚠  Feeds présents mais vides ({len(empty)}) :")
        for _, row in empty.iterrows():
            print(f"   {row['system_id']:<40} {row['city']}")

    if not nok.empty:
        print(f"\n✘  Pas de station_status ({len(nok)}) :")
        for _, row in nok.iterrows():
            print(f"   {row['system_id']:<40} {row['city']}")

    print()

    # Exporter les résultats
    out = _ROOT / "data" / "status_snapshots" / "feed_diagnostic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Rapport exporté : {out}")

    # Suggestion de commande collect_status
    if not ok.empty:
        sys_list = " ".join(ok["system_id"].tolist())
        print(f"\nPour lancer la collecte sur les {len(ok)} systèmes compatibles :")
        print(f"  python scripts/collect_status.py --systems {sys_list} --interval 60")


if __name__ == "__main__":
    main()
