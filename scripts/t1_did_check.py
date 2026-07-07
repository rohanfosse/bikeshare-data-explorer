#!/usr/bin/env python3
"""t1_did_check.py — suivi quotidien du DiD pré-enregistré (Toulouse, juillet-août 2026).

Prédiction gelée (microcity c47717b, docs/prediction_toulouse_2026.json) :
uplift DiD [+8, +32] %% (central +16 %%) des départs des stations exposées
(<500 m d'un arrêt fermé) vs contrôles, pendant vs hors fenêtre.
Fenêtres : tram_T1 2026-07-04 -> 2026-07-12 ; ligne_B_sud 2026-07-13 -> 2026-08-30.

Départs (proxy) = somme des baisses de num_bikes_available entre snapshots.
Baselines séparées jours ouvrés / week-end (médiane + MAD) sur les jours
antérieurs au 2026-07-04. Lecture 3 colonnes typées, jour par jour (pas d'OOM).

Usage : t1_did_check.py [tram_T1|ligne_B_sud]   (défaut : tram_T1)
"""
import glob
import sys

import numpy as np
import pandas as pd

EXPOSED = {"ligne_B_sud": ["155", "156", "157", "158", "159", "161", "162", "163", "164", "165", "166", "227", "228", "229", "230", "231", "233", "255", "281", "293", "295", "360", "5001"], "tram_T1": ["103", "104", "113", "116", "129", "136", "139", "140", "141", "142", "154", "183", "188", "189", "191", "193", "194", "2003", "2004", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "226", "245", "250", "268", "271", "278", "279", "289", "291", "292", "301", "302", "304", "305", "306", "37", "392", "393", "45", "47", "48", "50", "67", "68", "69", "70", "71", "72", "73", "74", "9"]}
WINDOWS = {"tram_T1": ("2026-07-04", "2026-07-12"),
           "ligne_B_sud": ("2026-07-13", "2026-08-30")}
DATA = "data/status_snapshots/toulouse"


def day_ratio(f, exposed):
    d = pd.read_parquet(f, columns=["fetched_at", "station_id", "num_bikes_available"])
    d = d.sort_values(["station_id", "fetched_at"])
    d["dep"] = (-d.groupby("station_id")["num_bikes_available"].diff()).clip(lower=0)
    g = d.groupby(d["station_id"].isin(exposed))["dep"].sum()
    e, c = float(g.get(True, 0.0)), float(g.get(False, 0.0))
    return e, c, (e / c if c > 0 else np.nan)


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else "tram_T1"
    exposed = set(EXPOSED[event])
    w0, w1 = WINDOWS[event]
    base_wd, base_we = [], []
    rows = []
    for f in sorted(glob.glob(f"{DATA}/2026-*.parquet")):
        day = f.rsplit("/", 1)[1][:10]
        e, c, r = day_ratio(f, exposed)
        if not np.isfinite(r):
            continue
        wd = pd.Timestamp(day).dayofweek < 5
        if day < "2026-07-04":
            (base_wd if wd else base_we).append(r)
        rows.append((day, wd, e, c, r))
    med_wd, mad_wd = np.median(base_wd), np.median(np.abs(np.array(base_wd) - np.median(base_wd)))
    med_we, mad_we = np.median(base_we), np.median(np.abs(np.array(base_we) - np.median(base_we)))
    print(f"{event} | exposees n={len(exposed)} | baseline ouvree: med={med_wd:.3f} "
          f"MAD={mad_wd:.3f} (n={len(base_wd)}) | week-end: med={med_we:.3f} "
          f"MAD={mad_we:.3f} (n={len(base_we)})")
    print("jour        type    expo   ctrl  ratio    DiD     MAD")
    for day, wd, e, c, r in rows:
        if day < w0 or day > w1:
            continue
        med, mad = (med_wd, mad_wd) if wd else (med_we, mad_we)
        did = 100 * (r - med) / med
        nmad = (r - med) / mad if mad > 0 else float("nan")
        print(f"{day}  {'ouvre' if wd else 'we   '} {e:>6.0f} {c:>6.0f}  {r:.3f}  "
              f"{did:+5.1f}%  {nmad:+5.1f}")
    print("rappel pre-enregistre (jours ouvres): [+8, +32] %, central +16 %")


if __name__ == "__main__":
    main()
