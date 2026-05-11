"""E4 pilot: dynamic A6-candidate detection from station_status.

Implements a 2-day pilot of experiment E4 of the Gold Standard
validation roadmap. The full protocol calls for 30 days of
``station_status`` snapshots; the present run uses what is
available locally (typically 2 day-files per system in
``data/status_snapshots/<system>/YYYY-MM-DD.parquet``).

For each certified station with sufficient snapshot coverage,
the script computes three dynamic indicators:

  rho_overflow    : P(num_bikes_available > capacity)
  rho_empty       : P(num_bikes_available == 0)
  rho_full_zero   : P(num_docks_available == 0 AND num_bikes == 0)

and flags A6 candidates per the paper's success thresholds:

  A6a (overflow)        : rho_overflow > 0.01
  A6b (saturated empty) : rho_empty    > 0.40
  A6c (degenerate)      : rho_full_zero > 0.20

Run from the repository root:

    python papers/01_gold_standard/experiments/e4_dynamic_pilot/run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir, repo_root  # noqa: E402

ROOT: Final[Path] = repo_root(__file__)
OUT_DIR: Final[Path] = outputs_dir(__file__)
SNAP: Final[Path] = ROOT / "data" / "status_snapshots"
STATIONS: Final[Path] = ROOT / "data" / "stations_gold_standard_final.parquet"


def _load_snapshots_for_system(sys_dir: Path) -> pd.DataFrame | None:
    """Concatenate all day-files for a system (excluding station_info)."""
    day_files = sorted(
        p for p in sys_dir.glob("*.parquet")
        if p.stem != "station_info"
    )
    if not day_files:
        return None
    frames = [pd.read_parquet(p) for p in day_files]
    return pd.concat(frames, ignore_index=True)


def _compute_indicators(snaps: pd.DataFrame, capacity: pd.Series) -> pd.DataFrame:
    """Return one row per station with the three rho indicators."""
    cap_map = capacity.to_dict()
    snaps = snaps.copy()
    snaps["capacity"] = snaps["station_id"].map(cap_map)

    def per_station(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        if n == 0 or g["capacity"].isna().all():
            return pd.Series({
                "n_obs": n, "capacity": float("nan"),
                "rho_overflow": float("nan"),
                "rho_empty": float("nan"),
                "rho_full_zero": float("nan"),
            })
        cap = float(g["capacity"].dropna().iloc[0])
        rho_overflow = (
            float(((g["num_bikes_available"] > cap)
                   & (cap > 0)).sum()) / n
        )
        rho_empty = float((g["num_bikes_available"] == 0).sum()) / n
        rho_full_zero = float(
            ((g["num_bikes_available"] == 0)
             & (g["num_docks_available"] == 0)).sum()
        ) / n
        return pd.Series({
            "n_obs": n, "capacity": cap,
            "rho_overflow": rho_overflow,
            "rho_empty": rho_empty,
            "rho_full_zero": rho_full_zero,
        })

    return snaps.groupby("station_id").apply(per_station).reset_index()


def main() -> None:
    print("Loading certified station catalogue...")
    stations = pd.read_parquet(STATIONS)
    capacity_map = stations.set_index("station_id")["capacity"]
    cert_by_system = stations.groupby("system_id")["station_id"].apply(set)
    print(f"  {len(stations):,} certified stations, "
          f"{stations['system_id'].nunique()} systems")

    sys_dirs = sorted(p for p in SNAP.iterdir() if p.is_dir())
    print(f"  {len(sys_dirs)} systems with snapshot data")
    print()

    all_indicators: list[pd.DataFrame] = []
    summary: list[dict] = []
    for sys_dir in sys_dirs:
        snaps = _load_snapshots_for_system(sys_dir)
        if snaps is None or snaps.empty:
            continue
        # Restrict to certified stations of the matching system.
        sys_name = snaps["system_id"].iloc[0] if "system_id" in snaps.columns else sys_dir.name
        cert_ids = cert_by_system.get(sys_name, set())
        if not cert_ids:
            continue
        snaps = snaps[snaps["station_id"].isin(cert_ids)]
        if snaps.empty:
            continue
        ind = _compute_indicators(snaps, capacity_map)
        ind["system_id"] = sys_name
        all_indicators.append(ind)
        summary.append({
            "system_id": sys_name,
            "snapshots": int(len(snaps)),
            "stations": int(ind["station_id"].nunique()),
            "a6a_overflow":     int((ind["rho_overflow"] > 0.01).sum()),
            "a6b_saturated":    int((ind["rho_empty"] > 0.40).sum()),
            "a6c_degenerate":   int((ind["rho_full_zero"] > 0.20).sum()),
        })
        print(f"  {sys_name:<32} "
              f"snaps={len(snaps):>6} "
              f"A6a={summary[-1]['a6a_overflow']:>3} "
              f"A6b={summary[-1]['a6b_saturated']:>3} "
              f"A6c={summary[-1]['a6c_degenerate']:>3}")

    if not all_indicators:
        print("No snapshots matched certified stations.")
        return

    indicators = pd.concat(all_indicators, ignore_index=True)
    indicators.to_csv(OUT_DIR / "indicators.csv", index=False)

    # Aggregate at the corpus level.
    total = {
        "n_stations_with_snapshots": int(indicators["station_id"].nunique()),
        "n_systems":  int(indicators["system_id"].nunique()),
        "n_snapshots": int(sum(s["snapshots"] for s in summary)),
        "a6a_overflow_total":   int((indicators["rho_overflow"] > 0.01).sum()),
        "a6b_saturated_total":  int((indicators["rho_empty"] > 0.40).sum()),
        "a6c_degenerate_total": int((indicators["rho_full_zero"] > 0.20).sum()),
    }
    total["a6_union_total"] = int(
        ((indicators["rho_overflow"] > 0.01)
         | (indicators["rho_empty"] > 0.40)
         | (indicators["rho_full_zero"] > 0.20)).sum()
    )
    print()
    print("E4 pilot aggregate:")
    for k, v in total.items():
        print(f"  {k:<32} {v}")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(
        json.dumps(
            {"per_system": summary, "aggregate": total},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
