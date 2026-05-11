"""
Quick check across Lyft-operated US systems (Citi Bike, BlueBikes,
Capital Bikeshare) and the French dock-based subset for the
fraction of stations declaring capacity == 0 in station_information.
This is a candidate new anomaly class ('A6 zero-capacity dock')
not currently in the A1-A5 taxonomy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

TMP = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp")


def load_stations(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data["data"]["stations"])


def report(name: str, df: pd.DataFrame) -> dict:
    n = len(df)
    n_zero = int((df["capacity"] == 0).sum())
    n_nan = int(df["capacity"].isna().sum())
    nonzero = df.loc[df["capacity"].fillna(0) > 0, "capacity"]
    return {
        "system": name,
        "stations": n,
        "capacity_zero": n_zero,
        "capacity_zero_pct": round(100 * n_zero / n, 2) if n else 0.0,
        "capacity_nan": n_nan,
        "capacity_mean_all": round(float(df["capacity"].fillna(0).mean()), 2) if n else 0.0,
        "capacity_mean_active": round(float(nonzero.mean()), 2) if len(nonzero) else 0.0,
        "capacity_unique_nonzero": int(nonzero.nunique()),
    }


def main() -> None:
    results = []
    for name, fn in [
        ("Citi Bike NYC",        "citibike_stations.json"),
        ("BlueBikes Boston",     "bluebikes_stations.json"),
        ("Capital Bikeshare DC", "capital_stations.json"),
    ]:
        results.append(report(name, load_stations(TMP / fn)))

    # French dock-based subset from the released parquet
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    fr = pd.read_parquet(repo_root / "data" / "stations_gold_standard_final.parquet")
    fr_dock = fr[fr["station_type"] == "docked_bike"]
    results.append(report("French dock-based (Audit Catalogue)", fr_dock))

    out = Path(__file__).parent / "c0_anomaly_probe.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
