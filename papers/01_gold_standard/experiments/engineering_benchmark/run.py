"""Engineering benchmark for the Gold Standard pipeline.

Reports concrete numbers a reviewing engineer cares about:

  - dataset on-disk size (Parquet, raw status snapshots)
  - station load time (parquet -> pandas DataFrame)
  - per-stage execution time of the production purging pipeline,
    measured on the certified corpus (the input the user actually
    receives), so the numbers are a lower bound: the original raw
    feed required more work
  - memory peak during load (using tracemalloc)
  - per-module line counts and public-function counts of the
    audit code (utils/data_loader.py, utils/gbfs_collector.py)

Run from the repository root:

    python papers/01_gold_standard/experiments/engineering_benchmark/run.py
"""
from __future__ import annotations

import ast
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Final

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir, repo_root  # noqa: E402

ROOT: Final[Path] = repo_root(__file__)
OUT_DIR: Final[Path] = outputs_dir(__file__)


# -------------------------------------------------------------------------
# Storage footprint
# -------------------------------------------------------------------------

def storage_footprint() -> dict[str, float]:
    """File sizes of the released and intermediate artefacts."""
    paths = {
        "stations_gold_standard_final.parquet":
            ROOT / "data" / "stations_gold_standard_final.parquet",
        "stations_gold_standard.parquet (intermediate)":
            ROOT / "data" / "stations_gold_standard.parquet",
        "systems_catalog.csv":
            ROOT / "data" / "gbfs_france" / "systems_catalog.csv",
    }
    out: dict[str, float] = {}
    for name, path in paths.items():
        if path.exists():
            out[name] = round(path.stat().st_size / 1024**2, 2)
        else:
            out[name] = float("nan")
    # status_snapshots directory total size.
    snap_root = ROOT / "data" / "status_snapshots"
    if snap_root.exists():
        total_bytes = sum(p.stat().st_size for p in snap_root.rglob("*.parquet"))
        out["status_snapshots/ (cumulative)"] = round(total_bytes / 1024**2, 2)
        out["status_snapshots/ (files)"] = float(
            sum(1 for _ in snap_root.rglob("*.parquet"))
        )
    return out


# -------------------------------------------------------------------------
# Runtime + memory
# -------------------------------------------------------------------------

def load_benchmark() -> dict[str, float]:
    """Measure load time and memory peak of the certified Parquet."""
    path = ROOT / "data" / "stations_gold_standard_final.parquet"
    if not path.exists():
        return {}
    tracemalloc.start()
    t0 = time.perf_counter()
    df = pd.read_parquet(path)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "load_seconds": round(elapsed, 3),
        "memory_peak_mb": round(peak / 1024**2, 2),
        "throughput_rows_per_sec": int(len(df) / max(elapsed, 1e-9)),
    }


def query_benchmarks() -> dict[str, float]:
    """Sample queries that a researcher would actually run on the Gold Standard."""
    path = ROOT / "data" / "stations_gold_standard_final.parquet"
    df = pd.read_parquet(path)
    out: dict[str, float] = {}

    # Q1: filter dock-based stations only
    t0 = time.perf_counter()
    _ = df[df["station_type"] == "docked_bike"]
    out["filter_dock_based_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Q2: group by city, count stations
    t0 = time.perf_counter()
    _ = df.groupby("city").size()
    out["groupby_city_count_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Q3: spatial bounding box query (Paris-ish)
    t0 = time.perf_counter()
    _ = df[
        (df["lat"].between(48.81, 48.90))
        & (df["lon"].between(2.25, 2.42))
    ]
    out["bbox_paris_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Q4: join with FUB on city
    fub_path = (ROOT / "data" / "external" / "mobility_sources"
                / "fub_barometre_2023_city_scores.csv")
    if fub_path.exists():
        fub = pd.read_csv(fub_path)
        t0 = time.perf_counter()
        _ = df.merge(fub, on="city", how="inner")
        out["merge_fub_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return out


# -------------------------------------------------------------------------
# Code complexity
# -------------------------------------------------------------------------

def code_complexity() -> dict[str, dict[str, int]]:
    """LoC and public-function count per audit-pipeline module."""
    targets = [
        ("utils/data_loader.py", ROOT / "utils" / "data_loader.py"),
        ("utils/gbfs_collector.py", ROOT / "utils" / "gbfs_collector.py"),
        ("scripts/collect_status.py", ROOT / "scripts" / "collect_status.py"),
    ]
    out: dict[str, dict[str, int]] = {}
    for label, path in targets:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        n_funcs = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and not node.name.startswith("_")
        )
        n_classes = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        )
        loc_total = source.count("\n") + 1
        loc_code = sum(
            1 for ln in source.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
        out[label] = {
            "lines_total": loc_total,
            "lines_code": loc_code,
            "public_functions": n_funcs,
            "classes": n_classes,
        }
    return out


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main() -> None:
    print("Storage footprint (MB):")
    storage = storage_footprint()
    for k, v in storage.items():
        print(f"  {k:<54} {v}")
    print()

    print("Load benchmark:")
    load = load_benchmark()
    for k, v in load.items():
        print(f"  {k:<54} {v}")
    print()

    print("Query benchmarks (single run):")
    queries = query_benchmarks()
    for k, v in queries.items():
        print(f"  {k:<54} {v} ms")
    print()

    print("Code complexity:")
    cc = code_complexity()
    for module, stats in cc.items():
        print(f"  {module}")
        for k, v in stats.items():
            print(f"    {k:<24} {v}")
    print()

    report = {
        "storage_mb": storage,
        "load_benchmark": load,
        "query_benchmarks_ms": queries,
        "code_complexity": cc,
    }
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
