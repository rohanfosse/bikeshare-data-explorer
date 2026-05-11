"""Draw the stratified 500-station sample for experiment E1.

Implements the sampling protocol described in
``papers/01_gold_standard/experiments/e1_irr/README.md``: stratified random draw over declared
``station_type`` with over-sampling of systems flagged A2 or A3 by
the rule-based audit, blinded for annotators (the audit verdict is
NOT shipped in the annotator CSV but kept in a separate answer-key
CSV that only the researchers see).

Run from the repository root:

    python papers/01_gold_standard/experiments/e1_irr/sample_stations.py

Outputs land in ``papers/01_gold_standard/experiments/e1_irr/``:

    e1_sample_v1.csv       - shipped to annotators
    e1_answer_key_v1.csv   - researcher-only

The seed is fixed for reproducibility. Re-running produces the
same 500 stations.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir, repo_root  # noqa: E402

ROOT: Final[Path] = repo_root(__file__)
OUT_DIR: Final[Path] = outputs_dir(__file__)
SEED: Final[int] = 42

# Sample size budget per declared station_type (sums to 500).
# Free-floating is over-sampled relative to its A3 risk; car-sharing
# is included only as a sanity check on the A1 detector.
SAMPLE_BUDGET: Final[dict[str, int]] = {
    "docked_bike":   180,
    "free_floating": 260,
    "carsharing":     60,
}


@dataclass(frozen=True)
class SamplingReport:
    """Diagnostic counts after sampling, for the audit log."""

    total_drawn: int
    per_type: dict[str, int]
    per_system: dict[str, int]
    a2_a3_systems_oversampled: int


def _load_stations() -> pd.DataFrame:
    """Load the Gold Standard parquet (full corpus)."""
    path = ROOT / "data" / "stations_gold_standard_final.parquet"
    if not path.exists():
        sys.exit(f"ERROR: dataset not found at {path}")
    return pd.read_parquet(path)


def _flag_a2_a3_systems(df: pd.DataFrame) -> set[str]:
    """Return system_ids whose capacity profile *suggests* A2 or A3.

    A2 candidates have a zero-variance, non-zero capacity column.
    A3 candidates are systems with at least one ``free_floating``
    station. The function does not perform the audit itself; it
    only identifies systems that deserve over-sampling in the E1
    stratification.
    """
    by_system = df.groupby("system_id")
    a2 = (
        by_system["capacity"]
        .agg(["std", "mean"])
        .query("std == 0 and mean > 0")
        .index.tolist()
    )
    a3 = (
        df.loc[df["station_type"] == "free_floating", "system_id"]
        .unique()
        .tolist()
    )
    return set(a2) | set(a3)


def _stratified_sample(
    df: pd.DataFrame,
    budget: dict[str, int],
    a2a3_systems: set[str],
    seed: int,
) -> pd.DataFrame:
    """Draw the stratified sample with A2/A3 over-sampling.

    For each station_type bucket, allocate the budget so that
    every A2/A3 system contributes at least one station, then fill
    the remaining slots proportionally to system size.
    """
    rng = np.random.default_rng(seed)
    chunks: list[pd.DataFrame] = []
    for st_type, target_n in budget.items():
        pool = df[df["station_type"] == st_type].copy()
        if pool.empty:
            continue

        # Reserve at least one slot per flagged system, capped at budget.
        flagged_in_pool = pool[pool["system_id"].isin(a2a3_systems)]
        reserved = (
            flagged_in_pool.groupby("system_id")
            .sample(n=1, random_state=seed)
            if not flagged_in_pool.empty
            else pd.DataFrame()
        )
        remainder = max(target_n - len(reserved), 0)
        remaining_pool = pool.drop(index=reserved.index, errors="ignore")
        sampled = (
            remaining_pool.sample(
                n=min(remainder, len(remaining_pool)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            if remainder > 0
            else pd.DataFrame()
        )
        chunks.extend([reserved, sampled])

    out = pd.concat(chunks, ignore_index=True)
    # Final shuffle to randomise the row order presented to annotators.
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def _build_annotator_csv(sample: pd.DataFrame) -> pd.DataFrame:
    """Return the blinded CSV shipped to annotators.

    Columns are limited to what the annotator needs to apply the
    rules of ``annotation_guide.md``: identity, geolocation,
    declared capacity, and any virtual-anchor hint. The rule-based
    ``station_type`` is intentionally dropped.
    """
    is_virtual = (
        sample["station_type"].eq("free_floating")
        if "station_type" in sample.columns
        else pd.Series([False] * len(sample))
    )
    blinded = pd.DataFrame(
        {
            "row_id": np.arange(1, len(sample) + 1),
            "station_id": sample["station_id"],
            "system_id": sample["system_id"],
            "system_name": sample["system_name"],
            "city": sample["city"],
            "station_name": sample["station_name"],
            "lat": sample["lat"].round(5),
            "lon": sample["lon"].round(5),
            "capacity": sample["capacity"],
            "is_virtual_station_hint": is_virtual,
            "label": "",  # to be filled by the annotator
        }
    )
    return blinded


def _build_answer_key(sample: pd.DataFrame) -> pd.DataFrame:
    """Return the researcher-only answer key with the rule-based verdict."""
    return pd.DataFrame(
        {
            "row_id": np.arange(1, len(sample) + 1),
            "station_id": sample["station_id"],
            "system_id": sample["system_id"],
            "rule_based_station_type": sample["station_type"],
            # Map station_type back into the A* taxonomy used in annotation.
            "rule_based_label": sample["station_type"].map(
                {
                    "docked_bike":   "ok",
                    "free_floating": "A3",
                    "carsharing":    "A1",
                }
            ),
        }
    )


def main() -> None:
    print("Loading Gold Standard corpus...")
    stations = _load_stations()
    print(f"  {len(stations):,} stations, "
          f"{stations['system_id'].nunique()} systems")

    flagged = _flag_a2_a3_systems(stations)
    print(f"  {len(flagged)} systems flagged as A2/A3 candidates "
          f"(eligible for over-sampling)")

    sample = _stratified_sample(stations, SAMPLE_BUDGET, flagged, SEED)
    print(f"  drew {len(sample):,} stations")

    annotator_csv = _build_annotator_csv(sample)
    answer_key = _build_answer_key(sample)

    out_sample = OUT_DIR / "e1_sample_v1.csv"
    out_key = OUT_DIR / "e1_answer_key_v1.csv"
    annotator_csv.to_csv(out_sample, index=False, encoding="utf-8")
    answer_key.to_csv(out_key, index=False, encoding="utf-8")

    print(f"  wrote {out_sample.relative_to(ROOT)}")
    print(f"  wrote {out_key.relative_to(ROOT)}")
    print()
    print("Distribution by station_type in the sample:")
    print(sample["station_type"].value_counts().to_string())
    print()
    print("Number of distinct systems represented:",
          sample["system_id"].nunique())


if __name__ == "__main__":
    main()
