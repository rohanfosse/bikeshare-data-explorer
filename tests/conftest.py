"""Shared pytest fixtures for the bikeshare-data-explorer test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def gold_standard_path() -> Path:
    """Path to the released Gold Standard Parquet."""
    return ROOT / "data" / "stations_gold_standard_final.parquet"


@pytest.fixture(scope="session")
def gold_standard(gold_standard_path: Path) -> pd.DataFrame:
    """The released Gold Standard, loaded with the public API."""
    pytest.importorskip("streamlit", reason="utils.data_loader imports streamlit")
    from utils.data_loader import load_stations  # noqa: WPS433

    if not gold_standard_path.exists():
        pytest.skip(f"Dataset not available at {gold_standard_path}")
    return load_stations()


@pytest.fixture
def synthetic_snapshots() -> pd.DataFrame:
    """Two-snapshot toy series for ``compute_pseudo_flows`` tests."""
    return pd.DataFrame(
        [
            # t0
            {"fetched_at": "2026-01-01T08:00:00Z", "system_id": "sys",
             "station_id": "s1", "num_bikes_available": 10,
             "num_docks_available": 5, "is_renting": True,
             "is_returning": True},
            {"fetched_at": "2026-01-01T08:00:00Z", "system_id": "sys",
             "station_id": "s2", "num_bikes_available": 3,
             "num_docks_available": 12, "is_renting": True,
             "is_returning": True},
            # t1, 1 hour later: s1 lost 4 bikes, s2 gained 2
            {"fetched_at": "2026-01-01T09:00:00Z", "system_id": "sys",
             "station_id": "s1", "num_bikes_available": 6,
             "num_docks_available": 9, "is_renting": True,
             "is_returning": True},
            {"fetched_at": "2026-01-01T09:00:00Z", "system_id": "sys",
             "station_id": "s2", "num_bikes_available": 5,
             "num_docks_available": 10, "is_renting": True,
             "is_returning": True},
        ]
    )
