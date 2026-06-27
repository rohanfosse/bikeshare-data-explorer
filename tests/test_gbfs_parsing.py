"""Migration regression tests: GBFS parsing now flows through gbfs-toolkit,
while preserving this app's snapshot schema/dtypes (incl. the GBFS 3.0 field rename)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from utils.gbfs_collector import _parse_status_snapshot, _STATUS_COLUMNS

_NOW = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)


def test_parse_status_v2_schema_and_dtypes():
    raw = [
        {"station_id": "a", "num_bikes_available": 5, "num_docks_available": 3},
        {"station_id": "b", "num_bikes_available": 0, "num_docks_available": 8, "is_renting": False},
    ]
    df = _parse_status_snapshot(raw, _NOW, "velib")
    assert list(df.columns) == _STATUS_COLUMNS
    assert df["num_bikes_available"].dtype.kind == "i"   # plain int preserved
    assert df["is_renting"].dtype == bool
    a = df[df.station_id == "a"].iloc[0]
    assert a["num_bikes_available"] == 5 and a["system_id"] == "velib"
    assert not bool(df[df.station_id == "b"].iloc[0]["is_renting"])


def test_parse_status_v3_num_vehicles_available():
    # GBFS 3.0 renamed the field — must still land in num_bikes_available
    raw = [{"station_id": "a", "num_vehicles_available": 7, "num_docks_available": 2}]
    df = _parse_status_snapshot(raw, _NOW, "velib")
    assert df.iloc[0]["num_bikes_available"] == 7


def test_parse_status_empty_returns_schema():
    df = _parse_status_snapshot([], _NOW, "velib")
    assert list(df.columns) == _STATUS_COLUMNS
    assert df.empty
