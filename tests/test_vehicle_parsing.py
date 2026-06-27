"""Migration regression tests for vehicle_collector: field parsing flows through
gbfs-toolkit while the app's snapshot schema/dtypes and reconstruct_trips are preserved."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from utils.vehicle_collector import (
    _VEHICLE_COLUMNS,
    _parse_vehicle_snapshot,
    reconstruct_trips,
)

_T0 = datetime(2026, 1, 5, 8, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 1, 5, 8, 2, tzinfo=timezone.utc)


def test_parse_vehicle_v2_bike_id_and_schema():
    raw = [
        {"bike_id": "b1", "lat": 48.85, "lon": 2.35, "is_disabled": True},
        {"bike_id": "b2", "lat": 48.86, "lon": 2.36, "station_id": "s9", "vehicle_type_id": "ebike"},
        {"lat": 48.87, "lon": 2.37},  # no id → dropped
    ]
    df = _parse_vehicle_snapshot(raw, _T0, "dott")
    assert list(df.columns) == _VEHICLE_COLUMNS
    assert len(df) == 2  # the id-less vehicle is dropped
    b1 = df[df.vehicle_id == "b1"].iloc[0]
    assert bool(b1["is_disabled"]) and b1["station_id"] is None
    assert df["lat"].dtype.kind == "f" and df["is_reserved"].dtype == bool
    b2 = df[df.vehicle_id == "b2"].iloc[0]
    assert b2["station_id"] == "s9" and b2["vehicle_type_id"] == "ebike"


def test_parse_vehicle_v3_vehicle_id():
    raw = [{"vehicle_id": "v1", "lat": 48.85, "lon": 2.35}]
    df = _parse_vehicle_snapshot(raw, _T0, "tier")
    assert df.iloc[0]["vehicle_id"] == "v1"


def test_parse_vehicle_empty():
    df = _parse_vehicle_snapshot([], _T0, "x")
    assert list(df.columns) == _VEHICLE_COLUMNS and df.empty


def test_reconstruct_trips_still_works():
    # one vehicle moves ~1.2 km between two snapshots → one trip
    snaps = pd.concat([
        _parse_vehicle_snapshot([{"vehicle_id": "v1", "lat": 48.850, "lon": 2.350}], _T0, "x"),
        _parse_vehicle_snapshot([{"vehicle_id": "v1", "lat": 48.860, "lon": 2.355}], _T1, "x"),
    ], ignore_index=True)
    snaps["fetched_at"] = pd.to_datetime(snaps["fetched_at"], utc=True)
    trips = reconstruct_trips(snaps, min_move_m=100.0)
    assert len(trips) == 1
    assert trips.iloc[0]["dist_m"] > 1000
    assert {"o_lat", "d_lat", "rebalancing"} <= set(trips.columns)
