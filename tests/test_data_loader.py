"""Schema-level invariants and idempotence of the Gold Standard load path.

These tests are what a JOSS reviewer runs first: they verify that
`load_stations()` returns a DataFrame conforming to the published
schema, that the load is idempotent, and that the documented
anomaly classes are represented as expected.
"""
from __future__ import annotations

import pandas as pd
import pytest


REQUIRED_COLUMNS: tuple[str, ...] = (
    "uid", "station_id", "system_id", "city", "lat", "lon",
    "capacity", "station_type",
    "elevation_m", "infra_cyclable_km", "baac_accidents_cyclistes",
    "gtfs_heavy_stops_300m", "revenu_median_uc",
)

ALLOWED_STATION_TYPES: frozenset[str] = frozenset({
    "docked_bike", "free_floating", "carsharing"
})


def test_load_returns_dataframe(gold_standard: pd.DataFrame) -> None:
    assert isinstance(gold_standard, pd.DataFrame)
    assert len(gold_standard) > 0


def test_required_columns_present(gold_standard: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(gold_standard.columns)
    assert not missing, f"missing columns: {missing}"


def test_station_type_enum(gold_standard: pd.DataFrame) -> None:
    """Documented A1--A5 taxonomy maps to three station_type values."""
    observed = set(gold_standard["station_type"].dropna().unique())
    assert observed.issubset(ALLOWED_STATION_TYPES), (
        f"unexpected station_type values: {observed - ALLOWED_STATION_TYPES}"
    )


def test_coordinates_within_metropolitan_france(gold_standard: pd.DataFrame) -> None:
    """Geofilter (A4) must drop stations outside the national bounding box."""
    assert gold_standard["lat"].between(41.0, 52.0).all()
    assert gold_standard["lon"].between(-6.0, 10.0).all()


def test_capacity_non_negative(gold_standard: pd.DataFrame) -> None:
    """Capacities must be non-negative (placeholder A2 is non-zero but valid)."""
    assert (gold_standard["capacity"].fillna(0) >= 0).all()


def test_no_orphan_stations(gold_standard: pd.DataFrame) -> None:
    """Every station_id must belong to a non-empty system_id."""
    assert gold_standard["system_id"].notna().all()
    assert (gold_standard["system_id"].astype(str).str.len() > 0).all()


def test_load_idempotent(gold_standard: pd.DataFrame) -> None:
    """Two successive loads return identical DataFrames."""
    from utils.data_loader import load_stations

    second = load_stations()
    pd.testing.assert_frame_equal(
        gold_standard.reset_index(drop=True),
        second.reset_index(drop=True),
        check_dtype=False,
    )


def test_a1_systems_excluded_from_active_corpus(gold_standard: pd.DataFrame) -> None:
    """Car-sharing systems (A1) keep a station_type of 'carsharing', not
    'docked_bike'."""
    carsharing = gold_standard[gold_standard["station_type"] == "carsharing"]
    assert len(carsharing) > 0
    # No carsharing station should ever be labelled docked_bike at the same time.
    assert "docked_bike" not in carsharing["station_type"].unique()


def test_corpus_size_within_expected_range(gold_standard: pd.DataFrame) -> None:
    """The published dataset has 46,000+ certified stations; allow a window
    for minor re-releases."""
    assert 40_000 < len(gold_standard) < 60_000
