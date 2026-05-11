"""Sanity tests for the IMD aggregate computed from the Gold Standard."""
from __future__ import annotations

import pandas as pd
import pytest


def test_imd_cities_shape(gold_standard: pd.DataFrame) -> None:
    from utils.data_loader import compute_imd_cities

    imd = compute_imd_cities(gold_standard)
    assert isinstance(imd, pd.DataFrame)
    assert len(imd) > 0
    assert "city" in imd.columns
    assert "IMD" in imd.columns


def test_imd_values_in_unit_range(gold_standard: pd.DataFrame) -> None:
    """The IMD composite is bounded in [0, 100] by construction."""
    from utils.data_loader import compute_imd_cities

    imd = compute_imd_cities(gold_standard).dropna(subset=["IMD"])
    assert (imd["IMD"] >= 0).all()
    assert (imd["IMD"] <= 100).all()


def test_no_macro_regional_cities_in_imd(gold_standard: pd.DataFrame) -> None:
    """Compute_imd_cities filters out macro-regional entries (A5)."""
    from utils.data_loader import compute_imd_cities

    imd = compute_imd_cities(gold_standard)
    excluded = {"France", "FR", "Grand Est", "Basque Country"}
    intersection = set(imd["city"]) & excluded
    assert not intersection, f"macro-regional entries leaked: {intersection}"


def test_imd_idempotent(gold_standard: pd.DataFrame) -> None:
    from utils.data_loader import compute_imd_cities

    a = compute_imd_cities(gold_standard).sort_values("city").reset_index(drop=True)
    b = compute_imd_cities(gold_standard).sort_values("city").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
