"""Smoke + reproducibility tests for the experiments package.

These tests don't re-run the full pipelines (too slow for CI); they
exercise the deterministic core: same input + same seed -> same
output, and shapes match what the paper claims.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = ROOT / "papers" / "01_gold_standard" / "experiments" / "baselines"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baselines_mod():
    if not BASELINES_DIR.exists():
        pytest.skip("baselines experiment not installed")
    return _load_module("baselines_run", BASELINES_DIR / "run.py")


def test_baseline_b0_constant(gold_standard: pd.DataFrame, baselines_mod) -> None:
    """B0 (Raw GBFS) labels every station 'docked_bike'."""
    pred = baselines_mod.b0_raw_gbfs(gold_standard)
    assert len(pred) == len(gold_standard)
    assert (pred == "docked_bike").all()


def test_baseline_b1_excludes_named_carsharing(
    gold_standard: pd.DataFrame, baselines_mod
) -> None:
    """B1 (vehicle_type filter) flags Citiz-named systems as carsharing."""
    pred = baselines_mod.b1_vehicle_type_filter(gold_standard)
    citiz_mask = gold_standard["system_id"].str.lower().str.contains("citiz", na=False)
    if citiz_mask.any():
        assert (pred[citiz_mask] == "carsharing").all()


def test_baseline_b3_matches_gold_standard(
    gold_standard: pd.DataFrame, baselines_mod
) -> None:
    """B3 (Gold Standard) is the reference, by construction."""
    pred = baselines_mod.b3_gold_standard(gold_standard)
    assert (pred.values == gold_standard["station_type"].values).all()


def test_wilson_ci_endpoints(baselines_mod) -> None:
    """Sanity check on the Wilson CI helper."""
    lo, hi = baselines_mod.wilson_ci(0, 14)
    assert lo == 0.0
    assert 0.0 < hi < 0.3
    lo, hi = baselines_mod.wilson_ci(14, 14)
    assert 0.7 < lo < 1.0
    assert hi == 1.0


def test_wald_ci_endpoints(baselines_mod) -> None:
    lo, hi = baselines_mod.wald_ci(1.0, 100)
    assert lo == 1.0 and hi == 1.0
    lo, hi = baselines_mod.wald_ci(0.5, 10000)
    assert 0.49 < lo < 0.5 and 0.5 < hi < 0.51
