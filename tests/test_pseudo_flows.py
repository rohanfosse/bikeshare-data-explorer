"""Tests for ``compute_pseudo_flows`` and the GBFSCollector skeleton."""
from __future__ import annotations

import pandas as pd
import pytest


def test_pseudo_flows_basic(synthetic_snapshots: pd.DataFrame) -> None:
    """Two snapshots, two stations, departures/arrivals are signed correctly."""
    from utils.gbfs_collector import compute_pseudo_flows

    flows = compute_pseudo_flows(synthetic_snapshots)
    # Two stations x one delta each => two rows.
    assert len(flows) == 2

    s1 = flows[flows["station_id"] == "s1"].iloc[0]
    assert s1["delta_bikes"] == -4
    assert s1["departures_est"] == 4
    assert s1["arrivals_est"] == 0
    assert s1["net_flow_est"] == -4

    s2 = flows[flows["station_id"] == "s2"].iloc[0]
    assert s2["delta_bikes"] == 2
    assert s2["departures_est"] == 0
    assert s2["arrivals_est"] == 2
    assert s2["net_flow_est"] == 2


def test_pseudo_flows_empty() -> None:
    """An empty snapshot frame must return an empty result, not raise."""
    from utils.gbfs_collector import compute_pseudo_flows

    empty = pd.DataFrame(
        columns=[
            "fetched_at", "system_id", "station_id",
            "num_bikes_available", "num_docks_available",
            "is_renting", "is_returning",
        ]
    )
    flows = compute_pseudo_flows(empty)
    assert isinstance(flows, pd.DataFrame)
    assert len(flows) == 0


def test_collector_instantiation() -> None:
    """``GBFSCollector`` must instantiate without network access."""
    from utils.gbfs_collector import GBFSCollector

    c = GBFSCollector(system_ids=["Paris"], min_stations=1)
    assert hasattr(c, "system_ids")
    assert "Paris" in c.system_ids
