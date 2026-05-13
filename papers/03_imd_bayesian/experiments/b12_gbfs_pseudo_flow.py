"""B12 -- GBFS-derived pseudo-flow proxy as 59-city usage validation.

The eco-counter panel covers only 28 cities. To validate the
IMD on the full IMD panel we derive a city-level pseudo-flow
from the station_status snapshots collected on 2026-03-02 and
2026-03-03. For each station and each consecutive snapshot pair
we compute |delta bikes| / capacity_proxy, then aggregate to
the city level by mean across stations and snapshots.

This metric is noisy (rebalancing operations are conflated with
rider trips) but its rank ordering tracks gross station-level
churn, which is a reasonable open-data proxy for cycling
activity intensity.

We then correlate the GBFS pseudo-flow against the IMD-3 and
IMD-4 city medians from B7.

Outputs:
    outputs/b12_pseudo_flow_results.json
    outputs/b12_pseudo_flow_correlation.pdf
"""
from __future__ import annotations

import json
import logging
import sys
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [HERE, *HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "papers" / "02_imd" / "experiments"))

from _common import load_panel  # noqa: E402

B7_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b7_imd4_with_density.py"
spec7 = importlib.util.spec_from_file_location("b7", B7_PATH)
b7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(b7)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def compute_pseudo_flow(snapshots_dir: Path, system_id: str) -> float | None:
    """Mean station-pseudo-flow per snapshot interval.

    For each pair of consecutive snapshots within a system:
      flow_s = |b_t1 - b_t0| / max(b_t0 + d_t0, 1)
    Then average across stations and across snapshot pairs.
    """
    sys_dir = snapshots_dir / system_id
    if not sys_dir.exists():
        return None
    files = sorted(sys_dir.glob("*.parquet"))
    files = [f for f in files if not f.name.startswith("station_info")]
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["station_id", "num_bikes_available",
                            "num_docks_available", "fetched_at"])
    df["capacity_proxy"] = df["num_bikes_available"] + df["num_docks_available"]
    df["capacity_proxy"] = df["capacity_proxy"].replace(0, 1)
    df = df.sort_values(["station_id", "fetched_at"])
    df["delta_b"] = df.groupby("station_id")["num_bikes_available"].diff().abs()
    df["flow"] = df["delta_b"] / df["capacity_proxy"]
    flow_clean = df["flow"].dropna()
    if len(flow_clean) < 10:
        return None
    return float(flow_clean.mean())


def main() -> None:
    log.info("Loading panel + IMD posteriors (via B7)...")
    panel = load_panel()
    dock, cmm4, _, _, _ = b7.build_design(panel)
    cmm3 = cmm4[:, :3]
    res3 = b7.calibrate_k3(cmm3, panel.fub, panel.emp)
    res4 = b7.calibrate_k4(cmm4, panel.fub, panel.emp)

    rng = np.random.default_rng(2026)
    idx = rng.choice(len(res3["w_samples"]), 300, replace=False)
    w3 = res3["w_samples"][idx]
    w4 = res4["w_samples"][idx]
    cs3 = dock[["M_norm", "I_norm", "T_norm"]].to_numpy()
    cs4 = dock[["M_norm", "I_norm", "T_norm", "D_norm"]].to_numpy()
    sta3 = (w3 @ cs3.T) * 100.0
    sta4 = (w4 @ cs4.T) * 100.0
    city_codes, city_index = pd.factorize(dock["city"].values)
    n_cities = len(city_index)
    city3 = np.zeros((n_cities, sta3.shape[0]))
    city4 = np.zeros((n_cities, sta4.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city3[ci] = sta3[:, mask].mean(axis=1)
        city4[ci] = sta4[:, mask].mean(axis=1)
    median3 = dict(zip(city_index, np.median(city3, axis=1)))
    median4 = dict(zip(city_index, np.median(city4, axis=1)))

    # GBFS pseudo-flow per system
    log.info("\nComputing GBFS pseudo-flow per system...")
    snapshots_dir = ROOT / "data" / "status_snapshots"
    feed_diag = pd.read_csv(snapshots_dir / "feed_diagnostic.csv")
    log.info("  feed_diagnostic.csv: %d systems documented", len(feed_diag))

    rows = []
    for _, r in feed_diag.iterrows():
        if not r.get("has_status_feed", False):
            continue
        sys_id = r["system_id"]
        city = r["city"]
        flow = compute_pseudo_flow(snapshots_dir, sys_id)
        if flow is None or not np.isfinite(flow):
            continue
        rows.append({"system_id": sys_id, "city": city,
                      "pseudo_flow": flow})

    flow_df = pd.DataFrame(rows)
    # Aggregate to city level (mean across systems sharing a city)
    flow_by_city = flow_df.groupby("city")["pseudo_flow"].mean().to_dict()
    log.info("  computed pseudo-flow for %d cities", len(flow_by_city))

    # Correlate with IMD-3, IMD-4
    pairs3 = []
    pairs4 = []
    for c, f in flow_by_city.items():
        if c in median3:
            pairs3.append((float(f), float(median3[c])))
        if c in median4:
            pairs4.append((float(f), float(median4[c])))

    if len(pairs3) >= 5:
        x = np.array([p[0] for p in pairs3])
        y = np.array([p[1] for p in pairs3])
        rho3 = sp_stats.spearmanr(x, y).statistic
        log.info("  rho(IMD-3, GBFS pseudo-flow) = %.3f  (n=%d)", rho3, len(pairs3))
    else:
        rho3 = None
    if len(pairs4) >= 5:
        x = np.array([p[0] for p in pairs4])
        y = np.array([p[1] for p in pairs4])
        rho4 = sp_stats.spearmanr(x, y).statistic
        log.info("  rho(IMD-4, GBFS pseudo-flow) = %.3f  (n=%d)", rho4, len(pairs4))
    else:
        rho4 = None

    # Bootstrap CI on rho(IMD-4, pseudo-flow)
    if len(pairs4) >= 10:
        boot_rhos = []
        rng_b = np.random.default_rng(2026)
        x = np.array([p[0] for p in pairs4])
        y = np.array([p[1] for p in pairs4])
        n = len(x)
        for _ in range(1000):
            idx_b = rng_b.choice(n, n, replace=True)
            r = sp_stats.spearmanr(x[idx_b], y[idx_b]).statistic
            if np.isfinite(r):
                boot_rhos.append(r)
        boot_rhos = np.array(boot_rhos)
        log.info("  bootstrap CI on rho(IMD-4, GBFS): 95%% = [%.3f, %.3f]",
                 float(np.percentile(boot_rhos, 2.5)),
                 float(np.percentile(boot_rhos, 97.5)))
        ci_low = float(np.percentile(boot_rhos, 2.5))
        ci_high = float(np.percentile(boot_rhos, 97.5))
    else:
        ci_low = ci_high = None

    # Save results
    results = {
        "n_systems_with_flow": int(len(rows)),
        "n_cities_with_flow": int(len(flow_by_city)),
        "n_cities_matched_imd3": int(len(pairs3)),
        "n_cities_matched_imd4": int(len(pairs4)),
        "rho_imd3_pseudo_flow": float(rho3) if rho3 is not None else None,
        "rho_imd4_pseudo_flow": float(rho4) if rho4 is not None else None,
        "bootstrap_ci_imd4": [ci_low, ci_high] if ci_low is not None else None,
        "city_pseudo_flows": flow_by_city,
    }
    out_json = OUT_DIR / "b12_pseudo_flow_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: scatter IMD-4 vs pseudo-flow
    if len(pairs4) >= 5:
        x = np.array([p[0] for p in pairs4])
        y = np.array([p[1] for p in pairs4])
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.scatter(x, y, s=50, color="#1F3A6B", alpha=0.7,
                    edgecolor="white", linewidth=0.5)
        # Label the city for each point
        # find labels
        labels = [c for c, f in flow_by_city.items() if c in median4]
        for label, xi, yi in zip(labels, x, y):
            if yi > np.median(y) or xi > np.median(x):
                ax.annotate(label, (xi, yi), fontsize=7,
                             xytext=(3, 3), textcoords="offset points",
                             color="#202020")
        ax.set_xlabel(r"GBFS pseudo-flow (mean $|\Delta\,$bikes$|$/capacity)")
        ax.set_ylabel("Bayesian IMD-4 (posterior median)")
        ax.set_title(f"B12: IMD-4 vs GBFS pseudo-flow  "
                     f"$\\rho = {rho4:+.3f}$ (n={len(pairs4)})",
                     fontsize=10)
        ax.grid(True, color="#E5E5E5", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "b12_pseudo_flow_correlation.pdf",
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("  wrote b12_pseudo_flow_correlation.pdf")


if __name__ == "__main__":
    main()
