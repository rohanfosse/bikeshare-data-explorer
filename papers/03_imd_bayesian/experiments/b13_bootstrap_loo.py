"""B13 -- Bootstrap CI and Leave-One-Out robustness for the tournament.

The B10 tournament reports point-estimate Spearman rho values
between five candidate metrics (IMD-4, IMD-3, volumetric,
Cerema, M alone) and four references (FUB, EMP, eco-counter,
BAAC). With panels of n in [25, 44] cities, point estimates
are not enough -- a single influential city can move rho by
0.1 or more.

This script adds two robustness checks to every cell:
  (i)  non-parametric bootstrap 95% CI on rho (1000 resamples)
  (ii) leave-one-out range: rho_min, rho_max across all
       n single-city deletions, with the most influential
       city named.

The output is the same tournament table as B10 plus
uncertainty bars and an influence-city annotation. The
intent is to be transparent about which results survive
under perturbation and which are fragile.

Outputs:
    outputs/b13_results.json
    outputs/b13_tournament_robust.pdf
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
from utils.data_loader import load_stations  # noqa: E402

B7_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b7_imd4_with_density.py"
spec7 = importlib.util.spec_from_file_location("b7", B7_PATH)
b7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(b7)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def bootstrap_rho(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 2026
) -> tuple[float, float, float]:
    """Return (rho, q025, q975) for non-parametric bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rho = sp_stats.spearmanr(x, y).statistic
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = sp_stats.spearmanr(x[idx], y[idx]).statistic
        boots[b] = r if np.isfinite(r) else np.nan
    boots = boots[np.isfinite(boots)]
    return float(rho), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def loo_rho(
    x: np.ndarray, y: np.ndarray, labels: list[str]
) -> tuple[float, float, str, str]:
    """Return (rho_min, rho_max, city_min, city_max) over leave-one-out."""
    n = len(x)
    rhos = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rhos[i] = sp_stats.spearmanr(x[mask], y[mask]).statistic
    i_min = int(np.argmin(rhos))
    i_max = int(np.argmax(rhos))
    return float(rhos[i_min]), float(rhos[i_max]), labels[i_min], labels[i_max]


def main() -> None:
    log.info("Loading panel + IMD-3 and IMD-4 city medians...")
    panel = load_panel()
    dock, cmm4, _, _, _ = b7.build_design(panel)
    cmm3 = cmm4[:, :3]
    res3 = b7.calibrate_k3(cmm3, panel.fub, panel.emp)
    res4 = b7.calibrate_k4(cmm4, panel.fub, panel.emp)

    rng = np.random.default_rng(2026)
    idx3 = rng.choice(len(res3["w_samples"]), 300, replace=False)
    idx4 = rng.choice(len(res4["w_samples"]), 300, replace=False)
    w3 = res3["w_samples"][idx3]
    w4 = res4["w_samples"][idx4]
    cs3 = dock[["M_norm", "I_norm", "T_norm"]].to_numpy()
    cs4 = dock[["M_norm", "I_norm", "T_norm", "D_norm"]].to_numpy()
    sta3 = (w3 @ cs3.T) * 100.0
    sta4 = (w4 @ cs4.T) * 100.0
    city_codes, city_index = pd.factorize(dock["city"].values)
    n_cities = len(city_index)
    city_imd3 = np.zeros((n_cities, sta3.shape[0]))
    city_imd4 = np.zeros((n_cities, sta4.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_imd3[ci] = sta3[:, mask].mean(axis=1)
        city_imd4[ci] = sta4[:, mask].mean(axis=1)
    median3 = dict(zip(city_index, np.median(city_imd3, axis=1)))
    median4 = dict(zip(city_index, np.median(city_imd4, axis=1)))

    # Other metrics
    stations = load_stations()
    dock_st = stations[stations["station_type"] == "docked_bike"]
    vol = dock_st.groupby("city").agg(
        n_stations=("station_id", "count"),
        mean_capacity=("capacity", "mean"),
    )
    vol["volumetric"] = np.log(vol["n_stations"] * vol["mean_capacity"].fillna(1))
    volumetric = dict(zip(vol.index, vol["volumetric"]))
    m_only = dock_st.groupby("city")["gtfs_heavy_stops_300m"].mean().to_dict()

    cer = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" / "cerema_cycling_infra_city.csv")
    cerema_dens = dict(zip(cer["city"], cer["infra_cyclable_km_per_km2"]))

    fub_lookup = dict(zip(panel.cities, panel.fub))
    emp_lookup = dict(zip(panel.cities, panel.emp))

    eco = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" / "eco_compteurs_city_usage.csv")
    eco_lookup = dict(zip(eco["city"], eco["eco_avg_daily_bike_counts"]))

    baac = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" / "baac_cyclist_accidents_city.csv")
    baac_col = next((c for c in baac.columns if c != "city"), None)
    baac_lookup = dict(zip(baac["city"], baac[baac_col])) if baac_col else {}

    metrics = {
        "IMD-4 (Bayesian)":   median4,
        "IMD-3 (Bayesian)":   median3,
        "Volumetric":         volumetric,
        "Cerema km/km^2":     cerema_dens,
        "M alone":            m_only,
    }
    references = {
        "FUB 2023":           fub_lookup,
        "EMP 2019":           emp_lookup,
        "Eco-counter":        eco_lookup,
        "BAAC accidents":     baac_lookup,
    }

    # Run bootstrap + LOO on every cell
    log.info("\nRunning bootstrap + LOO on all (metric, reference) pairs...")
    table: dict[str, dict[str, dict]] = {}
    for m_name, m_dict in metrics.items():
        table[m_name] = {}
        for r_name, r_dict in references.items():
            pairs = []
            labels = []
            for c, mv in m_dict.items():
                rv = r_dict.get(c)
                if rv is None or (isinstance(rv, float) and not np.isfinite(rv)):
                    continue
                if not np.isfinite(mv):
                    continue
                pairs.append((float(mv), float(rv)))
                labels.append(c)
            if len(pairs) < 7:
                table[m_name][r_name] = {"n": len(pairs), "rho": None}
                continue
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            rho, q025, q975 = bootstrap_rho(x, y, n_boot=1000)
            rho_min, rho_max, c_min, c_max = loo_rho(x, y, labels)
            table[m_name][r_name] = {
                "n": len(pairs), "rho": rho,
                "boot_q025": q025, "boot_q975": q975,
                "loo_min": rho_min, "loo_max": rho_max,
                "loo_city_min": c_min, "loo_city_max": c_max,
            }
            log.info("  %-22s vs %-18s  rho=%+.3f  CI=[%+.3f,%+.3f]  LOO=[%+.3f,%+.3f]  most-influential=%s",
                     m_name, r_name, rho, q025, q975, rho_min, rho_max, c_min)

    # Save
    out_json = OUT_DIR / "b13_results.json"
    out_json.write_text(json.dumps({
        "metrics": list(metrics.keys()),
        "references": list(references.keys()),
        "table": table,
    }, indent=2), encoding="utf-8")
    log.info("\nWrote %s", out_json)

    # Figure: heatmap with CI bars
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    metric_names = list(metrics.keys())
    ref_names = list(references.keys())
    mat = np.array([
        [table[m][r]["rho"] if table[m][r]["rho"] is not None else np.nan
         for r in ref_names] for m in metric_names
    ])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")
    ax.set_xticks(np.arange(len(ref_names)))
    ax.set_xticklabels(ref_names, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            cell = table[metric_names[i]][ref_names[j]]
            v = cell["rho"]
            n = cell["n"]
            if v is None or np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=8, color="#404040")
            else:
                ax.text(
                    j, i,
                    f"{v:+.2f}\n[{cell['boot_q025']:+.2f},{cell['boot_q975']:+.2f}]\nn={n}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.45 else "#202020",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman $\\rho$", fontsize=9)
    ax.set_title("B13: Tournament with 1000-boot 95% CI on $\\rho$",
                 fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b13_tournament_robust.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b13_tournament_robust.pdf")


if __name__ == "__main__":
    main()
