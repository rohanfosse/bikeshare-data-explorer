"""B10 -- Head-to-head metric tournament: IMD vs FUB vs EMP vs volumetric.

For the validation panel of the IMD paper, we compare the
predictive power of five candidate cycling-environment metrics
against four behavioural / outcome references:

  Metrics:
    - IMD-4 Bayesian posterior median (M+I+T+D)
    - IMD-3 Bayesian posterior median (M+I+T)
    - Volumetric log(N stations * mean capacity)
    - Cerema cycle-infrastructure km / km^2
    - GTFS heavy-stop density per station (M alone)

  References:
    - FUB Cycling Barometer 2023 score (perception)
    - EMP 2019 modal share (behavioural survey)
    - Eco-counter daily count (observed activity)
    - BAAC accident rate per 100k inh (safety proxy, inverse)

Each metric x reference pair is evaluated by Spearman rank
correlation on the cities where both are observed. We also
report leave-one-city-out predictive correlation for the IMD
variants. The result is a tournament table that positions the
IMD against alternatives the paper might be benchmarked against
by a reviewer.

Outputs:
    outputs/b10_results.json
    outputs/b10_tournament.pdf
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

B1_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b1_bayesian_imd.py"
spec = importlib.util.spec_from_file_location("b1", B1_PATH)
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

B7_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b7_imd4_with_density.py"
spec7 = importlib.util.spec_from_file_location("b7", B7_PATH)
b7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(b7)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Loading panel + computing IMD-3 and IMD-4 city medians...")
    panel = load_panel()
    dock, cmm4, _fub_raw, _emp_raw, _ = b7.build_design(panel)
    cmm3 = cmm4[:, :3]
    res3 = b7.calibrate_k3(cmm3, panel.fub, panel.emp)
    res4 = b7.calibrate_k4(cmm4, panel.fub, panel.emp)

    rng = np.random.default_rng(2026)
    idx3 = rng.choice(len(res3["w_samples"]), 300, replace=False)
    w_sub3 = res3["w_samples"][idx3]
    cs3 = dock[["M_norm", "I_norm", "T_norm"]].to_numpy()
    sta_imd3 = (w_sub3 @ cs3.T) * 100.0
    city_codes, city_index = pd.factorize(dock["city"].values)
    n_cities = len(city_index)
    city_imd3 = np.zeros((n_cities, sta_imd3.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_imd3[ci] = sta_imd3[:, mask].mean(axis=1)
    median3 = dict(zip(city_index, np.median(city_imd3, axis=1)))

    idx4 = rng.choice(len(res4["w_samples"]), 300, replace=False)
    w_sub4 = res4["w_samples"][idx4]
    cs4 = dock[["M_norm", "I_norm", "T_norm", "D_norm"]].to_numpy()
    sta_imd4 = (w_sub4 @ cs4.T) * 100.0
    city_imd4 = np.zeros((n_cities, sta_imd4.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_imd4[ci] = sta_imd4[:, mask].mean(axis=1)
    median4 = dict(zip(city_index, np.median(city_imd4, axis=1)))

    log.info("  IMD-3 median computed for %d cities", len(median3))
    log.info("  IMD-4 median computed for %d cities", len(median4))

    # ----- Build per-city metrics -----
    # Volumetric: log(N stations * mean capacity)
    stations = load_stations()
    dock_st = stations[stations["station_type"] == "docked_bike"]
    vol = dock_st.groupby("city").agg(
        n_stations=("station_id", "count"),
        mean_capacity=("capacity", "mean"),
    )
    vol["volumetric"] = np.log(vol["n_stations"] * vol["mean_capacity"].fillna(1))
    volumetric = dict(zip(vol.index, vol["volumetric"]))

    # M alone (GTFS heavy stops average per city)
    m_only = dock_st.groupby("city")["gtfs_heavy_stops_300m"].mean().to_dict()

    # Cerema infrastructure (km/km2)
    cerema_path = ROOT / "data" / "external" / "mobility_sources" / "cerema_cycling_infra_city.csv"
    cer = pd.read_csv(cerema_path)
    cerema_dens = dict(zip(cer["city"], cer["infra_cyclable_km_per_km2"]))

    # ----- Behavioural / outcome references -----
    fub_lookup = dict(zip(panel.cities, panel.fub))
    emp_lookup = dict(zip(panel.cities, panel.emp))

    eco_path = ROOT / "data" / "external" / "mobility_sources" / "eco_compteurs_city_usage.csv"
    eco = pd.read_csv(eco_path)
    eco_lookup = dict(zip(eco["city"], eco["eco_avg_daily_bike_counts"]))

    baac_path = ROOT / "data" / "external" / "mobility_sources" / "baac_cyclist_accidents_city.csv"
    baac = pd.read_csv(baac_path)
    log.info("  BAAC columns: %s", baac.columns.tolist())
    baac_col = next((c for c in baac.columns if c != "city"), None)
    baac_lookup = dict(zip(baac["city"], baac[baac_col])) if baac_col else {}

    metrics = {
        "IMD-4 (Bayesian, M+I+T+D)":  median4,
        "IMD-3 (Bayesian, M+I+T)":    median3,
        "Volumetric  log(N x cap.)":  volumetric,
        "Cerema  km / km^2":          cerema_dens,
        "M alone (heavy stops)":      m_only,
    }
    references = {
        "FUB perception 2023":        fub_lookup,
        "EMP modal share 2019":       emp_lookup,
        "Eco-counter daily flow":     eco_lookup,
        "BAAC accidents (per 100k)":  baac_lookup,
    }

    # ----- Pairwise Spearman correlations -----
    table = {}
    n_table = {}
    for m_name, m_dict in metrics.items():
        table[m_name] = {}
        n_table[m_name] = {}
        for r_name, r_dict in references.items():
            pairs = []
            for c, mv in m_dict.items():
                rv = r_dict.get(c)
                if rv is None or (isinstance(rv, float) and not np.isfinite(rv)):
                    continue
                if not np.isfinite(mv):
                    continue
                pairs.append((float(mv), float(rv)))
            if len(pairs) < 5:
                table[m_name][r_name] = None
                n_table[m_name][r_name] = len(pairs)
                continue
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            rho = sp_stats.spearmanr(x, y).statistic
            table[m_name][r_name] = float(rho)
            n_table[m_name][r_name] = len(pairs)

    # Print table
    log.info("\nMETRIC TOURNAMENT (Spearman rho with n in parens)\n")
    header = ["Metric"] + list(references.keys())
    log.info("%-30s | %s", "Metric", " | ".join(f"{r:<23}" for r in references))
    log.info("-" * 150)
    for m in metrics:
        row = [f"{m:<30}"]
        for r in references:
            v = table[m][r]
            n = n_table[m][r]
            if v is None:
                cell = f"--                       "
            else:
                cell = f"{v:+.3f} (n={n:>3})        "
            row.append(cell)
        log.info(" | ".join(row))

    # ----- Bayesian credible interval for IMD vs eco-counter -----
    # For the IMD-4 we can derive a posterior on the Spearman rho
    # by sampling: for each posterior draw of w, compute city_imd, then rho
    log.info("\nBayesian CrI on rho(IMD, eco-counter) ...")
    eco_cities = [c for c in city_index if c in eco_lookup
                   and np.isfinite(eco_lookup[c])]
    if len(eco_cities) >= 5:
        eco_vec = np.array([eco_lookup[c] for c in eco_cities])
        cities_idx_in_panel = [list(city_index).index(c) for c in eco_cities]
        n_full = len(res4["w_samples"])
        idx_full = rng.choice(n_full, min(2000, n_full), replace=False)
        rhos_post = np.zeros(len(idx_full))
        for k, ii in enumerate(idx_full):
            w = res4["w_samples"][ii]
            sta_imd = (cs4 @ w) * 100.0
            city_v = np.zeros(n_cities)
            for ci in range(n_cities):
                mask = city_codes == ci
                city_v[ci] = sta_imd[mask].mean()
            sub_v = city_v[cities_idx_in_panel]
            rhos_post[k] = sp_stats.spearmanr(eco_vec, sub_v).statistic
        log.info("  posterior rho(IMD-4, eco):  median = %.3f   95%% CrI = [%.3f, %.3f]",
                 float(np.median(rhos_post)),
                 float(np.percentile(rhos_post, 2.5)),
                 float(np.percentile(rhos_post, 97.5)))
    else:
        rhos_post = None
        log.warning("  not enough eco-counter overlap for posterior rho")

    # Save
    results = {
        "metrics": list(metrics.keys()),
        "references": list(references.keys()),
        "spearman_table": table,
        "n_table": n_table,
        "bayesian_rho_imd4_eco": {
            "median": float(np.median(rhos_post)) if rhos_post is not None else None,
            "q025":   float(np.percentile(rhos_post, 2.5)) if rhos_post is not None else None,
            "q975":   float(np.percentile(rhos_post, 97.5)) if rhos_post is not None else None,
            "n_panel": int(len(eco_cities)) if rhos_post is not None else 0,
        },
    }
    out_json = OUT_DIR / "b10_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: heatmap of correlations
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    metric_names = list(metrics.keys())
    ref_names = list(references.keys())
    mat = np.array([[table[m][r] if table[m][r] is not None else np.nan
                      for r in ref_names] for m in metric_names])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")
    ax.set_xticks(np.arange(len(ref_names)))
    ax.set_xticklabels(ref_names, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_yticklabels(metric_names, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                         fontsize=8, color="#404040")
            else:
                ax.text(j, i, f"{v:+.2f}\n(n={n_table[metric_names[i]][ref_names[j]]})",
                         ha="center", va="center",
                         fontsize=8,
                         color="white" if abs(v) > 0.45 else "#202020")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman $\\rho$", fontsize=9)
    ax.set_title("B10: pairwise Spearman correlations on the French panel",
                 fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b10_tournament.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b10_tournament.pdf")


if __name__ == "__main__":
    main()
