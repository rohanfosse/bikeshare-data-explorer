"""B7 -- IMD-4 with population density: D-component test.

Tests whether adding a commune-level population density component
to the Bayesian IMD-3 improves:
  H1: rho vs eco-counter daily flows (+0.05 expected)
  H2: w_M remains dominant (>0.5)
  H3: Top-3 ranking changes (challenge to "medium-sized > metro")

Fetches commune populations and surfaces from geo.api.gouv.fr,
joins to dock-based stations via code_commune, computes density
as pop / area_km2, adds log(density) as the 4th component D.
Re-runs Bayesian MH on the 4-simplex.

Outputs:
    outputs/b7_results.json
    outputs/b7_weights_imd4.pdf
    outputs/b7_ranking_compare.pdf
"""
from __future__ import annotations

import json
import logging
import sys
import importlib.util
from pathlib import Path
from urllib.request import Request, urlopen

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

OUT_DIR = Path(__file__).parent / "outputs"
CACHE_DIR = ROOT / "data" / "external" / "insee_communes"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch INSEE commune population + surface
# ---------------------------------------------------------------------------

def fetch_communes_meta() -> pd.DataFrame:
    """Fetch population and surface (hectares) for all French communes.

    Uses geo.api.gouv.fr which is free, stable, and serves the
    INSEE COG metadata.
    """
    cache_path = CACHE_DIR / "communes_meta.csv"
    if cache_path.exists():
        log.info("  reading cached communes metadata: %s", cache_path)
        return pd.read_csv(cache_path, dtype={"code": str})
    log.info("  fetching commune metadata from geo.api.gouv.fr...")
    url = "https://geo.api.gouv.fr/communes?fields=nom,code,population,surface&format=json"
    req = Request(url, headers={"User-Agent": "bikeshare-research/1.0"})
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    df = pd.DataFrame(data)
    df = df.rename(columns={"code": "code_commune"})
    df["code_commune"] = df["code_commune"].astype(str)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["surface_ha"] = pd.to_numeric(df["surface"], errors="coerce")
    df["surface_km2"] = df["surface_ha"] / 100.0
    df["density_per_km2"] = df["population"] / df["surface_km2"].replace(0, np.nan)
    log.info("  retrieved %d communes", len(df))
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Build IMD-4 design
# ---------------------------------------------------------------------------

def build_design(panel) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (dock-station df with 4 norm cols, city-mean component matrix
    of shape (n_cities, 4), and FUB, EMP, code_to_density)."""
    stations = load_stations()
    dock = b1.normalise_components(stations)
    log.info("  %d dock stations, %d cities", len(dock), dock["city"].nunique())

    # Population density via INSEE
    meta = fetch_communes_meta()
    meta["density_per_km2"] = meta["density_per_km2"].fillna(meta["density_per_km2"].median())
    code_to_density = dict(zip(meta["code_commune"], meta["density_per_km2"]))
    dock["density_per_km2"] = dock["code_commune"].astype(str).map(code_to_density)
    dock["log_density"] = np.log(dock["density_per_km2"].replace(0, np.nan).fillna(
        dock["density_per_km2"].median()
    ))
    # Normalise log-density on the station panel
    lo = dock["log_density"].quantile(0.01)
    hi = dock["log_density"].quantile(0.99)
    dock["D_norm"] = ((dock["log_density"].clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)
    log.info("  D component: log-density panel range [%.2f, %.2f] (clipped at 1-99%%)",
             lo, hi)

    # City means for the 4 components (in same order: M, I, T, D)
    means = dock.groupby("city")[["M_norm", "I_norm", "T_norm", "D_norm"]].mean()
    cmm = means.reindex(panel.cities).fillna(means.median())
    return dock, cmm.to_numpy(), panel.fub, panel.emp, code_to_density


# ---------------------------------------------------------------------------
# Run Bayesian MH on K=4
# ---------------------------------------------------------------------------

def calibrate_k4(component_matrix: np.ndarray,
                  fub: np.ndarray, emp: np.ndarray) -> dict:
    """Override b1.K=3 temporarily and run the MH sampler with 4 components."""
    backup_K = b1.K
    b1.K = 4

    # Manual MH (b1's mh_sample assumes K is set on the module)
    fub_std = b1.standardise(fub)
    emp_std = b1.standardise(np.log1p(emp))

    chain = b1.mh_sample(component_matrix, fub_std, emp_std,
                          n_burn=b1.N_BURN, n_keep=b1.N_KEEP)

    z_samples = chain["z"]
    w_samples = np.array([b1.softmax_with_floor(z) for z in z_samples])
    w_mean = w_samples.mean(axis=0)
    w_q025 = np.percentile(w_samples, 2.5, axis=0)
    w_q975 = np.percentile(w_samples, 97.5, axis=0)
    arg_dom = np.argmax(w_samples, axis=1)
    p_dom = [float((arg_dom == k).mean()) for k in range(4)]

    b1.K = backup_K
    return {
        "w_samples": w_samples,
        "w_mean": w_mean,
        "w_q025": w_q025,
        "w_q975": w_q975,
        "p_dominant": p_dom,
        "beta": chain["beta"],
    }


def calibrate_k3(component_matrix_3: np.ndarray,
                  fub: np.ndarray, emp: np.ndarray) -> dict:
    """Re-run K=3 for direct comparison."""
    b1.K = 3
    fub_std = b1.standardise(fub)
    emp_std = b1.standardise(np.log1p(emp))
    chain = b1.mh_sample(component_matrix_3, fub_std, emp_std,
                          n_burn=b1.N_BURN, n_keep=b1.N_KEEP)
    z_samples = chain["z"]
    w_samples = np.array([b1.softmax_with_floor(z) for z in z_samples])
    return {
        "w_samples": w_samples,
        "w_mean": w_samples.mean(axis=0),
        "w_q025": np.percentile(w_samples, 2.5, axis=0),
        "w_q975": np.percentile(w_samples, 97.5, axis=0),
        "beta": chain["beta"],
    }


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def correlate_with_eco(city_imd_median: dict, panel) -> tuple[float | None, int]:
    eco_path = ROOT / "data" / "external" / "mobility_sources" / "eco_compteurs_city_usage.csv"
    if not eco_path.exists():
        return None, 0
    eco = pd.read_csv(eco_path)
    obs = []
    for c, v in city_imd_median.items():
        eco_v = eco.loc[eco["city"] == c, "eco_avg_daily_bike_counts"]
        if len(eco_v) == 0:
            continue
        ev = eco_v.iloc[0]
        if pd.isna(ev):
            continue
        obs.append((float(ev), float(v)))
    if len(obs) < 5:
        return None, 0
    x = np.array([o[0] for o in obs])
    y = np.array([o[1] for o in obs])
    return float(sp_stats.spearmanr(x, y).statistic), len(obs)


def correlate_with_ref(city_imd_median: dict, panel_cities: list, ref_arr: np.ndarray) -> tuple[float | None, int]:
    pairs = []
    for c, v in city_imd_median.items():
        if c not in panel_cities:
            continue
        idx = panel_cities.index(c)
        rv = ref_arr[idx]
        if not np.isfinite(rv):
            continue
        pairs.append((float(rv), float(v)))
    if len(pairs) < 5:
        return None, 0
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return float(sp_stats.spearmanr(x, y).statistic), len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading panel + stations + INSEE commune metadata...")
    panel = load_panel()
    dock, cmm4, fub, emp, code_to_density = build_design(panel)

    log.info("\n=== IMD-3 baseline (K=3, no D) ===")
    cmm3 = cmm4[:, :3]
    res3 = calibrate_k3(cmm3, fub, emp)
    log.info("Posterior on IMD-3 weights:")
    for k, name in enumerate(["M", "I", "T"]):
        log.info("  w_%s  mean = %.3f  CrI [%.3f, %.3f]",
                 name, res3["w_mean"][k], res3["w_q025"][k], res3["w_q975"][k])

    # City IMD-3 medians (using 200 weight samples)
    rng = np.random.default_rng(2026)
    idx3 = rng.choice(len(res3["w_samples"]), 200, replace=False)
    w_sub3 = res3["w_samples"][idx3]
    cs3 = dock[["M_norm", "I_norm", "T_norm"]].to_numpy()
    sta_imd3 = (w_sub3 @ cs3.T) * 100.0  # (200, n_stations)
    city_codes, city_index = pd.factorize(dock["city"].values)
    n_cities = len(city_index)
    city_imd3 = np.zeros((n_cities, sta_imd3.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_imd3[ci] = sta_imd3[:, mask].mean(axis=1)
    median3 = dict(zip(city_index, np.median(city_imd3, axis=1)))

    log.info("\n=== IMD-4 with D (K=4) ===")
    res4 = calibrate_k4(cmm4, fub, emp)
    log.info("Posterior on IMD-4 weights:")
    for k, name in enumerate(["M", "I", "T", "D"]):
        log.info("  w_%s  mean = %.3f  CrI [%.3f, %.3f]  P(dom) = %.2f",
                 name, res4["w_mean"][k], res4["w_q025"][k], res4["w_q975"][k],
                 res4["p_dominant"][k])

    # City IMD-4 medians
    idx4 = rng.choice(len(res4["w_samples"]), 200, replace=False)
    w_sub4 = res4["w_samples"][idx4]
    cs4 = dock[["M_norm", "I_norm", "T_norm", "D_norm"]].to_numpy()
    sta_imd4 = (w_sub4 @ cs4.T) * 100.0
    city_imd4 = np.zeros((n_cities, sta_imd4.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_imd4[ci] = sta_imd4[:, mask].mean(axis=1)
    median4 = dict(zip(city_index, np.median(city_imd4, axis=1)))

    # Correlations against references
    log.info("\nCorrelation comparison (IMD-3 vs IMD-4):")
    rho_fub_3, n_fub3 = correlate_with_ref(median3, panel.cities, panel.fub)
    rho_emp_3, n_emp3 = correlate_with_ref(median3, panel.cities, panel.emp)
    rho_eco_3, n_eco3 = correlate_with_eco(median3, panel)
    rho_fub_4, n_fub4 = correlate_with_ref(median4, panel.cities, panel.fub)
    rho_emp_4, n_emp4 = correlate_with_ref(median4, panel.cities, panel.emp)
    rho_eco_4, n_eco4 = correlate_with_eco(median4, panel)
    log.info("  rho(FUB)  IMD-3 = %.3f (n=%d)   IMD-4 = %.3f (n=%d)   shift = %+.3f",
             rho_fub_3, n_fub3, rho_fub_4, n_fub4, rho_fub_4 - rho_fub_3)
    log.info("  rho(EMP)  IMD-3 = %.3f (n=%d)   IMD-4 = %.3f (n=%d)   shift = %+.3f",
             rho_emp_3, n_emp3, rho_emp_4, n_emp4, rho_emp_4 - rho_emp_3)
    log.info("  rho(eco)  IMD-3 = %.3f (n=%d)   IMD-4 = %.3f (n=%d)   shift = %+.3f",
             rho_eco_3, n_eco3, rho_eco_4, n_eco4, rho_eco_4 - rho_eco_3)

    # Top-10 comparison
    log.info("\nTop-10 IMD-3:")
    top3 = sorted(median3.items(), key=lambda x: -x[1])[:10]
    for i, (c, v) in enumerate(top3, 1):
        log.info("  %2d. %-22s  IMD-3 = %.1f", i, c, v)
    log.info("\nTop-10 IMD-4 (with D):")
    top4 = sorted(median4.items(), key=lambda x: -x[1])[:10]
    for i, (c, v) in enumerate(top4, 1):
        log.info("  %2d. %-22s  IMD-4 = %.1f", i, c, v)
    overlap = len(set(c for c, _ in top3[:10]) & set(c for c, _ in top4[:10]))
    log.info("\nTop-10 overlap: %d / 10", overlap)

    # Verdict on H1, H2, H3
    log.info("\n=== Hypothesis tests ===")
    h1 = (rho_eco_4 - rho_eco_3) >= 0.05 if (rho_eco_3 and rho_eco_4) else None
    h2 = res4["w_mean"][0] > 0.5
    h3 = overlap < 10
    log.info("  H1 (rho_eco shifts by +0.05): %s  (delta = %s)",
             "PASS" if h1 else ("FAIL" if h1 is False else "n/a"),
             f"{rho_eco_4-rho_eco_3:+.3f}" if (rho_eco_3 and rho_eco_4) else "n/a")
    log.info("  H2 (w_M > 0.5 in IMD-4):       %s  (w_M = %.3f)",
             "PASS" if h2 else "FAIL", res4["w_mean"][0])
    log.info("  H3 (Top-10 changes):           %s  (overlap = %d/10)",
             "PASS" if h3 else "FAIL", overlap)

    results = {
        "imd3": {
            "weights": {
                k: {"mean": float(res3["w_mean"][i]),
                    "q025": float(res3["w_q025"][i]),
                    "q975": float(res3["w_q975"][i])}
                for i, k in enumerate(["M", "I", "T"])
            },
            "rho_fub": rho_fub_3, "n_fub": n_fub3,
            "rho_emp": rho_emp_3, "n_emp": n_emp3,
            "rho_eco": rho_eco_3, "n_eco": n_eco3,
            "top10": [{"city": c, "imd": float(v)} for c, v in top3],
        },
        "imd4": {
            "weights": {
                k: {"mean": float(res4["w_mean"][i]),
                    "q025": float(res4["w_q025"][i]),
                    "q975": float(res4["w_q975"][i]),
                    "p_dominant": float(res4["p_dominant"][i])}
                for i, k in enumerate(["M", "I", "T", "D"])
            },
            "rho_fub": rho_fub_4, "n_fub": n_fub4,
            "rho_emp": rho_emp_4, "n_emp": n_emp4,
            "rho_eco": rho_eco_4, "n_eco": n_eco4,
            "top10": [{"city": c, "imd": float(v)} for c, v in top4],
        },
        "hypothesis_tests": {
            "H1_rho_eco_uplift_geq_0.05": bool(h1) if h1 is not None else None,
            "H2_wM_dominant": bool(h2),
            "H3_top10_changes": bool(h3),
        },
        "top10_overlap": int(overlap),
        "delta_rho_eco": float(rho_eco_4 - rho_eco_3) if (rho_eco_3 and rho_eco_4) else None,
    }
    out_json = OUT_DIR / "b7_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: weights bar comparison
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    x3 = np.arange(3)
    means3 = res3["w_mean"]
    err3 = np.vstack([means3 - res3["w_q025"], res3["w_q975"] - means3])
    ax.bar(x3, means3, color=["#1F3A6B", "#7095C8", "#5B7E4F"],
           yerr=err3, capsize=4, edgecolor="white", linewidth=0.4,
           ecolor="#404040")
    ax.set_xticks(x3); ax.set_xticklabels(["M", "I", "T"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weight posterior")
    ax.set_title("IMD-3 weights", fontsize=10)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)

    ax = axes[1]
    x4 = np.arange(4)
    means4 = res4["w_mean"]
    err4 = np.vstack([means4 - res4["w_q025"], res4["w_q975"] - means4])
    ax.bar(x4, means4, color=["#1F3A6B", "#7095C8", "#5B7E4F", "#D08020"],
           yerr=err4, capsize=4, edgecolor="white", linewidth=0.4,
           ecolor="#404040")
    ax.set_xticks(x4); ax.set_xticklabels(["M", "I", "T", "D"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weight posterior")
    ax.set_title("IMD-4 weights (with D = population density)", fontsize=10)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.suptitle("B7: posterior on the IMD-4 weights after adding density",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b7_weights_imd4.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b7_weights_imd4.pdf")

    # Figure: ranking compare
    cities_all = list(set(median3.keys()) | set(median4.keys()))
    rank3 = {c: i for i, (c, _) in enumerate(
        sorted(median3.items(), key=lambda x: -x[1]))}
    rank4 = {c: i for i, (c, _) in enumerate(
        sorted(median4.items(), key=lambda x: -x[1]))}
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    common = [c for c in cities_all if c in rank3 and c in rank4]
    for c in common:
        if rank3[c] < 15 or rank4[c] < 15:
            ax.plot([0, 1], [rank3[c], rank4[c]], "-", color="#7A7A7A",
                     linewidth=0.5, alpha=0.5)
            ax.scatter([0, 1], [rank3[c], rank4[c]], s=15,
                        color="#1F3A6B" if rank3[c] < 10 else "#7095C8",
                        edgecolor="white", linewidth=0.3)
            ax.annotate(c, (1.02, rank4[c]), fontsize=7, va="center")
    ax.invert_yaxis()
    ax.set_xticks([0, 1]); ax.set_xticklabels(["IMD-3", "IMD-4 with D"])
    ax.set_ylabel("Rank")
    ax.set_title("Ranking shift after adding population density (Top-15 union)",
                 fontsize=10)
    ax.set_xlim(-0.05, 1.4)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b7_ranking_compare.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b7_ranking_compare.pdf")


if __name__ == "__main__":
    main()
