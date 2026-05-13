"""E32 -- Station-level probabilistic IMD-3 and IMD-2.

Implements the indicator redesign scoped in the IMD paper:
  - IMD-3 = w_M*M + w_I*I + w_T*T  (drops Safety)
  - IMD-2 = w_M*M + w_I*I            (drops Safety and Topography)

Both variants are computed at \emph{station} granularity rather
than city granularity. For each city we then aggregate to
  - the bootstrap median IMD (point estimate),
  - the 95% percentile CI (uncertainty band),
  - the inter-station SD within the city (heterogeneity).

Weights are re-calibrated by differential evolution against the
FUB and EMP behavioural references at the city level (mean of
station IMDs per city).

We compare the IMD-3 and IMD-2 rankings against the published
IMD-4 ranking (Kendall tau and Top-10 overlap), and report
the calibration objective (mean Spearman against FUB+EMP) for
each variant. The result diagnoses whether dropping S and T
loses predictive power or simply cleans up the indicator.

Outputs:
    outputs/e32_results.json
    outputs/e32_imd_variants.pdf
    outputs/e32_per_city_ci.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import differential_evolution

from _common import COMPONENTS, ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

W_MIN = 0.05
N_BOOT = 1000


# -------------------------------------------------------------------------
# Station-level normalisation
# -------------------------------------------------------------------------

def _station_components(stations: pd.DataFrame) -> pd.DataFrame:
    """Normalise the four raw components at panel-station level.

    Returns dock-only stations with M_norm, I_norm, S_norm, T_norm in [0,1].
    Safety and topography are sign-inverted (high = good).
    """
    dock = stations[stations["station_type"] == "docked_bike"].copy()

    def _normalise(col: str, invert: bool = False) -> np.ndarray:
        s = dock[col].astype(float).fillna(dock[col].median())
        lo, hi = s.min(), s.max()
        if hi == lo:
            norm = np.full(len(s), 0.5)
        else:
            norm = (s - lo) / (hi - lo)
        return 1.0 - norm if invert else norm

    dock["M_norm"] = _normalise("gtfs_heavy_stops_300m")
    dock["I_norm"] = _normalise("infra_cyclable_pct")
    dock["S_norm"] = _normalise("baac_accidents_cyclistes", invert=True)
    dock["T_norm"] = _normalise("topography_roughness_index", invert=True)
    return dock


# -------------------------------------------------------------------------
# Weight calibration by differential evolution
# -------------------------------------------------------------------------

def _softmax_to_simplex(z: np.ndarray, k: int) -> np.ndarray:
    z_shift = z - np.max(z)
    soft = np.exp(z_shift)
    soft = soft / soft.sum()
    return W_MIN + (1.0 - k * W_MIN) * soft


def _calibrate(city_components: np.ndarray, fub: np.ndarray, emp: np.ndarray,
               k: int) -> tuple[np.ndarray, float]:
    """Calibrate weights on k-dim component matrix."""
    bounds = [(-5.0, 5.0)] * k

    def obj(z):
        w = _softmax_to_simplex(z, k)
        composite = city_components @ w
        mask_fub = np.isfinite(fub)
        mask_emp = np.isfinite(emp)
        rho_fub = (
            sp_stats.spearmanr(composite[mask_fub], fub[mask_fub]).statistic
            if mask_fub.sum() >= 5 else 0.0
        )
        rho_emp = (
            sp_stats.spearmanr(composite[mask_emp], emp[mask_emp]).statistic
            if mask_emp.sum() >= 5 else 0.0
        )
        return -0.5 * (rho_fub + rho_emp)

    result = differential_evolution(
        obj, bounds=bounds, seed=42, maxiter=200, popsize=15,
        tol=1e-7, polish=True, strategy="best1bin",
    )
    w_star = _softmax_to_simplex(result.x, k)
    return w_star, -result.fun


# -------------------------------------------------------------------------
# Probabilistic aggregation via station bootstrap
# -------------------------------------------------------------------------

def _city_imd_from_stations(dock: pd.DataFrame, w: np.ndarray,
                             component_cols: list[str]) -> pd.Series:
    """City-mean of station-level IMD."""
    station_imd = dock[component_cols].to_numpy() @ w * 100.0
    dock = dock.assign(_imd=station_imd)
    return dock.groupby("city")["_imd"].mean()


def _bootstrap_city_distribution(
    dock: pd.DataFrame, w: np.ndarray, component_cols: list[str],
    n_boot: int = N_BOOT, rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return per-city distribution of mean station IMD via bootstrap."""
    rng = rng or np.random.default_rng(2026)
    station_imd = dock[component_cols].to_numpy() @ w * 100.0
    dock = dock.assign(_imd=station_imd)
    boot_means = {city: np.zeros(n_boot) for city in dock["city"].unique()}
    for b in range(n_boot):
        sampled = (
            dock.groupby("city", group_keys=False)
                .apply(lambda g: g.sample(n=len(g), replace=True,
                                          random_state=rng))
        )
        means = sampled.groupby("city")["_imd"].mean()
        for c, v in means.items():
            boot_means[c][b] = v
    return pd.DataFrame({
        c: vals for c, vals in boot_means.items()
    })


def _summarise(boot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in boot_df.columns:
        vals = boot_df[city].dropna().to_numpy()
        rows.append({
            "city": city,
            "imd_median": float(np.median(vals)),
            "imd_mean": float(np.mean(vals)),
            "imd_q025": float(np.percentile(vals, 2.5)),
            "imd_q975": float(np.percentile(vals, 97.5)),
            "imd_sd": float(np.std(vals)),
        })
    return pd.DataFrame(rows).sort_values("imd_median", ascending=False)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    log.info("Loading panel and stations...")
    panel = load_panel()
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    stations = load_stations()
    dock = _station_components(stations)
    log.info("  %d dock-based stations across %d cities",
             len(dock), dock["city"].nunique())

    # IMD-4 reference (use the published city-level component matrix)
    log.info("\n=== IMD-4 (reference, current paper) ===")
    w4, rho4 = _calibrate(panel.components, panel.fub, panel.emp, k=4)
    log.info("  w = %s  rho = %.3f", dict(zip(COMPONENTS, np.round(w4, 3))), rho4)

    # For IMD-3 / IMD-2 we want the *city-mean of station-level normalised
    # components* (not the published panel.components which use a different
    # normalisation). Build them now.
    city_means = dock.groupby("city")[["M_norm", "I_norm", "S_norm", "T_norm"]].mean()
    # Align to panel order
    cmm = city_means.reindex(panel.cities).fillna(city_means.median())
    cmM = cmm["M_norm"].to_numpy()
    cmI = cmm["I_norm"].to_numpy()
    cmS = cmm["S_norm"].to_numpy()
    cmT = cmm["T_norm"].to_numpy()

    # IMD-3 = M + I + T
    log.info("\n=== IMD-3 (M+I+T, station-level normalisation) ===")
    comp3 = np.column_stack([cmM, cmI, cmT])
    w3, rho3 = _calibrate(comp3, panel.fub, panel.emp, k=3)
    log.info("  w = (M=%.3f, I=%.3f, T=%.3f)  rho = %.3f",
             w3[0], w3[1], w3[2], rho3)

    # IMD-2 = M + I
    log.info("\n=== IMD-2 (M+I, station-level normalisation) ===")
    comp2 = np.column_stack([cmM, cmI])
    w2, rho2 = _calibrate(comp2, panel.fub, panel.emp, k=2)
    log.info("  w = (M=%.3f, I=%.3f)  rho = %.3f",
             w2[0], w2[1], rho2)

    # Station-level IMD-3 and IMD-2
    log.info("\nComputing per-station IMD-3 and IMD-2...")
    dock["imd3_station"] = (dock[["M_norm", "I_norm", "T_norm"]].to_numpy()
                            @ w3 * 100.0)
    dock["imd2_station"] = (dock[["M_norm", "I_norm"]].to_numpy()
                            @ w2 * 100.0)

    # Probabilistic city-level distributions via station bootstrap
    log.info("Bootstrapping per-city IMD-3 distributions (N=%d)...", N_BOOT)
    boot3 = _bootstrap_city_distribution(
        dock, w3, ["M_norm", "I_norm", "T_norm"], n_boot=N_BOOT,
    )
    summary3 = _summarise(boot3)
    log.info("Bootstrapping per-city IMD-2 distributions (N=%d)...", N_BOOT)
    boot2 = _bootstrap_city_distribution(
        dock, w2, ["M_norm", "I_norm"], n_boot=N_BOOT,
    )
    summary2 = _summarise(boot2)

    # Compare rankings
    def _rank_compare(panel_imd_ref, summary_df, label):
        cities = list(summary_df["city"])
        ref = dict(zip(panel.cities, panel_imd_ref))
        new_imd = dict(zip(summary_df["city"], summary_df["imd_median"]))
        shared = [c for c in cities if c in ref]
        ref_vals = np.array([ref[c] for c in shared])
        new_vals = np.array([new_imd[c] for c in shared])
        tau, _ = sp_stats.kendalltau(ref_vals, new_vals)
        rho, _ = sp_stats.spearmanr(ref_vals, new_vals)
        top10_ref = sorted(shared, key=lambda c: -ref[c])[:10]
        top10_new = sorted(shared, key=lambda c: -new_imd[c])[:10]
        overlap = len(set(top10_ref) & set(top10_new))
        log.info("%s vs IMD-4: tau=%.3f  rho=%.3f  Top-10 overlap=%d/10",
                 label, tau, rho, overlap)
        log.info("  IMD-4 Top-10:  %s", top10_ref)
        log.info("  %s Top-10:     %s", label, top10_new)
        return {"tau": float(tau), "rho": float(rho),
                "top10_overlap": int(overlap),
                "top10_ref": top10_ref, "top10_new": top10_new}

    imd_ref = panel.components @ w4 * 100.0
    cmp3 = _rank_compare(imd_ref, summary3, "IMD-3")
    cmp2 = _rank_compare(imd_ref, summary2, "IMD-2")

    # Predictive performance against eco-counter daily flows
    log.info("\nPredictive comparison against eco-counter flows...")
    eco_path = ROOT / "data" / "external" / "mobility_sources" / "eco_compteurs_city_usage.csv"
    if eco_path.exists():
        eco = pd.read_csv(eco_path)
        # Spearman correlation of each IMD variant with eco-counter
        eco_lookup = dict(zip(eco["city"], eco["eco_avg_daily_bike_counts"]))
        def _eco_rho(summary):
            cities = summary["city"].tolist()
            obs = []
            for c in cities:
                eco_v = eco_lookup.get(c)
                if eco_v is not None and np.isfinite(eco_v):
                    obs.append((float(eco_v),
                                 float(summary.loc[summary["city"]==c, "imd_median"].iloc[0])))
            if len(obs) < 5:
                return None
            x = np.array([o[0] for o in obs])
            y = np.array([o[1] for o in obs])
            return float(sp_stats.spearmanr(x, y).statistic), len(obs)
        ref_rho_obs = []
        for c in panel.cities:
            eco_v = eco_lookup.get(c)
            if eco_v is not None and np.isfinite(eco_v):
                ref_rho_obs.append((float(eco_v),
                                     float(imd_ref[panel.cities.index(c)])))
        rho_eco_4 = sp_stats.spearmanr(
            [o[0] for o in ref_rho_obs], [o[1] for o in ref_rho_obs],
        ).statistic if len(ref_rho_obs) >= 5 else None
        rho_eco_3 = _eco_rho(summary3)
        rho_eco_2 = _eco_rho(summary2)
        log.info("  rho(IMD-4, eco) = %s (n=%d)",
                 f"{rho_eco_4:.3f}" if rho_eco_4 is not None else "na",
                 len(ref_rho_obs))
        if rho_eco_3 is not None:
            log.info("  rho(IMD-3, eco) = %.3f (n=%d)", rho_eco_3[0], rho_eco_3[1])
        if rho_eco_2 is not None:
            log.info("  rho(IMD-2, eco) = %.3f (n=%d)", rho_eco_2[0], rho_eco_2[1])
    else:
        rho_eco_4 = rho_eco_3 = rho_eco_2 = None

    # Save outputs
    results = {
        "imd4": {
            "k": 4,
            "weights": dict(zip(COMPONENTS, [float(x) for x in w4])),
            "in_sample_rho": float(rho4),
            "rho_vs_eco_counter": float(rho_eco_4) if rho_eco_4 else None,
        },
        "imd3": {
            "k": 3,
            "weights": {"M": float(w3[0]), "I": float(w3[1]), "T": float(w3[2])},
            "in_sample_rho": float(rho3),
            "rho_vs_eco_counter": float(rho_eco_3[0]) if rho_eco_3 else None,
            "vs_imd4_kendall_tau": cmp3["tau"],
            "vs_imd4_spearman_rho": cmp3["rho"],
            "vs_imd4_top10_overlap": cmp3["top10_overlap"],
            "top10": cmp3["top10_new"],
        },
        "imd2": {
            "k": 2,
            "weights": {"M": float(w2[0]), "I": float(w2[1])},
            "in_sample_rho": float(rho2),
            "rho_vs_eco_counter": float(rho_eco_2[0]) if rho_eco_2 else None,
            "vs_imd4_kendall_tau": cmp2["tau"],
            "vs_imd4_spearman_rho": cmp2["rho"],
            "vs_imd4_top10_overlap": cmp2["top10_overlap"],
            "top10": cmp2["top10_new"],
        },
        "imd3_per_city": summary3.to_dict("records"),
        "imd2_per_city": summary2.to_dict("records"),
    }
    out_json = OUT_DIR / "e32_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure 1: weights comparison
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    ax = axes[0]
    variants = ["IMD-4", "IMD-3", "IMD-2"]
    M_w = [w4[0], w3[0], w2[0]]
    I_w = [w4[1], w3[1], w2[1]]
    T_w = [w4[3], w3[2], 0.0]
    S_w = [w4[2], 0.0, 0.0]
    x = np.arange(3)
    width = 0.20
    ax.bar(x - 1.5*width, M_w, width, color="#1F3A6B", label="M multimodality")
    ax.bar(x - 0.5*width, I_w, width, color="#7095C8", label="I infrastructure")
    ax.bar(x + 0.5*width, T_w, width, color="#5B7E4F", label="T topography")
    ax.bar(x + 1.5*width, S_w, width, color="#A8201A", label="S safety")
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel("Calibrated weight")
    ax.set_title("Weight redistribution as components drop", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)

    ax = axes[1]
    rhos = [rho4, rho3, rho2]
    bars = ax.bar(variants, rhos, color=["#1F3A6B", "#5B7E4F", "#B07A30"],
                  edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.3f}", ha="center", fontsize=9,
                color="#202020", fontweight="bold")
    ax.set_ylabel(r"In-sample mean Spearman $\bar\rho$(FUB+EMP)")
    ax.set_ylim(0, max(rhos) * 1.15)
    ax.set_title("Calibration fit as components drop", fontsize=10)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.suptitle("E32: IMD-2/3/4 calibration on the FR panel", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e32_imd_variants.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e32_imd_variants.pdf")

    # Figure 2: probabilistic per-city CI for IMD-3, Top-20
    top20 = summary3.head(20)
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    y_pos = np.arange(len(top20))
    err = np.vstack([
        top20["imd_median"] - top20["imd_q025"],
        top20["imd_q975"] - top20["imd_median"],
    ])
    ax.errorbar(top20["imd_median"], y_pos, xerr=err,
                fmt="o", color="#1F3A6B", ecolor="#404040",
                capsize=3, markersize=5, elinewidth=0.9, capthick=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["city"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("IMD-3 (station bootstrap median, 95% CI)")
    ax.set_title(f"Probabilistic IMD-3 Top-20 (station-level, $N = {N_BOOT}$ bootstrap)",
                 fontsize=10)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e32_per_city_ci.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e32_per_city_ci.pdf")


if __name__ == "__main__":
    main()
