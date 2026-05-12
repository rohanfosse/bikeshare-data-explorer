"""E16 -- Cycling well-being: which IMD component drives the FUB score?

The Cycling Barometer published by the Federation des Usagers de la
Bicyclette (FUB) is a subjective cyclist-experience instrument: a
city's score reflects how cyclists feel rather than how many of them
ride. We use it here as the empirical proxy for cycling well-being
and ask:

  Q-A. Does the IMD as a whole predict FUB well-being beyond chance?
  Q-B. Which IMD component carries the most predictive signal once
       the others are partialled out?
  Q-C. Does IMD predict FUB independently of city size and of
       observed cycling usage (eco-counter)?

We fit two parsimonious regressions on the FUB sub-panel:

  Model 1 (component-only):
      FUB_i = beta_0 + beta_M M_i + beta_I I_i
                     + beta_S S_i + beta_T T_i + eps_i

  Model 2 (controls for behavioural usage and size):
      FUB_i = beta_0 + beta_IMD IMD_i + beta_logN log(N_i)
                     + beta_eco log(eco_i + 1) + eps_i

Coefficients are reported as standardised partial-correlation
estimates with 95% bootstrap CIs from 2000 resamples. The
component-only model isolates which dimension of the IMD captures
well-being signal; the controls model establishes whether the IMD
adds information beyond what city size and observed usage already
provide.

Outputs:
    outputs/e16_results.json
    outputs/e16_components_vs_fub.pdf
    outputs/e16_controls_partial.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import COMPONENTS, ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(2026)
N_BOOT = 2000


def _standardise(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (x - mu) / sd


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS with intercept. Returns (coefs, residuals)."""
    x_full = np.column_stack([np.ones(x.shape[0]), x])
    coefs, *_ = np.linalg.lstsq(x_full, y, rcond=None)
    pred = x_full @ coefs
    return coefs, y - pred


def _partial_r2(x: np.ndarray, y: np.ndarray) -> dict:
    """Per-predictor partial R^2 in standardised OLS."""
    n, p = x.shape
    base_coefs, base_res = _ols(x, y)
    rss_full = float((base_res ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2_full = 1.0 - rss_full / max(tss, 1e-12)
    partial = []
    for k in range(p):
        keep = [j for j in range(p) if j != k]
        _, res_k = _ols(x[:, keep], y)
        rss_k = float((res_k ** 2).sum())
        partial.append((rss_k - rss_full) / max(rss_k, 1e-12))
    return {
        "r2_full": r2_full,
        "partial_r2": partial,
        "coefs_intercept": float(base_coefs[0]),
        "coefs": base_coefs[1:].tolist(),
    }


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> np.ndarray:
    """Return (n_boot, p) coefficient bootstrap matrix."""
    n, p = x.shape
    out = np.zeros((n_boot, p))
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        coefs, _ = _ols(x[idx], y[idx])
        out[b] = coefs[1:]
    return out


def _load_external() -> pd.DataFrame:
    base = ROOT / "data" / "external" / "mobility_sources"
    fub = pd.read_csv(base / "fub_barometre_2023_city_scores.csv")
    eco = pd.read_csv(base / "eco_compteurs_city_usage.csv")
    return fub[["city", "fub_score_2023"]].merge(
        eco[["city", "eco_avg_daily_bike_counts"]], on="city", how="outer",
    )


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n=%d cities", panel.n)

    ext = _load_external()
    base = pd.DataFrame({
        "city": panel.cities,
        "IMD": panel.imd,
        **{c: panel.components[:, i] for i, c in enumerate(COMPONENTS)},
    })
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    stations = load_stations()
    dock_counts = (
        stations[stations["station_type"] == "docked_bike"]
        .groupby("city").size().rename("n_stations").reset_index()
    )
    base = base.merge(dock_counts, on="city", how="left")
    base = base.merge(ext, on="city", how="left")
    base = base.rename(columns={
        "fub_score_2023": "fub",
        "eco_avg_daily_bike_counts": "eco_count",
    })
    log.info("Sub-panel with FUB: n=%d cities",
             int(np.isfinite(base["fub"]).sum()))

    # ---- Model 1: components -> FUB ---------------------------------
    mask1 = np.isfinite(base["fub"].to_numpy()) & np.all(
        np.isfinite(base[list(COMPONENTS)].to_numpy()), axis=1,
    )
    x1 = _standardise(base.loc[mask1, list(COMPONENTS)].to_numpy(dtype=float))
    y1 = base.loc[mask1, "fub"].to_numpy(dtype=float)
    y1_std = _standardise(y1)
    log.info("Model 1 (components -> FUB) on n=%d cities", int(mask1.sum()))
    m1 = _partial_r2(x1, y1_std)
    boot1 = _bootstrap_ci(x1, y1_std)
    ci1 = np.percentile(boot1, [2.5, 97.5], axis=0)
    log.info("  R^2_full = %.3f", m1["r2_full"])
    for k, name in enumerate(COMPONENTS):
        log.info(
            "  %-12s  beta = %+.3f  95%% CI [%+.3f, %+.3f]  partial R^2 = %.3f",
            name, m1["coefs"][k], ci1[0, k], ci1[1, k], m1["partial_r2"][k],
        )

    # ---- Model 2: IMD + log(N) + log(eco) -> FUB ---------------------
    mask2 = (
        np.isfinite(base["fub"].to_numpy())
        & np.isfinite(base["IMD"].to_numpy())
        & np.isfinite(base["n_stations"].to_numpy())
    )
    # eco_count is allowed to be NaN -> imputed at panel mean
    eco_imputed = base["eco_count"].fillna(base["eco_count"].mean()).to_numpy()
    log_n = np.log1p(base["n_stations"].to_numpy(dtype=float))
    log_eco = np.log1p(eco_imputed)
    x2_raw = np.column_stack([base["IMD"].to_numpy(), log_n, log_eco])
    x2 = _standardise(x2_raw[mask2])
    y2 = base.loc[mask2, "fub"].to_numpy(dtype=float)
    y2_std = _standardise(y2)
    log.info("Model 2 (IMD + log(N) + log(eco) -> FUB) on n=%d cities",
             int(mask2.sum()))
    m2 = _partial_r2(x2, y2_std)
    boot2 = _bootstrap_ci(x2, y2_std)
    ci2 = np.percentile(boot2, [2.5, 97.5], axis=0)
    pred_labels = ["IMD", "log(stations)", "log(eco-counter)"]
    log.info("  R^2_full = %.3f", m2["r2_full"])
    for k, name in enumerate(pred_labels):
        log.info(
            "  %-18s  beta = %+.3f  95%% CI [%+.3f, %+.3f]  partial R^2 = %.3f",
            name, m2["coefs"][k], ci2[0, k], ci2[1, k], m2["partial_r2"][k],
        )

    # Save JSON
    results = {
        "n_panel": int(panel.n),
        "n_fub_subpanel": int(mask1.sum()),
        "n_full_subpanel": int(mask2.sum()),
        "model1_components_to_fub": {
            "r2_full": m1["r2_full"],
            "predictors": list(COMPONENTS),
            "coefficients": {
                name: {
                    "beta_std": float(m1["coefs"][k]),
                    "ci95": [float(ci1[0, k]), float(ci1[1, k])],
                    "partial_r2": float(m1["partial_r2"][k]),
                } for k, name in enumerate(COMPONENTS)
            },
        },
        "model2_imd_plus_controls": {
            "r2_full": m2["r2_full"],
            "predictors": pred_labels,
            "coefficients": {
                name: {
                    "beta_std": float(m2["coefs"][k]),
                    "ci95": [float(ci2[0, k]), float(ci2[1, k])],
                    "partial_r2": float(m2["partial_r2"][k]),
                } for k, name in enumerate(pred_labels)
            },
        },
    }
    out_json = OUT_DIR / "e16_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # ---- Figure 1: Model 1 coefficient forest -----------------------
    labels1 = ["M\nmultim.", "I\ninfra", "S\nsafety", "T\ntopo"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(4)
    err = np.vstack([np.array(m1["coefs"]) - ci1[0], ci1[1] - np.array(m1["coefs"])])
    bar_colors = ["#5B7E4F" if c > 0 else "#A8201A" for c in m1["coefs"]]
    ax.bar(x, m1["coefs"], color=bar_colors,
           yerr=err, capsize=4, edgecolor="white", linewidth=0.4,
           ecolor="#404040", error_kw={"linewidth": 0.8})
    ax.axhline(0, color="#404040", linewidth=0.6, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels1, fontsize=9)
    ax.set_ylabel(r"Standardised $\hat\beta$ (FUB outcome)")
    ax.set_title(
        f"Which IMD component drives cycling well-being? "
        f"$R^2 = {m1['r2_full']:.2f}$",
        fontsize=10,
    )
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e16_components_vs_fub.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e16_components_vs_fub.pdf")

    # ---- Figure 2: Model 2 forest of standardised coefficients -------
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    y_pos = np.arange(len(pred_labels))
    err2 = np.vstack([
        np.array(m2["coefs"]) - ci2[0], ci2[1] - np.array(m2["coefs"]),
    ])
    colors2 = ["#1F3A6B"] + ["#7A7A7A"] * (len(pred_labels) - 1)
    ax.errorbar(m2["coefs"], y_pos, xerr=err2, fmt="o",
                color="black", ecolor="#404040",
                capsize=4, markersize=0)
    for i, (c, color) in enumerate(zip(m2["coefs"], colors2)):
        ax.plot([c], [i], "o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.axvline(0, color="#A8201A", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pred_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(r"Standardised $\hat\beta$ with 95% CI")
    ax.set_title(
        f"Does IMD add information beyond size and usage? "
        f"$R^2 = {m2['r2_full']:.2f}$",
        fontsize=10,
    )
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e16_controls_partial.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e16_controls_partial.pdf")


if __name__ == "__main__":
    main()
