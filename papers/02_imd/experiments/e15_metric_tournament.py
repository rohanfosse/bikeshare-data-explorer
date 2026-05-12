"""E15 -- Metric tournament: is the IMD a better indicator?

Compares the IMD against four alternative indicators of bike-sharing
performance, each available at the city level on the dock-based
panel:
  1. IMD              -- our four-dimensional composite (paper)
  2. Volumetric       -- log(stations * mean capacity per station)
                          per 10k inhabitants
  3. FUB Cycling Barometer 2023 -- subjective cyclist perception
  4. Eco-counter daily counts   -- observed behavioural usage
  5. EMP 2019 modal share       -- aggregate modal share at work

The tournament reports three views:

  A. Inter-metric Spearman correlation matrix (consistency).
  B. Predictive performance of each metric for two outcomes:
        - FUB score      (cycling well-being)
        - eco-counter    (observed daily usage)
     measured by leave-one-city-out Spearman correlation.
  C. Discriminative power: normalised inter-quartile range,
     dynamic range, and the share of cities at the extremes.

Note: each comparison uses pairwise-complete cases because no single
metric covers the entire panel.

Outputs:
    outputs/e15_results.json
    outputs/e15_metric_matrix.pdf
    outputs/e15_predictive_power.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

METRIC_LABELS = {
    "IMD": "IMD",
    "volumetric": "Volumetric",
    "fub": "FUB perception",
    "eco_count": "Eco-counter",
    "emp": "EMP modal share",
}
METRIC_ORDER = ["IMD", "volumetric", "fub", "eco_count", "emp"]


def _volumetric_metric(stations_df: pd.DataFrame) -> pd.DataFrame:
    """Build a volumetric capacity metric per city."""
    dock = stations_df[stations_df["station_type"] == "docked_bike"].copy()
    if "capacity" not in dock.columns:
        dock["capacity"] = np.nan
    g = dock.groupby("city").agg(
        n_stations=("station_type", "size"),
        mean_capacity=("capacity", "mean"),
    ).reset_index()
    g["mean_capacity"] = g["mean_capacity"].fillna(15.0)
    g["volumetric_raw"] = g["n_stations"] * g["mean_capacity"]
    g["volumetric"] = np.log1p(g["volumetric_raw"])
    return g[["city", "n_stations", "volumetric"]]


def _load_external() -> pd.DataFrame:
    base = ROOT / "data" / "external" / "mobility_sources"
    fub = pd.read_csv(base / "fub_barometre_2023_city_scores.csv")
    eco = pd.read_csv(base / "eco_compteurs_city_usage.csv")
    emp = pd.read_csv(base / "emp_2019_city_modal_share.csv")
    df = fub[["city", "fub_score_2023"]].merge(
        eco[["city", "eco_avg_daily_bike_counts"]], on="city", how="outer",
    ).merge(emp, on="city", how="outer")
    df = df.rename(columns={
        "fub_score_2023": "fub",
        "eco_avg_daily_bike_counts": "eco_count",
        "emp_part_velo_2019": "emp",
    })
    return df


def _loo_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out Spearman: average rank-correlation over folds.

    Each fold removes one observation and measures the residual
    Spearman correlation; the average is reported. With one point
    removed the correlation is computed on n-1 observations.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    xs, ys = x[mask], y[mask]
    n = len(xs)
    rhos = []
    for i in range(n):
        idx = np.ones(n, dtype=bool)
        idx[i] = False
        rhos.append(sp_stats.spearmanr(xs[idx], ys[idx]).statistic)
    return float(np.mean(rhos))


def _discriminative(values: np.ndarray) -> dict:
    vals = values[np.isfinite(values)]
    if len(vals) < 5:
        return {"n": int(len(vals))}
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    rng = float(vals.max() - vals.min())
    sd = float(vals.std(ddof=1))
    return {
        "n": int(len(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(np.median(vals)),
        "iqr": iqr,
        "range": rng,
        "sd": sd,
        # Normalised dynamic range: how many SDs the panel spans
        "spans_in_sd": rng / max(sd, 1e-9),
    }


def main() -> None:
    log.info("Loading panel and stations...")
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    panel = load_panel()
    stations = load_stations()

    vol = _volumetric_metric(stations)
    ext = _load_external()

    df = (
        pd.DataFrame({"city": panel.cities, "IMD": panel.imd})
        .merge(vol, on="city", how="left")
        .merge(ext, on="city", how="left")
    )
    log.info("Panel of %d cities; metric coverage:", len(df))
    for col in METRIC_ORDER:
        n_obs = int(np.isfinite(df[col].to_numpy()).sum())
        log.info("  %-14s  n = %d", col, n_obs)

    # ---- A. Inter-metric Spearman matrix -----------------------------
    metrics = df[METRIC_ORDER].to_numpy(dtype=float)
    n_m = len(METRIC_ORDER)
    rho_mat = np.full((n_m, n_m), np.nan)
    n_mat = np.zeros((n_m, n_m), dtype=int)
    for i in range(n_m):
        for j in range(n_m):
            mask = np.isfinite(metrics[:, i]) & np.isfinite(metrics[:, j])
            if mask.sum() >= 5:
                rho_mat[i, j] = sp_stats.spearmanr(
                    metrics[mask, i], metrics[mask, j]
                ).statistic
                n_mat[i, j] = int(mask.sum())

    log.info("Inter-metric Spearman correlations:")
    header = "      " + "  ".join(f"{m:>10}" for m in METRIC_ORDER)
    log.info(header)
    for i, mi in enumerate(METRIC_ORDER):
        row = "  ".join(
            ("       --" if not np.isfinite(rho_mat[i, j])
             else f"{rho_mat[i, j]:>+10.2f}")
            for j in range(n_m)
        )
        log.info(f"{mi:>5}: {row}")

    # ---- B. Predictive performance for two outcomes ------------------
    outcomes = ["fub", "eco_count"]
    pred = {}
    for predictor in METRIC_ORDER:
        for outcome in outcomes:
            if predictor == outcome:
                continue
            x = df[predictor].to_numpy(dtype=float)
            y = df[outcome].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < 5:
                rho, rho_loo = float("nan"), float("nan")
            else:
                rho = float(sp_stats.spearmanr(x[mask], y[mask]).statistic)
                rho_loo = _loo_spearman(x, y)
            pred[(predictor, outcome)] = {
                "n": n, "rho": rho, "rho_loo": rho_loo,
            }
            log.info("  pred(%s -> %s)  n=%d  rho=%+.3f  rho_loo=%+.3f",
                     predictor, outcome, n, rho if np.isfinite(rho) else 0,
                     rho_loo if np.isfinite(rho_loo) else 0)

    # ---- C. Discriminative power -------------------------------------
    discrim = {m: _discriminative(df[m].to_numpy(dtype=float))
               for m in METRIC_ORDER}

    # ---- Save JSON ---------------------------------------------------
    results = {
        "n_cities_total": int(len(df)),
        "metric_coverage": {
            m: int(np.isfinite(df[m].to_numpy()).sum()) for m in METRIC_ORDER
        },
        "spearman_matrix": {
            mi: {mj: float(rho_mat[i, j]) for j, mj in enumerate(METRIC_ORDER)}
            for i, mi in enumerate(METRIC_ORDER)
        },
        "n_pairwise": {
            mi: {mj: int(n_mat[i, j]) for j, mj in enumerate(METRIC_ORDER)}
            for i, mi in enumerate(METRIC_ORDER)
        },
        "predictive_power": {
            f"{pred_k[0]}__{pred_k[1]}": v for pred_k, v in pred.items()
        },
        "discriminative": discrim,
    }
    out_json = OUT_DIR / "e15_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # ---- Figure 1: heatmap of inter-metric Spearman -------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    masked = np.ma.masked_invalid(rho_mat)
    im = ax.imshow(masked, vmin=-1.0, vmax=1.0, cmap="RdBu_r",
                   aspect="auto")
    ax.set_xticks(range(n_m))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRIC_ORDER],
                       fontsize=9)
    for i in range(n_m):
        for j in range(n_m):
            val = rho_mat[i, j]
            if np.isfinite(val):
                ax.text(j, i,
                        f"{val:+.2f}\nn={n_mat[i, j]}",
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(val) > 0.45 else "#202020")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman $\\rho$")
    ax.set_title("Inter-metric Spearman correlation", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e15_metric_matrix.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e15_metric_matrix.pdf")

    # ---- Figure 2: predictive performance bar chart -------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, outcome, title in zip(
        axes,
        outcomes,
        ["Predicting cycling well-being (FUB)",
         "Predicting observed usage (eco-counter)"],
    ):
        names = [m for m in METRIC_ORDER if m != outcome]
        vals = [pred[(m, outcome)]["rho_loo"] for m in names]
        n_used = [pred[(m, outcome)]["n"] for m in names]
        colors = ["#1F3A6B" if m == "IMD" else "#7A7A7A" for m in names]
        labels = [METRIC_LABELS[m] for m in names]
        order = np.argsort(vals)
        ax.barh(
            [labels[i] for i in order],
            [vals[i] for i in order],
            color=[colors[i] for i in order],
            edgecolor="white", linewidth=0.4,
        )
        for i, idx in enumerate(order):
            v = vals[idx]
            n_label = n_used[idx]
            ax.text(
                v + 0.01 if v >= 0 else v - 0.01, i,
                f"n={n_label}",
                ha="left" if v >= 0 else "right",
                va="center", fontsize=7, color="#404040",
            )
        ax.axvline(0, color="#404040", linewidth=0.5, alpha=0.7)
        ax.set_xlim(-0.2, 1.0)
        ax.set_xlabel(r"LOO Spearman $\rho$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.suptitle("Metric tournament: predictive power", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e15_predictive_power.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e15_predictive_power.pdf")


if __name__ == "__main__":
    main()
