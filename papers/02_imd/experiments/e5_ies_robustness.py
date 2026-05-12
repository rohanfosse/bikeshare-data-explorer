"""E5 -- IES specification robustness.

Tests how stable the social-mobility-deserts diagnostic is under:
    (a) four predictor blocks (income only / full INSEE / +topography /
        +density);
    (b) four estimators (Ridge / Lasso / ElasticNet / Random Forest);
    (c) inclusion vs exclusion of the part_velo_travail predictor,
        which is the most direct tautology risk.

Outputs:
    outputs/e5_results.json
    outputs/e5_kendall_matrix.pdf
    outputs/e5_desert_invariance.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from _common import COMPONENTS, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DESERT_THRESHOLD = 0.85


def _fit_predict(
    estimator_name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Fit estimator with sensible defaults, return predictions + meta."""
    if estimator_name == "Ridge":
        model = RidgeCV(alphas=np.logspace(-3, 2, 100), cv=min(len(y), 10))
        model.fit(x, y)
        return model.predict(x), {"alpha": float(model.alpha_)}
    if estimator_name == "Lasso":
        model = LassoCV(alphas=np.logspace(-3, 2, 100),
                        cv=min(len(y), 10), max_iter=5000, random_state=seed)
        model.fit(x, y)
        return model.predict(x), {"alpha": float(model.alpha_)}
    if estimator_name == "ElasticNet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-3, 2, 50),
            cv=min(len(y), 10), max_iter=5000, random_state=seed,
        )
        model.fit(x, y)
        return model.predict(x), {"alpha": float(model.alpha_),
                                   "l1_ratio": float(model.l1_ratio_)}
    if estimator_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=3,
            random_state=seed, n_jobs=-1,
        )
        model.fit(x, y)
        return model.predict(x), {"feature_importance":
                                   model.feature_importances_.tolist()}
    raise ValueError(estimator_name)


def _build_blocks(socio: pd.DataFrame) -> dict[str, list[str]]:
    """Return the predictor-column blocks."""
    return {
        "income_only":                 ["revenu_median_uc"],
        "income_no_cycling":           ["revenu_median_uc", "gini_revenu",
                                        "part_menages_voit0"],
        "full_INSEE":                  ["revenu_median_uc", "gini_revenu",
                                        "part_menages_voit0",
                                        "part_velo_travail"],
        "INSEE_no_velo_travail":       ["revenu_median_uc", "gini_revenu",
                                        "part_menages_voit0"],
    }


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    socio = panel.socio.copy()
    cities = panel.cities
    imd = panel.imd

    # Drop cities with missing income (anchor variable)
    keep_mask = socio["revenu_median_uc"].notna().to_numpy()
    socio = socio.loc[keep_mask].reset_index(drop=True)
    cities = [cities[i] for i, k in enumerate(keep_mask) if k]
    imd = imd[keep_mask]
    n = len(cities)
    log.info("  n=%d cities with complete income data", n)

    blocks = _build_blocks(socio)
    estimators = ["Ridge", "Lasso", "ElasticNet", "RandomForest"]

    all_results: dict = {"specifications": [], "deserts_per_spec": {}}
    ies_matrix = pd.DataFrame(index=cities)

    for block_name, block_cols in blocks.items():
        x_raw = socio[block_cols].fillna(socio[block_cols].median()).to_numpy()
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x_raw)
        for est in estimators:
            try:
                y_hat, meta = _fit_predict(est, x_std, imd)
            except Exception as exc:
                log.warning("%s/%s skipped: %s", block_name, est, exc)
                continue
            ies = imd / np.maximum(y_hat, 1.0)
            tag = f"{block_name}__{est}"
            ies_matrix[tag] = ies
            r2 = float(sp_stats.pearsonr(y_hat, imd).statistic ** 2)
            deserts = [c for c, v in zip(cities, ies) if v < DESERT_THRESHOLD]
            all_results["specifications"].append({
                "block": block_name,
                "estimator": est,
                "predictors": block_cols,
                "n_predictors": len(block_cols),
                "r2_in_sample": r2,
                "n_deserts": len(deserts),
                "deserts": deserts,
                "meta": meta,
            })
            all_results["deserts_per_spec"][tag] = deserts

    # Kendall tau matrix of IES rankings across specifications
    tau_matrix = ies_matrix.corr(method="kendall")
    tau_matrix.to_csv(OUT_DIR / "e5_kendall_matrix.csv")

    # Desert invariance: how often does each city appear as a desert?
    desert_counts = pd.Series(0, index=cities, dtype=int)
    for d_list in all_results["deserts_per_spec"].values():
        for city in d_list:
            desert_counts[city] += 1
    total_specs = len(all_results["deserts_per_spec"])
    desert_invariance = desert_counts.sort_values(ascending=False) / total_specs
    desert_invariance = desert_invariance[desert_invariance > 0]

    all_results["total_specifications"] = total_specs
    all_results["desert_invariance"] = desert_invariance.to_dict()
    all_results["robust_deserts"] = desert_invariance[
        desert_invariance >= 0.75
    ].index.tolist()

    # Reference specification (published in the paper)
    log.info("Reference spec (full_INSEE with part_velo_travail / Ridge):")
    for spec in all_results["specifications"]:
        if spec["block"] == "full_INSEE" and spec["estimator"] == "Ridge":
            log.info("  R^2 in-sample = %.3f, n deserts = %d",
                     spec["r2_in_sample"], spec["n_deserts"])
            log.info("  deserts: %s", spec["deserts"])
    log.info("Spec with bike-commute REMOVED (INSEE_no_velo_travail / Ridge):")
    for spec in all_results["specifications"]:
        if spec["block"] == "INSEE_no_velo_travail" and spec["estimator"] == "Ridge":
            log.info("  R^2 in-sample = %.3f, n deserts = %d",
                     spec["r2_in_sample"], spec["n_deserts"])
            log.info("  deserts: %s", spec["deserts"])

    # Robust deserts (>= 75 % of specs)
    log.info("Robust deserts (flagged in >= 75%% of %d specs): %s",
             total_specs, all_results["robust_deserts"])
    log.info("Median Kendall tau across IES rankings: %.3f",
             tau_matrix.values[np.triu_indices_from(tau_matrix.values, k=1)].mean())

    # Plot: desert invariance heatmap
    fig, ax = plt.subplots(figsize=(5.6, max(4.0, 0.18 * len(desert_invariance))))
    pct = desert_invariance * 100
    bars = ax.barh(
        pct.index[::-1], pct.values[::-1],
        color=["#A8201A" if v >= 75 else "#1F3A6B" if v >= 50 else "#7A7A7A"
               for v in pct.values[::-1]],
        edgecolor="white", linewidth=0.4,
    )
    ax.axvline(75, color="#A8201A", linestyle=":", linewidth=0.7,
               alpha=0.7, label="75% invariance")
    ax.set_xlabel("Fraction of IES specifications flagging the city as a desert (%)")
    ax.set_xlim(0, 105)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e5_desert_invariance.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e5_desert_invariance.pdf")

    out_json = OUT_DIR / "e5_results.json"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
