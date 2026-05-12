"""E1 -- Leave-one-city-out and spatial-block cross-validation.

Tests whether the supervised weights generalise out of sample. For each
city in the FUB-or-EMP panel, re-runs differential evolution on the
remaining cities and predicts the held-out city's IMD with the
re-trained weights. Also runs a NUTS-2 spatial-block CV.

Outputs:
    outputs/e1_results.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import (
    CITY_TO_NUTS2,
    COMPONENTS,
    Panel,
    calibrate_weights,
    composite_score,
    load_panel,
)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _loo_calibration(panel: Panel) -> dict:
    """Hold out one city at a time, re-calibrate, predict held-out city."""
    cities = panel.cities
    n = panel.n
    weights_per_fold: list[np.ndarray] = []
    pred_imd = np.full(n, np.nan)
    pred_fub = np.full(n, np.nan)
    pred_emp = np.full(n, np.nan)

    for i in range(n):
        if not (np.isfinite(panel.fub[i]) or np.isfinite(panel.emp[i])):
            continue
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        try:
            w, _ = calibrate_weights(
                panel.components[keep],
                panel.fub[keep],
                panel.emp[keep],
                seed=42 + i,
                maxiter=120,
            )
        except Exception as exc:
            log.warning("LOO fold %s failed: %s", cities[i], exc)
            continue
        weights_per_fold.append(w)
        pred_imd[i] = float(panel.components[i] @ w * 100)
        if np.isfinite(panel.fub[i]):
            pred_fub[i] = pred_imd[i]
        if np.isfinite(panel.emp[i]):
            pred_emp[i] = pred_imd[i]

    weights_arr = np.array(weights_per_fold)
    inter_fold_sd = weights_arr.std(axis=0).tolist()
    inter_fold_mean = weights_arr.mean(axis=0).tolist()

    mask_fub = np.isfinite(pred_fub) & np.isfinite(panel.fub)
    mask_emp = np.isfinite(pred_emp) & np.isfinite(panel.emp)
    rho_fub_loo = (
        sp_stats.spearmanr(pred_fub[mask_fub], panel.fub[mask_fub]).statistic
        if mask_fub.sum() >= 5 else float("nan")
    )
    rho_emp_loo = (
        sp_stats.spearmanr(pred_emp[mask_emp], panel.emp[mask_emp]).statistic
        if mask_emp.sum() >= 5 else float("nan")
    )

    return {
        "n_folds": int(weights_arr.shape[0]),
        "rho_loo_fub": float(rho_fub_loo),
        "rho_loo_emp": float(rho_emp_loo),
        "n_loo_fub": int(mask_fub.sum()),
        "n_loo_emp": int(mask_emp.sum()),
        "weights_mean": dict(zip(COMPONENTS, [float(x) for x in inter_fold_mean])),
        "weights_sd": dict(zip(COMPONENTS, [float(x) for x in inter_fold_sd])),
    }


def _spatial_block_cv(panel: Panel, min_holdout: int = 3) -> dict:
    """5-fold spatial-block CV using NUTS-2 regions."""
    regions = np.array([CITY_TO_NUTS2.get(c, "OTHER") for c in panel.cities])
    pred = np.full(panel.n, np.nan)
    fold_weights: list[tuple[str, np.ndarray]] = []

    for region in sorted(set(regions)):
        mask_test = regions == region
        if mask_test.sum() < min_holdout:
            continue
        mask_train = ~mask_test
        try:
            w, _ = calibrate_weights(
                panel.components[mask_train],
                panel.fub[mask_train],
                panel.emp[mask_train],
                seed=42,
                maxiter=150,
            )
        except Exception as exc:
            log.warning("Region %s skipped: %s", region, exc)
            continue
        fold_weights.append((region, w))
        pred[mask_test] = composite_score(w, panel.components[mask_test])

    if not fold_weights:
        return {"n_folds": 0}

    weights_arr = np.array([w for _, w in fold_weights])
    inter_fold_sd = weights_arr.std(axis=0).tolist()

    mask_fub = np.isfinite(pred) & np.isfinite(panel.fub)
    mask_emp = np.isfinite(pred) & np.isfinite(panel.emp)
    rho_fub = (
        sp_stats.spearmanr(pred[mask_fub], panel.fub[mask_fub]).statistic
        if mask_fub.sum() >= 5 else float("nan")
    )
    rho_emp = (
        sp_stats.spearmanr(pred[mask_emp], panel.emp[mask_emp]).statistic
        if mask_emp.sum() >= 5 else float("nan")
    )

    return {
        "n_folds": len(fold_weights),
        "regions": [r for r, _ in fold_weights],
        "weights_sd": dict(zip(COMPONENTS, [float(x) for x in inter_fold_sd])),
        "rho_block_fub": float(rho_fub),
        "rho_block_emp": float(rho_emp),
        "n_block_fub": int(mask_fub.sum()),
        "n_block_emp": int(mask_emp.sum()),
    }


def _plot_loo_weights(loo: dict, fig_path: Path) -> None:
    """Side-by-side bar chart of mean LOO weight vs published weight."""
    published = {"M_multi": 0.578, "I_infra": 0.184,
                 "S_securite": 0.142, "T_topo": 0.096}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(4)
    width = 0.38
    means = [loo["weights_mean"][c] for c in COMPONENTS]
    errs = [loo["weights_sd"][c] for c in COMPONENTS]
    pub = [published[c] for c in COMPONENTS]
    ax.bar(x - width / 2, pub, width, color="#7A7A7A", label="Published")
    ax.bar(
        x + width / 2, means, width, color="#1F3A6B",
        yerr=errs, capsize=3, label=f"LOO mean (n={loo['n_folds']})",
        ecolor="#404040", error_kw={"linewidth": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["M", "I", "S", "T"])
    ax.set_ylabel("Calibrated weight")
    ax.set_ylim(0, max(max(pub), max(means)) * 1.4)
    ax.axhline(0.05, color="#A8201A", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.text(3.4, 0.06, r"$w_{\min}$", color="#A8201A", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", fig_path.name)


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n=%d cities, FUB=%d, EMP=%d", panel.n,
             int(np.isfinite(panel.fub).sum()),
             int(np.isfinite(panel.emp).sum()))

    log.info("Running E1 LOO calibration...")
    loo = _loo_calibration(panel)
    log.info("  LOO rho_FUB=%.3f (n=%d), rho_EMP=%.3f (n=%d)",
             loo["rho_loo_fub"], loo["n_loo_fub"],
             loo["rho_loo_emp"], loo["n_loo_emp"])
    log.info("  weights_mean: %s", loo["weights_mean"])
    log.info("  weights_sd:   %s", loo["weights_sd"])

    log.info("Running E1 spatial-block CV...")
    block = _spatial_block_cv(panel)
    log.info("  block CV n_folds=%d, rho_FUB=%.3f, rho_EMP=%.3f",
             block.get("n_folds", 0),
             block.get("rho_block_fub", float("nan")),
             block.get("rho_block_emp", float("nan")))

    results = {"loo": loo, "spatial_block": block}
    out_json = OUT_DIR / "e1_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    _plot_loo_weights(loo, OUT_DIR / "e1_loo_weights.pdf")


if __name__ == "__main__":
    main()
