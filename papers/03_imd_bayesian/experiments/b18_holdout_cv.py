"""B18 -- Hold-out cross-validation of the IMD-4 as predictor.

The B14 result (rho = +0.62 on n=58 cities) was obtained with
weights calibrated on FUB+EMP using the SAME 59-city panel.
A reviewer will ask: does the IMD-4 ranking generalise to
cities not used to calibrate the weights, or is it overfit
to the panel ?

This experiment performs k-fold CV:

  1. Split the 59 panel cities into k = 5 folds.
  2. For each fold:
       - Calibrate the K = 4 Bayesian weights on the cities in
         the four other folds (~47 cities), using their FUB and
         EMP observations.
       - Apply the posterior-median weights to compute the IMD-4
         for the held-out fold's stations -> aggregate to city
         median.
       - Look up INSEE part-velo-travail for the held-out
         cities.
       - Record predicted-IMD vs observed-INSEE pairs.
  3. Concatenate held-out predictions across all 5 folds.
  4. Report out-of-sample Spearman rho and 95% bootstrap CI.

Compared to the in-sample rho = +0.62, the out-of-sample rho
tests whether the IMD-4 is a useful predictor on cities for
which no calibration label is available -- precisely the
operational setting (nowcasting INSEE between census waves
for any commune).

Outputs:
    outputs/b18_holdout_cv_results.json
    outputs/b18_holdout_cv.pdf
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


def main() -> None:
    log.info("Loading panel + components ...")
    panel = load_panel()
    dock, cmm4, _, _, _ = b7.build_design(panel)
    cs4 = dock[["M_norm", "I_norm", "T_norm", "D_norm"]].to_numpy()
    city_codes, city_index = pd.factorize(dock["city"].values)
    n_cities = len(city_index)

    # INSEE reference
    insee = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources" /
        "insee_part_velo_travail_2022.csv",
        dtype={"code_commune": str},
    )
    insee_lookup = dict(zip(insee["city"], insee["insee_part_velo_travail_2022"]))

    # Index panel cities (those with FUB or EMP) into our city_index
    panel_city_set = set(panel.cities)
    panel_idx_in_city = [i for i, c in enumerate(city_index) if c in panel_city_set]
    log.info("  %d panel cities, %d in dock city_index",
             len(panel.cities), len(panel_idx_in_city))

    # cmm4 is the city-level design at the panel ordering (per panel.cities)
    # We CV on the panel cities only.
    rng = np.random.default_rng(2026)
    n_panel = len(panel.cities)
    k = 5
    fold_assign = rng.choice(k, n_panel, replace=True)
    log.info("  k-fold CV with k=%d, fold sizes: %s",
             k, [int((fold_assign == f).sum()) for f in range(k)])

    held_out_predictions: list[dict] = []
    fold_train_weights: list[np.ndarray] = []
    for fold in range(k):
        train_mask = fold_assign != fold
        test_mask = fold_assign == fold
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) < 10 or len(test_idx) < 2:
            continue
        cmm4_tr = cmm4[train_idx]
        fub_tr = np.asarray(panel.fub)[train_idx]
        emp_tr = np.asarray(panel.emp)[train_idx]

        # Calibrate on train
        res = b7.calibrate_k4(cmm4_tr, fub_tr, emp_tr)
        w_med = np.median(res["w_samples"], axis=0)
        fold_train_weights.append(w_med)
        log.info("  fold %d: train_n=%d  test_n=%d  w_med = M=%.2f I=%.2f T=%.2f D=%.2f",
                 fold, len(train_idx), len(test_idx),
                 w_med[0], w_med[1], w_med[2], w_med[3])

        # Apply to test cities -> station-level IMD -> city median
        for ti in test_idx:
            test_city = panel.cities[ti]
            if test_city not in insee_lookup:
                continue
            if not np.isfinite(insee_lookup[test_city]):
                continue
            # Find stations of this city in dock
            ci_in_index = list(city_index).index(test_city) \
                if test_city in city_index else None
            if ci_in_index is None:
                continue
            station_mask = city_codes == ci_in_index
            sta_imd = (cs4[station_mask] @ w_med) * 100.0
            city_imd = float(np.median(sta_imd))
            held_out_predictions.append({
                "city": test_city,
                "fold": int(fold),
                "predicted_imd4": city_imd,
                "observed_insee": float(insee_lookup[test_city]),
            })

    log.info("\nHeld-out predictions collected for %d cities",
             len(held_out_predictions))
    df = pd.DataFrame(held_out_predictions)
    if df.empty:
        log.error("No held-out predictions; aborting.")
        return

    # Out-of-sample rho with bootstrap CI
    x = df["predicted_imd4"].values
    y = df["observed_insee"].values
    rho_oos = sp_stats.spearmanr(x, y).statistic
    n_boot = 1000
    boots = np.empty(n_boot)
    rng_b = np.random.default_rng(2026)
    for b in range(n_boot):
        idx = rng_b.choice(len(x), len(x), replace=True)
        r = sp_stats.spearmanr(x[idx], y[idx]).statistic
        boots[b] = r if np.isfinite(r) else np.nan
    boots = boots[np.isfinite(boots)]
    q025 = float(np.percentile(boots, 2.5))
    q975 = float(np.percentile(boots, 97.5))
    log.info("OUT-OF-SAMPLE rho(IMD-4 predicted, INSEE part-velo) = %+.3f"
             "  95%% CI = [%+.3f, %+.3f]   n = %d",
             rho_oos, q025, q975, len(df))

    # Compare to in-sample
    log.info("\nFor reference, in-sample (B14): rho = +0.621  CI = [+0.391, +0.778]   n = 58")

    # Save
    results = {
        "n_folds": k,
        "n_held_out": int(len(df)),
        "rho_oos": float(rho_oos),
        "ci_oos": [q025, q975],
        "fold_weights_M": [float(w[0]) for w in fold_train_weights],
        "fold_weights_I": [float(w[1]) for w in fold_train_weights],
        "fold_weights_T": [float(w[2]) for w in fold_train_weights],
        "fold_weights_D": [float(w[3]) for w in fold_train_weights],
        "predictions": held_out_predictions,
    }
    out_json = OUT_DIR / "b18_holdout_cv_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: scatter predicted vs observed
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(x, y, s=42, color="#1F3A6B", alpha=0.7,
               edgecolor="white", linewidth=0.5)
    for label, xi, yi in zip(df["city"], x, y):
        if yi > 5 or xi > 50:
            ax.annotate(label, (xi, yi), fontsize=7,
                        xytext=(3, 3), textcoords="offset points",
                        color="#202020")
    ax.set_xlabel("Predicted IMD-4 (calibrated on other folds)")
    ax.set_ylabel("Observed INSEE part-velo-travail 2022 (%)")
    ax.set_title(f"B18: 5-fold hold-out validation  "
                 f"out-of-sample $\\rho = {rho_oos:+.3f}$ "
                 f"[{q025:+.2f}, {q975:+.2f}], n={len(df)}",
                 fontsize=10)
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b18_holdout_cv.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b18_holdout_cv.pdf")


if __name__ == "__main__":
    main()
