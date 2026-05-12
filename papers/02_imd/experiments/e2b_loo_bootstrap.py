"""E2b -- Empirical weight-distribution Monte Carlo (LOO bootstrap).

The original E2 stress test draws weights uniformly on the simplex,
which is a worst-case robustness probe but conflates two distinct
questions: (i) is the ranking stable across the *full* set of
mathematically admissible weights, and (ii) is the ranking stable
across weights *consistent with the calibration data*. The latter
is the relevant test for any user who trusts the calibration
methodology. We address it here by sampling weights from the
empirical distribution produced by leave-one-city-out recalibration.

Outputs:
    outputs/e2b_results.json
    outputs/e2b_top10_freq_loo.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import COMPONENTS, calibrate_weights, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _loo_weights_bank(panel, n_max: int | None = None) -> np.ndarray:
    """Re-calibrate weights leaving one city out at a time.

    Returns a (n_folds, 4) array of weight vectors. Cities without
    FUB or EMP coverage are skipped (the calibration objective is
    not defined).
    """
    weights_list: list[np.ndarray] = []
    indices = list(range(panel.n))
    if n_max is not None:
        indices = indices[:n_max]
    for i in indices:
        if not (np.isfinite(panel.fub[i]) or np.isfinite(panel.emp[i])):
            continue
        keep = np.ones(panel.n, dtype=bool)
        keep[i] = False
        try:
            w, _ = calibrate_weights(
                panel.components[keep],
                panel.fub[keep],
                panel.emp[keep],
                seed=42 + i,
                maxiter=120,
            )
            weights_list.append(w)
        except Exception as exc:
            log.warning("LOO fold %d failed: %s", i, exc)
    return np.array(weights_list)


def _resample_weights(bank: np.ndarray, n_draws: int, rng) -> np.ndarray:
    """Bootstrap-sample with Gaussian jitter around the LOO mean.

    Draws are constructed as mean + Sigma^{1/2} z + noise, where
    Sigma is the empirical covariance of the bank. After sampling,
    each draw is renormalised onto the simplex with the floor
    constraint applied.
    """
    mean = bank.mean(axis=0)
    cov = np.cov(bank, rowvar=False)
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(len(mean)))
    z = rng.standard_normal((n_draws, len(mean)))
    raw = mean + z @ chol.T
    # Clip and renormalise to simplex with floor
    out = np.empty_like(raw)
    for i, w in enumerate(raw):
        clipped = np.clip(w, 0.05, 1.0)
        out[i] = clipped / clipped.sum()
    return out


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("Building LOO weight bank (this re-runs DE up to %d times)...",
             panel.n)
    bank = _loo_weights_bank(panel)
    log.info("  bank size = %d folds", len(bank))
    log.info("  bank mean:  %s", dict(zip(
        COMPONENTS, [round(float(v), 3) for v in bank.mean(axis=0)])))
    log.info("  bank sd:    %s", dict(zip(
        COMPONENTS, [round(float(v), 4) for v in bank.std(axis=0)])))

    rng = np.random.default_rng(42)
    n_draws = 50_000
    samples = _resample_weights(bank, n_draws, rng)

    counts = np.zeros(panel.n, dtype=int)
    for w in samples:
        score = panel.components @ w
        order = np.argsort(score)[::-1]
        counts[order[:10]] += 1
    freqs = counts / n_draws
    freq_df = pd.DataFrame({"city": panel.cities, "freq": freqs})

    published_top10 = [
        "Strasbourg", "Montpellier", "Nantes", "Mulhouse", "Paris",
        "Caen", "Brest", "Bordeaux", "Dijon", "Rennes",
    ]
    in_panel = [c for c in published_top10 if c in panel.cities]
    retained_at_50 = int(
        (freq_df.set_index("city").loc[in_panel, "freq"] > 0.50).sum()
    )
    retained_at_25 = int(
        (freq_df.set_index("city").loc[in_panel, "freq"] > 0.25).sum()
    )
    median_freq = float(
        freq_df.set_index("city").loc[in_panel, "freq"].median()
    )

    results = {
        "n_loo_folds": int(len(bank)),
        "n_resampled_draws": n_draws,
        "bank_mean": dict(zip(COMPONENTS, bank.mean(axis=0).tolist())),
        "bank_sd": dict(zip(COMPONENTS, bank.std(axis=0).tolist())),
        "published_top10_in_panel": in_panel,
        "published_top10_retained_at_50pct": retained_at_50,
        "published_top10_retained_at_25pct": retained_at_25,
        "published_top10_median_freq": median_freq,
        "top10_frequencies": freq_df.sort_values("freq", ascending=False)
                             .head(20).set_index("city")["freq"].to_dict(),
    }
    out_json = OUT_DIR / "e2b_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    log.info("Published Top-10 retained at >= 50%%: %d / %d",
             retained_at_50, len(in_panel))
    log.info("Published Top-10 retained at >= 25%%: %d / %d",
             retained_at_25, len(in_panel))
    log.info("Median Top-10 frequency of published Top-10 = %.2f", median_freq)

    # Plot
    df_plot = freq_df.nlargest(15, "freq").sort_values("freq")
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = ["#1F3A6B" if c in published_top10 else "#7A7A7A"
              for c in df_plot["city"]]
    ax.barh(df_plot["city"], df_plot["freq"] * 100,
            color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(50, color="#A8201A", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("Top-10 frequency under LOO-bootstrap weights (%)")
    ax.set_xlim(0, 105)
    ax.text(51, 0.3, "50%", color="#A8201A", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e2b_top10_freq_loo.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e2b_top10_freq_loo.pdf")


if __name__ == "__main__":
    main()
