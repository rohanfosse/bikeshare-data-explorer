"""E9 -- Bayesian Spatial Equity Index with credible intervals.

The published IES is a Ridge point estimate of the residual deviation
of an urban area's IMD from its socio-economic prediction. This
experiment replaces the Ridge point estimator by a fully Bayesian
linear model with a Gaussian prior on the coefficients and a
Jeffreys prior on the residual variance, yielding posterior credible
intervals for both the coefficients and the per-city IES.

The conjugate normal-inverse-gamma prior allows closed-form posterior
sampling without an MCMC dependency:
    beta | sigma^2  ~  N(0, sigma^2 / tau * I)
    sigma^2          ~  Inv-Gamma(a0, b0)

Posterior:
    beta | y, sigma^2  ~  N(mu_n, sigma^2 * V_n)
    sigma^2 | y         ~  Inv-Gamma(a_n, b_n)
where
    V_n   = (X' X + tau * I)^{-1}
    mu_n  = V_n * X' y
    a_n   = a0 + n/2
    b_n   = b0 + 0.5 * (y' y - mu_n' V_n^{-1} mu_n)

We draw 8000 posterior samples and compute per-city posterior IES
distributions IES_i = IMD_i / IMD_hat_i, then the probability that
city i is a mobility desert (IES_i < 0.85).

Outputs:
    outputs/e9_results.json
    outputs/e9_desert_posterior.pdf
    outputs/e9_coefficients_posterior.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(2026)
N_POST = 8000
DESERT_THRESHOLD = 0.85

PREDICTORS = [
    "revenu_median_uc",
    "gini_revenu",
    "part_menages_voit0",
    "part_velo_travail",
]
PREDICTOR_LABELS = {
    "revenu_median_uc": "Income / consumption unit",
    "gini_revenu": "Gini (income)",
    "part_menages_voit0": "Car-free household share",
    "part_velo_travail": "Cycling commute share",
}


def _standardise(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (x - mu) / sd


def _bayesian_linreg_posterior(
    x: np.ndarray, y: np.ndarray, tau: float = 1.0,
    a0: float = 0.01, b0: float = 0.01,
    n_post: int = N_POST, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Conjugate Bayesian linear regression posterior samples.

    Returns (beta_samples [n_post, p+1], sigma2_samples [n_post]).
    A unit-intercept column is added to x.
    """
    rng = rng or np.random.default_rng(0)
    x_full = np.column_stack([np.ones(x.shape[0]), x])
    p = x_full.shape[1]
    n = x_full.shape[0]
    xtx = x_full.T @ x_full
    v_n = np.linalg.inv(xtx + tau * np.eye(p))
    mu_n = v_n @ (x_full.T @ y)
    a_n = a0 + n / 2
    quad = float(y @ y - mu_n @ np.linalg.inv(v_n) @ mu_n)
    b_n = b0 + 0.5 * max(quad, 1e-12)

    # Sample sigma^2 from Inv-Gamma(a_n, b_n)
    sigma2 = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n, size=n_post)
    # Sample beta | sigma^2
    L = np.linalg.cholesky(v_n + 1e-9 * np.eye(p))
    z = rng.standard_normal((n_post, p))
    beta = mu_n[None, :] + np.sqrt(sigma2)[:, None] * (z @ L.T)
    return beta, sigma2


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    socio = panel.socio
    log.info("  n=%d cities", panel.n)

    # Predictors and target
    x_raw = socio[PREDICTORS].to_numpy(dtype=float)
    mask = np.all(np.isfinite(x_raw), axis=1) & np.isfinite(panel.imd)
    x = _standardise(x_raw[mask])
    y = panel.imd[mask]
    cities = [c for c, m in zip(panel.cities, mask) if m]
    log.info("  n_complete = %d", len(cities))

    # Bayesian linear regression
    log.info("Drawing %d posterior samples (tau=1.0)...", N_POST)
    beta, sigma2 = _bayesian_linreg_posterior(x, y, tau=1.0, rng=RNG)
    log.info("  beta shape = %s, sigma2 shape = %s",
             beta.shape, sigma2.shape)

    # Posterior predictive of IMD_hat for each city
    x_full = np.column_stack([np.ones(x.shape[0]), x])
    y_hat_post = x_full @ beta.T  # (n_cities, n_post)
    # Avoid division by very small means
    y_hat_pos = np.clip(y_hat_post, 1e-3, None)
    ies_post = y[:, None] / y_hat_pos  # (n_cities, n_post)

    # Summaries
    coef_means = beta.mean(axis=0)
    coef_q = np.percentile(beta, [2.5, 50, 97.5], axis=0)
    sigma_mean = float(np.sqrt(sigma2).mean())

    log.info("Posterior coefficients (standardised predictors):")
    intercept_lab = ["intercept"] + PREDICTORS
    for i, name in enumerate(intercept_lab):
        log.info(
            "  %-22s  mean=%+.3f   95%% CrI = [%+.3f, %+.3f]",
            name, coef_means[i], coef_q[0, i], coef_q[2, i],
        )

    ies_mean = ies_post.mean(axis=1)
    ies_q = np.percentile(ies_post, [2.5, 50, 97.5], axis=1)
    p_desert = (ies_post < DESERT_THRESHOLD).mean(axis=1)

    desert_summary = []
    for i in np.argsort(-p_desert):
        if p_desert[i] < 0.10:
            continue
        desert_summary.append({
            "city": cities[i],
            "imd_obs": float(y[i]),
            "ies_mean": float(ies_mean[i]),
            "ies_q025": float(ies_q[0, i]),
            "ies_q975": float(ies_q[2, i]),
            "p_desert_post": float(p_desert[i]),
        })

    log.info("Posterior desert probability (cities with P(IES < %.2f) >= 0.10):",
             DESERT_THRESHOLD)
    for ds in desert_summary[:15]:
        log.info(
            "  %-22s  IES = %.2f [%.2f, %.2f]   P(desert) = %.2f",
            ds["city"], ds["ies_mean"],
            ds["ies_q025"], ds["ies_q975"],
            ds["p_desert_post"],
        )

    # Save JSON
    results = {
        "n_posterior_samples": int(N_POST),
        "n_cities_used": int(len(cities)),
        "predictors": PREDICTORS,
        "tau_ridge_prior": 1.0,
        "sigma_posterior_mean": sigma_mean,
        "coefficients_posterior": {
            name: {
                "mean": float(coef_means[i]),
                "q025": float(coef_q[0, i]),
                "median": float(coef_q[1, i]),
                "q975": float(coef_q[2, i]),
                "p_positive": float((beta[:, i] > 0).mean()),
            } for i, name in enumerate(intercept_lab)
        },
        "per_city": [
            {
                "city": cities[i],
                "imd_obs": float(y[i]),
                "ies_mean": float(ies_mean[i]),
                "ies_q025": float(ies_q[0, i]),
                "ies_q975": float(ies_q[2, i]),
                "p_desert_post": float(p_desert[i]),
            } for i in range(len(cities))
        ],
        "robust_deserts_p_geq_0.9": [
            cities[i] for i in range(len(cities)) if p_desert[i] >= 0.9
        ],
        "robust_deserts_p_geq_0.75": [
            cities[i] for i in range(len(cities)) if p_desert[i] >= 0.75
        ],
        "n_robust_deserts_p_geq_0.9": int((p_desert >= 0.9).sum()),
        "n_robust_deserts_p_geq_0.75": int((p_desert >= 0.75).sum()),
    }
    out_json = OUT_DIR / "e9_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)
    log.info("Robust deserts at P >= 0.90: %d", (p_desert >= 0.9).sum())
    log.info("Robust deserts at P >= 0.75: %d", (p_desert >= 0.75).sum())

    # ---- Figure 1: posterior coefficients horizontal forest plot --------
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labels = [PREDICTOR_LABELS[p] for p in PREDICTORS]
    means = coef_means[1:]  # skip intercept
    lo = coef_q[0, 1:]
    hi = coef_q[2, 1:]
    y_pos = np.arange(len(labels))
    err = np.vstack([means - lo, hi - means])
    ax.errorbar(means, y_pos, xerr=err, fmt="o",
                color="#1F3A6B", ecolor="#404040",
                capsize=4, markersize=6,
                elinewidth=1.0, capthick=1.0)
    ax.axvline(0, color="#A8201A", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Posterior coefficient (standardised predictors)")
    ax.set_title("Bayesian IES: posterior coefficients with 95% CrI",
                 fontsize=10)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e9_coefficients_posterior.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e9_coefficients_posterior.pdf")

    # ---- Figure 2: per-city posterior desert probability bar chart -------
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    order = np.argsort(-p_desert)
    top = order[:25]
    colors = [
        "#A8201A" if p >= 0.9
        else "#D08020" if p >= 0.75
        else "#7095C8" if p >= 0.5
        else "#9A9A9A"
        for p in p_desert[top]
    ]
    ax.barh(
        [cities[i] for i in top][::-1],
        p_desert[top][::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.85, color="#A8201A", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axvline(0.75, color="#D08020", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.set_xlabel(r"Posterior $P(\mathrm{IES} < 0.85)$")
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(
        "Posterior probability of being a social mobility desert",
        fontsize=10,
    )
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e9_desert_posterior.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e9_desert_posterior.pdf")


if __name__ == "__main__":
    main()
