"""B3 -- Bayesian Spatial Equity Index with full posterior propagation.

Builds on B1: for each MCMC draw of the IMD weights, computes the
city-level IMD and then regresses it on standardised socio-economic
predictors via a conjugate Bayesian linear regression (normal-
inverse-gamma). The Bayesian IES of city i for posterior draw b is

    IES_i^{(b)} = IMD_i^{(b)} / IMD_hat_i^{(b)}

The posterior on IES_i^{(b)} therefore propagates two sources of
uncertainty: the weight posterior from B1 and the regression
coefficient posterior. We summarise per city by the posterior
median, 95% credible interval, and the posterior probability of
being a mobility desert P(IES < 0.85 | data).

Outputs:
    outputs/b3_ies_results.json
    outputs/b3_ies_top_deserts.pdf
    outputs/b3_ies_coefficients.pdf
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
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(2026)
DESERT_THRESHOLD = 0.85
N_WEIGHT_DRAWS = 500
N_BETA_DRAWS = 500  # posterior draws of regression coefficients per weight draw
TAU = 1.0


PREDICTORS = [
    "revenu_median_uc",
    "gini_revenu",
    "part_menages_voit0",
    "part_velo_travail",
]


def _standardise(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (x - mu) / sd


def _conjugate_posterior_draws(
    x: np.ndarray, y: np.ndarray,
    tau: float = TAU, a0: float = 0.01, b0: float = 0.01,
    n_draws: int = N_BETA_DRAWS, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Conjugate normal-inverse-gamma posterior on (beta, sigma^2).

    Returns (beta_samples (n_draws, p+1), sigma2_samples (n_draws,)).
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
    sigma2 = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n, size=n_draws)
    L = np.linalg.cholesky(v_n + 1e-9 * np.eye(p))
    z = rng.standard_normal((n_draws, p))
    beta = mu_n[None, :] + np.sqrt(sigma2)[:, None] * (z @ L.T)
    return beta, sigma2


def main() -> None:
    log.info("Loading panel and stations...")
    panel = load_panel()
    stations = load_stations()
    dock = b1.normalise_components(stations)
    city_means = dock.groupby("city")[["M_norm", "I_norm", "T_norm"]].mean()
    cmm = city_means.reindex(panel.cities).fillna(city_means.median())
    component_city_means = cmm.to_numpy()
    fub = b1.standardise(panel.fub)
    emp = b1.standardise(np.log1p(panel.emp))

    log.info("Running B1 MH sampler (12 000 keep) to get weight posterior...")
    chain = b1.mh_sample(component_city_means, fub, emp,
                          n_burn=b1.N_BURN, n_keep=b1.N_KEEP)
    z_samples = chain["z"]
    weight_idx = RNG.choice(len(z_samples), size=N_WEIGHT_DRAWS, replace=False)
    w_draws = np.array([b1.softmax_with_floor(z_samples[i])
                         for i in weight_idx])
    log.info("  drew %d weight posterior samples", N_WEIGHT_DRAWS)

    # Build socio-economic predictor matrix on the panel
    x_raw = panel.socio[PREDICTORS].to_numpy(dtype=float)
    mask = np.all(np.isfinite(x_raw), axis=1) & np.isfinite(panel.imd)
    cities = [c for c, m in zip(panel.cities, mask) if m]
    x = _standardise(x_raw[mask])
    n_cities = len(cities)

    # Per weight draw: compute city IMD, regress, sample betas, compute IES
    log.info("Propagating to per-city IES posterior...")
    ies_post = np.zeros((N_WEIGHT_DRAWS * N_BETA_DRAWS, n_cities))
    p_desert_per_city = np.zeros(n_cities)
    n_total = N_WEIGHT_DRAWS * N_BETA_DRAWS
    coef_post = []

    for w_idx, w in enumerate(w_draws):
        # Station-level IMD propagated to city level
        components_station = dock[["M_norm", "I_norm", "T_norm"]].to_numpy()
        station_imd = (components_station @ w) * 100.0
        city_imd = pd.Series(station_imd, index=dock.index).groupby(
            dock["city"]
        ).mean().reindex(cities).fillna(0).to_numpy()
        # Regress: city_imd ~ x with conjugate prior
        beta_samples, _ = _conjugate_posterior_draws(
            x, city_imd, tau=TAU, n_draws=N_BETA_DRAWS,
            rng=np.random.default_rng(2026 + w_idx),
        )
        coef_post.append(beta_samples)
        # Predict IMD_hat per city per beta draw
        x_full = np.column_stack([np.ones(n_cities), x])
        y_hat = x_full @ beta_samples.T  # (n_cities, n_beta_draws)
        ies = city_imd[:, None] / np.clip(y_hat, 1e-3, None)
        ies_post[w_idx*N_BETA_DRAWS:(w_idx+1)*N_BETA_DRAWS, :] = ies.T

    # Summaries
    ies_median = np.median(ies_post, axis=0)
    ies_q025 = np.percentile(ies_post, 2.5, axis=0)
    ies_q975 = np.percentile(ies_post, 97.5, axis=0)
    p_desert = (ies_post < DESERT_THRESHOLD).mean(axis=0)

    rows = []
    for i, c in enumerate(cities):
        rows.append({
            "city": c,
            "ies_median": float(ies_median[i]),
            "ies_q025": float(ies_q025[i]),
            "ies_q975": float(ies_q975[i]),
            "p_desert": float(p_desert[i]),
        })
    rows.sort(key=lambda r: -r["p_desert"])

    log.info("Cities with P(IES < %.2f) >= 0.50:", DESERT_THRESHOLD)
    for r in rows:
        if r["p_desert"] < 0.50:
            continue
        log.info("  %-22s  IES = %.2f [%.2f, %.2f]  P(desert) = %.2f",
                 r["city"], r["ies_median"], r["ies_q025"],
                 r["ies_q975"], r["p_desert"])
    n_robust = sum(1 for r in rows if r["p_desert"] >= 0.90)
    log.info("\nP >= 0.90: %d cities", n_robust)
    log.info("P >= 0.75: %d cities", sum(1 for r in rows if r["p_desert"] >= 0.75))
    log.info("P >= 0.50: %d cities", sum(1 for r in rows if r["p_desert"] >= 0.50))

    # Pool coefficients across weight draws
    coef_all = np.concatenate(coef_post, axis=0)
    coef_names = ["intercept"] + PREDICTORS
    log.info("\nPooled posterior coefficients (across all weight draws):")
    for i, name in enumerate(coef_names):
        m = float(coef_all[:, i].mean())
        q025 = float(np.percentile(coef_all[:, i], 2.5))
        q975 = float(np.percentile(coef_all[:, i], 97.5))
        p_pos = float((coef_all[:, i] > 0).mean())
        log.info("  %-22s  mean = %+.3f  CrI = [%+.3f, %+.3f]  P(>0) = %.2f",
                 name, m, q025, q975, p_pos)

    results = {
        "n_weight_draws": N_WEIGHT_DRAWS,
        "n_beta_draws_per_weight": N_BETA_DRAWS,
        "n_total_draws": n_total,
        "tau_prior": TAU,
        "n_cities": int(n_cities),
        "n_robust_p_geq_0.90": int(n_robust),
        "n_robust_p_geq_0.75": int(sum(1 for r in rows if r["p_desert"] >= 0.75)),
        "per_city_ies": rows,
        "pooled_coefficients": {
            name: {
                "mean": float(coef_all[:, i].mean()),
                "q025": float(np.percentile(coef_all[:, i], 2.5)),
                "q975": float(np.percentile(coef_all[:, i], 97.5)),
                "p_positive": float((coef_all[:, i] > 0).mean()),
            } for i, name in enumerate(coef_names)
        },
    }
    out_json = OUT_DIR / "b3_ies_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: top desert candidates by P(desert)
    sorted_rows = sorted(rows, key=lambda r: -r["p_desert"])[:25]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    y_pos = np.arange(len(sorted_rows))
    colors = []
    for r in sorted_rows:
        if r["p_desert"] >= 0.90:
            colors.append("#A8201A")
        elif r["p_desert"] >= 0.75:
            colors.append("#D08020")
        elif r["p_desert"] >= 0.50:
            colors.append("#7095C8")
        else:
            colors.append("#9A9A9A")
    ax.barh(y_pos, [r["p_desert"] for r in sorted_rows[::-1]],
            color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.axvline(0.90, color="#A8201A", linewidth=0.6,
                linestyle=":", alpha=0.7, label="P = 0.90")
    ax.axvline(0.75, color="#D08020", linewidth=0.6,
                linestyle=":", alpha=0.7, label="P = 0.75")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["city"] for r in sorted_rows[::-1]], fontsize=8)
    ax.set_xlabel(r"Posterior $P(\mathrm{IES} < 0.85 \mid \mathrm{data})$")
    ax.set_title("B3: Bayesian mobility-desert posterior", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b3_ies_top_deserts.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b3_ies_top_deserts.pdf")

    # Figure: pooled coefficient posteriors
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    labels = {
        "revenu_median_uc": "Income / consumption unit",
        "gini_revenu": "Gini (income)",
        "part_menages_voit0": "Car-free household share",
        "part_velo_travail": "Cycling commute share",
    }
    for j, key in enumerate(PREDICTORS):
        col = coef_all[:, j + 1]  # +1 to skip intercept
        ax = axes[j]
        ax.hist(col, bins=40, color="#1F3A6B", edgecolor="white",
                 linewidth=0.4, alpha=0.85)
        ax.axvline(0, color="#A8201A", linewidth=1.0, linestyle="--",
                    alpha=0.7)
        ax.axvline(col.mean(), color="#5B7E4F", linewidth=1.0)
        ax.set_title(labels[key], fontsize=9)
        ax.set_xlabel("Standardised coef.")
        ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    axes[0].set_ylabel("Posterior count")
    fig.suptitle("B3: pooled coefficient posteriors (across all weight draws)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b3_ies_coefficients.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b3_ies_coefficients.pdf")


if __name__ == "__main__":
    main()
