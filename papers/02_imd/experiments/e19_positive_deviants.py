"""E19 -- Positive and negative deviants in the city-level IMD.

Most policy literature treats the IMD as a ranking: the question
becomes "who is at the top, who is at the bottom?". This experiment
takes the complementary anomaly-detection view: for each city,
compute the residual between observed IMD and the IMD predicted by
the city's socio-economic profile, and identify cities whose
residual is unexpectedly positive (\emph{positive deviants}) or
unexpectedly negative (\emph{negative deviants}).

Positive deviants are the empirical "what works" anchors of the
panel: cities that out-perform their socio-economic profile by a
margin that cannot be explained by chance. Their component
profiles document the policy levers that other cities can copy.

We predict $\\widehat{IMD}_i$ by the same Bayesian normal-inverse-gamma
regression of E9, take the posterior median residual
$r_i = \\IMD_i - \\widehat{IMD}_i$, and flag deviants by the posterior
probability $P(|r_i| > \\rho \\cdot \\sigma_y)$ with $\\rho = 1$.

Outputs:
    outputs/e19_results.json
    outputs/e19_deviants.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import COMPONENTS, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(2026)
N_POST = 4000
PREDICTORS = [
    "revenu_median_uc",
    "gini_revenu",
    "part_menages_voit0",
    "part_velo_travail",
]


def _bayes_lr(x: np.ndarray, y: np.ndarray, tau: float = 1.0,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(0)
    x_full = np.column_stack([np.ones(x.shape[0]), x])
    p = x_full.shape[1]
    n = x_full.shape[0]
    v_n = np.linalg.inv(x_full.T @ x_full + tau * np.eye(p))
    mu_n = v_n @ (x_full.T @ y)
    a_n = 0.01 + n / 2
    quad = float(y @ y - mu_n @ np.linalg.inv(v_n) @ mu_n)
    b_n = 0.01 + 0.5 * max(quad, 1e-12)
    sigma2 = 1.0 / rng.gamma(shape=a_n, scale=1.0 / b_n, size=N_POST)
    L = np.linalg.cholesky(v_n + 1e-9 * np.eye(p))
    z = rng.standard_normal((N_POST, p))
    beta = mu_n[None, :] + np.sqrt(sigma2)[:, None] * (z @ L.T)
    return beta, sigma2


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n = %d cities", panel.n)

    x_raw = panel.socio[PREDICTORS].to_numpy(dtype=float)
    mask = np.all(np.isfinite(x_raw), axis=1) & np.isfinite(panel.imd)
    mu = np.nanmean(x_raw[mask], axis=0)
    sd = np.nanstd(x_raw[mask], axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    x = (x_raw[mask] - mu) / sd
    y = panel.imd[mask]
    cities = [c for c, m in zip(panel.cities, mask) if m]
    components = panel.components[mask]

    beta, sigma2 = _bayes_lr(x, y, tau=1.0, rng=RNG)
    x_full = np.column_stack([np.ones(x.shape[0]), x])
    y_hat = x_full @ beta.T  # (n_cities, n_post)
    residual = y[:, None] - y_hat
    sigma_post = np.sqrt(sigma2)  # (n_post,)

    # Standardised residual posterior
    r_std = residual / sigma_post[None, :]

    # Posterior probabilities
    p_pos_deviant = (r_std > 1.0).mean(axis=1)
    p_neg_deviant = (r_std < -1.0).mean(axis=1)
    p_strong_pos = (r_std > 2.0).mean(axis=1)
    p_strong_neg = (r_std < -2.0).mean(axis=1)
    r_median = np.median(residual, axis=1)
    r_q025 = np.percentile(residual, 2.5, axis=1)
    r_q975 = np.percentile(residual, 97.5, axis=1)

    table = []
    for i, c in enumerate(cities):
        table.append({
            "city": c,
            "imd_obs": float(y[i]),
            "imd_hat_median": float(np.median(y_hat[i])),
            "residual_median": float(r_median[i]),
            "residual_ci95": [float(r_q025[i]), float(r_q975[i])],
            "p_positive_deviant": float(p_pos_deviant[i]),
            "p_negative_deviant": float(p_neg_deviant[i]),
            "p_strong_positive": float(p_strong_pos[i]),
            "p_strong_negative": float(p_strong_neg[i]),
            "components": {
                "M": float(components[i, 0]),
                "I": float(components[i, 1]),
                "S": float(components[i, 2]),
                "T": float(components[i, 3]),
            },
        })

    pos_dev = sorted(
        [t for t in table if t["p_positive_deviant"] >= 0.85],
        key=lambda x: -x["p_positive_deviant"],
    )
    neg_dev = sorted(
        [t for t in table if t["p_negative_deviant"] >= 0.85],
        key=lambda x: -x["p_negative_deviant"],
    )
    log.info("Positive deviants (P(r > 1*sigma) >= 0.85):")
    for t in pos_dev[:10]:
        log.info(
            "  %-22s  IMD = %.1f   pred = %.1f   r = %+.1f   P+ = %.2f",
            t["city"], t["imd_obs"], t["imd_hat_median"],
            t["residual_median"], t["p_positive_deviant"],
        )
    log.info("Negative deviants (P(r < -1*sigma) >= 0.85):")
    for t in neg_dev[:10]:
        log.info(
            "  %-22s  IMD = %.1f   pred = %.1f   r = %+.1f   P- = %.2f",
            t["city"], t["imd_obs"], t["imd_hat_median"],
            t["residual_median"], t["p_negative_deviant"],
        )

    # Component profiles of positive deviants
    pos_cities = [t["city"] for t in pos_dev]
    if pos_cities:
        log.info("Positive deviants -- mean component profile:")
        idx = [cities.index(c) for c in pos_cities]
        for k, name in enumerate(COMPONENTS):
            log.info("  %s = %.3f  (panel mean %.3f)",
                     name,
                     float(components[idx, k].mean()),
                     float(components[:, k].mean()))

    results = {
        "n_cities": int(len(cities)),
        "predictors": PREDICTORS,
        "n_positive_deviants": int(len(pos_dev)),
        "n_negative_deviants": int(len(neg_dev)),
        "positive_deviants": pos_dev[:15],
        "negative_deviants": neg_dev[:15],
        "all_cities": table,
    }
    out_json = OUT_DIR / "e19_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: residual ranking with CIs, positive deviants highlighted
    order = np.argsort([t["residual_median"] for t in table])[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 7.5))
    for plot_i, idx in enumerate(order[:30]):
        t = table[idx]
        c_main = ("#5B7E4F" if t["p_positive_deviant"] >= 0.85
                  else "#A8201A" if t["p_negative_deviant"] >= 0.85
                  else "#7A7A7A")
        ax.errorbar(
            t["residual_median"], -plot_i,
            xerr=np.array([[t["residual_median"] - t["residual_ci95"][0]],
                           [t["residual_ci95"][1] - t["residual_median"]]]),
            fmt="o", color=c_main, ecolor="#404040",
            capsize=3, markersize=5, elinewidth=0.7, capthick=0.7,
        )
    ax.set_yticks(np.arange(0, -min(30, len(order)), -1))
    ax.set_yticklabels(
        [table[order[i]]["city"] for i in range(min(30, len(order)))],
        fontsize=8,
    )
    ax.axvline(0, color="#404040", linewidth=0.6, alpha=0.7)
    ax.set_xlabel(r"Residual $r_i = \mathrm{IMD}_i - \widehat{\mathrm{IMD}}_i$ "
                  r"(posterior median, 95% CI)")
    ax.set_title("Positive (green) and negative (red) deviants of the IMD",
                 fontsize=10)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e19_deviants.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e19_deviants.pdf")


if __name__ == "__main__":
    main()
