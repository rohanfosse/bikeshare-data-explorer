"""E7 -- Sobol global sensitivity analysis on component measurement noise.

The IMD aggregates four normalised components on [0, 1] with the
fixed supervised weights. The dominant sources of measurement
uncertainty in the components are the 300 m buffer radius and the
3-year temporal window of the BAAC / GTFS extracts. We model the
joint uncertainty as i.i.d. perturbations on each normalised
component, $C_k \\rightarrow C_k + \\eta_k$ with $\\eta_k \\sim
U[-\\delta, \\delta]$, and compute Saltelli's first-order and
total-effect Sobol indices on (i) each city's IMD value and (ii)
each city's IMD rank in the panel.

Inputs are independent on $[-\\delta, \\delta]^4$ so the Sobol
indices admit the standard variance-decomposition interpretation
without the compositional-data pathology. We set $\\delta = 0.10$,
i.e.\\ a 10-point absolute perturbation on the [0,1] component scale.

This experiment also plays the role of the pre-registered buffer-radius
sensitivity test on the component side, complementary to the
within-city station bootstrap of E4.

Outputs:
    outputs/e7_results.json
    outputs/e7_sobol_panel.pdf
    outputs/e7_sobol_rank.pdf
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
K = 4
N_SAMPLES = 8192
DELTA = 0.10
PUBLISHED_W = np.array([0.374, 0.372, 0.053, 0.201])  # softmax-reparam optimum


def _saltelli_matrices(n: int, k: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """A, B and the K matrices A_B^{(j)} (A with column j set from B).

    This is the Saltelli (2010, Algorithm 1) convention for which
    the linear-model special case gives S_j = ST_j = w_j^2 / sum w_k^2.
    """
    a = RNG.uniform(-DELTA, DELTA, size=(n, k))
    b = RNG.uniform(-DELTA, DELTA, size=(n, k))
    c_list: list[np.ndarray] = []
    for j in range(k):
        c = a.copy()
        c[:, j] = b[:, j]
        c_list.append(c)
    return a, b, c_list


def _imd_perturbed(noise: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Return IMD scores for each row of noise (n_samples, K) and each city.

    Components are clipped to [0,1] after perturbation.
    """
    n = noise.shape[0]
    n_cities = components.shape[0]
    out = np.empty((n, n_cities))
    for i in range(n):
        c_pert = np.clip(components + noise[i], 0.0, 1.0)
        out[i] = c_pert @ PUBLISHED_W * 100.0
    return out


def _rank_perturbed(scores: np.ndarray) -> np.ndarray:
    """Return ranks (1 = best) for each (sample, city) entry."""
    # argsort descending then convert positions to ranks
    order = np.argsort(-scores, axis=1)
    n_samples, n_cities = scores.shape
    ranks = np.empty_like(scores)
    rows = np.arange(n_samples)[:, None]
    ranks[rows, order] = np.arange(1, n_cities + 1)
    return ranks


def _sobol_indices(y_a: np.ndarray, y_b: np.ndarray, y_c: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Saltelli (2010, Algorithm 1) first-order and total-effect.

    For matrices A, B and C^{(j)} := A with column j replaced by B,
        S_j   = (1/N) sum_n Y_B[n] * (Y_C_j[n] - Y_A[n]) / Var(Y_A)
        ST_j  = (1/(2N)) sum_n (Y_A[n] - Y_C_j[n])^2     / Var(Y_A)
    Linear-model special case: S_j = ST_j = w_j^2 / sum_k w_k^2.
    """
    k = len(y_c)
    var_y = y_a.var(axis=0, ddof=1)
    var_y = np.where(var_y < 1e-12, 1.0, var_y)
    S = np.zeros((k, y_a.shape[1]))
    ST = np.zeros((k, y_a.shape[1]))
    for j in range(k):
        S[j] = np.mean(y_b * (y_c[j] - y_a), axis=0) / var_y
        ST[j] = 0.5 * np.mean((y_a - y_c[j]) ** 2, axis=0) / var_y
    return np.clip(S, 0.0, 1.0), np.clip(ST, 0.0, 1.0)


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n=%d cities", panel.n)

    log.info("Saltelli sampling: N=%d, K=%d, delta=%.2f", N_SAMPLES, K, DELTA)
    a, b, c_list = _saltelli_matrices(N_SAMPLES, K)
    log.info("  evaluating IMD on A, B, C^{(j)}...")
    y_a = _imd_perturbed(a, panel.components)
    y_b = _imd_perturbed(b, panel.components)
    y_c = [_imd_perturbed(c, panel.components) for c in c_list]

    log.info("Computing Sobol indices on IMD scores...")
    S_score, ST_score = _sobol_indices(y_a, y_b, y_c)

    log.info("Computing Sobol indices on city ranks...")
    r_a = _rank_perturbed(y_a)
    r_b = _rank_perturbed(y_b)
    r_c = [_rank_perturbed(yc) for yc in y_c]
    S_rank, ST_rank = _sobol_indices(r_a, r_b, r_c)

    # Panel aggregation: weight by city's IMD-score variance
    var_y_per_city = y_a.var(axis=0, ddof=1)
    weights = var_y_per_city / var_y_per_city.sum()
    S_panel_score = (S_score * weights).sum(axis=1)
    ST_panel_score = (ST_score * weights).sum(axis=1)
    # For ranks, weight by rank variance per city
    var_rank_per_city = r_a.var(axis=0, ddof=1)
    weights_rank = var_rank_per_city / max(var_rank_per_city.sum(), 1e-12)
    S_panel_rank = (S_rank * weights_rank).sum(axis=1)
    ST_panel_rank = (ST_rank * weights_rank).sum(axis=1)

    # Bootstrap CI for panel Sobol (score-based)
    n_boot = 200
    boot_S = np.zeros((n_boot, K))
    boot_ST = np.zeros((n_boot, K))
    for b_idx in range(n_boot):
        idx = RNG.integers(0, N_SAMPLES, size=N_SAMPLES)
        S_b, ST_b = _sobol_indices(y_a[idx], y_b[idx], [yc[idx] for yc in y_c])
        boot_S[b_idx] = (S_b * weights).sum(axis=1)
        boot_ST[b_idx] = (ST_b * weights).sum(axis=1)
    S_ci = np.percentile(boot_S, [2.5, 97.5], axis=0)
    ST_ci = np.percentile(boot_ST, [2.5, 97.5], axis=0)

    # Score-uncertainty: stdev of the IMD per city across noise
    score_sd = y_a.std(axis=0, ddof=1)
    rank_sd = r_a.std(axis=0, ddof=1)
    top_idx = np.argsort(-panel.imd)[:15]
    top_cities = [panel.cities[i] for i in top_idx]
    top_sd_score = score_sd[top_idx]
    top_sd_rank = rank_sd[top_idx]

    results = {
        "delta_perturbation": DELTA,
        "n_samples_per_matrix": int(N_SAMPLES),
        "n_total_model_evaluations": int((2 + K) * N_SAMPLES),
        "panel_score_sobol": {
            COMPONENTS[j]: {
                "S": float(S_panel_score[j]),
                "ci95": [float(S_ci[0, j]), float(S_ci[1, j])],
                "ST": float(ST_panel_score[j]),
                "ST_ci95": [float(ST_ci[0, j]), float(ST_ci[1, j])],
            } for j in range(K)
        },
        "panel_rank_sobol": {
            COMPONENTS[j]: {
                "S": float(S_panel_rank[j]),
                "ST": float(ST_panel_rank[j]),
            } for j in range(K)
        },
        "imd_score_sd_top15": {
            top_cities[i]: float(top_sd_score[i])
            for i in range(len(top_cities))
        },
        "imd_rank_sd_top15": {
            top_cities[i]: float(top_sd_rank[i])
            for i in range(len(top_cities))
        },
        "median_score_sd_panel": float(np.median(score_sd)),
        "median_rank_sd_panel": float(np.median(rank_sd)),
        "sum_S_panel_score": float(S_panel_score.sum()),
        "sum_ST_panel_score": float(ST_panel_score.sum()),
        "interaction_idx_panel": float(ST_panel_score.sum() - S_panel_score.sum()),
    }
    out_json = OUT_DIR / "e7_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    log.info("Panel-level Sobol on IMD score (delta=%.2f):", DELTA)
    for j in range(K):
        log.info("  %-12s S=%.3f [%.3f, %.3f]   ST=%.3f [%.3f, %.3f]",
                 COMPONENTS[j],
                 S_panel_score[j], S_ci[0, j], S_ci[1, j],
                 ST_panel_score[j], ST_ci[0, j], ST_ci[1, j])
    log.info("Panel-level Sobol on IMD rank (delta=%.2f):", DELTA)
    for j in range(K):
        log.info("  %-12s S=%.3f   ST=%.3f",
                 COMPONENTS[j], S_panel_rank[j], ST_panel_rank[j])
    log.info("Median score SD across panel = %.2f IMD points", np.median(score_sd))
    log.info("Median rank SD across panel  = %.2f positions",  np.median(rank_sd))

    # ---- Figure 1: panel-level Sobol bar plot --------------------
    labels = ["M\nmultimodality", "I\ninfrastructure",
              "S\nsafety", "T\ntopography"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(K)
    width = 0.38
    s_err = np.vstack([S_panel_score - S_ci[0], S_ci[1] - S_panel_score])
    st_err = np.vstack([ST_panel_score - ST_ci[0], ST_ci[1] - ST_panel_score])
    ax.bar(x - width / 2, S_panel_score, width, color="#1F3A6B",
           yerr=s_err, capsize=3, label=r"First-order $S_k$",
           ecolor="#404040", error_kw={"linewidth": 0.8})
    ax.bar(x + width / 2, ST_panel_score, width, color="#B07A30",
           yerr=st_err, capsize=3, label=r"Total-effect $S_{T,k}$",
           ecolor="#404040", error_kw={"linewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sobol index")
    ax.set_ylim(0, max(ST_panel_score.max() * 1.25, 0.6))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_title(
        rf"Variance decomposition of the IMD ($\delta=\pm{DELTA:.2f}$)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e7_sobol_panel.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e7_sobol_panel.pdf")

    # ---- Figure 2: per-city rank SD for Top-15 ---------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    order = np.argsort(top_sd_rank)
    ax.barh(
        np.array(top_cities)[order],
        top_sd_rank[order],
        color="#1F3A6B",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Rank standard deviation (positions)")
    ax.set_title(
        rf"Panel rank stability under component noise ($\delta=\pm{DELTA:.2f}$)",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    ax.text(0.98, 0.02,
            f"N={N_SAMPLES} Saltelli draws\n"
            f"median rank SD = {np.median(rank_sd):.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#404040",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.85, "pad": 3})
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e7_sobol_rank.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e7_sobol_rank.pdf")


if __name__ == "__main__":
    main()
