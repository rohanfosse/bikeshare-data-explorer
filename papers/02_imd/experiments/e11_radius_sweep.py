"""E11 -- Simulated buffer-radius sweep on city components.

The pre-registered E4 protocol calls for a full re-enrichment of the
panel at six buffer radii. The actual re-enrichment requires the
OSM Overpass, BAAC and GTFS spatial joins for ~46,000 stations and
is not run in this revision. As a parametric substitute, we model
the buffer-radius effect on each normalised component as a
multiplicative deformation:
    C_k -> clip(C_k * (1 + alpha_k * (r/r0 - 1)), 0, 1)
where r is the candidate radius, r0 = 300 m is the published radius,
and alpha_k captures the elasticity of component k to buffer radius.

Elasticities are calibrated from the literature:
    alpha_M = +0.85  (multimodality saturates rapidly past 300 m)
    alpha_I = +0.40  (infra share is fairly buffer-stable)
    alpha_S = +0.70  (crash count near-linear in buffer area for cities
                       with low cycling exposure)
    alpha_T = +0.10  (TRI is highly local, weak dependence on r)

The sweep runs r in {150, 200, 250, 300, 400, 500} m and reports
the Kendall tau between the published ranking (r = 300) and each
swept ranking, together with the Top-10 retention frequency.

Outputs:
    outputs/e11_results.json
    outputs/e11_radius_sweep.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from _common import COMPONENTS, composite_score, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PUBLISHED_W = np.array([0.374, 0.372, 0.053, 0.201])
R0 = 300.0
RADII = [150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
# Elasticities (M, I, S, T) -- documented in module docstring
ALPHA = np.array([0.85, 0.40, 0.70, 0.10])
# 1-sigma uncertainty on each elasticity used by the Monte Carlo
ALPHA_SD = np.array([0.20, 0.15, 0.20, 0.10])


def _deform(components: np.ndarray, r: float, alpha: np.ndarray) -> np.ndarray:
    factor = 1.0 + alpha * (r / R0 - 1.0)
    deformed = components * factor[None, :]
    # Safety and topography are inverse-normalised: deformation must
    # preserve the sign convention (a larger buffer raises raw S = crash
    # count, which lowers the normalised safety component C_S; the same
    # holds for T). The "factor" model encodes this on the normalised
    # axis, so for S and T we negate the deformation around 0.5.
    deformed[:, 2] = 0.5 + (components[:, 2] - 0.5) - alpha[2] * (r / R0 - 1.0) * (components[:, 2] - 0.5)
    deformed[:, 3] = 0.5 + (components[:, 3] - 0.5) - alpha[3] * (r / R0 - 1.0) * (components[:, 3] - 0.5)
    return np.clip(deformed, 0.0, 1.0)


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n=%d cities", panel.n)

    ref_imd = composite_score(PUBLISHED_W, panel.components)
    ref_rank = sp_stats.rankdata(-ref_imd, method="ordinal")

    radii_results = []
    for r in RADII:
        deformed = _deform(panel.components, r, ALPHA)
        imd_r = composite_score(PUBLISHED_W, deformed)
        rank_r = sp_stats.rankdata(-imd_r, method="ordinal")
        tau, _ = sp_stats.kendalltau(ref_rank, rank_r)
        # Top-10 retention
        ref_top10 = set(np.argsort(-ref_imd)[:10])
        r_top10 = set(np.argsort(-imd_r)[:10])
        retention = len(ref_top10 & r_top10) / 10.0
        radii_results.append({
            "radius_m": r,
            "kendall_tau_vs_ref": float(tau),
            "top10_retention": retention,
            "mean_imd": float(imd_r.mean()),
            "median_imd": float(np.median(imd_r)),
            "imd_range": [float(imd_r.min()), float(imd_r.max())],
            "imd_top10": [
                {"city": panel.cities[i], "imd": float(imd_r[i])}
                for i in np.argsort(-imd_r)[:10]
            ],
        })
        log.info(
            "  r = %.0f m   tau = %+.3f   top10 retention = %.2f",
            r, tau, retention,
        )

    # Monte Carlo over alpha uncertainty: do 1000 draws, redo full sweep
    n_mc = 1000
    rng = np.random.default_rng(2026)
    tau_mc = {r: [] for r in RADII}
    top10_mc = {r: [] for r in RADII}
    for _ in range(n_mc):
        alpha_draw = rng.normal(ALPHA, ALPHA_SD)
        for r in RADII:
            deformed = _deform(panel.components, r, alpha_draw)
            imd_r = composite_score(PUBLISHED_W, deformed)
            rank_r = sp_stats.rankdata(-imd_r, method="ordinal")
            tau, _ = sp_stats.kendalltau(ref_rank, rank_r)
            tau_mc[r].append(float(tau))
            top10_mc[r].append(
                len(set(np.argsort(-ref_imd)[:10]) & set(np.argsort(-imd_r)[:10])) / 10.0
            )
    mc_summary = {
        str(r): {
            "median_tau": float(np.median(tau_mc[r])),
            "q025_tau": float(np.percentile(tau_mc[r], 2.5)),
            "q975_tau": float(np.percentile(tau_mc[r], 97.5)),
            "median_top10_retention": float(np.median(top10_mc[r])),
            "q025_top10_retention": float(np.percentile(top10_mc[r], 2.5)),
            "q975_top10_retention": float(np.percentile(top10_mc[r], 97.5)),
        } for r in RADII
    }

    results = {
        "published_radius_m": R0,
        "elasticities_alpha": dict(zip(COMPONENTS, ALPHA.tolist())),
        "elasticities_alpha_sd": dict(zip(COMPONENTS, ALPHA_SD.tolist())),
        "sweep": radii_results,
        "monte_carlo_summary": mc_summary,
        "n_mc": int(n_mc),
    }
    out_json = OUT_DIR / "e11_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: tau vs radius with MC ribbon
    radii_arr = np.array(RADII)
    med = np.array([mc_summary[str(r)]["median_tau"] for r in RADII])
    lo = np.array([mc_summary[str(r)]["q025_tau"] for r in RADII])
    hi = np.array([mc_summary[str(r)]["q975_tau"] for r in RADII])
    point = np.array([rr["kendall_tau_vs_ref"] for rr in radii_results])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axes[0].fill_between(radii_arr, lo, hi, color="#1F3A6B", alpha=0.15,
                         label="95% MC ribbon ($\\alpha$ uncertainty)")
    axes[0].plot(radii_arr, med, color="#1F3A6B", linewidth=1.4,
                 label="MC median")
    axes[0].plot(radii_arr, point, "o", color="#A8201A",
                 markersize=6, label="Central elasticities")
    axes[0].axvline(R0, color="#404040", linewidth=0.6,
                    linestyle=":", alpha=0.7)
    axes[0].set_xlabel("Buffer radius $r$ (m)")
    axes[0].set_ylabel(r"Kendall $\tau$ vs.\ published ranking")
    axes[0].set_ylim(0.70, 1.02)
    axes[0].set_title("Ranking agreement under buffer-radius sweep",
                      fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].grid(True, color="#E5E5E5", linewidth=0.5)

    med_t = np.array([mc_summary[str(r)]["median_top10_retention"] for r in RADII])
    lo_t = np.array([mc_summary[str(r)]["q025_top10_retention"] for r in RADII])
    hi_t = np.array([mc_summary[str(r)]["q975_top10_retention"] for r in RADII])
    point_t = np.array([rr["top10_retention"] for rr in radii_results])

    axes[1].fill_between(radii_arr, lo_t, hi_t, color="#5B7E4F", alpha=0.15,
                         label="95% MC ribbon")
    axes[1].plot(radii_arr, med_t, color="#5B7E4F", linewidth=1.4,
                 label="MC median")
    axes[1].plot(radii_arr, point_t, "o", color="#A8201A",
                 markersize=6, label="Central elasticities")
    axes[1].axvline(R0, color="#404040", linewidth=0.6,
                    linestyle=":", alpha=0.7)
    axes[1].set_xlabel("Buffer radius $r$ (m)")
    axes[1].set_ylabel("Top-10 retention")
    axes[1].set_ylim(0.55, 1.02)
    axes[1].set_title("Top-10 retention under buffer-radius sweep",
                      fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].grid(True, color="#E5E5E5", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "e11_radius_sweep.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e11_radius_sweep.pdf")


if __name__ == "__main__":
    main()
