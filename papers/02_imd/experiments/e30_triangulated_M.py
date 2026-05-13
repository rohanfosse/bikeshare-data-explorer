"""E30 -- Re-run IMD with the A6-triangulated multimodality component.

E29 documented a feed-level completeness anomaly (A6) on the
regional GTFS feeds of Lyon, Marseille and Lille. The recommended
remediation from the Gold Standard data-quality programme is to
ingest OSM alongside GTFS and take the maximum.

This experiment applies the A6 patch in isolation: we keep all
other components (I, S, T) at their published Gold Standard
values and replace the M component only for the three affected
cities with the OSM-based count from E29. We then recompute (i)
the IMD for the panel, (ii) the Cook-style leverage of E12 under
the triangulated weights, and (iii) the Bayesian IES posterior
desert probabilities of E9 under the corrected IMD. The aim is
to test how much of the Cook-leverage finding on Lyon and the
desert finding on Lyon/Marseille/Lille are artefacts of the A6
anomaly rather than genuine substantive results.

Inputs:
    experiments/outputs/e29_results.json (M_osm per city)
    Gold Standard panel (loaded via _common.load_panel)

Outputs:
    outputs/e30_results.json
    outputs/e30_triangulated_ranking.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from _common import (
    COMPONENTS,
    calibrate_weights,
    composite_score,
    load_panel,
)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _load_osm_corrections() -> dict[str, float]:
    """Return {city: M_osm} for the three A6-affected cities."""
    e29 = json.loads((OUT_DIR / "e29_results.json").read_text(encoding="utf-8"))
    # A6 incidence threshold: OSM/GTFS ratio > 5 keeps Lyon/Marseille/Lille
    a6_threshold = 5.0
    out = {}
    for r in e29["per_city"]:
        if r.get("ratio_osm_to_gtfs") and r["ratio_osm_to_gtfs"] >= a6_threshold:
            out[r["city"]] = float(r["M_osm"])
    return out


def main() -> None:
    log.info("Loading panel and OSM corrections...")
    panel = load_panel()
    osm_corr = _load_osm_corrections()
    log.info("  A6 corrections (ratio OSM/GTFS >= 5): %s", osm_corr)

    # Raw M values that were Min-Max normalised inside compute_imd_cities.
    # Reverse-engineer by extracting the dock-based per-city raw M from
    # the stations table.
    import sys
    from _common import ROOT
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    st = load_stations()
    dock = st[st["station_type"] == "docked_bike"]
    raw_M = dock.groupby("city")["gtfs_heavy_stops_300m"].mean()

    # Build the triangulated raw M: max(GTFS, OSM) for A6 cities,
    # unchanged otherwise.
    triang_M = raw_M.copy()
    for c, m_osm in osm_corr.items():
        if c in triang_M.index:
            old = triang_M[c]
            new = max(old, m_osm)
            triang_M[c] = new
            log.info("  %-15s  GTFS = %.2f -> triangulated = %.2f",
                     c, old, new)

    # Re-normalise on the new raw M and recompute components matrix
    # only over the cities of the panel (n=59).
    panel_cities = panel.cities
    M_old_norm = panel.components[:, 0]
    M_new_raw = np.array([triang_M.get(c, raw_M.get(c, np.nan))
                          for c in panel_cities])
    raw_panel = np.array([raw_M.get(c, np.nan) for c in panel_cities])

    # Min-Max re-normalisation
    lo, hi = np.nanmin(M_new_raw), np.nanmax(M_new_raw)
    if hi == lo:
        M_new_norm = np.full_like(M_new_raw, 0.5)
    else:
        M_new_norm = (M_new_raw - lo) / (hi - lo)
    M_new_norm = np.nan_to_num(M_new_norm, nan=0.5)

    log.info("Raw M panel statistics (before / after triangulation):")
    log.info("  median = %.2f / %.2f", np.nanmedian(raw_panel), np.nanmedian(M_new_raw))
    log.info("  max    = %.2f / %.2f", np.nanmax(raw_panel), np.nanmax(M_new_raw))

    components_new = panel.components.copy()
    components_new[:, 0] = M_new_norm
    # Other components unchanged

    # Re-calibrate with the same DE procedure on the corrected matrix
    log.info("Re-calibrating weights on the corrected component matrix...")
    w_new, rho_new = calibrate_weights(
        components_new, panel.fub, panel.emp, seed=42, maxiter=200,
    )
    log.info("  w_new = %s   rho_new = %.3f",
             dict(zip(COMPONENTS, [round(float(x), 3) for x in w_new])),
             rho_new)

    # Reference weights for comparison
    w_ref, rho_ref = calibrate_weights(
        panel.components, panel.fub, panel.emp, seed=42, maxiter=200,
    )
    log.info("  w_ref = %s   rho_ref = %.3f",
             dict(zip(COMPONENTS, [round(float(x), 3) for x in w_ref])),
             rho_ref)

    # Two side-by-side IMDs
    imd_ref = composite_score(w_ref, panel.components)
    imd_new = composite_score(w_new, components_new)
    panel_rank_ref = np.argsort(-imd_ref)
    panel_rank_new = np.argsort(-imd_new)

    # Per-city deltas
    delta = imd_new - imd_ref
    tab = pd.DataFrame({
        "city": panel_cities,
        "imd_ref": imd_ref,
        "imd_triangulated": imd_new,
        "delta_imd": delta,
        "rank_ref": np.argsort(np.argsort(-imd_ref)) + 1,
        "rank_new": np.argsort(np.argsort(-imd_new)) + 1,
    })
    tab["rank_change"] = tab["rank_ref"] - tab["rank_new"]  # positive = improved
    tab = tab.sort_values("delta_imd", ascending=False)
    log.info("\nTop-15 cities by IMD uplift after A6 triangulation:")
    for _, r in tab.head(15).iterrows():
        log.info("  %-20s  IMD %.1f -> %.1f  (delta=%+5.1f)  rank %d -> %d (%+d)",
                 r["city"], r["imd_ref"], r["imd_triangulated"],
                 r["delta_imd"], r["rank_ref"], r["rank_new"],
                 r["rank_change"])

    # Cook-style leverage for Lyon under both weight schemes
    log.info("\nCook-style leverage check on Lyon:")
    try:
        idx_lyon = panel_cities.index("Lyon")
        keep = np.ones(panel.n, dtype=bool); keep[idx_lyon] = False
        w_lyon_out_ref, _ = calibrate_weights(
            panel.components[keep], panel.fub[keep], panel.emp[keep],
            seed=42, maxiter=120,
        )
        w_lyon_out_new, _ = calibrate_weights(
            components_new[keep], panel.fub[keep], panel.emp[keep],
            seed=42, maxiter=120,
        )
        d_ref = float(np.linalg.norm(w_lyon_out_ref - w_ref) / np.linalg.norm(w_ref))
        d_new = float(np.linalg.norm(w_lyon_out_new - w_new) / np.linalg.norm(w_new))
        log.info("  D_Lyon (ref weights, GTFS M)         = %.3f", d_ref)
        log.info("  D_Lyon (triangulated M)              = %.3f", d_new)
        log.info("  Lyon leverage change %+.3f (%s)",
                 d_new - d_ref,
                 "reduced" if d_new < d_ref else "increased")
    except ValueError:
        log.warning("Lyon not in panel")
        d_ref = d_new = float("nan")

    # Bayesian IES check on Lyon/Marseille/Lille
    log.info("\nIES Bayesian recheck for A6 cities under triangulated IMD:")
    # Use simple closed-form Bayesian regression as in E9
    from e9_bayesian_ies import _bayesian_linreg_posterior, _standardise, PREDICTORS
    socio = panel.socio
    x_raw = socio[PREDICTORS].to_numpy(dtype=float)
    mask = np.all(np.isfinite(x_raw), axis=1)
    x = _standardise(x_raw[mask])
    y_old = imd_ref[mask]
    y_new = imd_new[mask]
    cities_mask = [c for c, m in zip(panel_cities, mask) if m]
    rng = np.random.default_rng(2026)
    rng2 = np.random.default_rng(2026)
    beta_old, sigma2_old = _bayesian_linreg_posterior(x, y_old, tau=1.0, rng=rng)
    beta_new, sigma2_new = _bayesian_linreg_posterior(x, y_new, tau=1.0, rng=rng2)
    x_full = np.column_stack([np.ones(x.shape[0]), x])
    y_hat_old = x_full @ beta_old.T
    y_hat_new = x_full @ beta_new.T
    ies_old = y_old[:, None] / np.clip(y_hat_old, 1e-3, None)
    ies_new = y_new[:, None] / np.clip(y_hat_new, 1e-3, None)
    p_desert_old = (ies_old < 0.85).mean(axis=1)
    p_desert_new = (ies_new < 0.85).mean(axis=1)
    for city in ["Lyon", "Marseille", "Lille"]:
        if city in cities_mask:
            i = cities_mask.index(city)
            log.info("  %-12s  P(desert) %.2f -> %.2f   IMD %.1f -> %.1f",
                     city, p_desert_old[i], p_desert_new[i],
                     y_old[i], y_new[i])

    # Ranking stability
    tau, _ = kendalltau(imd_ref, imd_new)
    rho, _ = spearmanr(imd_ref, imd_new)
    top10_old = set(panel_cities[i] for i in panel_rank_ref[:10])
    top10_new = set(panel_cities[i] for i in panel_rank_new[:10])
    overlap = len(top10_old & top10_new)
    log.info("\nRanking stability under triangulation:")
    log.info("  Kendall tau(ref, triangulated) = %.3f", tau)
    log.info("  Spearman rho                    = %.3f", rho)
    log.info("  Top-10 overlap                  = %d / 10", overlap)
    log.info("  Lost from Top-10:    %s", sorted(top10_old - top10_new))
    log.info("  Gained in Top-10:    %s", sorted(top10_new - top10_old))

    results = {
        "a6_corrections": osm_corr,
        "weights_reference": dict(zip(COMPONENTS, [float(x) for x in w_ref])),
        "weights_triangulated": dict(zip(COMPONENTS, [float(x) for x in w_new])),
        "rho_reference": float(rho_ref),
        "rho_triangulated": float(rho_new),
        "kendall_tau_ref_vs_triangulated": float(tau),
        "spearman_rho_ref_vs_triangulated": float(rho),
        "top10_overlap": int(overlap),
        "top10_lost": sorted(top10_old - top10_new),
        "top10_gained": sorted(top10_new - top10_old),
        "lyon_cook_D_ref": d_ref,
        "lyon_cook_D_triangulated": d_new,
        "lyon_cook_delta": float(d_new - d_ref),
        "ies_p_desert_ref": {
            city: float(p_desert_old[cities_mask.index(city)])
            for city in ["Lyon", "Marseille", "Lille"]
            if city in cities_mask
        },
        "ies_p_desert_triangulated": {
            city: float(p_desert_new[cities_mask.index(city)])
            for city in ["Lyon", "Marseille", "Lille"]
            if city in cities_mask
        },
        "per_city": tab.to_dict("records"),
    }
    out_json = OUT_DIR / "e30_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: rank-change view focused on the 3 A6 cities + neighbours
    focus = [c for c in osm_corr.keys()] + ["Strasbourg", "Montpellier",
             "Nantes", "Paris", "Mulhouse", "Bordeaux", "Caen", "Toulouse",
             "Rouen", "Marseille"]
    focus = list(dict.fromkeys(focus))  # dedupe preserving order
    subset = tab[tab["city"].isin(focus)].copy()
    subset = subset.sort_values("imd_ref", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y_pos = np.arange(len(subset))
    for j, (_, r) in enumerate(subset.iterrows()):
        is_a6 = r["city"] in osm_corr
        color = "#A8201A" if is_a6 else "#1F3A6B"
        # Draw arrow from imd_ref to imd_triangulated
        ax.annotate("",
                    xy=(r["imd_triangulated"], j),
                    xytext=(r["imd_ref"], j),
                    arrowprops={"arrowstyle": "->",
                                "color": color, "linewidth": 1.2,
                                "alpha": 0.9})
        ax.scatter(r["imd_ref"], j, marker="o",
                   color=color, s=30,
                   edgecolor="white", linewidth=0.4)
        ax.scatter(r["imd_triangulated"], j, marker="s",
                   color=color, s=30,
                   edgecolor="white", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset["city"], fontsize=8)
    ax.set_xlabel("IMD score")
    ax.set_title(r"E30: IMD shift under A6 triangulation "
                 r"$M = \max(C_M^{\mathrm{GTFS}}, C_M^{\mathrm{OSM}})$",
                 fontsize=10)
    ax.text(0.98, 0.02,
            f"Red = A6 cities (Lyon/Marseille/Lille)\n"
            f"Kendall tau = {tau:.2f}, Top-10 overlap = {overlap}/10",
            transform=ax.transAxes, fontsize=8, color="#404040",
            ha="right", va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                  "alpha": 0.9, "pad": 4})
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e30_triangulated_ranking.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e30_triangulated_ranking.pdf")


if __name__ == "__main__":
    main()
