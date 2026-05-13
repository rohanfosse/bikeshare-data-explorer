"""B19 -- Feasibility of IMD scoring on the full set of 34,969
French communes.

The B14 result (rho = +0.62 on n=58 panel cities) and the
B18 out-of-sample test (rho = +0.42, k=5 CV) both work on
the 59-city VLS panel. A reviewer will ask: does the
methodology generalise to communes WITHOUT a bike-sharing
system ? Of France's ~35,000 communes, ours is < 0.2 %.

This experiment establishes feasibility on the full
national panel.

For all 34,969 communes we have direct data for:
  - D = log(population density)  -- computable everywhere
       (INSEE communes_meta)
  - INSEE part-velo-travail-2022 -- our reference
       (downloaded in B14)

Other IMD components are NOT yet computable at national
scale from our current pipeline:
  - M (multimodality) requires GTFS heavy stops within 300m
    of each commune centroid -- needs national GTFS aggregation
    (transport.data.gouv.fr publishes it; the join is feasible
    but heavy and out of scope for this paper).
  - I (infrastructure) requires the Cerema national cycling
    network at commune level -- the dataset exists but is
    distributed in OSM-tile form and requires aggregation.
  - T (topography) requires national elevation -- BD ALTI 25 m
    nationally, computable but heavy.

We therefore run TWO restricted national experiments:

  (a) D-alone vs INSEE on all 34,969 communes:
       gives the predictive power of the cheapest-to-compute
       single component at full national scale.

  (b) D-alone vs INSEE on the 59-city panel:
       comparison point against (a) and against the IMD-4
       composite on the panel (B14).

This bounds what is achievable now (one component on the
whole country) versus what could be achievable with the
full pipeline extended.

Outputs:
    outputs/b19_national_results.json
    outputs/b19_national_scatter.pdf
"""
from __future__ import annotations

import json
import logging
import sys
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

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def bootstrap_rho(x, y, n_boot=1000, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(x)
    rho = sp_stats.spearmanr(x, y).statistic
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r = sp_stats.spearmanr(x[idx], y[idx]).statistic
        boots[b] = r if np.isfinite(r) else np.nan
    boots = boots[np.isfinite(boots)]
    return (float(rho),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def main() -> None:
    log.info("Loading commune meta (population, surface, density)...")
    meta = pd.read_csv(
        ROOT / "data" / "external" / "insee_communes" / "communes_meta.csv",
        dtype={"code_commune": str},
    )
    log.info("  %d communes with meta", len(meta))
    meta["log_density"] = np.log(meta["density_per_km2"].clip(lower=1.0))

    log.info("Loading INSEE part-velo-travail (national)...")
    mobpro = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" /
        "part-actifs-modes-transport-com.csv",
        dtype={"code_com": str},
        low_memory=False,
    )
    velo = mobpro[(mobpro["mode_transport"].str.contains("V.lo", regex=True, na=False))
                   & (mobpro["annee"] == 2022)].copy()
    velo["valeur"] = pd.to_numeric(velo["valeur"], errors="coerce")
    log.info("  %d commune rows with velo value", len(velo))

    # Merge
    df = meta.merge(velo[["code_com", "valeur"]],
                     left_on="code_commune", right_on="code_com", how="inner")
    df = df.rename(columns={"valeur": "insee_part_velo"})
    df = df[np.isfinite(df["log_density"]) & np.isfinite(df["insee_part_velo"])]
    log.info("  %d communes after merge with finite values", len(df))
    log.info("  velo share quantiles: 10%%=%.2f  median=%.2f  90%%=%.2f",
             float(df["insee_part_velo"].quantile(0.10)),
             float(df["insee_part_velo"].median()),
             float(df["insee_part_velo"].quantile(0.90)))

    # === (a) D-alone vs INSEE on full national panel ===
    log.info("\n=== (a) Density alone vs INSEE on all 34k communes ===")
    rho_nat, q025_nat, q975_nat = bootstrap_rho(
        df["log_density"].values,
        df["insee_part_velo"].values,
        n_boot=500,
    )
    log.info("  rho(log_density, INSEE part-velo) = %+.3f  CI=[%+.3f, %+.3f]   n=%d",
             rho_nat, q025_nat, q975_nat, len(df))

    # Restrict to communes with non-zero velo (more meaningful)
    df_nz = df[df["insee_part_velo"] > 0.5]
    rho_nz, q025_nz, q975_nz = bootstrap_rho(
        df_nz["log_density"].values,
        df_nz["insee_part_velo"].values,
        n_boot=500,
    )
    log.info("  restricted to part-velo > 0.5%%: rho=%+.3f  CI=[%+.3f, %+.3f]   n=%d",
             rho_nz, q025_nz, q975_nz, len(df_nz))

    # === (b) D-alone vs INSEE on the 59 panel cities ===
    log.info("\n=== (b) Density alone vs INSEE on 59-city panel ===")
    panel_insee = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources" /
        "insee_part_velo_travail_2022.csv",
        dtype={"code_commune": str},
    )
    panel_merged = panel_insee.merge(meta[["code_commune", "log_density"]],
                                       on="code_commune", how="inner")
    rho_pan, q025_pan, q975_pan = bootstrap_rho(
        panel_merged["log_density"].values,
        panel_merged["insee_part_velo_travail_2022"].values,
        n_boot=500,
    )
    log.info("  rho(log_density, INSEE) on panel = %+.3f  CI=[%+.3f, %+.3f]   n=%d",
             rho_pan, q025_pan, q975_pan, len(panel_merged))

    # For reference, the full IMD-4 composite on the same panel
    # (read from B14 result)
    b14 = json.loads(
        (OUT_DIR / "b14_tournament_with_insee.json").read_text(encoding="utf-8")
    )
    rho_imd4_panel = b14["table"]["IMD-4 (Bayesian)"]["INSEE velo-travail"]["rho"]
    log.info("  for reference, full IMD-4 on the panel: rho = %+.3f  (n = 58)",
             rho_imd4_panel)

    # === Save ===
    results = {
        "n_communes_national": int(len(df)),
        "national_rho_density_only": rho_nat,
        "national_ci_density_only": [q025_nat, q975_nat],
        "non_zero_n_communes": int(len(df_nz)),
        "non_zero_rho_density_only": rho_nz,
        "non_zero_ci_density_only": [q025_nz, q975_nz],
        "panel_n_cities": int(len(panel_merged)),
        "panel_rho_density_only": rho_pan,
        "panel_ci_density_only": [q025_pan, q975_pan],
        "panel_rho_imd4_full": rho_imd4_panel,
    }
    out_json = OUT_DIR / "b19_national_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("\nWrote %s", out_json)

    # === Figure: log_density vs INSEE on all communes ===
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # Left: national hex bin
    ax = axes[0]
    hb = ax.hexbin(df["log_density"], df["insee_part_velo"],
                    gridsize=50, cmap="Blues", mincnt=1, bins="log")
    ax.set_xlabel(r"$\log_{10}$(commune density, hab./km$^{2}$)")
    ax.set_ylabel(r"INSEE part-velo-travail 2022 (\%)")
    ax.set_ylim(0, 25)
    ax.set_title(f"(a) National scale: 34,629 communes\n"
                 r"$\rho = " f"{rho_nat:+.3f}$ "
                 f"[{q025_nat:+.2f},{q975_nat:+.2f}]",
                 fontsize=10)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.04)
    cbar.set_label("log(commune count)", fontsize=8)
    ax.grid(True, color="#E5E5E5", linewidth=0.5)

    # Right: panel scatter
    ax = axes[1]
    ax.scatter(panel_merged["log_density"],
               panel_merged["insee_part_velo_travail_2022"],
               s=40, color="#1F3A6B", alpha=0.7,
               edgecolor="white", linewidth=0.5)
    for _, row in panel_merged.iterrows():
        if row["insee_part_velo_travail_2022"] > 8 \
           or row["log_density"] > 9.5:
            ax.annotate(row["city"],
                         (row["log_density"], row["insee_part_velo_travail_2022"]),
                         fontsize=7, xytext=(3, 3),
                         textcoords="offset points", color="#202020")
    ax.set_xlabel(r"$\log_{10}$(commune density, hab./km$^{2}$)")
    ax.set_ylabel(r"INSEE part-velo-travail 2022 (\%)")
    ax.set_title(f"(b) 59-city VLS panel\n"
                 r"$\rho_{D} = " f"{rho_pan:+.3f}$"
                 r"$,\ \rho_{IMD-4} = " f"{rho_imd4_panel:+.3f}$",
                 fontsize=10)
    ax.grid(True, color="#E5E5E5", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "b19_national_scatter.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b19_national_scatter.pdf")


if __name__ == "__main__":
    main()
