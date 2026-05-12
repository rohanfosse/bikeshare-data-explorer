"""E22 -- Mobility-precarity index: who is hit by a bad IMD?

A bad IMD score is more politically consequential in a city with
many car-free households -- they depend on cycling and public
transport for everyday mobility. We define a Mobility Precarity
Index (IPM, French \emph{Indice de Pr\'ecarit\'e de Mobilit\'e})
as

    IPM_i = part_menages_voit0_i * (1 - IMD_i / 100)

interpreted as the share of car-free households exposed to a
sub-optimal cycling environment. The IPM is in [0, 1], with high
values meaning many car-free households AND a low IMD. We rank the
panel by IPM and combine with the Bayesian IES deserts of E9 to
identify cities where the social cost of cycling-environment
deficits is largest.

This experiment substitutes for a direct health/equity coupling
(which would require district-level health indicators not present
in the current Gold Standard release). The IPM is a
policy-relevant downstream outcome under the available data.

Outputs:
    outputs/e22_results.json
    outputs/e22_precarity_ranking.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DESERT_BAYESIAN_9 = {  # from E9 results at tau=1
    "Amiens", "Laon", "Lille", "Lyon", "Nancy", "Niort",
    "Saumur", "Tarbes", "Troyes",
}
DESERT_INVARIANT_4 = {"Amiens", "Lyon", "Nancy", "Saumur"}


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()

    df = panel.socio.copy()
    df["IMD"] = panel.imd
    df["IPM"] = df["part_menages_voit0"] * (1.0 - df["IMD"] / 100.0)
    df["pop_at_risk_share"] = df["part_menages_voit0"]
    df["imd_deficit"] = 100.0 - df["IMD"]

    # Quartiles of IPM
    df["IPM_q"] = pd.qcut(df["IPM"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    log.info("IPM distribution:")
    log.info("  median = %.3f, p25 = %.3f, p75 = %.3f, max = %.3f",
             float(df["IPM"].median()),
             float(df["IPM"].quantile(0.25)),
             float(df["IPM"].quantile(0.75)),
             float(df["IPM"].max()))

    # Top-15 by IPM
    top_ipm = df.nlargest(15, "IPM")[[
        "city", "IMD", "part_menages_voit0", "IPM", "IPM_q",
    ]].reset_index(drop=True)
    log.info("Top-15 cities by mobility precarity:")
    for _, r in top_ipm.iterrows():
        in_desert = ""
        if r["city"] in DESERT_INVARIANT_4:
            in_desert = "  [prior-invariant desert]"
        elif r["city"] in DESERT_BAYESIAN_9:
            in_desert = "  [tau=1 desert]"
        # part_menages_voit0 is already in percent (range ~5-58), don't *100
        log.info(
            "  %-22s  car-free = %5.1f%%  IMD = %5.1f  IPM = %.2f%s",
            r["city"], r["part_menages_voit0"], r["IMD"], r["IPM"], in_desert,
        )

    # Cross-tab: high IPM AND Bayesian desert
    high_ipm_cities = set(df.nlargest(15, "IPM")["city"])
    overlap_inv = high_ipm_cities & DESERT_INVARIANT_4
    overlap_9 = high_ipm_cities & DESERT_BAYESIAN_9
    log.info("Overlap of top-15 IPM with prior-invariant desert set (n=4): %d",
             len(overlap_inv))
    log.info("Overlap of top-15 IPM with tau=1 desert set (n=9): %d",
             len(overlap_9))
    log.info("  intersection: %s", sorted(overlap_9))

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, p = spearmanr(df["IMD"], df["part_menages_voit0"])
    log.info("Spearman(IMD, part_menages_voit0) = %+.3f  p = %.3f",
             float(rho), float(p))

    results = {
        "ipm_distribution": {
            "median": float(df["IPM"].median()),
            "p25": float(df["IPM"].quantile(0.25)),
            "p75": float(df["IPM"].quantile(0.75)),
            "max": float(df["IPM"].max()),
        },
        "top_15_precarity": top_ipm.to_dict("records"),
        "spearman_imd_carfree": {"rho": float(rho), "p": float(p)},
        "overlap_top15_with_desert_invariant": sorted(overlap_inv),
        "overlap_top15_with_desert_tau1": sorted(overlap_9),
        "all_cities": df[[
            "city", "IMD", "part_menages_voit0", "IPM",
        ]].to_dict("records"),
    }
    out_json = OUT_DIR / "e22_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: top-15 IPM with IMD and car-free annotated, deserts highlighted
    top20 = df.nlargest(20, "IPM").sort_values("IPM", ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    colors = []
    for c in top20["city"]:
        if c in DESERT_INVARIANT_4:
            colors.append("#A8201A")  # red
        elif c in DESERT_BAYESIAN_9:
            colors.append("#D08020")  # orange
        else:
            colors.append("#1F3A6B")  # navy
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20["IPM"], color=colors,
            edgecolor="white", linewidth=0.4)
    for j, (_, r) in enumerate(top20.iterrows()):
        ax.text(
            r["IPM"] + 0.3, j,
            f"IMD={r['IMD']:.0f}  car-free={r['part_menages_voit0']:.0f}%",
            fontsize=7, color="#404040", va="center",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["city"], fontsize=8)
    ax.set_xlabel(
        r"Mobility-Precarity Index "
        r"$\mathrm{IPM} = \mathrm{car\text{-}free}\,\times\,(1 - \mathrm{IMD}/100)$"
    )
    ax.set_title("Top-20 cities by mobility precarity\n"
                 "(red = prior-invariant desert, orange = $\\tau=1$ desert)",
                 fontsize=10)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e22_precarity_ranking.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e22_precarity_ranking.pdf")


if __name__ == "__main__":
    main()
