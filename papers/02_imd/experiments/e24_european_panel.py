"""E24 -- European panel comparison: France vs the rest of Europe.

The MobilityData GBFS systems catalogue lists 984 European
bike-sharing deployments across 29 countries. France contributes
$\\sim 122$ to this total; the comparable national panels include
Germany (251), Poland (94), Norway (83), Spain (69), the United
Kingdom (56), Finland (55), the Netherlands (52), Switzerland (49)
and Czechia (45).

We use the catalogue as a panel-frame to test two cross-country
questions \\emph{without} re-querying each operator's live API:

  Q-A. Does the number of GBFS systems per country scale with
       cycling modal share? If yes, the volumetric prior is alive
       and well at the European scale. If no, the IMD-style
       contextual approach has cross-country relevance.

  Q-B. Where do the world's bike-sharing systems cluster? The
       answer informs the design of an enrichment-pipeline
       extension to non-French cities.

We cross-reference the catalogue with country-level cycling
indicators that have stable open sources: the Eurostat-derived
cycling modal share (Survey on Activity, average over urban areas)
and the European Cyclists' Federation cycling-friendliness ranking
2022 are encoded by hand below from publicly available reports.
Country-level matching is approximate by construction.

Outputs:
    outputs/e24_results.json
    outputs/e24_european_panel.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import ROOT

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

EU_COUNTRY_NAMES = {
    "AT": "Austria",   "BE": "Belgium",       "BG": "Bulgaria",
    "CH": "Switzerland","CY": "Cyprus",       "CZ": "Czechia",
    "DE": "Germany",   "DK": "Denmark",       "EE": "Estonia",
    "ES": "Spain",     "FI": "Finland",       "FR": "France",
    "GB": "United Kingdom","GR": "Greece",    "HR": "Croatia",
    "HU": "Hungary",   "IE": "Ireland",       "IT": "Italy",
    "LT": "Lithuania", "LU": "Luxembourg",    "LV": "Latvia",
    "MT": "Malta",     "NL": "Netherlands",   "NO": "Norway",
    "PL": "Poland",    "PT": "Portugal",      "RO": "Romania",
    "SE": "Sweden",    "SI": "Slovenia",      "SK": "Slovakia",
}

# Eurostat-derived cycling modal share (% of trips, urban areas, 2019).
# Source: Eurobarometer "Future of transport" (2019, EB 495). Approximate.
CYCLING_MODAL_SHARE = {
    "NL": 27.0, "DK": 23.0, "DE": 11.0, "FI": 10.0, "BE": 9.0,
    "SE": 8.0,  "HU": 7.0,  "PL": 7.0,  "IT": 6.0,  "CZ": 6.0,
    "AT": 6.0,  "FR": 4.0,  "ES": 4.0,  "PT": 3.0,  "GR": 2.0,
    "GB": 4.0,  "IE": 3.0,  "RO": 5.0,  "SK": 5.0,  "HR": 4.0,
    "BG": 4.0,  "LT": 5.0,  "LV": 4.0,  "EE": 5.0,  "SI": 5.0,
    "NO": 6.0,  "CH": 8.0,  "LU": 3.0,
}

# European Cyclists' Federation 2022 climate-index ranking
# (relative score, higher = more cycling-friendly). Source: ECF report 2022.
ECF_INDEX = {
    "NL": 95, "DK": 90, "DE": 82, "BE": 78, "AT": 75,
    "FI": 73, "SE": 72, "FR": 68, "CZ": 67, "HU": 66,
    "IT": 64, "ES": 63, "PL": 62, "GB": 60, "PT": 58,
    "CH": 78, "NO": 75, "IE": 55, "GR": 50, "HR": 55,
    "BG": 48, "RO": 45, "SI": 60, "SK": 58, "EE": 60,
    "LV": 55, "LT": 55, "LU": 65,
}


def main() -> None:
    log.info("Loading MobilityData GBFS catalogue...")
    cat = pd.read_csv(
        ROOT / "data" / "external" / "european" / "gbfs_systems.csv"
    )
    eu = cat[cat["Country Code"].isin(EU_COUNTRY_NAMES)].copy()
    log.info("  global N = %d, EU+CH+NO N = %d", len(cat), len(eu))

    by_country = (
        eu.groupby("Country Code").size().reset_index(name="n_systems")
    )
    by_country["country"] = by_country["Country Code"].map(EU_COUNTRY_NAMES)
    by_country["modal_share_pct"] = by_country["Country Code"].map(CYCLING_MODAL_SHARE)
    by_country["ecf_index"] = by_country["Country Code"].map(ECF_INDEX)
    by_country = by_country.sort_values("n_systems", ascending=False).reset_index(drop=True)

    log.info("Top-10 EU countries by GBFS system count:")
    for _, r in by_country.head(10).iterrows():
        log.info("  %-15s  N = %3d   modal = %4.1f%%   ECF = %d",
                 r["country"], r["n_systems"],
                 r["modal_share_pct"] if pd.notna(r["modal_share_pct"]) else 0,
                 int(r["ecf_index"]) if pd.notna(r["ecf_index"]) else 0)

    # Cross-country tests
    mask = by_country["modal_share_pct"].notna()
    sub = by_country[mask]
    rho_modal, p_modal = sp_stats.spearmanr(
        sub["n_systems"], sub["modal_share_pct"]
    )
    rho_ecf, p_ecf = sp_stats.spearmanr(
        sub["n_systems"],
        sub["ecf_index"].fillna(sub["ecf_index"].median()),
    )
    rho_modal_ecf, p_modal_ecf = sp_stats.spearmanr(
        sub["modal_share_pct"],
        sub["ecf_index"].fillna(sub["ecf_index"].median()),
    )
    log.info("Cross-country Spearman (n=%d EU+CH+NO countries):", len(sub))
    log.info("  # GBFS systems vs.\\ cycling modal share  rho = %+.3f  p = %.3f",
             rho_modal, p_modal)
    log.info("  # GBFS systems vs.\\ ECF index            rho = %+.3f  p = %.3f",
             rho_ecf, p_ecf)
    log.info("  Modal share vs.\\ ECF index               rho = %+.3f  p = %.3f",
             rho_modal_ecf, p_modal_ecf)

    # France's position
    fr_row = by_country[by_country["Country Code"] == "FR"].iloc[0] if "FR" in by_country["Country Code"].values else None
    if fr_row is not None:
        rank_systems = int((by_country["n_systems"] > fr_row["n_systems"]).sum()) + 1
        rank_modal = int((by_country["modal_share_pct"].fillna(0) > fr_row["modal_share_pct"]).sum()) + 1
        log.info("France: N = %d (rank %d), modal = %.1f%% (rank %d)",
                 fr_row["n_systems"], rank_systems,
                 fr_row["modal_share_pct"], rank_modal)
    else:
        log.warning("France not in catalogue; catalogue may be incomplete.")
        rank_systems, rank_modal = None, None

    results = {
        "n_total_global": int(len(cat)),
        "n_total_eu": int(len(eu)),
        "n_countries": int(by_country.shape[0]),
        "by_country": by_country.to_dict("records"),
        "cross_country": {
            "n_with_modal": int(mask.sum()),
            "spearman_n_systems_vs_modal_share": {
                "rho": float(rho_modal), "p": float(p_modal),
            },
            "spearman_n_systems_vs_ecf": {
                "rho": float(rho_ecf), "p": float(p_ecf),
            },
            "spearman_modal_vs_ecf": {
                "rho": float(rho_modal_ecf), "p": float(p_modal_ecf),
            },
        },
        "france_position": {
            "n_systems": int(fr_row["n_systems"]) if fr_row is not None else None,
            "rank_systems": rank_systems,
            "modal_share_pct": float(fr_row["modal_share_pct"]) if fr_row is not None else None,
            "rank_modal": rank_modal,
        },
    }
    out_json = OUT_DIR / "e24_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: scatter # systems vs modal share, with FR highlighted
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    ax = axes[0]
    sub_plot = by_country.dropna(subset=["modal_share_pct"])
    colors = ["#A8201A" if c == "FR" else "#1F3A6B"
              for c in sub_plot["Country Code"]]
    ax.scatter(sub_plot["modal_share_pct"], sub_plot["n_systems"],
               s=60, color=colors, alpha=0.75,
               edgecolor="white", linewidth=0.5)
    for _, r in sub_plot.iterrows():
        if r["n_systems"] >= 30 or r["Country Code"] == "FR":
            ax.annotate(r["Country Code"],
                        (r["modal_share_pct"], r["n_systems"]),
                        fontsize=8, color="#202020",
                        xytext=(4, 4), textcoords="offset points")
    ax.text(0.02, 0.98,
            f"$\\rho_{{Sp}}$ = {rho_modal:+.2f},  $p$ = {p_modal:.2f},  "
            f"$n$ = {int(mask.sum())}",
            transform=ax.transAxes, fontsize=8, color="#202020",
            ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                  "alpha": 0.9, "pad": 4})
    ax.set_xlabel(r"Cycling modal share, urban areas (\%, EB 2019)")
    ax.set_ylabel("GBFS systems in the country (MobilityData 2026)")
    ax.set_title("Volumetric prior at the European scale",
                 fontsize=10)
    ax.grid(True, color="#E5E5E5", linewidth=0.5)

    ax = axes[1]
    by_country.head(15).iloc[::-1].plot.barh(
        y="n_systems", x="country", ax=ax,
        color=["#A8201A" if c == "FR" else "#1F3A6B"
               for c in by_country.head(15).iloc[::-1]["Country Code"]],
        legend=False,
    )
    ax.set_xlabel("Number of GBFS systems")
    ax.set_ylabel("")
    ax.set_title("Top-15 EU countries by GBFS deployment count",
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)

    fig.suptitle("European bike-sharing landscape "
                 f"(N = {len(eu)} systems in {by_country.shape[0]} countries)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e24_european_panel.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e24_european_panel.pdf")


if __name__ == "__main__":
    main()
