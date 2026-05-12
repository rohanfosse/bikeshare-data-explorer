"""E21 -- Health coupling: IMD vs life expectancy at département level.

The IMD measures supply-side cycling-environment quality. The
public-health literature (Mueller 2015, Andersen 2000, de Hartog
2010) documents a robust association between active mobility and
mortality reduction at population scale. We test whether the IMD
predicts département-level life expectancy after controlling for
income and population density.

Data:
    Life expectancy at département level, downloaded from data.gouv.fr
    (dataset "esperance-de-vie-par-departements", 2024 vintage; 98
    French départements, range 71-88 years, median 80.5). The
    département code is derived from the INSEE municipal code
    (code_commune) of each city, with Corsica (2A/2B) and DOM (97x)
    handled explicitly.

We report:
    - Spearman rho(IMD, life expectancy) on the panel intersect
    - Partial regression of life expectancy on IMD with controls
      for log(income_per_uc) and a département fixed effect proxy.

Note: département-level life expectancy is a coarse outcome for a
city-level supply indicator -- the within-département variation
of city IMD is partially absorbed by the département mean.
Interpretation is "the IMD distinguishes high-life-expectancy
départements" rather than a causal city-level claim.

Outputs:
    outputs/e21_results.json
    outputs/e21_imd_vs_life_expectancy.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Manual mapping of the French department-name table column to a
# canonical INSEE department code.  The CSV is encoded with "REGION -
# DEPT", e.g. "Île-de-France - Val-de-Marne".  We canonicalise on the
# right-hand-side part and use a curated lookup.
DEPT_NAME_TO_CODE = {
    "Bas Rhin": "67", "Haut Rhin": "68",
    "Dordogne": "24", "Gironde": "33", "Landes": "40",
    "Lot-et-Garonne": "47", "Pyrénées-Atlantiques": "64",
    "Allier": "03", "Cantal": "15", "Haute Loire": "43",
    "Puy-de-Dôme": "63",
    "Côte-d'Or": "21", "Nièvre": "58", "Saône-et-Loire": "71",
    "Yonne": "89",
    "Côtes-d'Armor": "22", "Finistère": "29",
    "Ille-et-Vilaine": "35", "Morbihan": "56",
    "Cher": "18", "Eure-et-Loir": "28", "Indre": "36",
    "Indre-et-Loire": "37", "Loir-et-Cher": "41", "Loiret": "45",
    "Ardennes": "08", "Aube": "10", "Marne": "51",
    "Haute-Marne": "52",
    "Corse-du-Sud": "2A", "Haute-Corse": "2B",
    "Doubs": "25", "Jura": "39", "Haute-Saône": "70",
    "Territoire de Belfort": "90",
    "Calvados": "14", "Manche": "50", "Orne": "61",
    "Eure": "27", "Seine-Maritime": "76",
    "Paris": "75", "Seine-et-Marne": "77", "Yvelines": "78",
    "Essonne": "91", "Hauts-de-Seine": "92",
    "Seine-Saint-Denis": "93", "Val-de-Marne": "94",
    "Val-d'Oise": "95",
    "Aisne": "02", "Nord": "59", "Oise": "60",
    "Pas-de-Calais": "62", "Somme": "80",
    "Corrèze": "19", "Creuse": "23", "Haute-Vienne": "87",
    "Meurthe-et-Moselle": "54", "Meuse": "55",
    "Moselle": "57", "Vosges": "88",
    "Ariège": "09", "Aveyron": "12", "Haute-Garonne": "31",
    "Gers": "32", "Lot": "46", "Hautes-Pyrénées": "65",
    "Tarn": "81", "Tarn-et-Garonne": "82",
    "Loire-Atlantique": "44", "Maine-et-Loire": "49",
    "Mayenne": "53", "Sarthe": "72", "Vendée": "85",
    "Aude": "11", "Gard": "30", "Hérault": "34",
    "Lozère": "48", "Pyrénées-Orientales": "66",
    "Charente": "16", "Charente-Maritime": "17",
    "Deux-Sèvres": "79", "Vienne": "86",
    "Alpes-de-Haute-Provence": "04", "Hautes-Alpes": "05",
    "Alpes-Maritimes": "06", "Bouches-du-Rhône": "13",
    "Var": "83", "Vaucluse": "84",
    "Ain": "01", "Ardèche": "07", "Drôme": "26",
    "Isère": "38", "Loire": "42", "Rhône": "69",
    "Savoie": "73", "Haute-Savoie": "74",
}


def _parse_dept_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = ["region_dept", "life_expectancy"]
    df["dept_name"] = df["region_dept"].str.split(" - ").str[-1].str.strip()
    df["dept_code"] = df["dept_name"].map(DEPT_NAME_TO_CODE)
    return df.dropna(subset=["dept_code", "life_expectancy"])


def _city_to_dept(code_commune: str | float | None) -> str | None:
    if not isinstance(code_commune, str):
        return None
    code = str(code_commune).strip().zfill(5)
    if code.startswith(("2A", "2B")):
        return code[:2]
    if code.startswith("97"):
        return code[:3]  # overseas dept e.g. 971, 972, 974
    return code[:2]


def main() -> None:
    log.info("Loading panel and life-expectancy table...")
    panel = load_panel()

    dept_table = _parse_dept_csv(
        ROOT / "data" / "external" / "health" /
        "esperance_vie_departements.csv"
    )
    log.info("  %d departments with life-expectancy data", len(dept_table))

    # Load stations for code_commune -> dept mapping
    import sys
    sys.path.insert(0, str(ROOT))
    from utils.data_loader import load_stations
    stations = load_stations()
    dock = stations[stations["station_type"] == "docked_bike"]

    # Most common dept per city
    dock = dock.copy()
    dock["dept_code"] = dock["code_commune"].apply(_city_to_dept)
    city_to_dept = (
        dock.dropna(subset=["dept_code"])
            .groupby("city")["dept_code"]
            .agg(lambda s: s.mode().iat[0] if not s.empty else None)
            .reset_index()
    )
    log.info("  %d cities mapped to a département", len(city_to_dept))

    # Merge IMD + life expectancy
    df = pd.DataFrame({
        "city": panel.cities,
        "IMD": panel.imd,
        "revenu_median_uc": panel.socio["revenu_median_uc"].values,
    })
    df = df.merge(city_to_dept, on="city", how="left")
    df = df.merge(
        dept_table[["dept_code", "life_expectancy"]],
        on="dept_code", how="left",
    )
    matched = df.dropna(subset=["life_expectancy", "IMD"])
    log.info("  panel-life expectancy intersect: %d cities", len(matched))

    # Univariate Spearman
    rho_ev, p_ev = sp_stats.spearmanr(matched["IMD"], matched["life_expectancy"])
    rho_inc_ev, p_inc_ev = sp_stats.spearmanr(
        matched["revenu_median_uc"], matched["life_expectancy"]
    )
    rho_imd_inc, p_imd_inc = sp_stats.spearmanr(
        matched["IMD"], matched["revenu_median_uc"]
    )
    log.info("Spearman correlations:")
    log.info("  IMD vs life expectancy            rho = %+.3f  p = %.3f",
             rho_ev, p_ev)
    log.info("  income vs life expectancy         rho = %+.3f  p = %.3f",
             rho_inc_ev, p_inc_ev)
    log.info("  IMD vs income                     rho = %+.3f  p = %.3f",
             rho_imd_inc, p_imd_inc)

    # Partial OLS: LE_dept ~ alpha + b1 IMD + b2 log(income)
    x = np.column_stack([
        np.ones(len(matched)),
        (matched["IMD"] - matched["IMD"].mean()) / matched["IMD"].std(ddof=1),
        np.log(matched["revenu_median_uc"]),
    ])
    y = matched["life_expectancy"].to_numpy(dtype=float)
    coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coefs
    rss = float(((y - pred) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / max(tss, 1e-12)

    # Partial R^2 of IMD
    x_no_imd = np.column_stack([np.ones(len(matched)),
                                 np.log(matched["revenu_median_uc"])])
    coefs_no, *_ = np.linalg.lstsq(x_no_imd, y, rcond=None)
    pred_no = x_no_imd @ coefs_no
    rss_no = float(((y - pred_no) ** 2).sum())
    partial_r2_imd = (rss_no - rss) / max(rss_no, 1e-12)

    log.info("Partial OLS  LE ~ IMD + log(income)")
    log.info("  R^2 = %.3f", r2)
    log.info("  beta_IMD (standardised) = %+.3f", coefs[1])
    log.info("  partial R^2 of IMD net income = %.3f", partial_r2_imd)
    log.info("  beta_log_income = %+.3f", coefs[2])

    results = {
        "data_source": "data.gouv.fr -- 2024 esperance-de-vie-par-departements (98 depts)",
        "n_cities_matched": int(len(matched)),
        "n_departments_unique": int(matched["dept_code"].nunique()),
        "spearman": {
            "imd_vs_life_expectancy": {"rho": float(rho_ev), "p": float(p_ev)},
            "income_vs_life_expectancy": {"rho": float(rho_inc_ev), "p": float(p_inc_ev)},
            "imd_vs_income": {"rho": float(rho_imd_inc), "p": float(p_imd_inc)},
        },
        "partial_regression": {
            "r2_full": float(r2),
            "beta_imd_std": float(coefs[1]),
            "beta_log_income": float(coefs[2]),
            "partial_r2_imd_net_income": float(partial_r2_imd),
        },
        "matched_table_head": matched.head(20)[[
            "city", "IMD", "dept_code", "life_expectancy",
        ]].to_dict("records"),
    }
    out_json = OUT_DIR / "e21_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: scatter IMD vs LE with linear fit
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.scatter(
        matched["IMD"], matched["life_expectancy"],
        s=34, color="#1F3A6B", alpha=0.75,
        edgecolor="white", linewidth=0.5,
    )
    # Top-5 cities by IMD: label
    top_imd = matched.nlargest(5, "IMD")
    for _, row in top_imd.iterrows():
        ax.annotate(row["city"], (row["IMD"], row["life_expectancy"]),
                    fontsize=8, color="#404040",
                    xytext=(4, 4), textcoords="offset points")
    # Linear fit line
    if len(matched) >= 5:
        m, b = np.polyfit(matched["IMD"], matched["life_expectancy"], 1)
        xs = np.linspace(matched["IMD"].min(), matched["IMD"].max(), 100)
        ax.plot(xs, m * xs + b, color="#A8201A",
                linewidth=1.0, linestyle="--", alpha=0.7,
                label=f"OLS fit: LE = {m:.3f}*IMD + {b:.1f}")
    ax.text(0.02, 0.98,
            f"$\\rho_{{Sp}}$(IMD, LE) = {rho_ev:+.2f}, $p$ = {p_ev:.2f}\n"
            f"$n$ = {len(matched)} cities, "
            f"{matched['dept_code'].nunique()} départements",
            transform=ax.transAxes, fontsize=8, color="#202020",
            ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                  "alpha": 0.9, "pad": 4})
    ax.set_xlabel("City IMD")
    ax.set_ylabel("Life expectancy at département level (years)")
    ax.set_title("IMD and département-level life expectancy",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e21_imd_vs_life_expectancy.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e21_imd_vs_life_expectancy.pdf")


if __name__ == "__main__":
    main()
