"""B22 -- Regional decomposition by current and previous political
regime of the Conseil Regional.

Cycling infrastructure investment in France is partly driven
by Region-level subsidies (Plans Velo regionaux, TER cycling
accommodation) and by Region-led Schemas Regionaux de
Coherence Ecologique. A natural question is whether the
political regime of the Conseil Regional shapes the cycling
environment quality (IMD-4) and equity (IES) of communes
inside its territory.

We aggregate the IMD-4 and IES of B20 and B21 to the regional
level and stratify by:

  - 2015 election majority (effective 2016-2021)
  - 2021 election majority (effective 2021-2028)

Categories: 'droite' (LR / Centre / UDI), 'gauche' (PS / EELV
/ DVG / Communiste / Indep.), 'regionaliste' (Corse), and 'NA'
for missing.

This is descriptive, not causal -- many cycling investments
predate the current regime and most municipal cycling policy
is independent of the Region.  We report ANOVA-like means
across regimes, with bootstrap CIs.

Outputs:
    outputs/b22_regional_political_results.json
    outputs/b22_regional_political.pdf
"""
from __future__ import annotations

import json
import logging
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


# Mapping department code -> region code (INSEE / post-2016 regions).
DEP_TO_REGION: dict[str, str] = {}
# Metropolitan
for d, reg in [
    # Auvergne-Rhone-Alpes (84)
    *[(d, "84") for d in ["01","03","07","15","26","38","42","43","63","69","73","74"]],
    # Bourgogne-Franche-Comte (27)
    *[(d, "27") for d in ["21","25","39","58","70","71","89","90"]],
    # Bretagne (53)
    *[(d, "53") for d in ["22","29","35","56"]],
    # Centre-Val de Loire (24)
    *[(d, "24") for d in ["18","28","36","37","41","45"]],
    # Corse (94)
    *[(d, "94") for d in ["2A","2B"]],
    # Grand Est (44)
    *[(d, "44") for d in ["08","10","51","52","54","55","57","67","68","88"]],
    # Hauts-de-France (32)
    *[(d, "32") for d in ["02","59","60","62","80"]],
    # Ile-de-France (11)
    *[(d, "11") for d in ["75","77","78","91","92","93","94","95"]],
    # Normandie (28)
    *[(d, "28") for d in ["14","27","50","61","76"]],
    # Nouvelle-Aquitaine (75)
    *[(d, "75") for d in ["16","17","19","23","24","33","40","47","64","79","86","87"]],
    # Occitanie (76)
    *[(d, "76") for d in ["09","11","12","30","31","32","34","46","48","65","66","81","82"]],
    # Pays de la Loire (52)
    *[(d, "52") for d in ["44","49","53","72","85"]],
    # Provence-Alpes-Cote d'Azur (93)
    *[(d, "93") for d in ["04","05","06","13","83","84"]],
    # Overseas - separate codes
    ("971", "01"), ("972", "02"), ("973", "03"),
    ("974", "04"), ("976", "06"),
]:
    DEP_TO_REGION[d] = reg


REGION_NAME = {
    "11": "Ile-de-France", "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comte", "28": "Normandie",
    "32": "Hauts-de-France", "44": "Grand Est", "52": "Pays de la Loire",
    "53": "Bretagne", "75": "Nouvelle-Aquitaine", "76": "Occitanie",
    "84": "Auvergne-Rhone-Alpes", "93": "PACA", "94": "Corse",
    "01": "Guadeloupe", "02": "Martinique", "03": "Guyane",
    "04": "La Reunion", "06": "Mayotte",
}


# Political color: outcome of the 2nd-round Conseil Regional election.
# 'droite' = LR / UDI / Centre majority; 'gauche' = PS / EELV / DVG /
# communist; 'regionaliste' = Corse; 'NA' if unstable / unique.
REGIME_2015 = {
    "11": "droite",      # Pecresse (LR)
    "24": "gauche",      # Bonneau (PS)
    "27": "gauche",      # Dufay (PS)
    "28": "droite",      # Morin (UDI/Centre)
    "32": "droite",      # Bertrand (LR)
    "44": "droite",      # Richert (LR)
    "52": "droite",      # Retailleau then Morancais (LR)
    "53": "gauche",      # Le Drian then Chesnais-Girard (PS)
    "75": "gauche",      # Rousset (PS)
    "76": "gauche",      # Delga (PS)
    "84": "droite",      # Wauquiez (LR)
    "93": "droite",      # Estrosi (LR)
    "94": "regionaliste",  # Simeoni
    "01": "gauche",      # Chalus
    "02": "gauche",      # Letchimy
    "03": "gauche",      # divers gauche
    "04": "droite",      # Robert (LR) 2015-2021
    "06": "droite",      # MDM/LR
}

REGIME_2021 = {
    "11": "droite",       # Pecresse (LR) re-elected
    "24": "gauche",       # Bonneau (PS) re-elected
    "27": "gauche",       # Dufay (PS) re-elected
    "28": "droite",       # Morin re-elected
    "32": "droite",       # Bertrand re-elected
    "44": "droite",       # Rottner then Leroy (LR)
    "52": "droite",       # Morancais re-elected
    "53": "gauche",       # Chesnais-Girard re-elected
    "75": "gauche",       # Rousset re-elected
    "76": "gauche",       # Delga re-elected
    "84": "droite",       # Wauquiez re-elected
    "93": "droite",       # Muselier (LR)
    "94": "regionaliste",
    "01": "gauche",       # Chalus
    "02": "gauche",       # Letchimy
    "03": "gauche",
    "04": "gauche",       # Bello (PLR / gauche) FLIPPED 2021
    "06": "droite",
}


def commune_to_region(code_com: str) -> str | None:
    """Map an INSEE commune code to its INSEE region code."""
    if not isinstance(code_com, str) or len(code_com) < 2:
        return None
    if code_com.startswith("97") and len(code_com) >= 3:
        dep = code_com[:3]
    elif code_com.startswith("2A") or code_com.startswith("2B"):
        dep = code_com[:2]
    else:
        dep = code_com[:2]
    return DEP_TO_REGION.get(dep)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 500, seed: int = 2026):
    rng = np.random.default_rng(seed)
    mu = float(np.mean(x))
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boots[b] = float(np.mean(x[idx]))
    return (mu,
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def main() -> None:
    log.info("Loading IES and IMD-4 per commune ...")
    ies = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources" /
        "ies_national_communes.csv",
        dtype={"code_commune": str},
    )
    log.info("  %d communes with IES + IMD-4", len(ies))

    # Map to region
    ies["region_code"] = ies["code_commune"].map(commune_to_region)
    ies["region_name"] = ies["region_code"].map(REGION_NAME)
    ies["regime_2015"] = ies["region_code"].map(REGIME_2015)
    ies["regime_2021"] = ies["region_code"].map(REGIME_2021)
    n_mapped = ies["region_code"].notna().sum()
    log.info("  %d communes mapped to a region (%.1f%%)",
             n_mapped, 100.0 * n_mapped / len(ies))

    # Regional means
    log.info("\n===== Regional means of IMD-4 and IES =====")
    reg_means = ies.groupby(["region_code", "region_name",
                              "regime_2015", "regime_2021"]).agg(
        n=("code_commune", "count"),
        imd_mean=("IMD4_national", "mean"),
        imd_median=("IMD4_national", "median"),
        ies_mean=("IES", "mean"),
        ies_median=("IES", "median"),
        income_median=("income_median", "median"),
        poverty_mean=("poverty_rate", "mean"),
    ).reset_index().sort_values("imd_mean", ascending=False)
    log.info("\n%s", reg_means.to_string(index=False))

    # ===== Aggregate by regime (current 2021) =====
    log.info("\n===== Aggregation by 2021 regime =====")
    regime_2021_stats = {}
    for reg, grp in ies.groupby("regime_2021"):
        if reg not in ("droite", "gauche", "regionaliste"):
            continue
        mu_imd, lo_imd, hi_imd = bootstrap_mean_ci(grp["IMD4_national"].values)
        mu_ies, lo_ies, hi_ies = bootstrap_mean_ci(grp["IES"].values)
        regime_2021_stats[reg] = {
            "n_communes": int(len(grp)),
            "imd_mean": mu_imd, "imd_ci": [lo_imd, hi_imd],
            "ies_mean": mu_ies, "ies_ci": [lo_ies, hi_ies],
        }
        log.info("  %-15s n=%6d  IMD mean=%+.3f [%+.3f,%+.3f]  IES mean=%+.3f [%+.3f,%+.3f]",
                 reg, len(grp), mu_imd, lo_imd, hi_imd, mu_ies, lo_ies, hi_ies)

    # ===== Aggregation by 2015 regime =====
    log.info("\n===== Aggregation by 2015 regime =====")
    regime_2015_stats = {}
    for reg, grp in ies.groupby("regime_2015"):
        if reg not in ("droite", "gauche", "regionaliste"):
            continue
        mu_imd, lo_imd, hi_imd = bootstrap_mean_ci(grp["IMD4_national"].values)
        mu_ies, lo_ies, hi_ies = bootstrap_mean_ci(grp["IES"].values)
        regime_2015_stats[reg] = {
            "n_communes": int(len(grp)),
            "imd_mean": mu_imd, "imd_ci": [lo_imd, hi_imd],
            "ies_mean": mu_ies, "ies_ci": [lo_ies, hi_ies],
        }
        log.info("  %-15s n=%6d  IMD mean=%+.3f [%+.3f,%+.3f]  IES mean=%+.3f [%+.3f,%+.3f]",
                 reg, len(grp), mu_imd, lo_imd, hi_imd, mu_ies, lo_ies, hi_ies)

    # ===== Test difference between regimes (Mann-Whitney) =====
    log.info("\n===== Mann-Whitney U test between regimes =====")
    droite_2021 = ies[ies["regime_2021"] == "droite"]["IMD4_national"].values
    gauche_2021 = ies[ies["regime_2021"] == "gauche"]["IMD4_national"].values
    u_stat, p_val = sp_stats.mannwhitneyu(droite_2021, gauche_2021,
                                            alternative="two-sided")
    log.info("  IMD-4 (2021): droite vs gauche  U=%.0f  p=%.3g  "
             "n_droite=%d  n_gauche=%d",
             u_stat, p_val, len(droite_2021), len(gauche_2021))
    droite_ies = ies[ies["regime_2021"] == "droite"]["IES"].values
    gauche_ies = ies[ies["regime_2021"] == "gauche"]["IES"].values
    u_stat2, p_val2 = sp_stats.mannwhitneyu(droite_ies, gauche_ies,
                                              alternative="two-sided")
    log.info("  IES   (2021): droite vs gauche  U=%.0f  p=%.3g",
             u_stat2, p_val2)

    # Effect of La Reunion flip 2021: La Reunion changed regime from
    # droite to gauche between 2015 and 2021. Compare its values.
    reunion = ies[ies["region_code"] == "04"]
    log.info("\nLa Reunion (regime flipped droite -> gauche in 2021):")
    log.info("  n_communes = %d", len(reunion))
    log.info("  IMD-4 mean = %+.3f  IES mean = %+.3f  poverty mean = %.1f%%",
             reunion["IMD4_national"].mean(),
             reunion["IES"].mean(),
             reunion["poverty_rate"].mean())

    # Save
    out = {
        "n_communes_mapped": int(n_mapped),
        "regime_2021": regime_2021_stats,
        "regime_2015": regime_2015_stats,
        "mannwhitney_imd_2021_droite_vs_gauche": {
            "u": float(u_stat), "p": float(p_val),
        },
        "mannwhitney_ies_2021_droite_vs_gauche": {
            "u": float(u_stat2), "p": float(p_val2),
        },
        "regional_means": reg_means.to_dict("records"),
    }
    out_json = OUT_DIR / "b22_regional_political_results.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False,
                                     default=str),
                          encoding="utf-8")
    log.info("\nWrote %s", out_json)

    # ===== Figure: 4-panel =====
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.8))

    # (a) IMD-4 mean per region, colored by 2021 regime
    ax = axes[0, 0]
    color_map = {"droite": "#1F4E79", "gauche": "#C0392B",
                 "regionaliste": "#27AE60"}
    rdf = reg_means.dropna(subset=["regime_2021"]).copy()
    rdf = rdf.sort_values("imd_mean", ascending=True)
    ax.barh(np.arange(len(rdf)), rdf["imd_mean"],
             color=[color_map.get(r, "#888") for r in rdf["regime_2021"]],
             edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(len(rdf)))
    ax.set_yticklabels(rdf["region_name"], fontsize=8)
    ax.set_xlabel("Regional mean IMD-4 (commune average)")
    ax.set_title("(a) IMD-4 by region, coloured by 2021 regime\n"
                 "blue = droite, red = gauche, green = regionaliste",
                 fontsize=9)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    for i, v in enumerate(rdf["imd_mean"]):
        ax.text(v + 0.02, i, f"{v:+.2f}", va="center", fontsize=7)

    # (b) IES mean per region
    ax = axes[0, 1]
    rdf_ies = reg_means.dropna(subset=["regime_2021"]).copy()
    rdf_ies = rdf_ies.sort_values("ies_mean", ascending=True)
    ax.barh(np.arange(len(rdf_ies)), rdf_ies["ies_mean"],
             color=[color_map.get(r, "#888") for r in rdf_ies["regime_2021"]],
             edgecolor="white", linewidth=0.5)
    ax.set_yticks(np.arange(len(rdf_ies)))
    ax.set_yticklabels(rdf_ies["region_name"], fontsize=8)
    ax.set_xlabel("Regional mean IES (IMD residual after income)")
    ax.set_title("(b) IES by region, coloured by 2021 regime",
                 fontsize=9)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.5)
    ax.axvline(0, color="#404040", linewidth=0.5)

    # (c) Box plot: IMD-4 distribution by 2021 regime
    ax = axes[1, 0]
    data21 = [
        ies[ies["regime_2021"] == "droite"]["IMD4_national"].values,
        ies[ies["regime_2021"] == "gauche"]["IMD4_national"].values,
        ies[ies["regime_2021"] == "regionaliste"]["IMD4_national"].values,
    ]
    bp = ax.boxplot(data21, vert=True, patch_artist=True,
                     widths=0.55, showfliers=False,
                     tick_labels=["droite\n(2021)", "gauche\n(2021)",
                                   "regionaliste\n(2021)"])
    for patch, c in zip(bp["boxes"],
                         ["#1F4E79", "#C0392B", "#27AE60"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("IMD-4 (commune-level)")
    ax.set_title(f"(c) IMD-4 by 2021 regime\n"
                 f"Mann-Whitney U droite vs gauche: p = {p_val:.2g}",
                 fontsize=9)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_ylim(-1, 4)

    # (d) Box plot: IES distribution by 2021 regime
    ax = axes[1, 1]
    data21_ies = [
        ies[ies["regime_2021"] == "droite"]["IES"].values,
        ies[ies["regime_2021"] == "gauche"]["IES"].values,
        ies[ies["regime_2021"] == "regionaliste"]["IES"].values,
    ]
    bp2 = ax.boxplot(data21_ies, vert=True, patch_artist=True,
                      widths=0.55, showfliers=False,
                      tick_labels=["droite\n(2021)", "gauche\n(2021)",
                                    "regionaliste\n(2021)"])
    for patch, c in zip(bp2["boxes"],
                         ["#1F4E79", "#C0392B", "#27AE60"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_ylabel("IES (z-score residual)")
    ax.set_title(f"(d) IES by 2021 regime\n"
                 f"Mann-Whitney p = {p_val2:.2g}",
                 fontsize=9)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_ylim(-0.4, 0.4)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "b22_regional_political.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b22_regional_political.pdf")


if __name__ == "__main__":
    main()
