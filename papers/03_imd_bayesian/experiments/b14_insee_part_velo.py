"""B14 -- INSEE recensement: part-velo-travail as a 59-city
behavioural reference.

The eco-counter panel covers ~25 cities and the FUB barometer
~32 cities. A reference axis that covers the FULL 59-city IMD
panel would let us validate the indicator on every city it
ranks. We use the INSEE Recensement variable "Part d'actifs
selon le mode de transport principalement utilise pour aller
travailler" (data.gouv slug:
`part-dactifs-selon-le-mode-de-transport-principalement-utilise-
pour-aller-travailler-2`, source: Tableau de bord des mobilites
durables, last update 2026-04-14). This is a long-format CSV at
the commune level covering all ~35,000 French communes for
2016 and 2022 census waves.

We extract the cycling commute share ("Velo (depuis 2017)") for
year 2022 and map it to our 59-city IMD panel by matching the
panel city name to the largest-population INSEE commune with
the same name (communes_meta cache is already available).

The result is a brand-new behavioural reference covering the
panel completely. We add it as a fifth column to the B10
tournament.

Outputs:
    data/external/mobility_sources/insee_part_velo_travail_2022.csv
    outputs/b14_panel_match.json
    outputs/b14_tournament_with_insee.json
    outputs/b14_tournament_with_insee.pdf
"""
from __future__ import annotations

import json
import logging
import sys
import importlib.util
import unicodedata
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "papers" / "02_imd" / "experiments"))

from _common import load_panel  # noqa: E402
from utils.data_loader import load_stations  # noqa: E402

B7_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b7_imd4_with_density.py"
spec7 = importlib.util.spec_from_file_location("b7", B7_PATH)
b7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(b7)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def normalise(s: str) -> str:
    """Lowercase + strip accents + remove punctuation for matching."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().replace("-", " ").replace("'", " ").replace(".", "").strip()


def build_city_to_insee(panel_cities: list[str]) -> dict[str, str]:
    """Map each panel city to its principal INSEE commune code.

    Heuristic: the commune sharing the city name with the
    largest population. communes_meta has 34,969 rows with
    nom + population + code_commune.
    """
    meta = pd.read_csv(
        ROOT / "data" / "external" / "insee_communes" / "communes_meta.csv",
        dtype={"code_commune": str},
        encoding="utf-8",
    )
    meta["nom_norm"] = meta["nom"].map(normalise)

    # Manual overrides for special cases (Paris arrondissements, etc.)
    overrides = {
        "Paris":             "75056",
        "Lyon":              "69123",
        "Marseille":         "13055",
        "Cergy-Pontoise":    "95127",  # Cergy
        "La Baule":          "44055",  # La Baule-Escoublac
        "Argeles-sur-Mer":   "66008",
        "PAU":               "64445",
        "Pointe-a-Pitre":    "97120",
        "Saint-Etienne":     "42218",
        "Saint-Brieuc":      "22278",
        "Saint-Nazaire":     "44184",
    }
    overrides_norm = {normalise(k): v for k, v in overrides.items()}

    result: dict[str, str] = {}
    unmatched: list[str] = []
    for city in panel_cities:
        cn = normalise(city)
        if cn in overrides_norm:
            result[city] = overrides_norm[cn]
            continue
        cand = meta[meta["nom_norm"] == cn]
        if cand.empty:
            # Loose match
            cand = meta[meta["nom_norm"].str.contains(cn, regex=False)]
        if cand.empty:
            unmatched.append(city)
            continue
        # Pick largest population
        cand = cand.sort_values("population", ascending=False)
        result[city] = cand.iloc[0]["code_commune"]
    log.info("  matched %d/%d panel cities to INSEE communes",
             len(result), len(panel_cities))
    if unmatched:
        log.warning("  unmatched: %s", unmatched)
    return result


def main() -> None:
    log.info("Loading INSEE part-velo-travail at commune level...")
    mobpro = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" / "part-actifs-modes-transport-com.csv",
        dtype={"code_com": str},
        low_memory=False,
    )
    log.info("  %d rows total, %d unique communes, modes: %s",
             len(mobpro),
             mobpro["code_com"].nunique(),
             sorted(mobpro["mode_transport"].dropna().unique().tolist()))

    # Keep only Velo, year 2022
    velo = mobpro[
        (mobpro["mode_transport"].str.contains("V.lo", regex=True, na=False))
        & (mobpro["annee"] == 2022)
    ].copy()
    velo["valeur"] = pd.to_numeric(velo["valeur"], errors="coerce")
    log.info("  velo subset: %d rows", len(velo))
    log.info("  velo share quantiles: 10%%=%.2f  median=%.2f  90%%=%.2f",
             float(velo["valeur"].quantile(0.10)),
             float(velo["valeur"].median()),
             float(velo["valeur"].quantile(0.90)))

    # Build city -> INSEE map for our IMD panel cities
    log.info("\nMatching IMD panel cities to INSEE communes...")
    panel = load_panel()
    city_to_insee = build_city_to_insee(panel.cities)

    # Lookup velo share per city
    velo_by_code = dict(zip(velo["code_com"], velo["valeur"]))
    rows = []
    for city, code in city_to_insee.items():
        v = velo_by_code.get(code)
        if v is None or not np.isfinite(v):
            log.warning("  %s (code %s): no value in INSEE", city, code)
            continue
        rows.append({"city": city, "code_commune": code,
                     "insee_part_velo_travail_2022": float(v)})
    csv = pd.DataFrame(rows)
    out_csv = ROOT / "data" / "external" / "mobility_sources" / \
              "insee_part_velo_travail_2022.csv"
    csv.to_csv(out_csv, index=False, encoding="utf-8")
    log.info("\nWrote %s (%d cities)", out_csv, len(csv))
    log.info("Top 10 cycling-commute cities in our panel:")
    log.info("\n%s", csv.nlargest(10, "insee_part_velo_travail_2022")
             .to_string(index=False))

    # Match results
    match = {"matched": len(csv), "panel_size": len(panel.cities),
             "unmatched": [c for c in panel.cities
                            if c not in csv["city"].values]}
    (OUT_DIR / "b14_panel_match.json").write_text(
        json.dumps(match, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- Tournament with new reference column ----------
    log.info("\nRunning tournament with INSEE as 5th reference...")
    dock, cmm4, _, _, _ = b7.build_design(panel)
    cmm3 = cmm4[:, :3]
    res3 = b7.calibrate_k3(cmm3, panel.fub, panel.emp)
    res4 = b7.calibrate_k4(cmm4, panel.fub, panel.emp)
    rng = np.random.default_rng(2026)
    idx3 = rng.choice(len(res3["w_samples"]), 300, replace=False)
    idx4 = rng.choice(len(res4["w_samples"]), 300, replace=False)
    w3 = res3["w_samples"][idx3]; w4 = res4["w_samples"][idx4]
    cs3 = dock[["M_norm","I_norm","T_norm"]].to_numpy()
    cs4 = dock[["M_norm","I_norm","T_norm","D_norm"]].to_numpy()
    sta3 = (w3 @ cs3.T) * 100.0
    sta4 = (w4 @ cs4.T) * 100.0
    city_codes, city_index = pd.factorize(dock["city"].values)
    n_cities = len(city_index)
    city_imd3 = np.zeros((n_cities, sta3.shape[0]))
    city_imd4 = np.zeros((n_cities, sta4.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_imd3[ci] = sta3[:, mask].mean(axis=1)
        city_imd4[ci] = sta4[:, mask].mean(axis=1)
    median3 = dict(zip(city_index, np.median(city_imd3, axis=1)))
    median4 = dict(zip(city_index, np.median(city_imd4, axis=1)))

    stations = load_stations()
    dock_st = stations[stations["station_type"] == "docked_bike"]
    vol = dock_st.groupby("city").agg(
        n_stations=("station_id", "count"),
        mean_capacity=("capacity", "mean"),
    )
    vol["volumetric"] = np.log(vol["n_stations"] * vol["mean_capacity"].fillna(1))
    volumetric = dict(zip(vol.index, vol["volumetric"]))
    m_only = dock_st.groupby("city")["gtfs_heavy_stops_300m"].mean().to_dict()
    cer = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" /
                       "cerema_cycling_infra_city.csv")
    cerema_dens = dict(zip(cer["city"], cer["infra_cyclable_km_per_km2"]))

    fub_lookup = dict(zip(panel.cities, panel.fub))
    emp_lookup = dict(zip(panel.cities, panel.emp))
    eco = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" /
                       "eco_compteurs_city_usage.csv")
    eco_lookup = dict(zip(eco["city"], eco["eco_avg_daily_bike_counts"]))
    baac = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" /
                        "baac_cyclist_accidents_city.csv")
    baac_col = next((c for c in baac.columns if c != "city"), None)
    baac_lookup = dict(zip(baac["city"], baac[baac_col])) if baac_col else {}
    insee_lookup = dict(zip(csv["city"], csv["insee_part_velo_travail_2022"]))

    metrics = {
        "IMD-4 (Bayesian)":   median4,
        "IMD-3 (Bayesian)":   median3,
        "Volumetric":         volumetric,
        "Cerema km/km^2":     cerema_dens,
        "M alone":            m_only,
    }
    references = {
        "FUB 2023":             fub_lookup,
        "EMP 2019":             emp_lookup,
        "Eco-counter":          eco_lookup,
        "BAAC accidents":       baac_lookup,
        "INSEE velo-travail":   insee_lookup,
    }

    table: dict = {}
    rng_b = np.random.default_rng(2026)
    for m_name, m_dict in metrics.items():
        table[m_name] = {}
        for r_name, r_dict in references.items():
            pairs = []
            for c, mv in m_dict.items():
                rv = r_dict.get(c)
                if rv is None or (isinstance(rv, float) and not np.isfinite(rv)):
                    continue
                if not np.isfinite(mv):
                    continue
                pairs.append((float(mv), float(rv)))
            if len(pairs) < 7:
                table[m_name][r_name] = {"n": len(pairs), "rho": None}
                continue
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            rho = sp_stats.spearmanr(x, y).statistic
            # Bootstrap CI
            n = len(x); boots = np.empty(1000)
            for b in range(1000):
                idx = rng_b.choice(n, n, replace=True)
                r = sp_stats.spearmanr(x[idx], y[idx]).statistic
                boots[b] = r if np.isfinite(r) else np.nan
            boots = boots[np.isfinite(boots)]
            table[m_name][r_name] = {
                "n": len(pairs), "rho": float(rho),
                "q025": float(np.percentile(boots, 2.5)),
                "q975": float(np.percentile(boots, 97.5)),
            }

    log.info("\nTOURNAMENT WITH INSEE (Spearman rho, 95%% bootstrap CI, n)")
    log.info("%-22s | %s", "Metric",
             " | ".join(f"{r:<20s}" for r in references))
    log.info("-" * 160)
    for m in metrics:
        row = [f"{m:<22s}"]
        for r in references:
            cell = table[m][r]
            if cell["rho"] is None:
                row.append(f"{'n<7':>20s}")
            else:
                row.append(
                    f"{cell['rho']:+.2f} [{cell['q025']:+.2f},{cell['q975']:+.2f}]"
                    f" n={cell['n']}"
                )
        log.info(" | ".join(row))

    # Save
    out_json = OUT_DIR / "b14_tournament_with_insee.json"
    out_json.write_text(
        json.dumps({
            "metrics": list(metrics.keys()),
            "references": list(references.keys()),
            "table": table,
        }, indent=2),
        encoding="utf-8",
    )
    log.info("\nWrote %s", out_json)

    # Heatmap figure
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    mns = list(metrics.keys()); rns = list(references.keys())
    mat = np.array([[table[m][r]["rho"] if table[m][r]["rho"] is not None
                      else np.nan for r in rns] for m in mns])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")
    ax.set_xticks(np.arange(len(rns)))
    ax.set_xticklabels(rns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(mns)))
    ax.set_yticklabels(mns, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            cell = table[mns[i]][rns[j]]
            v = cell["rho"]
            if v is None or np.isnan(v):
                ax.text(j, i, "n<7", ha="center", va="center",
                        fontsize=8, color="#404040")
            else:
                ax.text(
                    j, i,
                    f"{v:+.2f}\n[{cell['q025']:+.2f},{cell['q975']:+.2f}]"
                    f"\nn={cell['n']}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.45 else "#202020",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman $\\rho$", fontsize=9)
    ax.set_title("B14: Tournament with INSEE part-velo-travail "
                 "(5th reference, 59-city panel)", fontsize=10, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b14_tournament_with_insee.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b14_tournament_with_insee.pdf")


if __name__ == "__main__":
    main()
