"""B21 -- National IES (Indice d'Equite Sociale) on 34,858 communes.

With the national IMD-4 in hand (B20) and the commune-level
median income and poverty rate published by the same
"Tableau de bord des mobilites durables" source, we can build
the first national cycling-environment equity diagnostic
ever computed at commune granularity in France.

Three questions guide the experiment:

  Q1  Is the IMD-4 systematically correlated with income across
      French communes ?  An indicator that just tracks income
      is not adding equity information.

  Q2  Stratified by city size, is the equity gap larger in
      big cities or rural communes ?  Within-city vs between-
      city decomposition.

  Q3  Can we identify "double-penalty" communes -- low IMD-4
      AND high poverty -- as concrete Plan Velo investment
      targets ?

The IES is defined per commune as the residual of IMD-4 after
projecting onto income.  A positive IES means the commune
has *better cycling environment than predicted by its income*
(equitable); a negative IES means worse cycling environment
than expected (inequitable).

Outputs:
    data/external/mobility_sources/ies_national_communes.csv
    outputs/b21_ies_results.json
    outputs/b21_ies_panel.pdf
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


def bootstrap_rho(x, y, n_boot=500, seed=2026):
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
    log.info("Loading national IMD-4 (from B20)...")
    imd = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources" /
        "imd4_national_communes.csv",
        dtype={"code_commune": str},
    )
    log.info("  %d communes", len(imd))

    log.info("Loading commune income (niveau-de-vie-median)...")
    inc = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" /
        "mediane-niveau-vie-com.csv",
        dtype={"code_com": str},
    )
    inc["valeur"] = pd.to_numeric(inc["valeur"], errors="coerce")
    inc = inc.rename(columns={"valeur": "income_median"})
    log.info("  %d commune rows, %d with income value",
             len(inc), inc["income_median"].notna().sum())

    log.info("Loading commune poverty rate...")
    pov = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" /
        "taux-pauvrete-com.csv",
        dtype={"code_com": str},
    )
    pov["valeur"] = pd.to_numeric(pov["valeur"], errors="coerce")
    pov = pov.rename(columns={"valeur": "poverty_rate"})
    log.info("  %d rows, %d with poverty value",
             len(pov), pov["poverty_rate"].notna().sum())

    # Merge
    df = imd.merge(inc[["code_com", "income_median"]],
                    left_on="code_commune", right_on="code_com", how="left")
    df = df.drop(columns=["code_com"])
    df = df.merge(pov[["code_com", "poverty_rate"]],
                    left_on="code_commune", right_on="code_com", how="left")
    df = df.drop(columns=["code_com"])

    n_with_inc = df["income_median"].notna().sum()
    n_with_pov = df["poverty_rate"].notna().sum()
    log.info("\nMerged: %d total, %d with IMD+income, %d with IMD+poverty",
             len(df), n_with_inc, n_with_pov)

    # ===== Q1: IMD vs income at national scale =====
    log.info("\n===== Q1: IMD-4 vs income / poverty (national) =====")
    inc_panel = df[np.isfinite(df["IMD4_national"]) &
                    np.isfinite(df["income_median"])].copy()
    inc_panel["log_income"] = np.log(inc_panel["income_median"].clip(lower=1))
    rho_inc, q025_inc, q975_inc = bootstrap_rho(
        inc_panel["IMD4_national"].values,
        inc_panel["log_income"].values,
        n_boot=500,
    )
    log.info("  rho(IMD-4, log_income) = %+.3f   CI = [%+.3f, %+.3f]   n = %d",
             rho_inc, q025_inc, q975_inc, len(inc_panel))

    pov_panel = df[np.isfinite(df["IMD4_national"]) &
                    np.isfinite(df["poverty_rate"])].copy()
    rho_pov, q025_pov, q975_pov = bootstrap_rho(
        pov_panel["IMD4_national"].values,
        pov_panel["poverty_rate"].values,
        n_boot=500,
    )
    log.info("  rho(IMD-4, poverty_rate) = %+.3f   CI = [%+.3f, %+.3f]   n = %d",
             rho_pov, q025_pov, q975_pov, len(pov_panel))

    # ===== Q2: stratified by commune size =====
    log.info("\n===== Q2: stratified by commune population =====")
    strat: dict = {}
    for thr in [0, 1000, 5000, 10000, 20000, 50000, 100000]:
        sub = inc_panel[inc_panel["population"] >= thr]
        if len(sub) < 5:
            continue
        rho_s, q025_s, q975_s = bootstrap_rho(
            sub["IMD4_national"].values, sub["log_income"].values,
            n_boot=300,
        )
        strat[f"pop_ge_{thr}"] = {
            "n": int(len(sub)),
            "rho_income": float(rho_s),
            "ci_income": [float(q025_s), float(q975_s)],
        }
        log.info("  pop >= %6d  n=%6d  rho=%+.3f  CI=[%+.3f, %+.3f]",
                 thr, len(sub), rho_s, q025_s, q975_s)

    # ===== Compute IES as residual =====
    # Model: IMD_z ~ alpha + beta * log_income_z + epsilon ; IES = epsilon
    log.info("\n===== Computing IES (residual of IMD-4 after income) =====")
    pan = inc_panel.dropna(subset=["IMD4_national", "log_income"]).copy()
    imd_z = (pan["IMD4_national"] - pan["IMD4_national"].mean()) / \
            pan["IMD4_national"].std(ddof=0)
    inc_z = (pan["log_income"] - pan["log_income"].mean()) / \
            pan["log_income"].std(ddof=0)
    beta = float((imd_z * inc_z).sum() / (inc_z * inc_z).sum())
    pan["IES"] = imd_z - beta * inc_z
    log.info("  beta (slope IMD_z on income_z) = %.3f", beta)
    log.info("  IES quantiles:  q10=%.2f  median=%.2f  q90=%.2f",
             float(pan["IES"].quantile(0.10)),
             float(pan["IES"].median()),
             float(pan["IES"].quantile(0.90)))

    # Most equitable and inequitable communes (with pop > 5000 for relevance)
    rel = pan[pan["population"] > 5000]
    eq = rel.nlargest(15, "IES")
    ineq = rel.nsmallest(15, "IES")
    log.info("\nTop 15 most CYCLING-EQUITABLE communes (IES highest)\n"
             "(better cycling environment than predicted by their income):")
    log.info("\n%s", eq[["nom", "code_commune", "population",
                          "IMD4_national", "income_median", "IES"]]
              .to_string(index=False))
    log.info("\nTop 15 most CYCLING-INEQUITABLE communes (IES lowest)\n"
             "(worse cycling environment than predicted by their income):")
    log.info("\n%s", ineq[["nom", "code_commune", "population",
                           "IMD4_national", "income_median", "IES"]]
              .to_string(index=False))

    # ===== Q3: double-penalty communes =====
    log.info("\n===== Q3: double-penalty communes "
             "(low IMD + high poverty) =====")
    # We need both IMD and poverty
    full = df[np.isfinite(df["IMD4_national"]) &
               np.isfinite(df["poverty_rate"])].copy()
    imd_q33 = full["IMD4_national"].quantile(0.33)
    pov_q66 = full["poverty_rate"].quantile(0.66)
    deserts = full[(full["IMD4_national"] < imd_q33) &
                    (full["poverty_rate"] > pov_q66)]
    log.info("  IMD-4 33rd percentile = %.3f", imd_q33)
    log.info("  poverty 66th percentile = %.2f%%", pov_q66)
    log.info("  %d 'cycling-poverty deserts' identified", len(deserts))
    log.info("  total panel: %d communes, deserts = %.1f%%",
             len(full), 100.0 * len(deserts) / len(full))

    # Show top 15 desert communes by population (most impactful)
    log.info("\nTop 15 desert communes by population:")
    top_deserts = deserts.nlargest(15, "population")
    log.info("\n%s", top_deserts[["nom", "code_commune", "population",
                                    "IMD4_national", "income_median",
                                    "poverty_rate"]].to_string(index=False))

    # ===== Save =====
    # Save the per-commune IES table
    out_csv = ROOT / "data" / "external" / "mobility_sources" / \
              "ies_national_communes.csv"
    keep = pan[["code_commune", "nom", "population", "IMD4_national",
                 "income_median", "IES"]].copy()
    keep = keep.merge(df[["code_commune", "poverty_rate"]],
                       on="code_commune", how="left")
    keep.to_csv(out_csv, index=False, encoding="utf-8")
    log.info("\nWrote %s (%d communes with IES)", out_csv, len(keep))

    out_json = OUT_DIR / "b21_ies_results.json"
    results = {
        "n_with_imd_income": int(len(inc_panel)),
        "n_with_imd_poverty": int(len(pov_panel)),
        "rho_imd_income": float(rho_inc),
        "ci_imd_income": [q025_inc, q975_inc],
        "rho_imd_poverty": float(rho_pov),
        "ci_imd_poverty": [q025_pov, q975_pov],
        "beta_income_slope": beta,
        "n_double_penalty_deserts": int(len(deserts)),
        "pct_double_penalty_deserts": float(100.0 * len(deserts) / len(full)),
        "imd_threshold_q33": float(imd_q33),
        "poverty_threshold_q66": float(pov_q66),
        "stratified_by_pop": strat,
        "top_15_equitable": eq[["nom", "code_commune", "population",
                                  "IES"]].to_dict("records"),
        "top_15_inequitable": ineq[["nom", "code_commune", "population",
                                       "IES"]].to_dict("records"),
        "top_15_deserts_by_pop": top_deserts[["nom", "code_commune",
                                                "population",
                                                "IMD4_national",
                                                "income_median",
                                                "poverty_rate"]]
                                          .to_dict("records"),
    }
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    log.info("Wrote %s", out_json)

    # ===== Figure: 4-panel IES analysis =====
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.8))

    # (a) IMD vs income hexbin
    ax = axes[0, 0]
    hb = ax.hexbin(inc_panel["log_income"], inc_panel["IMD4_national"],
                    gridsize=60, cmap="Blues", mincnt=1, bins="log")
    ax.set_xlabel(r"$\log$(median income per UC, EUR)")
    ax.set_ylabel("National IMD-4")
    ax.set_title(f"(a) IMD-4 vs income, n={len(inc_panel):,}, "
                 r"$\rho = " f"{rho_inc:+.3f}$",
                 fontsize=10)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.04)
    cbar.set_label("log(count)", fontsize=8)

    # (b) IMD vs poverty
    ax = axes[0, 1]
    hb2 = ax.hexbin(pov_panel["poverty_rate"], pov_panel["IMD4_national"],
                     gridsize=60, cmap="Reds", mincnt=1, bins="log")
    ax.set_xlabel(r"Poverty rate (\%)")
    ax.set_ylabel("National IMD-4")
    ax.set_title(f"(b) IMD-4 vs poverty, n={len(pov_panel):,}, "
                 r"$\rho = " f"{rho_pov:+.3f}$",
                 fontsize=10)
    cbar2 = fig.colorbar(hb2, ax=ax, fraction=0.04)
    cbar2.set_label("log(count)", fontsize=8)

    # (c) Stratified rho(IMD, income) by commune size
    ax = axes[1, 0]
    thr_list = [0, 1000, 5000, 10000, 20000, 50000, 100000]
    labels = ["all", "$\\geq$1k", "$\\geq$5k", "$\\geq$10k",
              "$\\geq$20k", "$\\geq$50k", "$\\geq$100k"]
    rhos = [strat[f"pop_ge_{t}"]["rho_income"] for t in thr_list]
    err_lo = [strat[f"pop_ge_{t}"]["rho_income"] -
              strat[f"pop_ge_{t}"]["ci_income"][0] for t in thr_list]
    err_hi = [strat[f"pop_ge_{t}"]["ci_income"][1] -
              strat[f"pop_ge_{t}"]["rho_income"] for t in thr_list]
    ax.bar(np.arange(len(thr_list)), rhos, yerr=[err_lo, err_hi],
           color="#5685B5", capsize=4, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xticks(np.arange(len(thr_list)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Communes by population threshold")
    ax.set_ylabel(r"$\rho$(IMD-4, log income)")
    ax.set_title("(c) Stratified equity-correlation by city size",
                 fontsize=10)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    for i in range(len(thr_list)):
        ax.text(i, rhos[i] + err_hi[i] + 0.01, f"{rhos[i]:+.2f}",
                ha="center", va="bottom", fontsize=8)

    # (d) IES distribution: histogram + named extremes
    ax = axes[1, 1]
    ax.hist(pan["IES"], bins=80, color="#5685B5",
            edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="#404040", linewidth=0.8)
    ax.set_xlabel("IES = IMD-4 residual after income (z-score)")
    ax.set_ylabel("Count of communes")
    # Mark some named cities
    targets = ["Strasbourg", "Grenoble", "Bordeaux", "Marseille",
                "Nice", "Paris", "Toulouse", "Montpellier"]
    for name in targets:
        rows = rel[rel["nom"] == name]
        if not rows.empty:
            val = float(rows["IES"].iloc[0])
            ax.axvline(val, color="#C0392B", linewidth=0.5, alpha=0.7)
            ax.text(val, ax.get_ylim()[1] * 0.85, name,
                     fontsize=7, rotation=90, ha="right", color="#C0392B",
                     va="top")
    ax.set_title("(d) IES distribution and named cities",
                 fontsize=10)
    ax.grid(True, color="#E5E5E5", linewidth=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "b21_ies_panel.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b21_ies_panel.pdf")


if __name__ == "__main__":
    main()
