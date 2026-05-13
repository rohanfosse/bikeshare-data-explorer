"""B15--B17 -- Stress tests on the INSEE part-velo-travail result.

The B14 tournament adds INSEE part-velo-travail-2022 as a fifth
reference column covering 58/59 cities. The IMD-4 wins this
column with rho = +0.62, bootstrap CI [+0.39, +0.78]. But three
questions need addressing before this result is bulletproof:

  B15  Per-component decomposition.
       Of M, I, T, D, which component(s) carry the rho = +0.62 ?
       If it is mostly D (density), the result is partially
       circular (dense communes mechanically have more cycling
       commuters because trips are shorter).

  B16  Marginal value over Cerema (residualised regression).
       Does the IMD-4 add explanatory power on top of Cerema
       infrastructure density alone ? We fit
         part_velo  =  beta_0 + beta_1 * Cerema + beta_2 * IMD-4-residual
       where IMD-4-residual is the part of IMD-4 not explained
       by Cerema. A significant beta_2 means the IMD captures
       cycling-environment signal that infrastructure alone
       misses.

  B17  Cycling specificity vs walking.
       If rho(IMD-4, walking-share) is comparable to
       rho(IMD-4, cycling-share), then the IMD is not a vello
       indicator -- it is a generic active-mobility indicator.
       The IMD must specifically associate with cycling.

Outputs:
    outputs/b15_decomposition.json
    outputs/b15_decomposition.pdf
    outputs/b16_residual.json
    outputs/b16_residual.pdf
    outputs/b17_specificity.json
"""
from __future__ import annotations

import json
import logging
import sys
import importlib.util
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

B7_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b7_imd4_with_density.py"
spec7 = importlib.util.spec_from_file_location("b7", B7_PATH)
b7 = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(b7)

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
    log.info("Loading panel + IMD-4 components...")
    panel = load_panel()
    dock, cmm4, _, _, _ = b7.build_design(panel)
    cmm3 = cmm4[:, :3]
    res4 = b7.calibrate_k4(cmm4, panel.fub, panel.emp)
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(res4["w_samples"]), 300, replace=False)
    w4 = res4["w_samples"][idx]
    cs4 = dock[["M_norm", "I_norm", "T_norm", "D_norm"]].to_numpy()
    sta4 = (w4 @ cs4.T) * 100.0
    city_codes, city_index = pd.factorize(dock["city"].values)
    n_cities = len(city_index)
    city_imd4 = np.zeros((n_cities, sta4.shape[0]))
    for ci in range(n_cities):
        mask = city_codes == ci
        city_imd4[ci] = sta4[:, mask].mean(axis=1)
    median4 = dict(zip(city_index, np.median(city_imd4, axis=1)))

    # Per-component city-level means
    comp_names = ["M_norm", "I_norm", "T_norm", "D_norm"]
    comp_city = {}
    for cname in comp_names:
        v = dock.groupby("city")[cname].mean().to_dict()
        comp_city[cname] = v

    # References
    insee = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" /
                         "insee_part_velo_travail_2022.csv")
    insee_lookup = dict(zip(insee["city"], insee["insee_part_velo_travail_2022"]))
    cer = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" /
                       "cerema_cycling_infra_city.csv")
    cerema_lookup = dict(zip(cer["city"], cer["infra_cyclable_km_per_km2"]))

    # ===== B15: Per-component decomposition vs INSEE =====
    log.info("\n===== B15: Per-component vs INSEE part-velo-travail =====")
    decomp = {}
    for cname in comp_names:
        pairs = [(comp_city[cname][c], insee_lookup[c])
                 for c in comp_city[cname] if c in insee_lookup
                 and np.isfinite(comp_city[cname][c])
                 and np.isfinite(insee_lookup[c])]
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        rho, q025, q975 = bootstrap_rho(x, y)
        decomp[cname] = {"rho": rho, "q025": q025, "q975": q975, "n": len(pairs)}
        log.info("  %s alone vs INSEE  rho=%+.3f  CI=[%+.3f,%+.3f]  n=%d",
                 cname, rho, q025, q975, len(pairs))

    # IMD-4 composite
    pairs = [(median4[c], insee_lookup[c]) for c in median4
             if c in insee_lookup and np.isfinite(insee_lookup[c])]
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    rho, q025, q975 = bootstrap_rho(x, y)
    decomp["IMD-4 composite"] = {"rho": rho, "q025": q025, "q975": q975,
                                  "n": len(pairs)}
    log.info("  %-15s vs INSEE  rho=%+.3f  CI=[%+.3f,%+.3f]  n=%d",
             "IMD-4 composite", rho, q025, q975, len(pairs))

    out_json = OUT_DIR / "b15_decomposition.json"
    out_json.write_text(json.dumps(decomp, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Decomposition figure
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    names = ["M_norm", "I_norm", "T_norm", "D_norm", "IMD-4 composite"]
    labels = ["M\n(multimodalite)", "I\n(infrastructure)",
              "T\n(topographie)", "D\n(densite)",
              "IMD-4\n(composite)"]
    rhos = [decomp[n]["rho"] for n in names]
    err_lo = [decomp[n]["rho"] - decomp[n]["q025"] for n in names]
    err_hi = [decomp[n]["q975"] - decomp[n]["rho"] for n in names]
    colors = ["#5685B5", "#5685B5", "#5685B5", "#5685B5", "#1F3A6B"]
    ax.bar(np.arange(5), rhos, yerr=[err_lo, err_hi], color=colors,
           capsize=4, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.axhline(0, color="#404040", linewidth=0.5)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman $\\rho$ vs INSEE part-velo-travail 2022", fontsize=9)
    ax.set_title(f"B15: per-component decomposition (n={decomp['M_norm']['n']})",
                 fontsize=10)
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    for i, n in enumerate(names):
        ax.text(i, rhos[i] + 0.04, f"{rhos[i]:+.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b15_decomposition.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote b15_decomposition.pdf")

    # ===== B16: Cerema-residual regression =====
    log.info("\n===== B16: Does IMD-4 add value beyond Cerema ? =====")
    cities_3 = [c for c in median4 if c in insee_lookup and c in cerema_lookup
                 and np.isfinite(insee_lookup[c]) and np.isfinite(cerema_lookup[c])]
    log.info("  cities with IMD+Cerema+INSEE: %d", len(cities_3))
    y_vec = np.array([insee_lookup[c] for c in cities_3])
    imd_vec = np.array([median4[c] for c in cities_3])
    cer_vec = np.array([cerema_lookup[c] for c in cities_3])

    # Z-score each predictor
    def z(v):
        return (v - v.mean()) / v.std(ddof=0)
    imd_z, cer_z, y_z = z(imd_vec), z(cer_vec), z(y_vec)

    # Residualise IMD with Cerema
    beta_cer = (imd_z * cer_z).sum() / (cer_z * cer_z).sum()
    imd_res = imd_z - beta_cer * cer_z

    # Fit y ~ Cerema + IMD_residual
    X = np.column_stack([np.ones(len(cities_3)), cer_z, imd_res])
    beta = np.linalg.lstsq(X, y_z, rcond=None)[0]
    pred = X @ beta
    rss = ((y_z - pred) ** 2).sum()
    tss = ((y_z - y_z.mean()) ** 2).sum()
    r2 = 1 - rss / tss

    # Cerema-only baseline
    Xc = np.column_stack([np.ones(len(cities_3)), cer_z])
    beta_c = np.linalg.lstsq(Xc, y_z, rcond=None)[0]
    pred_c = Xc @ beta_c
    rss_c = ((y_z - pred_c) ** 2).sum()
    r2_c = 1 - rss_c / tss

    # Standard errors via bootstrap
    n_boot = 2000
    beta_boot = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.choice(len(cities_3), len(cities_3), replace=True)
        Xb = X[idx]; yb = y_z[idx]
        try:
            beta_boot[b] = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        except Exception:
            beta_boot[b] = np.nan
    beta_boot = beta_boot[np.isfinite(beta_boot).all(axis=1)]
    ci_beta_imd = (float(np.percentile(beta_boot[:, 2], 2.5)),
                    float(np.percentile(beta_boot[:, 2], 97.5)))
    ci_beta_cer = (float(np.percentile(beta_boot[:, 1], 2.5)),
                    float(np.percentile(beta_boot[:, 1], 97.5)))

    log.info("  Cerema-only:       R^2 = %.3f", r2_c)
    log.info("  Cerema+IMD-resid:  R^2 = %.3f  (delta = %+.3f)",
             r2, r2 - r2_c)
    log.info("  beta_Cerema      = %+.3f  CI=[%+.3f, %+.3f]",
             beta[1], *ci_beta_cer)
    log.info("  beta_IMD-resid   = %+.3f  CI=[%+.3f, %+.3f]",
             beta[2], *ci_beta_imd)
    sig = ci_beta_imd[0] > 0 or ci_beta_imd[1] < 0
    log.info("  IMD-residual significant (CI excludes 0): %s", sig)

    residual_res = {
        "n": int(len(cities_3)),
        "R2_cerema_only": float(r2_c),
        "R2_cerema_plus_imd": float(r2),
        "delta_R2": float(r2 - r2_c),
        "beta_cerema": float(beta[1]),
        "ci_beta_cerema": ci_beta_cer,
        "beta_imd_residual": float(beta[2]),
        "ci_beta_imd_residual": ci_beta_imd,
        "imd_residual_significant": bool(sig),
    }
    (OUT_DIR / "b16_residual.json").write_text(
        json.dumps(residual_res, indent=2), encoding="utf-8")

    # ===== B17: Cycling specificity vs walking =====
    log.info("\n===== B17: IMD specificity, vello vs walking =====")
    mobpro = pd.read_csv(
        ROOT / "data" / "external" / "insee_mobpro" /
        "part-actifs-modes-transport-com.csv",
        dtype={"code_com": str},
        low_memory=False,
    )
    insee_map = pd.read_csv(ROOT / "data" / "external" / "mobility_sources" /
                             "insee_part_velo_travail_2022.csv",
                             dtype={"code_commune": str})
    city_to_insee = dict(zip(insee_map["city"], insee_map["code_commune"]))

    spec = {}
    for mode in ["Marche", "Voiture", "Transports en commun"]:
        sub = mobpro[(mobpro["mode_transport"] == mode) &
                      (mobpro["annee"] == 2022)].copy()
        sub["valeur"] = pd.to_numeric(sub["valeur"], errors="coerce")
        m_lookup = dict(zip(sub["code_com"], sub["valeur"]))
        pairs = []
        for city, code in city_to_insee.items():
            v = m_lookup.get(code)
            if v is None or not np.isfinite(v): continue
            if city not in median4: continue
            pairs.append((float(median4[city]), float(v)))
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        rho, q025, q975 = bootstrap_rho(x, y)
        spec[mode] = {"rho": rho, "q025": q025, "q975": q975, "n": len(pairs)}
        log.info("  rho(IMD-4, %s) = %+.3f  CI=[%+.3f, %+.3f]  n=%d",
                 mode, rho, q025, q975, len(pairs))

    # Compare to Velo
    spec["Velo (B14)"] = {"rho": decomp["IMD-4 composite"]["rho"],
                          "q025": decomp["IMD-4 composite"]["q025"],
                          "q975": decomp["IMD-4 composite"]["q975"],
                          "n": decomp["IMD-4 composite"]["n"]}

    log.info("\nSummary: IMD-4 vs other commute modes (2022):")
    for k, v in spec.items():
        log.info("  %-25s rho=%+.3f CI=[%+.3f, %+.3f]",
                 k, v["rho"], v["q025"], v["q975"])

    (OUT_DIR / "b17_specificity.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
