"""E23 -- Concurrent validity against Cerema and BAAC city aggregates.

The IMD's component matrix relies on station-level OSM and BAAC
records aggregated within a 300 m buffer. Two complementary
city-level sources are independently available and were not used in
the IMD construction:

    - Cerema cycling-infrastructure inventory (city-level kilometres
      of dedicated cycle facilities, both raw and per-km² of the
      city's footprint);
    - BAAC accident database aggregated per city + normalised per
      100 000 inhabitants (a rate, not the station-buffer count).

We test concurrent validity by comparing the IMD and its components
against these city-level aggregates. A good composite indicator
should:

    - correlate strongly with Cerema's per-km² infrastructure density
      (the construct most closely aligned with C_I);
    - correlate with the BAAC per-100 k rate after adjusting for the
      exposure confounding documented in E3-E4;
    - not be a direct re-expression of either aggregate (otherwise
      the IMD adds no information).

Outputs:
    outputs/e23_results.json
    outputs/e23_concurrent_validity.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from _common import COMPONENTS, ROOT, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Loading panel and external aggregates...")
    panel = load_panel()
    base = ROOT / "data" / "external" / "mobility_sources"
    cerema = pd.read_csv(base / "cerema_cycling_infra_city.csv")
    baac_rate = pd.read_csv(base / "baac_cyclist_accidents_city.csv")

    df = pd.DataFrame({
        "city": panel.cities,
        "IMD": panel.imd,
        **{c: panel.components[:, i] for i, c in enumerate(COMPONENTS)},
    })
    df = df.merge(cerema, on="city", how="left")
    df = df.merge(baac_rate, on="city", how="left")

    log.info("Coverage of external aggregates on the panel:")
    for col in ["infra_cyclable_km", "infra_cyclable_km_per_km2",
                "baac_accidents_cyclistes_per_100k"]:
        log.info("  %-40s  n = %d", col, int(df[col].notna().sum()))

    # ---- IMD vs Cerema infrastructure ----
    cerema_results = {}
    for col, label in [
        ("infra_cyclable_km", "Cerema km (raw)"),
        ("infra_cyclable_km_per_km2", "Cerema km / km² (density)"),
    ]:
        for target_name, target in [("IMD", "IMD"),
                                     *[(c, c) for c in COMPONENTS]]:
            mask = df[col].notna() & df[target].notna()
            if mask.sum() < 5:
                continue
            rho, p = sp_stats.spearmanr(df.loc[mask, col], df.loc[mask, target])
            cerema_results[f"{col}__{target}"] = {
                "label": label,
                "predictor": target_name,
                "n": int(mask.sum()),
                "rho": float(rho),
                "p": float(p),
            }
    log.info("IMD vs Cerema correlations:")
    for k, v in cerema_results.items():
        if v["predictor"] in ("IMD", "I_infra"):
            log.info("  %-35s  rho = %+.3f  p = %.3f  n = %d",
                     f"{v['label']} <-> {v['predictor']}",
                     v["rho"], v["p"], v["n"])

    # ---- IMD vs BAAC per-100k rate ----
    baac_results = {}
    for target_name, target in [("IMD", "IMD"),
                                 *[(c, c) for c in COMPONENTS]]:
        mask = (df["baac_accidents_cyclistes_per_100k"].notna()
                & df[target].notna())
        if mask.sum() < 5:
            continue
        rho, p = sp_stats.spearmanr(
            df.loc[mask, "baac_accidents_cyclistes_per_100k"],
            df.loc[mask, target],
        )
        baac_results[target_name] = {
            "n": int(mask.sum()),
            "rho": float(rho),
            "p": float(p),
        }
    log.info("BAAC per-100k vs IMD correlations:")
    for k, v in baac_results.items():
        log.info("  %-15s  rho = %+.3f  p = %.3f  n = %d",
                 k, v["rho"], v["p"], v["n"])

    # ---- Discriminative test: does IMD add information beyond Cerema? ----
    # Regress observed eco-counter (E3 outcome) on IMD vs on Cerema density.
    eco = pd.read_csv(base / "eco_compteurs_city_usage.csv")
    df = df.merge(eco, on="city", how="left")
    mask = (df["eco_avg_daily_bike_counts"].notna()
            & df["infra_cyclable_km_per_km2"].notna()
            & df["IMD"].notna())
    sub = df.loc[mask].copy()
    log.info("Triangulation panel (IMD + Cerema + eco-counter): n = %d",
             int(len(sub)))
    if len(sub) >= 10:
        y = np.log1p(sub["eco_avg_daily_bike_counts"].to_numpy())
        x_imd = sub["IMD"].to_numpy()
        x_cer = sub["infra_cyclable_km_per_km2"].to_numpy()
        x_size = np.log1p(sub["infra_cyclable_km"].to_numpy())
        # Standardise
        x_imd_s = (x_imd - x_imd.mean()) / x_imd.std(ddof=1)
        x_cer_s = (x_cer - x_cer.mean()) / x_cer.std(ddof=1)
        x_size_s = (x_size - x_size.mean()) / x_size.std(ddof=1)
        # Full model: log(eco) ~ IMD + Cerema_density + log(km_raw)
        X = np.column_stack([np.ones(len(sub)), x_imd_s, x_cer_s, x_size_s])
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coefs
        rss_full = float(((y - pred) ** 2).sum())
        tss = float(((y - y.mean()) ** 2).sum())
        r2_full = 1.0 - rss_full / max(tss, 1e-12)

        def _partial_r2(drop_col_idx):
            X_no = np.delete(X, drop_col_idx, axis=1)
            coefs_no, *_ = np.linalg.lstsq(X_no, y, rcond=None)
            pred_no = X_no @ coefs_no
            rss_no = float(((y - pred_no) ** 2).sum())
            return (rss_no - rss_full) / max(rss_no, 1e-12)

        partial_imd = _partial_r2(1)
        partial_cer = _partial_r2(2)
        partial_size = _partial_r2(3)

        triangulation = {
            "r2_full": r2_full,
            "beta_imd": float(coefs[1]),
            "beta_cerema_density": float(coefs[2]),
            "beta_log_km": float(coefs[3]),
            "partial_r2_imd": partial_imd,
            "partial_r2_cerema_density": partial_cer,
            "partial_r2_log_km": partial_size,
            "n_cities": int(len(sub)),
        }
        log.info("Triangulation R^2 = %.3f", r2_full)
        log.info("  partial R^2 of IMD net Cerema      = %.3f", partial_imd)
        log.info("  partial R^2 of Cerema density net IMD = %.3f", partial_cer)
        log.info("  partial R^2 of log(km) net all        = %.3f", partial_size)
    else:
        triangulation = {"n_cities": int(len(sub)), "note": "panel too small"}

    results = {
        "cerema_correlations": cerema_results,
        "baac_per100k_correlations": baac_results,
        "triangulation": triangulation,
        "coverage": {
            "cerema_km": int(df["infra_cyclable_km"].notna().sum()),
            "cerema_density": int(df["infra_cyclable_km_per_km2"].notna().sum()),
            "baac_per_100k": int(df["baac_accidents_cyclistes_per_100k"].notna().sum()),
        },
    }
    out_json = OUT_DIR / "e23_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # ---- Figure: 2-panel scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ax = axes[0]
    mask = df["infra_cyclable_km_per_km2"].notna() & df["IMD"].notna()
    x = df.loc[mask, "infra_cyclable_km_per_km2"]
    y = df.loc[mask, "IMD"]
    ax.scatter(x, y, s=34, color="#1F3A6B",
               alpha=0.75, edgecolor="white", linewidth=0.5)
    top5 = df.loc[mask].nlargest(5, "IMD")
    for _, row in top5.iterrows():
        ax.annotate(row["city"],
                    (row["infra_cyclable_km_per_km2"], row["IMD"]),
                    fontsize=8, color="#404040",
                    xytext=(4, 4), textcoords="offset points")
    rho_c = cerema_results.get(
        "infra_cyclable_km_per_km2__IMD", {}).get("rho", float("nan"))
    n_c = cerema_results.get(
        "infra_cyclable_km_per_km2__IMD", {}).get("n", 0)
    ax.text(0.02, 0.98,
            f"$\\rho_{{Sp}}$ = {rho_c:+.2f},  $n$ = {n_c}",
            transform=ax.transAxes, fontsize=9, color="#202020",
            ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                  "alpha": 0.9, "pad": 4})
    ax.set_xlabel(r"Cerema cycle network density (km / km$^{2}$)")
    ax.set_ylabel("IMD")
    ax.set_title("IMD vs.\\ Cerema cycle-network density",
                 fontsize=10)
    ax.grid(True, color="#E5E5E5", linewidth=0.5)

    ax = axes[1]
    mask = (df["baac_accidents_cyclistes_per_100k"].notna()
            & df["IMD"].notna())
    x = df.loc[mask, "baac_accidents_cyclistes_per_100k"]
    y = df.loc[mask, "IMD"]
    ax.scatter(x, y, s=34, color="#1F3A6B",
               alpha=0.75, edgecolor="white", linewidth=0.5)
    top5 = df.loc[mask].nlargest(5, "IMD")
    for _, row in top5.iterrows():
        ax.annotate(row["city"],
                    (row["baac_accidents_cyclistes_per_100k"], row["IMD"]),
                    fontsize=8, color="#404040",
                    xytext=(4, 4), textcoords="offset points")
    rho_b = baac_results.get("IMD", {}).get("rho", float("nan"))
    n_b = baac_results.get("IMD", {}).get("n", 0)
    ax.text(0.02, 0.98,
            f"$\\rho_{{Sp}}$ = {rho_b:+.2f},  $n$ = {n_b}",
            transform=ax.transAxes, fontsize=9, color="#202020",
            ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                  "alpha": 0.9, "pad": 4})
    ax.set_xlabel(r"BAAC cyclist crashes per 100 000 inhabitants")
    ax.set_ylabel("IMD")
    ax.set_title("IMD vs.\\ BAAC cyclist-crash rate", fontsize=10)
    ax.grid(True, color="#E5E5E5", linewidth=0.5)

    fig.suptitle("Concurrent validity of the IMD against Cerema and BAAC "
                 "external aggregates", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e23_concurrent_validity.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e23_concurrent_validity.pdf")


if __name__ == "__main__":
    main()
