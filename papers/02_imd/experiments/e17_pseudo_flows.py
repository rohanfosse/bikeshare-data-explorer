"""E17 -- Pseudo-flow observation from GBFS status snapshots.

The eco-counter validation of E3 used aggregated daily counter
averages and could only match 25 cities. The GBFS station_status
feed published by every dock-based operator offers a second,
network-internal usage signal: differences in num_bikes_available
between consecutive snapshots reveal station-level pseudo-flows
of departures and arrivals. We collect status snapshots in the
companion repository at a 30-minute cadence and use the resulting
panel to:

  Q-A. Provide a second behavioural validation of the IMD, on a
       different city set than the eco-counter sub-panel.
  Q-B. Identify which IMD component most strongly tracks
       station-level pseudo-flow intensity.
  Q-C. Quantify the unique signal in pseudo-flows beyond what
       the IMD already captures.

The pseudo-flow proxy follows utils/gbfs_collector.compute_pseudo_flows:
between two timestamps t1 < t2, the departures at station s are
estimated as max(0, bikes(t1) - bikes(t2)) and the arrivals as
max(0, bikes(t2) - bikes(t1)). The city-level intensity is the
mean of (departures_est + arrivals_est) per station over the
observed window.

Outputs:
    outputs/e17_results.json
    outputs/e17_pseudo_flow_vs_imd.pdf
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


def _system_to_city() -> dict[str, str]:
    """Map system_id slugs to canonical city names."""
    return {
        "amiens": "Amiens",
        "auxrmlevelo": "Auxerre",
        "beb": "Brest",
        "calais-velos": "Calais",
        "capcotentin": "Cherbourg-en-Cotentin",
        "cergy": "Cergy-Pontoise",
        "citiz_alpes_loire": "Saint-Étienne",
        "citiz_angers": "Angers",
        "citiz_bfc": "Bourgogne-Franche-Comté",
        "citiz_grand_est": "Strasbourg",
        "citiz_grand_poitiers": "Poitiers",
        "citiz_la_rochelle": "La Rochelle",
        "citiz_occitanie": "Toulouse",
        "citiz_rennes_metropole": "Rennes",
        "CVelo_FR_Clermont-Ferrand": "Clermont-Ferrand",
        "donkey_brest": "Brest",
        "donkey_valenciennes": "Valenciennes",
        "idecycle_pau": "Pau",
        "inurba-rouen": "Rouen",
        "levelo_inurba_marseille": "Marseille",
        "le_velo_star": "Rennes",
        "libelo": "Libourne",
        "lyon": "Lyon",
        "montpellier": "Montpellier",
        "nancy": "Nancy",
        "Optymo_Belfort_ALS": "Belfort",
        "Paris": "Paris",
        "tan": "Nantes",
        "tbm": "Bordeaux",
        "toulouse_vc": "Toulouse",
        "vcub": "Bordeaux",
        "velomagg": "Montpellier",
        "velhop": "Strasbourg",
        "viavelo_nimes": "Nimes",
        "vlille": "Lille",
    }


def _compute_pseudo_flow_intensity(parquets: list[Path]) -> dict | None:
    """Compute station-level pseudo-flow normalised by station capacity.

    Reads all snapshots, computes |delta_bikes| between consecutive
    snapshots per station, divides by the rolling capacity proxy
    (bikes + docks available, which equals dock capacity when both
    fields are reported correctly), and aggregates to a city-level
    average turnover rate.
    """
    if len(parquets) < 2:
        return None
    dfs = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(dfs, ignore_index=True)
    df["capacity_proxy"] = df["num_bikes_available"] + df["num_docks_available"]
    df = df.sort_values(["station_id", "fetched_at"])
    df["bikes_prev"] = df.groupby("station_id")["num_bikes_available"].shift(1)
    df["cap_prev"] = df.groupby("station_id")["capacity_proxy"].shift(1)
    df = df.dropna(subset=["bikes_prev"])
    if df.empty:
        return None
    df["abs_delta"] = (df["num_bikes_available"] - df["bikes_prev"]).abs()
    df["capacity"] = df[["capacity_proxy", "cap_prev"]].max(axis=1).clip(lower=1)
    df["turnover_rate"] = df["abs_delta"] / df["capacity"]
    station_rate = df.groupby("station_id")["turnover_rate"].sum()
    station_abs = df.groupby("station_id")["abs_delta"].sum()
    station_cap = df.groupby("station_id")["capacity"].median()
    return {
        "n_stations": int(len(station_rate)),
        "mean_flow_per_station": float(station_abs.mean()),
        "mean_turnover_rate": float(station_rate.mean()),
        "p75_turnover_rate": float(station_rate.quantile(0.75)),
        "mean_capacity": float(station_cap.mean()),
    }


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n=%d cities in IMD panel", panel.n)

    snap_root = ROOT / "data" / "status_snapshots"
    system_to_city = _system_to_city()

    rows = []
    for system_dir in sorted(snap_root.iterdir()):
        if not system_dir.is_dir():
            continue
        parquets = sorted(system_dir.glob("2026-*.parquet"))
        if len(parquets) < 2:
            continue
        city = system_to_city.get(system_dir.name)
        if city is None:
            log.info("  skip %s (no city mapping)", system_dir.name)
            continue
        try:
            stats = _compute_pseudo_flow_intensity(parquets)
        except Exception as exc:
            log.warning("  failed on %s: %s", system_dir.name, exc)
            continue
        if stats is None:
            continue
        stats["system_id"] = system_dir.name
        stats["city"] = city
        rows.append(stats)
    flow_df = pd.DataFrame(rows)
    log.info("Computed pseudo-flows for %d systems / %d distinct cities",
             len(flow_df), flow_df["city"].nunique())

    # Pick the largest network per city
    flow_df = (
        flow_df.sort_values("n_stations", ascending=False)
        .drop_duplicates("city")
        .reset_index(drop=True)
    )

    # Merge with IMD panel
    panel_df = pd.DataFrame({
        "city": panel.cities,
        "IMD": panel.imd,
        **{c: panel.components[:, i] for i, c in enumerate(COMPONENTS)},
    })
    merged = flow_df.merge(panel_df, on="city", how="inner")
    log.info("Matched %d cities to IMD panel", len(merged))

    # Two usage proxies: raw |delta| per station, and turnover rate
    # (which divides by capacity to correct for system maturity).
    corr = {"flow_raw": {}, "turnover_rate": {}}
    for usage_col, label_y in [
        ("mean_flow_per_station", "flow_raw"),
        ("mean_turnover_rate", "turnover_rate"),
    ]:
        log.info("Spearman against %s:", label_y)
        for col, label in [("IMD", "IMD"),
                           *[(c, c) for c in COMPONENTS]]:
            x = merged[col].to_numpy(dtype=float)
            y = merged[usage_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 5:
                rho = float(sp_stats.spearmanr(x[mask], y[mask]).statistic)
                p = float(sp_stats.spearmanr(x[mask], y[mask]).pvalue)
            else:
                rho, p = float("nan"), float("nan")
            corr[label_y][label] = {
                "rho": rho, "p": p, "n": int(mask.sum()),
            }
            log.info("  Spearman(%s, %-12s) = %+.3f  p=%.3f  n=%d",
                     label_y, label,
                     rho if np.isfinite(rho) else 0,
                     p if np.isfinite(p) else 1, int(mask.sum()))

    # Partial regression: turnover_rate ~ IMD + log(n_stations)
    use = (
        np.isfinite(merged["IMD"])
        & np.isfinite(merged["mean_turnover_rate"])
        & np.isfinite(merged["n_stations"])
    )
    sub = merged.loc[use].copy()
    if len(sub) >= 8:
        x = np.column_stack([
            np.ones(len(sub)),
            (sub["IMD"] - sub["IMD"].mean()) / sub["IMD"].std(ddof=1),
            np.log1p(sub["n_stations"]),
        ])
        y = sub["mean_turnover_rate"].to_numpy(dtype=float)
        coefs, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ coefs
        rss_full = float(((y - pred) ** 2).sum())
        tss = float(((y - y.mean()) ** 2).sum())
        r2_full = 1.0 - rss_full / max(tss, 1e-12)
        x_no_imd = x[:, [0, 2]]
        coefs_no, *_ = np.linalg.lstsq(x_no_imd, y, rcond=None)
        pred_no = x_no_imd @ coefs_no
        rss_no = float(((y - pred_no) ** 2).sum())
        partial_r2_imd = (rss_no - rss_full) / max(rss_no, 1e-12)
        log.info("Partial regression (turnover_rate ~ IMD + log(N))")
        log.info("  R^2_full = %.3f, partial R^2 of IMD = %.3f",
                 r2_full, partial_r2_imd)
    else:
        r2_full, partial_r2_imd = float("nan"), float("nan")
        log.info("  not enough cities for partial regression")

    results = {
        "n_systems": int(len(flow_df)),
        "n_cities_matched": int(len(merged)),
        "spearman_per_predictor": corr,
        "r2_full_imd_plus_logN_turnover": r2_full,
        "partial_r2_imd_net_size_turnover": partial_r2_imd,
        "matched_cities": merged[[
            "city", "IMD", "n_stations",
            "mean_flow_per_station", "mean_turnover_rate",
        ]].to_dict("records"),
    }
    out_json = OUT_DIR / "e17_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Two-panel figure: raw flow (left, sensitive to system maturity)
    # and capacity-normalised turnover rate (right, the proper proxy).
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, ycol, ylabel, label_key, panel_title in zip(
        axes,
        ["mean_flow_per_station", "mean_turnover_rate"],
        [r"Raw $|\Delta$bikes$|$ per station",
         r"Turnover rate $|\Delta$bikes$|$ / capacity"],
        ["flow_raw", "turnover_rate"],
        ["Raw pseudo-flow (capacity-confounded)",
         "Capacity-normalised turnover rate"],
    ):
        x = merged["IMD"].to_numpy(dtype=float)
        y = merged[ycol].to_numpy(dtype=float)
        sizes = 20 + 60 * (merged["n_stations"] / merged["n_stations"].max())
        ax.scatter(x, y, s=sizes, color="#1F3A6B",
                   alpha=0.75, edgecolor="white", linewidth=0.5)
        top5 = merged.nlargest(5, "IMD")
        for _, row in top5.iterrows():
            ax.annotate(row["city"], (row["IMD"], row[ycol]),
                        fontsize=8, color="#404040",
                        xytext=(4, 4), textcoords="offset points")
        rho = corr[label_key]["IMD"]["rho"]
        n = corr[label_key]["IMD"]["n"]
        ax.text(0.02, 0.98,
                f"$\\rho_{{Sp}}$ = {rho:+.2f},  $n$ = {n}",
                transform=ax.transAxes, fontsize=9, color="#202020",
                ha="left", va="top",
                bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                      "alpha": 0.9, "pad": 4})
        ax.set_xlabel("IMD")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, fontsize=10)
        ax.grid(True, color="#E5E5E5", linewidth=0.5)
    fig.suptitle("GBFS-derived usage proxies against the IMD",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e17_pseudo_flow_vs_imd.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e17_pseudo_flow_vs_imd.pdf")


if __name__ == "__main__":
    main()
