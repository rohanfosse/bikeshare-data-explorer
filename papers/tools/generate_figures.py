"""Generate PDF figures for the BikeShare-ICT research papers.

Outputs land in:
  papers/01_gold_standard/figures/
  papers/02_imd/figures/

All figures are vector PDFs sized for the academic page (single-column),
in a sober blue/grey palette consistent with the Streamlit dashboard.

Run from the repository root:

    python papers/tools/generate_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [_HERE, *_HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)
sys.path.insert(0, str(ROOT))

from utils.data_loader import compute_imd_cities, load_stations  # noqa: E402

# -------------------------------------------------------------------------
# Output paths and shared style
# -------------------------------------------------------------------------

GOLD_DIR = ROOT / "papers" / "01_gold_standard" / "figures"
IMD_DIR = ROOT / "papers" / "02_imd" / "figures"
GOLD_DIR.mkdir(parents=True, exist_ok=True)
IMD_DIR.mkdir(parents=True, exist_ok=True)

# Sober academic palette: deep navy as single accent, grey scale for the
# rest. No red/orange/green/purple. Designed to read well in greyscale
# print and to align with SCITEPRESS-style conference papers.
NAVY = "#1F3A6B"
DARK_GREY = "#404040"
MID_GREY = "#7A7A7A"
LIGHT_GREY = "#BFBFBF"
PALE_GREY = "#E5E5E5"
HIGHLIGHT = "#8C8C8C"  # used only when a second tone is unavoidable

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": DARK_GREY,
        "axes.linewidth": 0.6,
        "axes.labelcolor": DARK_GREY,
        "xtick.color": DARK_GREY,
        "ytick.color": DARK_GREY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": PALE_GREY,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# -------------------------------------------------------------------------
# Gold Standard figures
# -------------------------------------------------------------------------

def fig01_audit_status(catalog: pd.DataFrame) -> None:
    """Donut chart of system audit statuses."""
    labels = {
        "ok": "Certified (Gold Standard)",
        "too_small": "Excluded (micro-network)",
        "no_si_url": "Excluded (no SI URL)",
        "fetch_error": "Excluded (fetch error)",
        "autopartage": "Excluded (car-sharing, A1)",
        "dom_tom": "Out of perimeter (A5)",
    }
    colors = {
        "Certified (Gold Standard)": NAVY,
        "Excluded (micro-network)": MID_GREY,
        "Excluded (no SI URL)": LIGHT_GREY,
        "Excluded (fetch error)": DARK_GREY,
        "Excluded (car-sharing, A1)": HIGHLIGHT,
        "Out of perimeter (A5)": LIGHT_GREY,
    }
    counts = catalog["status"].value_counts()
    series = pd.Series({labels.get(k, k): v for k, v in counts.items()})

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    wedges, _texts = ax.pie(
        series.values,
        labels=None,
        colors=[colors.get(k, LIGHT_GREY) for k in series.index],
        wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 1.2},
        startangle=90,
        counterclock=False,
    )
    ax.text(
        0,
        0,
        f"{int(series.sum())}\nsystems",
        ha="center",
        va="center",
        fontsize=10,
        color=DARK_GREY,
    )
    ax.set_aspect("equal")
    ax.grid(False)
    legend_labels = [
        f"{name} (n={count})" for name, count in series.items()
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=8,
    )
    _save(fig, GOLD_DIR / "fig01_audit_status.pdf")


def fig02_capacity_violin(stations: pd.DataFrame) -> None:
    """Capacity distribution by station_type, illustrating A3 bias."""
    label_map = {
        "docked_bike": "Dock-based (VLS)",
        "free_floating": "Free-floating (A3)",
        "carsharing": "Car-sharing (A1)",
    }
    palette = {
        "Dock-based (VLS)": NAVY,
        "Free-floating (A3)": MID_GREY,
        "Car-sharing (A1)": LIGHT_GREY,
    }
    df = stations[stations["capacity"].notna() & (stations["capacity"] > 0)].copy()
    df["category"] = df["station_type"].map(label_map)
    df = df[df["category"].notna()]
    df["capacity_clipped"] = df["capacity"].clip(upper=80)

    categories = ["Dock-based (VLS)", "Free-floating (A3)", "Car-sharing (A1)"]
    data = [df.loc[df["category"] == c, "capacity_clipped"].values for c in categories]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    parts = ax.violinplot(
        data,
        positions=range(len(categories)),
        widths=0.85,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, cat in zip(parts["bodies"], categories):
        body.set_facecolor(palette[cat])
        body.set_alpha(0.45)
        body.set_edgecolor(DARK_GREY)
        body.set_linewidth(0.6)
    parts["cmedians"].set_color(DARK_GREY)
    parts["cmedians"].set_linewidth(1.0)

    for i, cat in enumerate(categories):
        vals = df.loc[df["category"] == cat, "capacity"]
        median = vals.median()
        ax.annotate(
            f"median = {median:.0f}",
            xy=(i, median),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
            color=DARK_GREY,
        )

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Declared capacity (docks, clipped at 80)")
    ax.set_xlabel("")
    ax.set_ylim(-2, 85)
    ax.grid(True, axis="y")
    _save(fig, GOLD_DIR / "fig02_capacity_violin.pdf")


def fig03_region_stations(catalog: pd.DataFrame) -> None:
    """Stations by region (certified systems only)."""
    ok = catalog[catalog["status"] == "ok"].copy()
    agg = (
        ok.groupby("region")["n_stations"]
        .agg(stations="sum", systems="count")
        .reset_index()
        .sort_values("stations", ascending=True)
    )
    agg = agg[agg["region"].notna()]

    fig, ax = plt.subplots(figsize=(5.4, max(3.0, len(agg) * 0.28)))
    bars = ax.barh(
        agg["region"],
        agg["stations"],
        color=NAVY,
        edgecolor="white",
        linewidth=0.4,
    )
    max_val = agg["stations"].max()
    for bar, n_sys in zip(bars, agg["systems"]):
        w = bar.get_width()
        ax.text(
            w + max_val * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,} ({n_sys} sys.)",
            va="center",
            fontsize=7.5,
            color=DARK_GREY,
        )
    ax.set_xlabel("Certified stations")
    ax.set_ylabel("")
    ax.set_xlim(0, max_val * 1.22)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    _save(fig, GOLD_DIR / "fig03_region_stations.pdf")


def fig04_bordeaux_before_after(catalog: pd.DataFrame, cities_dock: pd.DataFrame) -> None:
    """Top cities by raw vs certified station counts (illustrates A3 impact)."""
    raw = (
        catalog.groupby("city")["n_stations"].sum().reset_index()
        .rename(columns={"n_stations": "raw"})
    )
    cert = cities_dock[["city", "n_stations"]].rename(columns={"n_stations": "certified"})
    merged = raw.merge(cert, on="city", how="inner")
    merged = merged[merged["raw"] > 0].copy()
    merged["reduction_pct"] = (1 - merged["certified"] / merged["raw"]) * 100
    top = merged.sort_values("reduction_pct", ascending=False).head(12)
    top = top.sort_values("reduction_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(5.6, max(3.0, len(top) * 0.34)))
    y = np.arange(len(top))
    h = 0.4
    ax.barh(
        y + h / 2, top["raw"], height=h,
        color=MID_GREY, label="Raw GBFS",
        edgecolor="white", linewidth=0.4,
    )
    ax.barh(
        y - h / 2, top["certified"], height=h,
        color=NAVY, label="Gold Standard",
        edgecolor="white", linewidth=0.4,
    )
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["raw"] + top["raw"].max() * 0.01,
            i + h / 2,
            f"{int(row['raw']):,}",
            va="center",
            fontsize=7.5,
            color=DARK_GREY,
        )
        ax.text(
            row["certified"] + top["raw"].max() * 0.01,
            i - h / 2,
            f"{int(row['certified']):,}",
            va="center",
            fontsize=7.5,
            color=DARK_GREY,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(top["city"])
    ax.set_xlabel("Stations")
    ax.set_xlim(0, top["raw"].max() * 1.18)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    _save(fig, GOLD_DIR / "fig04_bordeaux_before_after.pdf")


def fig05_completeness(stations: pd.DataFrame) -> None:
    """Empirical completeness per enriched variable (dock-based subset)."""
    dock = stations[stations["station_type"] == "docked_bike"]
    items = [
        ("Distance to nearest transit stop", "gtfs_heavy_stops_300m"),
        ("Cycling linear 300 m", "infra_cyclable_km"),
        ("BAAC accident density 500 m", "baac_accidents_cyclistes"),
        ("Average slope (BD ALTI)", "elevation_m"),
        ("Median income per CU (Filosofi)", "revenu_median_uc"),
        ("FUB perception score", "part_velo_travail"),
        ("Local Gini index", "gini_revenu"),
    ]
    rows = []
    for label, col in items:
        if col in dock.columns:
            pct = dock[col].notna().mean() * 100
        else:
            pct = np.nan
        rows.append((label, pct))
    out = pd.DataFrame(rows, columns=["variable", "completeness"])
    out = out.sort_values("completeness", ascending=True)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bars = ax.barh(
        out["variable"], out["completeness"],
        color=NAVY, edgecolor="white", linewidth=0.4,
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            fontsize=7.5,
            color=DARK_GREY,
        )
    ax.set_xlim(0, 105)
    ax.set_xlabel("Completeness (%)")
    ax.axvline(95, color=MID_GREY, linewidth=0.6, linestyle="--", alpha=0.6)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    _save(fig, GOLD_DIR / "fig05_completeness.pdf")


# -------------------------------------------------------------------------
# IMD figures
# -------------------------------------------------------------------------

def fig01_imd_weights() -> None:
    """Calibrated weights of the IMD composite index."""
    weights = {
        "Multimodality (M)": 0.578,
        "Infrastructure (I)": 0.184,
        "Safety (S)": 0.142,
        "Topography (T)": 0.096,
    }
    s = pd.Series(weights).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bars = ax.barh(
        s.index, s.values,
        color=NAVY, edgecolor="white", linewidth=0.4,
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            fontsize=8,
            color=DARK_GREY,
        )
    ax.set_xlim(0, max(s.values) * 1.15)
    ax.set_xlabel("Calibrated weight (sum = 1)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    _save(fig, IMD_DIR / "fig01_weights.pdf")


def fig02_volume_vs_imd(imd_cities: pd.DataFrame) -> None:
    """Raw station volume vs IMD score, with city labels for outliers."""
    df = imd_cities.dropna(subset=["IMD", "n_stations"]).copy()
    df = df[df["n_stations"] > 0]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(
        df["n_stations"],
        df["IMD"],
        s=34,
        color=NAVY,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Dock-based stations (log scale)")
    ax.set_ylabel("IMD (0--100)")
    ax.grid(True, axis="both")

    highlight_top_imd = df.nlargest(6, "IMD")
    highlight_large = df.nlargest(4, "n_stations")
    labelled = pd.concat([highlight_top_imd, highlight_large]).drop_duplicates(subset="city")
    for _, row in labelled.iterrows():
        ax.annotate(
            row["city"],
            (row["n_stations"], row["IMD"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7.5,
            color=DARK_GREY,
        )

    median = df["IMD"].median()
    ax.axhline(median, color=MID_GREY, linewidth=0.6, linestyle="--", alpha=0.6)
    ax.text(
        df["n_stations"].max() * 0.6,
        median + 1.5,
        f"national median IMD = {median:.1f}",
        fontsize=7,
        color=MID_GREY,
    )
    _save(fig, IMD_DIR / "fig02_volume_vs_imd.pdf")


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    stations = load_stations()
    catalog = pd.read_csv(ROOT / "data" / "gbfs_france" / "systems_catalog.csv", encoding="utf-8")
    imd_cities = compute_imd_cities(stations)
    cities_dock = (
        stations[stations["station_type"] == "docked_bike"]
        .groupby("city")
        .agg(n_stations=("uid", "count"))
        .reset_index()
    )

    print("Gold Standard figures:")
    fig01_audit_status(catalog)
    fig02_capacity_violin(stations)
    fig03_region_stations(catalog)
    fig04_bordeaux_before_after(catalog, cities_dock)
    fig05_completeness(stations)

    print("IMD figures:")
    fig01_imd_weights()
    fig02_volume_vs_imd(imd_cities)

    print("Done.")


if __name__ == "__main__":
    main()
