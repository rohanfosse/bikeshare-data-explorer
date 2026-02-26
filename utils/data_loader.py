"""
data_loader.py — chargement et préparation du Gold Standard GBFS.
Toutes les fonctions sont mises en cache via @st.cache_data (TTL 1h).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Chemin vers le CSV exporté par le notebook 27
_ROOT     = Path(__file__).parent.parent
DATA_PATH = _ROOT / "data" / "stations_gold_standard.parquet"

# ── Métadonnées des métriques ──────────────────────────────────────────────────
METRICS: dict[str, dict] = {
    "infra_cyclable_pct": {
        "label": "Infrastructure cyclable (%)",
        "unit": "%",
        "description": "Part (%) du buffer 300 m autour de la station couverte par des aménagements cyclables OSM.",
        "color_scale": "Greens",
        "higher_is_better": True,
    },
    "infra_cyclable_km": {
        "label": "Infrastructure cyclable (km)",
        "unit": "km",
        "description": "Longueur totale (km) des aménagements cyclables dans un rayon de 300 m.",
        "color_scale": "Greens",
        "higher_is_better": True,
    },
    "baac_accidents_cyclistes": {
        "label": "Accidents cyclistes (300 m, 2021-2023)",
        "unit": "accidents",
        "description": "Nombre d'accidents impliquant un cycliste recensés dans un rayon de 300 m (BAAC 2021-2023).",
        "color_scale": "Reds",
        "higher_is_better": False,
    },
    "gtfs_heavy_stops_300m": {
        "label": "Arrêts TC lourds (300 m)",
        "unit": "arrêts",
        "description": "Nombre d'arrêts de transport en commun lourd (métro, tram, RER, train) dans un rayon de 300 m.",
        "color_scale": "Blues",
        "higher_is_better": True,
    },
    "gtfs_stops_within_300m_pct": {
        "label": "Couverture GTFS (%)",
        "unit": "%",
        "description": "Pourcentage d'arrêts GTFS du réseau local présents dans le buffer 300 m.",
        "color_scale": "Blues",
        "higher_is_better": True,
    },
    "elevation_m": {
        "label": "Altitude (m)",
        "unit": "m",
        "description": "Altitude de la station (SRTM 30 m via Open-Topo-Data). Disponible pour ~90 % des stations après recalcul.",
        "color_scale": "Oranges",
        "higher_is_better": None,  # neutre
    },
    "topography_roughness_index": {
        "label": "Rugosité topographique (TRI)",
        "unit": "m",
        "description": "Écart-type des dénivelés absolus entre la station et ses voisines à ≤ 500 m (proxy dénivelé local).",
        "color_scale": "Purples",
        "higher_is_better": False,
    },
}


@st.cache_data(ttl=3600, show_spinner="Chargement du Gold Standard…")
def load_stations() -> pd.DataFrame:
    """Charge et normalise le CSV Gold Standard."""
    df = pd.read_parquet(DATA_PATH)

    # Nettoyage minimal
    df = df.dropna(subset=["lat", "lon"])
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["n_stations_system"] = pd.to_numeric(df["n_stations_system"], errors="coerce")

    # Source lisible
    df["source_label"] = df["source"].map(
        {"MobilityData": "GBFS (MobilityData)", "Manuel": "GBFS (Manuel)", "OSM": "OSM (M1)"}
    ).fillna(df["source"])

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def city_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques agrégées par ville."""
    agg = {
        "uid": "count",
        "infra_cyclable_pct": "mean",
        "infra_cyclable_km": "mean",
        "baac_accidents_cyclistes": "mean",
        "gtfs_heavy_stops_300m": "mean",
        "gtfs_stops_within_300m_pct": "mean",
        "elevation_m": "mean",
        "topography_roughness_index": "mean",
        "capacity": "mean",
    }
    stats = df.groupby("city").agg(agg).reset_index()
    stats = stats.rename(columns={"uid": "n_stations"})
    return stats.sort_values("n_stations", ascending=False)


@st.cache_data(ttl=3600, show_spinner=False)
def completeness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau de complétude pour chaque métrique enrichie."""
    rows = []
    for col, meta in METRICS.items():
        n_valid = int(df[col].notna().sum()) if col in df.columns else 0
        rows.append(
            {
                "Métrique": meta["label"],
                "Valides": n_valid,
                "Total": len(df),
                "Complétude (%)": round(100 * n_valid / len(df), 1) if len(df) else 0,
            }
        )
    return pd.DataFrame(rows)


def color_scale_rgb(
    series: pd.Series,
    palette: str = "Greens",
    alpha: int = 200,
) -> list[list[int]]:
    """
    Mappe une série numérique vers une liste de couleurs RGBA [R,G,B,A]
    pour pydeck (ScatterplotLayer get_fill_color).
    Gère les NaN (couleur grise).
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap(palette)
    vmin, vmax = series.min(), series.max()
    norm_range = vmax - vmin if vmax != vmin else 1.0

    colors = []
    for v in series:
        if pd.isna(v):
            colors.append([160, 160, 160, 120])
        else:
            norm_v = (v - vmin) / norm_range
            r, g, b, _ = cmap(norm_v)
            colors.append([int(r * 255), int(g * 255), int(b * 255), alpha])
    return colors
