"""
data_loader.py — chargement et préparation du Gold Standard GBFS.
Toutes les fonctions sont mises en cache via @st.cache_data (TTL 1h).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Chemins des données
_ROOT     = Path(__file__).parent.parent
DATA_PATH = _ROOT / "data" / "stations_gold_standard_final.parquet"

# Catalogue des systèmes GBFS
SYSTEMS_PATH   = _ROOT / "data" / "gbfs_france" / "systems_catalog.csv"

# Indicateurs de mobilité à l'échelle des villes (sources externes)
FUB_PATH        = _ROOT / "data" / "external" / "mobility_sources" / "fub_barometre_2023_city_scores.csv"
EMP_PATH        = _ROOT / "data" / "external" / "mobility_sources" / "emp_2019_city_modal_share.csv"
BAAC_CITY_PATH  = _ROOT / "data" / "external" / "mobility_sources" / "baac_cyclist_accidents_city.csv"
CEREMA_PATH     = _ROOT / "data" / "external" / "mobility_sources" / "cerema_cycling_infra_city.csv"
ECO_PATH        = _ROOT / "data" / "external" / "mobility_sources" / "eco_compteurs_city_usage.csv"

# Données Montpellier Vélomagg (analyse de réseau)
SOCIO_MMM_PATH    = _ROOT / "data" / "processed" / "socioeconomic_analysis_results.csv"
TEMPORAL_PATH     = _ROOT / "data" / "processed" / "station_temporal_profiles.csv"
STRESS_PATH       = _ROOT / "data" / "processed" / "station_stress_ranking.csv"
BIKETRAM_PATH     = _ROOT / "data" / "processed" / "multimodal" / "bike_tram_proximity_matrix.csv"
HOURLY_PATH       = _ROOT / "data" / "processed" / "flow_analysis" / "hourly_flow_statistics.csv"
NETFLOW_PATH      = _ROOT / "data" / "processed" / "flow_analysis" / "net_flow_analysis.csv"
SYNTHESE_PATH     = _ROOT / "data" / "processed" / "ville_montpellier" / "analyses" / "synthese_velo_socio.csv"
TOP_QUART_PATH    = _ROOT / "data" / "processed" / "ville_montpellier" / "analyses" / "top_10_quartiers_velo.csv"
BOT_QUART_PATH    = _ROOT / "data" / "processed" / "ville_montpellier" / "analyses" / "bottom_10_quartiers_velo.csv"
COMMUNITY_PATH    = _ROOT / "data" / "processed" / "community_detection_results.csv"
NETWORK_TOPO_PATH = _ROOT / "data" / "processed" / "network_topology_results.csv"
VULNERABILITY_PATH = _ROOT / "data" / "processed" / "station_vulnerability_ranking.csv"
WEATHER_PATH      = _ROOT / "data" / "processed" / "weather_data_enriched.csv"
MODAL_PATH        = _ROOT / "data" / "processed" / "ville_montpellier" / "analyses" / "parts_modales_moyennes.csv"

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
    # ── Nouvelles colonnes socio-économiques INSEE (Gold Standard Final) ──────────
    "revenu_median_uc": {
        "label": "Revenu médian/UC (€/an, INSEE)",
        "unit": "€/an",
        "description": "Revenu médian par unité de consommation du carreau INSEE 200 m contenant la station (Filosofi).",
        "color_scale": "Greens",
        "higher_is_better": None,
    },
    "gini_revenu": {
        "label": "Indice de Gini (inégalités de revenu)",
        "unit": "",
        "description": "Coefficient de Gini mesurant les inégalités de revenu dans le carreau INSEE 200 m.",
        "color_scale": "Reds",
        "higher_is_better": False,
    },
    "part_menages_voit0": {
        "label": "Ménages sans voiture (%)",
        "unit": "%",
        "description": "Part des ménages sans voiture dans le carreau INSEE 200 m (RP 2020).",
        "color_scale": "Blues",
        "higher_is_better": None,
    },
    "part_velo_travail": {
        "label": "Part vélo domicile-travail (%)",
        "unit": "%",
        "description": "Part modale vélo pour les trajets domicile-travail dans le carreau INSEE 200 m (RP 2020).",
        "color_scale": "Greens",
        "higher_is_better": True,
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

    # Normalisation des noms de villes (doublons et variantes connues)
    _city_norm = {
        "Strasbourg, FR": "Strasbourg",
        "Nice, FR":       "Nice",
        "PAU":            "Pau",
    }
    df["city"] = df["city"].replace(_city_norm)

    # Source lisible
    df["source_label"] = df["source"].map(
        {"MobilityData": "GBFS (MobilityData)", "Manuel": "GBFS (Manuel)", "OSM": "OSM (M1)"}
    ).fillna(df["source"])

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def city_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques agrégées par ville."""
    agg: dict[str, str] = {
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
    # Colonnes socio-économiques optionnelles (Gold Standard Final — INSEE Filosofi)
    _socio: dict[str, str] = {
        "revenu_median_uc":  "median",   # médiane des médianes de carreau
        "gini_revenu":       "mean",
        "part_menages_voit0": "mean",
        "part_velo_travail": "mean",
    }
    for col, func in _socio.items():
        if col in df.columns:
            agg[col] = func
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


@st.cache_data(ttl=3600, show_spinner=False)
def compute_imd_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'Indice de Mobilité Douce (IMD) à l'échelle des villes.

    Composantes (pondération égale, 25 % chacune) :
    - S : Sécurité      = 1 − norm(baac_accidents_cyclistes)
    - I : Infrastructure = norm(infra_cyclable_pct)
    - M : Multimodalité  = norm(gtfs_heavy_stops_300m)
    - T : Topographie    = 1 − norm(topography_roughness_index)

    Normalisation min-max sur les villes avec au moins 5 stations.
    La médiane est utilisée pour imputer les valeurs manquantes.
    IMD ∈ [0, 100].

    Référence méthodologique : notebooks 21-25, CESI BikeShare-ICT 2025-2026.
    """
    stats = city_stats(df).query("n_stations >= 5").copy().reset_index(drop=True)

    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    def _fill(col: str) -> pd.Series:
        s = stats[col].copy() if col in stats.columns else pd.Series(np.nan, index=stats.index)
        return s.fillna(s.median())

    stats["S_securite"] = (1 - _minmax(_fill("baac_accidents_cyclistes"))).values
    stats["I_infra"]    = _minmax(_fill("infra_cyclable_pct")).values
    stats["M_multi"]    = _minmax(_fill("gtfs_heavy_stops_300m")).values
    stats["T_topo"]     = (1 - _minmax(_fill("topography_roughness_index"))).values

    comp_cols = ["S_securite", "I_infra", "M_multi", "T_topo"]
    stats["IMD"] = stats[comp_cols].mean(axis=1) * 100

    return stats.sort_values("IMD", ascending=False).reset_index(drop=True)


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


# ── Données nationales ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_systems_catalog() -> pd.DataFrame:
    """Catalogue des 122 systèmes GBFS français (notebook 20)."""
    df = pd.read_csv(SYSTEMS_PATH)
    df["n_stations"] = pd.to_numeric(df["n_stations"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_city_mobility() -> pd.DataFrame:
    """
    Fusion des indicateurs de mobilité à l'échelle des villes :
    FUB Baromètre 2023, EMP 2019, BAAC, Cerema, Eco-compteurs.
    """
    paths = [FUB_PATH, EMP_PATH, BAAC_CITY_PATH, CEREMA_PATH, ECO_PATH]
    merged: pd.DataFrame | None = None
    for path in paths:
        try:
            df = pd.read_csv(path)
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="city", how="outer")
        except Exception:
            pass
    return merged if merged is not None else pd.DataFrame()


# ── Données Montpellier Vélomagg ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_montpellier_stations() -> pd.DataFrame:
    """Profils des stations Vélomagg : centralité, trips, quartier."""
    df = pd.read_csv(SOCIO_MMM_PATH)
    for col in ("latitude", "longitude", "Total_Trips", "PageRank", "Betweenness"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["latitude", "longitude"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_station_temporal_profiles() -> pd.DataFrame:
    """Profils temporels des stations (commuter index, trips semaine/week-end)."""
    df = pd.read_csv(TEMPORAL_PATH)
    for col in ("Total_Trips", "Weekday_Departures", "Weekend_Departures", "Weekend_Ratio"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_station_stress() -> pd.DataFrame:
    """Classement des stations par indice de stress (demande vs capacité)."""
    return pd.read_csv(STRESS_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_hourly_flows() -> pd.DataFrame:
    """Statistiques agrégées des flux par heure (0-23h)."""
    return pd.read_csv(HOURLY_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_net_flows() -> pd.DataFrame:
    """Flux nets (entrées − sorties) par station et par heure."""
    return pd.read_csv(NETFLOW_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_bike_tram_proximity() -> pd.DataFrame:
    """Matrice de proximité stations vélo — arrêts tram (distance en m)."""
    df = pd.read_csv(BIKETRAM_PATH)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_synthese_velo_socio() -> pd.DataFrame:
    """Synthèse mobilité vélo × indicateurs socio-économiques par quartier."""
    return pd.read_csv(SYNTHESE_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_top_quartiers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top 10 et bottom 10 quartiers par usage vélo."""
    top = pd.read_csv(TOP_QUART_PATH)
    try:
        bot = pd.read_csv(BOT_QUART_PATH)
    except Exception:
        bot = pd.DataFrame()
    return top, bot


@st.cache_data(ttl=3600, show_spinner=False)
def load_community_detection() -> pd.DataFrame:
    """Détection de communautés Louvain + métriques de pontage (bridge stations)."""
    return pd.read_csv(COMMUNITY_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_network_topology() -> pd.DataFrame:
    """Topologie du graphe de flux : degrés, clustering, points d'articulation."""
    return pd.read_csv(NETWORK_TOPO_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_station_vulnerability() -> pd.DataFrame:
    """Indice de vulnérabilité structurelle par station (betweenness, population, clustering)."""
    return pd.read_csv(VULNERABILITY_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def load_weather_data() -> pd.DataFrame:
    """Données météorologiques horaires enrichies (Montpellier, 2021-2024)."""
    df = pd.read_csv(WEATHER_PATH)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_parts_modales() -> pd.DataFrame:
    """Parts modales moyennes sur l'ensemble des quartiers de Montpellier."""
    return pd.read_csv(MODAL_PATH)
