"""
1_Carte.py — Carte interactive des stations avec coloration par métrique.
Utilise pydeck (ScatterplotLayer) pour gérer 46 k points en WebGL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, color_scale_rgb, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Carte des stations — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Carte des stations GBFS françaises")
st.caption("Gold Standard GBFS · CESI BikeShare-ICT 2025-2026")

abstract_box(
    "Cette page propose une visualisation géospatiale des 46 000+ stations de vélos "
    "en libre-service (VLS) françaises issues du Gold Standard GBFS. "
    "Chaque point est coloré selon la métrique d'enrichissement sélectionnée, "
    "calculée dans un rayon standard de 300 m autour du point de stationnement. "
    "Le rendu utilise pydeck (ScatterplotLayer, WebGL) pour garantir des performances "
    "fluides sur l'ensemble du corpus national."
)

df = load_stations()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Filtres et options")

    all_cities = sorted(df["city"].unique())
    city_sel = st.multiselect(
        "Ville(s)",
        options=all_cities,
        default=[],
        placeholder="Toutes les villes",
    )

    metric_key = st.selectbox(
        "Métrique à afficher",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k]["label"],
        index=0,
    )

    point_size = st.slider(
        "Rayon des points (m)", min_value=20, max_value=200, value=60, step=10
    )
    show_tooltip = st.checkbox("Afficher les infobulles", value=True)

    st.divider()
    meta = METRICS[metric_key]
    st.markdown(f"**{meta['label']}**")
    st.caption(meta["description"])
    if meta["higher_is_better"] is True:
        st.info("Valeur élevée = favorable")
    elif meta["higher_is_better"] is False:
        st.warning("Valeur faible = favorable")

# ── Filtrage ──────────────────────────────────────────────────────────────────
dff = df[df["city"].isin(city_sel)] if city_sel else df

n_shown  = len(dff)
n_nodata = int(dff[metric_key].isna().sum()) if metric_key in dff else 0

# ── Section 1 — Contexte ─────────────────────────────────────────────────────
section(1, "Couverture géographique — 46 000+ stations VLS sur le territoire national")

col_info, col_na = st.columns([4, 1])
col_info.caption(
    f"**{n_shown:,}** stations affichées · "
    f"{dff['city'].nunique()} villes · "
    f"{dff['system_id'].nunique()} réseaux GBFS"
)
if n_nodata:
    col_na.caption(f"{n_nodata:,} sans données (gris)")

# ── Section 2 — Carte ────────────────────────────────────────────────────────
section(2, "Carte interactive — coloration par métrique d'enrichissement (rayon 300 m)")

palette = meta["color_scale"]
dff = dff.copy()
dff["_color"] = color_scale_rgb(dff[metric_key], palette=palette, alpha=210)

tooltip_html = (
    {
        "html": (
            "<b>{station_name}</b><br/>"
            "Ville : {city}<br/>"
            f"{meta['label']} : {{{metric_key}}}<br/>"
            "Capacité : {capacity}<br/>"
            "Source : {source}"
        ),
        "style": {
            "backgroundColor": "#1A2332",
            "color": "white",
            "fontSize": "13px",
            "padding": "8px 12px",
            "borderRadius": "4px",
        },
    }
    if show_tooltip
    else None
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=dff[["lat", "lon", "_color", "station_name", "city",
              metric_key, "capacity", "source"]],
    get_position="[lon, lat]",
    get_fill_color="_color",
    get_radius=point_size,
    pickable=show_tooltip,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=float(dff["lat"].mean()),
    longitude=float(dff["lon"].mean()),
    zoom=5 if not city_sel else 11,
    pitch=0,
)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip_html,
    map_style="light",
)

st.pydeck_chart(r, use_container_width=True, height=580)

# ── Légende ───────────────────────────────────────────────────────────────────
valid = dff[metric_key].dropna()
if len(valid) > 0:
    vmin, vmax = float(valid.min()), float(valid.max())
    vmean = float(valid.mean())
    unit = meta["unit"]
    st.caption(
        f"Figure 2.1. Carte des stations colorées selon **{meta['label']}**. "
        f"Min {vmin:.2f} {unit}  ·  Moy {vmean:.2f} {unit}  ·  Max {vmax:.2f} {unit}. "
        f"Palette *{palette}*. Points gris = données manquantes."
    )

# ── Section 3 — Statistiques de la sélection ─────────────────────────────────
st.divider()
section(3, "Statistiques descriptives de la sélection — min, moy, médiane, max")

if len(valid) > 0:
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric(f"Stations ({meta['label']})", f"{len(valid):,}")
    s2.metric("Minimum", f"{vmin:.2f} {unit}")
    s3.metric("Moyenne", f"{vmean:.2f} {unit}")
    s4.metric("Médiane", f"{float(valid.median()):.2f} {unit}")
    s5.metric("Maximum", f"{vmax:.2f} {unit}")
else:
    st.info("Aucune donnée disponible pour la métrique sélectionnée.")
