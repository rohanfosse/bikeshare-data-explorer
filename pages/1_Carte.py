"""
1_Carte.py — Carte interactive des stations avec coloration par métrique.
Utilise pydeck (ScatterplotLayer) pour gérer 46 k points en WebGL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, color_scale_rgb, load_stations

st.set_page_config(
    page_title="Carte · Gold Standard GBFS",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Carte interactive des stations")

df = load_stations()

# ── Sidebar — filtres ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtres")

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

    point_size = st.slider("Taille des points", min_value=20, max_value=200, value=60, step=10)

    show_tooltip = st.checkbox("Afficher les infobulles", value=True)

    st.divider()
    meta = METRICS[metric_key]
    st.markdown(f"**{meta['label']}**")
    st.caption(meta["description"])
    if meta["higher_is_better"] is True:
        st.success("↑ Plus élevé = meilleur")
    elif meta["higher_is_better"] is False:
        st.error("↓ Plus bas = meilleur")

# ── Filtrage ──────────────────────────────────────────────────────────────────
dff = df[df["city"].isin(city_sel)] if city_sel else df

n_shown  = len(dff)
n_nodata = int(dff[metric_key].isna().sum()) if metric_key in dff else 0

col_info, col_na = st.columns([4, 1])
col_info.caption(f"**{n_shown:,}** stations affichées")
if n_nodata:
    col_na.caption(f"⚪ {n_nodata:,} sans données (gris)")

# ── Couleurs ──────────────────────────────────────────────────────────────────
palette = meta["color_scale"]
dff = dff.copy()
dff["_color"] = color_scale_rgb(dff[metric_key], palette=palette, alpha=210)

# ── Pydeck layer ──────────────────────────────────────────────────────────────
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
            "backgroundColor": "steelblue",
            "color": "white",
            "fontSize": "13px",
            "padding": "8px 12px",
            "borderRadius": "6px",
        },
    }
    if show_tooltip
    else None
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=dff[["lat", "lon", "_color", "station_name", "city", metric_key, "capacity", "source"]],
    get_position="[lon, lat]",
    get_fill_color="_color",
    get_radius=point_size,
    pickable=show_tooltip,
    auto_highlight=True,
)

# Centre de la vue : centroïde des données affichées
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

# ── Légende colorimétrique ────────────────────────────────────────────────────
valid = dff[metric_key].dropna()
if len(valid) > 0:
    vmin, vmax = float(valid.min()), float(valid.max())
    vmean = float(valid.mean())
    unit = meta["unit"]

    st.caption(
        f"**{meta['label']}** | "
        f"min {vmin:.2f} {unit} · moy {vmean:.2f} {unit} · max {vmax:.2f} {unit}"
        f"  —  palette **{palette}** (gris = données manquantes)"
    )
