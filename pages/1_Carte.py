"""
1_Carte.py — Visualisation géospatiale du corpus Gold Standard GBFS.
Utilise pydeck (ScatterplotLayer, WebGL) pour rendre 46 000+ points en temps réel.
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
    page_title="Cartographie Spatiale — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Cartographie Spatiale du Corpus Gold Standard")
st.caption("Axe de Recherche 2 : Distribution Territoriale et Disparités Géographiques de l'Offre Cyclable")

abstract_box(
    "<b>Question de recherche :</b> Les disparités spatiales de l'offre cyclable partagée "
    "résultent-elles d'une fatalité géographique ou d'inégalités de gouvernance locale ?<br><br>"
    "Cette interface permet la visualisation géospatiale du corpus Gold Standard GBFS : "
    "46 312 stations de vélos en libre-service issues de 122 systèmes nationaux, enrichies "
    "selon cinq modules spatiaux dans un rayon normalisé de 300 m. "
    "Le calcul global de l'autocorrélation spatiale (indice de Moran, $I = -0{,}023$, $p = 0{,}765$) "
    "invalide l'hypothèse d'un déterminisme géographique : les stations performantes et "
    "sous-performantes ne forment pas de clusters territoriaux significatifs. "
    "Cette absence de structure spatiale oriente l'interprétation des disparités observées "
    "vers des facteurs de <em>gouvernance locale</em> et de <em>choix politiques d'aménagement</em>."
)

df = load_stations()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres de Visualisation")

    all_cities = sorted(df["city"].unique())
    city_sel = st.multiselect(
        "Agglomération(s)",
        options=all_cities,
        default=[],
        placeholder="Corpus national complet",
    )

    metric_key = st.selectbox(
        "Dimension d'enrichissement à cartographier",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k]["label"],
        index=0,
    )

    point_size = st.slider(
        "Rayon de représentation (m)", min_value=20, max_value=200, value=60, step=10
    )
    show_tooltip = st.checkbox("Activer les infobulles", value=True)

    st.divider()
    meta = METRICS[metric_key]
    st.markdown(f"**{meta['label']}**")
    st.caption(meta["description"])
    if meta["higher_is_better"] is True:
        st.info("Indicateur direct : valeur élevée = environnement favorable.")
    elif meta["higher_is_better"] is False:
        st.warning("Indicateur inverse : valeur faible = environnement favorable.")

# ── Filtrage ──────────────────────────────────────────────────────────────────
dff = df[df["city"].isin(city_sel)] if city_sel else df

n_shown  = len(dff)
n_nodata = int(dff[metric_key].isna().sum()) if metric_key in dff else 0

# ── Section 1 — Couverture ────────────────────────────────────────────────────
section(1, "Couverture Territoriale — Distribution Nationale des 46 312 Stations")

col_info, col_na = st.columns([4, 1])
col_info.markdown(
    f"**{n_shown:,}** stations affichées · "
    f"**{dff['city'].nunique()}** agglomérations · "
    f"**{dff['system_id'].nunique()}** réseaux GBFS certifiés"
)
col_info.caption(
    "La couverture nationale couvre l'ensemble des agglomérations françaises disposant "
    "d'un système VLS actif au standard GBFS. Les points grisés indiquent des stations "
    "pour lesquelles la métrique sélectionnée est manquante (données non disponibles dans "
    "le buffer de 300 m ou hors périmètre de la source primaire)."
)
if n_nodata:
    col_na.metric("Sans données", f"{n_nodata:,}", delta=None)

# ── Section 2 — Carte ─────────────────────────────────────────────────────────
section(2, "Carte Interactive — Coloration par Dimension d'Enrichissement Spatial (Rayon 300 m)")

palette = meta["color_scale"]
dff = dff.copy()
dff["_color"] = color_scale_rgb(dff[metric_key], palette=palette, alpha=210)

tooltip_html = (
    {
        "html": (
            "<b>{station_name}</b><br/>"
            "Agglomération : {city}<br/>"
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
        f"**Figure 2.1.** Distribution cartographique de la dimension **{meta['label']}** "
        f"sur le corpus Gold Standard. "
        f"Intervalle observé : [{vmin:.2f} — {vmax:.2f}] {unit} · "
        f"Moyenne nationale : {vmean:.2f} {unit}. "
        f"Palette chromatique : *{palette}*. "
        f"Points gris : valeur manquante (absence de données dans le rayon de 300 m)."
    )

# ── Section 3 — Statistiques descriptives ─────────────────────────────────────
st.divider()
section(3, "Statistiques Descriptives de la Sélection — Caractérisation Univariée")

st.markdown(r"""
L'absence d'autocorrélation spatiale significative (Moran's $I = -0{,}023$, $p = 0{,}765$)
implique que la variance inter-stations de la métrique visualisée relève davantage de
déterminants locaux (politique d'aménagement, topographie de quartier) que d'un gradient
territorial macroscopique. Les statistiques ci-dessous caractérisent la distribution empirique
de la sélection courante.
""")

if len(valid) > 0:
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric(f"Stations valides", f"{len(valid):,}")
    s2.metric("Minimum", f"{vmin:.2f} {unit}")
    s3.metric("Moyenne", f"{vmean:.2f} {unit}")
    s4.metric("Médiane", f"{float(valid.median()):.2f} {unit}")
    s5.metric("Maximum", f"{vmax:.2f} {unit}")
    st.caption(
        f"**Tableau 3.1.** Statistiques univariées de **{meta['label']}** "
        f"sur la sélection courante ({len(valid):,} stations valides / {n_shown:,} totales). "
        f"Écart-type : {float(valid.std()):.3f} {unit} · "
        f"Q25 : {float(valid.quantile(0.25)):.3f} · Q75 : {float(valid.quantile(0.75)):.3f} {unit}."
    )
else:
    st.info(
        "Aucune valeur valide disponible pour la dimension sélectionnée sur la sélection courante. "
        "Vérifiez le périmètre géographique ou sélectionnez une autre métrique d'enrichissement."
    )
