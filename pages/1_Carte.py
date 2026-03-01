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

# ── Chargement anticipé (abstract dynamique) ──────────────────────────────────
df = load_stations()
_n_total   = len(df)
_n_dock    = int((df["station_type"] == "docked_bike").sum()) if "station_type" in df.columns else _n_total
_n_cities  = df["city"].nunique()
_n_systems = df["system_id"].nunique() if "system_id" in df.columns else "—"

st.title("Cartographie Spatiale du Corpus Gold Standard")
st.caption("Axe de Recherche 2 : Distribution Territoriale et Disparités Géographiques de l'Offre Cyclable")

abstract_box(
    "<b>Question de recherche :</b> Les disparités spatiales de l'offre cyclable partagée "
    "résultent-elles d'une fatalité géographique ou d'inégalités de gouvernance locale ?<br><br>"
    "Cette interface permet la visualisation géospatiale du corpus Gold Standard GBFS : "
    f"<b>{_n_total:,} stations certifiées</b> (dont {_n_dock:,} stations dock-based VLS) "
    f"issues de {_n_systems} systèmes nationaux couvrant {_n_cities} agglomérations, enrichies "
    "selon cinq modules spatiaux dans un rayon normalisé de 300 m. "
    "Le calcul global de l'autocorrélation spatiale (indice de Moran, $I = -0{,}023$, $p = 0{,}765$) "
    "invalide l'hypothèse d'un déterminisme géographique : les stations performantes et "
    "sous-performantes ne forment pas de clusters territoriaux significatifs. "
    "Cette absence de structure spatiale oriente l'interprétation des disparités observées "
    "vers des facteurs de <em>gouvernance locale</em> et de <em>choix politiques d'aménagement</em>."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres de Visualisation")

    # Filtre type de station
    type_options = ["Toutes les stations", "Dock-based (VLS)", "Free-floating", "Autopartage"]
    type_map = {
        "Toutes les stations": None,
        "Dock-based (VLS)": "docked_bike",
        "Free-floating": "free_floating",
        "Autopartage": "carsharing",
    }
    station_type_sel = st.selectbox("Type de station", options=type_options, index=1)

    all_cities = sorted(df["city"].unique())
    city_sel = st.multiselect(
        "Agglomération(s)",
        options=all_cities,
        default=[],
        placeholder="Corpus national complet",
    )

    # Raccourci zoom Montpellier
    if st.button("Zoom Montpellier (Vélomagg — Rang IMD #2)"):
        city_sel = ["Montpellier"]

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
dff = df.copy()
if station_type_sel != "Toutes les stations" and "station_type" in dff.columns:
    _type_val = type_map[station_type_sel]
    dff = dff[dff["station_type"] == _type_val]
if city_sel:
    dff = dff[dff["city"].isin(city_sel)]

n_shown  = len(dff)
n_nodata = int(dff[metric_key].isna().sum()) if metric_key in dff.columns else 0

# ── Section 1 — Couverture ────────────────────────────────────────────────────
section(1, f"Couverture Territoriale — {_n_total:,} Stations Gold Standard ({_n_dock:,} Dock-Based VLS)")

col_info, col_na = st.columns([4, 1])
_type_label = f" ({station_type_sel})" if station_type_sel != "Toutes les stations" else ""
col_info.markdown(
    f"**{n_shown:,}** stations affichées{_type_label} · "
    f"**{dff['city'].nunique()}** agglomérations · "
    f"**{dff['system_id'].nunique() if 'system_id' in dff.columns else '—'}** réseaux GBFS certifiés"
)
col_info.caption(
    "La couverture nationale couvre l'ensemble des agglomérations françaises disposant "
    "d'un système VLS actif au standard GBFS. Les points grisés indiquent des stations "
    "pour lesquelles la métrique sélectionnée est manquante (données non disponibles dans "
    "le buffer de 300 m ou hors périmètre de la source primaire). "
    "**Conseil :** sélectionner 'Dock-based (VLS)' pour l'analyse IMD (5 341 stations certifiées)."
)
if n_nodata:
    col_na.metric("Sans données", f"{n_nodata:,}", delta=None)

# ── KPIs par type de station ──────────────────────────────────────────────────
if "station_type" in df.columns and station_type_sel == "Toutes les stations":
    _type_counts = df["station_type"].value_counts()
    _tc = st.columns(len(_type_counts))
    for i, (stype, cnt) in enumerate(_type_counts.items()):
        _tc[i].metric(stype.replace("_", " ").title(), f"{cnt:,}")

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
            "Type : {station_type}<br/>"
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

_map_cols = ["lat", "lon", "_color", "station_name", "city",
             metric_key, "capacity", "source"]
if "station_type" in dff.columns:
    _map_cols.append("station_type")

layer = pdk.Layer(
    "ScatterplotLayer",
    data=dff[[c for c in _map_cols if c in dff.columns]],
    get_position="[lon, lat]",
    get_fill_color="_color",
    get_radius=point_size,
    pickable=show_tooltip,
    auto_highlight=True,
)

# Centrage automatique sur Montpellier si sélectionné
if city_sel and "Montpellier" in city_sel and len(city_sel) == 1:
    _lat_c, _lon_c, _zoom = 43.6047, 3.8742, 13
elif city_sel and len(dff) > 0:
    _lat_c = float(dff["lat"].mean())
    _lon_c = float(dff["lon"].mean())
    _zoom  = 11
else:
    _lat_c = float(dff["lat"].mean()) if len(dff) > 0 else 46.5
    _lon_c = float(dff["lon"].mean()) if len(dff) > 0 else 2.35
    _zoom  = 5

view_state = pdk.ViewState(latitude=_lat_c, longitude=_lon_c, zoom=_zoom, pitch=0)

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
        f"sur le corpus Gold Standard{_type_label}. "
        f"Intervalle observé : [{vmin:.2f} — {vmax:.2f}] {unit} · "
        f"Moyenne : {vmean:.2f} {unit}. "
        f"Palette chromatique : *{palette}*. "
        f"Points gris : valeur manquante. "
        f"**Montpellier Vélomagg (rang IMD #2 national)** est accessible via le bouton 'Zoom Montpellier' en sidebar."
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
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Stations valides",   f"{len(valid):,}")
    s2.metric("Minimum",            f"{vmin:.2f} {unit}")
    s3.metric("Moyenne",            f"{vmean:.2f} {unit}")
    s4.metric("Médiane",            f"{float(valid.median()):.2f} {unit}")
    s5.metric("Maximum",            f"{vmax:.2f} {unit}")
    s6.metric("Écart-type",         f"{float(valid.std()):.3f} {unit}")
    st.caption(
        f"**Tableau 3.1.** Statistiques univariées de **{meta['label']}** "
        f"sur la sélection courante ({len(valid):,} stations valides / {n_shown:,} totales). "
        f"Q25 : {float(valid.quantile(0.25)):.3f} · Q75 : {float(valid.quantile(0.75)):.3f} {unit}."
    )
else:
    st.info(
        "Aucune valeur valide disponible pour la dimension sélectionnée sur la sélection courante. "
        "Vérifiez le périmètre géographique ou sélectionnez une autre métrique d'enrichissement."
    )
