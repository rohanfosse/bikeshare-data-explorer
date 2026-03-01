"""
1_Carte.py — Visualisation géospatiale du corpus Gold Standard GBFS.
Utilise pydeck (ScatterplotLayer, WebGL) pour rendre 46 000+ points en temps réel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, color_scale_rgb, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Cartographie Spatiale - Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Chargement anticipé (abstract dynamique) ──────────────────────────────────
df = load_stations()
_n_total   = len(df)
_n_dock    = int((df["station_type"] == "docked_bike").sum()) if "station_type" in df.columns else _n_total
_n_cities  = df["city"].nunique()
_n_systems = df["system_id"].nunique() if "system_id" in df.columns else "-"

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
    if st.button("Zoom Montpellier (Vélomagg - Rang IMD #2)"):
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

# ── Section 1 - Couverture ────────────────────────────────────────────────────
section(1, f"Couverture Territoriale - {_n_total:,} Stations Gold Standard ({_n_dock:,} Dock-Based VLS)")

col_info, col_na = st.columns([4, 1])
_type_label = f" ({station_type_sel})" if station_type_sel != "Toutes les stations" else ""
col_info.markdown(
    f"**{n_shown:,}** stations affichées{_type_label} · "
    f"**{dff['city'].nunique()}** agglomérations · "
    f"**{dff['system_id'].nunique() if 'system_id' in dff.columns else '-'}** réseaux GBFS certifiés"
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

# ── Section 2 - Carte ─────────────────────────────────────────────────────────
section(2, "Carte Interactive - Coloration par Dimension d'Enrichissement Spatial (Rayon 300 m)")

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
        f"Intervalle observé : [{vmin:.2f} - {vmax:.2f}] {unit} · "
        f"Moyenne : {vmean:.2f} {unit}. "
        f"Palette chromatique : *{palette}*. "
        f"Points gris : valeur manquante. "
        f"**Montpellier Vélomagg (rang IMD #2 national)** est accessible via le bouton 'Zoom Montpellier' en sidebar."
    )

# ── Section 3 - Statistiques descriptives ─────────────────────────────────────
st.divider()
section(3, "Statistiques Descriptives de la Sélection - Caractérisation Univariée")

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

# ── Section 4 — Distribution de la métrique ────────────────────────────────────
st.divider()
section(4, f"Distribution Empirique — {meta['label']}")

st.markdown(
    f"L'histogramme ci-dessous représente la distribution empirique de la dimension "
    f"**{meta['label']}** sur le corpus sélectionné. "
    f"L'axe des abscisses est exprimé en {meta['unit'] if meta['unit'] else 'unités brutes'}. "
    "Les statistiques de forme (asymétrie, aplatissement) orientent le choix des méthodes "
    "statistiques appropriées (paramétriques vs non-paramétriques)."
)

if len(valid) >= 5:
    _skew  = float(valid.skew())
    _kurt  = float(valid.kurtosis())  # excess kurtosis
    _iqr   = float(valid.quantile(0.75) - valid.quantile(0.25))

    fig_hist = px.histogram(
        valid.rename(meta["label"]),
        nbins=min(60, max(10, len(valid) // 20)),
        color_discrete_sequence=["#1A6FBF"],
        opacity=0.82,
        marginal="box",
    )
    fig_hist.add_vline(x=float(valid.mean()),   line_dash="dash",  line_color="#e74c3c",
                       annotation_text="Moyenne", annotation_position="top right")
    fig_hist.add_vline(x=float(valid.median()), line_dash="dot",   line_color="#27ae60",
                       annotation_text="Médiane", annotation_position="top left")
    fig_hist.update_layout(
        height=340,
        xaxis_title=f"{meta['label']} ({meta['unit']})" if meta["unit"] else meta["label"],
        yaxis_title="Effectif (stations)",
        showlegend=False,
        margin=dict(t=20, b=40, l=50, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_hist.update_xaxes(showgrid=True, gridcolor="#e8ecf1")
    fig_hist.update_yaxes(showgrid=True, gridcolor="#e8ecf1")
    st.plotly_chart(fig_hist, use_container_width=True)

    _norm_flag = "Distribution **non-normale**" if (abs(_skew) > 1 or abs(_kurt) > 3) else "Distribution proche de la normale"
    st.caption(
        f"**Figure 4.1.** Distribution empirique de **{meta['label']}** "
        f"({len(valid):,} stations valides{_type_label}). "
        f"Asymétrie $\\gamma_1 = {_skew:.3f}$ — Aplatissement $\\gamma_2 = {_kurt:.3f}$ (excédentaire). "
        f"IQR = {_iqr:.3f} {meta['unit']}. "
        f"{_norm_flag} — justifie l'usage de tests non-paramétriques (Spearman, Mann-Whitney)."
    )

# ── Section 5 — Classement par agglomération ──────────────────────────────────
st.divider()
section(5, f"Classement des Agglomérations — Moyenne de {meta['label']} par Ville")

st.markdown(
    "L'agrégation par agglomération permet de comparer les niveaux moyens d'enrichissement "
    "spatial entre villes. Cette synthèse révèle les disparités de gouvernance locale "
    "indépendamment du volume de stations. Seules les villes avec au moins 3 stations valides "
    "sont représentées."
)

_city_agg = (
    dff[dff[metric_key].notna()]
    .groupby("city")[metric_key]
    .agg(["mean", "median", "std", "count"])
    .rename(columns={"mean": "Moyenne", "median": "Médiane", "std": "Écart-type", "count": "N stations"})
    .query("`N stations` >= 3")
    .sort_values("Moyenne", ascending=False)
    .reset_index()
)
_city_agg.rename(columns={"city": "Agglomération"}, inplace=True)

if len(_city_agg) >= 2:
    _n_cities_agg = len(_city_agg)
    _slider_max   = min(_n_cities_agg, 40)
    _top_n = st.slider(
        "Nombre d'agglomérations affichées", min_value=5, max_value=_slider_max,
        value=min(20, _slider_max), step=5, key="city_ranking_slider"
    )

    _city_plot = _city_agg.head(_top_n).copy()

    # Couleur : Montpellier en rouge, autres en bleu
    _city_plot["_color"] = _city_plot["Agglomération"].apply(
        lambda c: "#e74c3c" if c == "Montpellier" else "#1A6FBF"
    )

    fig_city = go.Figure()
    fig_city.add_trace(go.Bar(
        x=_city_plot["Agglomération"],
        y=_city_plot["Moyenne"],
        error_y=dict(type="data", array=_city_plot["Écart-type"].tolist(), visible=True, color="#aab4c0"),
        marker_color=_city_plot["_color"].tolist(),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"Moyenne : %{{y:.3f}} {meta['unit']}<br>"
            "<extra></extra>"
        ),
    ))
    _dir_label = "valeur élevée favorable" if meta["higher_is_better"] else "valeur faible favorable"
    fig_city.update_layout(
        height=420,
        xaxis=dict(tickangle=-40, title="Agglomération"),
        yaxis=dict(title=f"{meta['label']} — Moyenne ({meta['unit']})" if meta["unit"] else f"{meta['label']} — Moyenne",
                   showgrid=True, gridcolor="#e8ecf1"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=20, b=100, l=60, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_city, use_container_width=True)
    st.caption(
        f"**Figure 5.1.** Classement des {_top_n} agglomérations par moyenne de **{meta['label']}** "
        f"({_dir_label}). "
        f"Barres d'erreur = écart-type intra-ville. "
        f"Montpellier (rouge) est mis en exergue à titre de référence (rang IMD #2 national). "
        f"Source : Gold Standard GBFS, enrichissement spatial 300 m."
    )

    # Tableau synthétique
    with st.expander(f"Tableau détaillé — {_n_cities_agg} agglomérations", expanded=False):
        _city_display = _city_agg.copy()
        for col in ["Moyenne", "Médiane", "Écart-type"]:
            _city_display[col] = _city_display[col].map(lambda v: f"{v:.3f}")
        _city_display["N stations"] = _city_display["N stations"].astype(int)
        _city_display.insert(0, "Rang", range(1, len(_city_display) + 1))
        st.dataframe(_city_display, use_container_width=True, hide_index=True)
        st.caption(
            f"**Tableau 5.1.** Synthèse par agglomération pour la dimension **{meta['label']}**. "
            f"{_n_cities_agg} villes avec $n \\geq 3$ stations valides{_type_label}."
        )
else:
    st.info(
        "Pas assez d'agglomérations avec données suffisantes pour établir un classement. "
        "Sélectionnez 'Toutes les stations' ou élargissez le filtre géographique."
    )

# ── Section 6 — Méthodologie ────────────────────────────────────────────────────
st.divider()
section(6, "Note Méthodologique — Enrichissement Spatial 300 m et Autocorrélation")

st.markdown(r"""
**Protocole d'enrichissement spatial.** Chaque station du corpus Gold Standard a été enrichie
selon un rayon normalisé de **300 mètres** à partir de cinq sources administratives :

| Dimension | Source primaire | Périmètre |
|---|---|---|
| Densité résidentielle | INSEE RP 2019 (carreaux 200 m) | 300 m |
| Accessibilité multimodale | GTFS national (SNCF, RATP, Métropoles) | 300 m isochrone marche |
| Mixité fonctionnelle | IGN BD TOPO — POI commerces/services | 300 m |
| Topographie (TRI) | SRTM 30 m (NASA/USGS) | 500 m × 500 m |
| Profil socio-économique | INSEE Filosofi 2019 (carreaux 200 m) | 300 m |

**Autocorrélation spatiale.** L'indice de Moran global ($I = -0{,}023$, $p = 0{,}765$,
permutation $n = 999$) calculé sur l'ensemble du corpus dock-based est **non significatif**
(seuil $\alpha = 0{,}05$). Cette absence de structure spatiale macroscopique invalide
l'hypothèse d'un déterminisme géographique et oriente l'interprétation vers des facteurs
de **gouvernance locale**. L'analyse à l'échelle intra-urbaine (LISA — Local Indicators of
Spatial Association) peut révéler des clusters locaux non détectés à l'échelle nationale.

**Limites.** (1) Le rayon de 300 m ne capture pas les discontinuités de voirie (coupures
autoroutières, voies ferrées). (2) Les données Filosofi sont interpolées à l'échelle de
carreaux 200 m — la résolution peut introduire un biais d'agrégation (MAUP). (3) Le TRI
SRTM 30 m présente une précision altimétrique de ±16 m en zone urbaine plane.
""")

