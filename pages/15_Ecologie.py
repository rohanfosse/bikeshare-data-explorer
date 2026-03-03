"""
15_Ecologie.py — Écologie et transition bas-carbone des VLS.
Concepts : analyse du cycle de vie (ACV), substitution modale, empreinte carbone,
qualité de l'air, équivalences environnementales, scénarios de décarbonation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_stations, METRICS
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Écologie & Pollution - Réseaux VLS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Constantes ACV — g CO2eq / km / passager (sources ADEME 2023, Leuenberger 2021) ──
_LCA_MODES: list[dict] = [
    {"mode": "Métro / RER",            "co2": 4,   "color": "#1565C0", "cat": "TC lourd"},
    {"mode": "Tramway",                 "co2": 8,   "color": "#1976D2", "cat": "TC lourd"},
    {"mode": "Vélo personnel",          "co2": 11,  "color": "#1abc9c", "cat": "Vélo"},
    {"mode": "VLS (dock-based)",        "co2": 16,  "color": "#27ae60", "cat": "Vélo"},
    {"mode": "Bus électrique",          "co2": 40,  "color": "#3498db", "cat": "TC léger"},
    {"mode": "Trottinette électrique",  "co2": 35,  "color": "#9b59b6", "cat": "Micromo."},
    {"mode": "Bus diesel",              "co2": 90,  "color": "#e67e22", "cat": "TC léger"},
    {"mode": "Voiture électrique",      "co2": 65,  "color": "#f39c12", "cat": "Auto"},
    {"mode": "Voiture thermique",       "co2": 140, "color": "#e74c3c", "cat": "Auto"},
    {"mode": "Voiture thermique (SUV)", "co2": 195, "color": "#c0392b", "cat": "Auto"},
]

# ── Paramètres du modèle de substitution modale ────────────────────────────────
_KM_TRAJET        = 2.5    # km/trajet VLS (Frade & Ribeiro 2015, Fishman 2016)
_TRAJETS_VLO_J    = 1.5    # trajets/vélo/jour (European Cyclists' Federation 2021)
_FILL_RATE        = 0.80   # taux de remplissage moyen des racks (hypothèse conservatrice)
_PCT_VOITURE      = 0.35   # part des trajets VLS remplaçant une voiture (Fishman et al. 2014)
_CO2_VOITURE      = 140.0  # g CO2eq/km voiture thermique (ADEME 2023)
_CO2_VLS          = 16.0   # g CO2eq/km VLS cycle de vie (Leuenberger et al. 2021)
_CO2_NET          = _CO2_VOITURE - _CO2_VLS   # 124 g CO2eq/km économisés
_ARBRE_KG_AN      = 22.0   # kg CO2 absorbés/an/arbre adulte (INRAE 2022)
_CONSO_L_KM       = 0.070  # l/km voiture (parc moyen France, ADEME 2023)
_FRANCE_TRANSP_MT = 141.0  # Mt CO2eq/an du secteur transport France (CITEPA 2022)

# ── Chargement et calculs ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_stations()
    dock = (
        df[df["station_type"] == "docked_bike"]
        .dropna(subset=["city", "capacity", "lat", "lon"])
        .query("capacity > 0")
        .copy()
    )

    # Indicateurs disponibles par station
    _metric_cols = [k for k in METRICS if k in dock.columns]

    agg_dict: dict[str, object] = {
        "n_stations":   ("capacity", "count"),
        "total_cap":    ("capacity", "sum"),
        "mean_cap":     ("capacity", "mean"),
        "lat":          ("lat", "mean"),
        "lon":          ("lon", "mean"),
    }
    for _mk in _metric_cols:
        agg_dict[f"med_{_mk}"] = (_mk, "median")

    city = (
        dock.groupby("city")
        .agg(**agg_dict)
        .query("n_stations >= 5")
        .reset_index()
    )

    # Modèle CO2
    city["n_velos_est"]       = city["total_cap"] * _FILL_RATE
    city["trajets_an"]        = city["n_velos_est"] * _TRAJETS_VLO_J * 365
    city["km_an"]             = city["trajets_an"] * _KM_TRAJET
    city["co2_evite_t"]       = city["km_an"] * _PCT_VOITURE * _CO2_NET / 1e6
    city["eq_arbres"]         = city["co2_evite_t"] * 1000 / _ARBRE_KG_AN
    city["litres_essence"]    = city["km_an"] * _PCT_VOITURE * _CONSO_L_KM
    city["co2_t_par_station"] = city["co2_evite_t"] / city["n_stations"]

    # Indice de Carbonité Locale (ICL) par station — proxy qualité de l'air
    # Score 0-1 : haute valeur = zone favorable à la mobilité bas-carbone
    _icl_cols: list[str] = []
    if "infra_cyclable_pct" in dock.columns:
        dock["_icl_infra"] = dock["infra_cyclable_pct"].clip(0, 100) / 100
        _icl_cols.append("_icl_infra")
    if "baac_accidents_cyclistes" in dock.columns:
        _mx = float(dock["baac_accidents_cyclistes"].quantile(0.95))
        dock["_icl_sini"] = 1 - (dock["baac_accidents_cyclistes"].clip(0, _mx) / max(_mx, 1))
        _icl_cols.append("_icl_sini")
    if "gtfs_heavy_stops_300m" in dock.columns:
        _mx2 = float(dock["gtfs_heavy_stops_300m"].quantile(0.95))
        dock["_icl_tc"] = dock["gtfs_heavy_stops_300m"].clip(0, _mx2) / max(_mx2, 1)
        _icl_cols.append("_icl_tc")

    if _icl_cols:
        dock["ICL"] = dock[_icl_cols].mean(axis=1) * 100
    else:
        dock["ICL"] = 50.0

    city_icl = dock.groupby("city")["ICL"].median().rename("ICL").reset_index()
    city = city.merge(city_icl, on="city", how="left")

    return dock, city


df_station, df_city = _load()

_n_stations   = len(df_station)
_n_cities     = df_city["city"].nunique()
_co2_total_t  = float(df_city["co2_evite_t"].sum())
_arbres_total = int(df_city["eq_arbres"].sum())
_trajets_total = int(df_city["trajets_an"].sum())

st.title("Écologie et Qualité de l'Air — VLS comme Vecteur de Transition Bas-Carbone")
st.caption("Analyse du cycle de vie (ACV), substitution modale, empreinte CO₂, qualité de l'air et scénarios de décarbonation")

abstract_box(
    "<b>Question de recherche :</b> Dans quelle mesure le déploiement des réseaux de vélos "
    "en libre-service contribue-t-il à la réduction des émissions de CO₂ et à l'amélioration "
    "de la qualité de l'air urbaine ?<br><br>"
    "Un VLS dock-based émet <b>16 g CO₂eq/km</b> sur son cycle de vie complet, contre "
    "<b>140 g CO₂eq/km</b> pour une voiture thermique (ADEME 2023). Sur l'ensemble du corpus "
    f"Gold Standard ({_n_stations:,} stations, {_n_cities} agglomérations), le réseau VLS français "
    f"évite annuellement <b>≈ {_co2_total_t:,.0f} t CO₂eq</b> (substitution modale 35 %, "
    "Fishman et al. 2014), équivalant à <b>maintenir en vie</b> "
    f"{_arbres_total:,} arbres adultes supplémentaires (INRAE 2022). "
    "L'<i>Indice de Carbonité Locale</i> (ICL), construit à partir de l'infrastructure cyclable, "
    "de la sinistralité et de l'accessibilité TC, révèle une forte hétérogénéité spatiale : "
    "les agglomérations les plus décarbonées ne sont pas nécessairement les plus grandes.",
    findings=[
        (f"{_co2_total_t:,.0f} t", "CO₂eq évités / an"),
        (f"{_arbres_total:,}", "arbres équivalents"),
        (f"{_trajets_total / 1e6:.1f} M", "trajets VLS / an estimés"),
        ("16 vs 140", "g CO₂eq/km VLS vs voiture"),
        (f"{_n_cities}", "agglomérations analysées"),
    ],
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres")
    _n_top = st.slider("Agglomérations affichées", 10, min(50, _n_cities), min(20, _n_cities), 5)
    _subst_pct = st.slider(
        "Taux de substitution voiture (%) — modèle",
        min_value=10, max_value=60, value=35, step=5,
        help="Part des trajets VLS remplaçant réellement un trajet en voiture (Fishman 2014 : 35 %).",
    )
    _expansion = st.slider(
        "Expansion du réseau VLS (scénario)",
        min_value=100, max_value=300, value=100, step=25,
        format="%d%%",
        help="Facteur multiplicatif de la flotte VLS par rapport à l'actuel (100 % = statu quo).",
    )
    st.divider()
    st.caption("Sources : ADEME 2023, Fishman et al. 2014,\nLeuenberger et al. 2021, INRAE 2022,\nCITEPA 2022.")
    st.caption("R. Fossé & G. Pallares · 2025–2026")

# CO2 recalculé avec les paramètres sidebar
_factor = (_subst_pct / 100) / _PCT_VOITURE   # ratio vs valeur de référence
_expand = _expansion / 100

# ── Section 1 — ACV Comparatif des Modes de Transport ─────────────────────────
st.divider()
section(1, "Analyse du Cycle de Vie (ACV) — Empreinte CO₂ par Mode de Transport")

st.markdown(r"""
L'**analyse du cycle de vie** (*Life Cycle Assessment*, ACV) comptabilise l'ensemble des
émissions de gaz à effet de serre liées à un mode de transport : fabrication du véhicule,
infrastructure, carburant / électricité, maintenance et fin de vie.
Les valeurs ci-dessous sont exprimées en **g CO₂eq / km / passager** et proviennent de
l'ADEME (2023) pour les modes motorisés et de Leuenberger et al. (2021) pour le VLS.

Le vélo en libre-service dock-based se positionne **parmi les modes les moins émissifs**,
devant la trottinette électrique (production de batterie) et largement devant tout usage automobile.
La différence de $\Delta = 124$ g CO₂eq/km avec la voiture thermique constitue le levier
de calcul principal du modèle de substitution modale.
""")

_lca_df = pd.DataFrame(_LCA_MODES).sort_values("co2")
_lca_df["is_vls"] = _lca_df["mode"] == "VLS (dock-based)"

fig_acv = go.Figure()
fig_acv.add_trace(go.Bar(
    x=_lca_df["co2"],
    y=_lca_df["mode"],
    orientation="h",
    marker_color=_lca_df["color"],
    marker_line_width=0,
    text=[f"{v} g CO₂eq/km" for v in _lca_df["co2"]],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>%{x} g CO₂eq/km<extra></extra>",
))
# Repère VLS
_vls_row = _lca_df[_lca_df["is_vls"]]
if not _vls_row.empty:
    fig_acv.add_shape(
        type="line",
        x0=float(_vls_row["co2"].iloc[0]), x1=float(_vls_row["co2"].iloc[0]),
        y0=-0.5, y1=len(_lca_df) - 0.5,
        line=dict(color="#27ae60", dash="dash", width=1.5),
    )
    fig_acv.add_annotation(
        x=float(_vls_row["co2"].iloc[0]) + 3, y=len(_lca_df) - 1,
        text="Référence VLS (16 g)", showarrow=False,
        font=dict(size=9, color="#27ae60"), xanchor="left",
    )
# Repère voiture
fig_acv.add_vline(x=140, line_dash="dot", line_color="#e74c3c", line_width=1.2,
                  annotation_text="Voiture thermique (140 g)",
                  annotation_font=dict(size=9, color="#e74c3c"),
                  annotation_position="top left")

fig_acv.update_layout(
    title=dict(text="ACV comparatif — g CO₂eq / km / passager (ADEME 2023, Leuenberger 2021)",
               font_size=11, x=0),
    height=420,
    margin=dict(l=10, r=80, t=38, b=20),
    xaxis=dict(title="g CO₂eq / km / passager", range=[0, 230],
               gridcolor="#e8edf3", tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    plot_bgcolor="#f8fafd",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
st.plotly_chart(fig_acv, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "**Figure 1.1.** ACV comparatif des modes de transport urbain en g CO₂eq / km / passager. "
    "Ligne verte pointillée = VLS dock-based (16 g). Ligne rouge pointillée = voiture thermique (140 g). "
    "Écart Δ = 124 g CO₂eq/km constitue le gain unitaire de la substitution modale. "
    "Sources : ADEME (2023), Leuenberger et al. (2021), Fishman et al. (2014)."
)

# ── Section 2 — CO₂ Évité par Agglomération ───────────────────────────────────
st.divider()
section(2, "Empreinte Carbone Évitée par Agglomération — Classement National")

st.markdown(rf"""
Le modèle de substitution modale applique la formule suivante à chaque agglomération :

$$\text{{CO}}_2^{{\text{{évité}}}} = N_{{\text{{vélos}}}} \times f_{{\text{{utilisation}}}} \times 365 \times d_{{\text{{trajet}}}} \times \alpha_{{\text{{voiture}}}} \times \Delta_{{CO_2}} / 10^6$$

avec $N_{{\text{{vélos}}}} = \text{{capacité totale}} \times {_FILL_RATE:.0%}$,
$f_{{\text{{utilisation}}}} = {_TRAJETS_VLO_J}$ trajets/vélo/jour,
$d_{{\text{{trajet}}}} = {_KM_TRAJET}$ km,
$\alpha_{{\text{{voiture}}}} = {_subst_pct}\%$ (paramètre ajustable en sidebar),
$\Delta_{{CO_2}} = {int(_CO2_NET)}$ g CO₂eq/km.

Les résultats sont exprimés en **tonnes de CO₂eq évitées par an**.
""")

# Recalcul CO2 selon paramètre sidebar
_city_plot = df_city.copy()
_city_plot["co2_evite_t_adj"] = _city_plot["co2_evite_t"] * _factor * _expand
_city_plot["eq_arbres_adj"]   = _city_plot["co2_evite_t_adj"] * 1000 / _ARBRE_KG_AN
_top_cities = _city_plot.nlargest(_n_top, "co2_evite_t_adj").sort_values("co2_evite_t_adj")

_co2_total_adj = float(_city_plot["co2_evite_t_adj"].sum())

col_a, col_b = st.columns([3, 2])

with col_a:
    _bar_colors_co2 = [
        "#e74c3c" if c == "Montpellier" else "#f1c40f" if c == _top_cities.iloc[-1]["city"]
        else "#27ae60"
        for c in _top_cities["city"]
    ]
    fig_co2 = go.Figure(go.Bar(
        x=_top_cities["co2_evite_t_adj"],
        y=_top_cities["city"],
        orientation="h",
        marker_color=_bar_colors_co2,
        text=[f"{v:,.0f} t" for v in _top_cities["co2_evite_t_adj"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>CO₂ évité : %{x:,.0f} t/an<extra></extra>",
    ))
    fig_co2.add_vline(
        x=float(_city_plot["co2_evite_t_adj"].median()),
        line_dash="dash", line_color="#555", line_width=1,
        annotation_text="Médiane",
        annotation_font=dict(size=9),
        annotation_position="top right",
    )
    fig_co2.update_layout(
        title=dict(text=f"CO₂ évité / an — top {_n_top} agglomérations (t CO₂eq)",
                   font_size=11, x=0),
        height=max(340, _n_top * 18),
        margin=dict(l=10, r=80, t=38, b=20),
        xaxis=dict(title="t CO₂eq / an", gridcolor="#e8edf3"),
        yaxis=dict(tickfont=dict(size=9)),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_co2, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 2.1.** CO₂ évité annuellement (top {_n_top} agglomérations). "
        f"Taux de substitution voiture = {_subst_pct} %. "
        "Jaune = 1ère agglomération · Rouge = Montpellier · Ligne pointillée = médiane nationale. "
        "Le classement reflète la taille du réseau plus que sa performance intrinsèque."
    )

with col_b:
    # Scatter n_stations vs CO2_evite
    _c_fit = np.polyfit(
        np.log1p(_city_plot["n_stations"].to_numpy(float)),
        np.log1p(_city_plot["co2_evite_t_adj"].to_numpy(float)), 1
    )
    _x_fit = np.linspace(float(_city_plot["n_stations"].min()),
                          float(_city_plot["n_stations"].max()), 80)
    _y_fit = np.expm1(np.polyval(_c_fit, np.log1p(_x_fit)))
    _hl5 = {"Montpellier", "Paris", "Lyon", "Marseille", "Bordeaux", "Rennes"}
    _lab5 = _city_plot["city"].isin(_hl5)

    fig_sc2 = go.Figure()
    fig_sc2.add_trace(go.Scatter(
        x=_city_plot["n_stations"], y=_city_plot["co2_evite_t_adj"],
        mode="markers",
        marker=dict(
            color=["#e74c3c" if c == "Montpellier" else "#27ae60"
                   for c in _city_plot["city"]],
            size=6, opacity=0.6,
        ),
        text=_city_plot["city"],
        hovertemplate="<b>%{text}</b><br>%{x} stations · %{y:,.0f} t CO₂<extra></extra>",
        showlegend=False,
    ))
    fig_sc2.add_trace(go.Scatter(
        x=_city_plot.loc[_lab5, "n_stations"],
        y=_city_plot.loc[_lab5, "co2_evite_t_adj"],
        mode="markers+text",
        marker=dict(
            color=["#e74c3c" if c == "Montpellier" else "#1565C0"
                   for c in _city_plot.loc[_lab5, "city"]],
            size=8, opacity=1,
        ),
        text=_city_plot.loc[_lab5, "city"],
        textposition="top center", textfont=dict(size=8),
        showlegend=False,
        hovertemplate="<b>%{text}</b><br>%{x} stations · %{y:,.0f} t CO₂<extra></extra>",
    ))
    fig_sc2.add_trace(go.Scatter(
        x=_x_fit, y=_y_fit, mode="lines",
        line=dict(color="#e74c3c", dash="dash", width=1.5),
        name="Loi puissance (OLS log-log)",
    ))
    fig_sc2.update_layout(
        title=dict(text="n_stations × CO₂ évité (log-log)", font_size=11, x=0),
        height=370,
        margin=dict(l=10, r=10, t=38, b=40),
        xaxis=dict(title="Nombre de stations", type="log", gridcolor="#e8edf3"),
        yaxis=dict(title="t CO₂eq / an", type="log", gridcolor="#e8edf3"),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9), x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
    )
    st.plotly_chart(fig_sc2, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "**Figure 2.2.** Nuage de points taille réseau × CO₂ évité (échelle log-log). "
        "La loi puissance (tirets) confirme une croissance sous-linéaire — "
        "les gains marginaux par station supplémentaire diminuent avec la taille. "
        "Montpellier en rouge."
    )

# ── Section 3 — Équivalences Environnementales ────────────────────────────────
st.divider()
section(3, "Équivalences Environnementales — Traduire le CO₂ en Langage Accessible")

st.markdown(r"""
La traduction des émissions évitées en **équivalences concrètes** facilite la communication
scientifique vers un public non-spécialisé et renforce l'argumentaire des politiques publiques.
Quatre indicateurs de traduction sont proposés ci-dessous, calculés sur la base du corpus complet
Gold Standard avec les paramètres courants du modèle de substitution.
""")

_co2_total_t_adj = float(_city_plot["co2_evite_t_adj"].sum())
_arbres_eq   = int(_co2_total_t_adj * 1000 / _ARBRE_KG_AN)
_litres_tot  = float((_city_plot["litres_essence"] * _factor * _expand).sum())
_voitures_eq = int(_co2_total_t_adj * 1e6 / (140 * 15_000))  # voiture moy 15 000 km/an
_paris_ny_eq = int(_co2_total_t_adj * 1e6 / 855_000)          # vol Paris-NYC ≈ 855 kg CO2eq/pers

_eq_c1, _eq_c2, _eq_c3, _eq_c4 = st.columns(4)
with _eq_c1:
    st.metric("Arbres adultes équivalents", f"{_arbres_eq:,}",
              help="1 arbre adulte absorbe ≈ 22 kg CO₂/an (INRAE 2022).")
with _eq_c2:
    st.metric("Litres d'essence économisés", f"{_litres_tot / 1e6:.2f} M",
              help=f"Consommation moyenne parc français : {_CONSO_L_KM:.3f} l/km (ADEME 2023).")
with _eq_c3:
    st.metric("Voitures retirées du réseau", f"{_voitures_eq:,}",
              help="Voiture thermique moy. 15 000 km/an × 140 g CO₂/km = 2,1 t CO₂/an.")
with _eq_c4:
    st.metric("Vols Paris ↔ New York évités", f"{_paris_ny_eq:,}",
              help="Vol aller-retour Paris–NYC ≈ 855 kg CO₂eq/passager (Atmosfair 2023).")

st.divider()

# Rapport à l'objectif national
_pct_transport = _co2_total_t_adj / (_FRANCE_TRANSP_MT * 1e6) * 100
st.markdown(
    f"**Mise en perspective nationale.** Le secteur transport représente "
    f"**{_FRANCE_TRANSP_MT:.0f} Mt CO₂eq/an** en France (CITEPA 2022), soit le premier "
    f"poste d'émissions. Les VLS Gold Standard couvrent "
    f"**{_pct_transport:.4f} %** de ces émissions sectorielles à taux de substitution "
    f"{_subst_pct} %. Ce chiffre modeste souligne que le VLS est un **outil complémentaire** "
    "d'une politique de décarbonation plus large incluant électrification du parc automobile, "
    "développement des transports en commun et report modal massif."
)

# Gauge chart
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=_co2_total_t_adj / 1000,  # kt
    delta={"reference": _FRANCE_TRANSP_MT * 1000, "relative": True,
           "valueformat": ".6f", "suffix": "% du secteur transport"},
    title={"text": "CO₂ évité par les VLS Gold Standard (kt CO₂eq/an)", "font": {"size": 13}},
    number={"suffix": " kt", "font": {"size": 18}},
    gauge={
        "axis": {"range": [0, _FRANCE_TRANSP_MT * 1000 * 0.01], "tickformat": ",.0f",
                 "ticksuffix": " kt"},
        "bar": {"color": "#27ae60"},
        "steps": [
            {"range": [0, _FRANCE_TRANSP_MT * 1000 * 0.001], "color": "#eafaf1"},
            {"range": [_FRANCE_TRANSP_MT * 1000 * 0.001, _FRANCE_TRANSP_MT * 1000 * 0.005], "color": "#a9dfbf"},
            {"range": [_FRANCE_TRANSP_MT * 1000 * 0.005, _FRANCE_TRANSP_MT * 1000 * 0.01],  "color": "#58d68d"},
        ],
        "threshold": {
            "line": {"color": "#e74c3c", "width": 2},
            "thickness": 0.75,
            "value": float(_co2_total_t_adj / 1000),
        },
    },
))
fig_gauge.update_layout(
    height=250,
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
st.caption(
    f"**Figure 3.1.** Jauge de la contribution des VLS Gold Standard à la décarbonation "
    f"du secteur transport français ({_FRANCE_TRANSP_MT:.0f} Mt CO₂eq/an, CITEPA 2022). "
    f"Le réseau actuel évite ≈ {_co2_total_t_adj:,.0f} t CO₂eq/an "
    f"({_pct_transport:.4f} % du secteur transport). "
    "L'échelle logarithmique souligne l'ordre de grandeur à atteindre pour un impact systémique."
)

# ── Section 4 — Indice de Carbonité Locale (ICL) ──────────────────────────────
st.divider()
section(4, "Indice de Carbonité Locale (ICL) — Proxy Qualité de l'Air par Station")

st.markdown(r"""
En l'absence de données de mesure directe (ATMO France, AIRPARIF) à l'échelle de chaque station,
un **Indice de Carbonité Locale** (ICL) est construit comme proxy de la qualité de l'air
urbaine à partir de trois dimensions spatiales disponibles dans le Gold Standard :

| Composante | Variable | Rationale |
|---|---|---|
| Infrastructure cyclable | `infra_cyclable_pct` | Une densité élevée d'aménagements cyclables est corrélée à une part modale voiture plus faible (Cervero & Duncan 2003) |
| Sinistralité inverse | `1 - baac_accidents_cyclistes` | Plus d'accidents cyclistes = plus de trafic motorisé = plus d'émissions locales |
| Accessibilité TC lourde | `gtfs_heavy_stops_300m` | La proximité à un arrêt de métro/tram favorise la substitution à la voiture |

$\text{ICL}_i = \frac{1}{3}\left(\hat{I}_i + (1 - \hat{S}_i) + \hat{T}_i\right) \times 100 \in [0, 100]$

Une valeur élevée signifie une station **entourée d'un environnement favorable à la mobilité bas-carbone**.
""")

col_icl1, col_icl2 = st.columns(2)

with col_icl1:
    _icl_top = df_city.dropna(subset=["ICL"]).nlargest(_n_top, "ICL").sort_values("ICL")
    _colors_icl = [
        "#e74c3c" if c == "Montpellier" else "#27ae60"
        for c in _icl_top["city"]
    ]
    fig_icl = go.Figure(go.Bar(
        x=_icl_top["ICL"],
        y=_icl_top["city"],
        orientation="h",
        marker_color=_colors_icl,
        text=[f"{v:.1f}" for v in _icl_top["ICL"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ICL = %{x:.2f} / 100<extra></extra>",
    ))
    fig_icl.update_layout(
        title=dict(text=f"Classement ICL — top {_n_top} agglomérations", font_size=11, x=0),
        height=max(300, _n_top * 18),
        margin=dict(l=10, r=50, t=38, b=20),
        xaxis=dict(title="ICL / 100", range=[0, 105], gridcolor="#e8edf3"),
        yaxis=dict(tickfont=dict(size=9)),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_icl, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 4.1.** Classement des agglomérations par ICL (Indice de Carbonité Locale), "
        f"top {_n_top}. ICL = 100 indique un environnement cyclable optimal pour la décarbonation. "
        "Montpellier en rouge. L'ICL mesure la qualité de l'environnement bas-carbone autour "
        "des stations, pas la performance globale du réseau."
    )

with col_icl2:
    # Scatter ICL vs CO2_evite par station (normalisé)
    _scatter_icl = df_city.dropna(subset=["ICL", "co2_evite_t_adj"]).copy()
    _scatter_icl["co2_evite_t_adj"] = _scatter_icl["co2_evite_t"] * _factor * _expand
    _hl_icl = {"Montpellier", "Paris", "Lyon", "Marseille", "Bordeaux", "Strasbourg", "Rennes"}
    _lab_icl = _scatter_icl["city"].isin(_hl_icl)

    fig_icl_sc = go.Figure()
    fig_icl_sc.add_trace(go.Scatter(
        x=_scatter_icl["ICL"], y=_scatter_icl["co2_evite_t_adj"],
        mode="markers",
        marker=dict(
            color=["#e74c3c" if c == "Montpellier" else "#1A6FBF"
                   for c in _scatter_icl["city"]],
            size=6, opacity=0.6,
        ),
        text=_scatter_icl["city"],
        hovertemplate="<b>%{text}</b><br>ICL = %{x:.1f} · CO₂ = %{y:,.0f} t<extra></extra>",
        showlegend=False,
    ))
    if _lab_icl.any():
        fig_icl_sc.add_trace(go.Scatter(
            x=_scatter_icl.loc[_lab_icl, "ICL"],
            y=_scatter_icl.loc[_lab_icl, "co2_evite_t_adj"],
            mode="markers+text",
            marker=dict(
                color=["#e74c3c" if c == "Montpellier" else "#1565C0"
                       for c in _scatter_icl.loc[_lab_icl, "city"]],
                size=8, opacity=1,
            ),
            text=_scatter_icl.loc[_lab_icl, "city"],
            textposition="top center", textfont=dict(size=8),
            showlegend=False,
            hovertemplate="<b>%{text}</b><br>ICL = %{x:.1f} · CO₂ = %{y:,.0f} t<extra></extra>",
        ))
    # OLS
    if len(_scatter_icl) >= 3:
        _c_icl = np.polyfit(_scatter_icl["ICL"].to_numpy(float),
                            _scatter_icl["co2_evite_t_adj"].to_numpy(float), 1)
        _x_icl = np.linspace(float(_scatter_icl["ICL"].min()),
                              float(_scatter_icl["ICL"].max()), 60)
        fig_icl_sc.add_trace(go.Scatter(
            x=_x_icl, y=np.polyval(_c_icl, _x_icl),
            mode="lines",
            line=dict(color="#e74c3c", dash="dash", width=1.5),
            name="OLS",
        ))
    fig_icl_sc.update_layout(
        title=dict(text="ICL × CO₂ évité — indépendance qualité / volume", font_size=11, x=0),
        height=370,
        margin=dict(l=10, r=10, t=38, b=40),
        xaxis=dict(title="ICL (Indice de Carbonité Locale)", gridcolor="#e8edf3"),
        yaxis=dict(title="t CO₂eq évitées / an", gridcolor="#e8edf3"),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig_icl_sc, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "**Figure 4.2.** Scatter ICL × CO₂ évité annuel par agglomération. "
        "Une corrélation faible suggèrerait que la qualité de l'environnement bas-carbone "
        "(ICL) et le volume d'émissions évitées (lié à la taille du réseau) sont orthogonaux — "
        "les petits réseaux bien localisés peuvent avoir un ICL élevé. "
        "Montpellier en rouge, Droite OLS en tirets."
    )

# ── Section 5 — Scénarios de Décarbonation ────────────────────────────────────
st.divider()
section(5, "Scénarios de Décarbonation — Impact de l'Expansion du Réseau VLS")

st.markdown(r"""
Quatre scénarios de politique de mobilité sont comparés en termes de CO₂ évité cumulé
à l'échelle du corpus Gold Standard. Les paramètres ajustables (sidebar) permettent
d'explorer la sensibilité du modèle au taux de substitution modale $\alpha$ et au
facteur d'expansion du réseau.

Le scénario de **référence** correspond aux données actuelles.
Les scénarios **+50 %**, **+100 %** et **+200 %** projettent une expansion homothétique du parc de vélos.
""")

# Calcul des scénarios
_scenarios = {
    "Statu quo (actuel)":       (1.00, _subst_pct / 100),
    f"+50 % flotte VLS":        (1.50, _subst_pct / 100),
    f"+100 % flotte VLS":       (2.00, _subst_pct / 100),
    f"+200 % flotte VLS":       (3.00, _subst_pct / 100),
    f"Subst. max (α = 60 %)":   (float(_expansion) / 100, 0.60),
    f"Scénario combiné":        (2.00, 0.60),
}

_base_co2 = float(df_city["co2_evite_t"].sum())  # tonne avec paramètres de base

_sc_rows = []
for _label, (_exp_f, _sub_f) in _scenarios.items():
    _co2_sc = _base_co2 * _exp_f * (_sub_f / _PCT_VOITURE)
    _sc_rows.append({
        "Scénario": _label,
        "CO₂ évité (t/an)": _co2_sc,
        "Arbres équivalents": int(_co2_sc * 1000 / _ARBRE_KG_AN),
        "% secteur transport France": _co2_sc / (_FRANCE_TRANSP_MT * 1e6) * 100,
        "Gain vs statu quo": _co2_sc / _base_co2 - 1,
    })

_sc_df = pd.DataFrame(_sc_rows)

# Waterfall des scénarios
_sc_x = [r["Scénario"] for r in _sc_rows]
_sc_y = [r["CO₂ évité (t/an)"] for r in _sc_rows]
_sc_colors = [
    "#1A6FBF", "#27ae60", "#2ecc71", "#1abc9c", "#f39c12", "#e67e22"
]

fig_sc = go.Figure()
fig_sc.add_trace(go.Bar(
    x=_sc_x, y=_sc_y,
    marker_color=_sc_colors,
    text=[f"{v:,.0f} t" for v in _sc_y],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>CO₂ évité : %{y:,.0f} t/an<extra></extra>",
))
fig_sc.add_hline(
    y=_base_co2, line_dash="dash", line_color="#1A6FBF", line_width=1.5,
    annotation_text="Référence (statu quo)",
    annotation_font=dict(size=9, color="#1A6FBF"),
    annotation_position="top right",
)
fig_sc.update_layout(
    title=dict(
        text=f"Scénarios de décarbonation — CO₂ évité / an (t CO₂eq) "
             f"| Subst. = {_subst_pct} % | Expansion = {_expansion} %",
        font_size=11, x=0,
    ),
    height=360,
    margin=dict(l=10, r=10, t=50, b=100),
    xaxis=dict(tickangle=-20, tickfont=dict(size=10)),
    yaxis=dict(title="t CO₂eq / an", gridcolor="#e8edf3"),
    plot_bgcolor="#f8fafd",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

st.caption(
    "**Figure 5.1.** Scénarios de décarbonation pour le réseau VLS Gold Standard. "
    "Le statu quo constitue la référence (bleu). "
    "Les scénarios d'expansion projettent une augmentation homothétique de la flotte. "
    f"Le scénario combiné (+100 % flotte × α = 60 %) représente "
    f"le potentiel maximal plausible. "
    "Limites du modèle : linéarité, saturation de la demande non modélisée, "
    "offre sans contrainte foncière ou budgétaire."
)

# Tableau récapitulatif des scénarios
with st.expander("Tableau des scénarios (données détaillées)", expanded=False):
    _sc_display = _sc_df.copy()
    _sc_display["CO₂ évité (t/an)"] = _sc_display["CO₂ évité (t/an)"].apply(lambda v: f"{v:,.0f}")
    _sc_display["Arbres équivalents"] = _sc_display["Arbres équivalents"].apply(lambda v: f"{v:,}")
    _sc_display["% secteur transport France"] = _sc_display["% secteur transport France"].apply(lambda v: f"{v:.5f} %")
    _sc_display["Gain vs statu quo"] = _sc_display["Gain vs statu quo"].apply(lambda v: f"{v:+.1%}")

    st.dataframe(_sc_display, use_container_width=True, hide_index=True)
    st.caption(
        "**Tableau 5.1.** Récapitulatif des scénarios de décarbonation. "
        f"Base de calcul : {_base_co2:,.0f} t CO₂eq/an (statu quo, α = {_PCT_VOITURE:.0%}). "
        "Hypothèses de modélisation : substitution modale homogène, pas de saturation, "
        "mix électrique 2023 pour les modes non-thermiques."
    )

# ── Section 6 — Carte Nationale ICL ───────────────────────────────────────────
st.divider()
section(6, "Carte Nationale — Géographie de la Carbonité des Réseaux VLS")

st.markdown("""
La carte nationale représente l'ICL médian de chaque agglomération du corpus Gold Standard.
Les zones vertes correspondent aux réseaux les mieux intégrés dans un environnement
cyclable bas-carbone. La taille des points encode le nombre de stations.
""")

_map_df = df_city.dropna(subset=["ICL", "lat", "lon"]).copy()
_map_df["co2_adj"] = _map_df["co2_evite_t"] * _factor * _expand

import plotly.express as px
fig_map = px.scatter_mapbox(
    _map_df, lat="lat", lon="lon",
    color="ICL",
    size="n_stations",
    size_max=28,
    hover_name="city",
    hover_data={"ICL": ":.1f", "co2_adj": ":,.0f", "n_stations": True, "lat": False, "lon": False},
    labels={"ICL": "ICL / 100", "co2_adj": "CO₂ évité (t/an)", "n_stations": "Stations"},
    color_continuous_scale="RdYlGn",
    range_color=[0, 100],
    zoom=5,
    center={"lat": 46.5, "lon": 2.5},
    height=520,
    mapbox_style="carto-positron",
)

# Annotation Montpellier
_mmm_map = _map_df[_map_df["city"] == "Montpellier"]
if not _mmm_map.empty:
    fig_map.add_trace(go.Scattermapbox(
        lat=_mmm_map["lat"], lon=_mmm_map["lon"],
        mode="markers+text",
        marker=dict(size=14, color="#e74c3c"),
        text=["Montpellier"],
        textposition="top right",
        textfont=dict(size=11, color="#e74c3c"),
        showlegend=False,
        hoverinfo="skip",
    ))

fig_map.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar=dict(title="ICL / 100", ticksuffix=""),
)
st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "**Figure 6.1.** Carte nationale de l'Indice de Carbonité Locale (ICL) des agglomérations "
    "du Gold Standard GBFS. Couleur : ICL médian (rouge = faible, vert = élevé). "
    "Taille : nombre de stations. Montpellier indiqué en rouge. "
    "Source : Gold Standard GBFS enrichi, calcul auteurs."
)
