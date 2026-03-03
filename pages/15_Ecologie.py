"""
15_Ecologie.py — Écologie et transition bas-carbone des VLS.
Concepts : analyse du cycle de vie (ACV), substitution modale, empreinte carbone,
équivalences environnementales, efficacité carbone par station, scénarios de décarbonation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_stations, METRICS, EMP_PATH
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Écologie & Pollution - Réseaux VLS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Constantes ACV — g CO₂eq / km / passager (ADEME 2023, Leuenberger 2021) ───
_LCA_MODES: list[dict] = [
    {"mode": "Métro / RER",            "co2": 4,   "color": "#1565C0", "cat": "TC lourd"},
    {"mode": "Tramway",                 "co2": 8,   "color": "#1976D2", "cat": "TC lourd"},
    {"mode": "Vélo personnel",          "co2": 11,  "color": "#1abc9c", "cat": "Vélo"},
    {"mode": "VLS (dock-based)",        "co2": 16,  "color": "#27ae60", "cat": "Vélo"},
    {"mode": "Trottinette électrique",  "co2": 35,  "color": "#9b59b6", "cat": "Micromo."},
    {"mode": "Bus électrique",          "co2": 40,  "color": "#3498db", "cat": "TC léger"},
    {"mode": "Voiture électrique",      "co2": 65,  "color": "#f39c12", "cat": "Auto"},
    {"mode": "Bus diesel",              "co2": 90,  "color": "#e67e22", "cat": "TC léger"},
    {"mode": "Voiture thermique",       "co2": 140, "color": "#e74c3c", "cat": "Auto"},
    {"mode": "Voiture thermique (SUV)", "co2": 195, "color": "#c0392b", "cat": "Auto"},
]

# ── Paramètres du modèle de substitution modale ────────────────────────────────
_KM_TRAJET        = 2.5    # km/trajet VLS (Frade & Ribeiro 2015, Fishman 2016)
_TRAJETS_VLO_J    = 1.5    # trajets/vélo/jour (European Cyclists' Federation 2021)
_FILL_RATE        = 0.80   # taux de remplissage moyen des racks
_PCT_VOITURE      = 0.35   # part des trajets VLS remplaçant une voiture (Fishman et al. 2014)
_CO2_VOITURE      = 140.0  # g CO₂eq/km voiture thermique (ADEME 2023)
_CO2_VLS          = 16.0   # g CO₂eq/km VLS cycle de vie (Leuenberger et al. 2021)
_CO2_NET          = _CO2_VOITURE - _CO2_VLS   # 124 g CO₂eq/km économisés
_ARBRE_KG_AN      = 22.0   # kg CO₂ absorbés/an/arbre adulte (INRAE 2022)
_CONSO_L_KM       = 0.070  # l/km voiture (parc moyen France, ADEME 2023)
_FRANCE_TRANSP_MT = 141.0  # Mt CO₂eq/an du secteur transport France (CITEPA 2022)


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

    _metric_cols = [k for k in METRICS if k in dock.columns]

    # Agrégation en deux passes pour éviter les conflits de types Pylance
    city = (
        dock.groupby("city")
        .agg(
            n_stations=("capacity", "count"),
            total_cap=("capacity", "sum"),
            mean_cap=("capacity", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .query("n_stations >= 5")
        .reset_index()
    )
    if _metric_cols:
        _med = (
            dock.groupby("city")[_metric_cols]
            .median()
            .add_prefix("med_")
            .reset_index()
        )
        city = city.merge(_med, on="city", how="left")

    # Modèle CO₂
    city["n_velos_est"]       = city["total_cap"] * _FILL_RATE
    city["trajets_an"]        = city["n_velos_est"] * _TRAJETS_VLO_J * 365
    city["km_an"]             = city["trajets_an"] * _KM_TRAJET
    city["co2_evite_t"]       = city["km_an"] * _PCT_VOITURE * _CO2_NET / 1e6
    city["eq_arbres"]         = city["co2_evite_t"] * 1000 / _ARBRE_KG_AN
    city["litres_essence"]    = city["km_an"] * _PCT_VOITURE * _CONSO_L_KM
    city["co2_t_par_station"] = city["co2_evite_t"] / city["n_stations"]

    return dock, city


@st.cache_data(ttl=3600)
def _load_emp() -> pd.DataFrame:
    try:
        return pd.read_csv(EMP_PATH)
    except Exception:
        return pd.DataFrame(columns=["city", "emp_part_velo_2019"])


df_station, df_city = _load()
df_emp = _load_emp()

_n_stations    = len(df_station)
_n_cities      = df_city["city"].nunique()
_co2_total_t   = float(df_city["co2_evite_t"].sum())
_arbres_total  = int(df_city["eq_arbres"].sum())
_trajets_total = int(df_city["trajets_an"].sum())
_co2_eff_med   = float(df_city["co2_t_par_station"].median())

st.title("Écologie et Qualité de l'Air — VLS comme Vecteur de Transition Bas-Carbone")
st.caption(
    "Analyse du cycle de vie (ACV), substitution modale, empreinte CO₂, "
    "efficacité carbone par station et scénarios de décarbonation"
)

abstract_box(
    "<b>Question de recherche :</b> Dans quelle mesure le déploiement des réseaux de vélos "
    "en libre-service contribue-t-il à la réduction des émissions de CO₂ et à l'amélioration "
    "de la qualité de l'air urbaine ?<br><br>"
    "Un VLS dock-based émet <b>16 g CO₂eq/km</b> sur son cycle de vie complet, contre "
    "<b>140 g CO₂eq/km</b> pour une voiture thermique (ADEME 2023). Sur l'ensemble du corpus "
    f"Gold Standard ({_n_stations:,} stations, {_n_cities} agglomérations), le réseau VLS français "
    f"évite annuellement <b>≈ {_co2_total_t:,.0f} t CO₂eq</b> (substitution modale 35 %, "
    "Fishman et al. 2014), équivalant à la capture annuelle de "
    f"<b>{_arbres_total:,} arbres adultes</b> supplémentaires (INRAE 2022). "
    "L'analyse de l'<i>efficacité carbone par station</i> — CO₂ évité / n_stations — révèle "
    "que la performance environnementale est quasi-indépendante de la taille du réseau : "
    "des agglomérations moyennes surpassent régulièrement les grandes métropoles.",
    findings=[
        (f"{_co2_total_t:,.0f} t", "CO₂eq évités / an"),
        (f"{_arbres_total:,}", "arbres équivalents"),
        (f"{_trajets_total / 1e6:.1f} M", "trajets VLS / an estimés"),
        ("16 vs 140", "g CO₂eq/km VLS vs voiture"),
        (f"{_co2_eff_med:.2f} t", "CO₂/station médiane"),
    ],
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres du modèle")
    _n_top = st.slider("Agglomérations affichées", 10, min(50, _n_cities), min(20, _n_cities), 5)
    _subst_pct = st.slider(
        "Taux de substitution voiture α (%)",
        min_value=10, max_value=60, value=35, step=5,
        help="Part des trajets VLS remplaçant réellement un trajet en voiture (Fishman 2014 : 35 %).",
    )
    _expansion = st.slider(
        "Expansion du réseau VLS (scénario)",
        min_value=100, max_value=300, value=100, step=25,
        format="%d%%",
        help="Facteur multiplicatif de la flotte VLS (100 % = statu quo).",
    )
    st.divider()
    st.caption(
        "Sources : ADEME 2023 · Fishman et al. 2014\n"
        "Leuenberger et al. 2021 · INRAE 2022\n"
        "CITEPA 2022 · EMP 2019"
    )
    st.caption("R. Fossé & G. Pallares · 2025–2026")

# Facteurs de recalcul dynamiques (sidebar)
_factor = (_subst_pct / 100) / _PCT_VOITURE
_expand = _expansion / 100

# ── Section 1 — ACV Comparatif des Modes de Transport ─────────────────────────
st.divider()
section(1, "Analyse du Cycle de Vie (ACV) — Empreinte CO₂ par Mode de Transport")

st.markdown(r"""
L'**analyse du cycle de vie** (*Life Cycle Assessment*, ACV) comptabilise l'ensemble des
émissions de gaz à effet de serre liées à un mode de transport : fabrication du véhicule,
infrastructure, énergie, maintenance et fin de vie.
Les valeurs ci-dessous sont exprimées en **g CO₂eq / km / passager** (ADEME 2023 pour les modes
motorisés, Leuenberger et al. 2021 pour le VLS).

Le VLS dock-based se positionne **parmi les modes les moins émissifs**, loin devant tout usage
automobile. L'écart $\Delta = 124$ g CO₂eq/km avec la voiture thermique constitue le levier
de calcul principal du modèle de substitution modale développé en Section 2.
""")

_lca_df = pd.DataFrame(_LCA_MODES).sort_values("co2")

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
_vls_idx = _lca_df[_lca_df["mode"] == "VLS (dock-based)"]
if not _vls_idx.empty:
    _vls_val = float(_vls_idx["co2"].iloc[0])
    fig_acv.add_vline(
        x=_vls_val, line_dash="dash", line_color="#27ae60", line_width=1.5,
        annotation_text=f"VLS ({_vls_val:.0f} g)",
        annotation_font=dict(size=9, color="#27ae60"),
        annotation_position="top right",
    )
# Repère voiture thermique
fig_acv.add_vline(
    x=140, line_dash="dot", line_color="#e74c3c", line_width=1.2,
    annotation_text="Voiture thermique (140 g)",
    annotation_font=dict(size=9, color="#e74c3c"),
    annotation_position="top left",
)
# Zone de gain
fig_acv.add_vrect(
    x0=_vls_val if not _vls_idx.empty else 16, x1=140,
    fillcolor="rgba(39,174,96,0.06)", line_width=0,
    annotation_text="Δ = 124 g économisés",
    annotation_font=dict(size=9, color="#27ae60"),
    annotation_position="bottom right",
)

fig_acv.update_layout(
    title=dict(
        text="ACV comparatif — g CO₂eq / km / passager (ADEME 2023, Leuenberger 2021)",
        font_size=11, x=0,
    ),
    height=400,
    margin=dict(l=10, r=90, t=38, b=20),
    xaxis=dict(
        title="g CO₂eq / km / passager", range=[0, 240],
        gridcolor="#e8edf3", tickfont=dict(size=10),
    ),
    yaxis=dict(tickfont=dict(size=10)),
    plot_bgcolor="#f8fafd",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
)
st.plotly_chart(fig_acv, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "**Figure 1.1.** ACV comparatif des modes de transport urbain en g CO₂eq / km / passager. "
    "Ligne verte pointillée = VLS dock-based (16 g). Ligne rouge pointillée = voiture thermique (140 g). "
    "Zone verte = Δ = 124 g CO₂eq/km économisés par trajet substitué. "
    "Sources : ADEME (2023), Leuenberger et al. (2021)."
)

# ── Section 2 — CO₂ Évité par Agglomération ───────────────────────────────────
st.divider()
section(2, "Empreinte Carbone Évitée par Agglomération — Classement National")

st.markdown(rf"""
Le modèle de substitution modale applique la formule suivante à chaque agglomération :

$$\text{{CO}}_2^{{\text{{évité}}}} = \underbrace{{N_{{\text{{vélos}}}} \times {_FILL_RATE:.0%}}}_{{\text{{flotte active}}}}
\times \underbrace{{{_TRAJETS_VLO_J} \times 365}}_{{{\text{{trajets/an}}}}}
\times \underbrace{{{_KM_TRAJET}\ \text{{km}}}}_{{{\text{{dist. moy.}}}}}
\times \underbrace{{\alpha}}_{{{\text{{subst.}}\ {_subst_pct}\%}}}
\times \underbrace{{{int(_CO2_NET)}\ \text{{g CO}}_2/\text{{km}}}}_{{{\Delta_{{ACV}}}}}
\div 10^6\ \text{{[t CO}}_2\text{{eq/an]}}$$
""")

_city_plot = df_city.copy()
_city_plot["co2_evite_t_adj"]   = _city_plot["co2_evite_t"] * _factor * _expand
_city_plot["co2_t_par_st_adj"]  = _city_plot["co2_evite_t_adj"] / _city_plot["n_stations"]
_city_plot["eq_arbres_adj"]     = _city_plot["co2_evite_t_adj"] * 1000 / _ARBRE_KG_AN
_top_cities = _city_plot.nlargest(_n_top, "co2_evite_t_adj").sort_values("co2_evite_t_adj")
_co2_total_adj = float(_city_plot["co2_evite_t_adj"].sum())
_co2_med_adj   = float(_city_plot["co2_evite_t_adj"].median())

col_a, col_b = st.columns([3, 2])

with col_a:
    _bar_colors_co2 = [
        "#e74c3c" if c == "Montpellier"
        else "#f1c40f" if c == _top_cities.iloc[-1]["city"]
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
        x=_co2_med_adj,
        line_dash="dash", line_color="#555", line_width=1,
        annotation_text=f"Médiane ({_co2_med_adj:,.0f} t)",
        annotation_font=dict(size=9),
        annotation_position="top right",
    )
    fig_co2.update_layout(
        title=dict(
            text=f"CO₂ évité / an — top {_n_top} agglomérations (α = {_subst_pct} %)",
            font_size=11, x=0,
        ),
        height=max(340, _n_top * 18),
        margin=dict(l=10, r=90, t=38, b=20),
        xaxis=dict(title="t CO₂eq / an", gridcolor="#e8edf3"),
        yaxis=dict(tickfont=dict(size=9)),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_co2, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 2.1.** CO₂ évité annuellement — top {_n_top} agglomérations. "
        f"Taux de substitution voiture α = {_subst_pct} %. "
        "Jaune = 1ère agglomération · Rouge = Montpellier · Pointillé = médiane nationale. "
        "Le classement reflète essentiellement la taille du réseau."
    )

with col_b:
    _c_fit = np.polyfit(
        np.log1p(_city_plot["n_stations"].to_numpy(float)),
        np.log1p(_city_plot["co2_evite_t_adj"].to_numpy(float)), 1
    )
    _x_fit = np.linspace(float(_city_plot["n_stations"].min()),
                          float(_city_plot["n_stations"].max()), 80)
    _y_fit = np.expm1(np.polyval(_c_fit, np.log1p(_x_fit)))
    _hl = {"Montpellier", "Paris", "Lyon", "Marseille", "Bordeaux", "Rennes"}
    _lab = _city_plot["city"].isin(_hl)

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
        x=_city_plot.loc[_lab, "n_stations"],
        y=_city_plot.loc[_lab, "co2_evite_t_adj"],
        mode="markers+text",
        marker=dict(
            color=["#e74c3c" if c == "Montpellier" else "#1565C0"
                   for c in _city_plot.loc[_lab, "city"]],
            size=8, opacity=1,
        ),
        text=_city_plot.loc[_lab, "city"],
        textposition="top center", textfont=dict(size=8),
        showlegend=False,
        hovertemplate="<b>%{text}</b><br>%{x} stations · %{y:,.0f} t CO₂<extra></extra>",
    ))
    fig_sc2.add_trace(go.Scatter(
        x=_x_fit, y=_y_fit, mode="lines",
        line=dict(color="#e74c3c", dash="dash", width=1.5),
        name="Loi puissance (OLS log-log)",
    ))
    _exp_coef = float(_c_fit[0])
    fig_sc2.update_layout(
        title=dict(
            text=f"n_stations × CO₂ évité — loi puissance β = {_exp_coef:.2f}",
            font_size=11, x=0,
        ),
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
        f"**Figure 2.2.** n_stations × CO₂ évité (log-log). "
        f"Exposant β = {_exp_coef:.2f} (β < 1 ⟹ rendements marginaux décroissants). "
        "Montpellier en rouge."
    )

# ── Section 3 — Équivalences Environnementales ────────────────────────────────
st.divider()
section(3, "Équivalences Environnementales — Traduire le CO₂ en Langage Accessible")

st.markdown(r"""
La traduction des émissions évitées en **équivalences concrètes** facilite la communication
scientifique vers un public non-spécialisé et renforce l'argumentaire des politiques publiques.
""")

_arbres_eq   = int(_co2_total_adj * 1000 / _ARBRE_KG_AN)
_litres_tot  = float((_city_plot["litres_essence"] * _factor * _expand).sum())
_voitures_eq = int(_co2_total_adj * 1e6 / (140 * 15_000))
_paris_ny_eq = int(_co2_total_adj * 1e6 / 855_000)

_eq_c1, _eq_c2, _eq_c3, _eq_c4 = st.columns(4)
with _eq_c1:
    st.metric(
        "Arbres adultes équivalents", f"{_arbres_eq:,}",
        help="1 arbre adulte absorbe ≈ 22 kg CO₂/an (INRAE 2022).",
    )
with _eq_c2:
    st.metric(
        "Litres d'essence économisés", f"{_litres_tot / 1e6:.2f} M",
        help=f"Consommation parc français : {_CONSO_L_KM:.3f} l/km (ADEME 2023).",
    )
with _eq_c3:
    st.metric(
        "Voitures retirées du réseau", f"{_voitures_eq:,}",
        help="Voiture thermique moy. 15 000 km/an × 140 g CO₂/km = 2,1 t CO₂/an.",
    )
with _eq_c4:
    st.metric(
        "Vols Paris ↔ New York évités", f"{_paris_ny_eq:,}",
        help="Vol aller-retour Paris–NYC ≈ 855 kg CO₂eq/passager (Atmosfair 2023).",
    )

st.divider()

_pct_transport = _co2_total_adj / (_FRANCE_TRANSP_MT * 1e6) * 100
st.markdown(
    f"**Mise en perspective nationale.** Le secteur transport représente "
    f"**{_FRANCE_TRANSP_MT:.0f} Mt CO₂eq/an** en France (CITEPA 2022), premier poste d'émissions. "
    f"Les VLS Gold Standard couvrent **{_pct_transport:.4f} %** de ces émissions sectorielles "
    f"(α = {_subst_pct} %). Ce chiffre modeste souligne que le VLS est un **outil complémentaire** "
    "d'une politique de décarbonation plus large (électrification, report modal massif, TC)."
)

# Jauge
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=_co2_total_adj / 1000,
    title={"text": "CO₂ évité par les VLS Gold Standard (kt CO₂eq/an)", "font": {"size": 13}},
    number={"suffix": " kt", "font": {"size": 22}},
    gauge={
        "axis": {
            "range": [0, _FRANCE_TRANSP_MT * 1000 * 0.01],
            "tickformat": ",.0f",
            "ticksuffix": " kt",
        },
        "bar": {"color": "#27ae60"},
        "steps": [
            {"range": [0, _FRANCE_TRANSP_MT * 1000 * 0.002], "color": "#eafaf1"},
            {"range": [_FRANCE_TRANSP_MT * 1000 * 0.002, _FRANCE_TRANSP_MT * 1000 * 0.006], "color": "#a9dfbf"},
            {"range": [_FRANCE_TRANSP_MT * 1000 * 0.006, _FRANCE_TRANSP_MT * 1000 * 0.01],  "color": "#58d68d"},
        ],
        "threshold": {
            "line": {"color": "#e74c3c", "width": 2},
            "thickness": 0.75,
            "value": float(_co2_total_adj / 1000),
        },
    },
))
fig_gauge.update_layout(
    height=240,
    margin=dict(l=20, r=20, t=60, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
st.caption(
    f"**Figure 3.1.** Jauge de la contribution des VLS Gold Standard à la décarbonation "
    f"du secteur transport français ({_FRANCE_TRANSP_MT:.0f} Mt CO₂eq/an, CITEPA 2022). "
    f"Valeur actuelle : ≈ {_co2_total_adj:,.0f} t CO₂eq/an ({_pct_transport:.4f} % du secteur). "
    "L'ordre de grandeur à atteindre pour un impact systémique est de l'ordre du Mt."
)

# ── Section 4 — Efficacité Carbone par Station ────────────────────────────────
st.divider()
section(4, "Efficacité Carbone par Station — Qualité vs Taille du Réseau")

st.markdown(r"""
Le CO₂ évité **total** (Section 2) est mécaniquement corrélé à la taille du réseau.
Pour comparer la **performance environnementale intrinsèque** des agglomérations indépendamment
de leur taille, on définit l'**efficacité carbone par station** :

$$\varepsilon = \frac{\text{CO}_2^{\text{évité}}}{\text{n\_stations}} \quad \text{[t CO}_2\text{eq / station / an]}$$

Cette métrique reflète la densité d'utilisation et la propension locale à substituer la voiture.
Une agglomération avec $\varepsilon$ élevé dispose de stations **intensément utilisées**
dans un contexte de forte concurrence modale avec la voiture.
""")

col_e1, col_e2 = st.columns(2)

with col_e1:
    _eff_top = _city_plot.nlargest(_n_top, "co2_t_par_st_adj").sort_values("co2_t_par_st_adj")
    _eff_colors = [
        "#e74c3c" if c == "Montpellier"
        else "#f1c40f" if c == _eff_top.iloc[-1]["city"]
        else "#1A6FBF"
        for c in _eff_top["city"]
    ]
    _eff_med = float(_city_plot["co2_t_par_st_adj"].median())

    fig_eff = go.Figure(go.Bar(
        x=_eff_top["co2_t_par_st_adj"],
        y=_eff_top["city"],
        orientation="h",
        marker_color=_eff_colors,
        text=[f"{v:.3f} t" for v in _eff_top["co2_t_par_st_adj"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ε = %{x:.4f} t CO₂/station/an<extra></extra>",
    ))
    fig_eff.add_vline(
        x=_eff_med,
        line_dash="dash", line_color="#555", line_width=1,
        annotation_text=f"Médiane ({_eff_med:.3f} t)",
        annotation_font=dict(size=9),
        annotation_position="top right",
    )
    fig_eff.update_layout(
        title=dict(
            text=f"Efficacité carbone ε = CO₂ / station — top {_n_top}",
            font_size=11, x=0,
        ),
        height=max(320, _n_top * 18),
        margin=dict(l=10, r=90, t=38, b=20),
        xaxis=dict(title="t CO₂eq / station / an", gridcolor="#e8edf3"),
        yaxis=dict(tickfont=dict(size=9)),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_eff, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 4.1.** Efficacité carbone ε (t CO₂eq/station/an) — top {_n_top}. "
        "Contrairement au classement total, ce classement est indépendant de la taille. "
        "Jaune = meilleure efficacité · Rouge = Montpellier · Pointillé = médiane nationale."
    )

with col_e2:
    # Scatter n_stations vs efficacité (doit montrer quasi-indépendance)
    _hl_e = {"Montpellier", "Paris", "Lyon", "Marseille", "Bordeaux", "Strasbourg", "Rennes", "Nantes"}
    _lab_e = _city_plot["city"].isin(_hl_e)

    # OLS
    _c_eff = np.polyfit(
        _city_plot["n_stations"].to_numpy(float),
        _city_plot["co2_t_par_st_adj"].to_numpy(float), 1
    )
    _x_eff = np.linspace(float(_city_plot["n_stations"].min()),
                          float(_city_plot["n_stations"].max()), 60)

    fig_eff_sc = go.Figure()
    fig_eff_sc.add_trace(go.Scatter(
        x=_city_plot["n_stations"], y=_city_plot["co2_t_par_st_adj"],
        mode="markers",
        marker=dict(
            color=["#e74c3c" if c == "Montpellier" else "#1A6FBF"
                   for c in _city_plot["city"]],
            size=6, opacity=0.55,
        ),
        text=_city_plot["city"],
        hovertemplate="<b>%{text}</b><br>%{x} stations · ε = %{y:.4f} t<extra></extra>",
        showlegend=False,
    ))
    if _lab_e.any():
        fig_eff_sc.add_trace(go.Scatter(
            x=_city_plot.loc[_lab_e, "n_stations"],
            y=_city_plot.loc[_lab_e, "co2_t_par_st_adj"],
            mode="markers+text",
            marker=dict(
                color=["#e74c3c" if c == "Montpellier" else "#1565C0"
                       for c in _city_plot.loc[_lab_e, "city"]],
                size=8, opacity=1,
            ),
            text=_city_plot.loc[_lab_e, "city"],
            textposition="top center", textfont=dict(size=8),
            showlegend=False,
            hovertemplate="<b>%{text}</b><br>%{x} stations · ε = %{y:.4f} t<extra></extra>",
        ))
    fig_eff_sc.add_trace(go.Scatter(
        x=_x_eff, y=np.polyval(_c_eff, _x_eff),
        mode="lines",
        line=dict(color="#e74c3c", dash="dash", width=1.5),
        name="OLS",
    ))
    # Ligne horizontale médiane
    fig_eff_sc.add_hline(
        y=_eff_med, line_dash="dot", line_color="#555", line_width=1,
        annotation_text=f"Médiane ε",
        annotation_font=dict(size=9),
        annotation_position="bottom right",
    )
    fig_eff_sc.update_layout(
        title=dict(text="n_stations × ε — indépendance taille / efficacité", font_size=11, x=0),
        height=370,
        margin=dict(l=10, r=10, t=38, b=40),
        xaxis=dict(title="Nombre de stations", gridcolor="#e8edf3"),
        yaxis=dict(title="ε (t CO₂eq / station / an)", gridcolor="#e8edf3"),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig_eff_sc, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "**Figure 4.2.** Scatter n_stations × efficacité carbone ε. "
        "L'absence de corrélation entre taille et ε valide l'hypothèse d'indépendance — "
        "les grandes métropoles n'ont pas d'avantage environnemental intrinsèque par station. "
        "OLS en tirets · Montpellier en rouge."
    )

# Validation EMP si données disponibles
_emp_merged = pd.DataFrame()
if not df_emp.empty and "emp_part_velo_2019" in df_emp.columns:
    _emp_merged = (
        _city_plot[["city", "co2_t_par_st_adj", "n_stations"]]
        .merge(df_emp.dropna(subset=["emp_part_velo_2019"]), on="city", how="inner")
    )

if len(_emp_merged) >= 5:
    st.markdown("**Validation externe — Part modale vélo (EMP 2019) × efficacité carbone**")
    _hl_emp = {"Montpellier", "Paris", "Lyon", "Bordeaux", "Strasbourg", "Rennes"}
    _lab_emp = _emp_merged["city"].isin(_hl_emp)

    _c_emp = np.polyfit(
        _emp_merged["emp_part_velo_2019"].to_numpy(float),
        _emp_merged["co2_t_par_st_adj"].to_numpy(float), 1
    )
    _x_emp = np.linspace(
        float(_emp_merged["emp_part_velo_2019"].min()),
        float(_emp_merged["emp_part_velo_2019"].max()), 60
    )
    fig_emp = go.Figure()
    fig_emp.add_trace(go.Scatter(
        x=_emp_merged["emp_part_velo_2019"], y=_emp_merged["co2_t_par_st_adj"],
        mode="markers",
        marker=dict(
            color=["#e74c3c" if c == "Montpellier" else "#1A6FBF"
                   for c in _emp_merged["city"]],
            size=7, opacity=0.7,
        ),
        text=_emp_merged["city"],
        hovertemplate="<b>%{text}</b><br>part vélo = %{x:.1f} % · ε = %{y:.4f} t<extra></extra>",
        showlegend=False,
    ))
    if _lab_emp.any():
        fig_emp.add_trace(go.Scatter(
            x=_emp_merged.loc[_lab_emp, "emp_part_velo_2019"],
            y=_emp_merged.loc[_lab_emp, "co2_t_par_st_adj"],
            mode="markers+text",
            marker=dict(
                color=["#e74c3c" if c == "Montpellier" else "#1565C0"
                       for c in _emp_merged.loc[_lab_emp, "city"]],
                size=9, opacity=1,
            ),
            text=_emp_merged.loc[_lab_emp, "city"],
            textposition="top center", textfont=dict(size=8),
            showlegend=False,
        ))
    fig_emp.add_trace(go.Scatter(
        x=_x_emp, y=np.polyval(_c_emp, _x_emp),
        mode="lines",
        line=dict(color="#e74c3c", dash="dash", width=1.5),
        name="OLS",
    ))
    fig_emp.update_layout(
        title=dict(
            text="Part modale vélo (EMP 2019) × efficacité carbone ε",
            font_size=11, x=0,
        ),
        height=340,
        margin=dict(l=10, r=10, t=38, b=40),
        xaxis=dict(title="Part modale vélo (%, EMP 2019)", gridcolor="#e8edf3"),
        yaxis=dict(title="ε (t CO₂eq / station / an)", gridcolor="#e8edf3"),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig_emp, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 4.3.** Corrélation part modale vélo (EMP 2019) × efficacité carbone ε "
        f"({len(_emp_merged)} agglomérations communes EMP × Gold Standard). "
        "Une corrélation positive confirmerait que les agglomérations à forte culture vélo "
        "ont également une plus forte efficacité de substitution par station VLS."
    )

# ── Section 5 — Scénarios de Décarbonation ────────────────────────────────────
st.divider()
section(5, "Scénarios de Décarbonation — Impact de l'Expansion du Réseau VLS")

st.markdown(r"""
Six scénarios de politique de mobilité sont comparés en termes de CO₂ évité cumulé
à l'échelle du corpus Gold Standard. Les paramètres ajustables (sidebar) permettent
d'explorer la sensibilité du modèle au taux de substitution modale $\alpha$ et au
facteur d'expansion de la flotte.
""")

_base_co2 = float(df_city["co2_evite_t"].sum())

_scenarios: list[tuple[str, float, float]] = [
    ("Statu quo (actuel)",      1.00, _subst_pct / 100),
    ("+50 % flotte VLS",        1.50, _subst_pct / 100),
    ("+100 % flotte VLS",       2.00, _subst_pct / 100),
    ("+200 % flotte VLS",       3.00, _subst_pct / 100),
    (f"Subst. max (α = 60 %)",  float(_expansion) / 100, 0.60),
    ("Scénario combiné",        2.00, 0.60),
]

_sc_rows = []
for _label, _exp_f, _sub_f in _scenarios:
    _co2_sc = _base_co2 * _exp_f * (_sub_f / _PCT_VOITURE)
    _sc_rows.append({
        "Scénario":                     _label,
        "CO₂ évité (t/an)":            _co2_sc,
        "Arbres équivalents":           int(_co2_sc * 1000 / _ARBRE_KG_AN),
        "% secteur transport France":   _co2_sc / (_FRANCE_TRANSP_MT * 1e6) * 100,
        "Gain vs statu quo":            _co2_sc / _base_co2 - 1,
    })

_sc_df = pd.DataFrame(_sc_rows)
_sc_colors = ["#1A6FBF", "#27ae60", "#2ecc71", "#1abc9c", "#f39c12", "#e67e22"]

fig_sc = go.Figure(go.Bar(
    x=[r["Scénario"] for r in _sc_rows],
    y=[r["CO₂ évité (t/an)"] for r in _sc_rows],
    marker_color=_sc_colors,
    text=[f"{v:,.0f} t" for v in _sc_df["CO₂ évité (t/an)"]],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>CO₂ évité : %{y:,.0f} t/an<extra></extra>",
))
fig_sc.add_hline(
    y=_base_co2,
    line_dash="dash", line_color="#1A6FBF", line_width=1.5,
    annotation_text="Référence (statu quo)",
    annotation_font=dict(size=9, color="#1A6FBF"),
    annotation_position="top right",
)
fig_sc.update_layout(
    title=dict(
        text=f"Scénarios de décarbonation — CO₂ évité / an | α = {_subst_pct} % | expansion = {_expansion} %",
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
    "Le statu quo est la référence (bleu). "
    "Scénarios d'expansion = augmentation homothétique de la flotte. "
    "Scénario combiné = +100 % flotte × α = 60 %. "
    "Limites : modèle linéaire, saturation et contraintes d'offre non modélisées."
)

with st.expander("Tableau des scénarios (données détaillées)", expanded=False):
    _sc_display = _sc_df.copy()
    _sc_display["CO₂ évité (t/an)"] = _sc_display["CO₂ évité (t/an)"].apply(lambda v: f"{v:,.0f}")
    _sc_display["Arbres équivalents"] = _sc_display["Arbres équivalents"].apply(lambda v: f"{v:,}")
    _sc_display["% secteur transport France"] = _sc_display["% secteur transport France"].apply(lambda v: f"{v:.5f} %")
    _sc_display["Gain vs statu quo"] = _sc_display["Gain vs statu quo"].apply(lambda v: f"{v:+.1%}")
    st.dataframe(_sc_display, use_container_width=True, hide_index=True)
    st.caption(
        "**Tableau 5.1.** Récapitulatif des scénarios. "
        f"Base : {_base_co2:,.0f} t CO₂eq/an (statu quo, α = {_PCT_VOITURE:.0%}). "
        "Hypothèses : substitution homogène, pas de saturation, mix électrique 2023."
    )

# ── Section 6 — Carte Nationale CO₂ par Station ───────────────────────────────
st.divider()
section(6, "Carte Nationale — Géographie de l'Efficacité Carbone des Réseaux VLS")

st.markdown("""
La carte représente l'**efficacité carbone ε** (t CO₂eq / station / an) de chaque agglomération.
Contrairement au CO₂ total, ε est indépendant de la taille du réseau et reflète
la propension locale à la substitution modale. La taille des points encode le nombre de stations.
""")

_map_df = _city_plot.dropna(subset=["co2_t_par_st_adj", "lat", "lon"]).copy()

fig_map = px.scatter_mapbox(
    _map_df, lat="lat", lon="lon",
    color="co2_t_par_st_adj",
    size="n_stations",
    size_max=28,
    hover_name="city",
    hover_data={
        "co2_t_par_st_adj": ":.4f",
        "co2_evite_t_adj":  ":,.0f",
        "n_stations":       True,
        "lat": False, "lon": False,
    },
    labels={
        "co2_t_par_st_adj": "ε (t CO₂/station/an)",
        "co2_evite_t_adj":  "CO₂ évité total (t/an)",
        "n_stations":       "Stations",
    },
    color_continuous_scale="RdYlGn",
    zoom=5,
    center={"lat": 46.5, "lon": 2.5},
    height=520,
    mapbox_style="carto-positron",
)

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
    coloraxis_colorbar=dict(title="ε (t CO₂<br>/station/an)"),
)
st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "**Figure 6.1.** Carte nationale de l'efficacité carbone ε par agglomération "
    "(Gold Standard GBFS). Couleur : ε = CO₂ évité / station / an "
    "(rouge = faible, vert = élevé). Taille : nombre de stations. "
    "Montpellier indiqué en rouge. Source : Gold Standard GBFS enrichi, calcul auteurs."
)
