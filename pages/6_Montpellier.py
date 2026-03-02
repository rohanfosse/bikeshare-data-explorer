"""
6_Montpellier.py - Validation micro-locale : réseau Vélomagg (Montpellier Méditerranée Métropole).
Sources : TAM/MMM trips, station metrics, community detection, weather, socioeconomic IRIS.
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
from utils.data_loader import (
    compute_imd_cities,
    load_bike_tram_proximity,
    load_community_detection,
    load_dynamic_neighborhoods,
    load_hourly_flows,
    load_montpellier_stations,
    load_net_flows,
    load_network_topology,
    load_parts_modales,
    load_station_stress,
    load_station_temporal_profiles,
    load_station_vulnerability,
    load_stations,
    load_super_spreaders,
    load_synthese_velo_socio,
    load_top_quartiers,
    load_weather_data,
)
from utils.styles import abstract_box, inject_css, section, sidebar_nav

_VIZ_PATH = Path(__file__).parent.parent / "data" / "processed" / "ville_montpellier" / "visualisations"

st.set_page_config(
    page_title="Montpellier Vélomagg - Validation Micro-Locale",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Validation Micro-Locale : Réseau Vélomagg de Montpellier")
st.caption("Axe de Recherche 5 : Friction Spatiale, Écosystème Multimodal et Fracture Socio-Spatiale")

abstract_box(
    "<b>Question de recherche :</b> Les déséquilibres opérationnels source/puits du réseau "
    "Vélomagg s'expliquent-ils par la topographie et la multimodalité ? "
    "Les modèles nationaux (IMD, IES) se confirment-ils à l'échelle micro-locale ?<br><br>"
    "Étude de cas du réseau <b>Vélomagg</b> (TAM / Montpellier Méditerranée Métropole, "
    "IMD national = 86,8/100, rang #2) sur cinq axes : "
    "<em>topologie du graphe</em> (Louvain, PageRank, points critiques), "
    "<em>dynamiques temporelles</em> (régime bimodal domicile-travail), "
    "<em>friction spatiale</em> (source/puits, indice de vulnérabilité V_i), "
    "<em>intégration GTFS-tramway</em> (coût de correspondance piéton-vélo), "
    "et <em>fracture socio-spatiale</em> (IES par quartier IRIS). "
    "Les résultats valident à l'échelle fine les conclusions nationales.",
    findings=[
        ("53", "stations Vélomagg"),
        ("5 ans", "historique de courses"),
        ("4 lignes", "tramway GTFS intégré"),
        ("86,8/100", "IMD national"),
        ("#2", "rang national"),
    ],
)

# ── Chargement ────────────────────────────────────────────────────────────────
stations    = load_montpellier_stations()
profiles    = load_station_temporal_profiles()
stress      = load_station_stress()
hourly      = load_hourly_flows()
net_flows   = load_net_flows()
biketram    = load_bike_tram_proximity()
synthese    = load_synthese_velo_socio()
top_q, bot_q = load_top_quartiers()
community   = load_community_detection()
topo        = load_network_topology()
vulnerability = load_station_vulnerability()
weather     = load_weather_data()
modal       = load_parts_modales()
super_spreaders  = load_super_spreaders()
dyn_neighborhoods = load_dynamic_neighborhoods()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()

# ── KPIs globaux ──────────────────────────────────────────────────────────────
n_stations  = len(stations)
total_trips = int(profiles["Total_Trips"].sum(skipna=True)) if "Total_Trips" in profiles.columns else 0
avg_trips   = round(profiles["Total_Trips"].mean(), 0) if "Total_Trips" in profiles.columns else 0
top_station = profiles.loc[profiles["Total_Trips"].idxmax(), "Station_Name"] if "Total_Trips" in profiles.columns else "-"
n_quartiers = stations["Quartier"].nunique() if "Quartier" in stations.columns else "-"
pct_5min    = 100 * biketram["walkable_5min"].mean() if "walkable_5min" in biketram.columns else 0
n_bridge    = int(community["Is_Bridge"].sum()) if "Is_Bridge" in community.columns else "-"

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Stations analysées", f"{n_stations}")
k2.metric("Trips totaux (5 ans)", f"{total_trips:,}")
k3.metric("Trips moyens / station", f"{avg_trips:,.0f}")
k4.metric("Station la plus active", top_station)
k5.metric("Stations < 5 min d'un tram", f"{pct_5min:.1f} %")
k6.metric("Stations-pivots (bridge)", f"{n_bridge}")

# ── Encart IMD national ────────────────────────────────────────────────────────
try:
    _gs_df  = load_stations()
    _imd_df = compute_imd_cities(_gs_df)
    _imd_ranked = _imd_df.sort_values("IMD", ascending=False).reset_index(drop=True)
    if "Montpellier" in _imd_ranked["city"].values:
        _mmm_pos = int(_imd_ranked[_imd_ranked["city"] == "Montpellier"].index[0]) + 1
        _mmm_imd = float(_imd_ranked.loc[_imd_ranked["city"] == "Montpellier", "IMD"].iloc[0])
        _n_ranked = len(_imd_ranked)
        _top_city = _imd_ranked.iloc[0]["city"]
        st.success(
            f"**Montpellier - Rang IMD #{_mmm_pos}/{_n_ranked} National (Vélomagg) - "
            f"IMD = {_mmm_imd:.1f}/100**  \n"
            f"Le réseau Vélomagg se classe parmi les environnements cyclables les plus favorables "
            f"de France (rang #{_mmm_pos} sur {_n_ranked} agglomérations dock-based), "
            f"derrière {_top_city} (rang #1). "
            "Ce positionnement valide le choix de Montpellier comme cas d'étude micro-local "
            "pour l'analyse des déterminants de performance : l'écart entre sa position IMD "
            "et son niveau de revenu médian national traduit l'efficacité de la politique "
            "locale d'aménagement cyclable (IES > 1)."
        )
except Exception:
    pass

# ── Section 1 - Topologie du réseau et communautés ───────────────────────────
st.divider()
section(1, "Topologie du Réseau et Détection de Communautés - Structure du Graphe de Flux")

st.markdown(r"""
L'analyse topologique du graphe de flux orienté (stations comme nœuds, flux OD comme arêtes
pondérées) révèle la structure organisationnelle profonde du réseau Vélomagg.
La **détection de communautés de Louvain** (*Blondel et al., 2008*) identifie des
sous-ensembles de stations échangeant préférentiellement entre elles - des *bassins de
mobilité fonctionnels* indépendants des découpages administratifs.

Les **stations-pivots** (*bridge stations*, coefficient de participation $P_i > 0{,}5$)
assurent la connexion entre plusieurs communautés distinctes. Leur défaillance opérationnelle
(stock nul, vélo défectueux) fragilise la connectivité globale du réseau bien au-delà de
leur seule zone d'influence locale.
""")

col_map, col_community = st.columns([3, 2])

color_map_profile = {"Residential": "#4A9FDF", "Commuter": "#c0392b", "Mixed-Use": "#1A6FBF"}

map_df = stations.merge(
    profiles[["Station_Name", "Total_Trips", "Profile"]].rename(
        columns={"Station_Name": "Station_Name_p"}),
    left_on="Station_Name", right_on="Station_Name_p", how="left",
)
map_df["Total_Trips"] = map_df.get("Total_Trips_x", map_df.get("Total_Trips", 0)).fillna(0)

with col_map:
    fig_map = px.scatter_mapbox(
        map_df.dropna(subset=["latitude", "longitude"]),
        lat="latitude", lon="longitude",
        size="Total_Trips" if "Total_Trips" in map_df.columns else None,
        color="Profile" if "Profile" in map_df.columns else None,
        color_discrete_map=color_map_profile,
        hover_name="Station_Name",
        hover_data={"latitude": False, "longitude": False,
                    "Quartier": True, "Total_Trips": True},
        mapbox_style="carto-positron",
        zoom=12,
        center={"lat": 43.611, "lon": 3.876},
        size_max=20,
        height=450,
        labels={"Profile": "Profil", "Total_Trips": "Trips totaux"},
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        "**Figure 1.1.** Carte des stations Vélomagg. "
        "La taille encode le volume de trips historiques ($\\sum_t n_t$ sur 5 ans). "
        "La couleur encode la taxonomie des profils d'usage : "
        "<em>Commuter</em> (nœuds domicile-travail, structure bimodale) ; "
        "<em>Residential</em> (usage résidentiel de proximité) ; "
        "<em>Mixed-Use</em> (usage diffus, pas de structure temporelle dominante)."
    )

with col_community:
    if "Is_Bridge" in community.columns and "Community_Louvain" in community.columns:
        n_comm = community["Community_Louvain"].nunique()
        bridges = community[community["Is_Bridge"] == True][["Station_Name", "Participation_Coef", "Community_Louvain", "PageRank"]].sort_values("Participation_Coef", ascending=False)
        st.markdown(f"**{n_comm} communautés de Louvain** détectées · **{len(bridges)} stations-pivots**")
        st.dataframe(
            bridges.rename(columns={
                "Station_Name": "Station",
                "Participation_Coef": "Coef. participation",
                "Community_Louvain": "Communauté",
                "PageRank": "PageRank",
            }).head(10),
            use_container_width=True, hide_index=True,
            column_config={
                "Coef. participation": st.column_config.ProgressColumn(
                    "Coef. participation", min_value=0, max_value=1, format="%.3f"
                )
            }
        )
        st.caption(
            "**Tableau 1.1.** Stations-pivots par ordre décroissant du coefficient "
            "de participation $P_i \\in [0, 1]$. Un $P_i > 0{,}5$ indique "
            "qu'une station distribue ses flux vers plusieurs communautés distinctes, "
            "lui conférant un rôle structurel critique dans la connectivité du réseau."
        )

# Section 1 - suite : profils d'usage
left_prof, right_prof = st.columns([2, 3])

with left_prof:
    if "Profile" in profiles.columns:
        prof_counts = profiles["Profile"].value_counts().reset_index()
        prof_counts.columns = ["Profil", "Stations"]
        fig_prof = px.bar(
            prof_counts, x="Stations", y="Profil", orientation="h",
            color="Profil", color_discrete_map=color_map_profile,
            text="Stations", height=240,
        )
        fig_prof.update_traces(textposition="outside")
        fig_prof.update_layout(
            showlegend=False, plot_bgcolor="white",
            margin=dict(l=10, r=40, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_prof, use_container_width=True)
        st.caption("**Figure 1.2.** Répartition des stations par profil d'usage fonctionnel.")

with right_prof:
    if {"Profile", "Weekday_Departures", "Weekend_Departures"}.issubset(profiles.columns):
        bp_df = profiles[["Station_Name", "Profile",
                           "Weekday_Departures", "Weekend_Departures"]].dropna()
        bp_melt = bp_df.melt(
            id_vars=["Station_Name", "Profile"],
            value_vars=["Weekday_Departures", "Weekend_Departures"],
            var_name="Type", value_name="Departures",
        )
        bp_melt["Type"] = bp_melt["Type"].map(
            {"Weekday_Departures": "Semaine (Commuter)", "Weekend_Departures": "Week-end (Loisir)"}
        )
        fig_bp = px.box(
            bp_melt, x="Profile", y="Departures", color="Type",
            color_discrete_map={"Semaine (Commuter)": "#1A6FBF", "Week-end (Loisir)": "#4A9FDF"},
            labels={"Profile": "Profil fonctionnel", "Departures": "Départs", "Type": ""},
            height=280, notched=False,
        )
        fig_bp.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bp, use_container_width=True)
        st.caption(
            "**Figure 1.3.** Distribution des départs semaine versus week-end par profil. "
            "L'asymétrie semaine/week-end des profils Commuter valide la classification "
            "fonctionnelle par analyse de la série temporelle horaire."
        )

# ── Section 2 - Dynamiques temporelles et friction météorologique ─────────────
st.divider()
section(2, "Dynamiques Temporelles - Régimes Bimodaux, Flux OD et Friction Météorologique")

st.markdown(r"""
Les dynamiques temporelles du réseau Vélomagg révèlent un **régime bimodal** caractéristique
des mobilités pendulaires : un pic matinal ($h \approx 8\,\text{h}$) et un pic vespéral
($h \approx 17$–$18\,\text{h}$) concentrent l'essentiel du volume de trips journalier.
La **friction météorologique** - quantifiée par un score de mauvais temps normalisé de 0 à 1
(0 = temps dégagé, 1 = épisode pluvieux intense) - constitue un déterminant externe des
déséquilibres source/puits : les précipitations induisent une réduction significative des
départs, exacerbant les stocks dans les stations-puits et aggravant les pénuries dans les
stations-sources, ce qui complexifie la politique de redistribution.
""")

tab_hourly, tab_weather = st.tabs(["Distribution Horaire des Flux", "Friction Météorologique"])

with tab_hourly:
    left_hr, right_hr = st.columns(2)

    with left_hr:
        if "total_trips" in hourly.columns:
            fig_hr = px.bar(
                hourly, x="hour", y="total_trips",
                color="total_trips", color_continuous_scale="Blues",
                labels={"hour": "Heure (0–23 h)", "total_trips": "Volume de trips"},
                height=300,
            )
            fig_hr.update_layout(
                coloraxis_showscale=False, plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(dtick=2),
            )
            st.plotly_chart(fig_hr, use_container_width=True)
            st.caption(
                "**Figure 2.1.** Distribution horaire agrégée du volume de trips "
                "($\\sum_j n_j(h)$ sur l'ensemble de l'historique). "
                "La structure bimodale traduit la dominance des mobilités pendulaires "
                "domicile-travail, cohérente avec la classification des profils de stations."
            )

    with right_hr:
        if "unique_od_pairs" in hourly.columns:
            fig_od = px.line(
                hourly, x="hour", y="unique_od_pairs",
                labels={"hour": "Heure (0–23 h)", "unique_od_pairs": "Paires OD actives"},
                height=300, markers=True,
                color_discrete_sequence=["#1A6FBF"],
            )
            fig_od.update_layout(
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(dtick=2),
            )
            st.plotly_chart(fig_od, use_container_width=True)
            st.caption(
                "**Figure 2.2.** Cardinalité de l'ensemble des paires origine-destination "
                "actives par tranche horaire, révélant la diversité des flux "
                "et la robustesse topologique du réseau sous contrainte temporelle."
            )

with tab_weather:
    if not weather.empty and "bad_weather_score" in weather.columns and "hour" in weather.columns:
        season_labels = {1: "Hiver", 2: "Printemps", 3: "Été", 4: "Automne"}

        col_w1, col_w2 = st.columns(2)

        with col_w1:
            if "season" in weather.columns and "is_raining" in weather.columns:
                rain_by_season = weather.groupby("season")["is_raining"].mean().reset_index()
                rain_by_season["Saison"] = rain_by_season["season"].map(season_labels)
                rain_by_season["Taux de précipitation (%)"] = (rain_by_season["is_raining"] * 100).round(1)

                fig_rain = px.bar(
                    rain_by_season, x="Saison", y="Taux de précipitation (%)",
                    color="Taux de précipitation (%)",
                    color_continuous_scale="Blues_r",
                    text="Taux de précipitation (%)",
                    height=300,
                    labels={"Saison": "Saison", "Taux de précipitation (%)": "Heures pluvieuses (%)"},
                )
                fig_rain.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
                fig_rain.update_layout(
                    plot_bgcolor="white", coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=30),
                )
                st.plotly_chart(fig_rain, use_container_width=True)
                st.caption(
                    "**Figure 2.3.** Taux horaire de précipitations par saison (Montpellier, 2021–2024). "
                    "L'été méditerranéen, quasi exempt de précipitations, favorise les pics d'usage "
                    "touristique non pendulaire."
                )

        with col_w2:
            # Score météo moyen par heure × trips horaires (corrélation friction)
            if not hourly.empty and "total_trips" in hourly.columns:
                _wx_hr = (
                    weather.groupby("hour")["bad_weather_score"]
                    .mean()
                    .reset_index()
                    .rename(columns={"bad_weather_score": "bws_mean"})
                )
                _wx_merged = hourly[["hour", "total_trips"]].merge(_wx_hr, on="hour", how="inner")
                if not _wx_merged.empty:
                    # Droite OLS manuelle
                    _wx_x = _wx_merged["bws_mean"].values
                    _wx_y = _wx_merged["total_trips"].values
                    _wx_coef = np.polyfit(_wx_x, _wx_y, 1)
                    _wx_xr = np.linspace(_wx_x.min(), _wx_x.max(), 100)

                    # Corrélation de Spearman
                    _wx_rho = float(
                        pd.Series(_wx_x).rank().corr(pd.Series(_wx_y).rank())
                    )

                    fig_wx = px.scatter(
                        _wx_merged,
                        x="bws_mean", y="total_trips",
                        text="hour",
                        labels={
                            "bws_mean":    "Score de mauvais temps moyen (0 = beau, 1 = pluie forte)",
                            "total_trips": "Volume moyen de trips",
                        },
                        height=300,
                        color="total_trips",
                        color_continuous_scale="RdBu",
                    )
                    fig_wx.add_trace(go.Scatter(
                        x=_wx_xr, y=np.polyval(_wx_coef, _wx_xr),
                        mode="lines",
                        line=dict(color="#1A2332", dash="dash", width=2),
                        name=f"Tendance (ρ = {_wx_rho:+.2f})",
                        showlegend=True,
                    ))
                    fig_wx.update_traces(
                        selector=dict(mode="markers+text"),
                        textposition="top center",
                        textfont=dict(size=8),
                        marker_size=9,
                    )
                    fig_wx.update_layout(
                        plot_bgcolor="white",
                        coloraxis_showscale=False,
                        margin=dict(l=10, r=10, t=10, b=30),
                        legend=dict(orientation="h", y=-0.25),
                    )
                    st.plotly_chart(fig_wx, use_container_width=True)
                    st.caption(
                        f"**Figure 2.4.** Score de mauvais temps moyen (axe x) versus "
                        f"volume de trips (axe y) par heure 0–23 h. "
                        f"ρ de Spearman = {_wx_rho:+.2f} : la friction météorologique "
                        "présente une corrélation négative avec le volume de trips, "
                        "confirmant l'effet dissuasif des précipitations sur l'usage du VLS."
                    )

        # Profil temperature × trips : scatter optionnel
        if "temperature" in weather.columns and not hourly.empty and "total_trips" in hourly.columns:
            _temp_hr = (
                weather.groupby("hour")["temperature"]
                .mean().reset_index().rename(columns={"temperature": "temp_mean"})
            )
            _temp_merged = hourly[["hour", "total_trips"]].merge(_temp_hr, on="hour", how="inner")
            if not _temp_merged.empty:
                with st.expander("Corrélation Température × Volume de Trips (par heure)"):
                    _tc = np.polyfit(_temp_merged["temp_mean"].values, _temp_merged["total_trips"].values, 1)
                    _txr = np.linspace(_temp_merged["temp_mean"].min(), _temp_merged["temp_mean"].max(), 100)
                    _t_rho = float(
                        pd.Series(_temp_merged["temp_mean"].values).rank()
                        .corr(pd.Series(_temp_merged["total_trips"].values).rank())
                    )
                    fig_temp = px.scatter(
                        _temp_merged,
                        x="temp_mean", y="total_trips",
                        text="hour",
                        color="total_trips",
                        color_continuous_scale="RdYlGn",
                        labels={
                            "temp_mean": "Température moyenne (°C)",
                            "total_trips": "Volume moyen de trips",
                        },
                        height=300,
                    )
                    fig_temp.add_trace(go.Scatter(
                        x=_txr, y=np.polyval(_tc, _txr),
                        mode="lines",
                        line=dict(color="#c0392b", dash="dash", width=1.5),
                        name=f"Tendance (ρ = {_t_rho:+.2f})",
                        showlegend=True,
                    ))
                    fig_temp.update_traces(
                        selector=dict(mode="markers+text"),
                        textposition="top center", textfont=dict(size=8),
                        marker_size=9,
                    )
                    fig_temp.update_layout(
                        plot_bgcolor="white", coloraxis_showscale=False,
                        margin=dict(l=10, r=10, t=10, b=30),
                        legend=dict(orientation="h", y=-0.25),
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
                    st.caption(
                        f"**Figure 2.5.** Température moyenne (°C) versus volume de trips "
                        f"par tranche horaire. ρ de Spearman = {_t_rho:+.2f}. "
                        "Les heures les plus chaudes (14–16 h) cumulent les températures "
                        "maximales mais pas nécessairement le volume le plus élevé, "
                        "reflétant l'effet de chaleur dissuasif en journée."
                    )
    else:
        st.info(
            "Les données météorologiques enrichies sont nécessaires pour cette analyse. "
            "Vérifiez la présence de `weather_data_enriched.csv` dans `data/processed/`."
        )

# ── Section 3 - Friction spatiale : déséquilibres et vulnérabilité ─────────────
st.divider()
section(3, "Modélisation de la Friction Spatiale - Déséquilibres Source/Puits et Vulnérabilité Structurelle")

st.markdown(r"""
Le flux net $f_i = \sum_t (\text{entrées}_t - \text{sorties}_t)$ quantifie la
<strong>friction opérationnelle</strong> de chaque station. Un flux net positif signale
une station <em>puits</em> (attractrice nette de vélos) ; un flux net négatif signale
une station <em>source</em> (génératrice nette de départs). Ces déséquilibres sont
intrinsèquement liés à la **friction topographique** : les stations situées en altitude
tendent à être des sources (les cyclistes descendent), tandis que les stations en bas
de pente tendent à être des puits.

L'**indice de vulnérabilité structurelle** ($V_i$) agrège trois indicateurs pour
identifier les stations dont la défaillance impacterait le plus fortement le réseau :
la centralité de betweenness ($b_i$, importance structurelle dans le graphe de flux),
la population dans le rayon de 300 m ($P_{300m}$, exposition de la demande potentielle)
et le coefficient de clustering ($C_i$, mesure de l'intégration locale).
""")

col_net, col_vuln = st.columns(2)

with col_net:
    if {"station", "net_flow"}.issubset(net_flows.columns):
        net_agg = (
            net_flows.groupby("station")["net_flow"]
            .mean().reset_index()
            .sort_values("net_flow")
        )
        n_show = st.slider("Stations aux extrêmes à afficher", 10, 40, 20, 5, key="net_slider")
        half   = n_show // 2
        extremes = pd.concat([net_agg.head(half), net_agg.tail(half)]).drop_duplicates()

        fig_net = px.bar(
            extremes, x="net_flow", y="station",
            orientation="h",
            color="net_flow",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"station": "Station", "net_flow": "Flux net moyen $\\bar{f}_i$"},
            height=max(350, len(extremes) * 22),
        )
        fig_net.add_vline(x=0, line_width=1.5, line_color="#1A2332")
        fig_net.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            margin=dict(l=10, r=60, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption(
            "**Figure 3.1.** Flux nets moyens $\\bar{f}_i$ par station (entrées $-$ sorties). "
            "Bleu : station <em>puits</em> (attractrice nette) ; "
            "Rouge : station <em>source</em> (génératrice nette). "
            "Les stations aux valeurs extrêmes constituent les cibles prioritaires "
            "de la politique de redistribution (<em>rebalancing</em>)."
        )

with col_vuln:
    if "Vulnerability_Index" in vulnerability.columns:
        top_vuln = vulnerability.nlargest(15, "Vulnerability_Index")[
            ["Station_Name", "Vulnerability_Index", "Betweenness", "Population_300m"]
        ]
        fig_vuln = px.bar(
            top_vuln, x="Vulnerability_Index", y="Station_Name",
            orientation="h",
            color="Vulnerability_Index", color_continuous_scale="OrRd",
            labels={"Station_Name": "Station", "Vulnerability_Index": "Indice de vulnérabilité $V_i$"},
            height=max(350, 15 * 24),
        )
        fig_vuln.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            margin=dict(l=10, r=60, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_vuln, use_container_width=True)
        st.caption(
            "**Figure 3.2.** Top 15 stations par indice de vulnérabilité structurelle $V_i$. "
            "L'indice agrège la centralité de betweenness $b_i$ (importance dans le graphe), "
            "la population exposée $P_{300m}$ (demande potentielle) et le coefficient "
            "de clustering $C_i$ (mesure de l'intégration locale). "
            "Ces stations constituent les <em>points de fragilité critiques</em> "
            "du réseau, dont la défaillance impacte le plus fortement la connectivité globale."
        )

# Centralité : PageRank vs Trips
st.markdown("#### Centralité Structurelle - PageRank versus Volume de Trips")
st.markdown(r"""
Le PageRank ($PR_i$) mesure l'importance structurelle d'une station dans le graphe de flux
orienté, indépendamment de son volume absolu de trips. Une station à fort $PR_i$ mais faible
volume constitue un **nœud pivot** dont la défaillance présente un risque élevé pour
la connectivité globale, même si son activité observée semble modeste.
""")

if {"PageRank", "Total_Trips", "Station_Name"}.issubset(stations.columns):
    fig_pr = px.scatter(
        stations.dropna(subset=["PageRank", "Total_Trips"]),
        x="PageRank", y="Total_Trips",
        hover_name="Station_Name",
        color="Quartier" if "Quartier" in stations.columns else None,
        labels={"PageRank": "PageRank ($PR_i$)", "Total_Trips": "Volume de trips totaux"},
        height=360, opacity=0.75,
    )
    fig_pr.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_pr, use_container_width=True)
    st.caption(
        "**Figure 3.3.** PageRank ($PR_i$) versus volume de trips totaux par station. "
        "Les stations en haut à gauche (fort $PR_i$, faible volume) "
        "sont des nœuds pivots structurellement critiques. "
        "Les stations en bas à droite présentent un volume élevé mais une centralité "
        "moindre dans le graphe de flux orienté."
    )

# Scatter : anatomie du V_i (Betweenness × Population)
st.markdown("#### Anatomie de l'Indice de Vulnérabilité — Betweenness × Population exposée")
st.markdown(r"""
Le nuage ci-dessous décompose l'indice $V_i$ en ses deux composantes principales.
Les stations en **haut à droite** (fort betweenness ET forte population) concentrent le
risque systémique maximal : leur défaillance coupe des flux structuraux et prive
une large zone de desserte. La taille encode $V_i$ ; la couleur encode le clustering $C_i$.
""")

if {"Vulnerability_Index", "Betweenness", "Population_300m", "Clustering"}.issubset(vulnerability.columns):
    _vuln_sc = vulnerability.dropna(
        subset=["Vulnerability_Index", "Betweenness", "Population_300m", "Clustering"]
    ).copy()
    # Nom court pour le hover
    _vuln_sc["label"] = _vuln_sc["Station_Name"].str.replace(r"^\S+\s+", "", regex=True).str[:25]

    fig_vi = px.scatter(
        _vuln_sc,
        x="Betweenness",
        y="Population_300m",
        size="Vulnerability_Index",
        color="Clustering",
        hover_name="Station_Name",
        size_max=35,
        color_continuous_scale="RdYlGn_r",
        labels={
            "Betweenness":        "Centralité de betweenness $b_i$",
            "Population_300m":    "Population exposée (300 m)",
            "Vulnerability_Index":"Indice V_i",
            "Clustering":         "Coeff. clustering $C_i$",
        },
        height=380,
        opacity=0.85,
    )
    # Annoter les 5 stations les plus vulnérables
    _top5_vi = _vuln_sc.nlargest(5, "Vulnerability_Index")
    for _, _r in _top5_vi.iterrows():
        fig_vi.add_annotation(
            x=_r["Betweenness"], y=_r["Population_300m"],
            text=_r["label"],
            showarrow=True, arrowhead=2, arrowwidth=1,
            arrowcolor="#1A2332", ax=30, ay=-20,
            font=dict(size=8),
        )
    fig_vi.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    st.plotly_chart(fig_vi, use_container_width=True)
    st.caption(
        "**Figure 3.4.** Betweenness ($b_i$, axe x) versus population exposée à 300 m (axe y). "
        "La **taille** encode l'indice de vulnérabilité $V_i$ ; "
        "la **couleur** encode le coefficient de clustering $C_i$ "
        "(rouge = station peu intégrée localement, donc plus fragile). "
        "Les 5 stations les plus vulnérables sont annotées. "
        "La quadrant supérieur droit identifie les nœuds à risque systémique maximal."
    )

# ── Section 4 - Écosystème multimodal ─────────────────────────────────────────
st.divider()
section(4, "Écosystème Multimodal - Intégration GTFS Tramway et Coût de Correspondance")

st.markdown(r"""
L'efficacité du réseau Vélomagg comme solution du **premier/dernier kilomètre** repose sur
son intégration à l'écosystème de transport lourd TAM (4 lignes de tramway, fréquence
$\sim 5$–$7$ min en heure de pointe). Le seuil d'interopérabilité confortable est fixé
à **400 m** ($\approx 5$ min à pied), standard UITP pour la correspondance multimodale.
Au-delà de **800 m** ($\approx 10$ min), le coût piéton devient dissuasif et la
complémentarité modale se dissout.

La distance à l'arrêt de tramway le plus proche constitue l'une des trois composantes de
la **friction spatiale totale** - aux côtés de la rugosité topographique et de la
friction météorologique - qui structurent collectivement les déséquilibres source/puits
du réseau et conditionnent l'efficacité opérationnelle du rebalancing.
""")

if {"bike_station", "distance_m", "walkable_5min", "walkable_10min"}.issubset(biketram.columns):
    closest = (
        biketram.groupby("bike_station")["distance_m"]
        .min().reset_index()
        .rename(columns={"distance_m": "dist_tram_min_m"})
    )
    closest["< 5 min (400 m)"]  = closest["dist_tram_min_m"] <= 400
    closest["5–10 min (800 m)"] = (closest["dist_tram_min_m"] > 400) & (closest["dist_tram_min_m"] <= 800)
    closest["> 10 min"]         = closest["dist_tram_min_m"] > 800

    pct_5  = 100 * closest["< 5 min (400 m)"].mean()
    pct_10 = 100 * closest["5–10 min (800 m)"].mean()
    pct_gt = 100 * closest["> 10 min"].mean()

    ml1, ml2, ml3 = st.columns(3)
    ml1.metric("Stations < 5 min d'un arrêt tram", f"{pct_5:.1f} %", help="Seuil UITP d'interopérabilité confortable")
    ml2.metric("Stations entre 5 et 10 min", f"{pct_10:.1f} %")
    ml3.metric("Stations > 10 min (friction élevée)", f"{pct_gt:.1f} %", delta=f"{pct_gt:.0f}% hors zone d'intégration", delta_color="inverse")

    fig_dist = px.histogram(
        closest, x="dist_tram_min_m", nbins=30,
        color_discrete_sequence=["#1A6FBF"],
        labels={"dist_tram_min_m": "Distance à l'arrêt tram le plus proche (m)", "count": "Stations"},
        height=300,
    )
    fig_dist.add_vline(
        x=400, line_dash="dash", line_color="#c0392b",
        annotation_text="Seuil UITP 5 min", annotation_position="top right",
    )
    fig_dist.add_vline(
        x=800, line_dash="dash", line_color="#e67e22",
        annotation_text="Seuil max 10 min", annotation_position="top right",
    )
    fig_dist.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption(
        "**Figure 4.1.** Distribution de la distance piétonne entre stations Vélomagg "
        "et l'arrêt de tramway TAM le plus proche. "
        "Les lignes pointillées indiquent les seuils UITP : 400 m (5 min, "
        "interopérabilité confortable) et 800 m (10 min, limite de l'écosystème intégré). "
        f"**{pct_5:.0f} %** des stations sont dans la zone d'intégration optimale (< 5 min) ; "
        f"**{pct_gt:.0f} %** présentent une friction multimodale élevée (> 10 min)."
    )

    # Scatter : distance tram × volume de trips
    st.markdown("#### Friction Multimodale × Usage — Distance au Tram versus Volume de Trips")
    st.markdown(r"""
    La relation entre la distance au tram et le volume de trips quantifie directement
    l'**effet de friction multimodale** : les stations éloignées des correspondances
    tramway perdent-elles des usagers au profit d'autres modes ?
    """)

    _trips_ref = profiles[["Station_Name", "Total_Trips", "Profile"]].dropna(subset=["Total_Trips"])
    _dist_trips = closest.merge(
        _trips_ref.rename(columns={"Station_Name": "bike_station"}),
        on="bike_station", how="inner",
    ).dropna(subset=["dist_tram_min_m", "Total_Trips"])

    if not _dist_trips.empty:
        _dt_x = _dist_trips["dist_tram_min_m"].values
        _dt_y = _dist_trips["Total_Trips"].values
        _dt_coef = np.polyfit(_dt_x, _dt_y, 1)
        _dt_xr   = np.linspace(_dt_x.min(), _dt_x.max(), 100)
        _dt_rho  = float(
            pd.Series(_dt_x).rank().corr(pd.Series(_dt_y).rank())
        )

        _dist_trips["Zone"] = pd.cut(
            _dist_trips["dist_tram_min_m"],
            bins=[0, 400, 800, float("inf")],
            labels=["< 5 min (optimal)", "5–10 min", "> 10 min (friction)"],
        ).astype(str)
        _zone_colors = {
            "< 5 min (optimal)": "#27ae60",
            "5–10 min":          "#e67e22",
            "> 10 min (friction)": "#c0392b",
        }

        fig_dt = px.scatter(
            _dist_trips,
            x="dist_tram_min_m", y="Total_Trips",
            color="Zone",
            color_discrete_map=_zone_colors,
            symbol="Profile" if "Profile" in _dist_trips.columns else None,
            hover_name="bike_station",
            hover_data={"dist_tram_min_m": ":.0f m", "Total_Trips": ":,.0f", "Zone": False},
            labels={
                "dist_tram_min_m": "Distance à l'arrêt tram le plus proche (m)",
                "Total_Trips":     "Volume de trips totaux (5 ans)",
                "Zone":            "Zone d'intégration",
                "Profile":         "Profil",
            },
            height=380,
            opacity=0.8,
        )
        fig_dt.add_trace(go.Scatter(
            x=_dt_xr, y=np.polyval(_dt_coef, _dt_xr),
            mode="lines",
            line=dict(color="#1A2332", dash="dash", width=2),
            name=f"Tendance OLS (ρ = {_dt_rho:+.2f})",
            showlegend=True,
        ))
        fig_dt.add_vline(x=400, line_dash="dot", line_color="#27ae60", line_width=1.5)
        fig_dt.add_vline(x=800, line_dash="dot", line_color="#e67e22", line_width=1.5)
        fig_dt.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_dt, use_container_width=True)
        st.caption(
            f"**Figure 4.2.** Distance au tramway le plus proche (axe x) versus "
            f"volume de trips totaux (axe y). ρ de Spearman = {_dt_rho:+.2f}. "
            "La couleur encode la zone d'intégration UITP ; le symbole encode le profil fonctionnel. "
            "Une corrélation négative confirme l'**effet friction multimodale** : "
            "les stations éloignées des correspondances tramway génèrent structurellement "
            "moins de trips, indépendamment de leur capacité."
        )

# ── Section 5 - Fracture socio-spatiale ───────────────────────────────────────
st.divider()
section(5, "Fracture Socio-Spatiale - Usage du Vélo et Profil Socio-Économique par Quartier")

st.markdown("""
Cette section confronte les données d'usage du réseau Vélomagg aux indicateurs
socio-économiques des quartiers de Montpellier (IRIS, INSEE Filosofi) pour tester
l'hypothèse d'un **biais socio-économique** dans l'adoption de la micromobilité partagée.
La question centrale est la suivante : les ménages précaires sous-utilisent-ils le VLS
en raison d'inégalités d'infrastructure (offre absente ou éloignée) ou de barrières
d'usage structurelles (coût, habitudes de mobilité, accessibilité culturelle) ?

Les quartiers cumulant revenu faible et usage faible constituent formellement des
**déserts de mobilité sociale** au sens de l'IES - la micro-validation de la
conclusion nationale ρ(IMD, revenu) ≈ 0 à l'échelle des quartiers montpelliérains.
""")

tab_viz, tab_scatter = st.tabs(["Figures Pré-calculées (Analyse IRIS)", "Analyse Interactive"])

with tab_viz:
    _syn = synthese.copy() if not synthese.empty else pd.DataFrame()

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        # Fig 5.1 — Revenu × Part vélo (scatter interactif)
        _req1 = {"nom", "revenu_fiscal_moyen_menage", "transport_deux_roues_velo_pct"}
        if _req1.issubset(_syn.columns):
            _s1 = _syn.dropna(subset=["revenu_fiscal_moyen_menage", "transport_deux_roues_velo_pct"])
            _r1x = _s1["revenu_fiscal_moyen_menage"].values
            _r1y = _s1["transport_deux_roues_velo_pct"].values
            _c1  = np.polyfit(_r1x, _r1y, 1)
            _xr1 = np.linspace(_r1x.min(), _r1x.max(), 100)
            _rho1 = float(pd.Series(_r1x).rank().corr(pd.Series(_r1y).rank()))

            fig_v1 = px.scatter(
                _s1,
                x="revenu_fiscal_moyen_menage",
                y="transport_deux_roues_velo_pct",
                text="nom",
                color="transport_voiture_camion_pct" if "transport_voiture_camion_pct" in _s1.columns else None,
                color_continuous_scale="RdBu_r",
                size="equipement_pas_de_voiture_pct" if "equipement_pas_de_voiture_pct" in _s1.columns else None,
                size_max=20,
                labels={
                    "revenu_fiscal_moyen_menage":    "Revenu fiscal médian (€/an)",
                    "transport_deux_roues_velo_pct": "Part modale vélo/deux-roues (%)",
                    "transport_voiture_camion_pct":  "Part voiture (%)",
                    "equipement_pas_de_voiture_pct": "% ménages sans voiture",
                },
                height=320,
                opacity=0.85,
            )
            fig_v1.add_trace(go.Scatter(
                x=_xr1, y=np.polyval(_c1, _xr1),
                mode="lines",
                line=dict(color="#1A2332", dash="dash", width=2),
                name=f"Tendance OLS (ρ = {_rho1:+.2f})",
                showlegend=True,
            ))
            fig_v1.update_traces(
                selector=dict(mode="markers+text"),
                textposition="top center", textfont=dict(size=8),
            )
            fig_v1.update_layout(
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.22),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_v1, use_container_width=True)
            st.caption(
                f"**Figure 5.1.** Revenu fiscal médian (axe x) versus part modale vélo (axe y) "
                f"par quartier IRIS. ρ Spearman = {_rho1:+.2f}. "
                "La couleur encode la part modale voiture ; "
                "la taille encode le % de ménages sans voiture. "
                "La relation positive revenu–vélo caractérise le biais socio-économique "
                "structurel de la micromobilité partagée à Montpellier."
            )
        else:
            _img = _VIZ_PATH / "revenu_vs_velo.png"
            if _img.exists():
                st.image(str(_img), use_container_width=True)

    with col_v2:
        # Fig 5.2 — Parts modales (donut interactif)
        if not modal.empty:
            _modal_clean = modal.reset_index()
            _modal_clean.columns = ["Mode", "Part (%)"]
            _modal_colors = {
                "Marche a pied":       "#27ae60",
                "Velo/Deux-roues":     "#1A6FBF",
                "Voiture":             "#c0392b",
                "Transport en commun": "#8e44ad",
            }
            fig_v2 = go.Figure(go.Pie(
                labels=_modal_clean["Mode"],
                values=_modal_clean["Part (%)"].round(1),
                hole=0.45,
                marker_colors=[_modal_colors.get(m, "#999") for m in _modal_clean["Mode"]],
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value:.1f} %<extra></extra>",
            ))
            fig_v2.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=10),
                annotations=[dict(
                    text="Parts<br>modales", x=0.5, y=0.5,
                    font_size=12, showarrow=False,
                )],
                showlegend=True,
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_v2, use_container_width=True)
            _voit_pct = float(_modal_clean.loc[_modal_clean["Mode"] == "Voiture", "Part (%)"].iloc[0]) \
                if "Voiture" in _modal_clean["Mode"].values else float("nan")
            _velo_pct = float(_modal_clean.loc[_modal_clean["Mode"] == "Velo/Deux-roues", "Part (%)"].iloc[0]) \
                if "Velo/Deux-roues" in _modal_clean["Mode"].values else float("nan")
            _s_voit = f"{_voit_pct:.1f} %" if not np.isnan(_voit_pct) else "n.d."
            _s_velo = f"{_velo_pct:.1f} %" if not np.isnan(_velo_pct) else "n.d."
            st.caption(
                f"**Figure 5.2.** Parts modales moyennes des quartiers montpelliérains. "
                f"La voiture reste dominante ({_s_voit}), révélant le poids de la "
                "<em>captivité automobile</em>. Le vélo/deux-roues représente "
                f"{_s_velo} — soit le niveau cible de développement du VLS."
            )
        else:
            _img = _VIZ_PATH / "parts_modales.png"
            if _img.exists():
                st.image(str(_img), use_container_width=True)

    col_v3, col_v4 = st.columns(2)

    with col_v3:
        # Fig 5.3 — Densité × Part vélo (scatter interactif)
        _req3 = {"nom", "densite_hab_km2", "transport_deux_roues_velo_pct"}
        if _req3.issubset(_syn.columns):
            _s3 = _syn.dropna(subset=["densite_hab_km2", "transport_deux_roues_velo_pct"])
            _r3x = _s3["densite_hab_km2"].values
            _r3y = _s3["transport_deux_roues_velo_pct"].values
            _c3  = np.polyfit(_r3x, _r3y, 1)
            _xr3 = np.linspace(_r3x.min(), _r3x.max(), 100)
            _rho3 = float(pd.Series(_r3x).rank().corr(pd.Series(_r3y).rank()))

            fig_v3 = px.scatter(
                _s3,
                x="densite_hab_km2",
                y="transport_deux_roues_velo_pct",
                text="nom",
                color="revenu_fiscal_moyen_menage" if "revenu_fiscal_moyen_menage" in _s3.columns else None,
                color_continuous_scale="Greens",
                labels={
                    "densite_hab_km2":               "Densité de population (hab./km²)",
                    "transport_deux_roues_velo_pct": "Part modale vélo/deux-roues (%)",
                    "revenu_fiscal_moyen_menage":    "Revenu médian (€/an)",
                },
                height=320,
                opacity=0.85,
            )
            fig_v3.add_trace(go.Scatter(
                x=_xr3, y=np.polyval(_c3, _xr3),
                mode="lines",
                line=dict(color="#1A2332", dash="dash", width=2),
                name=f"Tendance OLS (ρ = {_rho3:+.2f})",
                showlegend=True,
            ))
            fig_v3.update_traces(
                selector=dict(mode="markers+text"),
                textposition="top center", textfont=dict(size=8),
            )
            fig_v3.update_layout(
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.22),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_v3, use_container_width=True)
            st.caption(
                f"**Figure 5.3.** Densité de population (hab./km²) versus part modale vélo "
                f"par quartier IRIS. ρ Spearman = {_rho3:+.2f}. "
                "La couleur encode le revenu médian. "
                "La densité urbaine réduit la friction spatiale "
                "(distances plus courtes, moins d'avantage compétitif de la voiture), "
                "agissant comme prédicteur positif de l'usage cyclable indépendant du revenu."
            )
        else:
            _img = _VIZ_PATH / "densite_vs_velo.png"
            if _img.exists():
                st.image(str(_img), use_container_width=True)

    with col_v4:
        # Fig 5.4 — Radar des modes par quartier (top 5 et bottom 5 par vélo)
        _req4 = {
            "nom", "transport_deux_roues_velo_pct", "transport_voiture_camion_pct",
            "transport_marche_a_pied_pct", "transport_transport_commun_pct",
        }
        if _req4.issubset(_syn.columns):
            _s4 = _syn.dropna(subset=["transport_deux_roues_velo_pct"]).copy()
            _s4 = _s4.sort_values("transport_deux_roues_velo_pct")
            _bot5 = _s4.head(5)
            _top5 = _s4.tail(5)
            _sel4 = pd.concat([_bot5, _top5])
            _sel4["rang"] = ["Bottom 5"] * 5 + ["Top 5"] * 5

            _mode_cols = [
                ("transport_deux_roues_velo_pct",  "Vélo"),
                ("transport_voiture_camion_pct",    "Voiture"),
                ("transport_marche_a_pied_pct",     "Marche"),
                ("transport_transport_commun_pct",  "TC"),
            ]
            _mc_avail = [(c, l) for c, l in _mode_cols if c in _sel4.columns]
            _theta4   = [l for _, l in _mc_avail] + [[l for _, l in _mc_avail][0]]

            fig_v4 = go.Figure()
            _pal4 = {"Top 5": "#1A6FBF", "Bottom 5": "#c0392b"}
            for _, _row4 in _sel4.iterrows():
                _vals4 = [_row4[c] for c, _ in _mc_avail] + [_row4[_mc_avail[0][0]]]
                fig_v4.add_trace(go.Scatterpolar(
                    r=_vals4, theta=_theta4,
                    fill="toself",
                    name=str(_row4["nom"])[:18],
                    line_color=_pal4.get(str(_row4["rang"]), "#999"),
                    opacity=0.55,
                ))
            fig_v4.update_layout(
                polar=dict(radialaxis=dict(visible=True, tickfont=dict(size=8))),
                height=320,
                margin=dict(l=30, r=30, t=20, b=30),
                legend=dict(orientation="v", font=dict(size=8)),
                showlegend=True,
            )
            st.plotly_chart(fig_v4, use_container_width=True)
            st.caption(
                "**Figure 5.4.** Profil modal radar des 5 quartiers les plus cyclistes "
                "(bleu) et les 5 moins cyclistes (rouge). "
                "Les quartiers à forte pratique cyclable se distinguent par une part "
                "voiture bien plus faible et une part TC/marche plus élevée, "
                "révélant la complémentarité modale plutôt que la concurrence."
            )
        else:
            _img = _VIZ_PATH / "top_10_quartiers_velo.png"
            if _img.exists():
                st.image(str(_img), use_container_width=True)

with tab_scatter:
    left_sq, right_sq = st.columns(2)

    with left_sq:
        if not top_q.empty and "usage_velo_%" in top_q.columns:
            top_q["rang"] = "Top 10"
            if not bot_q.empty and "usage_velo_%" in bot_q.columns:
                bot_q["rang"] = "Bottom 10"
                compare_q = pd.concat([top_q, bot_q])
            else:
                compare_q = top_q

            fig_q = px.bar(
                compare_q.sort_values("usage_velo_%", ascending=True),
                x="usage_velo_%", y="nom",
                orientation="h",
                color="rang" if "rang" in compare_q.columns else "usage_velo_%",
                color_discrete_map={"Top 10": "#1A6FBF", "Bottom 10": "#c0392b"},
                labels={"nom": "Quartier / Sous-quartier", "usage_velo_%": "Taux d'usage vélo (%)"},
                height=400, text="usage_velo_%",
            )
            fig_q.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
            fig_q.update_layout(
                showlegend=True, plot_bgcolor="white",
                margin=dict(l=10, r=60, t=10, b=10),
            )
            st.plotly_chart(fig_q, use_container_width=True)
            st.caption(
                "**Figure 5.5.** Comparaison top/bottom 10 quartiers par taux d'usage cyclable. "
                "L'amplitude de l'écart quantifie l'intensité de la fracture socio-spatiale "
                "dans l'adoption du vélo en libre-service. "
                "Cette disparité, combinée à l'analyse du revenu (Figure 5.1), "
                "permet de calculer formellement l'IES par quartier."
            )

    with right_sq:
        if {"nom", "revenu_fiscal_moyen_menage", "transport_deux_roues_velo_pct"}.issubset(synthese.columns):
            syn_clean = synthese[
                ["nom", "revenu_fiscal_moyen_menage", "transport_deux_roues_velo_pct",
                 "transport_voiture_camion_pct", "equipement_pas_de_voiture_pct"]
            ].dropna(subset=["revenu_fiscal_moyen_menage", "transport_deux_roues_velo_pct"])

            fig_syn = px.scatter(
                syn_clean,
                x="revenu_fiscal_moyen_menage",
                y="transport_deux_roues_velo_pct",
                text="nom",
                size="equipement_pas_de_voiture_pct" if "equipement_pas_de_voiture_pct" in syn_clean.columns else None,
                size_max=25,
                color="transport_voiture_camion_pct" if "transport_voiture_camion_pct" in syn_clean.columns else None,
                color_continuous_scale="RdBu_r",
                labels={
                    "revenu_fiscal_moyen_menage":       "Revenu fiscal médian (€/an - INSEE Filosofi)",
                    "transport_deux_roues_velo_pct":    "Part modale vélo/deux-roues (%)",
                    "transport_voiture_camion_pct":     "Part modale voiture (%)",
                    "equipement_pas_de_voiture_pct":    "% ménages sans voiture",
                },
                height=400,
            )
            fig_syn.update_traces(textposition="top center", marker_opacity=0.8)
            fig_syn.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_syn, use_container_width=True)
            st.caption(
                "**Figure 5.6.** Revenu fiscal médian (axe horizontal) versus "
                "part modale vélo/deux-roues (axe vertical) par quartier IRIS. "
                "La couleur encode la part modale voiture ; "
                "la taille encode le % de ménages sans voiture. "
                "La relation positive revenu–vélo et la relation négative voiture–vélo "
                "constituent le signal empirique des <em>déserts de mobilité sociale</em> : "
                "les quartiers précaires cumulent forte captivité automobile et faible "
                "accès effectif au réseau de vélos partagés."
            )

# ── Section 6 - Super-Spreaders & Quartiers Fonctionnels Dynamiques ───────────
st.divider()
section(6, "Résilience Réseau : Super-Spreaders et Quartiers Fonctionnels Dynamiques")

st.markdown(r"""
Deux analyses complémentaires révèlent la **vulnérabilité structurelle** et la **dynamique
temporelle** du réseau Vélomagg.

**Super-Spreaders** (*Kitsak et al., 2010*) : le potentiel de contagion opérationnelle
$C_i = D_i \times K_i^2$ (demande × carré de la capacité) identifie les stations dont la
défaillance — stock nul ou vélo hors-service — propage une perturbation maximale dans le réseau.
Ces nœuds critiques doivent être prioritisés dans les stratégies de réapprovisionnement (*rebalancing*).

**Quartiers Fonctionnels Dynamiques** : la détection de communautés de Louvain appliquée
séparément à chaque tranche horaire (matin, midi, soir, nuit) révèle la *recomposition* des
bassins de mobilité au fil de la journée. Un score de stabilité $S \in [0, 1]$ mesure
la cohérence de l'appartenance communautaire d'une station sur les quatre tranches.
""")

tab_ss, tab_dyn = st.tabs(["Super-Spreaders", "Quartiers Dynamiques"])

# ── Onglet Super-Spreaders ─────────────────────────────────────────────────────
with tab_ss:
    if not super_spreaders.empty and "Contagion_Potential" in super_spreaders.columns:
        _n_ss = st.slider(
            "Nombre de stations affichées",
            min_value=5, max_value=min(30, len(super_spreaders)),
            value=min(15, len(super_spreaders)), step=5,
            key="ss_slider",
        )
        _ss_top = super_spreaders.head(_n_ss).copy()

        # Normalisation du potentiel de contagion pour la couleur
        _ss_max = _ss_top["Contagion_Potential"].max()
        _ss_top["CP_norm"] = _ss_top["Contagion_Potential"] / max(_ss_max, 1)

        col_ss_bar, col_ss_map = st.columns([3, 2])

        with col_ss_bar:
            # Nom court pour l'affichage
            _ss_top["label"] = _ss_top["Name"].str.replace(
                r"^\d+\s+", "", regex=True
            ).str[:30]

            fig_ss = go.Figure(go.Bar(
                x=_ss_top["Contagion_Potential"],
                y=_ss_top["label"],
                orientation="h",
                marker=dict(
                    color=_ss_top["CP_norm"],
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="C_i normalisé", thickness=12),
                ),
                text=_ss_top["Demand"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "C_i = %{x:.2e}<br>"
                    "Demande = %{text} trips<extra></extra>"
                ),
            ))
            fig_ss.update_layout(
                xaxis_title="Potentiel de contagion C_i = D × K²",
                yaxis=dict(autorange="reversed"),
                height=480,
                plot_bgcolor="white",
                margin=dict(l=10, r=80, t=20, b=40),
                xaxis=dict(showgrid=True, gridcolor="#eee"),
            )
            st.plotly_chart(fig_ss, use_container_width=True)
            st.caption(
                f"**Figure 6.1.** Top {_n_ss} stations Vélomagg par potentiel de contagion "
                "$C_i = D_i \\times K_i^2$ (Demande × Capacité²). "
                "La couleur encode l'intensité relative du potentiel (rouge = maximal). "
                "Les stations autour de la **Gare Saint-Roch** concentrent un potentiel "
                "disproportionné, justifiant une priorité opérationnelle de réapprovisionnement."
            )

        with col_ss_map:
            _ss_map = _ss_top.dropna(subset=["Latitude", "Longitude"])
            if not _ss_map.empty:
                fig_ss_map = px.scatter_mapbox(
                    _ss_map,
                    lat="Latitude", lon="Longitude",
                    size="Contagion_Potential",
                    color="CP_norm",
                    color_continuous_scale="Reds",
                    hover_name="Name",
                    hover_data={"Demand": True, "Capacity": True,
                                "Contagion_Potential": ":.2e",
                                "Latitude": False, "Longitude": False, "CP_norm": False},
                    mapbox_style="carto-positron",
                    zoom=12,
                    center={"lat": 43.611, "lon": 3.876},
                    size_max=30,
                    height=480,
                    labels={"CP_norm": "C_i normalisé"},
                )
                fig_ss_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_ss_map, use_container_width=True)
                st.caption(
                    "**Figure 6.2.** Carte des super-spreaders. "
                    "La taille et la couleur encodent le potentiel de contagion $C_i$. "
                    "La concentration spatiale autour du centre-ville et de la gare "
                    "matérialise le risque systémique localisé du réseau."
                )

        # Tableau synthétique
        _ss_display = _ss_top[["Name", "Demand", "Capacity", "Contagion_Potential"]].copy()
        _ss_display.insert(0, "Rang", range(1, len(_ss_display) + 1))
        _ss_display = _ss_display.rename(columns={
            "Name": "Station",
            "Demand": "Demande (trips)",
            "Capacity": "Capacité (bornes)",
            "Contagion_Potential": "C_i (potentiel)",
        })
        with st.expander("Tableau complet des super-spreaders"):
            st.dataframe(
                _ss_display.style.format({
                    "Demande (trips)": "{:,.0f}",
                    "Capacité (bornes)": "{:.0f}",
                    "C_i (potentiel)": "{:.2e}",
                }),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Données super-spreaders non disponibles.")

# ── Onglet Quartiers Dynamiques ────────────────────────────────────────────────
with tab_dyn:
    _comm_cols = {
        "Community_Morning_Rush": "Matin (pointe AM)",
        "Community_Lunch_Time":   "Midi",
        "Community_Evening_Rush": "Soir (pointe PM)",
        "Community_Night_Life":   "Nuit",
    }
    _avail_cols = {k: v for k, v in _comm_cols.items() if k in dyn_neighborhoods.columns}

    if not dyn_neighborhoods.empty and _avail_cols:
        col_dyn_left, col_dyn_right = st.columns([3, 2])

        with col_dyn_left:
            # Heatmap station × tranche horaire (appartenance de communauté)
            _hm_data = dyn_neighborhoods[["Name"] + list(_avail_cols.keys())].dropna()
            _hm_data = _hm_data.sort_values(
                "Community_Morning_Rush" if "Community_Morning_Rush" in _hm_data.columns
                else _hm_data.columns[1]
            )
            _hm_labels = [_avail_cols[c] for c in _avail_cols]
            _hm_z = _hm_data[list(_avail_cols.keys())].values.T
            _hm_names = _hm_data["Name"].str.replace(r"^\d+\s+", "", regex=True).str[:25].tolist()

            fig_hm = go.Figure(go.Heatmap(
                z=_hm_z,
                x=_hm_names,
                y=_hm_labels,
                colorscale="Spectral",
                showscale=True,
                colorbar=dict(title="Communauté", thickness=12),
                hoverongaps=False,
                hovertemplate="Station: %{x}<br>Tranche: %{y}<br>Communauté: %{z}<extra></extra>",
            ))
            fig_hm.update_layout(
                xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=10)),
                height=320,
                margin=dict(l=120, r=20, t=20, b=120),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption(
                "**Figure 6.3.** Heatmap d'appartenance communautaire (Louvain) "
                "par station et par tranche horaire. "
                "Les changements de couleur entre colonnes révèlent les stations "
                "à *double vie* fonctionnelle : leur bassin de mobilité se reconfigure "
                "selon que l'heure est une pointe AM, une pause méridienne ou une pointe PM."
            )

        with col_dyn_right:
            # Distribution du score de stabilité
            if "Stability_Score" in dyn_neighborhoods.columns:
                _stab = dyn_neighborhoods["Stability_Score"].dropna()
                fig_stab = px.histogram(
                    dyn_neighborhoods.dropna(subset=["Stability_Score"]),
                    x="Stability_Score",
                    nbins=20,
                    color_discrete_sequence=["#1A6FBF"],
                    labels={"Stability_Score": "Score de stabilité (0 = volatile, 1 = stable)"},
                    height=260,
                )
                fig_stab.update_layout(
                    plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=20, b=40),
                    yaxis_title="Nombre de stations",
                )
                fig_stab.add_vline(
                    x=float(_stab.median()),
                    line_dash="dash", line_color="#e74c3c", line_width=2,
                    annotation_text=f"Médiane = {_stab.median():.2f}",
                    annotation_position="top right",
                )
                st.plotly_chart(fig_stab, use_container_width=True)
                st.caption(
                    "**Figure 6.4.** Distribution du score de stabilité communautaire $S \\in [0,1]$. "
                    f"Médiane = {_stab.median():.2f}. "
                    "Les stations stables ($S \\approx 1$) conservent la même communauté "
                    "quelle que soit la tranche horaire — elles constituent le *squelette* "
                    "permanent du réseau. Les stations volatiles ($S < 0{,}5$) participent "
                    "à plusieurs bassins selon l'heure, révélant leur rôle d'interface entre "
                    "zones résidentielles et zones d'activité."
                )

            # Résumé par tranche horaire : nombre de communautés distinctes
            _n_comm_per_slot = {}
            for col, label in _avail_cols.items():
                _n_comm_per_slot[label] = int(dyn_neighborhoods[col].dropna().nunique())

            if _n_comm_per_slot:
                _slot_df = pd.DataFrame(
                    list(_n_comm_per_slot.items()),
                    columns=["Tranche horaire", "Communautés actives"],
                )
                fig_slots = px.bar(
                    _slot_df,
                    x="Tranche horaire",
                    y="Communautés actives",
                    color="Communautés actives",
                    color_continuous_scale="Blues",
                    text="Communautés actives",
                    height=240,
                )
                fig_slots.update_traces(textposition="outside")
                fig_slots.update_layout(
                    plot_bgcolor="white",
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=20, b=40),
                )
                st.plotly_chart(fig_slots, use_container_width=True)
                st.caption(
                    "**Figure 6.5.** Nombre de communautés Louvain actives par tranche horaire. "
                    "La fragmentation communautaire maximale en pointe AM/PM illustre "
                    "l'éclatement fonctionnel du réseau aux heures de mobilité intense."
                )
    else:
        st.info("Données de quartiers fonctionnels dynamiques non disponibles.")
