"""
6_Montpellier.py — Validation micro-locale : réseau Vélomagg (Montpellier Méditerranée Métropole).
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
    load_bike_tram_proximity,
    load_community_detection,
    load_hourly_flows,
    load_montpellier_stations,
    load_net_flows,
    load_network_topology,
    load_parts_modales,
    load_station_stress,
    load_station_temporal_profiles,
    load_station_vulnerability,
    load_synthese_velo_socio,
    load_top_quartiers,
    load_weather_data,
)
from utils.styles import abstract_box, inject_css, section, sidebar_nav

_VIZ_PATH = Path(__file__).parent.parent / "data" / "processed" / "ville_montpellier" / "visualisations"

st.set_page_config(
    page_title="Montpellier Vélomagg — Validation Micro-Locale",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Validation Micro-Locale : Réseau Vélomagg de Montpellier")
st.caption("Axe de Recherche 5 : Friction Spatiale, Écosystème Multimodal et Fracture Socio-Spatiale")

abstract_box(
    "<b>Question de recherche :</b> La rugosité topographique et la distance aux nœuds "
    "modaux expliquent-elles les déséquilibres source/puits opérationnels du réseau Vélomagg ? "
    "Dans quelle mesure l'écosystème multimodal GTFS-tramway réduit-il ces frictions ?<br><br>"
    "Cette étude de cas constitue l'étape de <em>validation micro-locale</em> du cadre "
    "analytique développé à l'échelle nationale. Le réseau Vélomagg "
    "(TAM / Montpellier Méditerranée Métropole) est analysé selon cinq axes complémentaires : "
    "(1) la <em>topologie du réseau</em> — structure du graphe de flux, détection de communautés "
    "de Louvain, stations-pivots et points d'articulation structurellement critiques ; "
    "(2) les <em>dynamiques temporelles</em> — patterns horaires, régimes bimodaux "
    "domicile-travail et sensibilité aux conditions météorologiques ; "
    "(3) la <em>modélisation de la friction spatiale</em> — déséquilibres source/puits "
    "et indice de vulnérabilité opérationnelle ; "
    "(4) l'<em>écosystème multimodal</em> — intégration GTFS du réseau tramway TAM "
    "et quantification du coût de correspondance piéton-vélo ; "
    "(5) la <em>fracture socio-spatiale</em> — corrélation entre usage du vélo et "
    "profil socio-économique par quartier (IRIS), révélant les déserts de mobilité sociale. "
    "Les données couvrent 5 ans d'historique de courses Vélomagg (notebooks 22–26)."
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()

# ── KPIs globaux ──────────────────────────────────────────────────────────────
n_stations  = len(stations)
total_trips = int(profiles["Total_Trips"].sum(skipna=True)) if "Total_Trips" in profiles.columns else 0
avg_trips   = round(profiles["Total_Trips"].mean(), 0) if "Total_Trips" in profiles.columns else 0
top_station = profiles.loc[profiles["Total_Trips"].idxmax(), "Station_Name"] if "Total_Trips" in profiles.columns else "—"
n_quartiers = stations["Quartier"].nunique() if "Quartier" in stations.columns else "—"
pct_5min    = 100 * biketram["walkable_5min"].mean() if "walkable_5min" in biketram.columns else 0
n_bridge    = int(community["Is_Bridge"].sum()) if "Is_Bridge" in community.columns else "—"

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Stations analysées", f"{n_stations}")
k2.metric("Trips totaux (5 ans)", f"{total_trips:,}")
k3.metric("Trips moyens / station", f"{avg_trips:,.0f}")
k4.metric("Station la plus active", top_station)
k5.metric("Stations < 5 min d'un tram", f"{pct_5min:.1f} %")
k6.metric("Stations-pivots (bridge)", f"{n_bridge}")

# ── Section 1 — Topologie du réseau et communautés ───────────────────────────
st.divider()
section(1, "Topologie du Réseau et Détection de Communautés — Structure du Graphe de Flux")

st.markdown(r"""
L'analyse de la topologie du graphe de flux (*directed weighted graph*, $n = 56$ nœuds)
permet de caractériser la structure organisationnelle profonde du réseau Vélomagg.
La **détection de communautés de Louvain** (*Blondel et al., 2008*) identifie des
sous-ensembles de stations échangeant préférentiellement entre elles, révélant des
bassins de mobilité fonctionnels indépendants des découpages administratifs.

Les **stations-pivots** (*bridge stations*, coefficient de participation $P_i > 0{,}5$)
sont les nœuds assurant la connexion entre plusieurs communautés distinctes.
Leur défaillance opérationnelle (stock nul, vélo défectueux) présente un risque
structurel élevé pour la connectivité globale du réseau.
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

# Section 1 — suite : profils d'usage
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

# ── Section 2 — Dynamiques temporelles et friction météorologique ─────────────
st.divider()
section(2, "Dynamiques Temporelles — Régimes Bimodaux, Flux OD et Friction Météorologique")

st.markdown(r"""
Les dynamiques temporelles du réseau Vélomagg révèlent un **régime bimodal** caractéristique
des mobilités pendulaires : un pic matinal ($h \approx 8$) et un pic vespéral
($h \approx 17$–$18$) concentrent respectivement $\sim 18\,\%$ et $\sim 22\,\%$
du volume de trips journalier. La friction météorologique — quantifiée par le
score de mauvais temps ($\text{bad\_weather\_score} \in [0, 1]$) — constitue
un déterminant externe des déséquilibres source/puits : les précipitations induisent
une réduction significative des départs, exacerbant les stocks dans les stations-puits
et aggravant les pénuries dans les stations-sources.
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
        weather_hourly = weather.groupby("hour").agg(
            trips_mean=("bad_weather_score", lambda x: (x == 0).mean() * 100),
            bad_weather_pct=("bad_weather_score", lambda x: (x > 0).mean() * 100),
        ).reset_index()

        season_labels = {1: "Hiver", 2: "Printemps", 3: "Été", 4: "Automne"}
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
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            st.caption(
                "**Figure 2.3.** Taux horaire de précipitations par saison (Montpellier, 2021–2024). "
                "La friction météorologique saisonnière modifie la distribution des déséquilibres "
                "source/puits et conditionne la politique de redistribution (<em>rebalancing</em>). "
                "L'été méditerranéen, quasi exempt de précipitations, favorise les pics d'usage "
                "touristique non pendulaire."
            )
    else:
        st.info(
            "Les données météorologiques enrichies sont nécessaires pour cette analyse. "
            "Vérifiez la présence de `weather_data_enriched.csv` dans `data/processed/`."
        )

# ── Section 3 — Friction spatiale : déséquilibres et vulnérabilité ─────────────
st.divider()
section(3, "Modélisation de la Friction Spatiale — Déséquilibres Source/Puits et Vulnérabilité Structurelle")

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
st.markdown("#### Centralité Structurelle — PageRank versus Volume de Trips")
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

# ── Section 4 — Écosystème multimodal ─────────────────────────────────────────
st.divider()
section(4, "Écosystème Multimodal — Intégration GTFS Tramway et Coût de Correspondance")

st.markdown(r"""
L'efficacité du réseau Vélomagg comme solution du **premier/dernier kilomètre** repose sur
son intégration à l'écosystème de transport lourd TAM (4 lignes de tramway, fréquence
$\sim 5$–$7$ min en heure de pointe). Le seuil d'interopérabilité confortable est fixé
à **400 m** ($\approx 5$ min à 80 m/min), standard UITP pour la correspondance multimodale.
Au-delà de **800 m** ($\approx 10$ min), le coût de correspondance piéton-vélo devient
dissuasif et la complémentarité modale se dissout.

Le coût de correspondance constitue une composante de la **friction spatiale totale** :
$\text{friction}_{\text{totale}} = \text{friction}_{\text{topo}} + \text{friction}_{\text{multimodale}} + \text{friction}_{\text{météo}}$.
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

# ── Section 5 — Fracture socio-spatiale ───────────────────────────────────────
st.divider()
section(5, "Fracture Socio-Spatiale — Usage du Vélo et Profil Socio-Économique par Quartier")

st.markdown(r"""
L'analyse de la fracture socio-spatiale confronte les données d'usage du réseau Vélomagg
aux indicateurs socio-économiques des quartiers de Montpellier (IRIS, données INSEE Filosofi).
L'objectif est de tester empiriquement l'hypothèse d'un **biais socio-économique**
dans l'adoption de la micromobilité partagée : les ménages à revenus élevés
utilisent-ils davantage le vélo en libre-service que les ménages précaires,
et cette disparité est-elle expliquée par des inégalités d'infrastructure
ou par des barrières d'usage indépendantes de l'offre physique ?

Si tel est le cas, les zones à revenu faible et faible usage constituent formellement
des **déserts de mobilité sociale** au sens de l'Indice d'Équité Sociale (IES).
""")

tab_viz, tab_scatter = st.tabs(["Figures Pré-calculées (Analyse IRIS)", "Analyse Interactive"])

with tab_viz:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        img_revenu = _VIZ_PATH / "revenu_vs_velo.png"
        if img_revenu.exists():
            st.image(str(img_revenu), use_container_width=True)
            st.caption(
                "**Figure 5.1.** Revenu fiscal médian des ménages (€/an, INSEE Filosofi) "
                "versus part modale vélo/deux-roues par quartier IRIS. "
                "La pente positive de la relation atteste d'un biais socio-économique "
                "structurel dans l'adoption de la micromobilité partagée : "
                "les quartiers aisés cyclent davantage, indépendamment de la densité "
                "d'infrastructure. Ce résultat caractérise une situation de "
                "<em>désert de mobilité sociale</em> dans les quartiers périphériques précaires."
            )

    with col_v2:
        img_modal = _VIZ_PATH / "parts_modales.png"
        if img_modal.exists():
            st.image(str(img_modal), use_container_width=True)
            st.caption(
                "**Figure 5.2.** Parts modales moyennes des quartiers montpelliérains "
                "(marche à pied, vélo/deux-roues, voiture, transports en commun). "
                "La part modale voiture reste dominante ($\\approx 59\\,\\%$), "
                "révélant le poids de la <em>captivité automobile</em> "
                "dans les mobilités quotidiennes."
            )

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        img_densite = _VIZ_PATH / "densite_vs_velo.png"
        if img_densite.exists():
            st.image(str(img_densite), use_container_width=True)
            st.caption(
                "**Figure 5.3.** Densité de population (hab./km²) versus part modale vélo. "
                "La densité urbaine constitue un prédicteur positif de l'usage cyclable, "
                "indépendamment du revenu, en réduisant la friction spatiale totale "
                "(distances plus courtes, moins d'avantage compétitif de la voiture)."
            )

    with col_v4:
        img_top = _VIZ_PATH / "top_10_quartiers_velo.png"
        if img_top.exists():
            st.image(str(img_top), use_container_width=True)
            st.caption(
                "**Figure 5.4.** Top 10 quartiers par taux d'usage cyclable. "
                "Le Centre, l'Écusson et les quartiers universitaires concentrent "
                "l'essentiel de la pratique cyclable, révélant l'interaction entre "
                "densité étudiante, proximité aux nœuds tramway et disponibilité des stations."
            )

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
                    "revenu_fiscal_moyen_menage":       "Revenu fiscal médian (€/an — INSEE Filosofi)",
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
