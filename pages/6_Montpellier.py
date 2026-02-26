"""
6_Montpellier.py — Analyse approfondie du réseau Vélomagg (Montpellier / TAM-MMM).
Sources : socioeconomic_analysis_results, station_temporal_profiles, flow_analysis,
          bike_tram_proximity_matrix, synthese_velo_socio.
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
    load_hourly_flows,
    load_montpellier_stations,
    load_net_flows,
    load_station_stress,
    load_station_temporal_profiles,
    load_synthese_velo_socio,
    load_top_quartiers,
)

st.set_page_config(
    page_title="Montpellier Vélomagg — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)

st.title("Réseau Vélomagg — Montpellier Méditerranée Métropole")
st.markdown(
    """
    Analyse approfondie du système de vélos en libre-service Vélomagg (TAM / MMM)
    à partir des profils de stations, des matrices de flux origine-destination,
    du maillage multimodal tram-vélo et des indicateurs socio-économiques
    par quartier montpelliérain.
    """
)

# ── Chargement ────────────────────────────────────────────────────────────────
stations  = load_montpellier_stations()
profiles  = load_station_temporal_profiles()
stress    = load_station_stress()
hourly    = load_hourly_flows()
net_flows = load_net_flows()
biketram  = load_bike_tram_proximity()
synthese  = load_synthese_velo_socio()
top_q, bot_q = load_top_quartiers()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    st.page_link("app.py",                        label="Accueil")
    st.page_link("pages/1_Carte.py",              label="Carte des stations")
    st.page_link("pages/2_Villes.py",             label="Comparaison des villes")
    st.page_link("pages/3_Distributions.py",      label="Distributions statistiques")
    st.page_link("pages/4_Export.py",             label="Export des données")
    st.page_link("pages/5_Mobilite_France.py",    label="Indicateurs nationaux")
    st.page_link("pages/6_Montpellier.py",        label="Montpellier — Velomagg")

# ── KPI ───────────────────────────────────────────────────────────────────────
st.markdown("#### Vue d'ensemble du réseau Vélomagg")

n_stations   = len(stations)
total_trips  = int(profiles["Total_Trips"].sum(skipna=True)) if "Total_Trips" in profiles.columns else 0
avg_trips    = round(profiles["Total_Trips"].mean(), 0) if "Total_Trips" in profiles.columns else 0
top_station  = profiles.loc[profiles["Total_Trips"].idxmax(), "Station_Name"] if "Total_Trips" in profiles.columns else "—"
n_quartiers  = stations["Quartier"].nunique() if "Quartier" in stations.columns else "—"
pct_5min     = 100 * biketram["walkable_5min"].mean() if "walkable_5min" in biketram.columns else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Stations analysées", f"{n_stations}")
k2.metric("Trips totaux (historique)", f"{total_trips:,}")
k3.metric("Trips moyens / station", f"{avg_trips:,.0f}")
k4.metric("Station la plus active", top_station)
k5.metric("Quartiers couverts", f"{n_quartiers}")
k6.metric("Stations < 5 min d'un tram", f"{pct_5min:.1f} %")

st.divider()

# ── Carte des stations ────────────────────────────────────────────────────────
st.subheader("Carte des stations — volume de trips et profil")
st.caption(
    "Taille des points proportionnelle au volume de trips historiques. "
    "Couleur selon le profil d'usage (Commuter, Residential, Mixed-Use)."
)

map_df = stations.merge(
    profiles[["Station_Name", "Total_Trips", "Profile"]].rename(
        columns={"Station_Name": "Station_Name_p"}),
    left_on="Station_Name", right_on="Station_Name_p", how="left",
)
map_df["Total_Trips"] = map_df["Total_Trips"].fillna(map_df.get("Total_Trips_x", 0))

color_map_profile = {"Residential": "#4A9FDF", "Commuter": "#c0392b", "Mixed-Use": "#1A6FBF"}

fig_map = px.scatter_mapbox(
    map_df.dropna(subset=["latitude", "longitude"]),
    lat="latitude",
    lon="longitude",
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
    height=500,
    labels={"Profile": "Profil", "Total_Trips": "Trips"},
)
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ── Profils temporels ─────────────────────────────────────────────────────────
left_prof, right_prof = st.columns([2, 3])

with left_prof:
    st.subheader("Distribution des profils de stations")
    st.caption(
        "Classement des stations selon leur pattern d'usage : "
        "**Commuter** (départs concentrés matin/soir), "
        "**Residential** (usage résidentiel), "
        "**Mixed-Use** (usage diffus)."
    )
    if "Profile" in profiles.columns:
        prof_counts = profiles["Profile"].value_counts().reset_index()
        prof_counts.columns = ["Profil", "Stations"]
        fig_prof = px.bar(
            prof_counts, x="Stations", y="Profil", orientation="h",
            color="Profil",
            color_discrete_map=color_map_profile,
            text="Stations", height=280,
        )
        fig_prof.update_traces(textposition="outside")
        fig_prof.update_layout(showlegend=False, plot_bgcolor="white",
                               margin=dict(l=10, r=40, t=10, b=10),
                               yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_prof, use_container_width=True)

with right_prof:
    st.subheader("Trips semaine vs week-end par profil")
    st.caption("Comparaison de la distribution des départs selon le type de jour.")
    if {"Profile", "Weekday_Departures", "Weekend_Departures"}.issubset(profiles.columns):
        bp_df = profiles[["Station_Name", "Profile",
                           "Weekday_Departures", "Weekend_Departures"]].dropna()
        bp_melt = bp_df.melt(
            id_vars=["Station_Name", "Profile"],
            value_vars=["Weekday_Departures", "Weekend_Departures"],
            var_name="Type", value_name="Departures",
        )
        bp_melt["Type"] = bp_melt["Type"].map(
            {"Weekday_Departures": "Semaine", "Weekend_Departures": "Week-end"}
        )
        fig_bp = px.box(
            bp_melt, x="Profile", y="Departures", color="Type",
            color_discrete_map={"Semaine": "#1A6FBF", "Week-end": "#4A9FDF"},
            labels={"Profile": "Profil", "Departures": "Départs", "Type": ""},
            height=300,
            notched=False,
        )
        fig_bp.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bp, use_container_width=True)

st.divider()

# ── Flux horaires ─────────────────────────────────────────────────────────────
st.subheader("Patterns horaires du réseau (0 h — 23 h)")

left_hr, right_hr = st.columns(2)

with left_hr:
    st.caption(
        "Volume total de flux agrégé sur l'ensemble de l'historique, "
        "normalisé par le nombre de jours. Deux pics attendus : "
        "matin (~8 h) et fin d'après-midi (~17-18 h)."
    )
    if "total_trips" in hourly.columns:
        fig_hr = px.bar(
            hourly, x="hour", y="total_trips",
            color="total_trips", color_continuous_scale="Blues",
            labels={"hour": "Heure", "total_trips": "Trips moyens / heure"},
            height=300,
        )
        fig_hr.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                             margin=dict(l=10, r=10, t=10, b=10),
                             xaxis=dict(dtick=2))
        st.plotly_chart(fig_hr, use_container_width=True)

with right_hr:
    st.caption(
        "Nombre de paires origine-destination actives par heure. "
        "Reflète la diversité des trajets effectués selon l'heure."
    )
    if "unique_od_pairs" in hourly.columns:
        fig_od = px.line(
            hourly, x="hour", y="unique_od_pairs",
            labels={"hour": "Heure", "unique_od_pairs": "Paires OD actives"},
            height=300,
            markers=True,
            color_discrete_sequence=["#1A6FBF"],
        )
        fig_od.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                             xaxis=dict(dtick=2))
        st.plotly_chart(fig_od, use_container_width=True)

st.divider()

# ── Flux nets : déséquilibres ─────────────────────────────────────────────────
st.subheader("Déséquilibres de flux — stations source et puits")
st.caption(
    "Un flux net positif indique un excédent d'entrées (station *puits* : "
    "attractrice de vélos). Un flux net négatif indique une station *source* "
    "(génératrice de départs). Les stations extrêmes nécessitent "
    "une redistribution active des vélos."
)

if {"station", "net_flow"}.issubset(net_flows.columns):
    net_agg = (
        net_flows.groupby("station")["net_flow"]
        .mean()
        .reset_index()
        .sort_values("net_flow")
    )
    n_show = st.slider("Stations à afficher (extrêmes)", 10, 40, 20, 5)
    half = n_show // 2
    extremes = pd.concat([net_agg.head(half), net_agg.tail(half)]).drop_duplicates()

    fig_net = px.bar(
        extremes, x="net_flow", y="station",
        orientation="h",
        color="net_flow",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={"station": "Station", "net_flow": "Flux net moyen"},
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

st.divider()

# ── Centralité réseau ─────────────────────────────────────────────────────────
st.subheader("Centralité du réseau et indice de stress")

left_cent, right_cent = st.columns(2)

with left_cent:
    st.caption(
        "PageRank vs volume de trips : les stations avec un fort PageRank "
        "sont des nœuds structurellement centraux dans le graphe de flux, "
        "indépendamment de leur volume absolu."
    )
    if {"PageRank", "Total_Trips", "Station_Name"}.issubset(stations.columns):
        fig_pr = px.scatter(
            stations.dropna(subset=["PageRank", "Total_Trips"]),
            x="PageRank", y="Total_Trips",
            hover_name="Station_Name",
            color="Quartier" if "Quartier" in stations.columns else None,
            labels={"PageRank": "PageRank", "Total_Trips": "Trips totaux"},
            height=360,
            opacity=0.75,
        )
        fig_pr.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                             showlegend=False)
        st.plotly_chart(fig_pr, use_container_width=True)

with right_cent:
    st.caption(
        "Top 15 stations par indice de stress. "
        "L'indice combine la demande, la centralité et la population environnante "
        "pour identifier les points de fragilité opérationnelle du réseau."
    )
    if "Stress_Index" in stress.columns:
        top_stress = stress.nlargest(15, "Stress_Index")
        fig_stress = px.bar(
            top_stress, x="Stress_Index", y="Station_Name",
            orientation="h",
            color="Stress_Index", color_continuous_scale="Reds",
            labels={"Station_Name": "Station", "Stress_Index": "Indice de stress"},
            height=360,
        )
        fig_stress.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            margin=dict(l=10, r=60, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_stress, use_container_width=True)

st.divider()

# ── Intégration multimodale ───────────────────────────────────────────────────
st.subheader("Intégration multimodale — proximité vélo / tram")
st.caption(
    "Distance à pied entre chaque station Vélomagg et l'arrêt de tram le plus proche. "
    "Le seuil de 5 min (400 m) est le standard d'intermodalité confortable ; "
    "10 min (800 m) représente la limite acceptable."
)

if {"bike_station", "distance_m", "walkable_5min", "walkable_10min"}.issubset(biketram.columns):
    closest = (
        biketram.groupby("bike_station")["distance_m"]
        .min()
        .reset_index()
        .rename(columns={"distance_m": "dist_tram_min_m"})
    )
    closest["< 5 min (400 m)"]  = closest["dist_tram_min_m"] <= 400
    closest["5-10 min (800 m)"] = (closest["dist_tram_min_m"] > 400) & (closest["dist_tram_min_m"] <= 800)
    closest["> 10 min"]         = closest["dist_tram_min_m"] > 800

    pct_5  = 100 * closest["< 5 min (400 m)"].mean()
    pct_10 = 100 * closest["5-10 min (800 m)"].mean()
    pct_gt = 100 * closest["> 10 min"].mean()

    ml1, ml2, ml3 = st.columns(3)
    ml1.metric("Stations < 5 min d'un tram", f"{pct_5:.1f} %")
    ml2.metric("Stations entre 5 et 10 min", f"{pct_10:.1f} %")
    ml3.metric("Stations > 10 min", f"{pct_gt:.1f} %")

    fig_dist = px.histogram(
        closest, x="dist_tram_min_m", nbins=30,
        color_discrete_sequence=["#1A6FBF"],
        labels={"dist_tram_min_m": "Distance au tram le plus proche (m)", "count": "Stations"},
        height=300,
    )
    fig_dist.add_vline(x=400, line_dash="dash", line_color="#c0392b",
                       annotation_text="5 min", annotation_position="top right")
    fig_dist.add_vline(x=800, line_dash="dash", line_color="#e67e22",
                       annotation_text="10 min", annotation_position="top right")
    fig_dist.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ── Analyse socio-économique par quartier ─────────────────────────────────────
st.subheader("Mobilité vélo et profil socio-économique par quartier")

left_sq, right_sq = st.columns(2)

with left_sq:
    st.caption(
        "Top 10 et bottom 10 quartiers par taux d'usage du vélo. "
        "L'écart entre quartiers révèle des inégalités de pratique "
        "qui peuvent ne pas correspondre aux inégalités d'infrastructure."
    )
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
            labels={"nom": "Quartier / Sous-quartier", "usage_velo_%": "Usage vélo (%)"},
            height=400,
            text="usage_velo_%",
        )
        fig_q.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig_q.update_layout(showlegend=True, plot_bgcolor="white",
                            margin=dict(l=10, r=60, t=10, b=10))
        st.plotly_chart(fig_q, use_container_width=True)

with right_sq:
    st.caption(
        "Revenu fiscal moyen des ménages versus part modale vélo par quartier. "
        "Un fort coefficient de corrélation signalerait un biais socio-économique "
        "dans l'adoption du vélo."
    )
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
                "revenu_fiscal_moyen_menage": "Revenu fiscal moyen (euros)",
                "transport_deux_roues_velo_pct": "Part modale vélo (%)",
                "transport_voiture_camion_pct": "Part modale voiture (%)",
                "equipement_pas_de_voiture_pct": "% sans voiture",
            },
            height=400,
        )
        fig_syn.update_traces(textposition="top center", marker_opacity=0.8)
        fig_syn.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_syn, use_container_width=True)
