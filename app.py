"""
app.py — Page d'accueil du tableau de bord Gold Standard GBFS France.

Lancer :
    streamlit run streamlit/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Assurer que le dossier parent est dans le path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import METRICS, completeness_report, load_stations

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Standard GBFS · France",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/anthropics/claude-code",
        "About": "Dashboard Gold Standard enrichi — Notebook 27\nRecherche CESI BikeShare",
    },
)

# ── CSS minimal ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .metric-card {
        background: #f8f9fa; border-radius: 8px; padding: 1rem 1.25rem;
        border-left: 4px solid #2ecc71; margin-bottom: .5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chargement ────────────────────────────────────────────────────────────────
df = load_stations()

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("🚲 Gold Standard GBFS — France")
st.markdown(
    """
    Jeu de données enrichi produit par le **notebook 27** :
    **46 k+ stations** issues des flux GBFS français enrichies de 5 dimensions spatiales
    (infrastructure cyclable, accidentologie, multimodalité, topographie).
    """
)
st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
n_total   = len(df)
n_cities  = df["city"].nunique()
n_systems = df["system_id"].nunique()
avg_infra = df["infra_cyclable_pct"].mean()
avg_gtfs  = df["gtfs_heavy_stops_300m"].mean()
avg_baac  = df["baac_accidents_cyclistes"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Stations", f"{n_total:,}")
col2.metric("Villes", f"{n_cities}")
col3.metric("Réseaux", f"{n_systems}")
col4.metric("Infra cyclable moy.", f"{avg_infra:.1f} %")
col5.metric("Arrêts TC lourds moy.", f"{avg_gtfs:.2f}")
col6.metric("Accidents moy. (300 m)", f"{avg_baac:.3f}")

st.divider()

# ── Complétude ────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Complétude des métriques enrichies")
    comp_df = completeness_report(df)

    # Barre de progression via plotly horizontal bar
    fig_comp = px.bar(
        comp_df,
        x="Complétude (%)",
        y="Métrique",
        orientation="h",
        color="Complétude (%)",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        text="Complétude (%)",
        height=320,
    )
    fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_comp.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=40, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 115]),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with right:
    st.subheader("Répartition des sources")
    src_counts = df["source_label"].value_counts().reset_index()
    src_counts.columns = ["Source", "Stations"]
    fig_pie = px.pie(
        src_counts,
        names="Source",
        values="Stations",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.45,
        height=320,
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Top villes par nombre de stations ─────────────────────────────────────────
st.subheader("Top 15 villes — nombre de stations")
top_cities = (
    df.groupby("city")
    .agg(
        n_stations=("uid", "count"),
        infra_pct=("infra_cyclable_pct", "mean"),
        gtfs=("gtfs_heavy_stops_300m", "mean"),
        accidents=("baac_accidents_cyclistes", "mean"),
    )
    .reset_index()
    .sort_values("n_stations", ascending=False)
    .head(15)
)

fig_top = px.bar(
    top_cities,
    x="city",
    y="n_stations",
    color="infra_pct",
    color_continuous_scale="Greens",
    labels={"city": "Ville", "n_stations": "Stations", "infra_pct": "Infra cyclable (%)"},
    text="n_stations",
    height=380,
)
fig_top.update_traces(textposition="outside")
fig_top.update_layout(
    coloraxis_colorbar=dict(title="Infra cyclable (%)"),
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="white",
    xaxis_tickangle=-30,
)
st.plotly_chart(fig_top, use_container_width=True)

st.divider()

# ── Description des métriques ─────────────────────────────────────────────────
with st.expander("ℹ️ Description des métriques enrichies", expanded=False):
    for col, meta in METRICS.items():
        st.markdown(f"**{meta['label']}** (`{col}`)  \n{meta['description']}")
        st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/115px-Python-logo-notext.svg.png",
        width=40,
    )
    st.markdown("### Navigation")
    st.page_link("app.py",                  label="🏠 Accueil",          icon="🏠")
    st.page_link("pages/1_Carte.py",        label="Carte interactive",   icon="🗺️")
    st.page_link("pages/2_Villes.py",       label="Comparaison villes",  icon="🏙️")
    st.page_link("pages/3_Distributions.py", label="Distributions",      icon="📊")
    st.divider()
    st.caption(
        "Gold Standard GBFS · Notebook 27  \n"
        "CESI BikeShare-Graph-Forecasting  \n"
        f"**{n_total:,} stations** · {n_cities} villes"
    )
