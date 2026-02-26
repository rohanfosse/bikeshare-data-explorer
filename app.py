"""
app.py — Page d'accueil du tableau de bord Gold Standard GBFS France.
Pipeline d'enrichissement : notebooks/27_gold_standard_enrichment.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import METRICS, completeness_report, load_stations

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Standard GBFS — France",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Gold Standard GBFS — Pipeline d'enrichissement spatial\n"
            "Recherche BikeShare-ICT — CESI 2025-2026"
        ),
    },
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: .8rem; text-transform: uppercase;
                                    letter-spacing: .05em; color: #5a7a99; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chargement ────────────────────────────────────────────────────────────────
df = load_stations()

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("Gold Standard GBFS — Micromobilité française")
st.markdown(
    """
    Tableau de bord issu du **pipeline d'enrichissement spatial (notebook 27)**
    appliqué aux 46 000+ stations GBFS françaises auditées (notebook 20).
    Chaque station est enrichie selon cinq modules thématiques
    calculés dans un rayon standard de **300 m** autour du point de stationnement.
    """
)

# ── Méthodologie ──────────────────────────────────────────────────────────────
with st.expander("Méthodologie du pipeline d'enrichissement", expanded=True):
    st.markdown(
        """
| Module | Axe d'enrichissement | Colonnes produites | Source de données |
|:------:|:---------------------|:-------------------|:------------------|
| 1 | Comblement des zones blanches OSM | `source`, `osm_node_id` | OpenStreetMap |
| 2 | Topographie nationale (SRTM 30 m) | `elevation_m`, `topography_roughness_index` | Open-Elevation / SRTM |
| 3A | Continuité cyclable (cycleways OSM) | `infra_cyclable_km`, `infra_cyclable_pct` | OSM Overpass API |
| 3B | Sécurité — accidents cyclistes | `baac_accidents_cyclistes` | BAAC 2021-2023 (ONISR) |
| 4 | Multimodalité lourde (métro, tram, RER) | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | Flux GTFS nationaux |

**Stratégie d'implémentation** : traitement par lots (*batch processing*) avec
requêtes HTTP asynchrones (`aiohttp`) et mise en cache locale.
Le rayon de 300 m correspond au standard last-mile pour l'analyse
de la continuité des déplacements.
        """
    )

st.divider()

# ── KPI ───────────────────────────────────────────────────────────────────────
n_total   = len(df)
n_cities  = df["city"].nunique()
n_systems = df["system_id"].nunique()
avg_infra = df["infra_cyclable_pct"].mean()
avg_gtfs  = df["gtfs_heavy_stops_300m"].mean()
avg_baac  = df["baac_accidents_cyclistes"].mean()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Stations", f"{n_total:,}")
col2.metric("Villes", f"{n_cities}")
col3.metric("Réseaux GBFS", f"{n_systems}")
col4.metric("Infra cyclable moy.", f"{avg_infra:.1f} %")
col5.metric("Arrêts TC lourds moy.", f"{avg_gtfs:.2f}")
col6.metric("Accidents moy. (300 m)", f"{avg_baac:.3f}")

st.divider()

# ── Complétude & Sources ───────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Complétude des métriques enrichies")
    st.caption(
        "Pourcentage de stations disposant d'une valeur valide pour chaque "
        "dimension d'enrichissement."
    )
    comp_df = completeness_report(df)

    fig_comp = px.bar(
        comp_df,
        x="Complétude (%)",
        y="Métrique",
        orientation="h",
        color="Complétude (%)",
        color_continuous_scale="Blues",
        range_color=[0, 100],
        text="Complétude (%)",
        height=320,
    )
    fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_comp.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=50, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 115], title=""),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with right:
    st.subheader("Provenance des données stations")
    st.caption(
        "Répartition des stations selon leur source d'origine "
        "(flux GBFS officiels ou complément OSM)."
    )
    src_counts = df["source_label"].value_counts().reset_index()
    src_counts.columns = ["Source", "Stations"]
    fig_pie = px.pie(
        src_counts,
        names="Source",
        values="Stations",
        color_discrete_sequence=["#1A6FBF", "#4A9FDF", "#A8CFEF"],
        hole=0.5,
        height=320,
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Top villes ─────────────────────────────────────────────────────────────────
st.subheader("Top 15 villes — nombre de stations et infrastructure cyclable")
st.caption(
    "Classement par volume de stations. La couleur indique la part moyenne "
    "d'infrastructure cyclable dans un rayon de 300 m."
)
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
    color_continuous_scale="Blues",
    labels={
        "city": "Ville",
        "n_stations": "Nombre de stations",
        "infra_pct": "Infra cyclable (%)",
    },
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
with st.expander("Définition des métriques enrichies", expanded=False):
    for col, meta in METRICS.items():
        directionality = ""
        if meta["higher_is_better"] is True:
            directionality = " — *valeur élevée favorable*"
        elif meta["higher_is_better"] is False:
            directionality = " — *valeur faible favorable*"
        st.markdown(
            f"**{meta['label']}** (`{col}`, {meta['unit']}){directionality}  \n"
            f"{meta['description']}"
        )
        st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    st.page_link("app.py",                   label="Accueil")
    st.page_link("pages/1_Carte.py",         label="Carte des stations")
    st.page_link("pages/2_Villes.py",        label="Comparaison des villes")
    st.page_link("pages/3_Distributions.py", label="Distributions statistiques")
    st.divider()
    st.markdown(
        "**Gold Standard GBFS**  \n"
        "Pipeline d'enrichissement spatial  \n"
        "Notebook 27 — CESI BikeShare-ICT  \n"
        f"`{n_total:,}` stations · {n_cities} villes"
    )
    st.caption("Recherche CESI 2025-2026")
