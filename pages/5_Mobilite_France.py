"""
5_Mobilite_France.py — Indicateurs comparatifs de mobilité douce nationale.
Combine le catalogue GBFS (122 systèmes) avec les indicateurs de ville :
FUB Baromètre 2023, EMP 2019, BAAC, Cerema, Eco-compteurs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_city_mobility, load_systems_catalog

st.set_page_config(
    page_title="Mobilité nationale — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)

st.title("Indicateurs nationaux de mobilité douce")
st.markdown(
    """
    Croisement du catalogue GBFS France (122 systèmes, notebook 20) avec
    des indicateurs externes à l'échelle des villes :
    **FUB Baromètre 2023** (perception cycliste), **EMP 2019** (part modale vélo),
    **BAAC** (accidentologie), **Cerema** (infrastructure) et **Eco-compteurs** (fréquentation).
    """
)

systems = load_systems_catalog()
city_df  = load_city_mobility()

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
    st.divider()
    st.header("Filtres")
    regions = sorted(systems["region"].dropna().unique())
    region_sel = st.multiselect("Région(s)", options=regions, default=[],
                                placeholder="Toutes les régions")
    min_stations = st.number_input("Nb min. stations (systèmes)", min_value=1, value=5)

# ── Filtrage systèmes ─────────────────────────────────────────────────────────
sys_f = systems[systems["n_stations"] >= min_stations]
if region_sel:
    sys_f = sys_f[sys_f["region"].isin(region_sel)]

# ── KPI ───────────────────────────────────────────────────────────────────────
st.markdown("#### Vue d'ensemble du réseau national")

n_sys    = len(sys_f)
n_reg    = sys_f["region"].nunique()
n_dep    = sys_f["department"].nunique()
tot_sta  = int(sys_f["n_stations"].sum(skipna=True))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Systèmes GBFS actifs", f"{n_sys}")
k2.metric("Régions couvertes", f"{n_reg}")
k3.metric("Départements couverts", f"{n_dep}")
k4.metric("Stations totales (catalogue)", f"{tot_sta:,}")

if not city_df.empty and "fub_score_2023" in city_df.columns:
    fub_valid = city_df["fub_score_2023"].dropna()
    best_city = city_df.loc[city_df["fub_score_2023"].idxmax(), "city"]
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Villes avec score FUB", f"{len(fub_valid)}")
    k6.metric("Score FUB moyen", f"{fub_valid.mean():.2f} / 6")
    k7.metric("Meilleure ville (FUB)", best_city)
    if "emp_part_velo_2019" in city_df.columns:
        emp_valid = city_df["emp_part_velo_2019"].dropna()
        k8.metric("Part modale vélo moy. (EMP 2019)", f"{emp_valid.mean():.1f} %")

st.divider()

# ── Catalogue des systèmes par région ─────────────────────────────────────────
st.subheader("Catalogue des systèmes GBFS par région")
st.caption(
    "Chaque barre représente le nombre de systèmes actifs dans la région. "
    "La couleur indique le nombre total de stations. "
    f"Seuil minimum : {min_stations} stations par système."
)

reg_agg = (
    sys_f.groupby("region")
    .agg(n_systemes=("system_id", "count"), n_stations=("n_stations", "sum"))
    .reset_index()
    .sort_values("n_systemes", ascending=False)
)

fig_reg = px.bar(
    reg_agg,
    x="n_systemes",
    y="region",
    orientation="h",
    color="n_stations",
    color_continuous_scale="Blues",
    text="n_systemes",
    labels={"region": "Région", "n_systemes": "Systèmes", "n_stations": "Stations totales"},
    height=max(320, len(reg_agg) * 28),
)
fig_reg.update_traces(textposition="outside")
fig_reg.update_layout(
    coloraxis_colorbar=dict(title="Stations"),
    margin=dict(l=10, r=60, t=10, b=10),
    plot_bgcolor="white",
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_reg, use_container_width=True)

# ── Top systèmes ──────────────────────────────────────────────────────────────
left_sys, right_sys = st.columns([3, 2])

with left_sys:
    st.subheader("Top 15 systèmes par nombre de stations")
    top_sys = (
        sys_f.nlargest(15, "n_stations")[["title", "city", "region", "n_stations", "source"]]
        .rename(columns={"title": "Système", "city": "Ville", "region": "Région",
                         "n_stations": "Stations", "source": "Source"})
    )
    st.dataframe(top_sys, use_container_width=True, hide_index=True,
                 column_config={"Stations": st.column_config.ProgressColumn(
                     "Stations", min_value=0,
                     max_value=int(sys_f["n_stations"].max()), format="%d")})

with right_sys:
    st.subheader("Répartition par source de données")
    src = sys_f["source"].value_counts().reset_index()
    src.columns = ["Source", "Systèmes"]
    fig_src = px.pie(src, names="Source", values="Systèmes",
                     color_discrete_sequence=["#1A6FBF", "#4A9FDF"],
                     hole=0.5, height=320)
    fig_src.update_traces(textinfo="percent+label")
    fig_src.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_src, use_container_width=True)

st.divider()

# ── Indicateurs comparatifs des villes ────────────────────────────────────────
if city_df.empty:
    st.info("Données de mobilité par ville non disponibles.")
    st.stop()

st.subheader("Indicateurs comparatifs des villes")
st.caption(
    "Fusion de cinq sources d'indicateurs à l'échelle des villes. "
    "Les cellules vides indiquent l'absence de donnée pour cette ville dans la source concernée."
)

col_labels = {
    "city": "Ville",
    "fub_score_2023": "FUB 2023 (/6)",
    "fub_rank_2023": "Rang FUB",
    "emp_part_velo_2019": "Part modale vélo % (EMP 2019)",
    "infra_cyclable_km": "Infra cyclable (km)",
    "infra_cyclable_km_per_km2": "Infra/km² (Cerema)",
    "baac_accidents_cyclistes": "Accidents cyclistes",
    "baac_accidents_cyclistes_per_100k": "Accidents/100k hab.",
    "eco_avg_daily_bike_counts": "Comptages vélo/jour",
}
disp_cols = [c for c in col_labels if c in city_df.columns]
disp = city_df[disp_cols].rename(columns=col_labels).sort_values(
    "FUB 2023 (/6)" if "FUB 2023 (/6)" in [col_labels[c] for c in disp_cols] else disp_cols[1],
    ascending=False,
)
st.dataframe(disp, use_container_width=True, hide_index=True)

st.divider()

# ── FUB Baromètre — classement ────────────────────────────────────────────────
if "fub_score_2023" in city_df.columns:
    st.subheader("Classement FUB Baromètre 2023")
    st.caption(
        "Le FUB Baromètre mesure la perception de la qualité cyclable par les usagers "
        "(note de 1 à 6). Données issues de l'enquête nationale 2023."
    )
    fub_sorted = city_df[["city", "fub_score_2023"]].dropna().sort_values(
        "fub_score_2023", ascending=False
    )
    fig_fub = px.bar(
        fub_sorted,
        x="fub_score_2023",
        y="city",
        orientation="h",
        color="fub_score_2023",
        color_continuous_scale="Blues",
        text="fub_score_2023",
        labels={"city": "Ville", "fub_score_2023": "Score FUB 2023"},
        height=max(400, len(fub_sorted) * 24),
    )
    fig_fub.update_traces(texttemplate="%{x:.2f}", textposition="outside")
    fig_fub.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 6.5], title="Score FUB (1 = hostile, 6 = accueillant)"),
    )
    st.plotly_chart(fig_fub, use_container_width=True)

st.divider()

# ── Scatter : FUB vs part modale ──────────────────────────────────────────────
if {"fub_score_2023", "emp_part_velo_2019"}.issubset(city_df.columns):
    st.subheader("Perception cycliste (FUB) et usage effectif (EMP 2019)")
    st.caption(
        "Chaque point est une ville. Une corrélation positive indiquerait que "
        "les villes perçues comme cyclables sont aussi celles avec une forte part modale vélo. "
        "Les valeurs extrêmes révèlent des villes sur- ou sous-performantes."
    )
    sc_df = city_df[["city", "fub_score_2023", "emp_part_velo_2019",
                      "infra_cyclable_km"]].dropna(subset=["fub_score_2023", "emp_part_velo_2019"])

    fig_sc = px.scatter(
        sc_df,
        x="fub_score_2023",
        y="emp_part_velo_2019",
        text="city",
        size="infra_cyclable_km" if "infra_cyclable_km" in sc_df.columns else None,
        size_max=30,
        color="fub_score_2023",
        color_continuous_scale="Blues",
        labels={
            "fub_score_2023": "Score FUB 2023",
            "emp_part_velo_2019": "Part modale vélo % (EMP 2019)",
            "infra_cyclable_km": "Infra cyclable (km)",
        },
        height=480,
    )
    fig_sc.update_traces(textposition="top center", marker_opacity=0.8)
    fig_sc.add_vline(x=sc_df["fub_score_2023"].mean(), line_dash="dot",
                     line_color="#1A6FBF", opacity=0.5, annotation_text="Moy. FUB")
    fig_sc.add_hline(y=sc_df["emp_part_velo_2019"].mean(), line_dash="dot",
                     line_color="#c0392b", opacity=0.5, annotation_text="Moy. part modale")
    fig_sc.update_layout(plot_bgcolor="white", coloraxis_showscale=False,
                         margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_sc, use_container_width=True)
    st.divider()

# ── Infrastructure vs accidentologie ─────────────────────────────────────────
if {"infra_cyclable_km_per_km2", "baac_accidents_cyclistes_per_100k"}.issubset(city_df.columns):
    st.subheader("Infrastructure cyclable et sinistralité")
    st.caption(
        "Densité d'infrastructure cyclable (Cerema) versus taux d'accidents "
        "cyclistes pour 100 000 habitants (BAAC). Un effet protecteur de "
        "l'infrastructure serait visible dans le quadrant supérieur gauche "
        "(forte infra, faible accidentologie)."
    )
    safety_df = city_df[["city", "infra_cyclable_km_per_km2",
                          "baac_accidents_cyclistes_per_100k",
                          "emp_part_velo_2019"]].dropna(
        subset=["infra_cyclable_km_per_km2", "baac_accidents_cyclistes_per_100k"]
    )
    fig_saf = px.scatter(
        safety_df,
        x="infra_cyclable_km_per_km2",
        y="baac_accidents_cyclistes_per_100k",
        text="city",
        color="emp_part_velo_2019" if "emp_part_velo_2019" in safety_df.columns else None,
        color_continuous_scale="Greens",
        labels={
            "infra_cyclable_km_per_km2": "Densité infrastructure (km/km²)",
            "baac_accidents_cyclistes_per_100k": "Accidents cyclistes / 100k hab.",
            "emp_part_velo_2019": "Part modale % (EMP)",
        },
        height=480,
    )
    fig_saf.update_traces(textposition="top center", marker_opacity=0.85)
    fig_saf.add_vline(x=safety_df["infra_cyclable_km_per_km2"].mean(), line_dash="dot",
                      line_color="#1A6FBF", opacity=0.5, annotation_text="Moy. infra")
    fig_saf.add_hline(y=safety_df["baac_accidents_cyclistes_per_100k"].mean(), line_dash="dot",
                      line_color="#c0392b", opacity=0.5, annotation_text="Moy. accidents")
    fig_saf.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_saf, use_container_width=True)
