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
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_city_mobility, load_systems_catalog
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Mobilité nationale — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Indicateurs nationaux de mobilité douce")
st.caption("Gold Standard GBFS · CESI BikeShare-ICT 2025-2026")

abstract_box(
    "Cette analyse croise le catalogue GBFS France — 122 systèmes actifs couvrant "
    "l'ensemble du territoire national (notebook 20) — avec cinq sources d'indicateurs "
    "à l'échelle des villes : le <em>FUB Baromètre 2023</em> (perception cycliste, note /6), "
    "l'<em>EMP 2019</em> (part modale vélo), la base <em>BAAC</em> (accidentologie cycliste), "
    "les données <em>Cerema</em> (infrastructure cyclable) et les <em>Eco-compteurs</em> "
    "(fréquentation). L'objectif est de positionner la France dans sa globalité et "
    "de caractériser les disparités inter-urbaines selon des sources indépendantes du Gold Standard."
)

systems  = load_systems_catalog()
city_df  = load_city_mobility()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Filtres")
    regions = sorted(systems["region"].dropna().unique())
    region_sel = st.multiselect(
        "Région(s)", options=regions, default=[], placeholder="Toutes les régions"
    )
    min_stations = st.number_input("Nb min. stations (systèmes)", min_value=1, value=5)

# ── Filtrage systèmes ─────────────────────────────────────────────────────────
sys_f = systems[systems["n_stations"] >= min_stations]
if region_sel:
    sys_f = sys_f[sys_f["region"].isin(region_sel)]

# ── Section 1 — Vue d'ensemble ────────────────────────────────────────────────
section(1, "Vue d'ensemble — 122 systèmes GBFS actifs sur le territoire français")

n_sys   = len(sys_f)
n_reg   = sys_f["region"].nunique()
n_dep   = sys_f["department"].nunique()
tot_sta = int(sys_f["n_stations"].sum(skipna=True))

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

# ── Section 2 — Catalogue par région ─────────────────────────────────────────
st.divider()
section(2, "Catalogue GBFS — répartition régionale des systèmes et stations")

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
    x="n_systemes", y="region",
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
st.caption(
    "Figure 2.1. Nombre de systèmes GBFS actifs par région administrative. "
    "La couleur encode le volume total de stations recensées."
)

# Top systèmes + répartition source
left_sys, right_sys = st.columns([3, 2])

with left_sys:
    st.markdown("**Top 15 systèmes par nombre de stations**")
    top_sys = (
        sys_f.nlargest(15, "n_stations")[["title", "city", "region", "n_stations", "source"]]
        .rename(columns={
            "title": "Système", "city": "Ville",
            "region": "Région", "n_stations": "Stations", "source": "Source",
        })
    )
    st.dataframe(
        top_sys, use_container_width=True, hide_index=True,
        column_config={
            "Stations": st.column_config.ProgressColumn(
                "Stations", min_value=0,
                max_value=int(sys_f["n_stations"].max()), format="%d",
            )
        },
    )
    st.caption("Tableau 2.1. Les 15 systèmes GBFS les plus importants par volume de stations.")

with right_sys:
    st.markdown("**Répartition par source de données**")
    src = sys_f["source"].value_counts().reset_index()
    src.columns = ["Source", "Systèmes"]
    fig_src = px.pie(
        src, names="Source", values="Systèmes",
        color_discrete_sequence=["#1A6FBF", "#4A9FDF"],
        hole=0.5, height=280,
    )
    fig_src.update_traces(textinfo="percent+label")
    fig_src.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_src, use_container_width=True)
    st.caption("Figure 2.2. Répartition des systèmes par source (MobilityData vs GBFS Manuel).")

# ── Section 3 — Indicateurs comparatifs des villes ───────────────────────────
if city_df.empty:
    st.info("Données de mobilité par ville non disponibles.")
    st.stop()

st.divider()
section(3, "Tableau comparatif — cinq sources d'indicateurs croisées par ville")

st.caption(
    "Fusion des cinq sources d'indicateurs à l'échelle des villes. "
    "Les cellules vides indiquent l'absence de donnée dans la source concernée."
)

col_labels = {
    "city":                             "Ville",
    "fub_score_2023":                   "FUB 2023 (/6)",
    "fub_rank_2023":                    "Rang FUB",
    "emp_part_velo_2019":               "Part modale vélo % (EMP 2019)",
    "infra_cyclable_km":                "Infra cyclable (km)",
    "infra_cyclable_km_per_km2":        "Infra/km² (Cerema)",
    "baac_accidents_cyclistes":         "Accidents cyclistes",
    "baac_accidents_cyclistes_per_100k": "Accidents/100k hab.",
    "eco_avg_daily_bike_counts":        "Comptages vélo/jour",
}
disp_cols = [c for c in col_labels if c in city_df.columns]
sort_col  = "FUB 2023 (/6)" if "fub_score_2023" in city_df.columns else disp_cols[1]
disp = (
    city_df[disp_cols]
    .rename(columns=col_labels)
    .sort_values(sort_col, ascending=False)
)
st.dataframe(disp, use_container_width=True, hide_index=True)
st.caption(
    "Tableau 3.1. Indicateurs comparatifs des villes françaises. "
    "Sources : FUB Baromètre 2023, EMP 2019, BAAC, Cerema, Eco-compteurs."
)

# ── Section 4 — Classement FUB ───────────────────────────────────────────────
if "fub_score_2023" in city_df.columns:
    st.divider()
    section(4, "FUB Baromètre 2023 — classement des villes par perception cycliste (/6)")

    st.caption(
        "Le FUB Baromètre mesure la perception de la qualité cyclable par les usagers (1 à 6). "
        "Les données sont issues de l'enquête nationale 2023."
    )
    fub_sorted = city_df[["city", "fub_score_2023"]].dropna().sort_values(
        "fub_score_2023", ascending=False
    )
    fig_fub = px.bar(
        fub_sorted,
        x="fub_score_2023", y="city",
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
    st.caption(
        "Figure 4.1. Classement des villes par score FUB Baromètre 2023. "
        "Les barres indiquent la note de perception cyclable (/6)."
    )

# ── Section 5 — Perception vs usage effectif ─────────────────────────────────
if {"fub_score_2023", "emp_part_velo_2019"}.issubset(city_df.columns):
    st.divider()
    section(5, "Perception cycliste (FUB) × usage réel (EMP 2019) — cohérence des indicateurs")

    st.caption(
        "Chaque point représente une ville. "
        "Une corrélation positive indiquerait que les villes perçues comme cyclables "
        "sont aussi celles avec une forte part modale vélo. "
        "Les valeurs extrêmes révèlent des villes sur- ou sous-performantes."
    )
    sc_df = city_df[
        ["city", "fub_score_2023", "emp_part_velo_2019", "infra_cyclable_km"]
    ].dropna(subset=["fub_score_2023", "emp_part_velo_2019"])

    fig_sc = px.scatter(
        sc_df,
        x="fub_score_2023", y="emp_part_velo_2019",
        text="city",
        size="infra_cyclable_km" if "infra_cyclable_km" in sc_df.columns else None,
        size_max=30,
        color="fub_score_2023",
        color_continuous_scale="Blues",
        labels={
            "fub_score_2023":   "Score FUB 2023",
            "emp_part_velo_2019": "Part modale vélo % (EMP 2019)",
            "infra_cyclable_km":  "Infra cyclable (km)",
        },
        height=480,
    )
    fig_sc.update_traces(textposition="top center", marker_opacity=0.8)
    fig_sc.add_vline(
        x=sc_df["fub_score_2023"].mean(), line_dash="dot",
        line_color="#1A6FBF", opacity=0.5, annotation_text="Moy. FUB",
    )
    fig_sc.add_hline(
        y=sc_df["emp_part_velo_2019"].mean(), line_dash="dot",
        line_color="#c0392b", opacity=0.5, annotation_text="Moy. part modale",
    )
    fig_sc.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "Figure 5.1. Score FUB 2023 (axe horizontal) versus part modale vélo EMP 2019 (axe vertical). "
        "La taille des points est proportionnelle au linéaire d'infrastructure cyclable (Cerema). "
        "Les lignes pointillées indiquent les moyennes nationales."
    )

# ── Section 6 — Infrastructure vs accidentologie ─────────────────────────────
if {"infra_cyclable_km_per_km2", "baac_accidents_cyclistes_per_100k"}.issubset(city_df.columns):
    st.divider()
    section(6, "Infrastructure cyclable (Cerema) × sinistralité cycliste (BAAC) — effet protecteur")

    st.caption(
        "Densité d'infrastructure cyclable (Cerema) versus taux d'accidents cyclistes "
        "pour 100 000 habitants (BAAC). Un effet protecteur de l'infrastructure "
        "serait visible dans le quadrant supérieur gauche (forte infra, faible accidentologie)."
    )
    safety_df = city_df[
        ["city", "infra_cyclable_km_per_km2",
         "baac_accidents_cyclistes_per_100k", "emp_part_velo_2019"]
    ].dropna(subset=["infra_cyclable_km_per_km2", "baac_accidents_cyclistes_per_100k"])

    fig_saf = px.scatter(
        safety_df,
        x="infra_cyclable_km_per_km2",
        y="baac_accidents_cyclistes_per_100k",
        text="city",
        color="emp_part_velo_2019" if "emp_part_velo_2019" in safety_df.columns else None,
        color_continuous_scale="Greens",
        labels={
            "infra_cyclable_km_per_km2":          "Densité infrastructure (km/km²)",
            "baac_accidents_cyclistes_per_100k":  "Accidents cyclistes / 100k hab.",
            "emp_part_velo_2019":                  "Part modale % (EMP)",
        },
        height=480,
    )
    fig_saf.update_traces(textposition="top center", marker_opacity=0.85)
    fig_saf.add_vline(
        x=safety_df["infra_cyclable_km_per_km2"].mean(), line_dash="dot",
        line_color="#1A6FBF", opacity=0.5, annotation_text="Moy. infra",
    )
    fig_saf.add_hline(
        y=safety_df["baac_accidents_cyclistes_per_100k"].mean(), line_dash="dot",
        line_color="#c0392b", opacity=0.5, annotation_text="Moy. accidents",
    )
    fig_saf.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_saf, use_container_width=True)
    st.caption(
        "Figure 6.1. Densité d'infrastructure cyclable (Cerema) versus taux d'accidents "
        "cyclistes pour 100 000 habitants (BAAC). "
        "La couleur encode la part modale vélo (EMP 2019)."
    )
