"""
5_Mobilite_France.py — Triangulation multi-sources des indicateurs de mobilité cyclable nationale.
Combine le catalogue GBFS (122 systèmes) avec FUB 2023, EMP 2019, BAAC, Cerema, Eco-compteurs.
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
    page_title="Indicateurs Nationaux — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Triangulation Multi-Sources des Indicateurs Nationaux")
st.caption("Axe de Recherche 4 : Validation Croisée et Positionnement Comparatif des Agglomérations Françaises")

abstract_box(
    "<b>Question de recherche :</b> Les indicateurs Gold Standard construits à partir "
    "des données GBFS enrichies sont-ils cohérents avec les mesures indépendantes "
    "issues d'enquêtes déclaratives et de bases administratives ?<br><br>"
    "Cette analyse croise le catalogue GBFS national — 122 systèmes actifs collectés "
    "via MobilityData et la collecte manuelle (Notebook 20) — avec cinq sources "
    "d'indicateurs indépendantes : le <em>FUB Baromètre 2023</em> (perception déclarative "
    "de la qualité cyclable, $\\in [1, 6]$), l'<em>EMP 2019</em> (part modale vélo mesurée "
    "par enquête), la base <em>BAAC</em> (sinistralité cycliste objective), "
    "les données <em>Cerema</em> (linéaire d'infrastructure cyclable) et les "
    "<em>Eco-compteurs</em> (fréquentation observée). "
    "L'objectif est de valider les dimensions Gold Standard par triangulation "
    "avec ces sources externes et de caractériser les disparités inter-urbaines "
    "selon une approche multi-sources indépendante — condition nécessaire à la "
    "validité de construit du modèle IMD."
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
    min_stations = st.number_input("Seuil min. stations (systèmes)", min_value=1, value=5)

# ── Filtrage systèmes ─────────────────────────────────────────────────────────
sys_f = systems[systems["n_stations"] >= min_stations]
if region_sel:
    sys_f = sys_f[sys_f["region"].isin(region_sel)]

# ── Section 1 — Vue d'ensemble ────────────────────────────────────────────────
section(1, "Couverture Nationale — 122 Systèmes GBFS Actifs sur le Territoire Français")

st.markdown(r"""
Le catalogue GBFS français recense l'ensemble des systèmes de vélos en libre-service
opérationnels ayant publié un flux conforme au standard *General Bikeshare Feed Specification*
(v2.x ou v3.0). Ce catalogue constitue le périmètre d'audit initial, avant application
du protocole de purge en 5 étapes documenté dans la page *Gold Standard*.
""")

n_sys   = len(sys_f)
n_reg   = sys_f["region"].nunique()
n_dep   = sys_f["department"].nunique()
tot_sta = int(sys_f["n_stations"].sum(skipna=True))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Systèmes GBFS actifs", f"{n_sys}")
k2.metric("Régions couvertes", f"{n_reg}")
k3.metric("Départements couverts", f"{n_dep}")
k4.metric("Stations totales (catalogue brut)", f"{tot_sta:,}")

if not city_df.empty and "fub_score_2023" in city_df.columns:
    fub_valid = city_df["fub_score_2023"].dropna()
    best_city = city_df.loc[city_df["fub_score_2023"].idxmax(), "city"]
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Agglomérations avec score FUB 2023", f"{len(fub_valid)}")
    k6.metric("Score FUB moyen national", f"{fub_valid.mean():.2f} / 6")
    k7.metric("Meilleure agglomération (FUB)", best_city)
    if "emp_part_velo_2019" in city_df.columns:
        emp_valid = city_df["emp_part_velo_2019"].dropna()
        k8.metric("Part modale vélo moy. (EMP 2019)", f"{emp_valid.mean():.1f} %")

# ── Section 2 — Catalogue par région ─────────────────────────────────────────
st.divider()
section(2, "Catalogue GBFS — Répartition Régionale des Systèmes et Concentration de l'Offre")

st.markdown(r"""
La répartition géographique des systèmes GBFS révèle une **concentration structurelle**
de l'offre dans les régions Île-de-France et Nouvelle-Aquitaine, tandis que plusieurs
régions présentent un sous-équipement marqué. Cette asymétrie territoriale constitue
le premier indice d'une fracture socio-spatiale dans l'accès à la micromobilité partagée.
""")

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
    labels={"region": "Région", "n_systemes": "Systèmes GBFS", "n_stations": "Stations totales"},
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
    "**Figure 2.1.** Nombre de systèmes GBFS actifs par région administrative française. "
    "La couleur encode le volume total de stations recensées dans la région. "
    f"Seuil d'inclusion : $\\geq {min_stations}$ stations par système."
)

left_sys, right_sys = st.columns([3, 2])

with left_sys:
    st.markdown("**Top 15 systèmes par volumétrie de stations**")
    top_sys = (
        sys_f.nlargest(15, "n_stations")[["title", "city", "region", "n_stations", "source"]]
        .rename(columns={
            "title": "Système", "city": "Agglomération",
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
    st.caption(
        "**Tableau 2.1.** Les 15 systèmes GBFS les plus importants par volume de stations. "
        "Note : les volumes élevés pour les opérateurs *free-floating* (Pony, Bird) "
        "incluent des stations d'ancrage virtuel potentiellement affectées par "
        "l'anomalie A3 (biais de surcapacité), corrigée dans le Gold Standard."
    )

with right_sys:
    st.markdown("**Répartition par source de collecte**")
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
    st.caption(
        "**Figure 2.2.** Répartition des systèmes par source de collecte. "
        "MobilityData fournit les flux certifiés ; "
        "la collecte manuelle complète les systèmes non encore déclarés sur la plateforme."
    )

# ── Section 3 — Tableau comparatif multi-sources ─────────────────────────────
if city_df.empty:
    st.info(
        "Les données de mobilité multi-sources ne sont pas disponibles. "
        "Vérifiez la présence des fichiers CSV dans `data/external/mobility_sources/`."
    )
    st.stop()

st.divider()
section(3, "Tableau de Triangulation — Cinq Sources d'Indicateurs Croisées par Agglomération")

st.markdown(r"""
La triangulation multi-sources constitue le test de validité externe du modèle Gold Standard.
Si les dimensions construites à partir des données GBFS enrichies (modules 2–4) sont
cohérentes avec les indicateurs issus d'enquêtes indépendantes (FUB, EMP) et de bases
administratives (BAAC, Cerema, Eco-compteurs), la **validité de construit** de l'IMD est établie.
Les cellules vides indiquent l'absence de donnée disponible dans la source concernée
pour l'agglomération correspondante.
""")

col_labels = {
    "city":                             "Agglomération",
    "fub_score_2023":                   "FUB 2023 (/6)",
    "fub_rank_2023":                    "Rang FUB",
    "emp_part_velo_2019":               "Part modale vélo % (EMP 2019)",
    "infra_cyclable_km":                "Infra cyclable (km)",
    "infra_cyclable_km_per_km2":        "Densité infra (km/km²)",
    "baac_accidents_cyclistes":         "Sinistralité cycliste (BAAC)",
    "baac_accidents_cyclistes_per_100k": "Sinistralité/100k hab.",
    "eco_avg_daily_bike_counts":        "Fréquentation (comptages/j)",
}
disp_cols = [c for c in col_labels if c in city_df.columns]
sort_col  = "FUB 2023 (/6)" if "fub_score_2023" in city_df.columns else col_labels[disp_cols[1]]
disp = (
    city_df[disp_cols]
    .rename(columns=col_labels)
    .sort_values(sort_col, ascending=False)
)
st.dataframe(disp, use_container_width=True, hide_index=True)
st.caption(
    "**Tableau 3.1.** Tableau de triangulation multi-sources des indicateurs de mobilité cyclable "
    "par agglomération française. Sources : FUB Baromètre 2023, EMP 2019 (INSEE), "
    "BAAC (ONISR), Cerema, Eco-compteurs. "
    "Cellules vides = absence de donnée dans la source concernée."
)

# ── Section 4 — Baromètre FUB ─────────────────────────────────────────────────
if "fub_score_2023" in city_df.columns:
    st.divider()
    section(4, "FUB Baromètre 2023 — Perception Déclarative de la Qualité Cyclable (/6)")

    st.markdown(r"""
    Le FUB Baromètre 2023 constitue la principale mesure déclarative de la qualité cyclable
    perçue par les usagers en France. Fondé sur une enquête nationale auprès de cyclistes
    et non-cyclistes, il évalue six dimensions : praticabilité, sécurité, confort,
    attractivité, partage de la voirie et potentiel de développement.
    Le score agrégé $\in [1, 6]$ (1 = ville hostile, 6 = ville accueillante) sert ici
    d'indicateur de **validation externe perceptuelle** de l'IMD.
    """)

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
        labels={"city": "Agglomération", "fub_score_2023": "Score FUB 2023"},
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
        "**Figure 4.1.** Classement des agglomérations par score FUB Baromètre 2023. "
        "Cet indicateur déclaratif mesure le *vécu perçu* de la qualité cyclable, "
        "indépendamment de toute mesure physique de l'infrastructure. "
        "Sa corrélation avec l'IMD (construit sur données objectives) teste la validité "
        "de construit du modèle : un $r$ élevé attesterait que les conditions objectives "
        "modélisées se traduisent en expérience cyclable positive."
    )

# ── Section 5 — Perception vs usage effectif ──────────────────────────────────
if {"fub_score_2023", "emp_part_velo_2019"}.issubset(city_df.columns):
    st.divider()
    section(5, "Cohérence Perception × Pratique Réelle — FUB 2023 versus EMP 2019")

    st.markdown(r"""
    La triangulation entre perception déclarative (FUB Baromètre) et pratique comportementale
    (part modale vélo EMP 2019) constitue un test de cohérence critique.
    Une corrélation positive significative entre ces deux indicateurs indépendants
    attesterait de leur validité convergente : les villes perçues comme cyclables sont
    aussi celles où la pratique effective est la plus développée.
    Les agglomérations hors de la diagonale principale représentent des cas d'anomalie
    nécessitant une investigation qualitative :
    **sous-performance** (bonne infrastructure, faible usage — barrières socio-économiques ?)
    ou **sur-performance** (usage élevé malgré des conditions objectives médiocres —
    culture cyclable indépendante de l'infrastructure ?).
    """)

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
            "fub_score_2023":   "Score FUB 2023 (/6) — Perception déclarative",
            "emp_part_velo_2019": "Part modale vélo % (EMP 2019) — Pratique effective",
            "infra_cyclable_km":  "Linéaire d'infrastructure (km)",
        },
        height=480,
    )
    fig_sc.update_traces(textposition="top center", marker_opacity=0.8)
    fig_sc.add_vline(
        x=sc_df["fub_score_2023"].mean(), line_dash="dot",
        line_color="#1A6FBF", opacity=0.5, annotation_text="Moy. nationale FUB",
    )
    fig_sc.add_hline(
        y=sc_df["emp_part_velo_2019"].mean(), line_dash="dot",
        line_color="#c0392b", opacity=0.5, annotation_text="Moy. nationale part modale",
    )
    fig_sc.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "**Figure 5.1.** Score FUB 2023 (axe horizontal, perception déclarative) versus "
        "part modale vélo EMP 2019 (axe vertical, pratique comportementale). "
        "La taille des points est proportionnelle au linéaire d'infrastructure Cerema. "
        "Les lignes pointillées indiquent les moyennes nationales. "
        "Les agglomérations hors-diagonale identifient des situations de "
        "*déserts de mobilité sociale* (faible usage malgré une bonne perception) "
        "ou de *résilience cyclable* (usage élevé malgré une perception dégradée)."
    )

# ── Section 6 — Infrastructure vs sinistralité ─────────────────────────────────
if {"infra_cyclable_km_per_km2", "baac_accidents_cyclistes_per_100k"}.issubset(city_df.columns):
    st.divider()
    section(6, "Effet Protecteur de l'Infrastructure — Densité Cerema × Sinistralité BAAC")

    st.markdown(r"""
    L'hypothèse d'un effet protecteur de l'infrastructure cyclable sur la sinistralité
    — dite hypothèse de *safety in numbers* (*Jacobsen, 2003*) — prédit une relation
    négative entre la densité d'infrastructure et le taux d'accidents cyclistes.
    Ce nuage de points teste cette relation à l'échelle nationale en croisant
    les données Cerema (linéaire d'infrastructure en km/km²) avec les statistiques
    BAAC d'accidents cyclistes normalisées par population (taux pour 100 000 habitants).
    """)

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
            "infra_cyclable_km_per_km2":          "Densité d'infrastructure cyclable (km/km² — Cerema)",
            "baac_accidents_cyclistes_per_100k":  "Taux de sinistralité cycliste (/100k hab. — BAAC)",
            "emp_part_velo_2019":                  "Part modale vélo % (EMP 2019)",
        },
        height=480,
    )
    fig_saf.update_traces(textposition="top center", marker_opacity=0.85)
    fig_saf.add_vline(
        x=safety_df["infra_cyclable_km_per_km2"].mean(), line_dash="dot",
        line_color="#1A6FBF", opacity=0.5, annotation_text="Moy. densité infra",
    )
    fig_saf.add_hline(
        y=safety_df["baac_accidents_cyclistes_per_100k"].mean(), line_dash="dot",
        line_color="#c0392b", opacity=0.5, annotation_text="Moy. sinistralité",
    )
    fig_saf.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_saf, use_container_width=True)
    st.caption(
        "**Figure 6.1.** Densité d'infrastructure cyclable Cerema (axe horizontal, km/km²) "
        "versus taux de sinistralité cycliste BAAC (axe vertical, pour 100 000 habitants). "
        "La couleur encode la part modale vélo (EMP 2019). "
        "Le quadrant supérieur gauche (forte densité, faible sinistralité) valide "
        "l'hypothèse de *safety in numbers* à l'échelle des agglomérations françaises. "
        "Les villes hors-quadrant constituent des anomalies à investiguer qualitativement."
    )
