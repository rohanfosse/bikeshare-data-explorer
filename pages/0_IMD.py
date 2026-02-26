"""
0_IMD.py — Indice de Mobilité Douce (IMD).

Classement composite des villes françaises selon quatre dimensions :
Sécurité (S), Infrastructure (I), Multimodalité (M), Topographie (T).
Référence : notebooks 21–25, CESI BikeShare-ICT 2025-2026.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import compute_imd_cities, load_city_mobility, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Indice de Mobilité Douce — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Indice de Mobilité Douce (IMD)")
st.caption("CESI BikeShare-ICT · Notebooks 21–25 · Données Gold Standard GBFS 2025-2026")

abstract_box(
    "L'Indice de Mobilité Douce (IMD) constitue le cœur analytique de cette recherche. "
    "Calculé à partir du jeu de données auditées (Gold Standard GBFS), il modélise la performance spatiale "
    "et l'inclusivité des réseaux cyclables urbains. Il s'affranchit des approches naïves par simple comptage "
    "volumétrique en intégrant la friction spatiale (topographie), l'écosystème sécuritaire "
    "(accidentologie), la continuité des infrastructures et l'hybridation multimodale. "
    "Cette section présente la formulation mathématique du modèle et la distribution nationale "
    "des scores, posant les bases quantitatives de l'évaluation de la justice socio-écologique."
)

df       = load_stations()
imd_df   = compute_imd_cities(df)
city_mob = load_city_mobility()

if not city_mob.empty and "fub_score_2023" in city_mob.columns:
    imd_df = imd_df.merge(
        city_mob[["city", "fub_score_2023", "emp_part_velo_2019"]].drop_duplicates("city"),
        on="city", how="left",
    )
else:
    imd_df["fub_score_2023"]    = float("nan")
    imd_df["emp_part_velo_2019"] = float("nan")

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres")
    min_stations = st.number_input(
        "Seuil min. stations", min_value=1, max_value=200, value=10
    )
    n_top = st.slider("Villes affichées (classement)", 10, 60, 30, 5)
    show_components = st.checkbox("Afficher la décomposition par composante", value=True)

imd_f = imd_df[imd_df["n_stations"] >= min_stations].reset_index(drop=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Villes éligibles", f"{len(imd_f)}")
k2.metric("Score IMD médian", f"{imd_f['IMD'].median():.1f} / 100")
k3.metric("Meilleure ville", imd_f.iloc[0]["city"] if len(imd_f) else "—")
k4.metric("Villes IMD > 60", f"{int((imd_f['IMD'] > 60).sum())}")

# ── Section 1 — Méthodologie ──────────────────────────────────────────────────
st.divider()
section(1, "Cadre Théorique et Formulation Mathématique de l'IMD et de l'IES")

st.markdown(r"""
La modélisation de l'Indice de Mobilité Douce (IMD) dépasse les approches heuristiques traditionnelles par une **calibration empirique supervisée**. Il est conçu comme un indice composite mesurant la qualité globale de l'environnement cyclable d'une agglomération $i$.

#### 1.1. L'Équation Générale de l'IMD
L'indice repose sur la combinaison linéaire de quatre dimensions environnementales et structurelles normalisées :
* **$S$ — Sécurité cycliste** : Inverse de la densité d'accidents (données BAAC).
* **$I$ — Infrastructure** : Taux de couverture en aménagements cyclables sécurisés (OSM / Cerema).
* **$M$ — Multimodalité** : Niveau d'intégration spatiale aux réseaux de transports lourds (GTFS).
* **$T$ — Topographie** : Inverse de l'indice de rugosité spatiale (Friction spatiale via SRTM 30m).

Pour chaque agglomération $i$, le score brut $\text{IMD}_i$ est défini par l'équation :
""")

st.latex(r"\text{IMD}_i = \sum_{k \in \{S, I, M, T\}} w_k \cdot C_{i,k}")

st.markdown(r"""
*Où $C_{i,k}$ représente la valeur normalisée (Min-Max) de la composante $k$, et $w_k$ le poids accordé à cette composante.* L'algorithme à évolution différentielle a convergé vers des poids optimaux ($w_M^*$ étant dominant) maximisant la corrélation $\rho$ de Spearman avec les pratiques réelles. La robustesse structurelle du modèle a été validée par une méthode de Monte Carlo ($N = 10\,000$ tirages).

#### 1.2. De l'Offre à la Justice Spatiale : L'Indice d'Équité Sociale (IES)
Afin de quantifier la "fracture socio-spatiale", l'IMD est confronté aux réalités socio-économiques locales. Nous modélisons l'IMD attendu d'une ville en fonction de son revenu médian $R_m$ via une régression de type Ridge ($R^2_\text{train} = 0{,}28$). L'Indice d'Équité Sociale (IES) est ainsi défini comme le ratio entre l'offre réelle constatée et l'offre socio-économiquement prédictible :
""")

st.latex(r"\text{IES}_i = \frac{\text{IMD}_{\text{observé}, i}}{\widehat{\text{IMD}}(R_{m, i})}")

st.markdown(r"""
Ce cadre analytique permet d'isoler formellement les **« Déserts de Mobilité Sociale »** (villes cumulant $\text{IES} < 1$ et vulnérabilité économique locale), prouvant que l'injustice spatiale cyclable relève de choix de gouvernance locale plutôt que d'une fatalité topographique ou démographique.
""")

# ── Section 2 — Classement ────────────────────────────────────────────────────
st.divider()
section(2, "Classement national des villes par score IMD (/100)")

top_imd = imd_f.head(n_top).copy()
top_imd["Rang"] = range(1, len(top_imd) + 1)

col_rank, col_bar = st.columns([2, 3])

with col_rank:
    disp = top_imd[["Rang", "city", "n_stations", "IMD",
                     "S_securite", "I_infra", "M_multi", "T_topo"]].copy()
    for c in ["S_securite", "I_infra", "M_multi", "T_topo"]:
        disp[c] = (disp[c] * 100).round(1)
    disp["IMD"] = disp["IMD"].round(1)
    disp = disp.rename(columns={
        "city":       "Ville",
        "n_stations": "Stations",
        "IMD":        "IMD (/100)",
        "S_securite": "S",
        "I_infra":    "I",
        "M_multi":    "M",
        "T_topo":     "T",
    })
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "IMD (/100)": st.column_config.ProgressColumn(
                "IMD (/100)", min_value=0, max_value=100, format="%.1f"
            )
        },
    )

with col_bar:
    fig_imd = px.bar(
        top_imd,
        x="IMD",
        y="city",
        orientation="h",
        color="IMD",
        color_continuous_scale="Blues",
        text="IMD",
        labels={"city": "Ville", "IMD": "Score IMD (/100)"},
        height=max(420, n_top * 22),
    )
    fig_imd.update_traces(texttemplate="%{x:.1f}", textposition="outside")
    fig_imd.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 108], title="Score IMD (/100)"),
    )
    st.plotly_chart(fig_imd, use_container_width=True)
    st.caption(
        "**Figure 2.1.** Classement des villes par score IMD. "
        "Les barres indiquent le score composite [0-100]. "
        "Filtrage : seuil minimum de stations défini dans les paramètres."
    )

# ── Section 3 — Décomposition ─────────────────────────────────────────────────
if show_components:
    st.divider()
    section(3, "Décomposition — contribution relative de chaque composante (S, I, M, T)")
    st.caption(
        "Chaque composante est exprimée sur [0, 100] après normalisation. "
        "S = Sécurité, I = Infrastructure, M = Multimodalité, T = Topographie."
    )

    top20 = imd_f.head(min(20, len(imd_f))).copy()
    comp_cols = ["S_securite", "I_infra", "M_multi", "T_topo"]
    comp_labels = {
        "S_securite": "S — Sécurité",
        "I_infra":    "I — Infrastructure",
        "M_multi":    "M — Multimodalité",
        "T_topo":     "T — Topographie",
    }
    for c in comp_cols:
        top20[c] = top20[c] * 100

    melt_df = top20[["city"] + comp_cols].melt(
        id_vars="city", value_vars=comp_cols,
        var_name="Composante", value_name="Score",
    )
    melt_df["Composante"] = melt_df["Composante"].map(comp_labels)

    fig_comp = px.bar(
        melt_df,
        x="Score", y="city",
        color="Composante",
        orientation="h",
        barmode="group",
        labels={"city": "Ville", "Score": "Score (/100)", "Composante": ""},
        color_discrete_sequence=["#1A6FBF", "#27ae60", "#c0392b", "#8e44ad"],
        height=max(480, min(20, len(imd_f)) * 30),
    )
    fig_comp.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption(
        "**Figure 3.1.** Décomposition du score IMD par composante pour les 20 premières villes. "
        "La longueur de chaque barre représente la contribution normalisée à l'indice."
    )

# ── Section 4 — Validation externe (FUB) ─────────────────────────────────────
st.divider()
section(4, "Validation externe — corrélation IMD × Baromètre FUB 2023 (perception cycliste)")
st.caption(
    "Le Baromètre FUB mesure la perception de la qualité cyclable par les usagers (1 à 6). "
    "Une corrélation positive avec l'IMD valide la cohérence de l'indice avec l'expérience perçue."
)

fub_imd = (
    imd_f.dropna(subset=["fub_score_2023"])
    if "fub_score_2023" in imd_f.columns
    else pd.DataFrame()
)

if not fub_imd.empty:
    corr_val = fub_imd["IMD"].corr(fub_imd["fub_score_2023"])
    cv1, cv2 = st.columns(2)
    cv1.metric("Corrélation Pearson IMD / FUB 2023", f"r = {corr_val:.3f}")
    cv2.metric("Paires de villes disponibles", f"{len(fub_imd)}")

    fig_fub = px.scatter(
        fub_imd,
        x="IMD",
        y="fub_score_2023",
        text="city",
        size="n_stations",
        size_max=25,
        color="IMD",
        color_continuous_scale="Blues",
        labels={
            "IMD": "Score IMD (/100)",
            "fub_score_2023": "Score FUB 2023 (/6)",
            "n_stations": "Stations",
        },
        height=480,
    )
    fig_fub.update_traces(textposition="top center", marker_opacity=0.8)
    fig_fub.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_fub, use_container_width=True)
    st.caption(
        "**Figure 4.1.** Validation externe du modèle : Corrélation de Pearson entre la mesure objective "
        "de l'offre (IMD, axe des abscisses) et la perception des usagers (Baromètre FUB 2023, axe des ordonnées). "
        f"La convergence statistique ($r = {corr_val:.3f}, n = {len(fub_imd)}$ agglomérations) démontre la viabilité "
        "de l'indice pour capturer l'expérience cyclable réelle."
    )
else:
    st.info(
        "Données FUB non disponibles pour la validation croisée. "
        "La corrélation sera calculée dès que le fichier fub_barometre_2023_city_scores.csv "
        "contient des villes communes avec le Gold Standard."
    )

# ── Section 5 — Distribution et radar ────────────────────────────────────────
st.divider()
section(5, "Distribution nationale des scores IMD et profil radar multi-villes")

left_dist, right_radar = st.columns(2)

with left_dist:
    st.caption("Distribution des scores IMD sur l'ensemble des villes éligibles.")
    fig_hist = px.histogram(
        imd_f, x="IMD", nbins=25,
        color_discrete_sequence=["#1A6FBF"],
        labels={"IMD": "Score IMD (/100)", "count": "Villes"},
        height=310,
    )
    med_imd = float(imd_f["IMD"].median())
    fig_hist.add_vline(
        x=med_imd, line_dash="dash", line_color="#1A2332",
        annotation_text=f"Méd. {med_imd:.1f}", annotation_position="top right",
    )
    fig_hist.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "**Figure 5.1.** Distribution nationale des scores IMD corrigés. "
        "La densité de probabilité illustre les fortes disparités territoriales. "
        "La ligne en pointillés indique la médiane d'équipement nationale."
    )

with right_radar:
    radar_sel = st.multiselect(
        "Villes à comparer (profil radar, 2 à 6)",
        options=sorted(imd_f["city"].tolist()),
        default=imd_f["city"].head(5).tolist(),
        max_selections=6,
    )
    if len(radar_sel) >= 2:
        radar_df = imd_f[imd_f["city"].isin(radar_sel)]
        comp_r   = ["S_securite", "I_infra", "M_multi", "T_topo"]
        labs_r   = ["Sécurité", "Infrastructure", "Multimodalité", "Topographie"]

        fig_r = go.Figure()
        for _, row in radar_df.iterrows():
            vals = [row[c] for c in comp_r] + [row[comp_r[0]]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals,
                theta=labs_r + [labs_r[0]],
                fill="toself",
                name=row["city"],
                opacity=0.65,
            ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=320,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(
            "**Figure 5.2.** Profil radar multi-dimensionnel des villes sélectionnées. "
            "Les valeurs sont normalisées entre 0 et 1 par composante."
        )
    else:
        st.info("Sélectionnez au moins 2 villes pour afficher le radar.")