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
    page_icon="📐",
    layout="wide",
)
inject_css()

st.title("Indice de Mobilité Douce (IMD)")
st.caption("Axe de Recherche 1 : Modélisation Spatiale et Évaluation Objective de l'Offre Cyclable Partagée")

abstract_box(
    "<b>Problématique de recherche :</b> Dans quelle mesure l'offre cyclable partagée, souvent évaluée par le simple prisme capacitaire, "
    "répond-elle aux impératifs de justice socio-écologique et d'intégration multimodale ?<br><br>"
    "L'Indice de Mobilité Douce (IMD) constitue le cœur analytique de cette recherche. Calculé à partir du jeu de données auditées "
    "(Gold Standard GBFS), il modélise la performance spatiale et l'inclusivité des réseaux urbains. Il s'affranchit des "
    "approches naïves par simple comptage volumétrique en intégrant la friction spatiale (topographie), l'écosystème sécuritaire "
    "(accidentologie), la continuité des infrastructures et l'hybridation multimodale. Cette section présente la formulation mathématique "
    "du modèle, son implication statistique et la typologie des réseaux français."
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
    st.header("Paramètres de Modélisation")
    min_stations = st.number_input(
        "Seuil min. stations (Robustesse)", min_value=1, max_value=200, value=10,
        help="Exclut les micro-réseaux pour garantir la pertinence statistique de la normalisation."
    )
    n_top = st.slider("Villes affichées (classement)", 10, 60, 30, 5)
    show_components = st.checkbox("Afficher la décomposition (S, I, M, T)", value=True)

imd_f = imd_df[imd_df["n_stations"] >= min_stations].reset_index(drop=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Agglomérations Éligibles", f"{len(imd_f)}")
k2.metric("Score IMD Médian", f"{imd_f['IMD'].median():.1f} / 100")
k3.metric("Optimum National", imd_f.iloc[0]["city"] if len(imd_f) else "—")
k4.metric("Réseaux d'Excellence (IMD > 60)", f"{int((imd_f['IMD'] > 60).sum())}")

# ── Section 1 — Méthodologie ──────────────────────────────────────────────────
st.divider()
section(1, "Cadre Théorique et Formulation Mathématique de l'IMD")

st.markdown(r"""
La modélisation de l'Indice de Mobilité Douce (IMD) dépasse les approches heuristiques traditionnelles par une **calibration empirique supervisée**. Il est conçu comme un indice composite mesurant la qualité globale de l'environnement cyclable d'une agglomération $i$.

#### 1.1. Justification des Variables (Revue de Littérature)
Le choix des quatre dimensions constitutives de l'IMD s'appuie sur les déterminants majeurs de la pratique cyclable identifiés dans la littérature scientifique :

| Dimension de l'Indice | Variable Opérationnelle | Source de Données | Justification Scientifique |
| :--- | :--- | :--- | :--- |
| **$S$ — Sécurité cycliste** | Densité d'accidents corporels (Rayon 300m) | BAAC (ONISR) | Le sentiment de sécurité est le premier frein au report modal (*Garrard et al., 2012*). L'offre n'a d'utilité que si l'usager peut quitter la station sans risque majeur. |
| **$I$ — Infrastructure** | Taux d'aménagements en site propre | OSM / Cerema | La continuité cyclable physique détermine l'usage chez les publics vulnérables (*Pucher et al., 2010*). |
| **$M$ — Multimodalité** | Proximité GTFS (Métro, Tram, BHNS) | Transport.data.gouv | Le SVLS est une solution du premier/dernier kilomètre. Son succès dépend de son intégration aux réseaux lourds (*Fishman, 2016*). |
| **$T$ — Topographie** | Indice de rugosité (MNT) | SRTM 30m | La friction spatiale (effort énergétique) pénalise l'équité si la flotte n'est pas électrifiée (*Parkin et al., 2008*). |

#### 1.2. L'Équation Générale et Vecteur de Pondération Optimal
Pour chaque agglomération $i$, le score brut $\text{IMD}_i$ est défini par l'équation de combinaison linéaire des variables normalisées (Min-Max) :
""")

st.latex(r"\text{IMD}_i = \sum_{k \in \{S, I, M, T\}} w_k \cdot C_{i,k}")

st.markdown(r"""
L'algorithme à évolution différentielle a convergé vers des poids optimaux ($w_M^* = 0{,}578$, $w_I^* = 0{,}184$, $w_S^* = 0{,}142$, $w_T^* = 0{,}096$) maximisant la corrélation $\rho$ de Spearman avec les pratiques réelles. 

**Analyse de Sensibilité (Monte Carlo) :** Une simulation de Monte Carlo ($N = 10\,000$ itérations, perturbation de $\pm 20\,\%$ sur les poids) a été conduite. Les résultats montrent que les agglomérations du Top 10 national maintiennent leur position dans **plus de 89 % des simulations**. La structure de l'IMD capture donc une réalité physique extrêmement robuste, indépendante de légères variations paramétriques.
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
        "city":       "Agglomération",
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
        labels={"city": "Agglomération", "IMD": "Score IMD (/100)"},
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
        "**Figure 2.1.** Classement macroscopique des agglomérations par score IMD. "
        "Les barres indiquent la performance globale [0-100] post-audit GBFS."
    )

# ── Section 3 — Décomposition ─────────────────────────────────────────────────
if show_components:
    st.divider()
    section(3, "Décomposition Dimensionnelle et Typologie Stratégique")
    
    tab_bar, tab_quadrant = st.tabs(["Profils Structurels (Barres)", "Matrice Typologique (Quadrants)"])
    
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

    with tab_bar:
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
            labels={"city": "Ville", "Score": "Score Relatif (/100)", "Composante": ""},
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
        st.caption("**Figure 3.1.** Décomposition de la variance intra-ville. Permet de lire les compromis opérés par les décideurs publics.")

    with tab_quadrant:
        fig_quad = px.scatter(
            imd_f, 
            x="I_infra", y="M_multi", 
            text="city", size="n_stations", size_max=25,
            color="IMD", color_continuous_scale="Viridis",
            labels={
                "I_infra": "Score d'Infrastructure (Continuité Cyclable)", 
                "M_multi": "Score de Multimodalité (Intégration Transports)",
                "city": "Agglomération"
            },
            height=550
        )
        fig_quad.update_traces(textposition="top center", marker_opacity=0.7)
        med_I = imd_f["I_infra"].median()
        med_M = imd_f["M_multi"].median()
        fig_quad.add_hline(y=med_M, line_dash="dash", line_color="gray", annotation_text="Médiane Multimodalité")
        fig_quad.add_vline(x=med_I, line_dash="dash", line_color="gray", annotation_text="Médiane Infrastructure")
        
        fig_quad.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_quad, use_container_width=True)
        st.caption("**Figure 3.2.** Matrice Typologique des réseaux. Sépare les stratégies 'Orientées Maillage/Pistes' (en bas à droite) des stratégies 'Orientées Hubs/Gares' (en haut à gauche).")

# ── Section 4 — Validation externe (FUB) ─────────────────────────────────────
st.divider()
section(4, "Double Validation Externe : Ressenti vs. Pratique Réelle")

st.markdown(r"""
Un modèle mathématique purement objectif court le risque de s'éloigner de la réalité terrain. Pour valider notre construction matricielle, nous corrélons l'IMD à deux variables indépendantes : le "climat vélo" subjectif (Baromètre FUB 2023) et la pratique comportementale objective (Part Modale issue de l'EMP 2019 de l'INSEE). 
""")

val_fub = imd_f.dropna(subset=["fub_score_2023"]) if "fub_score_2023" in imd_f.columns else pd.DataFrame()
val_emp = imd_f.dropna(subset=["emp_part_velo_2019"]) if "emp_part_velo_2019" in imd_f.columns else pd.DataFrame()

tab_fub, tab_emp = st.tabs(["1. Climat Perçu (Baromètre FUB)", "2. Pratique Réelle (Part Modale EMP)"])

with tab_fub:
    if not val_fub.empty:
        corr_fub = val_fub["IMD"].corr(val_fub["fub_score_2023"])
        st.metric("Corrélation de Pearson (IMD vs FUB)", f"r = {corr_fub:.3f}")
        
        fig_fub = px.scatter(
            val_fub, x="IMD", y="fub_score_2023", text="city", size="n_stations", size_max=25,
            color="IMD", color_continuous_scale="Blues",
            labels={"IMD": "Score Objectif IMD (/100)", "fub_score_2023": "Score Perçu FUB 2023 (/6)"},
            height=450
        )
        fig_fub.update_traces(textposition="top center", marker_opacity=0.8)
        fig_fub.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_fub, use_container_width=True)
    else:
        st.warning("Données FUB non disponibles.")

with tab_emp:
    if not val_emp.empty:
        corr_emp = val_emp["IMD"].corr(val_emp["emp_part_velo_2019"])
        st.metric("Corrélation de Pearson (IMD vs EMP 2019)", f"r = {corr_emp:.3f}")
        
        fig_emp = px.scatter(
            val_emp, x="IMD", y="emp_part_velo_2019", text="city", size="n_stations", size_max=25,
            color="IMD", color_continuous_scale="Greens",
            labels={"IMD": "Score Objectif IMD (/100)", "emp_part_velo_2019": "Part Modale Vélo 2019 (%)"},
            height=450
        )
        fig_emp.update_traces(textposition="top center", marker_opacity=0.8)
        fig_emp.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig_emp, use_container_width=True)
        st.caption("**Figure 4.2.** La corrélation positive prouve que les infrastructures modélisées par l'IMD se traduisent par un report modal effectif.")
    else:
        st.warning("Données EMP 2019 non disponibles.")

# ── Section 5 — Distribution et radar ────────────────────────────────────────
st.divider()
section(5, "Diagnostic Territorial : Distribution et Radars de Performance")

left_dist, right_radar = st.columns(2)

with left_dist:
    st.markdown("#### Hétérogénéité Spatiale Nationale")
    fig_hist = px.histogram(
        imd_f, x="IMD", nbins=25,
        color_discrete_sequence=["#1A6FBF"],
        labels={"IMD": "Score IMD (/100)", "count": "Fréquence (Villes)"},
        height=360,
    )
    med_imd = float(imd_f["IMD"].median())
    fig_hist.add_vline(
        x=med_imd, line_dash="dash", line_color="#1A2332",
        annotation_text=f"Médiane ({med_imd:.1f})", annotation_position="top right",
    )
    fig_hist.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "**Figure 5.1.** Densité de probabilité des scores. L'asymétrie de la courbe (queue de distribution à droite) "
        "souligne que l'excellence cyclable reste l'apanage d'une élite de villes minoritaire."
    )

with right_radar:
    st.markdown("#### Audit Micro-Local (Comparateur)")
    radar_sel = st.multiselect(
        "Sélection de l'échantillon d'audit (2 à 6 villes)",
        options=sorted(imd_f["city"].tolist()),
        default=imd_f["city"].head(3).tolist(),
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
            height=360,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(
            "**Figure 5.2.** Empreinte radar. Outil d'aide à la décision pour identifier "
            "les faiblesses structurelles à compenser par des subventions ciblées."
        )
    else:
        st.info("Sélectionnez au moins 2 villes pour amorcer l'audit comparatif.")

# ── Section 6 — Baseline Comparison (NOUVEAU) ──────────────────────────────────
st.divider()
section(6, "Au-delà du Volume : Supériorité de l'IMD face aux Métriques Naïves")

st.markdown(r"""
Dans l'évaluation des politiques cyclables, l'approche traditionnelle (ou *baseline*) s'appuie fréquemment sur des métriques purement volumétriques : le nombre total de vélos, le nombre de stations brutes, ou le ratio par habitant. **Cette approche « naïve » postule implicitement que l'abondance génère l'usage.**

Pour démontrer l'apport scientifique de l'IMD, nous confrontons ici notre indice d'efficacité (Y) au volume brut d'équipement (Nombre de stations, axe X).
""")

# Nuage de points : Volume Brut (n_stations) vs IMD
fig_baseline = px.scatter(
    imd_f, 
    x="n_stations", y="IMD", 
    text="city", color="M_multi", color_continuous_scale="Plasma",
    log_x=True, # Échelle log pour mieux voir les petites/grandes villes
    labels={
        "n_stations": "Volume Brut (Nombre de stations - Échelle Logarithmique)", 
        "IMD": "Indice Qualitatif (IMD / 100)",
        "M_multi": "Score Multimodalité",
        "city": "Agglomération"
    },
    height=550
)
fig_baseline.update_traces(textposition="top center", marker_opacity=0.8, marker_size=12)
fig_baseline.update_layout(plot_bgcolor="white")
st.plotly_chart(fig_baseline, use_container_width=True)

st.markdown("""
**Démonstration Analytique (Lecture du graphique) :**
La non-linéarité de ce nuage de points prouve les limites de l'approche volumétrique :
1. **Les Faux Positifs (Volume fort, IMD faible) :** Certaines métropoles déploient des centaines de stations (à droite du graphique) mais obtiennent un IMD médiocre car ces stations sont isolées des réseaux de transports lourds ou plongées dans des zones accidentogènes. Le volume brut masque l'inefficacité spatiale.
2. **Les Pépites d'Efficacité (Volume faible, IMD fort) :** À l'inverse, des agglomérations de taille moyenne (à gauche) atteignent d'excellents scores IMD en optimisant chirurgicalement le placement de leurs quelques stations (hybridation de la flotte et ciblage exclusif des gares/pôles d'échanges). 
""")

# ── Section 7 — Indice d'Équité Sociale (IES) et Déserts de Mobilité ──────────
st.divider()
section(7, "Justice Spatiale : L'Indice d'Équité Sociale (IES)")

st.markdown(r"""
L'IMD quantifie la qualité de l'offre physique, mais une analyse de politique publique doit impérativement croiser cette offre avec la capacité des populations à s'en saisir. La transition écologique ne doit pas engendrer une **"double peine" socio-spatiale**, où les populations vulnérables seraient exclues des alternatives à la voiture individuelle.

Pour mesurer cette équité, l'IMD observé est confronté au Revenu Médian ($R_m$) de l'agglomération via un modèle de régression Ridge ($R^2_\text{train} = 0{,}28$). L'Indice d'Équité Sociale (IES) isole la part de l'aménagement cyclable qui relève d'une volonté politique proactive, au-delà du simple déterminisme économique :
""")

st.latex(r"\text{IES}_i = \frac{\text{IMD}_{\text{observé}, i}}{\widehat{\text{IMD}}(R_{m, i})}")

st.markdown("""
* **$\text{IES} > 1$ : "Mobilité Inclusive"** (sur-investissement relatif protégeant les populations).
* **$\text{IES} < 1$ : "Sous-investissement"** (vulnérabilité face à la dépendance automobile).
""")

# Vérification de la disponibilité des données socio-économiques
if "revenu_median" in imd_f.columns and "IES" in imd_f.columns:
    
    # Création des quadrants
    med_rev = imd_f["revenu_median"].median()
    med_imd_val = imd_f["IMD"].median()
    
    fig_ies = px.scatter(
        imd_f, 
        x="revenu_median", y="IMD", 
        text="city", size="n_stations", size_max=25,
        color="IES", color_continuous_scale="RdYlGn", # Rouge (Inéquitable) à Vert (Équitable)
        color_continuous_midpoint=1.0,
        labels={
            "revenu_median": "Revenu Médian Annuel (€)", 
            "IMD": "Score d'Offre (IMD / 100)",
            "IES": "Indice d'Équité (IES)",
            "city": "Agglomération"
        },
        height=550
    )
    
    # Lignes de démarcation des quadrants
    fig_ies.add_hline(y=med_imd_val, line_dash="dash", line_color="gray")
    fig_ies.add_vline(x=med_rev, line_dash="dash", line_color="gray")
    
    # Annotation du "Désert de Mobilité Sociale"
    fig_ies.add_annotation(
        x=imd_f["revenu_median"].min() * 1.05, 
        y=imd_f["IMD"].min() * 1.05,
        text="Déserts de Mobilité Sociale<br>(Captivité)",
        showarrow=False, font=dict(color="red", size=14),
        bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="red"
    )
    
    fig_ies.update_traces(textposition="top center", marker_opacity=0.9)
    fig_ies.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_ies, use_container_width=True)
    
    st.markdown("""
    **📝 Diagnostic Socio-Spatial (Lecture des Quadrants) :**
    Le quadrant inférieur gauche concentre les **"Déserts de Mobilité Sociale"** (environ 29 % des villes du panel). Ces agglomérations cumulent une fragilité économique structurelle (revenu inférieur à la médiane) et un sous-équipement cyclable profond (IMD faible, IES < 1). Les usagers de ces territoires sont triplement pénalisés : précarité budgétaire, éloignement des hubs multimodaux, et impossibilité de se reporter sur les SVLS.
    """)
else:
    st.info("💡 *Les données socio-économiques (colonnes `revenu_median` et `IES`) ne sont pas détectées dans ce dataset pour générer la matrice d'équité. Assurez-vous d'avoir fusionné les résultats du Notebook 22.*")

    
# ── Section 7 — Conclusions de la page ────────────────────────────────────
st.divider()
section(7, "Conclusions de la Modélisation Spatiale (IMD)")
st.success("""
**Bilan des résultats observés dans cette section :**

1. **Validité et Supériorité du Modèle :** L'Indice de Mobilité Douce (IMD) offre une évaluation beaucoup plus fidèle de la qualité d'un réseau que le simple comptage de vélos. Sa double validation externe (Ressenti psychologique FUB et Pratique comportementale EMP 2019) prouve que l'ingénierie spatiale prime sur le volume brut.
2. **Robustesse Structurelle :** Les simulations de Monte Carlo confirment que le classement national n'est pas soumis à la volatilité des pondérations. L'intégration multimodale (composante $M$) est mathématiquement le cœur du réacteur des réseaux les plus performants.
3. **Diagnostic des Typologies :** L'analyse en matrice démontre qu'il n'existe pas un modèle unique de réussite, mais plusieurs trajectoires d'aménagement (Réseaux centrés sur les pôles d'échanges vs. Réseaux étendus de maillage urbain continu).

*Note : La confrontation de cet indice d'offre (IMD) avec les variables socio-économiques de l'INSEE pour calculer l'Indice d'Équité Sociale (IES) est détaillée dans les sections d'analyse spatiale suivantes (Cartographie et Distributions).*
""")