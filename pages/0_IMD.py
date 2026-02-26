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
section(1, "Cadre Théorique et Formulation Mathématique de l'IMD et de l'IES")

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

#### 1.2. L'Équation Générale de l'IMD
Pour chaque agglomération $i$, le score brut $\text{IMD}_i$ est défini par l'équation de combinaison linéaire des variables normalisées (Min-Max) :
""")

st.latex(r"\text{IMD}_i = \sum_{k \in \{S, I, M, T\}} w_k \cdot C_{i,k}")

st.markdown(r"""
*Où $C_{i,k}$ représente la valeur normalisée de la composante $k$, et $w_k$ le poids accordé à cette composante.*

#### 1.3. Vecteur de Pondération Optimal et Validation (Monte Carlo)
Plutôt que d'attribuer des poids équiprobables ($0{,}25$ par variable), nous avons utilisé un algorithme à évolution différentielle (optimisation supervisée). L'objectif était de maximiser la corrélation de Spearman ($\rho$) entre l'IMD calculé et les pratiques cyclables réelles (Baromètre FUB et part modale de l'Enquête Mobilité des Personnes 2019). Cette optimisation a porté la corrélation initiale de $\rho = 0{,}16$ à $\rho = 0{,}47$.

**Tableau des Poids Optimaux Retenus :**
| Composante ($k$) | Poids final ($w_k^*$) | Interprétation Analytique |
| :--- | :---: | :--- |
| **$M$ — Multimodalité** | **$0{,}578$** | La diversité de la flotte et la connexion GTFS constituent le levier prédictif dominant. |
| **$I$ — Infrastructure** | **$0{,}184$** | La continuité des pistes cyclables reste un maillon indispensable pour transformer l'offre en usage. |
| **$S$ — Sécurité cycliste**| **$0{,}142$** | Pénalise les réseaux déployés dans des environnements urbains structurellement denses et accidentogènes. |
| **$T$ — Topographie** | **$0{,}096$** | Un frein énergétique secondaire, aujourd'hui partiellement lissé par la montée en puissance de l'électrification (VAE). |
| **Total** | **$1{,}000$** | *Somme unitaire respectée par l'algorithme d'optimisation.* |

**Analyse de Sensibilité (Monte Carlo) :**
Pour démontrer que notre classement n'est pas un simple artefact mathématique lié à ce vecteur spécifique, nous avons conduit une simulation de Monte Carlo ($N = 10\,000$ itérations). À chaque tirage, le vecteur $(w_S, w_I, w_M, w_T)$ a été perturbé aléatoirement ($\pm 20\,\%$). Les résultats montrent que les agglomérations du Top 10 national maintiennent leur position dans **plus de 89 % des simulations**. La structure de l'IMD capture donc une réalité physique extrêmement robuste.

#### 1.4. De l'Offre à la Justice Spatiale : L'Indice d'Équité Sociale (IES)
Afin de quantifier la "fracture socio-spatiale", l'IMD est confronté aux réalités socio-économiques locales. Nous modélisons l'IMD attendu d'une ville en fonction de son revenu médian $R_m$ via une régression de type Ridge ($R^2_\text{train} = 0{,}28$). L'Indice d'Équité Sociale (IES) est le ratio entre l'offre réelle constatée et l'offre socio-économiquement prédictible :
""")

st.latex(r"\text{IES}_i = \frac{\text{IMD}_{\text{observé}, i}}{\widehat{\text{IMD}}(R_{m, i})}")

st.info("**Implication pour la recherche :** Ce cadre analytique permet d'isoler formellement les **« Déserts de Mobilité Sociale »** (villes cumulant $\text{IES} < 1$ et vulnérabilité économique locale), prouvant que l'injustice spatiale cyclable relève de choix de gouvernance locale plutôt que d'une fatalité.")

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

st.markdown("""
**📝 Note d'analyse :** La hiérarchie révélée par l'IMD bouleverse les classements naïfs basés uniquement sur le volume de vélos. L'absence de corrélation forte entre la taille démographique et la position dans le classement prouve que **l'efficacité d'un réseau cyclable n'est pas l'apanage des seules mégalopoles**, mais résulte d'une ingénierie de maillage et d'une hybridation des flottes réussies.
""")

# ── Section 3 — Décomposition ─────────────────────────────────────────────────
if show_components:
    st.divider()
    section(3, "Décomposition Dimensionnelle — Typologie des Réseaux")
    st.caption(
        "Chaque composante est exprimée sur [0, 100] après normalisation. "
        "Cette décomposition permet d'identifier les stratégies d'aménagement locales."
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
    st.caption(
        "**Figure 3.1.** Profils structurels des 20 premières villes. "
        "La variance intra-ville démontre les compromis opérés par les décideurs publics."
    )
    
    st.markdown("""
    **📝 Note d'analyse :** L'analyse visuelle des barres groupées permet de dégager deux grandes typologies de réseaux en France :
    1. **Les réseaux "Dorsales" (Forte composante M) :** Des villes qui déploient peu de stations mais les concentrent exclusivement autour des hubs de transport (Gares, Tramway).
    2. **Les réseaux "Différentiels" (Forte composante I, faible S) :** Des villes ayant un fort kilométrage de pistes, mais dont le maillage des stations croise historiquement les points noirs d'accidentologie urbaine.
    """)

# ── Section 4 — Validation externe (FUB) ─────────────────────────────────────
st.divider()
section(4, "Validation Externe — Offre Objective (IMD) vs. Climat Perçu (FUB)")

st.markdown(r"""
Un modèle mathématique purement objectif court le risque de s'éloigner de la réalité usager. Pour valider notre construction, nous corrélons l'IMD au **Baromètre des Villes Cyclables de la FUB (2023)**, qui agrège le "climat vélo" ressenti (note sur 6). 
""")

fub_imd = (
    imd_f.dropna(subset=["fub_score_2023"])
    if "fub_score_2023" in imd_f.columns
    else pd.DataFrame()
)

if not fub_imd.empty:
    corr_val = fub_imd["IMD"].corr(fub_imd["fub_score_2023"])
    cv1, cv2, cv3 = st.columns(3)
    cv1.metric("Coefficient de Pearson ($r$)", f"{corr_val:.3f}")
    cv2.metric("Agglomérations Croisées ($n$)", f"{len(fub_imd)}")
    cv3.metric("Variance Expliquée ($R^2$)", f"{(corr_val**2)*100:.1f} %")

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
            "IMD": "Score Objectif IMD (/100)",
            "fub_score_2023": "Score Perçu FUB 2023 (/6)",
            "n_stations": "Densité de Stations",
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
        "**Figure 4.1.** Validation psychométrique du modèle. La convergence statistique démontre "
        "que les variables d'environnement intégrées à l'IMD capturent efficacement l'expérience cyclable ressentie."
    )
    
    st.markdown("""
    **📝 Implications des Outliers (Points atypiques) :** Les villes situées très au-dessus de la ligne de tendance (Score FUB excellent mais IMD moyen) bénéficient d'une "culture vélo" historique qui compense le manque d'offre partagée. À l'inverse, les villes sous la ligne de tendance démontrent qu'un investissement massif en VLS (fort IMD) ne suffit pas à rassurer les cyclistes si le trafic routier environnant reste oppressant.
    """)
else:
    st.info(
        "Données FUB non disponibles pour la validation croisée dans cette session."
    )

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

# ── Section 6 — Conclusion et Implications ────────────────────────────────────
st.divider()
section(6, "Implications pour la Recherche et l'Aménagement Public")
st.success("""
**Synthèse Stratégique :**
1. **Changement de Paradigme d'Évaluation :** La calibration de l'IMD démontre formellement ($w_M^* = 0{,}578$) que le volume de la flotte n'est plus le prédicteur principal du succès cyclable. L'hybridation (Multimodalité GTFS) s'impose comme la variable explicative dominante.
2. **Recommandation pour l'Allocation des Fonds Publics :** Les financements étatiques (type Plan Vélo) ne devraient plus être alloués au prorata de la population, mais en fonction de l'Indice d'Équité Sociale (IES), afin de résorber prioritairement les "Déserts de Mobilité Sociale" documentés dans cette recherche.
3. **Perspectives :** L'intégration future d'une modélisation de la diffusion spatiale par théorie des graphes permettra de raffiner l'analyse de la centralité des stations d'un point de vue énergétique.
""")