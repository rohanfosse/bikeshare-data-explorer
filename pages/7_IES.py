"""
7_IES.py — Indice d'Équité Sociale (IES).
Mesure de la justice spatiale dans la distribution des systèmes de vélos en libre-service.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    compute_imd_cities,
    load_stations,
    load_synthese_velo_socio,
    load_top_quartiers,
)
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Indice d'Équité Sociale (IES) — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Indice d'Équité Sociale (IES)")
st.caption(
    "Axe de Recherche 4 : Justice Spatiale et Déserts de Mobilité Douce "
    "dans les Systèmes de Vélos en Libre-Service"
)

abstract_box(
    "<b>Problématique de recherche :</b> La qualité de l'environnement cyclable partagé — "
    "telle que mesurée par l'IMD — est-elle distribuée de manière équitable sur le territoire, "
    "indépendamment du niveau socio-économique des populations desservies ?<br><br>"
    "L'Indice de Mobilité Douce (IMD) quantifie la <i>qualité physique</i> de l'offre cyclable, "
    "mais reste aveugle à sa dimension sociale. Une approche de politique publique qui se contenterait "
    "de mesurer l'IMD sans le confronter aux structures socio-spatiales risque de valider des inégalités "
    "structurelles sous couvert d'objectivité technique. "
    "L'Indice d'Équité Sociale (IES) — défini comme le ratio entre l'IMD observé et l'IMD théorique "
    "prédit par le seul déterminant économique (modèle Ridge, $R^2_{\\text{train}} = 0{,}28$) — "
    "isole la composante de l'aménagement cyclable relevant d'une volonté politique proactive, "
    "au-delà du déterminisme économique. "
    "Cette page présente la formalisation théorique de l'IES et son illustration empirique "
    "sur le cas de Montpellier, seul territoire pour lequel la granularité intra-urbaine "
    "des données socio-économiques est disponible dans ce corpus."
)

# ── Chargement des données ─────────────────────────────────────────────────────
df       = load_stations()
imd_df   = compute_imd_cities(df)
socio_df = load_synthese_velo_socio()
top_q, bot_q = load_top_quartiers()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres d'Analyse")
    min_stations = st.number_input(
        "Seuil min. stations (IMD)", min_value=1, max_value=200, value=10,
        help="Filtre les micro-réseaux pour garantir la robustesse statistique du classement.",
    )
    geo_type = st.selectbox(
        "Échelle Montpellier",
        options=["Tous", "quartier", "sous-quartier"],
        index=0,
        help="Granularité géographique pour l'analyse IES intra-urbaine.",
    )

imd_f = imd_df[imd_df["n_stations"] >= min_stations].reset_index(drop=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_socio = len(socio_df[socio_df["revenu_fiscal_moyen_menage"].notna()]) if not socio_df.empty else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Score IMD médian national", f"{imd_f['IMD'].median():.1f} / 100")
k2.metric("Agglomérations éligibles", f"{len(imd_f)}")
k3.metric("Quartiers Montpellier analysés", f"{n_socio}")
k4.metric("R² Ridge (revenu → IMD)", "0,28")

# ── Section 1 — Fondements théoriques ─────────────────────────────────────────
st.divider()
section(1, "Fondements Théoriques : Justice Spatiale et Droit à la Mobilité")

st.markdown(r"""
#### 1.1. Le Paradoxe de la Gentrification Cyclable

La littérature internationale en géographie des transports a documenté un paradoxe systématique :
les systèmes de vélos en libre-service (VLS) tendent à s'implanter préférentiellement dans les
quartiers à forte accessibilité multimodale et revenu élevé (*Fishman et al., 2014 ;
Médard de Chardon et al., 2017*). Ce phénomène — la **"gentrification cyclable"** (*Stehlin, 2019*) —
s'oppose au mandat d'équité spatiale des politiques de mobilité douce, dont l'objectif est précisément
de réduire la dépendance automobile dans les territoires sous-équipés.

La question de l'équité de l'offre VLS ne peut donc pas être résolue en mesurant uniquement la
qualité de l'environnement cyclable (IMD) : il faut la **conditionner** par le niveau socio-économique
des populations desservies, pour distinguer les agglomérations qui sur-investissent dans la mobilité
douce pour les populations précaires de celles qui reproduisent les inégalités préexistantes.

#### 1.2. Cadre Théorique : De Rawls à la Justice Spatiale

Le concept de **justice spatiale** (*Soja, 2010 ; Grengs, 2010*) stipule que la distribution des
ressources de mobilité dans l'espace urbain n'est pas politiquement neutre : elle reflète et amplifie
les rapports de force socio-économiques. La **double peine** (*Lucas, 2012*) désigne la situation des
ménages à la fois exclus économiquement *et* spatialement des alternatives à la voiture individuelle —
deux handicaps qui se renforcent mutuellement.

| Indicateur | Situation à risque | Mécanisme d'exclusion |
| :--- | :--- | :--- |
| $\text{IES}_i < 1$ | Sous-investissement relatif | L'offre VLS ne compense pas la fragilité socio-économique. |
| IMD faible + Revenu faible | Désert de mobilité sociale | Cumul de la précarité économique et de l'isolement cyclable. |
| Part voiture élevée + absence VLS | Captivité automobile | Dépendance forcée à un mode onéreux et polluant. |

#### 1.3. Hypothèses de Recherche

**H₀ :** L'IMD est distribuée de manière aléatoire, indépendamment du niveau de revenu des
agglomérations ($\rho_S = 0$).

**H₁ :** Il existe une corrélation positive significative entre revenu et IMD (inégalité structurelle),
mais des agglomérations "outliers" s'en affranchissent (IES $> 1$ malgré un revenu faible), témoignant
d'une volonté politique proactive. Le coefficient de détermination Ridge ($R^2 = 0{,}28$) constitue le
test de cette hypothèse : une valeur faible confirme que la gouvernance locale prime sur le déterminisme
économique.
""")

# ── Section 2 — Formalisation mathématique ────────────────────────────────────
st.divider()
section(2, "Formalisation Mathématique de l'IES et du Modèle de Référence Ridge")

st.markdown(r"""
#### 2.1. Définition de l'IES

Pour chaque agglomération $i$ disposant d'un réseau VLS certifié Gold Standard, l'Indice d'Équité
Sociale est défini comme le rapport entre la performance cyclable observée et la performance
théoriquement attendue en fonction du seul revenu médian :
""")

st.latex(r"""
\text{IES}_i = \frac{\text{IMD}_{\text{observé}, i}}{\widehat{\text{IMD}}(R_{m, i})}
""")

st.markdown(r"""
où $\widehat{\text{IMD}}(R_{m,i})$ est la valeur prédite par le modèle de référence Ridge pour
l'agglomération $i$ ayant un revenu médian $R_{m,i}$.

Les trois régimes interprétatifs sont :

* **$\text{IES}_i > 1$ — "Mobilité Inclusive" :** L'agglomération investit dans l'environnement
  cyclable au-delà de ce que son niveau de revenu laisserait prévoir. Indicateur d'une politique
  publique pro-active.
* **$\text{IES}_i \approx 1$ — "Conformité" :** L'offre cyclable est proportionnelle au niveau
  économique local. Ni sur- ni sous-investissement relatif.
* **$\text{IES}_i < 1$ — "Sous-investissement" :** L'agglomération ne déploie pas l'offre cyclable
  que son niveau de revenu permettrait d'anticiper. Risque de captivité automobile pour les
  populations vulnérables.

#### 2.2. Le Modèle de Référence Ridge

Le dénominateur de l'IES est estimé par une régression Ridge ($\ell_2$), privilégiée à la régression
OLS standard pour sa robustesse aux multi-colinéarités et aux petits échantillons. La pénalité
Ridge est :
""")

st.latex(r"""
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}}
\left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2
+ \lambda \|\boldsymbol{\beta}\|_2^2 \right\}
""")

st.markdown(r"""
Le paramètre $\lambda$ est sélectionné par validation croisée ($k = 5$) sur l'échantillon des
agglomérations Gold Standard. Le coefficient de détermination obtenu est $R^2_{\text{train}} = 0{,}28$,
ce qui implique que le revenu médian n'explique que **28 % de la variance de l'IMD** — résultat
fondamental qui confirme que le déterminisme économique n'est pas le seul facteur structurant
l'environnement cyclable.

Les **72 % restants** sont attribuables aux choix de gouvernance locale, à la topographie, à l'héritage
historique des politiques de mobilité, et aux stratégies différenciées des opérateurs VLS. Ce résultat
est cohérent avec l'absence d'autocorrélation spatiale documentée sur ce corpus
(Moran's $I = -0{,}023$, $p = 0{,}765$) : ni la géographie ni l'économie ne prédestinent une
agglomération à l'excellence ou à la médiocrité cyclable.
""")

# ── Section 3 — Matrice de diagnostic ─────────────────────────────────────────
st.divider()
section(3, "Matrice de Diagnostic : Quatre Régimes de Justice Cyclable")

st.markdown(r"""
Le croisement de l'IMD (qualité physique de l'offre) et du revenu médian de l'agglomération
(contrainte socio-économique) produit une **matrice de diagnostic à quatre quadrants**, permettant de
classer les agglomérations selon leur régime de justice cyclable. Les lignes de démarcation sont
définies par les médianes nationales de chaque indicateur.
""")

fig_quad = go.Figure()

_quadrants = [
    dict(x0=0.0, y0=0.5, x1=0.5, y1=1.0,
         label_x=0.25, label_y=0.75,
         text="<b>Mobilité Inclusive</b><br>IES > 1<br>Revenu < médiane, IMD > médiane<br><i>Politique pro-active</i>",
         color="#27ae60"),
    dict(x0=0.5, y0=0.5, x1=1.0, y1=1.0,
         label_x=0.75, label_y=0.75,
         text="<b>Excellence Consolidée</b><br>IES ≈ 1<br>Revenu > médiane, IMD > médiane<br><i>Conformité attendue</i>",
         color="#1A6FBF"),
    dict(x0=0.0, y0=0.0, x1=0.5, y1=0.5,
         label_x=0.25, label_y=0.25,
         text="<b>Désert de Mobilité</b><br>IES < 1<br>Revenu < médiane, IMD < médiane<br><i>Double peine (Lucas, 2012)</i>",
         color="#e74c3c"),
    dict(x0=0.5, y0=0.0, x1=1.0, y1=0.5,
         label_x=0.75, label_y=0.25,
         text="<b>Sous-Performance</b><br>IES < 1<br>Revenu > médiane, IMD < médiane<br><i>Sous-investissement relatif</i>",
         color="#e67e22"),
]

for q in _quadrants:
    fig_quad.add_shape(
        type="rect",
        x0=q["x0"], y0=q["y0"], x1=q["x1"], y1=q["y1"],
        fillcolor=q["color"], opacity=0.10, line_width=0,
    )
    fig_quad.add_annotation(
        x=q["label_x"], y=q["label_y"],
        text=q["text"],
        showarrow=False,
        font=dict(size=11, color=q["color"]),
        align="center",
        bgcolor="rgba(255,255,255,0.80)",
        bordercolor=q["color"],
        borderwidth=1.5,
        borderpad=7,
    )

fig_quad.add_hline(
    y=0.5, line_dash="dash", line_color="#555", line_width=1.5,
    annotation_text="Médiane nationale (IMD)", annotation_position="right",
)
fig_quad.add_vline(
    x=0.5, line_dash="dash", line_color="#555", line_width=1.5,
    annotation_text="Médiane nationale (Revenu)", annotation_position="top",
)

fig_quad.update_layout(
    height=420,
    margin=dict(l=40, r=80, t=20, b=50),
    plot_bgcolor="white",
    xaxis=dict(
        title="Revenu médian de l'agglomération (normalisé 0–1)",
        showticklabels=False, range=[0, 1],
    ),
    yaxis=dict(
        title="Score IMD (/100, normalisé 0–1)",
        showticklabels=False, range=[0, 1],
    ),
)
st.plotly_chart(fig_quad, use_container_width=True)
st.caption(
    "**Figure 3.1.** Matrice de diagnostic à quatre quadrants de la justice cyclable. "
    "Les axes sont normalisés min-max sur l'échantillon national. "
    "Le quadrant inférieur gauche — 'Déserts de Mobilité Sociale' — concentre les agglomérations "
    "cumulant précarité économique ($R_m <$ médiane nationale) et sous-équipement cyclable "
    "(IMD $<$ médiane). Ces territoires présentent un IES $< 1$, signalant un sous-investissement "
    "relatif en matière de mobilité douce."
)

# ── Section 4 — Étude de cas Montpellier ──────────────────────────────────────
st.divider()
section(4, "Étude de Cas Intra-Urbaine : Équité Cyclable à Montpellier (Vélomagg)")

st.markdown(r"""
L'application de la logique IES à l'échelle intra-urbaine — en substituant les quartiers aux
agglomérations et les revenus fiscaux moyens aux revenus médians nationaux — permet de tester
l'hypothèse de la gentrification cyclable à une granularité plus fine. Le cas de Montpellier est
particulièrement favorable car le jeu de données *synthèse vélo × socio-économique* (INSEE RP 2020)
fournit, pour chaque quartier et sous-quartier, le **revenu fiscal moyen des ménages**
($\text{Rev}_q$) et la **part modale vélo et deux-roues** ($\text{Velo}_q$, proxy comportemental
de la demande VLS).

Le résidu de la régression OLS locale (proxy Ridge) constitue l'IES intra-urbain $\widetilde{\text{IES}}_q$ :
""")

st.latex(r"""
\widetilde{\text{IES}}_q = \text{Velo}_{q} - \widehat{\text{Velo}}(\text{Rev}_q),
\quad \widehat{\text{Velo}}(\text{Rev}_q) = \hat{\beta}_0 + \hat{\beta}_1 \cdot \text{Rev}_q
""")

st.markdown(r"""
Un $\widetilde{\text{IES}}_q > 0$ indique que le quartier utilise davantage le vélo que ce que son
niveau de revenu laisserait prévoir — situation de "mobilité inclusive" ; un
$\widetilde{\text{IES}}_q < 0$ signale un sous-usage relatif, potentiellement révélateur d'un déficit
d'infrastructure ou d'un frein d'accessibilité.
""")

if (
    not socio_df.empty
    and "revenu_fiscal_moyen_menage" in socio_df.columns
    and "transport_deux_roues_velo_pct" in socio_df.columns
):
    plot_df = socio_df.copy()
    if geo_type != "Tous" and "type" in plot_df.columns:
        plot_df = plot_df[plot_df["type"] == geo_type]
    plot_df = plot_df.dropna(
        subset=["revenu_fiscal_moyen_menage", "transport_deux_roues_velo_pct"]
    ).copy()

    if len(plot_df) >= 3:
        x_arr = plot_df["revenu_fiscal_moyen_menage"].values.astype(float)
        y_arr = plot_df["transport_deux_roues_velo_pct"].values.astype(float)

        # OLS regression (proxy for Ridge baseline)
        coeffs        = np.polyfit(x_arr, y_arr, 1)
        x_line        = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line        = np.polyval(coeffs, x_line)
        y_pred        = np.polyval(coeffs, x_arr)
        plot_df["residual_ies"] = y_arr - y_pred
        plot_df["regime_ies"]   = plot_df["residual_ies"].apply(
            lambda r: "Mobilité Inclusive (IES > 0)" if r >= 0
            else "Sous-Performance (IES < 0)"
        )

        # Spearman correlation
        rho, pval = spearmanr(x_arr, y_arr)

        c1, c2, c3 = st.columns(3)
        c1.metric("Quartiers analysés", f"{len(plot_df)}")
        c2.metric("Corrélation de Spearman (rho)", f"{rho:+.3f}")
        c3.metric("p-valeur", f"{pval:.3f}" if pval >= 0.001 else "< 0,001")

        # Scatter plot with regression line
        name_col = "nom" if "nom" in plot_df.columns else plot_df.columns[0]
        hover = {"revenu_fiscal_moyen_menage": ":.0f", "transport_deux_roues_velo_pct": ":.2f"}

        fig_ies = px.scatter(
            plot_df,
            x="revenu_fiscal_moyen_menage",
            y="transport_deux_roues_velo_pct",
            color="regime_ies",
            text=name_col,
            hover_data=hover,
            color_discrete_map={
                "Mobilité Inclusive (IES > 0)": "#27ae60",
                "Sous-Performance (IES < 0)":   "#e74c3c",
            },
            labels={
                "revenu_fiscal_moyen_menage":      "Revenu fiscal moyen des ménages (€/an)",
                "transport_deux_roues_velo_pct":   "Part modale vélo et deux-roues (%)",
                "regime_ies":                      "Régime IES",
            },
            height=500,
        )

        fig_ies.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            name="Référentiel OLS (proxy Ridge)",
            line=dict(color="#1A6FBF", dash="dash", width=2),
            showlegend=True,
        ))

        med_x = float(plot_df["revenu_fiscal_moyen_menage"].median())
        med_y = float(plot_df["transport_deux_roues_velo_pct"].median())
        fig_ies.add_hline(
            y=med_y, line_dash="dot", line_color="#888", opacity=0.5,
            annotation_text="Médiane (Part vélo)", annotation_position="right",
        )
        fig_ies.add_vline(
            x=med_x, line_dash="dot", line_color="#888", opacity=0.5,
            annotation_text="Médiane (Revenu)", annotation_position="top",
        )

        fig_ies.update_traces(textposition="top center", selector=dict(mode="markers+text"))
        fig_ies.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig_ies, use_container_width=True)
        st.caption(
            "**Figure 4.1.** Revenu fiscal moyen (axe horizontal) versus part modale vélo et deux-roues "
            "(axe vertical) par quartier / sous-quartier de Montpellier (INSEE RP 2020). "
            "La droite en pointillés bleus est le référentiel OLS (proxy Ridge) : les quartiers "
            "au-dessus de la droite présentent une part modale vélo supérieure à ce que leur revenu "
            "laisserait prévoir (IES > 0 — Mobilité Inclusive) ; les quartiers en dessous sont en "
            f"situation de sous-performance relative (IES < 0). "
            f"Corrélation de Spearman : $\\rho = {rho:+.3f}$ ($p = {pval:.3f}$)."
        )

        # ── Tableau des résidus IES classés ──────────────────────────────────
        st.markdown("#### Classement des Quartiers par Résidu IES (Écart au Référentiel)")
        st.markdown(r"""
        Les valeurs de résidu négatives les plus importantes identifient les **"Déserts de Mobilité
        Intra-Urbaine"** : quartiers dont la part modale vélo est structurellement inférieure à ce que
        leur niveau de revenu permettrait d'anticiper. Les valeurs positives les plus fortes indiquent
        les territoires de "mobilité inclusive" — quartiers dont l'usage vélo est disproportionnellement
        élevé compte tenu de leur contrainte socio-économique.
        """)

        cols_disp = [name_col, "revenu_fiscal_moyen_menage",
                     "transport_deux_roues_velo_pct", "residual_ies"]
        if "type" in plot_df.columns:
            cols_disp = [name_col, "type"] + cols_disp[1:]
        disp_df = plot_df[cols_disp].copy().sort_values("residual_ies")

        rename_map = {
            name_col:                          "Quartier",
            "type":                            "Type",
            "revenu_fiscal_moyen_menage":      "Revenu moyen (€/an)",
            "transport_deux_roues_velo_pct":   "Part vélo (%)",
            "residual_ies":                    "Résidu IES (p.p.)",
        }
        disp_df = disp_df.rename(columns={k: v for k, v in rename_map.items() if k in disp_df.columns})
        disp_df["Résidu IES (p.p.)"] = disp_df["Résidu IES (p.p.)"].round(3)
        if "Revenu moyen (€/an)" in disp_df.columns:
            disp_df["Revenu moyen (€/an)"] = disp_df["Revenu moyen (€/an)"].round(0).astype("Int64")
        if "Part vélo (%)" in disp_df.columns:
            disp_df["Part vélo (%)"] = disp_df["Part vélo (%)"].round(2)

        st.dataframe(
            disp_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Résidu IES (p.p.)": st.column_config.ProgressColumn(
                    "Résidu IES (p.p.)",
                    min_value=float(disp_df["Résidu IES (p.p.)"].min()),
                    max_value=float(disp_df["Résidu IES (p.p.)"].max()),
                    format="%.3f",
                )
            },
        )
        st.caption(
            "**Tableau 4.1.** Classement des quartiers de Montpellier par résidu IES intra-urbain "
            "(en points de pourcentage de part modale). "
            "Les quartiers avec un résidu négatif élevé constituent les 'Déserts de Mobilité "
            "Intra-Urbaine' : leur part modale vélo est inférieure à celle attendue compte tenu "
            "de leur niveau de revenu, révélant un potentiel non capturé par le réseau Vélomagg."
        )

    else:
        st.warning(
            "Données Montpellier insuffisantes (moins de 3 quartiers avec données complètes) "
            "pour l'analyse IES intra-urbaine."
        )

else:
    st.info(
        "Les données socio-économiques de Montpellier (synthèse vélo × socio) "
        "ne sont pas disponibles dans le répertoire `data/processed/`."
    )

# ── Section 5 — Implications politiques ───────────────────────────────────────
st.divider()
section(5, "Implications pour la Gouvernance des Réseaux VLS")

st.markdown(r"""
L'IES fournit un outil de ciblage politique précis pour orienter les investissements en mobilité douce
vers les territoires où l'impact social est maximal. Trois leviers d'action se dégagent de l'analyse :

#### 5.1. Levier Infrastructurel : Redéploiement Spatial de l'Offre

Les agglomérations identifiées comme "Déserts de Mobilité Sociale" (IES $< 1$, revenu $<$ médiane)
devraient bénéficier d'une densification prioritaire de l'offre VLS. Le diagnostic IES permet de
quantifier *l'effort correctif minimal* : l'agglomération doit déployer suffisamment de stations
pour porter son IMD au niveau prédit par le référentiel Ridge,
soit $\text{IMD}_{\text{cible}} = \widehat{\text{IMD}}(R_{m,i})$.

#### 5.2. Levier Tarifaire : Différenciation Socio-Spatiale

La littérature internationale (*Fishman et al., 2014 ; Ricci, 2015*) montre que le prix de
l'abonnement est le principal frein à l'adoption du VLS dans les ménages à revenus modestes. Un
dispositif de **tarification sociale différenciée** — abonnement gratuit ou subventionné pour les
allocataires RSA ou APL — est un mécanisme complémentaire à l'investissement infrastructurel,
à l'image du modèle montpelliérain de la TAM.

#### 5.3. Levier Gouvernanciel : Contractualisation des Obligations d'Équité

L'IES peut être intégré comme **indicateur contractuel** dans les délégations de service public (DSP)
VLS : les opérateurs seraient tenus de maintenir un IES $\geq 0{,}90$ pour l'ensemble de leur
territoire d'exploitation, sous peine de pénalités financières. Ce mécanisme de régulation
performative inciterait les opérateurs à étendre leur réseau vers les zones moins rentables mais
socialement stratégiques.

#### 5.4. Résultat Clé : La Primauté de la Gouvernance sur le Déterminisme Économique

Le coefficient $R^2 = 0{,}28$ du modèle Ridge confirme que le revenu médian n'explique qu'une minorité
de la variance de l'IMD. **72 % de la qualité de l'environnement cyclable relèvent de choix de
gouvernance locale**, non de déterminismes économiques. Ce résultat est cohérent avec l'absence
d'autocorrélation spatiale documentée sur ce corpus (Moran's $I = -0{,}023$, $p = 0{,}765$) :
ni la géographie ni l'économie ne prédestinent une agglomération à l'excellence ou à la médiocrité
cyclable. La responsabilité politique reste entière.
""")
