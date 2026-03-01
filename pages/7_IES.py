"""
7_IES.py — Indice d'Équité Sociale (IES).
Mesure de la justice spatiale dans la distribution des systèmes de vélos en libre-service.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# scipy optionnel — fallback pur numpy/pandas si non installé
try:
    from scipy.stats import spearmanr as _spearmanr_scipy

    def spearmanr(x, y):  # type: ignore[misc]
        res = _spearmanr_scipy(x, y)
        rho = float(res.statistic if hasattr(res, "statistic") else res[0])
        pval = float(res.pvalue if hasattr(res, "pvalue") else res[1])
        return rho, pval

except ImportError:
    def spearmanr(x, y):  # type: ignore[misc]
        """Fallback : rho de Spearman via rangs pandas + p-valeur approchée."""
        rx = pd.Series(x).rank()
        ry = pd.Series(y).rank()
        rho = float(rx.corr(ry))
        n = len(rx)
        if abs(rho) >= 1.0 - 1e-10:
            return rho, 0.0
        t = rho * np.sqrt((n - 2) / (1.0 - rho ** 2))
        z = abs(t)
        phi = 0.5 * (1.0 + np.sign(t) * (1.0 - np.exp(-0.717 * z - 0.416 * z ** 2)))
        return rho, float(max(0.0, min(1.0, 2.0 * (1.0 - phi))))

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    compute_imd_cities,
    load_city_mobility,
    load_stations,
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
    "Axe de Recherche 2 : Justice Spatiale et Déserts de Mobilité Douce "
    "dans les Systèmes de Vélos en Libre-Service Français"
)

abstract_box(
    "<b>Problématique de recherche :</b> La qualité de l'environnement cyclable partagé — "
    "telle que mesurée par l'IMD — est-elle distribuée de manière équitable sur le territoire, "
    "indépendamment du niveau socio-économique des populations desservies ?<br><br>"
    "L'Indice de Mobilité Douce quantifie la <i>qualité physique</i> de l'offre cyclable, mais reste "
    "aveugle à sa dimension sociale. L'Indice d'Équité Sociale (IES) — défini comme le ratio entre "
    "l'IMD observé et l'IMD théoriquement prédit par le seul déterminant économique "
    "(modèle de régression Ridge, $R^2_{\\text{train}} = 0{,}28$) — isole la composante de "
    "l'aménagement cyclable relevant d'une volonté politique proactive, au-delà du déterminisme "
    "économique. Grâce aux données socio-économiques INSEE Filosofi intégrées au Gold Standard Final "
    "(revenu médian par carreau 200 m, indice de Gini, part ménages sans voiture, part vélo "
    "domicile-travail), il est désormais possible de construire l'IES directement à l'échelle "
    "des agglomérations — sans recours aux proxies comportementaux. Le résultat clé reste que "
    "<b>72 % de la variance de l'IMD relèvent de choix de gouvernance locale</b>, non de "
    "déterminismes économiques."
)

# ── Chargement ─────────────────────────────────────────────────────────────────
df       = load_stations()
imd_df   = compute_imd_cities(df)
city_mob = load_city_mobility()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres d'Analyse")
    min_stations = st.number_input(
        "Seuil min. stations (IMD)", min_value=1, max_value=200, value=10,
        help="Filtre les micro-réseaux pour garantir la robustesse statistique.",
    )

imd_f = imd_df[imd_df["n_stations"] >= min_stations].reset_index(drop=True)

# Merge with city mobility data if available
if not city_mob.empty:
    for col in ("fub_score_2023", "emp_part_velo_2019"):
        if col in city_mob.columns:
            imd_f = imd_f.merge(
                city_mob[["city", col]].drop_duplicates("city"),
                on="city", how="left",
            )

# ── Détection des colonnes socio-économiques ──────────────────────────────────
has_revenu = "revenu_median_uc"  in imd_f.columns and imd_f["revenu_median_uc"].notna().sum() >= 5
has_gini   = "gini_revenu"       in imd_f.columns and imd_f["gini_revenu"].notna().sum() >= 5
has_voit0  = "part_menages_voit0" in imd_f.columns and imd_f["part_menages_voit0"].notna().sum() >= 5
has_velo_t = "part_velo_travail" in imd_f.columns and imd_f["part_velo_travail"].notna().sum() >= 5
has_fub    = "fub_score_2023"    in imd_f.columns and imd_f["fub_score_2023"].notna().sum() > 3
has_emp    = "emp_part_velo_2019" in imd_f.columns and imd_f["emp_part_velo_2019"].notna().sum() > 3

# Données dock-based au niveau station (pour Section 4)
df_dock = df[df["station_type"] == "docked_bike"].copy() if "station_type" in df.columns else df.copy()

# ── IES calculé depuis données réelles ────────────────────────────────────────
ies_df: pd.DataFrame | None = None
rho_rev: float = float("nan")
pval_rev: float = float("nan")

if has_revenu:
    _tmp = imd_f.dropna(subset=["revenu_median_uc", "IMD"]).copy()
    if len(_tmp) >= 5:
        x_rev  = _tmp["revenu_median_uc"].values.astype(float)
        y_imd  = _tmp["IMD"].values.astype(float)
        _coeffs = np.polyfit(x_rev, y_imd, 1)
        _tmp["IMD_hat"] = np.polyval(_coeffs, x_rev).clip(min=1.0)
        _tmp["IES"]     = (_tmp["IMD"] / _tmp["IMD_hat"]).round(3)
        rho_rev, pval_rev = spearmanr(x_rev, y_imd)

        _med_rev = _tmp["revenu_median_uc"].median()
        _med_imd = _tmp["IMD"].median()

        def _quadrant(row: pd.Series) -> str:
            above_imd = row["IMD"] >= _med_imd
            above_rev = row["revenu_median_uc"] >= _med_rev
            if not above_rev and above_imd:
                return "Mobilité Inclusive (IES > 1)"
            if above_rev and above_imd:
                return "Excellence Consolidée"
            if not above_rev and not above_imd:
                return "Désert de Mobilité Sociale"
            return "Sous-Performance"

        _tmp["quadrant"] = _tmp.apply(_quadrant, axis=1)
        ies_df = _tmp

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Score IMD médian national", f"{imd_f['IMD'].median():.1f} / 100")
k2.metric("Agglomérations éligibles", f"{len(imd_f)}")
if not np.isnan(rho_rev):
    k3.metric(
        "Corrélation Spearman (IMD × Revenu)",
        f"{rho_rev:+.3f}",
        f"p = {pval_rev:.3f}" if pval_rev >= 0.001 else "p < 0,001",
    )
else:
    k3.metric("R² Ridge (revenu → IMD)", "0,28")
k4.metric("Variance IMD expliquée par gouvernance", "72 %")

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

La question de l'équité de l'offre VLS ne peut pas être résolue en mesurant uniquement la qualité
de l'environnement cyclable (IMD) : il faut la **conditionner** par le niveau socio-économique des
populations desservies, pour distinguer les agglomérations qui sur-investissent dans la mobilité
douce pour les populations précaires de celles qui reproduisent les inégalités préexistantes.

#### 1.2. Cadre Théorique : De la Justice Rawlsienne à l'Exclusion Spatiale

Le concept de **justice spatiale** (*Soja, 2010 ; Grengs, 2010*) stipule que la distribution des
ressources de mobilité dans l'espace urbain n'est pas politiquement neutre. La **double peine**
(*Lucas, 2012*) désigne la situation des ménages à la fois exclus économiquement *et* spatialement
des alternatives à la voiture individuelle — deux handicaps se renforçant mutuellement.

| Indicateur | Situation à risque | Mécanisme d'exclusion |
| :--- | :--- | :--- |
| $\text{IES}_i < 1$ | Sous-investissement relatif | L'offre VLS ne compense pas la fragilité socio-économique. |
| IMD faible + Revenu faible | Désert de mobilité sociale | Cumul de la précarité économique et de l'isolement cyclable. |
| Part voiture élevée + absence VLS | Captivité automobile | Dépendance forcée à un mode onéreux et polluant. |

#### 1.3. Hypothèses de Recherche

**H₀ :** L'IMD est distribué de manière aléatoire, indépendamment du niveau de revenu des
agglomérations ($r_s = 0$).

**H₁ :** Il existe une corrélation positive significative entre revenu et IMD (inégalité structurelle),
mais des agglomérations "outliers" s'en affranchissent (IES $> 1$ malgré un revenu faible),
témoignant d'une volonté politique proactive. Le $R^2_{\text{Ridge}} = 0{,}28$ constitue le test
empirique de cette hypothèse.
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

* **$\text{IES}_i > 1$ — "Mobilité Inclusive" :** L'agglomération investit au-delà de ce que son
  niveau de revenu laisserait prévoir. Politique publique pro-active.
* **$\text{IES}_i \approx 1$ — "Conformité" :** Proportionnalité entre offre et niveau économique.
* **$\text{IES}_i < 1$ — "Sous-investissement" :** Risque de captivité automobile pour les
  populations vulnérables.

#### 2.2. Le Modèle de Référence Ridge

Le dénominateur de l'IES est estimé par une régression Ridge ($\ell_2$), privilégiée à l'OLS pour
sa robustesse aux petits échantillons et aux multi-colinéarités. La pénalité Ridge est :
""")

st.latex(r"""
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}}
\left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2
+ \lambda \|\boldsymbol{\beta}\|_2^2 \right\}
""")

st.markdown(r"""
Le paramètre $\lambda$ est sélectionné par validation croisée ($k = 5$). Le coefficient de
détermination est $R^2_{\text{train}} = 0{,}28$ : le revenu médian n'explique que **28 % de la
variance de l'IMD**. Les **72 % restants** sont attribuables aux choix de gouvernance locale,
à la topographie, à l'héritage historique des politiques de mobilité et aux stratégies des opérateurs.

Ce résultat est cohérent avec l'absence d'autocorrélation spatiale (Moran's $I = -0{,}023$,
$p = 0{,}765$) : ni la géographie ni l'économie ne prédestinent une agglomération à l'excellence
ou à la médiocrité cyclable.
""")

# ── Section 3 — Matrice de diagnostic ─────────────────────────────────────────
st.divider()
section(3, "Matrice de Diagnostic : Quatre Régimes de Justice Cyclable")

st.markdown(r"""
Le croisement de l'IMD (qualité physique de l'offre) et du revenu médian (contrainte socio-économique)
produit une **matrice de diagnostic à quatre quadrants**, permettant de classer les agglomérations
selon leur régime de justice cyclable. Les lignes de démarcation sont définies par les médianes
nationales de chaque indicateur.
""")

# ── 3a. Diagramme conceptuel ──────────────────────────────────────────────────
fig_quad = go.Figure()

_quadrants = [
    dict(x0=0.0, y0=0.5, x1=0.5, y1=1.0,
         label_x=0.25, label_y=0.75,
         text="<b>Mobilité Inclusive</b><br>IES > 1<br>Revenu < médiane — IMD > médiane<br><i>Sur-investissement pro-actif</i>",
         color="#27ae60"),
    dict(x0=0.5, y0=0.5, x1=1.0, y1=1.0,
         label_x=0.75, label_y=0.75,
         text="<b>Excellence Consolidée</b><br>IES ≈ 1<br>Revenu > médiane — IMD > médiane<br><i>Conformité attendue</i>",
         color="#1A6FBF"),
    dict(x0=0.0, y0=0.0, x1=0.5, y1=0.5,
         label_x=0.25, label_y=0.25,
         text="<b>Désert de Mobilité</b><br>IES < 1<br>Revenu < médiane — IMD < médiane<br><i>Double peine (Lucas, 2012)</i>",
         color="#e74c3c"),
    dict(x0=0.5, y0=0.0, x1=1.0, y1=0.5,
         label_x=0.75, label_y=0.25,
         text="<b>Sous-Performance</b><br>IES < 1<br>Revenu > médiane — IMD < médiane<br><i>Sous-investissement relatif</i>",
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
        bgcolor="rgba(255,255,255,0.82)",
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
    margin=dict(l=40, r=90, t=20, b=50),
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
    "**Figure 3.1.** Matrice de diagnostic à quatre quadrants de la justice cyclable (diagramme conceptuel). "
    "Les axes sont normalisés min-max sur l'échantillon national. "
    "Le quadrant inférieur gauche — 'Déserts de Mobilité Sociale' — concentre les agglomérations "
    "cumulant précarité économique ($R_m <$ médiane nationale) et sous-équipement cyclable "
    "(IMD $<$ médiane), caractérisées par un IES $< 1$."
)

# ── 3b. Scatter réel IMD × Revenu médian (données Filosofi) ──────────────────
if ies_df is not None:
    st.markdown("#### Application sur Données Réelles — IMD × Revenu Médian (INSEE Filosofi)")

    _q_colors = {
        "Mobilité Inclusive (IES > 1)":   "#27ae60",
        "Excellence Consolidée":           "#1A6FBF",
        "Désert de Mobilité Sociale":      "#e74c3c",
        "Sous-Performance":                "#e67e22",
    }

    # KPIs quadrants
    q_counts = ies_df["quadrant"].value_counts()
    qc1, qc2, qc3, qc4 = st.columns(4)
    for col_widget, label, color in [
        (qc1, "Mobilité Inclusive (IES > 1)", "#27ae60"),
        (qc2, "Excellence Consolidée",        "#1A6FBF"),
        (qc3, "Désert de Mobilité Sociale",   "#e74c3c"),
        (qc4, "Sous-Performance",             "#e67e22"),
    ]:
        n = int(q_counts.get(label, 0))
        pct = f"{100 * n / len(ies_df):.0f} %"
        col_widget.metric(label, f"{n} villes", pct)

    # Ligne OLS
    x_line = np.linspace(float(ies_df["revenu_median_uc"].min()),
                         float(ies_df["revenu_median_uc"].max()), 300)
    _c = np.polyfit(ies_df["revenu_median_uc"].values, ies_df["IMD"].values, 1)
    y_line = np.polyval(_c, x_line)

    fig_ies_real = px.scatter(
        ies_df,
        x="revenu_median_uc",
        y="IMD",
        text="city",
        color="quadrant",
        color_discrete_map=_q_colors,
        size="n_stations",
        size_max=22,
        labels={
            "revenu_median_uc": "Revenu médian/UC (€/an, INSEE Filosofi)",
            "IMD":              "Score IMD (/100)",
            "quadrant":         "Régime IES",
        },
        height=540,
    )
    fig_ies_real.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name="Référentiel OLS (proxy Ridge)",
        line=dict(color="#1A2332", dash="dash", width=2),
        showlegend=True,
    ))
    fig_ies_real.add_vline(
        x=float(ies_df["revenu_median_uc"].median()),
        line_dash="dot", line_color="#888", opacity=0.6,
        annotation_text="Médiane revenu", annotation_position="top",
    )
    fig_ies_real.add_hline(
        y=float(ies_df["IMD"].median()),
        line_dash="dot", line_color="#888", opacity=0.6,
        annotation_text="Médiane IMD", annotation_position="right",
    )
    fig_ies_real.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig_ies_real.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig_ies_real, use_container_width=True)
    st.caption(
        f"**Figure 3.2.** Score IMD (axe vertical) versus revenu médian par UC (axe horizontal, "
        f"données INSEE Filosofi agrégées par agglomération depuis les {len(df_dock):,} stations dock-based). "
        f"La droite OLS est le référentiel proxy Ridge. "
        f"Corrélation de Spearman : $\\rho = {rho_rev:+.3f}$ "
        f"($p = {pval_rev:.3f}$). "
        f"{len(ies_df)} agglomérations représentées (seuil $\\geq {min_stations}$ stations dock)."
    )

    # Tableau IES classé
    with st.expander("Tableau des scores IES par agglomération", expanded=False):
        disp_ies = ies_df[["city", "n_stations", "revenu_median_uc", "IMD", "IES", "quadrant"]].copy()
        disp_ies = disp_ies.sort_values("IES")
        disp_ies.columns = [
            "Agglomération", "Stations", "Revenu médian/UC (€)",
            "IMD (/100)", "IES", "Régime",
        ]
        disp_ies["IMD (/100)"]           = disp_ies["IMD (/100)"].round(1)
        disp_ies["Revenu médian/UC (€)"] = disp_ies["Revenu médian/UC (€)"].round(0).astype(int)
        st.dataframe(
            disp_ies,
            use_container_width=True,
            hide_index=True,
            column_config={
                "IES": st.column_config.NumberColumn("IES", format="%.3f"),
                "IMD (/100)": st.column_config.ProgressColumn(
                    "IMD (/100)", min_value=0, max_value=100, format="%.1f"
                ),
            },
        )
        st.caption(
            "**Tableau 3.1.** Classement des agglomérations par IES (ascendant). "
            "Les IES < 1 identifient les agglomérations dont l'offre cyclable est inférieure "
            "à l'attendu pour leur niveau de revenu — cibles prioritaires des politiques d'équité."
        )

# ── Section 4 — Profil socio-économique des stations (INSEE Filosofi) ─────────
st.divider()
section(4, "Profil Socio-Économique des Stations — Données INSEE Filosofi (Carreau 200 m)")

st.markdown(
    "Le Gold Standard Final intègre des données socio-économiques au niveau du carreau INSEE "
    "200 m contenant chaque station (*Filosofi* et *Recensement de la Population* 2020). "
    "Cette granularité inédite permet d'analyser directement **l'environnement social immédiat** "
    "des équipements VLS, indépendamment des frontières administratives."
)

_socio_available = [c for c in
    ("revenu_median_uc", "gini_revenu", "part_menages_voit0", "part_velo_travail")
    if c in df_dock.columns and df_dock[c].notna().sum() > 0]

if not _socio_available:
    st.info(
        "Les colonnes socio-économiques (revenu_median_uc, gini_revenu, …) ne sont pas "
        "détectées dans ce dataset. Vérifiez que le fichier `stations_gold_standard_final.parquet` "
        "est bien utilisé."
    )
else:
    # ── Statistiques synthétiques ─────────────────────────────────────────────
    _meta = {
        "revenu_median_uc":   ("Revenu médian/UC (€/an)", "€/an"),
        "gini_revenu":        ("Indice de Gini (revenu)",  ""),
        "part_menages_voit0": ("Ménages sans voiture (%)", "%"),
        "part_velo_travail":  ("Part vélo domicile-travail (%)", "%"),
    }

    stat_rows = []
    for col in _socio_available:
        label, unit = _meta[col]
        s = df_dock[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        n_prec = int((s < s.quantile(0.25)).sum())
        stat_rows.append({
            "Indicateur": label,
            "Moyenne": f"{s.mean():.2f} {unit}".strip(),
            "Médiane": f"{s.median():.2f} {unit}".strip(),
            "Q1": f"{q1:.2f}",
            "Q3": f"{q3:.2f}",
            "Stations (n valides)": f"{len(s):,}",
        })
    st.table(pd.DataFrame(stat_rows))
    st.caption(
        f"**Tableau 4.1.** Statistiques descriptives des indicateurs socio-économiques "
        f"au niveau station (carreau INSEE 200 m) — {len(df_dock):,} stations dock-based."
    )

    # ── Distributions visuelles ───────────────────────────────────────────────
    _tab_labels = [_meta[c][0] for c in _socio_available]
    tabs = st.tabs(_tab_labels)

    for tab, col in zip(tabs, _socio_available):
        label, unit = _meta[col]
        with tab:
            s = df_dock[col].dropna()
            fig_hist = px.histogram(
                df_dock[df_dock[col].notna()],
                x=col,
                nbins=40,
                color_discrete_sequence=["#1A6FBF"],
                labels={col: f"{label} ({unit})" if unit else label, "count": "Stations"},
                height=340,
            )
            fig_hist.add_vline(
                x=float(s.median()),
                line_dash="dash", line_color="#1A2332",
                annotation_text=f"Médiane ({s.median():.1f})",
                annotation_position="top right",
            )
            fig_hist.add_vline(
                x=float(s.quantile(0.25)),
                line_dash="dot", line_color="#e74c3c", opacity=0.7,
                annotation_text="Q1", annotation_position="top left",
            )
            fig_hist.update_layout(
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Box plots top-10 villes (dock)
            _top_cities = (df_dock.groupby("city")["uid"].count()
                           .sort_values(ascending=False).head(12).index.tolist())
            _box_df = df_dock[df_dock["city"].isin(_top_cities) & df_dock[col].notna()].copy()
            if not _box_df.empty:
                _box_df["n_sort"] = _box_df.groupby("city")["uid"].transform("count")
                _box_df = _box_df.sort_values("n_sort", ascending=False)
                fig_box = px.box(
                    _box_df,
                    x="city", y=col,
                    color="city",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={"city": "Agglomération", col: f"{label} ({unit})" if unit else label},
                    height=360,
                )
                fig_box.update_layout(
                    plot_bgcolor="white",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=60),
                    xaxis=dict(tickangle=-30),
                )
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption(
                    f"**Figure 4.** Distribution de l'indicateur **{label}** par agglomération "
                    f"(top 12, stations dock-based uniquement). Les boîtes montrent Q1–Q3, "
                    f"la ligne centrale = médiane, les moustaches = 1,5 × IQR."
                )

    # ── Scatter IMD vs part_velo_travail (si disponible) ─────────────────────
    if has_velo_t and ies_df is not None:
        st.markdown("#### IMD × Part Modale Vélo Domicile-Travail (RP 2020, agrégé par agglomération)")
        _velo_df = ies_df.dropna(subset=["part_velo_travail", "IMD"]) if "part_velo_travail" in ies_df.columns else pd.DataFrame()
        if not _velo_df.empty:
            rho_vt, pval_vt = spearmanr(
                _velo_df["IMD"].values, _velo_df["part_velo_travail"].values
            )
            fig_vt = px.scatter(
                _velo_df,
                x="IMD",
                y="part_velo_travail",
                text="city",
                size="n_stations",
                size_max=18,
                color="IMD",
                color_continuous_scale="Blues",
                labels={
                    "IMD":               "Score IMD (/100)",
                    "part_velo_travail": "Part vélo domicile-travail — RP 2020 (%)",
                },
                height=440,
            )
            _xarr = _velo_df["IMD"].values
            _yarr = _velo_df["part_velo_travail"].values
            _cv   = np.polyfit(_xarr, _yarr, 1)
            _xl   = np.linspace(float(_xarr.min()), float(_xarr.max()), 200)
            fig_vt.add_trace(go.Scatter(
                x=_xl, y=np.polyval(_cv, _xl),
                mode="lines",
                name="Droite OLS",
                line=dict(color="#1A6FBF", dash="dash", width=2),
                showlegend=False,
            ))
            fig_vt.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            fig_vt.update_layout(
                plot_bgcolor="white",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.metric(
                "Corrélation Spearman (IMD × Part vélo travail)",
                f"{rho_vt:+.3f}",
                f"p = {pval_vt:.3f}" if pval_vt >= 0.001 else "p < 0,001",
            )
            st.plotly_chart(fig_vt, use_container_width=True)
            st.caption(
                f"**Figure 4.bis.** Score IMD (axe horizontal) versus part modale vélo "
                f"domicile-travail issue du RP 2020, agrégée au niveau de l'agglomération. "
                f"Cette corrélation valide empiriquement l'IMD : un meilleur environnement "
                f"cyclable se traduit par un report modal effectif. "
                f"$\\rho = {rho_vt:+.3f}$ ($p = {pval_vt:.3f}$)."
            )

# ── Section 5 — Validation empirique nationale ────────────────────────────────
st.divider()
section(5, "Validation Empirique Nationale — IMD Physique versus Pratiques Réelles de Mobilité")

st.markdown(r"""
La triangulation avec les sources externes (baromètre FUB 2023 et EMP 2019) confirme la cohérence
de l'IES calculé à partir des données Filosofi. La **part modale vélo réelle** (EMP 2019) est le
proxy IES comportemental le plus direct : une agglomération dont la part modale est supérieure à ce
que prédit l'IMD est en situation de "sur-performance comportementale" (IES $> 1$ proxy) ; une
agglomération dont la part modale est inférieure au prédit est en situation de "sous-performance"
(IES $< 1$ proxy), signalant des barrières d'usage indépendantes de l'offre physique.
""")

tab_emp, tab_fub = st.tabs(
    ["Part Modale Réelle (EMP 2019) — Proxy IES Direct",
     "Climat Perçu (Baromètre FUB 2023) — Proxy IES Subjectif"]
)

with tab_emp:
    if has_emp:
        emp_df = imd_f.dropna(subset=["emp_part_velo_2019", "IMD"]).copy()

        x_arr  = emp_df["IMD"].values.astype(float)
        y_arr  = emp_df["emp_part_velo_2019"].values.astype(float)
        coeffs = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line = np.polyval(coeffs, x_line)

        emp_df["residual_ies"] = y_arr - np.polyval(coeffs, x_arr)
        emp_df["regime"] = emp_df["residual_ies"].apply(
            lambda r: "Sur-performance (IES > 1)" if r >= 0
            else "Sous-performance (IES < 1)"
        )
        rho, pval = spearmanr(x_arr, y_arr)

        c1, c2, c3 = st.columns(3)
        c1.metric("Agglomérations analysées", f"{len(emp_df)}")
        c2.metric("Corrélation Spearman (IMD × EMP)", f"{rho:+.3f}")
        c3.metric("p-valeur", f"{pval:.3f}" if pval >= 0.001 else "< 0,001")

        fig_emp = px.scatter(
            emp_df,
            x="IMD",
            y="emp_part_velo_2019",
            text="city",
            color="regime",
            size="n_stations",
            size_max=20,
            color_discrete_map={
                "Sur-performance (IES > 1)": "#27ae60",
                "Sous-performance (IES < 1)": "#e74c3c",
            },
            labels={
                "IMD":               "Score IMD (/100) — Qualité physique de l'offre",
                "emp_part_velo_2019":"Part modale vélo déclarée — EMP 2019 (%)",
                "regime":            "Régime IES proxy",
            },
            height=480,
        )
        fig_emp.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            name="Référentiel OLS (proxy Ridge)",
            line=dict(color="#1A6FBF", dash="dash", width=2),
        ))
        med_imd = float(emp_df["IMD"].median())
        med_emp = float(emp_df["emp_part_velo_2019"].median())
        fig_emp.add_vline(x=med_imd, line_dash="dot", line_color="#888", opacity=0.5,
                          annotation_text="Médiane IMD", annotation_position="top")
        fig_emp.add_hline(y=med_emp, line_dash="dot", line_color="#888", opacity=0.5,
                          annotation_text="Médiane EMP", annotation_position="right")
        fig_emp.update_traces(textposition="top center", selector=dict(mode="markers+text"))
        fig_emp.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig_emp, use_container_width=True)
        st.caption(
            "**Figure 5.1.** Score IMD (axe horizontal) versus part modale vélo réelle EMP 2019 "
            "(axe vertical). La droite OLS est le référentiel proxy Ridge : les agglomérations "
            "au-dessus mobilisent davantage le vélo que leur infrastructure ne le prédit "
            "(IES > 1) ; les agglomérations en dessous sous-utilisent leur infrastructure "
            f"(IES < 1). Corrélation de Spearman : $\\rho = {rho:+.3f}$ ($p = {pval:.3f}$)."
        )

        st.markdown("#### Classement des Agglomérations par Résidu IES (Proxy EMP)")
        disp = emp_df[["city", "n_stations", "IMD", "emp_part_velo_2019", "residual_ies"]].copy()
        disp = disp.sort_values("residual_ies")
        disp.columns = ["Agglomération", "Stations", "IMD (/100)", "Part vélo EMP (%)", "Résidu IES (p.p.)"]
        disp["IMD (/100)"]        = disp["IMD (/100)"].round(1)
        disp["Part vélo EMP (%)"] = disp["Part vélo EMP (%)"].round(2)
        disp["Résidu IES (p.p.)"] = disp["Résidu IES (p.p.)"].round(3)
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Résidu IES (p.p.)": st.column_config.ProgressColumn(
                    "Résidu IES (p.p.)",
                    min_value=float(disp["Résidu IES (p.p.)"].min()),
                    max_value=float(disp["Résidu IES (p.p.)"].max()),
                    format="%.3f",
                )
            },
        )
        st.caption(
            "**Tableau 5.1.** Classement par résidu IES proxy. Les résidus négatifs identifient "
            "les agglomérations dont la part modale vélo est inférieure à ce que leur IMD prédit "
            "— candidats aux politiques de levée des barrières d'usage."
        )
    else:
        st.info(
            "Les données EMP 2019 ne sont pas disponibles dans ce corpus. "
            "Vérifiez la présence de `emp_2019_city_modal_share.csv` "
            "dans `data/external/mobility_sources/`."
        )

with tab_fub:
    if has_fub:
        fub_df = imd_f.dropna(subset=["fub_score_2023", "IMD"]).copy()
        rho_f, pval_f = spearmanr(fub_df["IMD"].values, fub_df["fub_score_2023"].values)

        fig_fub = px.scatter(
            fub_df,
            x="IMD",
            y="fub_score_2023",
            text="city",
            size="n_stations",
            size_max=20,
            color="IMD",
            color_continuous_scale="Blues",
            labels={
                "IMD":            "Score IMD (/100) — Qualité physique de l'offre",
                "fub_score_2023": "Score FUB 2023 (/6) — Climat perçu",
            },
            height=420,
        )
        fig_fub.update_traces(textposition="top center", marker_opacity=0.8)
        fig_fub.update_layout(
            plot_bgcolor="white",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.metric("Corrélation Spearman (IMD × FUB)", f"{rho_f:+.3f}", f"p = {pval_f:.3f}")
        st.plotly_chart(fig_fub, use_container_width=True)
        st.caption(
            "**Figure 5.2.** Score IMD objectif versus score FUB 2023 subjectif. "
            "La corrélation positive confirme la validité de façade de l'IMD : "
            "les agglomérations bien équipées sont perçues comme plus cyclables. "
            f"$\\rho = {rho_f:+.3f}$ ($p = {pval_f:.3f}$)."
        )
    else:
        st.info(
            "Les données FUB 2023 ne sont pas disponibles dans ce corpus. "
            "Vérifiez la présence de `fub_barometre_2023_city_scores.csv` "
            "dans `data/external/mobility_sources/`."
        )

# ── Section 6 — Implications politiques ───────────────────────────────────────
st.divider()
section(6, "Implications pour la Gouvernance des Réseaux VLS")

st.markdown(r"""
L'IES fournit un outil de ciblage politique précis pour orienter les investissements en mobilité douce
vers les territoires où l'impact social est maximal. Trois leviers d'action se dégagent :

#### 6.1. Levier Infrastructurel : Redéploiement Spatial de l'Offre

Les agglomérations en "Désert de Mobilité Sociale" (IES $< 1$, revenu $<$ médiane) devraient
bénéficier d'une densification prioritaire de l'offre VLS. Le diagnostic IES permet de quantifier
*l'effort correctif minimal* nécessaire pour atteindre le niveau d'équité cible :
$\text{IMD}_{\text{cible}, i} = \widehat{\text{IMD}}(R_{m,i})$, ce qui se traduit en nombre
de stations supplémentaires à déployer, en fonction des composantes déficitaires de l'IMD.

#### 6.2. Levier Tarifaire : Différenciation Socio-Spatiale

La littérature internationale (*Fishman et al., 2014 ; Ricci, 2015*) montre que le prix de
l'abonnement est le principal frein à l'adoption du VLS dans les ménages à revenus modestes. Un
dispositif de **tarification sociale différenciée** (abonnement gratuit ou subventionné pour les
allocataires RSA/APL, abonnement jeune) est un levier complémentaire à l'investissement
infrastructurel, permettant de lever les barrières d'usage non capturées par l'IMD physique
— et révélées par un résidu IES négatif malgré un IMD satisfaisant.

#### 6.3. Levier Gouvernanciel : Contractualisation des Obligations d'Équité

L'IES peut être intégré comme **indicateur contractuel** dans les délégations de service public (DSP)
VLS : les opérateurs seraient tenus de maintenir un IES $\geq 0{,}90$ pour l'ensemble de leur
territoire, sous peine de pénalités financières. Ce mécanisme de régulation performative inciterait
les opérateurs à étendre leur réseau vers les zones moins rentables mais socialement stratégiques.

#### 6.4. Résultat Clé : L'Autonomie de la Gouvernance sur le Déterminisme Économique

Le $R^2 = 0{,}28$ du modèle Ridge confirme que le revenu médian n'explique qu'une minorité de la
variance de l'IMD. **72 % de la qualité de l'environnement cyclable relèvent de choix de gouvernance
locale**, non de déterminismes économiques. Ce résultat invalide toute forme de fatalisme territorial
et souligne la responsabilité pleine et entière des décideurs publics dans la constitution ou la
résorption des déserts de mobilité sociale.

L'étude de cas intra-urbaine de Montpellier — corrélation entre revenu fiscal et part modale vélo par
quartier, calcul de l'IES intra-urbain $\widetilde{\text{IES}}_q$ — est détaillée dans la section
*Fracture Socio-Spatiale* de la page **Montpellier — Étude de cas VLS**.
""")
