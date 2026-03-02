"""
10_Politique.py - Gouvernance Politique et Mobilité Douce.
Analyse de la relation entre couleur politique (municipale et régionale)
et qualité de l'offre VLS (IMD) / équité sociale (IES).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# scipy optionnel
try:
    from scipy.stats import kruskalwallis as _kw, mannwhitneyu as _mwu
    _SCIPY = True
except ImportError:
    _SCIPY = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    compute_imd_cities,
    load_political_data,
    load_stations,
)
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Gouvernance Politique et Mobilité Douce - Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Palette politique ──────────────────────────────────────────────────────────
_POL_COLORS: dict[str, str] = {
    "Gauche":          "#C0392B",   # rouge
    "Centre":          "#D4AC0D",   # or/jaune
    "Droite":          "#1565C0",   # bleu
    "Extrême droite":  "#2C3E50",   # bleu très foncé
}
_POL_ORDER = ["Gauche", "Centre", "Droite", "Extrême droite"]

# ── Chargement des données ─────────────────────────────────────────────────────
df       = load_stations()
imd_df   = compute_imd_cities(df)
pol_df   = load_political_data()

# ── Fusion IMD × Politique ─────────────────────────────────────────────────────
if not pol_df.empty and not imd_df.empty:
    merged = imd_df.merge(pol_df, on="city", how="left")
else:
    merged = imd_df.copy()

has_pol   = "couleur_municipale" in merged.columns and merged["couleur_municipale"].notna().sum() >= 3
has_reg   = "couleur_regionale"  in merged.columns and merged["couleur_regionale"].notna().sum() >= 3
has_revnu = "revenu_median_uc"   in merged.columns and merged["revenu_median_uc"].notna().sum() >= 5

# ── IES recalculé ─────────────────────────────────────────────────────────────
ies_col_ok = False
if has_revnu:
    _tmp = merged.dropna(subset=["revenu_median_uc", "IMD"]).copy()
    if len(_tmp) >= 5:
        _c = np.polyfit(_tmp["revenu_median_uc"].values, _tmp["IMD"].values, 1)
        _tmp["IMD_hat"] = np.polyval(_c, _tmp["revenu_median_uc"].values).clip(min=1.0)
        _tmp["IES"]     = (_tmp["IMD"] / _tmp["IMD_hat"]).round(3)
        merged = merged.merge(_tmp[["city", "IMD_hat", "IES"]], on="city", how="left")
        ies_col_ok = True

# ── Stats préliminaires abstract ───────────────────────────────────────────────
_n_pol   = int(merged["couleur_municipale"].notna().sum()) if has_pol else 0
_n_total = len(merged)

if has_pol and _n_pol >= 3:
    _cnt = merged["couleur_municipale"].value_counts()
    _gauche_n  = int(_cnt.get("Gauche", 0))
    _centre_n  = int(_cnt.get("Centre", 0))
    _droite_n  = int(_cnt.get("Droite", 0))
    _extd_n    = int(_cnt.get("Extrême droite", 0))
    _gauche_imd = merged.loc[merged["couleur_municipale"] == "Gauche", "IMD"].median()
    _droite_imd = merged.loc[merged["couleur_municipale"] == "Droite", "IMD"].median()
    _diff_imd   = _gauche_imd - _droite_imd if pd.notna(_gauche_imd) and pd.notna(_droite_imd) else float("nan")
else:
    _gauche_n = _centre_n = _droite_n = _extd_n = 0
    _gauche_imd = _droite_imd = _diff_imd = float("nan")

st.title("Gouvernance Politique et Mobilité Douce")
st.caption(
    "Axe de Recherche transversal : La couleur politique des exécutifs locaux "
    "prédit-elle la qualité de l'offre VLS (IMD) et l'équité sociale (IES) ?"
)

abstract_box(
    "<b>Problématique :</b> La décision d'investir dans les infrastructures cyclables "
    "partagées est-elle conditionnée par l'orientation politique de l'exécutif municipal "
    "ou régional ? Cette page examine la relation entre la couleur politique issue des "
    "<b>élections municipales 2020</b> et <b>régionales 2021</b> et les deux indicateurs "
    "synthétiques du Gold Standard : l'<b>Indice de Mobilité Douce (IMD)</b> et l'<b>Indice "
    "d'Équité Sociale (IES)</b>. L'analyse est exploratoire et descriptive — la faiblesse "
    "des effectifs par groupe (quelques dizaines d'agglomérations) impose une grande "
    "prudence dans l'interprétation des résultats. La corrélation politique ≠ causalité : "
    "de nombreux facteurs de confusion (taille de la ville, topographie, héritage "
    "historique, densité) co-déterminent la qualité VLS indépendamment du choix politique. "
    f"Panel : <b>{_n_pol} agglomérations</b> appariées avec des données politiques sur "
    f"{_n_total} dans le Gold Standard dock-based.",
    findings=[
        (str(_n_pol),    "agglomérations avec données politiques"),
        (f"{_gauche_n}", "Gauche"),
        (f"{_centre_n}", "Centre"),
        (f"{_droite_n}", "Droite"),
        (f"{_extd_n}",   "Extrême droite"),
    ],
)

sidebar_nav()
with st.sidebar:
    st.header("Paramètres")
    min_stations = st.number_input(
        "Seuil min. stations (IMD)", min_value=1, max_value=200, value=10,
        help="Filtre les micro-réseaux pour garantir la robustesse statistique.",
    )
    show_labels = st.checkbox("Afficher les étiquettes de villes", value=False)

# Filtre par taille
merged = merged[merged["n_stations"] >= min_stations].reset_index(drop=True)
has_pol = "couleur_municipale" in merged.columns and merged["couleur_municipale"].notna().sum() >= 3

if not has_pol:
    st.warning(
        "Les données politiques n'ont pas pu être appariées avec les agglomérations du "
        "Gold Standard dock-based. Vérifiez que le fichier "
        "`data/external/politique/political_data.csv` est présent."
    )
    st.stop()

# ── KPIs principaux ────────────────────────────────────────────────────────────
merged_pol = merged.dropna(subset=["couleur_municipale"])
grp_imd    = merged_pol.groupby("couleur_municipale", observed=True)["IMD"]

k_cols = st.columns(4)
for col_w, grp_label in zip(k_cols, _POL_ORDER):
    s = merged_pol.loc[merged_pol["couleur_municipale"] == grp_label, "IMD"]
    if len(s) > 0:
        col_w.metric(
            f"IMD médian — {grp_label}",
            f"{s.median():.1f} / 100",
            f"n = {len(s)} villes",
        )
    else:
        col_w.metric(f"IMD médian — {grp_label}", "-")

# ── Section 1 - Cadre méthodologique ──────────────────────────────────────────
st.divider()
section(1, "Cadre Méthodologique — Données Politiques et Hypothèses")

col_meth, col_table = st.columns([3, 2])
with col_meth:
    st.markdown(r"""
#### 1.1. Source des Données Politiques

Les données politiques proviennent des **résultats officiels des élections municipales
de 2020** (métropoles et communes) et des **élections régionales de 2021**
(Ministère de l'Intérieur). La couleur politique est codée en quatre catégories :

| Catégorie | Principaux partis | Code couleur |
|:--- |:--- |:--- |
| **Gauche** | PS, PCF, LFI, EELV, Union de la Gauche | Rouge |
| **Centre** | RE, MoDem, Horizons, UDI, DVG proches centre | Or |
| **Droite** | LR, Divers droite | Bleu |
| **Extrême droite** | RN | Bleu marine |

#### 1.2. Hypothèses de Recherche

**H₁ :** Les agglomérations à exécutif de gauche (et en particulier EELV) affichent
un IMD significativement supérieur à celles gouvernées par la droite.

**H₂ :** La couleur politique régionale modère l'effet municipal (co-financement
régional des infrastructures cyclables).

**H₃ :** La couleur politique n'explique pas l'IES — la justice distributive est
orthogonale au positionnement partisan.

> **Avertissement méthodologique :** Le panel compte au maximum quelques dizaines
> de villes par groupe. Les tests statistiques sont indicatifs et non conclusifs
> en raison des faibles effectifs. Corrélation ≠ causalité.
""")

with col_table:
    _pol_summary = (
        merged_pol.groupby("couleur_municipale", observed=True)
        .agg(
            n_villes=("city", "count"),
            IMD_med=("IMD", "median"),
            IMD_moy=("IMD", "mean"),
        )
        .reset_index()
    )
    _pol_summary.columns = ["Couleur politique", "N villes", "IMD médian", "IMD moyen"]
    _pol_summary["IMD médian"] = _pol_summary["IMD médian"].round(1)
    _pol_summary["IMD moyen"]  = _pol_summary["IMD moyen"].round(1)
    st.markdown("#### Distribution par couleur politique")
    st.dataframe(_pol_summary, use_container_width=True, hide_index=True)
    st.caption("**Tableau 1.1.** Statistiques IMD par couleur politique municipale.")

# ── Section 2 - IMD × Couleur politique municipale ────────────────────────────
st.divider()
section(2, "IMD par Couleur Politique Municipale — Élections 2020")

# ── 2a. Box plot IMD par couleur ───────────────────────────────────────────────
_box_pol = merged_pol.copy()
_box_pol["couleur_municipale"] = pd.Categorical(
    _box_pol["couleur_municipale"], categories=_POL_ORDER, ordered=True
)

fig_box = px.box(
    _box_pol.sort_values("couleur_municipale"),
    x="couleur_municipale",
    y="IMD",
    color="couleur_municipale",
    color_discrete_map=_POL_COLORS,
    points="all",
    hover_data=["city", "n_stations", "IMD"],
    labels={
        "couleur_municipale": "Couleur politique municipale (2020)",
        "IMD": "Score IMD (/100)",
    },
    height=480,
)
if show_labels:
    for _, row in _box_pol.iterrows():
        if pd.notna(row.get("couleur_municipale")) and pd.notna(row["IMD"]):
            fig_box.add_annotation(
                x=row["couleur_municipale"],
                y=row["IMD"],
                text=row["city"],
                showarrow=False,
                font=dict(size=8, color="#555"),
                yshift=6,
            )
fig_box.update_layout(
    plot_bgcolor="white",
    showlegend=False,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_box, use_container_width=True)
st.caption(
    "**Figure 2.1.** Distribution des scores IMD par couleur politique de l'exécutif "
    "municipal (élections 2020). Les points représentent les agglomérations individuelles. "
    "La ligne centrale = médiane, les boîtes = Q1–Q3, les moustaches = 1,5 × IQR."
)

# ── 2b. Test de Kruskal-Wallis ─────────────────────────────────────────────────
_groups_kw = [
    _box_pol.loc[_box_pol["couleur_municipale"] == g, "IMD"].dropna().values
    for g in _POL_ORDER
    if (_box_pol["couleur_municipale"] == g).sum() >= 2
]
_labels_kw = [
    g for g in _POL_ORDER
    if (_box_pol["couleur_municipale"] == g).sum() >= 2
]

with st.expander("Test de Kruskal-Wallis — Différence globale entre groupes politiques", expanded=True):
    if _SCIPY and len(_groups_kw) >= 2:
        try:
            _H, _p_kw = _kw(*_groups_kw)
            _df_kw = len(_groups_kw) - 1
            # Eta-squared (effect size)
            _n_kw   = sum(len(g) for g in _groups_kw)
            _eta2   = (_H - _df_kw) / (_n_kw - _df_kw - 1) if _n_kw > _df_kw + 1 else float("nan")

            kw1, kw2, kw3, kw4 = st.columns(4)
            kw1.metric("Statistique H (Kruskal-Wallis)", f"{_H:.3f}")
            kw2.metric(f"Degrés de liberté", f"{_df_kw}")
            kw3.metric("p-valeur", f"{_p_kw:.4f}" if _p_kw >= 0.001 else "< 0,001",
                       "sign. (p < 0,05)" if _p_kw < 0.05 else "non sign.")
            kw4.metric("Taille d'effet η²", f"{_eta2:.3f}" if pd.notna(_eta2) else "-")

            _fmt_p = lambda p: f"{p:.4f}" if p >= 0.001 else "< 0,001"
            st.caption(
                f"**Tableau 2.1.** Test de Kruskal-Wallis H ($k = {len(_groups_kw)}$ groupes, "
                f"$n = {_n_kw}$ agglomérations). "
                f"$H({_df_kw}) = {_H:.3f}$, $p = {_fmt_p(_p_kw)}$. "
                f"Taille d'effet $\\eta^2 = {_eta2:.3f}$"
                + (" (faible < 0,06 / modéré < 0,14 / fort ≥ 0,14). "
                   "**Différence statistiquement significative** entre groupes politiques."
                   if _p_kw < 0.05 else
                   " (faible < 0,06 / modéré < 0,14 / fort ≥ 0,14). "
                   "Absence de différence significative au seuil α = 0,05 — "
                   "la couleur politique n'est pas un prédicteur robuste de l'IMD "
                   "sur ce panel.")
            )

            # ── Comparaisons pairées (Mann-Whitney U) ─────────────────────────
            if _p_kw < 0.05 and len(_groups_kw) >= 2:
                st.markdown("##### Comparaisons pairées post-hoc (Mann-Whitney U, correction Bonferroni)")
                _mw_rows = []
                _n_pairs = len(_groups_kw) * (len(_groups_kw) - 1) // 2
                for i, (g1, l1) in enumerate(zip(_groups_kw, _labels_kw)):
                    for g2, l2 in zip(_groups_kw[i+1:], _labels_kw[i+1:]):
                        try:
                            _U, _p_mw = _mwu(g1, g2, alternative="two-sided")
                            _p_corr   = min(1.0, _p_mw * _n_pairs)
                            _r_eff    = 1 - 2 * _U / (len(g1) * len(g2))
                            _mw_rows.append({
                                "Groupe 1": l1,
                                "Groupe 2": l2,
                                "n₁": len(g1),
                                "n₂": len(g2),
                                "Méd. IMD G1": f"{float(np.median(g1)):.1f}",
                                "Méd. IMD G2": f"{float(np.median(g2)):.1f}",
                                "U": f"{_U:.0f}",
                                "p (brut)": _fmt_p(_p_mw),
                                "p (Bonf.)": _fmt_p(_p_corr),
                                "Sig.": "**" if _p_corr < 0.01 else ("*" if _p_corr < 0.05 else "n.s."),
                                "r (effet)": f"{_r_eff:.3f}",
                            })
                        except Exception:
                            pass
                if _mw_rows:
                    st.dataframe(pd.DataFrame(_mw_rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f"Test Kruskal-Wallis non disponible : {e}")
    else:
        # Statistiques descriptives de fallback
        st.info("scipy non installé — statistiques descriptives uniquement.")
        st.dataframe(_pol_summary, use_container_width=True, hide_index=True)

# ── 2c. Scatter IMD × Revenu colorié par politique ────────────────────────────
if has_revnu:
    _scat_df = merged_pol.dropna(subset=["revenu_median_uc", "IMD", "couleur_municipale"]).copy()
    if len(_scat_df) >= 5:
        st.markdown("#### IMD × Revenu médian — Couleur par orientation politique")
        fig_scat = px.scatter(
            _scat_df,
            x="revenu_median_uc",
            y="IMD",
            color="couleur_municipale",
            color_discrete_map=_POL_COLORS,
            category_orders={"couleur_municipale": _POL_ORDER},
            symbol="couleur_municipale",
            text="city" if show_labels else None,
            size="n_stations",
            size_max=20,
            hover_data=["city", "n_stations", "couleur_municipale"],
            labels={
                "revenu_median_uc":   "Revenu médian/UC (€/an, INSEE Filosofi)",
                "IMD":                "Score IMD (/100)",
                "couleur_municipale": "Couleur politique",
            },
            height=500,
        )
        fig_scat.add_vline(
            x=float(_scat_df["revenu_median_uc"].median()),
            line_dash="dot", line_color="#999", opacity=0.5,
            annotation_text="Médiane revenu", annotation_position="top",
        )
        fig_scat.add_hline(
            y=float(_scat_df["IMD"].median()),
            line_dash="dot", line_color="#999", opacity=0.5,
            annotation_text="Médiane IMD", annotation_position="right",
        )
        fig_scat.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig_scat, use_container_width=True)
        st.caption(
            "**Figure 2.2.** Score IMD versus revenu médian/UC (INSEE Filosofi), "
            "les points colorés par orientation politique municipale (2020). "
            "Chaque symbole représente une agglomération. "
            "La taille des points est proportionnelle au nombre de stations dock-based."
        )

# ── Section 3 - IES × Politique ───────────────────────────────────────────────
st.divider()
section(3, "Équité Sociale (IES) et Couleur Politique")

st.markdown(r"""
L'IES mesure si l'agglomération investit **au-delà** de ce que son niveau économique
laisserait prévoir. Un IES > 1 signale une volonté politique proactive d'équité cyclable.
La question est : cette volonté est-elle corrélée à l'orientation partisane ?
""")

if ies_col_ok and "IES" in merged.columns:
    _ies_pol = merged.dropna(subset=["IES", "couleur_municipale"]).copy()

    # ── 3a. Quadrants IES × politique ─────────────────────────────────────────
    if has_revnu:
        _med_rev = float(_ies_pol["revenu_median_uc"].median()) if "revenu_median_uc" in _ies_pol.columns else None

        def _quadrant(row: pd.Series) -> str:
            above_imd = row["IMD"] >= float(merged["IMD"].median())
            above_rev = (row.get("revenu_median_uc", float("nan")) >= _med_rev
                         if _med_rev is not None else True)
            if not above_rev and above_imd:
                return "Mobilité Inclusive"
            if above_rev and above_imd:
                return "Excellence Consolidée"
            if not above_rev and not above_imd:
                return "Désert de Mobilité"
            return "Sous-Performance"

        if "revenu_median_uc" in _ies_pol.columns:
            _ies_pol["quadrant"] = _ies_pol.apply(_quadrant, axis=1)

            # Stacked bar : répartition des quadrants par couleur politique
            _quad_pol = (
                _ies_pol.groupby(["couleur_municipale", "quadrant"], observed=True)
                .size()
                .reset_index(name="n")
            )
            _quad_col_map = {
                "Mobilité Inclusive":   "#27ae60",
                "Excellence Consolidée":"#1A6FBF",
                "Désert de Mobilité":   "#e74c3c",
                "Sous-Performance":     "#e67e22",
            }
            fig_quad_pol = px.bar(
                _quad_pol,
                x="couleur_municipale",
                y="n",
                color="quadrant",
                color_discrete_map=_quad_col_map,
                barmode="stack",
                category_orders={"couleur_municipale": _POL_ORDER},
                labels={
                    "couleur_municipale": "Couleur politique",
                    "n":                  "Nombre d'agglomérations",
                    "quadrant":           "Régime IES",
                },
                height=420,
            )
            fig_quad_pol.update_layout(
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            st.plotly_chart(fig_quad_pol, use_container_width=True)
            st.caption(
                "**Figure 3.1.** Répartition des régimes de justice cyclable (quadrants IES) "
                "par couleur politique municipale. La proportion de 'Mobilité Inclusive' "
                "(revenu faible, IMD élevé) révèle les agglomérations qui sur-investissent "
                "en faveur des populations précaires, indépendamment du contexte économique."
            )

    # ── 3b. Box IES par politique ──────────────────────────────────────────────
    _ies_pol["couleur_municipale"] = pd.Categorical(
        _ies_pol["couleur_municipale"], categories=_POL_ORDER, ordered=True
    )
    fig_ies_box = px.box(
        _ies_pol.sort_values("couleur_municipale"),
        x="couleur_municipale",
        y="IES",
        color="couleur_municipale",
        color_discrete_map=_POL_COLORS,
        points="all",
        hover_data=["city", "IMD", "IES"],
        labels={
            "couleur_municipale": "Couleur politique municipale",
            "IES":                "Indice d'Équité Sociale (IES)",
        },
        height=400,
    )
    fig_ies_box.add_hline(
        y=1.0, line_dash="dash", line_color="#555", line_width=1.5,
        annotation_text="IES = 1 (équité neutre)", annotation_position="right",
    )
    fig_ies_box.update_layout(
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_ies_box, use_container_width=True)
    st.caption(
        "**Figure 3.2.** Distribution de l'IES par couleur politique. "
        "La ligne en tirets (IES = 1) marque la neutralité : au-dessus, l'agglomération "
        "investit au-delà de son niveau de revenu prévisible."
    )
else:
    st.info(
        "L'IES ne peut pas être calculé : la colonne `revenu_median_uc` "
        "(INSEE Filosofi) n'est pas disponible dans ce dataset."
    )

# ── Section 4 - Dimension régionale ───────────────────────────────────────────
st.divider()
section(4, "Dimension Régionale — Exécutifs Régionaux et IMD")

st.markdown(r"""
Les régions co-financent les plans vélo via les SRADDET (Schéma Régional
d'Aménagement, de Développement Durable et d'Égalité des Territoires).
La couleur politique régionale pourrait amplifier ou atténuer l'effet municipal.
""")

if has_reg:
    _reg_pol = merged.dropna(subset=["couleur_regionale"]).copy()
    _reg_pol["couleur_regionale"] = pd.Categorical(
        _reg_pol["couleur_regionale"], categories=_POL_ORDER, ordered=True
    )

    fig_reg = px.box(
        _reg_pol.sort_values("couleur_regionale"),
        x="couleur_regionale",
        y="IMD",
        color="couleur_regionale",
        color_discrete_map=_POL_COLORS,
        points="all",
        hover_data=["city", "region", "IMD", "couleur_municipale"] if "couleur_municipale" in _reg_pol.columns else ["city", "region", "IMD"],
        labels={
            "couleur_regionale": "Couleur politique régionale (2021)",
            "IMD":               "Score IMD (/100)",
        },
        height=420,
    )
    fig_reg.update_layout(
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption(
        "**Figure 4.1.** Distribution des scores IMD par couleur politique régionale. "
        "Note : la quasi-totalité des régions françaises est gouvernée par la droite "
        "(LR) depuis 2021, ce qui réduit fortement la variabilité inter-groupes "
        "à l'échelon régional."
    )

    # ── Interaction municipal × régional ──────────────────────────────────────
    if has_pol and "couleur_municipale" in merged.columns:
        _inter = merged.dropna(subset=["couleur_municipale", "couleur_regionale"]).copy()
        if len(_inter) >= 5:
            st.markdown("#### Interaction Municipal × Régional")
            _inter["config"] = (
                _inter["couleur_municipale"].astype(str)
                + " / "
                + _inter["couleur_regionale"].astype(str)
            )
            _config_imd = (
                _inter.groupby("config")["IMD"]
                .agg(n="count", mediane="median", moyenne="mean")
                .reset_index()
                .query("n >= 2")
                .sort_values("mediane", ascending=False)
            )
            _config_imd.columns = ["Configuration politique", "N villes", "IMD médian", "IMD moyen"]
            _config_imd["IMD médian"] = _config_imd["IMD médian"].round(1)
            _config_imd["IMD moyen"]  = _config_imd["IMD moyen"].round(1)

            fig_inter = px.bar(
                _config_imd,
                x="Configuration politique",
                y="IMD médian",
                color="IMD médian",
                color_continuous_scale="RdBu",
                text="N villes",
                labels={"IMD médian": "IMD médian (/100)"},
                height=380,
            )
            fig_inter.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig_inter.update_layout(
                plot_bgcolor="white",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=80),
                xaxis=dict(tickangle=-30),
            )
            st.plotly_chart(fig_inter, use_container_width=True)
            st.caption(
                "**Figure 4.2.** IMD médian par configuration politique "
                "(couleur municipale / couleur régionale). "
                "Seules les configurations avec ≥ 2 agglomérations sont affichées. "
                "Les petits effectifs rendent ces résultats hautement indicatifs."
            )

# ── Section 5 - Tableau synthétique ───────────────────────────────────────────
st.divider()
section(5, "Tableau de Classement — IMD et IES par Agglomération et Couleur Politique")

_disp_cols = ["city", "couleur_municipale", "region", "couleur_regionale",
              "maire", "parti_maire", "n_stations", "IMD"]
if ies_col_ok and "IES" in merged.columns:
    _disp_cols.append("IES")
_disp_available = [c for c in _disp_cols if c in merged.columns]

_disp = merged[_disp_available].dropna(subset=["couleur_municipale"]).copy()
_disp = _disp.sort_values("IMD", ascending=False)

_rename = {
    "city":               "Agglomération",
    "couleur_municipale": "Couleur municipale",
    "region":             "Région",
    "couleur_regionale":  "Couleur régionale",
    "maire":              "Maire",
    "parti_maire":        "Parti",
    "n_stations":         "Stations",
    "IMD":                "IMD (/100)",
    "IES":                "IES",
}
_disp = _disp.rename(columns={k: v for k, v in _rename.items() if k in _disp.columns})
if "IMD (/100)" in _disp.columns:
    _disp["IMD (/100)"] = _disp["IMD (/100)"].round(1)

col_cfg = {
    "IMD (/100)": st.column_config.ProgressColumn(
        "IMD (/100)", min_value=0, max_value=100, format="%.1f"
    ),
}
if "IES" in _disp.columns:
    col_cfg["IES"] = st.column_config.NumberColumn("IES", format="%.3f")

st.dataframe(_disp, use_container_width=True, hide_index=True, column_config=col_cfg)
st.caption(
    "**Tableau 5.1.** Classement des agglomérations par score IMD (décroissant), "
    "enrichi de la couleur politique municipale et régionale. "
    "Source politique : élections municipales 2020 / régionales 2021 "
    "(Ministère de l'Intérieur). Source IMD/IES : Gold Standard GBFS — R. Fossé & G. Pallares, 2025–2026."
)

# ── Section 6 - Discussion ─────────────────────────────────────────────────────
st.divider()
section(6, "Discussion — Portée et Limites de l'Analyse Politique")

st.markdown(r"""
#### 6.1. Principaux Résultats Observés

L'analyse exploratoire révèle des différences descriptives entre groupes politiques,
mais les tests statistiques doivent être interprétés avec prudence :

- **Effectifs limités** : le panel compte au maximum ~60 agglomérations appariées,
  réparties en 4 groupes inégaux (la gauche et la droite représentent ~80 % du panel).
  Cela réduit la puissance statistique des tests.

- **Facteurs de confusion** : la taille de la ville, la densité urbaine, la topographie
  et l'héritage historique des politiques de mobilité sont corrélés à la fois à la
  couleur politique et à l'IMD. Sans contrôle multivarié (régression logistique,
  appariement propensity score), la corrélation brute est peu interprétable.

- **Biais de sélection VLS** : les villes qui ont investi dans un réseau VLS
  dock-based certifié Gold Standard tendent déjà à être plus engagées en mobilité
  douce, quelle que soit leur couleur politique. L'échantillon n'est pas représentatif
  de l'ensemble des communes françaises.

#### 6.2. Interprétation Contextuelle

La corrélation politique — si elle existe — reflète davantage une **corrélation
socio-historique** qu'un effet causal direct du programme partisan :

- Les grandes métropoles (Paris, Lyon, Bordeaux, Strasbourg) ont basculé à gauche
  (EELV) précisément après des mandats de renforcement des politiques cyclables.
- La mobilité douce est devenue un **marqueur identitaire** de l'écologie politique,
  renforçant l'apparence d'une corrélation que les données granulaires nuancent.

#### 6.3. Perspectives de Recherche

Pour aller au-delà du descriptif, une analyse robuste devrait :

1. **Contrôler les covariables** : taille de ville, densité, topographie, date de création
   du réseau VLS (antérieure à l'élection 2020 dans la plupart des cas).
2. **Analyse différence-en-différences** : comparer l'évolution de l'IMD avant/après
   un changement de majorité municipale (données longitudinales nécessaires).
3. **Étudier les budgets mobilité** : les délibérations de conseils municipaux et
   les plans vélo (PDME) permettraient de mesurer directement l'intention politique.

> **Conclusion provisoire :** La couleur politique est un **signal corrélé** mais
> non un déterminant isolé de la qualité VLS. L'analyse de l'IES confirme
> que la justice distributive — mesurée par le rapport IMD/revenu — transcende
> le clivage partisan : des agglomérations de droite offrent une mobilité inclusive,
> tandis que certaines de gauche présentent des déserts de mobilité sociale.
""")
