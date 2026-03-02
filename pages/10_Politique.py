"""
10_Politique.py - Gouvernance Politique et Mobilité Douce.
Analyse de la relation entre le parti politique de l'exécutif municipal
et la qualité de l'offre VLS (IMD) / équité sociale (IES).
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
from utils.data_loader import compute_imd_cities, load_political_data, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Gouvernance Politique et Mobilité Douce - Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Palettes ───────────────────────────────────────────────────────────────────
_COULEUR_COLORS: dict[str, str] = {
    "Gauche":         "#C0392B",
    "Centre":         "#D4AC0D",
    "Droite":         "#1565C0",
    "Extrême droite": "#2C3E50",
}
_COULEUR_ORDER = ["Gauche", "Centre", "Droite", "Extrême droite"]

# Couleurs distinctes par parti (intérieur de chaque bloc)
_PARTI_COLORS: dict[str, str] = {
    "EELV":     "#27ae60",  # vert écologie
    "PS":       "#C0392B",  # rouge socialiste
    "PCF":      "#8B0000",  # rouge foncé communiste
    "DVG":      "#E8A090",  # rose clair divers gauche
    "MoDem":    "#F1C40F",  # jaune centre
    "Horizons": "#E67E22",  # orange centre
    "UDI":      "#D4AC0D",  # or centre
    "Centre":   "#BFA04A",  # centre neutre
    "LR":       "#1565C0",  # bleu républicain
    "RN":       "#2C3E50",  # bleu nuit extrême droite
    "Corse":    "#7F8C8D",  # gris régionaliste
}

# ── Chargement des données ─────────────────────────────────────────────────────
df     = load_stations()
imd_df = compute_imd_cities(df)
pol_df = load_political_data()

# ── Fusion IMD × Politique ─────────────────────────────────────────────────────
if not pol_df.empty and not imd_df.empty:
    merged = imd_df.merge(pol_df, on="city", how="left")
else:
    merged = imd_df.copy()

has_pol   = "couleur_municipale" in merged.columns and merged["couleur_municipale"].notna().sum() >= 3
has_parti = "parti_maire" in merged.columns and merged["parti_maire"].notna().sum() >= 3
has_reg   = "couleur_regionale"  in merged.columns and merged["couleur_regionale"].notna().sum() >= 3
has_revnu = "revenu_median_uc"   in merged.columns and merged["revenu_median_uc"].notna().sum() >= 5

# ── IES recalculé ──────────────────────────────────────────────────────────────
ies_col_ok = False
if has_revnu:
    _tmp = merged.dropna(subset=["revenu_median_uc", "IMD"]).copy()
    if len(_tmp) >= 5:
        _c = np.polyfit(_tmp["revenu_median_uc"].values, _tmp["IMD"].values, 1)
        _tmp["IMD_hat"] = np.polyval(_c, _tmp["revenu_median_uc"].values).clip(min=1.0)
        _tmp["IES"]     = (_tmp["IMD"] / _tmp["IMD_hat"]).round(3)
        merged = merged.merge(_tmp[["city", "IMD_hat", "IES"]], on="city", how="left")
        ies_col_ok = True

# ── Stats préliminaires (pour l'abstract, avant filtre sidebar) ────────────────
_n_pol    = int(merged["parti_maire"].notna().sum()) if has_parti else 0
_n_total  = len(merged)
_n_partis = int(merged["parti_maire"].nunique()) if has_parti else 0

if has_parti and _n_pol >= 3:
    _parti_counts = merged["parti_maire"].value_counts()
    _top_parti    = str(_parti_counts.index[0]) if len(_parti_counts) > 0 else "—"
    _top_n        = int(_parti_counts.iloc[0]) if len(_parti_counts) > 0 else 0
else:
    _top_parti = "—"
    _top_n = 0

# ── Titre et résumé ────────────────────────────────────────────────────────────
st.title("Gouvernance Politique et Mobilité Douce")
st.caption(
    "Axe de Recherche transversal : Le parti politique des exécutifs locaux "
    "prédit-il la qualité de l'offre VLS (IMD) et l'équité sociale (IES) ?"
)

abstract_box(
    "<b>Problématique :</b> La décision d'investir dans les infrastructures de "
    "micromobilité partagée est-elle conditionnée par l'appartenance partisane de "
    "l'exécutif municipal ? Cette page examine la relation entre le <b>parti du maire</b> "
    "issu des <b>élections municipales 2020</b> et les indicateurs du Gold Standard : "
    "l'<b>IMD</b> (Indice de Mobilité Douce) et l'<b>IES</b> (Indice d'Équité Sociale). "
    "L'analyse est exploratoire — la faiblesse des effectifs par parti impose une grande "
    "prudence dans l'interprétation. Corrélation politique ≠ causalité : taille de ville, "
    "topographie, héritage historique et densité co-déterminent la qualité VLS "
    f"indépendamment du choix partisan. Panel : <b>{_n_pol} agglomérations</b>, "
    f"<b>{_n_partis} partis</b> représentés.",
    findings=[
        (str(_n_pol),    "agglomérations"),
        (str(_n_partis), "partis représentés"),
        (_top_parti,     f"parti le plus fréquent (n={_top_n})"),
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
    min_villes_parti = st.slider(
        "Min. agglomérations par parti (regroupement)",
        min_value=1, max_value=5, value=2,
        help="Les partis avec moins de N villes sont fusionnés en 'Autres'.",
    )

# ── Filtre par taille et recalcul des flags ────────────────────────────────────
merged    = merged[merged["n_stations"] >= min_stations].reset_index(drop=True)
has_pol   = "couleur_municipale" in merged.columns and merged["couleur_municipale"].notna().sum() >= 3
has_parti = "parti_maire" in merged.columns and merged["parti_maire"].notna().sum() >= 3
has_reg   = "couleur_regionale"  in merged.columns and merged["couleur_regionale"].notna().sum() >= 3

if not has_pol and not has_parti:
    st.warning(
        "Les données politiques n'ont pas pu être appariées. "
        "Vérifiez que `data/external/politique/political_data.csv` est présent."
    )
    st.stop()

# ── Préparation : regroupement des petits partis ───────────────────────────────
merged_pol = merged.dropna(subset=["parti_maire"]).copy()
_pcounts   = merged_pol["parti_maire"].value_counts()
_partis_maj = _pcounts[_pcounts >= min_villes_parti].index.tolist()


def _grp_parti(p: str) -> str:
    return p if p in _partis_maj else "Autres"


merged_pol["parti_grp"] = merged_pol["parti_maire"].apply(_grp_parti)

_parti_grp_colors: dict[str, str] = {p: _PARTI_COLORS.get(p, "#95a5a6") for p in _partis_maj}
_parti_grp_colors["Autres"] = "#95a5a6"

# Ordre des partis par médiane IMD décroissante (Autres en dernier)
_parti_order_by_imd: list[str] = (
    merged_pol[merged_pol["parti_grp"] != "Autres"]
    .groupby("parti_grp")["IMD"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)
if (merged_pol["parti_grp"] == "Autres").any():
    _parti_order_by_imd.append("Autres")

# ── KPIs principaux (par couleur) ──────────────────────────────────────────────
merged_couleur = merged.dropna(subset=["couleur_municipale"]) if has_pol else pd.DataFrame()
kpi_cols = st.columns(4)
for col_w, grp_label in zip(kpi_cols, _COULEUR_ORDER):
    if not merged_couleur.empty:
        s = merged_couleur.loc[merged_couleur["couleur_municipale"] == grp_label, "IMD"]
        col_w.metric(grp_label, f"{s.median():.1f} / 100" if len(s) > 0 else "—",
                     f"n = {len(s)} villes" if len(s) > 0 else "0 ville")
    else:
        col_w.metric(grp_label, "—")

# ── Section 1 - Cadre méthodologique ──────────────────────────────────────────
st.divider()
section(1, "Cadre Méthodologique — Données Politiques et Hypothèses")

col_meth, col_table = st.columns([3, 2])
with col_meth:
    st.markdown(r"""
#### 1.1. Source des Données Politiques

Les données politiques proviennent des **résultats officiels des élections municipales
2020** et des **élections régionales 2021** (Ministère de l'Intérieur). L'unité
d'analyse est le **parti du maire** — catégorie plus fine que la couleur politique
agrégée :

| Parti | Bloc | Description |
|:--- |:--- |:--- |
| **EELV** | Gauche | Europe Écologie Les Verts |
| **PS** | Gauche | Parti Socialiste |
| **PCF** | Gauche | Parti Communiste Français |
| **DVG** | Centre | Divers gauche / liste citoyenne |
| **MoDem** | Centre | Mouvement Démocrate |
| **Horizons** | Centre | Horizons (É. Philippe) |
| **UDI** | Centre | Union des Démocrates et Indépendants |
| **LR** | Droite | Les Républicains |
| **RN** | Extrême droite | Rassemblement National |

#### 1.2. Hypothèses de Recherche

**H₁ :** Les villes à majorité EELV affichent un IMD significativement supérieur
aux autres partis (programmes vélo explicites dans les mandats 2020).

**H₂ :** La couleur politique régionale modère l'effet municipal via le
co-financement des plans vélo (SRADDET).

**H₃ :** L'IES est orthogonal au positionnement partisan — la justice distributive
transcende les clivages partisans.

> **Avertissement :** Panel de ~60 agglomérations. Effectifs très faibles par parti.
> Tests statistiques indicatifs uniquement. Corrélation ≠ causalité.
""")

with col_table:
    _parti_tbl = (
        merged_pol.groupby("parti_grp")
        .agg(n_villes=("city", "count"), IMD_med=("IMD", "median"), IMD_moy=("IMD", "mean"))
        .reset_index()
        .sort_values("IMD_med", ascending=False)
    )
    _parti_tbl.columns = ["Parti", "N villes", "IMD médian", "IMD moyen"]
    _parti_tbl["IMD médian"] = _parti_tbl["IMD médian"].round(1)
    _parti_tbl["IMD moyen"]  = _parti_tbl["IMD moyen"].round(1)
    st.markdown("#### IMD par parti")
    st.dataframe(
        _parti_tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "IMD médian": st.column_config.ProgressColumn(
                "IMD médian", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )
    st.caption("**Tableau 1.1.** IMD par parti du maire (triés par médiane décroissante).")

# ── Section 2 - IMD × Parti ────────────────────────────────────────────────────
st.divider()
section(2, "IMD par Parti Politique Municipale — Élections 2020")

# ── 2.1 Bar chart IMD médian par parti ────────────────────────────────────────
st.markdown("#### 2.1. Score IMD médian par Parti")

_bar_df = (
    merged_pol.groupby("parti_grp")["IMD"]
    .agg(mediane="median", moyenne="mean", count="count")
    .reindex(_parti_order_by_imd)
    .reset_index()
)

fig_bar = go.Figure()
for _, row in _bar_df.iterrows():
    _col = _parti_grp_colors.get(str(row["parti_grp"]), "#95a5a6")
    fig_bar.add_trace(go.Bar(
        x=[row["parti_grp"]],
        y=[row["mediane"]],
        name=str(row["parti_grp"]),
        marker_color=_col,
        text=[f"n={int(row['count'])}  méd.={row['mediane']:.1f}"],
        textposition="outside",
        hovertemplate=(
            f"<b>{row['parti_grp']}</b><br>"
            f"Médiane : {row['mediane']:.1f}<br>"
            f"Moyenne : {row['moyenne']:.1f}<br>"
            f"N villes : {int(row['count'])}"
            "<extra></extra>"
        ),
    ))

fig_bar.update_layout(
    showlegend=False,
    plot_bgcolor="white",
    yaxis_title="IMD médian (/100)",
    xaxis_title="Parti du maire (municipales 2020)",
    margin=dict(l=10, r=10, t=30, b=10),
    height=420,
    yaxis=dict(range=[0, 105]),
)
st.plotly_chart(fig_bar, use_container_width=True)
st.caption(
    "**Figure 2.1.** Score IMD médian par parti politique de l'exécutif municipal "
    "(élections 2020). Les partis avec moins de "
    f"{min_villes_parti} agglomération(s) sont regroupés en 'Autres'. "
    "Triés par médiane IMD décroissante."
)

# ── 2.2 Box plot IMD par parti ─────────────────────────────────────────────────
st.markdown("#### 2.2. Distribution des Scores IMD par Parti")

_partis_box = [p for p in _parti_order_by_imd if p != "Autres"]
_box_df = merged_pol[merged_pol["parti_grp"].isin(_partis_box)].copy()

if len(_partis_box) >= 2 and not _box_df.empty:
    _box_df["parti_grp"] = pd.Categorical(_box_df["parti_grp"], categories=_partis_box, ordered=True)
    _hover_cols = ["city", "n_stations", "IMD"]
    if "couleur_municipale" in _box_df.columns:
        _hover_cols.append("couleur_municipale")

    fig_box = px.box(
        _box_df.sort_values("parti_grp"),
        x="parti_grp",
        y="IMD",
        color="parti_grp",
        color_discrete_map=_parti_grp_colors,
        points="all",
        hover_data=_hover_cols,
        labels={"parti_grp": "Parti du maire (2020)", "IMD": "Score IMD (/100)"},
        height=450,
    )
    if show_labels:
        for _, row in _box_df.iterrows():
            if pd.notna(row.get("parti_grp")) and pd.notna(row["IMD"]):
                fig_box.add_annotation(
                    x=row["parti_grp"], y=row["IMD"],
                    text=row["city"], showarrow=False,
                    font=dict(size=8, color="#555"), yshift=6,
                )
    fig_box.update_layout(
        plot_bgcolor="white", showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption(
        "**Figure 2.2.** Distribution des scores IMD par parti (hors 'Autres'). "
        "Chaque point représente une agglomération. "
        "Médiane (ligne centrale), Q1–Q3 (boîte), 1,5×IQR (moustaches)."
    )

# ── 2.3 Test de Kruskal-Wallis par parti ───────────────────────────────────────
_groups_kw = [
    merged_pol.loc[merged_pol["parti_grp"] == g, "IMD"].dropna().values
    for g in _partis_box
    if (merged_pol["parti_grp"] == g).sum() >= 2
]
_labels_kw = [g for g in _partis_box if (merged_pol["parti_grp"] == g).sum() >= 2]

with st.expander("Test de Kruskal-Wallis — Différence globale entre partis", expanded=True):
    if _SCIPY and len(_groups_kw) >= 2:
        try:
            _H, _p_kw = _kw(*_groups_kw)
            _df_kw    = len(_groups_kw) - 1
            _n_kw     = sum(len(g) for g in _groups_kw)
            _eta2     = (_H - _df_kw) / (_n_kw - _df_kw - 1) if _n_kw > _df_kw + 1 else float("nan")

            kw1, kw2, kw3, kw4 = st.columns(4)
            kw1.metric("Statistique H (K-W)", f"{_H:.3f}")
            kw2.metric("Degrés de liberté", f"{_df_kw}")
            kw3.metric(
                "p-valeur",
                f"{_p_kw:.4f}" if _p_kw >= 0.001 else "< 0,001",
                "sign. (p < 0,05)" if _p_kw < 0.05 else "non sign.",
            )
            kw4.metric("Taille d'effet η²", f"{_eta2:.3f}" if pd.notna(_eta2) else "—")

            _fmt_p = lambda p: f"{p:.4f}" if p >= 0.001 else "< 0,001"
            st.caption(
                f"**Tableau 2.1.** Kruskal-Wallis H ($k = {len(_groups_kw)}$ partis, "
                f"$n = {_n_kw}$ agglomérations). "
                f"$H({_df_kw}) = {_H:.3f}$, $p = {_fmt_p(_p_kw)}$, "
                f"$\\eta^2 = {_eta2:.3f}$ (faible < 0,06 / modéré < 0,14 / fort ≥ 0,14). "
                + ("**Différence statistiquement significative** entre partis au seuil α = 0,05."
                   if _p_kw < 0.05 else
                   "Absence de différence significative au seuil α = 0,05 — "
                   "le parti n'est pas un prédicteur robuste de l'IMD sur ce panel.")
            )

            if _p_kw < 0.05 and len(_groups_kw) >= 2:
                st.markdown("##### Comparaisons pairées post-hoc (Mann-Whitney U, correction Bonferroni)")
                _mw_rows = []
                _n_pairs = len(_groups_kw) * (len(_groups_kw) - 1) // 2
                for i, (g1, l1) in enumerate(zip(_groups_kw, _labels_kw)):
                    for g2, l2 in zip(_groups_kw[i + 1:], _labels_kw[i + 1:]):
                        try:
                            _U, _p_mw = _mwu(g1, g2, alternative="two-sided")
                            _p_corr   = min(1.0, _p_mw * _n_pairs)
                            _r_eff    = 1 - 2 * _U / (len(g1) * len(g2))
                            _mw_rows.append({
                                "Groupe 1": l1, "Groupe 2": l2,
                                "n₁": len(g1), "n₂": len(g2),
                                "Méd. G1": f"{float(np.median(g1)):.1f}",
                                "Méd. G2": f"{float(np.median(g2)):.1f}",
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
        st.info("scipy non installé — statistiques descriptives uniquement.")
        st.dataframe(_parti_tbl, use_container_width=True, hide_index=True)

# ── 2.4 Scatter IMD × Revenu, coloré par parti ────────────────────────────────
if has_revnu:
    _scat_df = merged_pol.dropna(subset=["revenu_median_uc", "IMD", "parti_grp"]).copy()
    if len(_scat_df) >= 5:
        st.markdown("#### 2.3. IMD × Revenu médian — Coloré par Parti")
        _hover_scat = ["city", "n_stations", "parti_maire"]
        if "couleur_municipale" in _scat_df.columns:
            _hover_scat.append("couleur_municipale")

        fig_scat = px.scatter(
            _scat_df,
            x="revenu_median_uc",
            y="IMD",
            color="parti_grp",
            color_discrete_map=_parti_grp_colors,
            category_orders={"parti_grp": _parti_order_by_imd},
            text="city" if show_labels else None,
            size="n_stations",
            size_max=22,
            hover_data=_hover_scat,
            labels={
                "revenu_median_uc": "Revenu médian/UC (€/an, INSEE Filosofi)",
                "IMD":              "Score IMD (/100)",
                "parti_grp":        "Parti",
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
            "**Figure 2.3.** Score IMD versus revenu médian/UC (INSEE Filosofi), "
            "coloré par parti de l'exécutif municipal. "
            "La taille des points est proportionnelle au nombre de stations dock-based. "
            "Les lignes en pointillés indiquent les médianes (revenu et IMD)."
        )

# ── Section 3 - IES × Parti ────────────────────────────────────────────────────
st.divider()
section(3, "Équité Sociale (IES) par Parti Politique")

st.markdown(r"""
L'IES mesure si l'agglomération investit **au-delà** de ce que son niveau économique
laisserait prévoir (OLS : IMD ~ revenu médian/UC). Un IES > 1 signale une volonté
proactive d'équité cyclable. La question est : cette volonté est-elle corrélée
à l'appartenance partisane ?
""")

if ies_col_ok and "IES" in merged.columns:
    _ies_pol = merged.dropna(subset=["IES", "parti_maire"]).copy()
    _ies_pol["parti_grp"] = _ies_pol["parti_maire"].apply(_grp_parti)

    # ── 3.1 Stacked bar quadrants par parti ────────────────────────────────────
    if has_revnu and "revenu_median_uc" in _ies_pol.columns:
        _med_rev = float(_ies_pol["revenu_median_uc"].median())
        _med_imd = float(merged["IMD"].median())

        def _quadrant(row: pd.Series) -> str:
            above_imd = row["IMD"] >= _med_imd
            above_rev = row.get("revenu_median_uc", float("nan")) >= _med_rev
            if not above_rev and above_imd:
                return "Mobilité Inclusive"
            if above_rev and above_imd:
                return "Excellence Consolidée"
            if not above_rev and not above_imd:
                return "Désert de Mobilité"
            return "Sous-Performance"

        _ies_pol["quadrant"] = _ies_pol.apply(_quadrant, axis=1)

        _quad_pol = (
            _ies_pol.groupby(["parti_grp", "quadrant"], observed=True)
            .size()
            .reset_index(name="n")
        )
        _quad_col_map = {
            "Mobilité Inclusive":    "#27ae60",
            "Excellence Consolidée": "#1A6FBF",
            "Désert de Mobilité":    "#e74c3c",
            "Sous-Performance":      "#e67e22",
        }
        fig_quad = px.bar(
            _quad_pol,
            x="parti_grp",
            y="n",
            color="quadrant",
            color_discrete_map=_quad_col_map,
            barmode="stack",
            category_orders={"parti_grp": _parti_order_by_imd},
            labels={
                "parti_grp": "Parti du maire",
                "n":         "Nombre d'agglomérations",
                "quadrant":  "Régime IES",
            },
            height=420,
        )
        fig_quad.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        st.plotly_chart(fig_quad, use_container_width=True)
        st.caption(
            "**Figure 3.1.** Répartition des régimes de justice cyclable (quadrants IES) "
            "par parti de l'exécutif municipal. 'Mobilité Inclusive' = revenu faible, "
            "IMD élevé (sur-investissement en faveur des populations précaires). "
            "Triés par médiane IMD décroissante."
        )

    # ── 3.2 Bar chart IES médian par parti ────────────────────────────────────
    _ies_bar = (
        _ies_pol.groupby("parti_grp")["IES"]
        .agg(mediane="median", count="count")
        .reindex(_parti_order_by_imd)
        .reset_index()
    )
    fig_ies_bar = go.Figure()
    for _, row in _ies_bar.iterrows():
        if pd.isna(row["mediane"]):
            continue
        _col = _parti_grp_colors.get(str(row["parti_grp"]), "#95a5a6")
        fig_ies_bar.add_trace(go.Bar(
            x=[row["parti_grp"]],
            y=[row["mediane"]],
            name=str(row["parti_grp"]),
            marker_color=_col,
            text=[f"n={int(row['count'])}"],
            textposition="outside",
            hovertemplate=(
                f"<b>{row['parti_grp']}</b><br>"
                f"IES médian : {row['mediane']:.3f}<br>"
                f"N villes : {int(row['count'])}"
                "<extra></extra>"
            ),
        ))
    fig_ies_bar.add_hline(
        y=1.0, line_dash="dash", line_color="#555", line_width=1.5,
        annotation_text="IES = 1 (neutralité)", annotation_position="right",
    )
    fig_ies_bar.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        yaxis_title="IES médian",
        xaxis_title="Parti du maire",
        margin=dict(l=10, r=10, t=30, b=10),
        height=380,
    )
    st.plotly_chart(fig_ies_bar, use_container_width=True)
    st.caption(
        "**Figure 3.2.** IES médian par parti de l'exécutif municipal. "
        "La ligne en tirets (IES = 1) marque la neutralité distributive. "
        "Au-dessus : l'agglomération sur-investit par rapport à son niveau de revenu prévisible."
    )

    # ── 3.3 Box IES par parti (partis ≥ 2 villes) ─────────────────────────────
    _ies_partis = [p for p in _parti_order_by_imd if (merged_pol["parti_grp"] == p).sum() >= 2 and p != "Autres"]
    _ies_box_df = _ies_pol[_ies_pol["parti_grp"].isin(_ies_partis)].copy()

    if len(_ies_partis) >= 2 and not _ies_box_df.empty:
        _ies_box_df["parti_grp"] = pd.Categorical(
            _ies_box_df["parti_grp"], categories=_ies_partis, ordered=True
        )
        fig_ies_box = px.box(
            _ies_box_df.sort_values("parti_grp"),
            x="parti_grp",
            y="IES",
            color="parti_grp",
            color_discrete_map=_parti_grp_colors,
            points="all",
            hover_data=["city", "IMD", "IES"],
            labels={"parti_grp": "Parti du maire", "IES": "Indice d'Équité Sociale (IES)"},
            height=380,
        )
        fig_ies_box.add_hline(
            y=1.0, line_dash="dash", line_color="#555", line_width=1.5,
            annotation_text="IES = 1", annotation_position="right",
        )
        fig_ies_box.update_layout(
            plot_bgcolor="white", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_ies_box, use_container_width=True)
        st.caption(
            "**Figure 3.3.** Distribution de l'IES par parti (≥ 2 agglomérations, 'Autres' exclus). "
            "La ligne en tirets marque la neutralité (IES = 1)."
        )
else:
    st.info(
        "L'IES ne peut pas être calculé : la colonne `revenu_median_uc` "
        "(INSEE Filosofi) n'est pas disponible dans ce dataset."
    )

# ── Section 4 - Vue agrégée par couleur politique ──────────────────────────────
st.divider()
section(4, "Vue Agrégée par Couleur Politique — Blocs Gauche / Centre / Droite")

st.markdown(r"""
En complément de l'analyse par parti, le regroupement en **quatre blocs politiques**
augmente les effectifs par groupe et améliore la robustesse des tests statistiques.
""")

if has_pol and not merged_couleur.empty:
    _mc = merged_couleur.copy()
    _mc["couleur_municipale"] = pd.Categorical(
        _mc["couleur_municipale"], categories=_COULEUR_ORDER, ordered=True
    )

    col_c1, col_c2 = st.columns([2, 1])

    with col_c1:
        _hov_c = ["city", "n_stations", "IMD"]
        if "parti_maire" in _mc.columns:
            _hov_c.append("parti_maire")
        fig_box_c = px.box(
            _mc.sort_values("couleur_municipale"),
            x="couleur_municipale",
            y="IMD",
            color="couleur_municipale",
            color_discrete_map=_COULEUR_COLORS,
            points="all",
            hover_data=_hov_c,
            labels={"couleur_municipale": "Couleur politique (2020)", "IMD": "Score IMD (/100)"},
            height=380,
        )
        fig_box_c.update_layout(
            plot_bgcolor="white", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_box_c, use_container_width=True)
        st.caption("**Figure 4.1.** IMD par bloc politique (vue agrégée).")

    with col_c2:
        _grps_c = [
            _mc.loc[_mc["couleur_municipale"] == g, "IMD"].dropna().values
            for g in _COULEUR_ORDER
            if (_mc["couleur_municipale"] == g).sum() >= 2
        ]
        _lbls_c = [g for g in _COULEUR_ORDER if (_mc["couleur_municipale"] == g).sum() >= 2]

        _couleur_tbl = (
            _mc.groupby("couleur_municipale", observed=True)["IMD"]
            .agg(n="count", mediane="median", moyenne="mean")
            .reset_index()
        )
        _couleur_tbl.columns = ["Couleur", "N", "Méd. IMD", "Moy. IMD"]
        _couleur_tbl["Méd. IMD"] = _couleur_tbl["Méd. IMD"].round(1)
        _couleur_tbl["Moy. IMD"] = _couleur_tbl["Moy. IMD"].round(1)
        st.dataframe(_couleur_tbl, use_container_width=True, hide_index=True)
        st.caption("**Tableau 4.1.**")

        if _SCIPY and len(_grps_c) >= 2:
            try:
                _Hc, _pc = _kw(*_grps_c)
                _df_c    = len(_grps_c) - 1
                _n_c     = sum(len(g) for g in _grps_c)
                _eta2_c  = (_Hc - _df_c) / (_n_c - _df_c - 1) if _n_c > _df_c + 1 else float("nan")
                st.markdown("**Kruskal-Wallis — Blocs**")
                r1, r2 = st.columns(2)
                r1.metric("H", f"{_Hc:.3f}")
                r2.metric("p", f"{_pc:.4f}" if _pc >= 0.001 else "< 0,001",
                          "sign." if _pc < 0.05 else "n.s.")
                st.caption(
                    f"η² = {_eta2_c:.3f}, k = {len(_grps_c)} blocs, n = {_n_c}. "
                    + ("**Diff. significative** (α = 0,05)."
                       if _pc < 0.05 else "Non significatif (α = 0,05).")
                )
            except Exception:
                pass

# ── Section 5 - Dimension régionale ───────────────────────────────────────────
st.divider()
section(5, "Dimension Régionale — Exécutifs Régionaux et IMD")

st.markdown(r"""
Les régions co-financent les plans vélo via les SRADDET. La couleur politique
régionale pourrait amplifier ou atténuer l'effet municipal sur l'IMD.
""")

if has_reg:
    _reg_pol = merged.dropna(subset=["couleur_regionale"]).copy()
    _reg_pol["couleur_regionale"] = pd.Categorical(
        _reg_pol["couleur_regionale"], categories=_COULEUR_ORDER, ordered=True
    )

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        _hov_r = ["city", "region", "IMD"]
        if "couleur_municipale" in _reg_pol.columns:
            _hov_r.append("couleur_municipale")
        if "parti_maire" in _reg_pol.columns:
            _hov_r.append("parti_maire")
        fig_reg = px.box(
            _reg_pol.sort_values("couleur_regionale"),
            x="couleur_regionale",
            y="IMD",
            color="couleur_regionale",
            color_discrete_map=_COULEUR_COLORS,
            points="all",
            hover_data=_hov_r,
            labels={"couleur_regionale": "Couleur régionale (2021)", "IMD": "Score IMD (/100)"},
            height=380,
        )
        fig_reg.update_layout(
            plot_bgcolor="white", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_reg, use_container_width=True)
        st.caption(
            "**Figure 5.1.** IMD par couleur régionale. "
            "Note : la quasi-totalité des régions est LR depuis 2021, "
            "ce qui réduit fortement la variabilité inter-groupes."
        )

    with col_r2:
        _reg_tbl = (
            _reg_pol.groupby("couleur_regionale", observed=True)["IMD"]
            .agg(n="count", mediane="median")
            .reset_index()
        )
        _reg_tbl.columns = ["Couleur régionale", "N villes", "IMD médian"]
        _reg_tbl["IMD médian"] = _reg_tbl["IMD médian"].round(1)
        st.dataframe(_reg_tbl, use_container_width=True, hide_index=True)
        st.caption("**Tableau 5.1.** IMD par couleur régionale.")

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
            if not _config_imd.empty:
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
                    "**Figure 5.2.** IMD médian par configuration politique "
                    "(couleur municipale / couleur régionale). "
                    "Configurations ≥ 2 agglomérations uniquement. "
                    "Résultats hautement indicatifs (faibles effectifs)."
                )

# ── Section 6 - Tableau de classement ─────────────────────────────────────────
st.divider()
section(6, "Tableau de Classement — Agglomérations, Partis et Scores")

_disp_cols = ["city", "parti_maire", "couleur_municipale", "region",
              "couleur_regionale", "maire", "n_stations", "IMD"]
if ies_col_ok and "IES" in merged.columns:
    _disp_cols.append("IES")
_disp_available = [c for c in _disp_cols if c in merged.columns]

_disp = merged[_disp_available].dropna(subset=["parti_maire"]).copy()
_disp = _disp.sort_values("IMD", ascending=False)

_rename = {
    "city":               "Agglomération",
    "parti_maire":        "Parti",
    "couleur_municipale": "Bloc politique",
    "region":             "Région",
    "couleur_regionale":  "Bloc régional",
    "maire":              "Maire",
    "n_stations":         "Stations",
    "IMD":                "IMD (/100)",
    "IES":                "IES",
}
_disp = _disp.rename(columns={k: v for k, v in _rename.items() if k in _disp.columns})
if "IMD (/100)" in _disp.columns:
    _disp["IMD (/100)"] = _disp["IMD (/100)"].round(1)
if "IES" in _disp.columns:
    _disp["IES"] = _disp["IES"].round(3)

col_cfg_6: dict = {
    "IMD (/100)": st.column_config.ProgressColumn(
        "IMD (/100)", min_value=0, max_value=100, format="%.1f"
    ),
}
if "IES" in _disp.columns:
    col_cfg_6["IES"] = st.column_config.NumberColumn("IES", format="%.3f")

st.dataframe(_disp, use_container_width=True, hide_index=True, column_config=col_cfg_6)
st.caption(
    "**Tableau 6.1.** Classement des agglomérations par IMD (décroissant), "
    "avec le parti du maire et le bloc politique. "
    "Source politique : élections municipales 2020 / régionales 2021 "
    "(Ministère de l'Intérieur). Source IMD/IES : Gold Standard GBFS — R. Fossé & G. Pallares, 2025–2026."
)

# ── Section 7 - Discussion ─────────────────────────────────────────────────────
st.divider()
section(7, "Discussion — Portée et Limites de l'Analyse Politique")

st.markdown(r"""
#### 7.1. Principaux Résultats Observés

L'analyse par parti révèle des différences descriptives, mais plusieurs nuances
s'imposent :

- **EELV vs. PS :** au sein du bloc de gauche, les maires EELV gouvernent des villes
  (Grenoble, Lyon, Bordeaux, Strasbourg, Besançon, Poitiers) aux scores IMD parmi les
  plus élevés du panel, cohérents avec des programmes vélo ambitieux. Les villes PS
  présentent une dispersion plus forte — de Paris et Nantes à des villes moyennes
  moins bien dotées en infrastructure cyclable.

- **LR hétérogène :** le groupe LR (le plus nombreux du panel) est le plus dispersé,
  allant de Nice et Reims à des villes à faible score, ce qui rend toute généralisation
  fragile.

- **Effectifs très limités :** RN (Perpignan), PCF (Bourges), MoDem et Horizons
  comptent une seule agglomération — aucune conclusion statistique n'est possible.

#### 7.2. Interprétation Contextuelle

La corrélation EELV–IMD élevé reflète davantage une **sélection électorale** qu'un
effet causal direct du programme partisan :

- Les électeurs des grandes métropoles pro-vélo ont souvent porté au pouvoir des
  listes écolos parce que l'environnement urbain leur était déjà favorable.
- Les réseaux VLS existaient avant les élections 2020 dans la majorité des villes :
  Paris (2007), Lyon (2005), Strasbourg (2013), Bordeaux (2010).
- La mobilité douce est devenue un **marqueur identitaire** de l'écologie politique,
  renforçant l'apparence d'une corrélation que les données granulaires nuancent.

#### 7.3. Perspectives de Recherche

1. **Contrôler les covariables :** taille de ville, densité, topographie, date de
   création du réseau VLS (antérieure aux élections dans la plupart des cas).
2. **Analyse différence-en-différences :** comparer l'évolution de l'IMD avant/après
   un changement de majorité municipale (données longitudinales nécessaires).
3. **Budgets mobilité :** délibérations de conseils municipaux et plans vélo (PDME)
   permettraient de mesurer directement l'intention politique.
4. **Analyse longitudinale :** suivre l'évolution des scores sur plusieurs mandats
   (2014 → 2020 → 2026).

> **Conclusion provisoire :** Le parti politique est un **signal corrélé** mais non un
> déterminant isolé de la qualité VLS. L'analyse de l'IES confirme que la justice
> distributive transcende les clivages partisans : des agglomérations LR ou DVG
> offrent une mobilité inclusive, tandis que certaines villes EELV ou PS présentent
> des sous-performances relatives à leur niveau de revenu.
""")
