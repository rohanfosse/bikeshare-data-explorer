"""
10_Politique.py - Gouvernance Politique et Mobilité Douce.
Analyse multidimensionnelle de la relation entre le parti politique de l'exécutif
municipal et les indicateurs Gold Standard : IMD, IES, FUB, EMP, Gini.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.stats import kruskalwallis as _kw, mannwhitneyu as _mwu, spearmanr as _spr
    _SCIPY = True
except ImportError:
    _SCIPY = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    compute_imd_cities,
    load_city_mobility,
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

# ── Palettes ───────────────────────────────────────────────────────────────────
_COULEUR_COLORS: dict[str, str] = {
    "Gauche":         "#C0392B",
    "Centre":         "#D4AC0D",
    "Droite":         "#1565C0",
    "Extrême droite": "#2C3E50",
}
_COULEUR_ORDER = ["Gauche", "Centre", "Droite", "Extrême droite"]

_PARTI_COLORS: dict[str, str] = {
    "EELV":     "#27ae60",
    "PS":       "#C0392B",
    "PCF":      "#8B0000",
    "DVG":      "#E8A090",
    "MoDem":    "#F1C40F",
    "Horizons": "#E67E22",
    "UDI":      "#D4AC0D",
    "Centre":   "#BFA04A",
    "LR":       "#1565C0",
    "RN":       "#2C3E50",
    "Corse":    "#7F8C8D",
}

# ── Variables Y disponibles ────────────────────────────────────────────────────
_Y_META: dict[str, dict] = {
    "IMD":               {"label": "IMD (/100)",             "fmt": ".1f", "title": "Indice de Mobilité Douce"},
    "IES":               {"label": "IES",                    "fmt": ".3f", "title": "Indice d'Équité Sociale"},
    "fub_score_2023":    {"label": "FUB score (/6)",         "fmt": ".2f", "title": "FUB Baromètre 2023 (subjectif)"},
    "emp_part_velo_2019":{"label": "Part modale vélo (%)",   "fmt": ".1f", "title": "EMP 2019 - Part modale cycliste"},
    "part_velo_travail": {"label": "% vélo domicile-travail","fmt": ".2f", "title": "RP 2020 - Navette vélo"},
    "gini_revenu":       {"label": "Gini (inégalités)",      "fmt": ".3f", "title": "Gini - Inégalités de revenu"},
}

# ── Chargement des données ─────────────────────────────────────────────────────
df     = load_stations()
imd_df = compute_imd_cities(df)
pol_df = load_political_data()
mob_df = load_city_mobility()

# ── Fusion principale ──────────────────────────────────────────────────────────
if not pol_df.empty and not imd_df.empty:
    merged = imd_df.merge(pol_df, on="city", how="left")
else:
    merged = imd_df.copy()

# Fusion données de mobilité externe (FUB, EMP…)
if not mob_df.empty:
    _mob_cols = [c for c in mob_df.columns if c != "city"]
    merged = merged.merge(mob_df[["city"] + _mob_cols], on="city", how="left")

has_pol   = "couleur_municipale" in merged.columns and merged["couleur_municipale"].notna().sum() >= 3
has_parti = "parti_maire"        in merged.columns and merged["parti_maire"].notna().sum() >= 3
has_reg   = "couleur_regionale"  in merged.columns and merged["couleur_regionale"].notna().sum() >= 3
has_revnu = "revenu_median_uc"   in merged.columns and merged["revenu_median_uc"].notna().sum() >= 5

# ── Calcul IES ─────────────────────────────────────────────────────────────────
ies_col_ok = False
if has_revnu:
    _tmp = merged.dropna(subset=["revenu_median_uc", "IMD"]).copy()
    if len(_tmp) >= 5:
        _c = np.polyfit(_tmp["revenu_median_uc"].values, _tmp["IMD"].values, 1)
        _tmp["IMD_hat"] = np.polyval(_c, _tmp["revenu_median_uc"].values).clip(min=1.0)
        _tmp["IES"]     = (_tmp["IMD"] / _tmp["IMD_hat"]).round(3)
        merged = merged.merge(_tmp[["city", "IMD_hat", "IES"]], on="city", how="left")
        ies_col_ok = True

# ── Stats abstract (avant filtre) ──────────────────────────────────────────────
_n_pol    = int(merged["parti_maire"].notna().sum()) if has_parti else 0
_n_total  = len(merged)
_n_partis = int(merged["parti_maire"].nunique()) if has_parti else 0
if has_parti and _n_pol >= 3:
    _top_parti = str(merged["parti_maire"].value_counts().index[0])
    _top_n     = int(merged["parti_maire"].value_counts().iloc[0])
else:
    _top_parti, _top_n = "-", 0

# ── Titre + résumé ─────────────────────────────────────────────────────────────
st.title("Gouvernance Politique et Mobilité Douce")
st.caption(
    "Axe transversal : le parti politique des exécutifs locaux prédit-il "
    "IMD, IES, score FUB et part modale cycliste ?"
)
abstract_box(
    "<b>Problématique :</b> La décision d'investir dans la micromobilité partagée "
    "est-elle conditionnée par l'appartenance partisane de l'exécutif municipal ? "
    "Cette page croise le <b>parti du maire</b> (élections 2020) avec six indicateurs : "
    "l'<b>IMD</b> (Infrastructure, Multimodalité, Sécurité, Topographie), "
    "l'<b>IES</b> (équité distributive), le <b>FUB Baromètre 2023</b> (satisfaction subjective), "
    "la <b>part modale EMP 2019</b>, la <b>part vélo domicile-travail</b> et le <b>Gini</b> "
    "(inégalités de revenu). L'analyse est exploratoire - faibles effectifs, "
    "absence de contrôle causal. L'année de création du réseau VLS permet de distinguer "
    "les villes pionnières (héritage) des villes récentes (décision politique directe).",
    findings=[
        (str(_n_pol),    "agglomérations"),
        (str(_n_partis), "partis représentés"),
        (_top_parti,     f"parti le plus fréquent (n={_top_n})"),
    ],
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres")
    min_stations = st.number_input(
        "Seuil min. stations (IMD)", min_value=1, max_value=200, value=10,
    )
    _y_opts = {k: v["title"] for k, v in _Y_META.items()
               if k == "IMD" or (k in merged.columns and merged[k].notna().sum() >= 5)}
    y_col = st.selectbox(
        "Variable d'analyse (Y)",
        options=list(_y_opts.keys()),
        format_func=lambda k: _y_opts[k],
        help="Toutes les visualisations de la section 2 s'adaptent à cette variable.",
    )
    show_labels    = st.checkbox("Afficher les étiquettes de villes", value=False)
    min_villes_parti = st.slider("Min. villes par parti (regroupement)", 1, 5, 2)
    exclude_idf    = st.checkbox(
        "Exclure Île-de-France",
        value=False,
        help="Paris, Versailles, Marne-la-Vallée, Cergy - géants démographiques atypiques.",
    )

# ── Filtre ─────────────────────────────────────────────────────────────────────
merged = merged[merged["n_stations"] >= min_stations].reset_index(drop=True)
if exclude_idf:
    merged = merged[merged.get("region", pd.Series(dtype=str)) != "Île-de-France"].reset_index(drop=True)

has_pol   = "couleur_municipale" in merged.columns and merged["couleur_municipale"].notna().sum() >= 3
has_parti = "parti_maire"        in merged.columns and merged["parti_maire"].notna().sum() >= 3
has_reg   = "couleur_regionale"  in merged.columns and merged["couleur_regionale"].notna().sum() >= 3

if not has_pol and not has_parti:
    st.warning("Données politiques non disponibles - vérifiez `data/external/politique/political_data.csv`.")
    st.stop()

# ── Regroupement des petits partis ────────────────────────────────────────────
merged_pol    = merged.dropna(subset=["parti_maire"]).copy()
_pcounts      = merged_pol["parti_maire"].value_counts()
_partis_maj   = _pcounts[_pcounts >= min_villes_parti].index.tolist()


def _grp_parti(p: str) -> str:
    return p if p in _partis_maj else "Autres"


merged_pol["parti_grp"] = merged_pol["parti_maire"].apply(_grp_parti)

_parti_grp_colors: dict[str, str] = {p: _PARTI_COLORS.get(p, "#95a5a6") for p in _partis_maj}
_parti_grp_colors["Autres"] = "#95a5a6"

_parti_order_by_imd: list[str] = (
    merged_pol[merged_pol["parti_grp"] != "Autres"]
    .groupby("parti_grp")["IMD"].median()
    .sort_values(ascending=False)
    .index.tolist()
)
if (merged_pol["parti_grp"] == "Autres").any():
    _parti_order_by_imd.append("Autres")

# Y metadata actif
_ym       = _Y_META.get(y_col, _Y_META["IMD"])
_y_label  = _ym["label"]
_y_fmt    = _ym["fmt"]
_y_avail  = y_col in merged_pol.columns and merged_pol[y_col].notna().sum() >= 5

# ── KPIs principaux ────────────────────────────────────────────────────────────
merged_couleur = merged.dropna(subset=["couleur_municipale"]) if has_pol else pd.DataFrame()
kpi_cols = st.columns(4)
for col_w, grp_label in zip(kpi_cols, _COULEUR_ORDER):
    if not merged_couleur.empty:
        s = merged_couleur.loc[merged_couleur["couleur_municipale"] == grp_label, "IMD"]
        col_w.metric(grp_label, f"{s.median():.1f} / 100" if len(s) > 0 else "-",
                     f"n = {len(s)} villes" if len(s) > 0 else "0 ville")
    else:
        col_w.metric(grp_label, "-")

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 - Cadre méthodologique
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(1, "Cadre Méthodologique - Sources, Indicateurs et Hypothèses")

col_meth, col_src = st.columns([3, 2])
with col_meth:
    st.markdown(r"""
#### 1.1. Données Politiques et Historiques

Les données politiques proviennent des **résultats officiels des élections municipales
2020** et régionales 2021. Deux variables historiques enrichissent l'analyse :

| Variable | Description |
|:--- |:--- |
| `annee_vls` | Année de création du réseau VLS dans la ville |
| `couleur_precedente` | Couleur politique du mandat précédent (2014–2020) |

Le croisement `annee_vls × parti_maire` permet de distinguer les villes qui ont
**hérité** d'un réseau créé sous un mandat antérieur de celles qui l'ont **initié**
durant le mandat actuel.

#### 1.2. Indicateurs Analysés

| Code | Source | Description |
|:--- |:--- |:--- |
| **IMD** | Gold Standard | Composite infra + multi + sécurité + topo (/100) |
| **IES** | Calculé | IMD / IMD_hat(revenu) - équité distributive |
| **FUB** | FUB 2023 | Satisfaction subjective des cyclistes (/6) |
| **EMP** | EMP 2019 | Part modale vélo (%) |
| **Gini** | INSEE Filosofi | Inégalités de revenu (0→1) |

#### 1.3. Hypothèses

**H₁ :** Les villes EELV affichent un IMD et FUB significativement supérieurs.

**H₂ :** L'effet partisan est conditionné par l'âge du réseau : les villes pionnières
(annee_vls < 2015) ont un IMD élevé indépendamment de la couleur politique actuelle.

**H₃ :** L'IES est orthogonal au parti - la justice distributive transcende le clivage.

> **Avertissement :** Panel ≤ 60 agglomérations, effectifs faibles par parti.
> Corrélation ≠ causalité.
""")

with col_src:
    _parti_tbl = (
        merged_pol.groupby("parti_grp")
        .agg(n_villes=("city", "count"), IMD_med=("IMD", "median"), IMD_moy=("IMD", "mean"))
        .reset_index().sort_values("IMD_med", ascending=False)
    )
    _parti_tbl.columns = ["Parti", "N villes", "IMD médian", "IMD moyen"]
    _parti_tbl[["IMD médian", "IMD moyen"]] = _parti_tbl[["IMD médian", "IMD moyen"]].round(1)
    st.markdown("#### IMD par parti")
    st.dataframe(
        _parti_tbl, use_container_width=True, hide_index=True,
        column_config={"IMD médian": st.column_config.ProgressColumn(
            "IMD médian", min_value=0, max_value=100, format="%.1f"
        )},
    )
    st.caption("**Tableau 1.1.** IMD par parti (trié par médiane).")

    # Âge des réseaux si disponible
    if "annee_vls" in merged_pol.columns:
        merged_pol["age_vls_2020"] = 2020 - pd.to_numeric(merged_pol["annee_vls"], errors="coerce")
        _age_era = merged_pol.groupby("parti_grp")["age_vls_2020"].median().reset_index()
        _age_era.columns = ["Parti", "Âge médian du réseau en 2020"]
        _age_era["Âge médian du réseau en 2020"] = _age_era["Âge médian du réseau en 2020"].round(1)
        st.markdown("#### Âge médian du réseau VLS")
        st.dataframe(_age_era.sort_values("Âge médian du réseau en 2020", ascending=False),
                     use_container_width=True, hide_index=True)
        st.caption("**Tableau 1.2.** Âge du réseau (années) en 2020 par parti.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 - Analyse principale (variable Y sélectable)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
_sec2_title = f"Analyse par Parti - {_ym['title']}"
section(2, _sec2_title)

if not _y_avail:
    st.info(f"La variable **{_y_label}** n'est pas disponible pour ce panel. "
            f"Sélectionnez une autre variable Y dans la barre latérale.")
else:
    _ana_df = merged_pol.dropna(subset=[y_col, "parti_grp"]).copy()

    # ── 2.1 Bar chart Y médian par parti ──────────────────────────────────────
    st.markdown(f"#### 2.1. {_ym['title']} médian par Parti")
    _bar_df = (
        _ana_df.groupby("parti_grp")[y_col]
        .agg(mediane="median", moyenne="mean", count="count")
        .reindex(_parti_order_by_imd).reset_index()
    )
    fig_bar = go.Figure()
    for _, row in _bar_df.iterrows():
        if pd.isna(row["mediane"]):
            continue
        _col = _parti_grp_colors.get(str(row["parti_grp"]), "#95a5a6")
        fig_bar.add_trace(go.Bar(
            x=[row["parti_grp"]], y=[row["mediane"]],
            name=str(row["parti_grp"]), marker_color=_col,
            text=[f"n={int(row['count'])}  méd.={row['mediane']:{_y_fmt}}"],
            textposition="outside",
            hovertemplate=(
                f"<b>{row['parti_grp']}</b><br>"
                f"Médiane : {row['mediane']:{_y_fmt}}<br>"
                f"Moyenne : {row['moyenne']:{_y_fmt}}<br>"
                f"N villes : {int(row['count'])}<extra></extra>"
            ),
        ))
    _y_max = float(_bar_df["mediane"].max()) * 1.2 if not _bar_df["mediane"].isna().all() else 1
    fig_bar.update_layout(
        showlegend=False, plot_bgcolor="white",
        yaxis_title=_y_label, xaxis_title="Parti du maire",
        margin=dict(l=10, r=10, t=30, b=10), height=400,
        yaxis=dict(range=[0, _y_max]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        f"**Figure 2.1.** {_ym['title']} médian par parti (élections 2020). "
        f"Partis < {min_villes_parti} villes regroupés en 'Autres'. "
        "Trié par médiane IMD décroissante."
    )

    # ── 2.2 Box + KW ──────────────────────────────────────────────────────────
    st.markdown(f"#### 2.2. Distribution et Test de Kruskal-Wallis")
    _partis_box = [p for p in _parti_order_by_imd if p != "Autres"]
    _box_df = _ana_df[_ana_df["parti_grp"].isin(_partis_box)].copy()

    if len(_partis_box) >= 2 and not _box_df.empty:
        _box_df["parti_grp"] = pd.Categorical(_box_df["parti_grp"], categories=_partis_box, ordered=True)
        _hov = ["city", "n_stations", y_col]
        if "couleur_municipale" in _box_df.columns:
            _hov.append("couleur_municipale")
        fig_box = px.box(
            _box_df.sort_values("parti_grp"), x="parti_grp", y=y_col,
            color="parti_grp", color_discrete_map=_parti_grp_colors,
            points="all", hover_data=_hov,
            labels={"parti_grp": "Parti", y_col: _y_label}, height=420,
        )
        if show_labels:
            for _, row in _box_df.iterrows():
                if pd.notna(row.get("parti_grp")) and pd.notna(row[y_col]):
                    fig_box.add_annotation(
                        x=row["parti_grp"], y=row[y_col], text=row["city"],
                        showarrow=False, font=dict(size=8, color="#555"), yshift=6,
                    )
        if y_col == "IES":
            fig_box.add_hline(y=1.0, line_dash="dash", line_color="#555",
                              annotation_text="IES = 1", annotation_position="right")
        fig_box.update_layout(plot_bgcolor="white", showlegend=False,
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

    # KW test
    _groups_kw = [
        _ana_df.loc[_ana_df["parti_grp"] == g, y_col].dropna().values
        for g in _partis_box if (_ana_df["parti_grp"] == g).sum() >= 2
    ]
    _labels_kw = [g for g in _partis_box if (_ana_df["parti_grp"] == g).sum() >= 2]

    with st.expander("Test de Kruskal-Wallis - significativité globale", expanded=True):
        if _SCIPY and len(_groups_kw) >= 2:
            try:
                _H, _p_kw = _kw(*_groups_kw)
                _df_kw    = len(_groups_kw) - 1
                _n_kw     = sum(len(g) for g in _groups_kw)
                _eta2     = (_H - _df_kw) / (_n_kw - _df_kw - 1) if _n_kw > _df_kw + 1 else float("nan")
                kw1, kw2, kw3, kw4 = st.columns(4)
                kw1.metric("H (K-W)", f"{_H:.3f}")
                kw2.metric("ddl", f"{_df_kw}")
                _fmt_p = lambda p: f"{p:.4f}" if p >= 0.001 else "< 0,001"
                kw3.metric("p-valeur", _fmt_p(_p_kw),
                           "sign." if _p_kw < 0.05 else "non sign.")
                kw4.metric("η²", f"{_eta2:.3f}" if pd.notna(_eta2) else "-")
                st.caption(
                    f"$k = {len(_groups_kw)}$ partis, $n = {_n_kw}$ agglomérations. "
                    f"$H({_df_kw}) = {_H:.3f}$, $p = {_fmt_p(_p_kw)}$, $\\eta^2 = {_eta2:.3f}$. "
                    + ("**Différence significative** entre partis (α = 0,05)."
                       if _p_kw < 0.05 else
                       "Absence de différence significative (α = 0,05).")
                )
                if _p_kw < 0.05 and len(_groups_kw) >= 2:
                    st.markdown("##### Post-hoc Mann-Whitney U (Bonferroni)")
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
                                    f"Méd. {_y_label} G1": f"{float(np.median(g1)):{_y_fmt}}",
                                    f"Méd. {_y_label} G2": f"{float(np.median(g2)):{_y_fmt}}",
                                    "p (brut)": _fmt_p(_p_mw),
                                    "p (Bonf.)": _fmt_p(_p_corr),
                                    "Sig.": "**" if _p_corr < 0.01 else ("*" if _p_corr < 0.05 else "n.s."),
                                    "r": f"{_r_eff:.3f}",
                                })
                            except Exception:
                                pass
                    if _mw_rows:
                        st.dataframe(pd.DataFrame(_mw_rows), use_container_width=True, hide_index=True)
            except Exception as e:
                st.info(f"Test non disponible : {e}")
        else:
            st.info("scipy non installé - statistiques descriptives uniquement.")

    # ── 2.3 Scatter Y × revenu ────────────────────────────────────────────────
    if has_revnu:
        _scat_df = _ana_df.dropna(subset=["revenu_median_uc", y_col]).copy()
        if len(_scat_df) >= 5:
            st.markdown(f"#### 2.3. {_ym['title']} × Revenu médian - Coloré par Parti")
            _hov_s = ["city", "n_stations", "parti_maire"]
            if "couleur_municipale" in _scat_df.columns:
                _hov_s.append("couleur_municipale")
            fig_scat = px.scatter(
                _scat_df, x="revenu_median_uc", y=y_col,
                color="parti_grp", color_discrete_map=_parti_grp_colors,
                category_orders={"parti_grp": _parti_order_by_imd},
                text="city" if show_labels else None,
                size="n_stations", size_max=22,
                hover_data=_hov_s,
                labels={"revenu_median_uc": "Revenu médian/UC (€/an)",
                        y_col: _y_label, "parti_grp": "Parti"},
                height=480,
            )
            fig_scat.add_vline(
                x=float(_scat_df["revenu_median_uc"].median()),
                line_dash="dot", line_color="#999", opacity=0.5,
                annotation_text="Médiane revenu", annotation_position="top",
            )
            fig_scat.add_hline(
                y=float(_scat_df[y_col].median()),
                line_dash="dot", line_color="#999", opacity=0.5,
                annotation_text=f"Médiane {_y_label}", annotation_position="right",
            )
            fig_scat.update_layout(
                plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            st.caption(
                f"**Figure 2.3.** {_ym['title']} versus revenu médian/UC. "
                "Chaque point = une agglomération (taille ∝ n° stations)."
            )

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 - Composantes IMD par Parti (Radar)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(3, "Décomposition IMD par Parti - Radar des Composantes")

_comp_cols  = ["S_securite", "I_infra", "M_multi", "T_topo"]
_comp_names = ["Sécurité (S)", "Infrastructure (I)", "Multimodalité (M)", "Topographie (T)"]
_comp_ok    = all(c in merged_pol.columns for c in _comp_cols)

if _comp_ok:
    col_radar, col_bar_comp = st.columns([1, 1])

    # Valeurs moyennes par parti (normalisées 0→1 pour le radar)
    _radar_df = (
        merged_pol[merged_pol["parti_grp"] != "Autres"]
        .groupby("parti_grp")[_comp_cols]
        .mean()
        .reindex([p for p in _parti_order_by_imd if p != "Autres"])
    )

    with col_radar:
        st.markdown("#### Radar des composantes par Parti")
        fig_radar = go.Figure()
        _theta = _comp_names + [_comp_names[0]]
        for parti, row in _radar_df.iterrows():
            _vals = row[_comp_cols].tolist() + [row[_comp_cols[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=_vals, theta=_theta,
                fill="toself", name=str(parti),
                line_color=_parti_grp_colors.get(str(parti), "#999"),
                opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                       tickformat=".2f", tickfont_size=9)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            margin=dict(l=30, r=30, t=30, b=60),
            height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "**Figure 3.1.** Composantes IMD moyennes (normalisées 0–1) par parti. "
            "Un profil étendu sur tous les axes signifie une excellence globale."
        )

    with col_bar_comp:
        st.markdown("#### Décomposition par Composante")
        # Heatmap partis × composantes
        _heat = _radar_df.copy()
        _heat.columns = _comp_names
        _heat.index.name = "Parti"
        _heat_reset = _heat.reset_index().melt(id_vars="Parti", var_name="Composante", value_name="Score moyen")

        fig_heat = px.bar(
            _heat_reset,
            x="Composante", y="Score moyen",
            color="Parti", color_discrete_map=_parti_grp_colors,
            barmode="group",
            labels={"Score moyen": "Score normalisé (0–1)"},
            height=380,
        )
        fig_heat.update_layout(
            plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "**Figure 3.2.** Scores moyens par composante IMD et par parti. "
            "Permet d'identifier sur quelle dimension les partis se distinguent."
        )

    # Tableau synthétique composantes
    with st.expander("Tableau complet des composantes par parti"):
        _comp_tbl = _radar_df.copy()
        _comp_tbl.columns = _comp_names
        _comp_tbl["IMD médian"] = (
            merged_pol[merged_pol["parti_grp"] != "Autres"]
            .groupby("parti_grp")["IMD"].median()
            .reindex(_comp_tbl.index)
        )
        _comp_tbl = _comp_tbl.round(3).reset_index()
        _comp_tbl.columns = ["Parti"] + _comp_names + ["IMD médian"]
        st.dataframe(_comp_tbl, use_container_width=True, hide_index=True)
        st.caption("**Tableau 3.1.** Composantes IMD moyennes (0–1) et IMD médian par parti.")
else:
    st.info("Les colonnes de composantes IMD (S_securite, I_infra, M_multi, T_topo) "
            "ne sont pas disponibles dans ce panel.")

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 - Équité Sociale (IES) par Parti
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(4, "Équité Sociale (IES) par Parti")

st.markdown(r"""
L'IES = IMD / IMD_hat(revenu). Un IES > 1 signale un sur-investissement
en mobilité douce relativement au niveau de revenu prévisible.
""")

if ies_col_ok and "IES" in merged.columns:
    _ies_pol = merged.dropna(subset=["IES", "parti_maire"]).copy()
    _ies_pol["parti_grp"] = _ies_pol["parti_maire"].apply(_grp_parti)

    col_i1, col_i2 = st.columns(2)

    # IES médian par parti
    with col_i1:
        _ies_bar = (
            _ies_pol.groupby("parti_grp")["IES"]
            .agg(mediane="median", count="count")
            .reindex(_parti_order_by_imd).reset_index()
        )
        fig_ies_bar = go.Figure()
        for _, row in _ies_bar.iterrows():
            if pd.isna(row["mediane"]):
                continue
            fig_ies_bar.add_trace(go.Bar(
                x=[row["parti_grp"]], y=[row["mediane"]],
                marker_color=_parti_grp_colors.get(str(row["parti_grp"]), "#95a5a6"),
                text=[f"n={int(row['count'])}"], textposition="outside",
                hovertemplate=f"<b>{row['parti_grp']}</b><br>IES médian : {row['mediane']:.3f}<extra></extra>",
            ))
        fig_ies_bar.add_hline(y=1.0, line_dash="dash", line_color="#555",
                              annotation_text="IES = 1 (neutre)", annotation_position="right")
        fig_ies_bar.update_layout(
            showlegend=False, plot_bgcolor="white",
            yaxis_title="IES médian", xaxis_title="Parti",
            margin=dict(l=10, r=10, t=20, b=10), height=360,
        )
        st.plotly_chart(fig_ies_bar, use_container_width=True)
        st.caption("**Figure 4.1.** IES médian par parti. Au-dessus de 1 : sur-investissement équitable.")

    # Quadrants par parti
    with col_i2:
        if has_revnu and "revenu_median_uc" in _ies_pol.columns:
            _med_rev = float(_ies_pol["revenu_median_uc"].median())
            _med_imd = float(merged["IMD"].median())

            def _quad(row: pd.Series) -> str:
                ai = row["IMD"] >= _med_imd
                ar = row.get("revenu_median_uc", float("nan")) >= _med_rev
                if not ar and ai:  return "Mobilité Inclusive"
                if ar and ai:      return "Excellence Consolidée"
                if not ar and not ai: return "Désert de Mobilité"
                return "Sous-Performance"

            _ies_pol["quadrant"] = _ies_pol.apply(_quad, axis=1)
            _quad_pol = (
                _ies_pol.groupby(["parti_grp", "quadrant"], observed=True)
                .size().reset_index(name="n")
            )
            _quad_col_map = {
                "Mobilité Inclusive":    "#27ae60",
                "Excellence Consolidée": "#1A6FBF",
                "Désert de Mobilité":    "#e74c3c",
                "Sous-Performance":      "#e67e22",
            }
            fig_quad = px.bar(
                _quad_pol, x="parti_grp", y="n", color="quadrant",
                color_discrete_map=_quad_col_map, barmode="stack",
                category_orders={"parti_grp": _parti_order_by_imd},
                labels={"parti_grp": "Parti", "n": "N agglomérations", "quadrant": "Régime IES"},
                height=360,
            )
            fig_quad.update_layout(
                plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            st.plotly_chart(fig_quad, use_container_width=True)
            st.caption("**Figure 4.2.** Régimes IES par parti (quadrants).")
else:
    st.info("IES non disponible (revenu_median_uc absent).")

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 - Validation Externe : FUB et EMP
# ══════════════════════════════════════════════════════════════════════════════
_has_fub = "fub_score_2023"     in merged.columns and merged["fub_score_2023"].notna().sum() >= 5
_has_emp = "emp_part_velo_2019" in merged.columns and merged["emp_part_velo_2019"].notna().sum() >= 5

if _has_fub or _has_emp:
    st.divider()
    section(5, "Validation Externe - FUB Baromètre et EMP Modal Share")

    st.markdown(r"""
La confrontation de l'IMD (objectif, infrastructure) avec le **FUB Baromètre**
(satisfaction subjective des cyclistes) et la **part modale EMP 2019** (usage réel)
permet de tester la validité externe du classement politique.
""")

    if _has_fub:
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            # FUB par parti
            _fub_df = merged_pol.dropna(subset=["fub_score_2023"]).copy()
            if len(_fub_df) >= 3:
                _fub_bar = (
                    _fub_df.groupby("parti_grp")["fub_score_2023"]
                    .agg(mediane="median", count="count")
                    .reindex(_parti_order_by_imd).reset_index()
                )
                fig_fub_bar = go.Figure()
                for _, row in _fub_bar.iterrows():
                    if pd.isna(row["mediane"]):
                        continue
                    fig_fub_bar.add_trace(go.Bar(
                        x=[row["parti_grp"]], y=[row["mediane"]],
                        marker_color=_parti_grp_colors.get(str(row["parti_grp"]), "#95a5a6"),
                        text=[f"n={int(row['count'])}"], textposition="outside",
                        hovertemplate=f"<b>{row['parti_grp']}</b><br>FUB médian : {row['mediane']:.2f}<extra></extra>",
                    ))
                fig_fub_bar.add_hline(y=3.0, line_dash="dot", line_color="#999",
                                      annotation_text="Score médian national (/6)")
                fig_fub_bar.update_layout(
                    showlegend=False, plot_bgcolor="white",
                    yaxis_title="FUB score (/6)", xaxis_title="Parti",
                    margin=dict(l=10, r=10, t=20, b=10), height=360,
                    yaxis=dict(range=[0, 6.5]),
                )
                st.plotly_chart(fig_fub_bar, use_container_width=True)
                st.caption("**Figure 5.1.** FUB Baromètre 2023 médian par parti.")

        with col_f2:
            # IMD vs FUB scatter
            _fub_scat = merged_pol.dropna(subset=["IMD", "fub_score_2023"]).copy()
            if len(_fub_scat) >= 5:
                fig_fub_scat = px.scatter(
                    _fub_scat, x="IMD", y="fub_score_2023",
                    color="parti_grp", color_discrete_map=_parti_grp_colors,
                    text="city" if show_labels else None,
                    size="n_stations", size_max=18,
                    hover_data=["city", "parti_maire"],
                    labels={"IMD": "IMD (/100)", "fub_score_2023": "FUB score (/6)",
                            "parti_grp": "Parti"},
                    height=360,
                )
                # Droite de régression globale (numpy, sans statsmodels)
                _fv = _fub_scat[["IMD", "fub_score_2023"]].dropna()
                if len(_fv) >= 3:
                    _cx = np.polyfit(_fv["IMD"].values, _fv["fub_score_2023"].values, 1)
                    _xr = np.linspace(_fv["IMD"].min(), _fv["IMD"].max(), 100)
                    fig_fub_scat.add_trace(go.Scatter(
                        x=_xr, y=np.polyval(_cx, _xr), mode="lines",
                        line=dict(color="#555", dash="dash", width=1.5),
                        name="Tendance globale", showlegend=False,
                    ))
                fig_fub_scat.update_layout(
                    plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                # Corrélation
                if _SCIPY:
                    _valid = _fub_scat[["IMD", "fub_score_2023"]].dropna()
                    _rho_fub, _p_fub = _spr(_valid["IMD"], _valid["fub_score_2023"])
                    st.plotly_chart(fig_fub_scat, use_container_width=True)
                    st.caption(
                        f"**Figure 5.2.** IMD objectif vs FUB subjectif. "
                        f"ρ de Spearman = {_rho_fub:.3f}, "
                        f"p = {_p_fub:.4f if _p_fub >= 0.001 else '< 0,001'}."
                    )
                else:
                    st.plotly_chart(fig_fub_scat, use_container_width=True)
                    st.caption("**Figure 5.2.** IMD objectif vs FUB Baromètre subjectif.")

    if _has_emp:
        _emp_df = merged_pol.dropna(subset=["emp_part_velo_2019"]).copy()
        if len(_emp_df) >= 3:
            _emp_bar = (
                _emp_df.groupby("parti_grp")["emp_part_velo_2019"]
                .agg(mediane="median", count="count")
                .reindex(_parti_order_by_imd).reset_index()
            )
            fig_emp = go.Figure()
            for _, row in _emp_bar.iterrows():
                if pd.isna(row["mediane"]):
                    continue
                fig_emp.add_trace(go.Bar(
                    x=[row["parti_grp"]], y=[row["mediane"]],
                    marker_color=_parti_grp_colors.get(str(row["parti_grp"]), "#95a5a6"),
                    text=[f"n={int(row['count'])}  {row['mediane']:.1f}%"],
                    textposition="outside",
                    hovertemplate=f"<b>{row['parti_grp']}</b><br>Part modale : {row['mediane']:.1f}%<extra></extra>",
                ))
            fig_emp.update_layout(
                showlegend=False, plot_bgcolor="white",
                yaxis_title="Part modale vélo EMP 2019 (%)", xaxis_title="Parti",
                margin=dict(l=10, r=10, t=20, b=10), height=360,
            )
            st.plotly_chart(fig_emp, use_container_width=True)
            st.caption(
                "**Figure 5.3.** Part modale vélo (EMP 2019) médiane par parti. "
                "Indicateur comportemental mesurant l'usage réel, "
                "indépendamment des scores d'infrastructure."
            )

# ══════════════════════════════════════════════════════════════════════════════
# Section 6 - Historique VLS : Âge du réseau et Continuité Politique
# ══════════════════════════════════════════════════════════════════════════════
if "annee_vls" in merged_pol.columns:
    st.divider()
    section(6, "Historique VLS - Âge du Réseau et Continuité Politique")

    st.markdown(r"""
L'âge du réseau VLS en 2020 est un facteur de confusion majeur : les villes pionnières
(réseau créé avant 2015) ont eu 5–15 ans pour développer leur infrastructure cyclable,
**indépendamment** de la couleur politique actuelle. La colonne `couleur_precedente`
permet de tester si le mandat 2020 a **changé** ou **maintenu** l'orientation politique.
""")

    merged_pol["annee_vls_num"] = pd.to_numeric(merged_pol["annee_vls"], errors="coerce")
    merged_pol["age_vls"] = 2020 - merged_pol["annee_vls_num"]
    merged_pol["ere_vls"] = pd.cut(
        merged_pol["annee_vls_num"],
        bins=[0, 2010, 2015, 2019, 2025],
        labels=["Pionnière (≤2010)", "Expansion (2011–2015)",
                "Consolidation (2016–2019)", "Récente (2020+)"],
        right=True,
    )

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        # IMD vs âge réseau
        _age_scat = merged_pol.dropna(subset=["age_vls", "IMD"]).copy()
        if len(_age_scat) >= 5:
            fig_age = px.scatter(
                _age_scat, x="age_vls", y="IMD",
                color="parti_grp", color_discrete_map=_parti_grp_colors,
                text="city" if show_labels else None,
                hover_data=["city", "parti_maire", "annee_vls"],
                labels={"age_vls": "Âge du réseau VLS en 2020 (années)",
                        "IMD": "Score IMD (/100)", "parti_grp": "Parti"},
                height=380,
            )
            # Droite de régression globale (numpy, sans statsmodels)
            _av = _age_scat[["age_vls", "IMD"]].dropna()
            if len(_av) >= 3:
                _ca = np.polyfit(_av["age_vls"].values, _av["IMD"].values, 1)
                _xra = np.linspace(_av["age_vls"].min(), _av["age_vls"].max(), 100)
                fig_age.add_trace(go.Scatter(
                    x=_xra, y=np.polyval(_ca, _xra), mode="lines",
                    line=dict(color="#555", dash="dash", width=1.5),
                    name="Tendance globale", showlegend=False,
                ))
            fig_age.update_layout(
                plot_bgcolor="white", showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_age, use_container_width=True)
            if _SCIPY:
                _valid_age = _age_scat[["age_vls", "IMD"]].dropna()
                _rho_age, _p_age = _spr(_valid_age["age_vls"], _valid_age["IMD"])
                st.caption(
                    f"**Figure 6.1.** IMD vs âge du réseau VLS. "
                    f"ρ = {_rho_age:+.3f}, p = {_p_age:.4f if _p_age >= 0.001 else '< 0,001'}. "
                    "Une corrélation positive confirmée indique un **effet d'héritage** "
                    "dominant sur l'effet partisan actuel."
                )
            else:
                st.caption("**Figure 6.1.** IMD vs âge du réseau VLS.")

    with col_h2:
        # Continuité politique
        if "couleur_precedente" in merged_pol.columns and "couleur_municipale" in merged_pol.columns:
            _cont = merged_pol.dropna(subset=["couleur_precedente", "couleur_municipale"]).copy()
            _cont["continuité"] = (
                _cont["couleur_municipale"].astype(str)
                == _cont["couleur_precedente"].astype(str)
            ).map({True: "Continuité politique", False: "Alternance politique"})

            _cont_imd = (
                _cont.groupby("continuité")["IMD"]
                .agg(n="count", mediane="median", moyenne="mean")
                .reset_index()
            )
            fig_cont = px.bar(
                _cont_imd, x="continuité", y="mediane",
                color="continuité",
                color_discrete_map={
                    "Continuité politique": "#1A6FBF",
                    "Alternance politique": "#e74c3c",
                },
                text="n",
                labels={"mediane": "IMD médian (/100)", "continuité": ""},
                height=260,
            )
            fig_cont.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig_cont.update_layout(
                showlegend=False, plot_bgcolor="white",
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_cont, use_container_width=True)
            st.caption(
                "**Figure 6.2.** IMD médian selon la continuité ou alternance politique "
                "(mandat 2014–2020 → mandat 2020)."
            )

    # Tableau des villes pionnières vs récentes
    with st.expander("Tableau : ère de création VLS × parti × IMD"):
        if "ere_vls" in merged_pol.columns:
            _ere_tbl = (
                merged_pol.groupby(["ere_vls", "parti_grp"], observed=True)
                .agg(n=("city", "count"), IMD_med=("IMD", "median"))
                .reset_index()
                .dropna(subset=["IMD_med"])
                .sort_values(["ere_vls", "IMD_med"], ascending=[True, False])
            )
            _ere_tbl.columns = ["Ère VLS", "Parti", "N villes", "IMD médian"]
            _ere_tbl["IMD médian"] = _ere_tbl["IMD médian"].round(1)
            st.dataframe(_ere_tbl, use_container_width=True, hide_index=True)
            st.caption(
                "**Tableau 6.1.** IMD médian par ère de création du réseau VLS et par parti. "
                "Permet de contrôler visuellement l'effet d'héritage."
            )

# ══════════════════════════════════════════════════════════════════════════════
# Section 7 - Vue agrégée par couleur (condensé)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(7, "Vue Agrégée par Couleur Politique - Blocs Gauche / Centre / Droite")

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
            _mc.sort_values("couleur_municipale"), x="couleur_municipale", y="IMD",
            color="couleur_municipale", color_discrete_map=_COULEUR_COLORS,
            points="all", hover_data=_hov_c,
            labels={"couleur_municipale": "Couleur politique", "IMD": "Score IMD (/100)"},
            height=360,
        )
        fig_box_c.update_layout(plot_bgcolor="white", showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_box_c, use_container_width=True)
        st.caption("**Figure 7.1.** IMD par bloc politique (vue agrégée).")

    with col_c2:
        _couleur_tbl = (
            _mc.groupby("couleur_municipale", observed=True)["IMD"]
            .agg(n="count", mediane="median", moyenne="mean").reset_index()
        )
        _couleur_tbl.columns = ["Couleur", "N", "Méd.", "Moy."]
        _couleur_tbl[["Méd.", "Moy."]] = _couleur_tbl[["Méd.", "Moy."]].round(1)
        st.dataframe(_couleur_tbl, use_container_width=True, hide_index=True)
        st.caption("**Tableau 7.1.**")
        _grps_c = [
            _mc.loc[_mc["couleur_municipale"] == g, "IMD"].dropna().values
            for g in _COULEUR_ORDER if (_mc["couleur_municipale"] == g).sum() >= 2
        ]
        if _SCIPY and len(_grps_c) >= 2:
            try:
                _Hc, _pc = _kw(*_grps_c)
                st.metric("K-W H", f"{_Hc:.3f}")
                st.metric("p", f"{_pc:.4f}" if _pc >= 0.001 else "< 0,001",
                          "sign." if _pc < 0.05 else "n.s.")
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# Section 8 - Dimension Régionale
# ══════════════════════════════════════════════════════════════════════════════
if has_reg:
    st.divider()
    section(8, "Dimension Régionale - Exécutifs Régionaux et IMD")

    _reg_pol = merged.dropna(subset=["couleur_regionale"]).copy()
    _reg_pol["couleur_regionale"] = pd.Categorical(
        _reg_pol["couleur_regionale"], categories=_COULEUR_ORDER, ordered=True
    )
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        _hov_r = ["city", "region", "IMD"]
        if "parti_maire" in _reg_pol.columns:
            _hov_r.append("parti_maire")
        fig_reg = px.box(
            _reg_pol.sort_values("couleur_regionale"), x="couleur_regionale", y="IMD",
            color="couleur_regionale", color_discrete_map=_COULEUR_COLORS,
            points="all", hover_data=_hov_r,
            labels={"couleur_regionale": "Couleur régionale (2021)", "IMD": "Score IMD (/100)"},
            height=360,
        )
        fig_reg.update_layout(plot_bgcolor="white", showlegend=False,
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_reg, use_container_width=True)
        st.caption("**Figure 8.1.** IMD par couleur régionale. Note : majorité LR depuis 2021.")
    with col_r2:
        _reg_tbl = (
            _reg_pol.groupby("couleur_regionale", observed=True)["IMD"]
            .agg(n="count", mediane="median").reset_index()
        )
        _reg_tbl.columns = ["Couleur régionale", "N", "IMD médian"]
        _reg_tbl["IMD médian"] = _reg_tbl["IMD médian"].round(1)
        st.dataframe(_reg_tbl, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 9 - Fiche par Ville
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(9, "Fiche par Ville - Profil Politique, IMD et Contexte Historique")

_city_list = sorted(merged_pol["city"].dropna().unique().tolist())
if _city_list:
    _sel_city = st.selectbox("Sélectionner une agglomération", _city_list, index=0)
    _row = merged_pol[merged_pol["city"] == _sel_city].iloc[0] if _sel_city in merged_pol["city"].values else None

    if _row is not None:
        c1, c2, c3, c4 = st.columns(4)

        # Colonne 1 : contexte politique
        with c1:
            st.markdown("**Contexte politique**")
            _parti  = _row.get("parti_maire", "-")
            _couleur = _row.get("couleur_municipale", "-")
            _couleur_prec = _row.get("couleur_precedente", "-")
            _continuity = "✓ Continuité" if str(_couleur) == str(_couleur_prec) else "↔ Alternance"
            st.markdown(
                f"<div style='font-size:0.85rem; line-height:1.8;'>"
                f"<b>Maire :</b> {_row.get('maire', '-')}<br>"
                f"<b>Parti :</b> {_parti}<br>"
                f"<b>Bloc :</b> {_couleur}<br>"
                f"<b>Mandat précédent :</b> {_couleur_prec}<br>"
                f"<b>Transition :</b> {_continuity}<br>"
                f"<b>Région :</b> {_row.get('region', '-')}<br>"
                f"<b>Région (couleur) :</b> {_row.get('couleur_regionale', '-')}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Colonne 2 : réseau VLS
        with c2:
            st.markdown("**Réseau VLS**")
            _ann_vls  = _row.get("annee_vls", "-")
            _ann_vls  = "-" if pd.isna(_ann_vls) else str(_ann_vls)
            try:
                _age_net = str(int(2020 - float(_ann_vls))) if _ann_vls != "-" else "-"
            except (ValueError, TypeError):
                _age_net = "-"
            _cap_v   = _row.get("capacity", float("nan"))
            _s_cap   = f"{_cap_v:.0f}" if pd.notna(_cap_v) else "-"
            st.markdown(
                f"<div style='font-size:0.85rem; line-height:1.8;'>"
                f"<b>Création du réseau :</b> {_ann_vls}<br>"
                f"<b>Âge en 2020 :</b> {_age_net} ans<br>"
                f"<b>N° stations :</b> {int(_row.get('n_stations', 0))}<br>"
                f"<b>Capacité moyenne :</b> {_s_cap} points<br>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Colonne 3 : IMD et composantes
        with c3:
            st.markdown("**Scores IMD**")
            _imd_val  = _row.get("IMD", float("nan"))
            _ies_val  = _row.get("IES", float("nan"))
            _s_val    = _row.get("S_securite", float("nan"))
            _i_val    = _row.get("I_infra", float("nan"))
            _m_val    = _row.get("M_multi", float("nan"))
            _t_val    = _row.get("T_topo", float("nan"))
            _s_imd  = f"{_imd_val:.1f}" if pd.notna(_imd_val) else "-"
            _s_ies  = f"{_ies_val:.3f}" if pd.notna(_ies_val) else "-"
            _s_sec  = f"{_s_val:.3f}"   if pd.notna(_s_val)   else "-"
            _s_inf  = f"{_i_val:.3f}"   if pd.notna(_i_val)   else "-"
            _s_mul  = f"{_m_val:.3f}"   if pd.notna(_m_val)   else "-"
            _s_top  = f"{_t_val:.3f}"   if pd.notna(_t_val)   else "-"
            st.markdown(
                f"<div style='font-size:0.85rem; line-height:1.8;'>"
                f"<b>IMD (/100) :</b> {_s_imd}<br>"
                f"<b>IES :</b> {_s_ies}<br>"
                f"<b>S_sécurité :</b> {_s_sec}<br>"
                f"<b>I_infra :</b> {_s_inf}<br>"
                f"<b>M_multi :</b> {_s_mul}<br>"
                f"<b>T_topo :</b> {_s_top}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Colonne 4 : données externes et socioéconomiques
        with c4:
            st.markdown("**Données externes**")
            _fub_v   = _row.get("fub_score_2023", float("nan"))
            _emp_v   = _row.get("emp_part_velo_2019", float("nan"))
            _rev_v   = _row.get("revenu_median_uc", float("nan"))
            _gin_v   = _row.get("gini_revenu", float("nan"))
            _voit_v  = _row.get("part_menages_voit0", float("nan"))
            _vtrav_v = _row.get("part_velo_travail", float("nan"))
            # Pré-formatage pour éviter les f-strings avec format spec conditionnel
            _s_fub   = f"{_fub_v:.2f}"   if pd.notna(_fub_v)   else "-"
            _s_emp   = f"{_emp_v:.1f}"   if pd.notna(_emp_v)   else "-"
            _s_rev   = f"{_rev_v:,.0f}"  if pd.notna(_rev_v)   else "-"
            _s_gin   = f"{_gin_v:.3f}"   if pd.notna(_gin_v)   else "-"
            _s_voit  = f"{_voit_v:.1f}"  if pd.notna(_voit_v)  else "-"
            _s_vtrav = f"{_vtrav_v:.2f}" if pd.notna(_vtrav_v) else "-"
            st.markdown(
                f"<div style='font-size:0.85rem; line-height:1.8;'>"
                f"<b>FUB score (/6) :</b> {_s_fub}<br>"
                f"<b>EMP part vélo :</b> {_s_emp} %<br>"
                f"<b>Revenu médian/UC :</b> {_s_rev} €<br>"
                f"<b>Gini revenu :</b> {_s_gin}<br>"
                f"<b>Ménages sans voiture :</b> {_s_voit} %<br>"
                f"<b>Vélo domicile-travail :</b> {_s_vtrav} %"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Mini radar comparatif ville vs médiane de son parti
        if _comp_ok:
            st.markdown("#### Comparaison avec la médiane du parti")
            _parti_grp_sel = _grp_parti(str(_parti))
            _parti_med = (
                merged_pol[merged_pol["parti_grp"] == _parti_grp_sel][_comp_cols].median()
                if (merged_pol["parti_grp"] == _parti_grp_sel).sum() >= 1
                else pd.Series([0.5] * 4, index=_comp_cols)
            )
            _city_vals  = [_row.get(c, float("nan")) for c in _comp_cols]
            _parti_vals = [_parti_med.get(c, float("nan")) for c in _comp_cols]

            fig_city_radar = go.Figure()
            _theta_r = _comp_names + [_comp_names[0]]
            _v_city  = [v if pd.notna(v) else 0 for v in _city_vals] + [_city_vals[0] if pd.notna(_city_vals[0]) else 0]
            _v_parti = [v if pd.notna(v) else 0 for v in _parti_vals] + [_parti_vals[0] if pd.notna(_parti_vals[0]) else 0]

            fig_city_radar.add_trace(go.Scatterpolar(
                r=_v_city, theta=_theta_r, fill="toself", name=_sel_city,
                line_color=_parti_grp_colors.get(_parti_grp_sel, "#1A6FBF"), opacity=0.8,
            ))
            fig_city_radar.add_trace(go.Scatterpolar(
                r=_v_parti, theta=_theta_r, fill="toself",
                name=f"Médiane {_parti_grp_sel}",
                line_color="#cccccc", opacity=0.5,
            ))
            fig_city_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=30, r=30, t=20, b=60),
                height=320,
            )
            st.plotly_chart(fig_city_radar, use_container_width=True)
            st.caption(
                f"**Figure 9.1.** Profil IMD de {_sel_city} (plein) vs médiane du groupe "
                f"**{_parti_grp_sel}** (grisé)."
            )

# ══════════════════════════════════════════════════════════════════════════════
# Section 10 - Analyse de Régression Contrôlée
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(10, "Analyse de Régression Contrôlée - Effet Partisan Net")

st.markdown(f"""
OLS multivarié : **{_ym['title']}** ~ covariables numériques + indicatrices partisanes.
Objectif : estimer l'effet partisan *net* une fois contrôlées la taille du réseau,
le revenu médian, les inégalités et la topographie.
""")

_cov_candidates = {
    "log_n_stations":            "log(N stations)",
    "revenu_median_uc":          "Revenu médian/UC",
    "gini_revenu":               "Gini revenu",
    "topography_roughness_index":"Rugosité topographique",
    "age_vls":                   "Âge réseau VLS",
}

_reg_df = merged_pol.dropna(subset=[y_col, "parti_grp"]).copy()
if "annee_vls" in _reg_df.columns:
    _reg_df["annee_vls_num"] = pd.to_numeric(_reg_df["annee_vls"], errors="coerce")
    _reg_df["age_vls"] = 2020 - _reg_df["annee_vls_num"]
if "n_stations" in _reg_df.columns:
    _reg_df["log_n_stations"] = np.log1p(_reg_df["n_stations"].values.astype(float))

_avail_covs = [c for c in _cov_candidates if c in _reg_df.columns and _reg_df[c].notna().sum() >= 5]
_ref_parti  = _parti_order_by_imd[-2] if len(_parti_order_by_imd) >= 2 else _parti_order_by_imd[-1]

with st.expander("Réglages de la régression"):
    _sel_covs = st.multiselect(
        "Covariables numériques", _avail_covs,
        default=_avail_covs,
        format_func=lambda c: _cov_candidates.get(c, c),
    )
    _ref_parti = st.selectbox(
        "Parti de référence (intercept)", _partis_maj,
        index=min(len(_partis_maj) - 1, len(_partis_maj) - 1),
    )

if len(_reg_df) >= 10 and _sel_covs:
    _reg_sub = _reg_df.dropna(subset=_sel_covs + [y_col]).copy()
    _parti_dummies = pd.get_dummies(_reg_sub["parti_grp"], prefix="").astype(float)
    _dummy_cols = [c for c in _parti_dummies.columns if c != f"_{_ref_parti}" and c != f" {_ref_parti}"]
    # Drop reference
    _parti_dummies = _parti_dummies.drop(columns=[c for c in _parti_dummies.columns
                                                   if _ref_parti in c], errors="ignore")
    _dummy_cols = list(_parti_dummies.columns)

    # Standardise les covariables numériques
    _X_cov = _reg_sub[_sel_covs].copy()
    for col in _sel_covs:
        _std = _X_cov[col].std()
        if _std > 0:
            _X_cov[col] = (_X_cov[col] - _X_cov[col].mean()) / _std

    _X_mat = np.column_stack([
        np.ones(len(_reg_sub)),
        _X_cov.values,
        _parti_dummies.values,
    ])
    _y_vec = _reg_sub[y_col].values.astype(float)

    try:
        _beta = np.linalg.lstsq(_X_mat, _y_vec, rcond=None)[0]
        _y_hat    = _X_mat @ _beta
        _resids   = _y_vec - _y_hat
        _n, _p    = _X_mat.shape
        _sigma2   = (_resids @ _resids) / max(_n - _p, 1)
        _XtX      = _X_mat.T @ _X_mat
        _XtX_inv  = np.linalg.pinv(_XtX)
        _se       = np.sqrt(_sigma2 * np.diag(_XtX_inv))
        _t_stat   = _beta / np.where(_se > 0, _se, np.nan)
        _SStot    = float(np.sum((_y_vec - _y_vec.mean()) ** 2))
        _SSres    = float(_resids @ _resids)
        _R2       = 1 - _SSres / _SStot if _SStot > 0 else float("nan")
        _R2_adj   = 1 - (1 - _R2) * (_n - 1) / (_n - _p) if _n > _p else float("nan")

        _feat_names = (
            ["(Constante)"]
            + [_cov_candidates.get(c, c) for c in _sel_covs]
            + [f"Parti = {c.strip('_').strip()}" for c in _dummy_cols]
        )
        _ols_tbl = pd.DataFrame({
            "Variable":  _feat_names,
            "β (std)":   [f"{b:+.3f}" for b in _beta],
            "Err. std.": [f"{s:.3f}" for s in _se],
            "t":         [f"{t:+.2f}" if pd.notna(t) else "-" for t in _t_stat],
            "Sig.":      [
                "***" if abs(t) > 3.29 else "**" if abs(t) > 2.58 else "*" if abs(t) > 1.96 else "."
                for t in [tv if pd.notna(tv) else 0 for tv in _t_stat]
            ],
        })

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        r2_c1.metric("R²", f"{_R2:.3f}")
        r2_c2.metric("R² ajusté", f"{_R2_adj:.3f}")
        r2_c3.metric("N obs.", f"{_n}")

        st.dataframe(_ols_tbl, use_container_width=True, hide_index=True)
        st.caption(
            f"**Tableau 10.1.** Régression OLS - {_ym['title']} sur covariables "
            f"numériques standardisées et indicatrices partisanes. "
            f"Référence : **{_ref_parti}**. "
            "Sig. : *** p<0,001 ; ** p<0,01 ; * p<0,05 ; . p<0,1. "
            "β (std) = coefficients sur variables centrées réduites - interprétables "
            "comme contributions relatives."
        )
    except Exception as e:
        st.info(f"Régression non disponible : {e}")
else:
    st.info(
        f"Données insuffisantes pour la régression "
        f"({len(_reg_df)} obs., {len(_sel_covs)} covariables sélectionnées)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Section 11 - Tableau de classement
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(11, "Tableau de Classement - Agglomérations, Partis et Scores")

_disp_cols = ["city", "parti_maire", "couleur_municipale", "annee_vls",
              "couleur_precedente", "region", "couleur_regionale", "maire",
              "n_stations", "IMD"]
if ies_col_ok and "IES" in merged.columns:
    _disp_cols.append("IES")
for _ext in ["fub_score_2023", "emp_part_velo_2019", "gini_revenu"]:
    if _ext in merged.columns and merged[_ext].notna().sum() >= 3:
        _disp_cols.append(_ext)

_disp_available = [c for c in _disp_cols if c in merged.columns]
_disp = merged[_disp_available].dropna(subset=["parti_maire"]).copy()
_disp = _disp.sort_values("IMD", ascending=False)

_rename = {
    "city":               "Agglomération",
    "parti_maire":        "Parti",
    "couleur_municipale": "Bloc",
    "annee_vls":          "Création VLS",
    "couleur_precedente": "Bloc précédent",
    "region":             "Région",
    "couleur_regionale":  "Bloc régional",
    "maire":              "Maire",
    "n_stations":         "Stations",
    "IMD":                "IMD (/100)",
    "IES":                "IES",
    "fub_score_2023":     "FUB (/6)",
    "emp_part_velo_2019": "EMP (%)",
    "gini_revenu":        "Gini",
}
_disp = _disp.rename(columns={k: v for k, v in _rename.items() if k in _disp.columns})
if "IMD (/100)" in _disp.columns:
    _disp["IMD (/100)"] = _disp["IMD (/100)"].round(1)
if "IES" in _disp.columns:
    _disp["IES"] = _disp["IES"].round(3)

_col_cfg: dict = {
    "IMD (/100)": st.column_config.ProgressColumn("IMD (/100)", min_value=0, max_value=100, format="%.1f"),
}
if "IES" in _disp.columns:
    _col_cfg["IES"] = st.column_config.NumberColumn("IES", format="%.3f")
if "FUB (/6)" in _disp.columns:
    _col_cfg["FUB (/6)"] = st.column_config.NumberColumn("FUB (/6)", format="%.2f")

st.dataframe(_disp, use_container_width=True, hide_index=True, column_config=_col_cfg)
st.caption(
    "**Tableau 11.1.** Classement par IMD décroissant. "
    "'Création VLS' = année d'ouverture du réseau dock-based. "
    "'Bloc précédent' = couleur politique 2014–2020. "
    "Source : élections municipales 2020 / régionales 2021, Gold Standard GBFS - "
    "R. Fossé & G. Pallares, 2025–2026."
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 12 - Discussion
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(12, "Discussion - Synthèse, Biais et Perspectives")

st.markdown(r"""
#### 12.1. Principaux Résultats

L'analyse multidimensionnelle révèle plusieurs résultats descriptifs convergents :

- **L'héritage du réseau domine l'effet partisan.** La corrélation IMD × âge du réseau
  VLS est la plus robuste observée sur ce panel. Les villes pionnières (Lyon, Paris,
  Nantes, Rennes, créées avant 2010) affichent des scores IMD élevés *indépendamment*
  de leur couleur politique en 2020. C'est la principale limite à toute interprétation
  causale des comparaisons politiques.

- **EELV se distingue mais dans un contexte favorable.** Les villes à maire EELV
  (Grenoble, Lyon, Bordeaux, Strasbourg, Besançon) cumulent réseau ancien, grande
  taille, densité élevée et électeurs pro-mobilité douce. L'analyse de régression
  contrôlée (Section 10) permet de décomposer la part de cet avantage expliquée
  par le parti vs les covariables structurelles.

- **LR est hétérogène.** Le groupe le plus nombreux (LR) présente la plus grande
  dispersion IMD, allant de Toulouse (réseau ancien, score élevé) à des villes
  récentes et plus petites. La médiane LR est inférieure à PS et EELV mais supérieure
  à DVG sur ce panel.

- **L'IES est plus homogène entre partis.** Conformément à H₃, l'analyse du Gini
  et de l'IES suggère que la justice distributive ne suit pas le clivage partisan :
  des villes LR ou DVG offrent une mobilité inclusive, certaines villes PS présentent
  des sous-performances relatives.

#### 12.2. Biais et Limites

| Biais | Description | Correction possible |
|:--- |:--- |:--- |
| **Sélection** | Panel limité aux villes avec réseau VLS dock-based certifié | Extension aux free-floating (données GBFS brutes) |
| **Confusion taille** | Les EELV gouvernent surtout de grandes métropoles | Contrôle par log(n_stations) en Section 10 |
| **Confusion historique** | Réseau créé avant 2020 pour ~85 % des villes | Contrôle par âge_vls dans la régression |
| **Confusion topographie** | Les villes plates ont naturellement un meilleur IMD | Contrôle par topography_roughness_index |
| **Faibles effectifs** | ≤ 6 villes par parti pour la plupart des partis | Tests non-paramétriques, intervalles de confiance Bootstrap |

#### 12.3. Perspectives de Recherche

1. **Analyse longitudinale (2014 → 2020 → 2026)** : mesurer l'évolution de l'IMD
   avant et après un changement de majorité municipale (différence-en-différences).
2. **Données budgétaires** : croiser avec les plans vélo (PDME) et les délibérations
   de conseils municipaux pour mesurer l'intention politique directement.
3. **Variables instrumentales** : utiliser les résultats du premier tour comme
   instrument pour l'appartenance partisane et estimer un effet causal local.
4. **Enquête qualitative** : compléter par des entretiens avec élus et directeurs
   de la mobilité pour contextualiser les scores quantitatifs.

> **Conclusion provisoire :** La couleur politique est un **signal corrélé** mais
> non un déterminant isolé de la qualité VLS. Une fois contrôlés l'âge du réseau,
> la taille de la ville et la topographie, l'effet partisan net est plus faible que
> les comparaisons brutes ne le suggèrent. La vraie variable prédictive semble être
> le **moment de création du réseau** et la **taille démographique**, qui déterminent
> en amont la capacité d'investissement infrastructurel.
""")
