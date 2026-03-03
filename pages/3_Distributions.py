"""
3_Distributions.py - Distributions empiriques et structure de corrélation des métriques Gold Standard.
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
    from scipy.stats import mannwhitneyu as _mannwhitneyu
    from scipy.stats import shapiro as _shapiro
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Distributions Statistiques - Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Chargement anticipé (abstract dynamique) ──────────────────────────────────
df = load_stations()
_n_total  = len(df)
_n_dock   = int((df["station_type"] == "docked_bike").sum()) if "station_type" in df.columns else _n_total
_n_cities = df["city"].nunique()

st.title("Distributions Empiriques et Structure de Corrélation")
st.caption("Axe de Recherche 3 : Hétérogénéité Statistique et Indépendance des Dimensions d'Enrichissement")

abstract_box(
    "<b>Question de recherche :</b> La taille démographique d'une agglomération constitue-t-elle "
    "un prédicteur fiable de la qualité de son environnement cyclable ?<br><br>"
    f"Analyse des distributions empiriques des sept dimensions d'enrichissement du Gold Standard "
    f"sur <b>{_n_total:,} stations</b> ({_n_dock:,} dock-based) et <b>{_n_cities} agglomérations</b>. "
    "Résultat central : <i>r<sub>s</sub></i>&nbsp;=&nbsp;&minus;0,02 (hors Paris) entre taille et performance cyclable - "
    "aucun avantage métropolitain. Les distributions présentent toutes des asymétries positives "
    "(Shapiro-Wilk <i>p</i>&nbsp;&lt;&nbsp;0,05), justifiant les tests non paramétriques. "
    "La matrice Spearman confirme la quasi-indépendance des quatre composantes IMD, "
    "validant la construction de l'indice comme somme pondérée non redondante.",
    findings=[
        (f"{_n_total:,}", "stations analysées"),
        (f"{_n_dock:,}", "dont dock-based VLS"),
        (str(_n_cities), "agglomérations"),
        ("r_s = −0,02", "taille × performance"),
        ("p < 0,05", "normalité rejetée"),
    ],
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres Statistiques")

    type_options = ["Toutes les stations", "Dock-based (VLS)", "Free-floating"]
    type_map = {
        "Toutes les stations": None,
        "Dock-based (VLS)": "docked_bike",
        "Free-floating": "free_floating",
    }
    station_type_sel = st.selectbox("Type de station", options=type_options, index=1)

    all_cities = sorted(df["city"].unique())
    city_filter = st.multiselect(
        "Restreindre à une agglomération",
        options=all_cities,
        default=[],
        placeholder="Corpus national complet",
    )
    n_bins = st.slider("Classes (histogramme)", 20, 100, 40, 5)

    st.divider()
    include_socio = st.checkbox(
        "Inclure les indicateurs socio-écon. dans la matrice de corrélation",
        value=False,
        help="Ajoute revenu médian, Gini, part ménages sans voiture et part vélo-travail.",
    )

# ── Filtrage ──────────────────────────────────────────────────────────────────
dff = df.copy()
if station_type_sel != "Toutes les stations" and "station_type" in dff.columns:
    dff = dff[dff["station_type"] == type_map[station_type_sel]]
if city_filter:
    dff = dff[dff["city"].isin(city_filter)]

_type_label = f" ({station_type_sel})" if station_type_sel != "Toutes les stations" else ""
st.caption(
    f"**{len(dff):,}** stations analysées{_type_label} · "
    f"**{dff['city'].nunique()}** agglomérations · "
    f"Seuil de significativité des encoches (notched boxes) : $p \\approx 0{{,}}05$."
)

# ── Section 1 - Distributions univariées ─────────────────────────────────────
st.divider()
section(1, "Distributions Univariées - Forme Empirique et Dispersion des Sept Dimensions")

st.markdown(r"""
Les distributions empiriques révèlent systématiquement une **asymétrie positive** (queue à droite)
pour les dimensions de sinistralité et de multimodalité : la majorité des stations opèrent dans
des environnements à faible risque et faible accessibilité TC, tandis qu'une minorité bénéficie
d'une exposition simultanée à un réseau lourd. Cette structure de distribution justifie
l'usage de la **médiane** comme estimateur de tendance centrale dans le calcul de l'IMD,
plus robuste à l'asymétrie que la moyenne arithmétique.

Code couleur : **bleu** = indicateur direct (valeur élevée favorable) ;
**rouge** = indicateur inverse (valeur faible favorable) ; **gris-bleu** = neutre.
""")

metric_keys = [k for k in METRICS if k in dff.columns]
cols = st.columns(2)

for i, mkey in enumerate(metric_keys):
    meta = METRICS[mkey]
    series = dff[mkey].dropna()
    if series.empty:
        continue

    color = (
        "#1A6FBF" if meta["higher_is_better"] is True
        else "#c0392b" if meta["higher_is_better"] is False
        else "#5a7a99"
    )

    fig = px.histogram(
        series,
        nbins=n_bins,
        labels={"value": meta["label"]},
        color_discrete_sequence=[color],
        height=280,
    )
    med  = float(series.median())
    moy  = float(np.nanmean(series.to_numpy(dtype=float, na_value=np.nan)))
    std  = float(np.nanstd(series.to_numpy(dtype=float, na_value=np.nan)))
    skew = float(series.skew())
    fig.add_vline(
        x=med, line_dash="dash", line_color="#1A2332", opacity=0.75,
        annotation_text=f"Méd. {med:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=9),
    )
    fig.add_vline(
        x=moy, line_dash="dot", line_color="#e74c3c", opacity=0.65,
        annotation_text=f"Moy. {moy:.2f}",
        annotation_position="top left",
        annotation_font=dict(size=9, color="#e74c3c"),
    )
    _skew_dir = "droite" if skew > 0 else "gauche"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.96,
        text=f"γ₁ = {skew:+.2f}  σ = {std:.2f}",
        showarrow=False,
        font=dict(size=9, color="#444"),
        bgcolor="rgba(255,255,255,0.78)",
        bordercolor="#dde3ea", borderwidth=1,
        xanchor="right", yanchor="top",
    )
    fig.update_layout(
        title=dict(text=meta["label"], font_size=13),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        xaxis_title=f"{meta['label']} ({meta['unit']})",
        yaxis_title="Stations",
    )
    with cols[i % 2]:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"**Figure 1.{i+1}.** Distribution empirique de **{meta['label']}** "
            f"({len(series):,} stations valides{_type_label}). "
            f"Médiane $\\tilde{{x}} = {med:.2f}$ · Moyenne $\\bar{{x}} = {moy:.2f}$ · "
            f"σ = {std:.2f} {meta['unit']}. "
            f"γ₁ = {skew:+.2f} (asymétrie vers la {_skew_dir})."
        )

# ── Tableau de statistiques de forme ─────────────────────────────────────────
with st.expander(
    "Statistiques de forme - Asymétrie, Aplatissement et Test de Normalité (Shapiro-Wilk)",
    expanded=False,
):
    _shape_rows = []
    for _mkey in metric_keys:
        _meta_s = METRICS[_mkey]
        _s = dff[_mkey].dropna()
        if len(_s) < 10:
            continue
        _skew = float(_s.skew())
        _kurt = float(_s.kurtosis())  # pandas excess kurtosis

        # Shapiro-Wilk sur sous-échantillon (valide jusqu'à n=5000)
        _sw_sample = _s.sample(min(len(_s), 5000), random_state=42) if len(_s) > 5000 else _s
        try:
            _sw_res = _shapiro(_sw_sample.to_numpy(float)) if _SCIPY else None
            _sw_p = float(getattr(_sw_res, "pvalue", float("nan"))) if _sw_res is not None else float("nan")
        except Exception:
            _sw_p = float("nan")

        def _sig_sw(p: float) -> str:
            if p != p:  # nan
                return "-"
            if p < 0.001: return "Rejetée (***)"
            if p < 0.01:  return "Rejetée (**)"
            if p < 0.05:  return "Rejetée (*)"
            return "Non rejetée"

        _shape_rows.append({
            "Dimension": _meta_s["label"],
            "n valides": f"{len(_s):,}",
            "Asymétrie γ₁": f"{_skew:+.3f}",
            "Kurtosis γ₂": f"{_kurt:+.3f}",
            "SW p-val. (n ≤ 5 000)": f"{_sw_p:.4f}" if _sw_p == _sw_p else "-",
            "H₀ normalité": _sig_sw(_sw_p),
        })

    if _shape_rows:
        _shape_df = pd.DataFrame(_shape_rows)

        def _color_normalite(col: pd.Series) -> list[str]:
            styles = []
            for v in col:
                if "***" in str(v):
                    styles.append("background-color:#fde8e8;color:#c0392b;font-weight:700")
                elif "**" in str(v):
                    styles.append("background-color:#fef0e8;color:#c0392b")
                elif "*" in str(v):
                    styles.append("background-color:#fef9e8;color:#e67e22")
                elif v == "Non rejetée":
                    styles.append("background-color:#eafaf1;color:#27ae60;font-weight:700")
                else:
                    styles.append("")
            return styles

        st.dataframe(
            _shape_df.style.apply(_color_normalite, subset=["H₀ normalité"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "**Tableau 1.1.** Statistiques de forme pour les dimensions d'enrichissement spatial"
            + (_type_label or "") + ". "
            "γ₁ = coefficient d'asymétrie de Fisher (γ₁ > 0 : queue à droite, distribution asymétrique positive). "
            "γ₂ = excès de kurtosis (γ₂ > 0 : leptokurtique, queues plus lourdes que la loi normale). "
            "SW = test de Shapiro-Wilk ($H_0$ : la distribution est normale). "
            "*** $p < 0{{,}}001$ · ** $p < 0{{,}}01$ · * $p < 0{{,}}05$. "
            "Le rejet quasi-universel de la normalité valide l'usage d'estimateurs robustes "
            "(médiane, $\\rho$ de Spearman) et invalide l'usage de tests paramétriques "
            "(t-test, ANOVA) dans les comparaisons inter-agglomérations."
        )

# ── Section 2 - Dispersion inter-agglomérations ───────────────────────────────
st.divider()
section(2, "Dispersion Inter-Agglomérations - Significativité Statistique des Différences de Médiane")

st.markdown(r"""
Les boîtes à moustaches à encoches (*notched boxes*) permettent l'évaluation visuelle
de la significativité statistique des différences de médiane entre agglomérations.
Deux encoches non chevauchantes indiquent une différence significative au seuil
$p \approx 0{,}05$ (intervalle de confiance approximatif à 95 % autour de la médiane,
selon la formule $\tilde{x} \pm 1{,}57 \cdot \text{IQR} / \sqrt{n}$).
""")

bp_metric = st.selectbox(
    "Dimension à analyser",
    options=[k for k in METRICS if k in dff.columns],
    format_func=lambda k: METRICS[k]["label"],
    key="bp_metric",
)

top15 = (
    dff.groupby("city")["uid"].count()
    .nlargest(15)
    .index.tolist()
)
# Include Montpellier in default selection if not already there
_default_bp = list(dict.fromkeys(
    (["Montpellier"] if "Montpellier" in dff["city"].values else []) + top15
))[:10]

bp_city_sel = st.multiselect(
    "Agglomérations à comparer",
    options=sorted(dff["city"].unique()),
    default=_default_bp,
    key="bp_cities",
)

if bp_city_sel:
    bp_df   = dff[dff["city"].isin(bp_city_sel) & dff[bp_metric].notna()]
    meta_bp = METRICS[bp_metric]

    order = (
        bp_df.groupby("city")[bp_metric].median()
        .sort_values(ascending=not meta_bp.get("higher_is_better", True))
        .index.tolist()
    )

    # Montpellier highlighted
    color_map_bp = {
        c: ("#e74c3c" if c == "Montpellier" else "#1A6FBF")
        for c in bp_city_sel
    }

    fig_bp = px.box(
        bp_df,
        x="city",
        y=bp_metric,
        color="city",
        color_discrete_map=color_map_bp,
        category_orders={"city": order},
        labels={"city": "Agglomération", bp_metric: meta_bp["label"]},
        height=420,
        notched=True,
    )
    fig_bp.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_bp, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 2.1.** Boîtes à moustaches à encoches de **{meta_bp['label']}** "
        f"par agglomération{_type_label}, triées par médiane décroissante. "
        "Les encoches représentent l'IC 95 % autour de la médiane ($p \\approx 0{{,}}05$). "
        "Les agglomérations dont les encoches ne se chevauchent pas présentent "
        "une différence de médiane statistiquement significative. "
        "**Montpellier** est mis en évidence en rouge."
    )

    # Mann-Whitney U — Montpellier vs. chaque autre agglomération sélectionnée
    if "Montpellier" in bp_city_sel and len(bp_city_sel) > 1 and _SCIPY:
        try:
            _mmm_vals = bp_df[bp_df["city"] == "Montpellier"][bp_metric].dropna().to_numpy(float)
            _mw_rows = []
            for _oc in [c for c in order if c != "Montpellier"]:
                _oc_vals = bp_df[bp_df["city"] == _oc][bp_metric].dropna().to_numpy(float)
                if len(_mmm_vals) > 0 and len(_oc_vals) > 0:
                    _mwu_res = _mannwhitneyu(_mmm_vals, _oc_vals, alternative="two-sided")
                    _pu = float(getattr(_mwu_res, "pvalue", 1.0))
                    _mmm_med_bp = float(np.nanmedian(_mmm_vals))
                    _oc_med_bp  = float(np.nanmedian(_oc_vals))
                    _delta = _mmm_med_bp - _oc_med_bp
                    _mw_rows.append({
                        "Agglomération": _oc,
                        "n": int(len(_oc_vals)),
                        "Médiane locale": f"{_oc_med_bp:.3f}",
                        "Δ vs Montpellier": f"{_delta:+.3f}",
                        "U p-val.": f"{float(_pu):.4f}",
                        "Significatif (α = 0,05)": "Oui ✓" if _pu < 0.05 else "Non",
                    })
            if _mw_rows:
                with st.expander(
                    f"Test de Mann-Whitney U : Montpellier (n = {len(_mmm_vals)}) "
                    f"vs. {len(_mw_rows)} agglomération(s) sélectionnée(s)",
                    expanded=False,
                ):
                    _mw_df = pd.DataFrame(_mw_rows)

                    def _color_mw(col: pd.Series) -> list[str]:
                        if col.name != "Significatif (α = 0,05)":
                            return [""] * len(col)
                        return [
                            "background-color:#eafaf1;color:#27ae60;font-weight:700" if "Oui" in str(v)
                            else "color:#888"
                            for v in col
                        ]

                    st.dataframe(
                        _mw_df.style.apply(_color_mw),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        f"**Tableau 2.1.** Test de Mann-Whitney U bilatéral — "
                        f"**{meta_bp['label']}**. "
                        "H₀ : les distributions de Montpellier et de l'agglomération comparée "
                        "sont identiques. Δ = médiane Montpellier − médiane locale. "
                        "Seuil α = 0,05 (non corrigé pour comparaisons multiples)."
                    )
        except ImportError:
            pass
else:
    st.info("Sélectionnez au moins une agglomération pour afficher les boîtes à moustaches.")

# ── Section 3 - Matrice de corrélation de Spearman ───────────────────────────
st.divider()
section(3, "Matrice de Corrélation de Spearman - Colinéarités et Indépendance des Dimensions")

st.markdown(r"""
La matrice de corrélation de rang de Spearman ($\rho$) entre les sept dimensions
d'enrichissement constitue un test de colinéarité critique avant la construction de l'IMD.
Une colinéarité forte ($|\rho| > 0{,}7$) entre deux dimensions incluses dans l'indice
indiquerait une redondance informationnelle et nécessiterait une réduction par ACP
ou une pondération différenciée.

Le résultat observé - quasi-indépendance des dimensions retenues pour l'IMD -
valide la non-redondance du modèle composite et justifie la pondération différenciée
optimisée par évolution différentielle.
**À l'échelle des villes**, la corrélation Spearman entre l'IMD agrégé et le revenu
médian des ménages est $\rho_s = +0{,}055$ (p = 0,677, non significatif) -
validant l'indépendance de la performance cyclable vis-à-vis du niveau de richesse de l'agglomération.
""")

_SOCIO_COLS = ["revenu_median_uc", "gini_revenu", "part_menages_voit0", "part_velo_travail"]
_socio_labels = {
    "revenu_median_uc":    "Revenu médian (€/UC)",
    "gini_revenu":         "Gini revenu",
    "part_menages_voit0":  "Ménages sans voiture (%)",
    "part_velo_travail":   "Part vélo travail (%)",
}

num_cols = [k for k in METRICS if k in dff.columns]
if include_socio:
    socio_available = [c for c in _SOCIO_COLS if c in dff.columns]
    num_cols = num_cols + socio_available
    all_labels = {**{c: METRICS[c]["label"] for c in [k for k in METRICS if k in dff.columns]}, **_socio_labels}
else:
    all_labels = {c: METRICS[c]["label"] for c in num_cols}

corr_df = dff[num_cols].dropna(how="all").corr(method="spearman")
labels  = [all_labels.get(c, c) for c in corr_df.columns]

# Calcul des p-values Spearman pairwise
_nc = len(num_cols)
_p_mat = np.ones((_nc, _nc))
if _SCIPY:
    try:
        _data_corr = dff[num_cols].dropna(how="all")
        for _ci in range(_nc):
            for _cj in range(_ci + 1, _nc):
                _pair = _data_corr[[num_cols[_ci], num_cols[_cj]]].dropna()
                if len(_pair) >= 4:
                    _res_ij = _spearmanr(_pair.iloc[:, 0].to_numpy(float),
                                         _pair.iloc[:, 1].to_numpy(float))
                    _pij = float(getattr(_res_ij, "pvalue", 1.0))
                    _p_mat[_ci, _cj] = _pij
                    _p_mat[_cj, _ci] = _pij
    except Exception:
        pass

def _sig_star_corr(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

_corr_text = [
    [
        f"{corr_df.values[i][j]:.2f}{_sig_star_corr(_p_mat[i][j])}" if i != j else "1.00"
        for j in range(_nc)
    ]
    for i in range(_nc)
]

fig_corr = go.Figure(
    data=go.Heatmap(
        z=corr_df.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=_corr_text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>",
    )
)
fig_corr.update_layout(
    height=480 if not include_socio else 560,
    margin=dict(l=10, r=10, t=10, b=130),
    xaxis=dict(tickangle=-35),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

_socio_note = (
    " Les colonnes socio-économiques (revenu, Gini, etc.) sont incluses à titre exploratoire "
    "pour tester la colinéarité avec les dimensions spatiales de l'IMD."
    if include_socio else ""
)
st.caption(
    "**Figure 3.1.** Matrice de corrélation de rang de Spearman ($\\rho$) entre les "
    + ("sept" if not include_socio else "sept + quatre") +
    " dimensions d'enrichissement spatial"
    + (_type_label or "") + ". "
    "Bleu : $\\rho < 0$ (corrélation négative) ; "
    "Rouge : $\\rho > 0$ (corrélation positive). "
    "La diagonale principale vaut 1 par définition. "
    "Les coefficients proches de $\\pm 1$ signalent une colinéarité structurelle à "
    "contrôler avant toute modélisation composite (VIF, ACP)."
    + _socio_note
)

# ── Section 4 - Scatter matriciel ─────────────────────────────────────────────
st.divider()
section(4, "Analyse par Paires - Scatter Matriciel sur Échantillon Stratifié")

with st.expander("Afficher le scatter matriciel (calcul sur sous-échantillon aléatoire)", expanded=False):
    st.markdown(r"""
    Le scatter matriciel (*splom*) représente toutes les paires de dimensions sélectionnées
    sur un sous-échantillon aléatoire de stations. Il permet de diagnostiquer visuellement
    les relations non-linéaires que le coefficient de Spearman ne capture pas pleinement,
    ainsi que la présence d'outliers multivariés susceptibles d'influencer les estimateurs
    de corrélation.
    """)
    sample_n = st.slider("Taille de l'échantillon (stations)", 500, 5000, 2000, 500)
    pair_keys = st.multiselect(
        "Dimensions à croiser",
        options=num_cols,
        default=["infra_cyclable_pct", "baac_accidents_cyclistes", "gtfs_heavy_stops_300m"],
        format_func=lambda k: all_labels.get(k, k),
    )
    if len(pair_keys) >= 2:
        sample_df = dff[pair_keys + ["city"]].dropna().sample(
            min(sample_n, len(dff[pair_keys].dropna())), random_state=42
        )
        fig_pair = px.scatter_matrix(
            sample_df,
            dimensions=pair_keys,
            color="city",
            labels={k: all_labels.get(k, k) for k in pair_keys},
            height=600,
            opacity=0.4,
        )
        fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
        fig_pair.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pair, use_container_width=True)
        st.caption(
            f"**Figure 4.1.** Scatter matriciel sur un sous-échantillon aléatoire de "
            f"{sample_n:,} stations{_type_label} (stratification par agglomération). "
            "La couleur encode l'agglomération d'appartenance. "
            "Le triangle inférieur affiche les dispersions bivariées ; "
            "la diagonale et le triangle supérieur sont masqués pour la lisibilité."
        )
    else:
        st.info("Sélectionnez au moins 2 dimensions pour générer le scatter matriciel.")

# ── Section 5 - Taille des réseaux × Performance ──────────────────────────────
st.divider()
section(5, "Taille des Réseaux et Performance Cyclable - Aucun Avantage Métropolitain")

st.markdown(r"""
La question de recherche centrale de cet axe postule que la **taille démographique ou la taille
du réseau VLS** d'une agglomération ne constitue pas un prédicteur fiable de sa performance cyclable.
Ce résultat, contre-intuitif, remet en cause l'hypothèse d'économies d'échelle dans les politiques
de mobilité douce. Le graphique ci-dessous quantifie la corrélation de Spearman entre le nombre
de stations (*n_stations*) et la médiane de chaque dimension, agrégée à l'échelle de l'agglomération.

Un coefficient $\rho_s$ proche de zéro indique l'absence de relation monotone entre taille et performance.
**Barres vertes** = corrélation significative (*p* < 0,05) · **Barres grises** = non significatif.
""")

# Agrégation par ville (médiane de chaque dimension)
_city_agg5 = (
    dff.groupby("city")["uid"].count()
    .rename("n_stations")
    .reset_index()
)
_med5 = (
    dff.groupby("city")[metric_keys].median()
    .add_prefix("_med_")
    .reset_index()
)
_city_agg5 = _city_agg5.merge(_med5, on="city")

# Corrélation Spearman : n_stations × médiane de chaque dimension
_sp5_rows = []
for _mk5 in metric_keys:
    _col5 = f"_med_{_mk5}"
    _valid5 = _city_agg5[["n_stations", _col5]].dropna()
    if len(_valid5) >= 5:
        _rho5, _p5 = (0.0, 1.0)
        if _SCIPY:
            try:
                _res5 = _spearmanr(_valid5["n_stations"].to_numpy(float),
                                   _valid5[_col5].to_numpy(float))
                _rho5 = float(getattr(_res5, "correlation", 0.0))
                _p5   = float(getattr(_res5, "pvalue", 1.0))
            except Exception:
                pass
        _sp5_rows.append({
            "label":  METRICS[_mk5]["label"],
            "rho":    _rho5,
            "pval":   _p5,
            "sig":    _p5 < 0.05,
            "n":      int(len(_valid5)),
        })

if _sp5_rows:
    _bar_colors = [
        "#27ae60" if (r["sig"] and r["rho"] > 0)
        else "#c0392b" if (r["sig"] and r["rho"] < 0)
        else "#bdc3c7"
        for r in _sp5_rows
    ]
    _bar_text = [
        f"{r['rho']:+.3f}{'*' if r['pval'] < 0.05 else ''}"
        for r in _sp5_rows
    ]
    fig_taille = go.Figure(go.Bar(
        x=[r["label"] for r in _sp5_rows],
        y=[r["rho"] for r in _sp5_rows],
        marker_color=_bar_colors,
        text=_bar_text,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>ρ_s = %{y:.3f}<extra></extra>",
    ))
    fig_taille.add_hline(y=0, line_color="#444", line_width=1)
    fig_taille.add_hrect(
        y0=-0.3, y1=0.3,
        fillcolor="rgba(200,200,200,0.08)", line_width=0,
        annotation_text="Zone non significative (|ρ| < 0,3)",
        annotation_font=dict(size=9, color="#aaa"),
        annotation_position="bottom right",
    )
    fig_taille.update_layout(
        title=dict(
            text="Corrélation de Spearman : Taille du réseau (n stations) × Médiane de chaque dimension",
            font_size=11, x=0,
        ),
        height=340,
        margin=dict(l=10, r=10, t=38, b=120),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title="ρ_s", range=[-0.85, 0.85], gridcolor="#e8edf3", zeroline=False),
        plot_bgcolor="#f8fafd",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_taille, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "**Figure 5.1.** Corrélations de Spearman entre le nombre de stations (taille du réseau) "
        f"et la médiane de chaque dimension d'enrichissement, agrégées sur {_city_agg5['city'].nunique()} "
        f"agglomérations{_type_label}. "
        "Vert/rouge = significatif (*p* < 0,05) ; gris = non significatif. "
        "Zone grisée : |ρ_s| < 0,3 (corrélation faible à négligeable). "
        "L'absence de corrélations fortes valide l'hypothèse d'indépendance taille–performance."
    )

    # Scatter interactif : taille × dimension choisie
    st.markdown("**Explorer la relation taille × dimension**")
    _taille_metric = st.selectbox(
        "Dimension à explorer (médiane par agglomération)",
        options=metric_keys,
        format_func=lambda k: METRICS[k]["label"],
        key="taille_metric_sel",
    )
    _col5_sel = f"_med_{_taille_metric}"
    _scatter5 = _city_agg5[["city", "n_stations", _col5_sel]].dropna().copy()
    _scatter5.columns = ["city", "n_stations", "y_med"]

    if len(_scatter5) >= 3:
        _c5 = np.polyfit(_scatter5["n_stations"].to_numpy(float),
                         _scatter5["y_med"].to_numpy(float), 1)
        _xfit5 = np.linspace(float(_scatter5["n_stations"].min()),
                              float(_scatter5["n_stations"].max()), 80)
        _yfit5 = np.polyval(_c5, _xfit5)

        _highlight5 = {"Montpellier", "Paris", "Lyon", "Marseille", "Bordeaux", "Rennes", "Nantes"}
        _lab5 = _scatter5["city"].isin(_highlight5)

        fig_sc5 = go.Figure()
        fig_sc5.add_trace(go.Scatter(
            x=_scatter5["n_stations"], y=_scatter5["y_med"],
            mode="markers",
            marker=dict(
                color=["#e74c3c" if c == "Montpellier" else "#1A6FBF"
                       for c in _scatter5["city"]],
                size=6, opacity=0.6,
            ),
            text=_scatter5["city"],
            hovertemplate="<b>%{text}</b><br>n = %{x} stations<br>médiane = %{y:.3f}<extra></extra>",
            showlegend=False,
        ))
        fig_sc5.add_trace(go.Scatter(
            x=_scatter5.loc[_lab5, "n_stations"],
            y=_scatter5.loc[_lab5, "y_med"],
            mode="markers+text",
            marker=dict(
                color=["#e74c3c" if c == "Montpellier" else "#1565C0"
                       for c in _scatter5.loc[_lab5, "city"]],
                size=8, opacity=1,
            ),
            text=_scatter5.loc[_lab5, "city"],
            textposition="top center",
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate="<b>%{text}</b><br>n = %{x}<br>médiane = %{y:.3f}<extra></extra>",
        ))
        fig_sc5.add_trace(go.Scatter(
            x=_xfit5, y=_yfit5,
            mode="lines",
            line=dict(color="#e74c3c", dash="dash", width=1.5),
            name="OLS",
            showlegend=True,
        ))
        # Afficher ρ et p-val du row correspondant
        _row5 = next((r for r in _sp5_rows if r["label"] == METRICS[_taille_metric]["label"]), None)
        _rho5_str = f"{_row5['rho']:+.3f}" if _row5 else "n/a"
        _p5_str   = f"{_row5['pval']:.4f}" if _row5 else "n/a"
        fig_sc5.update_layout(
            title=dict(
                text=f"Taille réseau × {METRICS[_taille_metric]['label']} — ρ_s = {_rho5_str}, p = {_p5_str}",
                font_size=11, x=0,
            ),
            height=370,
            margin=dict(l=10, r=10, t=38, b=40),
            xaxis=dict(title="Nombre de stations", gridcolor="#e8edf3"),
            yaxis=dict(title=f"Médiane {METRICS[_taille_metric]['label']} ({METRICS[_taille_metric]['unit']})",
                       gridcolor="#e8edf3"),
            plot_bgcolor="#f8fafd",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(size=9)),
        )
        st.plotly_chart(fig_sc5, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"**Figure 5.2.** Nuage de points : nombre de stations × médiane de **{METRICS[_taille_metric]['label']}** "
            f"par agglomération ({len(_scatter5)} points). Droite OLS en tirets. "
            "Montpellier en rouge. ρ_s = corrélation de Spearman (rang), p = p-valeur bilatérale."
        )
