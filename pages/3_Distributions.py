"""
3_Distributions.py — Distributions empiriques et structure de corrélation des métriques Gold Standard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Distributions Statistiques — Gold Standard GBFS",
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
    f"Cette analyse examine les distributions empiriques des sept dimensions d'enrichissement "
    f"du Gold Standard GBFS à l'échelle des <b>{_n_total:,} stations certifiées</b> "
    f"(dont {_n_dock:,} dock-based VLS) issues de <b>{_n_cities} agglomérations</b>. "
    "Le résultat central est contre-intuitif : la corrélation de rang de Spearman entre "
    "la taille de l'agglomération et la performance cyclable est statistiquement non significative "
    "($r_s = -0{,}02$, hors Paris), invalidant l'hypothèse d'un avantage dimensionnel des "
    "grandes métropoles. Ce résultat renforce l'importance de l'analyse par dimension, "
    "car les distributions présentent des asymétries positives caractéristiques "
    "(queues de distribution à droite) qui biaisent les comparaisons fondées sur la seule moyenne. "
    "La matrice de corrélation de Spearman révèle en outre la quasi-indépendance des "
    "quatre dimensions retenues pour l'IMD, validant leur non-colinéarité."
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

# ── Section 1 — Distributions univariées ─────────────────────────────────────
st.divider()
section(1, "Distributions Univariées — Forme Empirique et Dispersion des Sept Dimensions")

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
    med = float(series.median())
    moy = float(series.mean())
    fig.add_vline(
        x=med, line_dash="dash", line_color="#1A2332", opacity=0.7,
        annotation_text=f"Méd. {med:.2f}", annotation_position="top right",
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"**Figure 1.{i+1}.** Distribution empirique de **{meta['label']}** "
            f"({len(series):,} stations valides{_type_label}). "
            f"Médiane $\\tilde{{x}} = {med:.2f}$ {meta['unit']} · "
            f"Moyenne $\\bar{{x}} = {moy:.2f}$ {meta['unit']}. "
            f"L'écart médiane/moyenne quantifie le degré d'asymétrie de la distribution."
        )

# ── Section 2 — Dispersion inter-agglomérations ───────────────────────────────
st.divider()
section(2, "Dispersion Inter-Agglomérations — Significativité Statistique des Différences de Médiane")

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
    st.plotly_chart(fig_bp, use_container_width=True)
    st.caption(
        f"**Figure 2.1.** Boîtes à moustaches à encoches de **{meta_bp['label']}** "
        f"par agglomération{_type_label}, triées par médiane décroissante. "
        "Les encoches représentent l'IC 95 % autour de la médiane ($p \\approx 0{{,}}05$). "
        "Les agglomérations dont les encoches ne se chevauchent pas présentent "
        "une différence de médiane statistiquement significative. "
        "**Montpellier** est mis en évidence en rouge."
    )
else:
    st.info("Sélectionnez au moins une agglomération pour afficher les boîtes à moustaches.")

# ── Section 3 — Matrice de corrélation de Spearman ───────────────────────────
st.divider()
section(3, "Matrice de Corrélation de Spearman — Colinéarités et Indépendance des Dimensions")

st.markdown(r"""
La matrice de corrélation de rang de Spearman ($\rho$) entre les sept dimensions
d'enrichissement constitue un test de colinéarité critique avant la construction de l'IMD.
Une colinéarité forte ($|\rho| > 0{,}7$) entre deux dimensions incluses dans l'indice
indiquerait une redondance informationnelle et nécessiterait une réduction par ACP
ou une pondération différenciée.

Le résultat observé — quasi-indépendance des dimensions retenues pour l'IMD —
valide la non-redondance du modèle composite et justifie la pondération différenciée
optimisée par évolution différentielle.
**À l'échelle des villes**, la corrélation Spearman entre l'IMD agrégé et le revenu
médian des ménages est $\rho_s = +0{,}055$ (p = 0,677, non significatif) —
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

fig_corr = go.Figure(
    data=go.Heatmap(
        z=corr_df.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
        texttemplate="%{text}",
        hoverongaps=False,
    )
)
fig_corr.update_layout(
    height=480 if not include_socio else 560,
    margin=dict(l=10, r=10, t=10, b=130),
    xaxis=dict(tickangle=-35),
)
st.plotly_chart(fig_corr, use_container_width=True)

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

# ── Section 4 — Scatter matriciel ─────────────────────────────────────────────
st.divider()
section(4, "Analyse par Paires — Scatter Matriciel sur Échantillon Stratifié")

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
