"""
2_Villes.py — Analyse comparative inter-urbaine des métriques d'enrichissement spatial.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, city_stats, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Analyse Comparative Inter-Urbaine — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Analyse Comparative Inter-Urbaine")
st.caption("Axe de Recherche 2 : Gouvernance Locale et Disparités Structurelles de l'Environnement Cyclable")

abstract_box(
    "<b>Question de recherche :</b> Les disparités inter-urbaines de qualité cyclable "
    "sont-elles le produit d'une fatalité géographique ou d'inégalités de gouvernance ?<br><br>"
    "Cette analyse comparative classe les agglomérations françaises dotées d'un réseau VLS "
    "selon les dimensions d'enrichissement spatial du Gold Standard GBFS. "
    "Le résultat clé de l'analyse spatiale globale — l'absence d'autocorrélation significative "
    "(Moran's $I = -0{,}023$, $p = 0{,}765$) — invalide l'hypothèse d'un déterminisme "
    "géographique structurant les disparités. Les villes performantes et sous-performantes "
    "ne forment pas de clusters territoriaux cohérents : ce sont les choix de gouvernance "
    "locale, et non la localisation géographique, qui expliquent l'hétérogénéité observée. "
    "Trois niveaux d'analyse sont proposés : classement univarié, nuage de points "
    "infrastructure × sinistralité, et profil radar multi-dimensionnel."
)

df     = load_stations()
cities = city_stats(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres d'Analyse")
    n_top = st.slider("Nombre d'agglomérations", min_value=5, max_value=40, value=20, step=5)
    metric_key = st.selectbox(
        "Dimension principale",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k]["label"],
        index=0,
    )
    min_stations = st.number_input(
        "Seuil de robustesse (min. stations)", min_value=1, max_value=500, value=10,
        help="Exclut les micro-réseaux pour garantir la pertinence statistique de la comparaison."
    )
    meta = METRICS[metric_key]
    st.divider()
    st.markdown(f"**{meta['label']}**")
    st.caption(meta["description"])

# ── Filtrage ──────────────────────────────────────────────────────────────────
cities_f = cities[cities["n_stations"] >= min_stations].copy()

if metric_key not in cities_f.columns:
    st.warning(f"Dimension `{metric_key}` absente des données agrégées.")
    st.stop()

ascending = not meta.get("higher_is_better", True)
cities_sorted = cities_f.dropna(subset=[metric_key]).sort_values(
    metric_key, ascending=ascending
)

# ── Section 1 — Classement univarié ──────────────────────────────────────────
section(1, "Classement Univarié — Agglomérations Triées par la Dimension Sélectionnée")

st.markdown(r"""
Le classement univarié constitue le premier niveau de diagnostic territorial. Il met en évidence
les agglomérations se situant aux extrêmes de la distribution nationale pour la dimension
sélectionnée, révélant des situations de sur-performance ou de sous-investissement structurel
dans un environnement cyclable de qualité.
""")

col_tab, col_chart = st.columns([2, 3])

with col_tab:
    extra_cols = [
        c for c in ["infra_cyclable_pct", "baac_accidents_cyclistes", "gtfs_heavy_stops_300m"]
        if c != metric_key
    ]
    display = cities_sorted.head(n_top)[
        ["city", "n_stations", metric_key] + extra_cols
    ].rename(columns={
        "city":                       "Agglomération",
        "n_stations":                 "Stations",
        metric_key:                   meta["label"],
        "infra_cyclable_pct":         "Infra cyclable (%)",
        "baac_accidents_cyclistes":   "Sinistralité (moy.)",
        "gtfs_heavy_stops_300m":      "TC lourds (moy.)",
    })
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            meta["label"]: st.column_config.ProgressColumn(
                meta["label"],
                min_value=float(cities_sorted[metric_key].min()),
                max_value=float(cities_sorted[metric_key].max()),
                format=f"%.2f {meta['unit']}",
            )
        },
    )
    st.caption(
        f"**Tableau 1.1.** Top {n_top} agglomérations classées par **{meta['label']}**. "
        f"Seuil de robustesse : ≥ {min_stations} stations Gold Standard certifiées. "
        f"Les valeurs représentent la moyenne de la dimension sur l'ensemble des stations "
        f"de chaque agglomération."
    )

with col_chart:
    plot_df = cities_sorted.head(n_top).copy()
    fig = px.bar(
        plot_df,
        x=metric_key,
        y="city",
        orientation="h",
        color=metric_key,
        color_continuous_scale=meta["color_scale"],
        text=metric_key,
        labels={"city": "Agglomération", metric_key: meta["label"]},
        height=max(400, n_top * 22),
    )
    fig.update_traces(texttemplate=f"%{{x:.2f}} {meta['unit']}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=80, t=10, b=10),
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Figure 1.1.** Classement des {n_top} premières agglomérations par **{meta['label']}**. "
        "Chaque barre représente la moyenne de la dimension sur les stations de l'agglomération. "
        "Les écarts inter-urbains, non expliqués par la géographie (Moran's $I = -0{,}023$), "
        "reflètent des choix différenciés de politique d'aménagement cyclable."
    )

# ── Section 2 — Infrastructure et accidentologie ──────────────────────────────
st.divider()
section(2, "Analyse Croisée Infrastructure × Sinistralité — Effet Protecteur de l'Aménagement Cyclable")

st.markdown(r"""
L'hypothèse d'un effet protecteur de l'infrastructure cyclable sur la sinistralité est centrale
dans la littérature (*Pucher et al., 2010 ; Jacobsen, 2003*). Ce nuage de points teste cette
relation à l'échelle des agglomérations françaises : les villes situées dans le quadrant
supérieur gauche (forte densité d'infrastructure, faible sinistralité) valident l'hypothèse ;
les villes hors-diagonale identifient des situations d'anomalie nécessitant une investigation
qualitative de gouvernance.
""")

scatter_df = cities_f.dropna(subset=["infra_cyclable_pct", "baac_accidents_cyclistes"])

fig_sc = px.scatter(
    scatter_df,
    x="infra_cyclable_pct",
    y="baac_accidents_cyclistes",
    size="n_stations",
    color="gtfs_heavy_stops_300m",
    color_continuous_scale="Blues",
    hover_name="city",
    hover_data={
        "n_stations": True,
        "infra_cyclable_pct": ":.2f",
        "baac_accidents_cyclistes": ":.3f",
    },
    labels={
        "infra_cyclable_pct":       "Couverture en infrastructure cyclable (%)",
        "baac_accidents_cyclistes": "Densité de sinistralité cycliste (BAAC, 300 m)",
        "gtfs_heavy_stops_300m":    "Accessibilité multimodale (arrêts TC lourds)",
    },
    size_max=40,
    height=480,
)
fig_sc.update_layout(
    plot_bgcolor="white",
    coloraxis_colorbar=dict(title="TC lourds"),
    margin=dict(l=10, r=10, t=10, b=10),
)
fig_sc.add_hline(
    y=float(scatter_df["baac_accidents_cyclistes"].mean()),
    line_dash="dot", line_color="#e74c3c", opacity=0.5,
    annotation_text="Moyenne nationale (sinistralité)", annotation_position="right",
)
fig_sc.add_vline(
    x=float(scatter_df["infra_cyclable_pct"].mean()),
    line_dash="dot", line_color="#1A6FBF", opacity=0.5,
    annotation_text="Moyenne nationale (infrastructure)", annotation_position="top",
)
st.plotly_chart(fig_sc, use_container_width=True)
st.caption(
    "**Figure 2.1.** Couverture en infrastructure cyclable (axe horizontal) versus densité "
    "de sinistralité cycliste (axe vertical) par agglomération. "
    "La taille encode le volume de stations Gold Standard ; la couleur encode l'accessibilité "
    "aux transports en commun lourds (GTFS). "
    "Les lignes pointillées indiquent les moyennes nationales. "
    "Le quadrant supérieur gauche (forte infrastructure, faible sinistralité) constitue "
    "la cible normative des politiques d'aménagement cyclable sécurisé."
)

# ── Section 3 — Profil radar ──────────────────────────────────────────────────
st.divider()
section(3, "Profil Radar Multi-Dimensionnel — Audit Comparatif des Agglomérations")

st.markdown(r"""
Le profil radar permet de visualiser simultanément les quatre dimensions de l'environnement
cyclable normalisées min-max sur l'échantillon sélectionné. Chaque axe est exprimé selon
la relation $\tilde{c}(v) = \frac{c(v) - \min_v c}{\max_v c - \min_v c} \in [0, 1]$,
où les indicateurs inverses (sinistralité) sont retournés de sorte qu'une valeur élevée
corresponde systématiquement à un environnement favorable. Cet outil permet d'identifier
les profils d'aménagement différenciés et les dimensions structurellement déficitaires
dans chaque agglomération.
""")

radar_cols = {
    "infra_cyclable_pct":        "Infrastructure cyclable",
    "gtfs_heavy_stops_300m":     "Multimodalité (TC lourds)",
    "baac_accidents_cyclistes":  "Sécurité (inv. sinistralité)",
    "gtfs_stops_within_300m_pct": "Couverture GTFS",
}
top_radar_cities = (
    cities_f.dropna(subset=list(radar_cols))
    .nlargest(5, "n_stations")["city"]
    .tolist()
)

radar_city_sel = st.multiselect(
    "Sélection de l'échantillon d'audit (2 à 8 agglomérations)",
    options=sorted(cities_f["city"].unique()),
    default=top_radar_cities[:5],
    max_selections=8,
)

if len(radar_city_sel) >= 2:
    radar_df = cities_f[cities_f["city"].isin(radar_city_sel)].dropna(subset=list(radar_cols))

    ndf = radar_df[list(radar_cols)].copy()
    for c in ndf.columns:
        rng = ndf[c].max() - ndf[c].min()
        ndf[c] = (ndf[c] - ndf[c].min()) / rng if rng else 0.0
    if "baac_accidents_cyclistes" in ndf.columns:
        ndf["baac_accidents_cyclistes"] = 1 - ndf["baac_accidents_cyclistes"]

    ndf["city"] = radar_df["city"].values
    categories = list(radar_cols.values())

    fig_radar = go.Figure()
    for _, row in ndf.iterrows():
        vals = [row[c] for c in radar_cols]
        vals += vals[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["city"],
            opacity=0.65,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
        margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "**Figure 3.1.** Profil radar des agglomérations sélectionnées. "
        "Chaque axe est normalisé min-max ($\\tilde{c} \\in [0, 1]$) sur l'échantillon affiché. "
        "La composante sécurité est retournée (valeur haute = faible sinistralité). "
        "L'aire de la figure est proportionnelle à la performance globale de l'environnement cyclable. "
        "Les villes présentant des profils asymétriques révèlent des stratégies d'aménagement "
        "sectorisées plutôt qu'une approche intégrée de la mobilité douce."
    )
else:
    st.info(
        "Sélectionnez au moins 2 agglomérations pour initier l'audit radar comparatif."
    )
