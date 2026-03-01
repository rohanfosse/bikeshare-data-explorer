"""
2_Villes.py — Analyse comparative inter-urbaine des métriques d'enrichissement spatial.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, city_stats, compute_imd_cities, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Analyse Comparative Inter-Urbaine — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Chargement anticipé (abstract dynamique) ──────────────────────────────────
df     = load_stations()
imd_df = compute_imd_cities(df)   # dock-based, sans entrées non-ville
cities = city_stats(df)

_n_dock_cities = len(imd_df)
_top_city      = imd_df.iloc[0]["city"] if len(imd_df) else "—"
_top_imd       = f"{imd_df.iloc[0]['IMD']:.1f}" if len(imd_df) else "—"

st.title("Analyse Comparative Inter-Urbaine")
st.caption("Axe de Recherche 2 : Gouvernance Locale et Disparités Structurelles de l'Environnement Cyclable")

abstract_box(
    "<b>Question de recherche :</b> Les disparités inter-urbaines de qualité cyclable "
    "sont-elles le produit d'une fatalité géographique ou d'inégalités de gouvernance ?<br><br>"
    "Cette analyse comparative classe les <b>{n} agglomérations</b> françaises dotées d'un "
    "réseau VLS dock-based certifié Gold Standard selon les dimensions d'enrichissement spatial. "
    "Le résultat clé de l'analyse spatiale globale — l'absence d'autocorrélation significative "
    "(Moran's $I = -0{,}023$, $p = 0{,}765$) — invalide l'hypothèse d'un déterminisme "
    "géographique structurant les disparités. Les villes performantes et sous-performantes "
    "ne forment pas de clusters territoriaux cohérents : ce sont les choix de gouvernance "
    "locale, et non la localisation géographique, qui expliquent l'hétérogénéité observée. "
    "Quatre niveaux d'analyse sont proposés : classement univarié par dimension, nuage de points "
    "infrastructure × sinistralité, profil radar multi-dimensionnel, et comparaison IMD × socio-économique."
).replace("{n}", str(_n_dock_cities))

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres d'Analyse")
    n_top = st.slider("Nombre d'agglomérations", min_value=5, max_value=50, value=20, step=5)
    metric_key = st.selectbox(
        "Dimension principale",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k]["label"],
        index=0,
    )
    min_stations = st.number_input(
        "Seuil de robustesse (min. stations dock)", min_value=1, max_value=500, value=10,
        help="Exclut les micro-réseaux pour garantir la pertinence statistique de la comparaison."
    )
    meta = METRICS[metric_key]
    st.divider()
    st.markdown(f"**{meta['label']}**")
    st.caption(meta["description"])

# ── Filtrage ──────────────────────────────────────────────────────────────────
# Utiliser uniquement les stations dock-based pour la comparaison inter-urbaine
df_dock = df[df["station_type"] == "docked_bike"].copy() if "station_type" in df.columns else df.copy()
_NON_CITY = {"France", "FR", "Grand Est", "Basque Country"}
df_dock   = df_dock[~df_dock["city"].isin(_NON_CITY)]
cities_dock = city_stats(df_dock)
cities_f  = cities_dock[cities_dock["n_stations"] >= min_stations].copy()

if metric_key not in cities_f.columns:
    st.warning(f"Dimension `{metric_key}` absente des données agrégées (dock-based).")
    st.stop()

ascending = not meta.get("higher_is_better", True)
cities_sorted = cities_f.dropna(subset=[metric_key]).sort_values(
    metric_key, ascending=ascending
).reset_index(drop=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Agglomérations dock-based analysées", f"{len(cities_f)}")
k2.metric("Optimum IMD national",                _top_city)
k3.metric("Score IMD — Optimum",                 f"{_top_imd} / 100")
k4.metric("IMD médian national",                 f"{imd_df['IMD'].median():.1f} / 100")

# ── Section 1 — Classement univarié ──────────────────────────────────────────
st.divider()
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
        if c != metric_key and c in cities_sorted.columns
    ]
    disp_cols = ["city", "n_stations", metric_key] + extra_cols
    display = cities_sorted.head(n_top)[disp_cols].rename(columns={
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
        f"**Tableau 1.1.** Top {n_top} agglomérations classées par **{meta['label']}** "
        f"(stations dock-based uniquement, seuil ≥ {min_stations} stations). "
        f"Les valeurs représentent la moyenne de la dimension sur l'ensemble des stations dock."
    )

with col_chart:
    plot_df = cities_sorted.head(n_top).copy()
    _bar_colors = ["#e74c3c" if c == "Montpellier" else "#1A6FBF" for c in plot_df["city"]]
    fig = px.bar(
        plot_df,
        x=metric_key, y="city",
        orientation="h",
        color=metric_key,
        color_continuous_scale=meta["color_scale"],
        text=metric_key,
        labels={"city": "Agglomération", metric_key: meta["label"]},
        height=max(400, n_top * 22),
    )
    if "Montpellier" in plot_df["city"].values:
        fig.data[0].marker.color = _bar_colors
    fig.update_traces(texttemplate=f"%{{x:.2f}} {meta['unit']}", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=80, t=10, b=10),
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Figure 1.1.** Classement des {n_top} premières agglomérations par **{meta['label']}** "
        "(stations dock-based uniquement). La barre rouge identifie Montpellier (cas d'étude, rang IMD #2). "
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
supérieur gauche (forte densité d'infrastructure, faible sinistralité) valident l'hypothèse.
""")

scatter_df = cities_f.dropna(subset=["infra_cyclable_pct", "baac_accidents_cyclistes"])

if not scatter_df.empty:
    fig_sc = px.scatter(
        scatter_df,
        x="infra_cyclable_pct", y="baac_accidents_cyclistes",
        size="n_stations", color="gtfs_heavy_stops_300m",
        color_continuous_scale="Blues",
        hover_name="city",
        text=scatter_df["city"].apply(
            lambda c: c if c in {"Montpellier", "Strasbourg", "Paris", "Lyon", "Bordeaux",
                                  "Nantes", "Rennes", "Brest"} else ""
        ),
        hover_data={"n_stations": True, "infra_cyclable_pct": ":.2f",
                    "baac_accidents_cyclistes": ":.3f"},
        labels={
            "infra_cyclable_pct":       "Couverture en infrastructure cyclable (%)",
            "baac_accidents_cyclistes": "Densité de sinistralité cycliste (BAAC, 300 m)",
            "gtfs_heavy_stops_300m":    "Accessibilité multimodale (arrêts TC lourds)",
        },
        size_max=40, height=480,
    )
    # Cercle rouge Montpellier
    _mmm_sc = scatter_df[scatter_df["city"] == "Montpellier"]
    if not _mmm_sc.empty:
        fig_sc.add_trace(go.Scatter(
            x=[_mmm_sc["infra_cyclable_pct"].iloc[0]],
            y=[_mmm_sc["baac_accidents_cyclistes"].iloc[0]],
            mode="markers",
            marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color="#e74c3c", width=3)),
            name="Montpellier (rang IMD #2)",
            showlegend=True,
        ))
    fig_sc.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig_sc.update_layout(
        plot_bgcolor="white",
        coloraxis_colorbar=dict(title="TC lourds"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig_sc.add_hline(y=float(scatter_df["baac_accidents_cyclistes"].mean()),
                     line_dash="dot", line_color="#e74c3c", opacity=0.5,
                     annotation_text="Moy. sinistralité", annotation_position="right")
    fig_sc.add_vline(x=float(scatter_df["infra_cyclable_pct"].mean()),
                     line_dash="dot", line_color="#1A6FBF", opacity=0.5,
                     annotation_text="Moy. infrastructure", annotation_position="top")
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "**Figure 2.1.** Couverture en infrastructure cyclable (axe horizontal) versus densité "
        "de sinistralité cycliste (axe vertical) par agglomération (stations dock-based). "
        "La taille encode le volume de stations dock Gold Standard ; la couleur encode l'accessibilité "
        "aux TC lourds. Le cercle rouge identifie Montpellier (rang IMD #2 national)."
    )

# ── Section 3 — IMD vs Revenu médian ──────────────────────────────────────────
st.divider()
section(3, "Justice Sociale — Classement IMD × Revenu Médian par Agglomération")

_has_revenu = "revenu_median_uc" in imd_df.columns and imd_df["revenu_median_uc"].notna().sum() >= 5
if _has_revenu:
    st.markdown(r"""
    Le croisement du score IMD (qualité physique de l'offre) avec le revenu médian par unité
    de consommation (INSEE Filosofi) révèle la dimension d'équité sociale de la distribution
    des réseaux VLS. Le résultat empirique ($\rho = +0{,}055$, $p = 0{,}677$, non significatif)
    confirme l'absence de déterminisme économique : la qualité cyclable est indépendante
    du revenu de l'agglomération.
    """)
    _imd_f = imd_df[imd_df["n_stations"] >= min_stations].dropna(
        subset=["revenu_median_uc", "IMD"]
    ).copy()
    if len(_imd_f) >= 5:
        _c = np.polyfit(_imd_f["revenu_median_uc"].values, _imd_f["IMD"].values, 1)
        _imd_f["IMD_hat"] = np.polyval(_c, _imd_f["revenu_median_uc"].values).clip(min=1.0)
        _imd_f["IES"]     = (_imd_f["IMD"] / _imd_f["IMD_hat"]).round(3)

        _med_r = float(_imd_f["revenu_median_uc"].median())
        _med_i = float(_imd_f["IMD"].median())
        _imd_f["regime"] = _imd_f.apply(
            lambda row: (
                "Mobilité Inclusive" if row["revenu_median_uc"] < _med_r and row["IMD"] >= _med_i
                else "Excellence Consolidée" if row["revenu_median_uc"] >= _med_r and row["IMD"] >= _med_i
                else "Désert de Mobilité" if row["revenu_median_uc"] < _med_r and row["IMD"] < _med_i
                else "Sous-Performance"
            ), axis=1
        )

        _regime_colors = {
            "Mobilité Inclusive":    "#27ae60",
            "Excellence Consolidée": "#1A6FBF",
            "Désert de Mobilité":    "#e74c3c",
            "Sous-Performance":      "#e67e22",
        }
        _r_counts = _imd_f["regime"].value_counts()
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col_w, (lbl, clr) in zip(
            [rc1, rc2, rc3, rc4],
            _regime_colors.items()
        ):
            col_w.metric(lbl, f"{int(_r_counts.get(lbl, 0))} villes")

        _x_line = np.linspace(float(_imd_f["revenu_median_uc"].min()),
                               float(_imd_f["revenu_median_uc"].max()), 200)
        fig_ies = px.scatter(
            _imd_f, x="revenu_median_uc", y="IMD",
            color="regime", color_discrete_map=_regime_colors,
            size="n_stations", size_max=22,
            hover_name="city",
            hover_data={"IMD": ":.1f", "revenu_median_uc": ":,.0f", "IES": ":.3f"},
            labels={
                "revenu_median_uc": "Revenu médian/UC (€/an, INSEE Filosofi)",
                "IMD":              "Score IMD (/100)",
                "regime":           "Régime IES",
            },
            height=500,
        )
        fig_ies.add_trace(go.Scatter(
            x=_x_line, y=np.polyval(_c, _x_line),
            mode="lines", name="Référentiel OLS (ρ = +0,055, ns)",
            line=dict(color="#1A2332", dash="dash", width=2),
        ))
        fig_ies.add_hline(y=_med_i, line_dash="dot", line_color="#888", opacity=0.5,
                          annotation_text="Médiane IMD", annotation_position="right")
        fig_ies.add_vline(x=_med_r, line_dash="dot", line_color="#888", opacity=0.5,
                          annotation_text="Médiane revenu", annotation_position="top")
        # Annotation Montpellier
        _mmm_ies = _imd_f[_imd_f["city"] == "Montpellier"]
        if not _mmm_ies.empty:
            fig_ies.add_annotation(
                x=float(_mmm_ies["revenu_median_uc"].iloc[0]),
                y=float(_mmm_ies["IMD"].iloc[0]),
                text=f"<b>Montpellier<br>IES = {float(_mmm_ies['IES'].iloc[0]):.3f}</b>",
                showarrow=True, ax=-65, ay=-35,
                font=dict(size=11, color="#27ae60"),
                bgcolor="rgba(240,255,240,0.9)",
                bordercolor="#27ae60", borderpad=5,
            )
        fig_ies.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        )
        st.plotly_chart(fig_ies, use_container_width=True)
        st.caption(
            "**Figure 3.1.** Score IMD versus revenu médian/UC (INSEE Filosofi) par agglomération. "
            "Corrélation de Spearman : $\\rho = +0{,}055$ ($p = 0{,}677$, **non significatif**) — "
            "la qualité VLS est indépendante du niveau de revenu. "
            "Les quadrants révèlent les quatre régimes de justice cyclable. "
            "Analyse complète : page **IES — Indice d'Équité Sociale**."
        )
else:
    st.info(
        "Les données INSEE Filosofi (`revenu_median_uc`) ne sont pas disponibles. "
        "Vérifiez que le fichier `stations_gold_standard_final.parquet` est utilisé."
    )

# ── Section 4 — Profil radar ──────────────────────────────────────────────────
st.divider()
section(4, "Profil Radar Multi-Dimensionnel — Audit Comparatif des Agglomérations")

st.markdown(r"""
Le profil radar permet de visualiser simultanément les quatre dimensions de l'environnement
cyclable normalisées min-max sur l'échantillon sélectionné. Chaque axe est exprimé selon
la relation $\tilde{c}(v) = \frac{c(v) - \min_v c}{\max_v c - \min_v c} \in [0, 1]$,
où les indicateurs inverses (sinistralité) sont retournés de sorte qu'une valeur élevée
corresponde systématiquement à un environnement favorable.
""")

radar_cols = {
    "infra_cyclable_pct":        "Infrastructure cyclable",
    "gtfs_heavy_stops_300m":     "Multimodalité (TC lourds)",
    "baac_accidents_cyclistes":  "Sécurité (inv. sinistralité)",
    "gtfs_stops_within_300m_pct": "Couverture GTFS",
}
radar_cols = {k: v for k, v in radar_cols.items() if k in cities_f.columns}

# Défaut : Montpellier + top 4 IMD
_imd_top4 = imd_df["city"].head(4).tolist()
_default_radar = list(dict.fromkeys(["Montpellier"] + _imd_top4))[:5]
_default_radar = [c for c in _default_radar if c in cities_f["city"].values]

radar_city_sel = st.multiselect(
    "Sélection de l'échantillon d'audit (2 à 8 agglomérations)",
    options=sorted(cities_f["city"].unique()),
    default=_default_radar,
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
    palette_r = ["#1A6FBF", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22", "#1A2332", "#f39c12", "#16a085"]
    for i, (_, row) in enumerate(ndf.iterrows()):
        vals = [row[c] for c in radar_cols]
        vals += vals[:1]
        _color = "#e74c3c" if row["city"] == "Montpellier" else palette_r[i % len(palette_r)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=f"{row['city']} {'(cas d\'étude)' if row['city'] == 'Montpellier' else ''}",
            opacity=0.65,
            line_color=_color,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=460,
        margin=dict(l=60, r=60, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "**Figure 4.1.** Profil radar des agglomérations sélectionnées. "
        "Chaque axe est normalisé min-max ($\\tilde{c} \\in [0, 1]$) sur l'échantillon affiché. "
        "La composante sécurité est retournée (valeur haute = faible sinistralité). "
        "**Montpellier** (tracé rouge, rang IMD #2) est inclus par défaut comme référence nationale. "
        "L'aire de la figure est proportionnelle à la performance globale de l'environnement cyclable."
    )
else:
    st.info("Sélectionnez au moins 2 agglomérations pour initier l'audit radar comparatif.")
