"""
2_Villes.py — Analyse comparative des villes sur les métriques enrichies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, city_stats, load_stations

st.set_page_config(
    page_title="Comparaison des villes — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)

st.title("Analyse comparative des villes")
st.markdown(
    "Classement et profil multi-dimensionnel des agglomérations françaises "
    "selon les métriques d'enrichissement spatial du Gold Standard GBFS."
)

df     = load_stations()
cities = city_stats(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    st.page_link("app.py",                   label="Accueil")
    st.page_link("pages/1_Carte.py",         label="Carte des stations")
    st.page_link("pages/2_Villes.py",        label="Comparaison des villes")
    st.page_link("pages/3_Distributions.py", label="Distributions statistiques")
    st.page_link("pages/4_Export.py",        label="Export des données")
    st.page_link("pages/5_Mobilite_France.py",    label="Indicateurs nationaux")
    st.page_link("pages/6_Montpellier.py",        label="Montpellier — Velomagg")
    st.divider()
    st.header("Paramètres")
    n_top = st.slider("Nombre de villes", min_value=5, max_value=40, value=20, step=5)
    metric_key = st.selectbox(
        "Métrique principale",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k]["label"],
        index=0,
    )
    min_stations = st.number_input(
        "Seuil minimum de stations", min_value=1, max_value=500, value=10
    )
    meta = METRICS[metric_key]
    st.divider()
    st.markdown(f"**{meta['label']}**")
    st.caption(meta["description"])

# ── Filtrage ──────────────────────────────────────────────────────────────────
cities_f = cities[cities["n_stations"] >= min_stations].copy()

if metric_key not in cities_f.columns:
    st.warning(f"Métrique `{metric_key}` absente des données agrégées.")
    st.stop()

ascending = not meta.get("higher_is_better", True)
cities_sorted = cities_f.dropna(subset=[metric_key]).sort_values(
    metric_key, ascending=ascending
)

# ── Tableau + classement ──────────────────────────────────────────────────────
col_tab, col_chart = st.columns([2, 3])

with col_tab:
    st.subheader(f"Top {n_top} — {meta['label']}")
    extra_cols = [
        c for c in ["infra_cyclable_pct", "baac_accidents_cyclistes", "gtfs_heavy_stops_300m"]
        if c != metric_key
    ]
    display = cities_sorted.head(n_top)[
        ["city", "n_stations", metric_key] + extra_cols
    ].rename(columns={
        "city": "Ville",
        "n_stations": "Stations",
        metric_key: meta["label"],
        "infra_cyclable_pct": "Infra cyclable (%)",
        "baac_accidents_cyclistes": "Accidents (moy.)",
        "gtfs_heavy_stops_300m": "TC lourds (moy.)",
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

with col_chart:
    st.subheader("Classement — barres horizontales")
    plot_df = cities_sorted.head(n_top).copy()

    fig = px.bar(
        plot_df,
        x=metric_key,
        y="city",
        orientation="h",
        color=metric_key,
        color_continuous_scale=meta["color_scale"],
        text=metric_key,
        labels={"city": "Ville", metric_key: meta["label"]},
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

st.divider()

# ── Scatter : infrastructure vs accidentologie ────────────────────────────────
st.subheader("Infrastructure cyclable et accidentologie")
st.caption(
    "Chaque point représente une ville. "
    "La taille est proportionnelle au nombre de stations ; "
    "la couleur indique l'accessibilité aux transports lourds (TC). "
    "Le quadrant idéal se situe en haut à gauche : "
    "forte infrastructure cyclable, faible sinistralité."
)

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
        "infra_cyclable_pct": "Infrastructure cyclable moyenne (%)",
        "baac_accidents_cyclistes": "Accidents cyclistes moyens (300 m)",
        "gtfs_heavy_stops_300m": "Arrêts TC lourds (moy.)",
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
    annotation_text="Moyenne accidents", annotation_position="right",
)
fig_sc.add_vline(
    x=float(scatter_df["infra_cyclable_pct"].mean()),
    line_dash="dot", line_color="#1A6FBF", opacity=0.5,
    annotation_text="Moyenne infra", annotation_position="top",
)
st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# ── Radar multi-villes ────────────────────────────────────────────────────────
st.subheader("Profil radar — comparaison multi-dimensionnelle")
st.caption(
    "Les valeurs sont normalisées entre 0 et 1 par métrique pour permettre "
    "la comparaison entre dimensions hétérogènes. "
    "Pour les accidents, la valeur est inversée (1 = moins d'accidents)."
)

radar_cols = {
    "infra_cyclable_pct": "Infra cyclable",
    "gtfs_heavy_stops_300m": "TC lourds",
    "baac_accidents_cyclistes": "Sécurité (inv.)",
    "gtfs_stops_within_300m_pct": "Couv. GTFS",
}
top_radar_cities = (
    cities_f.dropna(subset=list(radar_cols))
    .nlargest(5, "n_stations")["city"]
    .tolist()
)

radar_city_sel = st.multiselect(
    "Villes à comparer (2 à 8)",
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
else:
    st.info("Sélectionnez au moins 2 villes pour afficher le radar.")
