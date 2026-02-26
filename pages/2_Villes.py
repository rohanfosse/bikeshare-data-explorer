"""
2_Villes.py — Comparaison des villes sur les métriques enrichies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, city_stats, load_stations

st.set_page_config(
    page_title="Villes · Gold Standard GBFS",
    page_icon="🏙️",
    layout="wide",
)

st.title("🏙️ Comparaison des villes")

df     = load_stations()
cities = city_stats(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    n_top = st.slider("Nombre de villes", min_value=5, max_value=40, value=20, step=5)
    metric_key = st.selectbox(
        "Métrique principale",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k]["label"],
        index=0,
    )
    min_stations = st.number_input(
        "Nombre min. de stations", min_value=1, max_value=500, value=10
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

ascending = not meta.get("higher_is_better", True)  # trier du meilleur au moins bon
cities_sorted = cities_f.dropna(subset=[metric_key]).sort_values(
    metric_key, ascending=ascending
)

# ── Tableau récapitulatif ─────────────────────────────────────────────────────
col_tab, col_chart = st.columns([2, 3])

with col_tab:
    st.subheader(f"Top {n_top} villes — {meta['label']}")
    display = cities_sorted.head(n_top)[
        ["city", "n_stations", metric_key,
         "infra_cyclable_pct", "baac_accidents_cyclistes", "gtfs_heavy_stops_300m"]
    ].rename(columns={
        "city": "Ville",
        "n_stations": "Stations",
        metric_key: meta["label"],
        "infra_cyclable_pct": "Infra cyclable (%)",
        "baac_accidents_cyclistes": "Accidents (moy)",
        "gtfs_heavy_stops_300m": "TC lourds (moy)",
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
    st.subheader("Classement (barres horizontales)")
    plot_df = cities_sorted.head(n_top).copy()
    color_scale = meta["color_scale"]

    fig = px.bar(
        plot_df,
        x=metric_key,
        y="city",
        orientation="h",
        color=metric_key,
        color_continuous_scale=color_scale,
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

# ── Scatter : infra vs accidents ──────────────────────────────────────────────
st.subheader("Infrastructure cyclable vs. Accidentologie")
st.caption(
    "Chaque point est une ville. Taille ∝ nombre de stations. "
    "Idéalement : coin supérieur gauche (bonne infra, peu d'accidents)."
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
    hover_data={"n_stations": True, "infra_cyclable_pct": ":.2f", "baac_accidents_cyclistes": ":.3f"},
    labels={
        "infra_cyclable_pct": "Infrastructure cyclable moyenne (%)",
        "baac_accidents_cyclistes": "Accidents cyclistes moyens (300 m)",
        "gtfs_heavy_stops_300m": "TC lourds (moy.)",
    },
    size_max=40,
    height=480,
)
fig_sc.update_layout(
    plot_bgcolor="white",
    coloraxis_colorbar=dict(title="TC lourds"),
    margin=dict(l=10, r=10, t=10, b=10),
)
# Quadrant helper lines
fig_sc.add_hline(
    y=float(scatter_df["baac_accidents_cyclistes"].mean()),
    line_dash="dot", line_color="red", opacity=0.4,
    annotation_text="Moy. accidents", annotation_position="right",
)
fig_sc.add_vline(
    x=float(scatter_df["infra_cyclable_pct"].mean()),
    line_dash="dot", line_color="green", opacity=0.4,
    annotation_text="Moy. infra", annotation_position="top",
)
st.plotly_chart(fig_sc, use_container_width=True)

st.divider()

# ── Radar multi-villes ────────────────────────────────────────────────────────
st.subheader("Profil radar — comparaison multi-villes")

radar_cols = {
    "infra_cyclable_pct": "Infra cyclable",
    "gtfs_heavy_stops_300m": "TC lourds",
    "baac_accidents_cyclistes": "Accidents (inv.)",
    "gtfs_stops_within_300m_pct": "Couv. GTFS",
}
top_radar_cities = cities_f.dropna(subset=list(radar_cols)).nlargest(5, "n_stations")["city"].tolist()

radar_city_sel = st.multiselect(
    "Sélectionner les villes à comparer",
    options=sorted(cities_f["city"].unique()),
    default=top_radar_cities[:5],
    max_selections=8,
)

if len(radar_city_sel) >= 2:
    radar_df = cities_f[cities_f["city"].isin(radar_city_sel)].dropna(subset=list(radar_cols))

    # Normaliser 0-1 par colonne
    ndf = radar_df[list(radar_cols)].copy()
    for c in ndf.columns:
        rng = ndf[c].max() - ndf[c].min()
        ndf[c] = (ndf[c] - ndf[c].min()) / rng if rng else 0.0
    # Inverser les accidents (plus bas = mieux → normaliser inversé)
    if "baac_accidents_cyclistes" in ndf.columns:
        ndf["baac_accidents_cyclistes"] = 1 - ndf["baac_accidents_cyclistes"]

    ndf["city"] = radar_df["city"].values
    categories = list(radar_cols.values())

    fig_radar = go.Figure()
    for _, row in ndf.iterrows():
        vals = [row[c] for c in radar_cols]
        vals += vals[:1]  # fermer le polygone
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["city"],
            opacity=0.7,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=450,
        margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "Valeurs normalisées 0-1 par métrique. "
        "Pour 'Accidents (inv.)' : 1 = moins d'accidents = meilleur."
    )
else:
    st.info("Sélectionnez au moins 2 villes pour afficher le radar.")
