"""
3_Distributions.py — Distributions et corrélations des métriques enrichies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, load_stations

st.set_page_config(
    page_title="Distributions · Gold Standard GBFS",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Distributions des métriques")

df = load_stations()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    all_cities = sorted(df["city"].unique())
    city_filter = st.multiselect(
        "Filtrer par ville(s)",
        options=all_cities,
        default=[],
        placeholder="Toutes les villes",
    )
    n_bins = st.slider("Nombre de bins (histogramme)", 20, 100, 40, 5)

dff = df[df["city"].isin(city_filter)] if city_filter else df
st.caption(f"**{len(dff):,}** stations · {dff['city'].nunique()} villes")

st.divider()

# ── Histogrammes (grille 2×3) ─────────────────────────────────────────────────
st.subheader("Histogrammes des métriques enrichies")

metric_keys = [k for k in METRICS if k in dff.columns]
cols = st.columns(2)

for i, mkey in enumerate(metric_keys):
    meta = METRICS[mkey]
    series = dff[mkey].dropna()
    if series.empty:
        continue

    fig = px.histogram(
        series,
        nbins=n_bins,
        labels={"value": meta["label"]},
        color_discrete_sequence=[
            "#2ecc71" if meta["higher_is_better"] is True
            else "#e74c3c" if meta["higher_is_better"] is False
            else "#3498db"
        ],
        height=280,
    )
    fig.update_layout(
        title=dict(text=meta["label"], font_size=14),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
        xaxis_title=f"{meta['label']} ({meta['unit']})",
        yaxis_title="Stations",
    )
    # Ligne médiane
    med = float(series.median())
    fig.add_vline(
        x=med, line_dash="dash", line_color="black", opacity=0.6,
        annotation_text=f"médiane {med:.2f}", annotation_position="top right",
    )
    cols[i % 2].plotly_chart(fig, use_container_width=True)

st.divider()

# ── Box-plots par ville (top 15) ──────────────────────────────────────────────
st.subheader("Comparaison inter-villes (box-plots)")

bp_metric = st.selectbox(
    "Métrique",
    options=[k for k in METRICS if k in dff.columns],
    format_func=lambda k: METRICS[k]["label"],
    key="bp_metric",
)

top15 = (
    dff.groupby("city")["uid"].count()
    .nlargest(15)
    .index.tolist()
)
bp_city_sel = st.multiselect(
    "Villes à comparer",
    options=sorted(dff["city"].unique()),
    default=top15[:10],
    key="bp_cities",
)

if bp_city_sel:
    bp_df = dff[dff["city"].isin(bp_city_sel) & dff[bp_metric].notna()]
    meta_bp = METRICS[bp_metric]

    # Trier par médiane
    order = (
        bp_df.groupby("city")[bp_metric].median()
        .sort_values(ascending=not meta_bp.get("higher_is_better", True))
        .index.tolist()
    )

    fig_bp = px.box(
        bp_df,
        x="city",
        y=bp_metric,
        color="city",
        category_orders={"city": order},
        labels={"city": "Ville", bp_metric: meta_bp["label"]},
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
else:
    st.info("Sélectionnez au moins une ville.")

st.divider()

# ── Matrice de corrélation ─────────────────────────────────────────────────────
st.subheader("Matrice de corrélation des métriques")

num_cols = [k for k in METRICS if k in dff.columns]
corr_df  = dff[num_cols].dropna(how="all").corr(method="spearman")

labels = [METRICS[c]["label"] for c in corr_df.columns]

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
    height=480,
    margin=dict(l=10, r=10, t=10, b=120),
    xaxis=dict(tickangle=-30),
)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption(
    "Corrélation de Spearman (rang). "
    "Rouge = corrélation positive, Bleu = corrélation négative."
)

st.divider()

# ── Scatter matriciel (pairplot) ──────────────────────────────────────────────
with st.expander("🔍 Scatter matriciel (pairplot) — peut être lent sur 46k points", expanded=False):
    sample_n = st.slider("Échantillon (stations)", 500, 5000, 2000, 500)
    pair_keys = st.multiselect(
        "Variables",
        options=num_cols,
        default=["infra_cyclable_pct", "baac_accidents_cyclistes", "gtfs_heavy_stops_300m"],
        format_func=lambda k: METRICS[k]["label"],
    )
    if len(pair_keys) >= 2:
        sample_df = dff[pair_keys + ["city"]].dropna().sample(
            min(sample_n, len(dff[pair_keys].dropna())), random_state=42
        )
        fig_pair = px.scatter_matrix(
            sample_df,
            dimensions=pair_keys,
            color="city",
            labels={k: METRICS[k]["label"] for k in pair_keys},
            height=600,
            opacity=0.4,
        )
        fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
        fig_pair.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.info("Sélectionnez au moins 2 variables.")
