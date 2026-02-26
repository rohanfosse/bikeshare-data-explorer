"""
app.py — Page d'accueil du tableau de bord Gold Standard GBFS France.
Pipeline d'enrichissement : notebooks/27_gold_standard_enrichment.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import METRICS, completeness_report, load_stations

st.set_page_config(
    page_title="Gold Standard GBFS — France",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Gold Standard GBFS — Pipeline d'enrichissement spatial\n"
            "Recherche BikeShare-ICT — CESI 2025-2026"
        ),
    },
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: .78rem; text-transform: uppercase;
                                    letter-spacing: .05em; color: #5a7a99; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_stations()

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("Gold Standard GBFS — Micromobilité française")
st.markdown(
    """
    Tableau de bord issu du **pipeline d'enrichissement spatial (notebook 27)**
    appliqué aux 46 000+ stations GBFS françaises auditées (notebook 20).
    Chaque station est enrichie selon cinq modules thématiques
    calculés dans un rayon standard de **300 m** autour du point de stationnement.
    """
)

with st.expander("Méthodologie du pipeline d'enrichissement", expanded=True):
    st.markdown(
        """
| Module | Axe d'enrichissement | Colonnes produites | Source de données |
|:------:|:---------------------|:-------------------|:------------------|
| 1 | Comblement des zones blanches OSM | `source`, `osm_node_id` | OpenStreetMap |
| 2 | Topographie nationale (SRTM 30 m) | `elevation_m`, `topography_roughness_index` | Open-Elevation / SRTM |
| 3A | Continuité cyclable (cycleways OSM) | `infra_cyclable_km`, `infra_cyclable_pct` | OSM Overpass API |
| 3B | Sécurité — accidents cyclistes | `baac_accidents_cyclistes` | BAAC 2021-2023 (ONISR) |
| 4 | Multimodalité lourde (métro, tram, RER) | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | Flux GTFS nationaux |

**Stratégie d'implémentation** : traitement par lots (*batch processing*) avec
requêtes HTTP asynchrones (`aiohttp`) et mise en cache locale.
Le rayon de 300 m correspond au standard last-mile pour l'analyse de la continuité des déplacements.
        """
    )

st.divider()

# ── KPI — volumétrie ──────────────────────────────────────────────────────────
st.markdown("#### Vue d'ensemble du corpus")

n_total    = len(df)
n_cities   = df["city"].nunique()
n_systems  = df["system_id"].nunique()
cap_total  = int(df["capacity"].sum(skipna=True))
avg_cap    = df["capacity"].mean()
n_complete = int(df[[k for k in METRICS if k in df.columns]].dropna().shape[0])

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Stations", f"{n_total:,}")
c2.metric("Villes", f"{n_cities}")
c3.metric("Réseaux GBFS", f"{n_systems}")
c4.metric("Capacité totale", f"{cap_total:,} places")
c5.metric("Capacité moyenne", f"{avg_cap:.1f} places")
c6.metric("Enrichissement complet", f"{100*n_complete/n_total:.1f} %")

# ── KPI — métriques d'enrichissement ─────────────────────────────────────────
st.markdown("#### Indicateurs d'accessibilité et de sécurité (rayon 300 m)")

avg_infra      = df["infra_cyclable_pct"].mean()
med_infra      = df["infra_cyclable_pct"].median()
pct_good_infra = 100 * (df["infra_cyclable_pct"] > 50).mean()
avg_gtfs       = df["gtfs_heavy_stops_300m"].mean()
pct_tc_access  = 100 * (df["gtfs_heavy_stops_300m"] >= 1).mean()
avg_baac       = df["baac_accidents_cyclistes"].mean()
pct_no_acc     = 100 * (df["baac_accidents_cyclistes"] == 0).mean()
avg_elev       = df["elevation_m"].mean()

c7, c8, c9, c10, c11, c12, c13, c14 = st.columns(8)
c7.metric("Infra cyclable moy.", f"{avg_infra:.1f} %")
c8.metric("Médiane infra cyclable", f"{med_infra:.1f} %")
c9.metric("Stations infra > 50 %", f"{pct_good_infra:.1f} %")
c10.metric("TC lourd moy. (arrêts)", f"{avg_gtfs:.2f}")
c11.metric("Stations TC accessibles", f"{pct_tc_access:.1f} %")
c12.metric("Accidents moy. (300 m)", f"{avg_baac:.3f}")
c13.metric("Stations sans accident", f"{pct_no_acc:.1f} %")
c14.metric("Altitude moy. (m)", f"{avg_elev:.0f} m")

st.divider()

# ── Résumé statistique ────────────────────────────────────────────────────────
left_stat, right_stat = st.columns([3, 2])

with left_stat:
    st.subheader("Résumé statistique des métriques enrichies")
    st.caption(
        "Statistiques descriptives calculées sur l'ensemble des stations "
        "disposant d'une valeur valide pour chaque métrique."
    )
    stat_rows = []
    for col_name, meta in METRICS.items():
        if col_name not in df.columns:
            continue
        s = df[col_name].dropna()
        stat_rows.append({
            "Métrique": meta["label"],
            "n": f"{len(s):,}",
            "Moyenne": round(s.mean(), 3),
            "Médiane": round(s.median(), 3),
            "Éc. type": round(s.std(), 3),
            "Q25": round(s.quantile(0.25), 3),
            "Q75": round(s.quantile(0.75), 3),
            "Unité": meta["unit"],
        })
    st.dataframe(
        pd.DataFrame(stat_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Moyenne":  st.column_config.NumberColumn(format="%.3f"),
            "Médiane":  st.column_config.NumberColumn(format="%.3f"),
            "Éc. type": st.column_config.NumberColumn(format="%.3f"),
            "Q25":      st.column_config.NumberColumn(format="%.3f"),
            "Q75":      st.column_config.NumberColumn(format="%.3f"),
        },
    )

with right_stat:
    st.subheader("Complétude des métriques enrichies")
    st.caption(
        "Pourcentage de stations disposant d'une valeur valide "
        "pour chaque dimension d'enrichissement."
    )
    comp_df = completeness_report(df)
    fig_comp = px.bar(
        comp_df,
        x="Complétude (%)",
        y="Métrique",
        orientation="h",
        color="Complétude (%)",
        color_continuous_scale="Blues",
        range_color=[0, 100],
        text="Complétude (%)",
        height=320,
    )
    fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_comp.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=50, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 115], title=""),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.divider()

# ── Distributions rapides ──────────────────────────────────────────────────────
st.subheader("Distribution des principales métriques — profil national")
st.caption(
    "Histogrammes en densité normalisée. La ligne pointillée indique la médiane nationale."
)

dist_keys = ["infra_cyclable_pct", "gtfs_heavy_stops_300m", "baac_accidents_cyclistes", "elevation_m"]
dist_cols = st.columns(len(dist_keys))

for i, mkey in enumerate(dist_keys):
    if mkey not in df.columns:
        continue
    meta = METRICS[mkey]
    series = df[mkey].dropna()
    med = float(series.median())

    color = (
        "#1A6FBF" if meta["higher_is_better"] is True
        else "#c0392b" if meta["higher_is_better"] is False
        else "#5a7a99"
    )
    fig = px.histogram(
        series,
        nbins=50,
        histnorm="density",
        color_discrete_sequence=[color],
        height=220,
        labels={"value": meta["label"]},
    )
    fig.add_vline(x=med, line_dash="dash", line_color="#1A2332", opacity=0.7,
                  annotation_text=f"Méd. {med:.2f}", annotation_position="top right")
    fig.update_layout(
        title=dict(text=meta["label"], font_size=12),
        showlegend=False,
        margin=dict(l=5, r=5, t=35, b=5),
        plot_bgcolor="white",
        xaxis_title=meta["unit"],
        yaxis_title="Densité",
    )
    dist_cols[i].plotly_chart(fig, use_container_width=True)

st.divider()

# ── Top villes + sources ──────────────────────────────────────────────────────
left_top, right_top = st.columns([3, 2])

with left_top:
    st.subheader("Top 15 villes — nombre de stations et infrastructure cyclable")
    st.caption(
        "Classement par volume de stations. "
        "La couleur indique la part moyenne d'infrastructure cyclable (300 m)."
    )
    top_cities = (
        df.groupby("city")
        .agg(
            n_stations=("uid", "count"),
            infra_pct=("infra_cyclable_pct", "mean"),
            gtfs=("gtfs_heavy_stops_300m", "mean"),
            accidents=("baac_accidents_cyclistes", "mean"),
        )
        .reset_index()
        .sort_values("n_stations", ascending=False)
        .head(15)
    )
    fig_top = px.bar(
        top_cities,
        x="city",
        y="n_stations",
        color="infra_pct",
        color_continuous_scale="Blues",
        labels={"city": "Ville", "n_stations": "Stations", "infra_pct": "Infra cyclable (%)"},
        text="n_stations",
        height=360,
    )
    fig_top.update_traces(textposition="outside")
    fig_top.update_layout(
        coloraxis_colorbar=dict(title="Infra (%)"),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_top, use_container_width=True)

with right_top:
    st.subheader("Provenance des données stations")
    st.caption(
        "Répartition des stations selon leur source d'origine "
        "(flux GBFS officiels ou complément OSM)."
    )
    src_counts = df["source_label"].value_counts().reset_index()
    src_counts.columns = ["Source", "Stations"]
    fig_pie = px.pie(
        src_counts,
        names="Source",
        values="Stations",
        color_discrete_sequence=["#1A6FBF", "#4A9FDF", "#A8CFEF"],
        hole=0.5,
        height=360,
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── Tableau des extrêmes ──────────────────────────────────────────────────────
st.subheader("Extrêmes par métrique — meilleures et moins bonnes villes")
st.caption(
    "Pour chaque dimension, la ville avec la meilleure valeur et celle "
    "avec la moins bonne (parmi les villes avec au moins 10 stations)."
)

city_agg = (
    df.groupby("city")
    .agg({k: "mean" for k in METRICS if k in df.columns} | {"uid": "count"})
    .rename(columns={"uid": "n_stations"})
    .query("n_stations >= 10")
    .reset_index()
)

extremes_rows = []
for mkey, meta in METRICS.items():
    if mkey not in city_agg.columns:
        continue
    sub = city_agg[["city", mkey]].dropna()
    if sub.empty:
        continue
    if meta["higher_is_better"] is True:
        best = sub.loc[sub[mkey].idxmax()]
        worst = sub.loc[sub[mkey].idxmin()]
    elif meta["higher_is_better"] is False:
        best = sub.loc[sub[mkey].idxmin()]
        worst = sub.loc[sub[mkey].idxmax()]
    else:
        continue
    extremes_rows.append({
        "Métrique": meta["label"],
        "Meilleure ville": best["city"],
        f"Valeur ({meta['unit']})": round(best[mkey], 3),
        "Moins bonne ville": worst["city"],
        f"Valeur ({meta['unit']}) ": round(worst[mkey], 3),
    })

st.dataframe(pd.DataFrame(extremes_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Définitions ───────────────────────────────────────────────────────────────
with st.expander("Définition des métriques enrichies", expanded=False):
    for col, meta in METRICS.items():
        directionality = ""
        if meta["higher_is_better"] is True:
            directionality = " — *valeur élevée favorable*"
        elif meta["higher_is_better"] is False:
            directionality = " — *valeur faible favorable*"
        st.markdown(
            f"**{meta['label']}** (`{col}`, {meta['unit']}){directionality}  \n"
            f"{meta['description']}"
        )
        st.markdown("---")

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
    st.markdown(
        "**Gold Standard GBFS**  \n"
        "Pipeline d'enrichissement spatial  \n"
        "Notebook 27 — CESI BikeShare-ICT  \n"
        f"`{n_total:,}` stations · {n_cities} villes"
    )
    st.caption("Recherche CESI 2025-2026")
