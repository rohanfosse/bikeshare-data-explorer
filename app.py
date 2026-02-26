"""
app.py — Page d'introduction du tableau de bord de recherche.
Micromobilité française — Gold Standard GBFS — CESI BikeShare-ICT 2025-2026.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import METRICS, completeness_report, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Micromobilité française — Tableau de bord de recherche",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Micromobilité française — Gold Standard GBFS\n"
            "Pipeline d'enrichissement spatial · CESI BikeShare-ICT 2025-2026"
        ),
    },
)

inject_css()

st.title("Micromobilité française — Tableau de bord de recherche")
st.caption(
    "Gold Standard GBFS · Pipeline d'enrichissement spatial (Notebook 27) · "
    "CESI BikeShare-ICT 2025-2026"
)

abstract_box(
    "Ce tableau de bord présente les résultats du pipeline d'enrichissement spatial "
    "appliqué aux 46 000+ stations de vélos en libre-service (VLS) françaises auditées "
    "dans le cadre du projet BikeShare-ICT (CESI, 2025-2026). "
    "À partir des données GBFS collectées auprès de 122 systèmes nationaux, "
    "chaque station est enrichie selon cinq modules thématiques calculés dans un rayon de 300 m : "
    "topographie, continuité cyclable, accidentologie, multimodalité et comblement OSM. "
    "Le corpus ainsi constitué — désigné <em>Gold Standard GBFS</em> — "
    "sert de base à l'élaboration d'un Indice de Mobilité Douce (IMD) "
    "et à une série d'analyses comparatives à l'échelle nationale et locale (Montpellier / Vélomagg)."
)

df = load_stations()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    n_total_s  = len(df)
    n_cities_s = df["city"].nunique()
    st.markdown(
        f"**Gold Standard GBFS**  \n"
        f"`{n_total_s:,}` stations · {n_cities_s} villes  \n"
        f"Recherche CESI 2025-2026"
    )

# ── Section 1 — Questions de recherche ───────────────────────────────────────
section(1, "Questions de recherche et objectifs scientifiques")

st.markdown(
    """
1. **Couverture et qualité des données GBFS françaises** — Dans quelle mesure les flux GBFS officiels couvrent-ils l'offre nationale de VLS, et quelles lacunes l'enrichissement OSM permet-il de combler ?
2. **Différenciation spatiale des conditions cyclables** — Comment les métriques d'infrastructure, de sécurité et d'accessibilité multimodale varient-elles entre les agglomérations françaises ?
3. **Indice de Mobilité Douce (IMD)** — Un indice composite peut-il synthétiser les conditions objectives de pratique du vélo en ville et se corréler avec des indicateurs de perception (FUB Baromètre) ?
4. **Dynamiques d'usage — cas de Montpellier** — Quels patterns temporels, spatiaux et socio-économiques caractérisent l'usage du réseau Vélomagg, et comment optimiser la redistribution des vélos ?
    """
)

# ── Section 2 — Structure du tableau de bord ─────────────────────────────────
st.divider()
section(2, "Présentation des sept analyses — structure du tableau de bord")

pages_data = [
    ("Indice de Mobilité Douce (IMD)",
     "Classement composite des villes selon 4 dimensions normalisées (S, I, M, T). "
     "Validation croisée avec le FUB Baromètre 2023."),
    ("Carte des stations",
     "Visualisation géospatiale des 46 000+ stations. "
     "Coloration par métrique d'enrichissement (pydeck WebGL)."),
    ("Comparaison des villes",
     "Classement, profil radar et analyse de la relation "
     "infrastructure / accidentologie à l'échelle des agglomérations."),
    ("Distributions statistiques",
     "Distributions univariées, boîtes à moustaches inter-villes "
     "et matrice de corrélation de Spearman."),
    ("Mobilité nationale",
     "Croisement du catalogue GBFS (122 systèmes) avec FUB Baromètre, "
     "EMP 2019, BAAC, Cerema et Eco-compteurs."),
    ("Montpellier — Vélomagg",
     "Analyse approfondie : profils de stations, flux OD horaires, "
     "intégration multimodale tram-vélo, inégalités socio-économiques."),
    ("Export des données",
     "Filtrage, prévisualisation et téléchargement du Gold Standard GBFS "
     "en CSV ou Parquet."),
]

for name, description in pages_data:
    st.markdown(f"**{name}** — {description}")

# ── Section 3 — Corpus Gold Standard ─────────────────────────────────────────
st.divider()
section(3, "Corpus Gold Standard GBFS — 46 000+ stations enrichies dans 300 m")

n_total   = len(df)
n_cities  = df["city"].nunique()
n_systems = df["system_id"].nunique()
cap_total = int(df["capacity"].sum(skipna=True))
avg_cap   = df["capacity"].mean()
metric_cols = [k for k in METRICS if k in df.columns]
n_complete  = int(df[metric_cols].dropna().shape[0])

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Stations", f"{n_total:,}")
c2.metric("Villes", f"{n_cities}")
c3.metric("Réseaux GBFS", f"{n_systems}")
c4.metric("Capacité totale", f"{cap_total:,}")
c5.metric("Capacité moyenne", f"{avg_cap:.1f}")
c6.metric("Enrichissement complet", f"{100*n_complete/n_total:.1f} %")

st.divider()

avg_infra      = df["infra_cyclable_pct"].mean()
pct_infra_gt50 = 100 * (df["infra_cyclable_pct"] > 50).mean()
avg_gtfs       = df["gtfs_heavy_stops_300m"].mean()
pct_tc_access  = 100 * (df["gtfs_heavy_stops_300m"] >= 1).mean()
avg_baac       = df["baac_accidents_cyclistes"].mean()
pct_no_acc     = 100 * (df["baac_accidents_cyclistes"] == 0).mean()
avg_elev       = df["elevation_m"].mean()

c7, c8, c9, c10, c11, c12, c13, c14 = st.columns(8)
c7.metric("Infra cyclable moy.", f"{avg_infra:.1f} %")
c8.metric("Stations infra > 50 %", f"{pct_infra_gt50:.1f} %")
c9.metric("TC lourd moy.", f"{avg_gtfs:.2f}")
c10.metric("Stations TC accessibles", f"{pct_tc_access:.1f} %")
c11.metric("Accidents moy. (300 m)", f"{avg_baac:.3f}")
c12.metric("Stations sans accident", f"{pct_no_acc:.1f} %")
c13.metric("Altitude moy. (m)", f"{avg_elev:.0f}")
c14.metric("Enrichissement complet", f"{100*n_complete/n_total:.1f} %")

# ── Section 4 — Pipeline d'enrichissement ────────────────────────────────────
st.divider()
section(4, "Pipeline d'enrichissement spatial — cinq modules thématiques")

with st.expander("Description détaillée des cinq modules", expanded=True):
    st.markdown(
        """
| Module | Axe | Colonnes produites | Source |
|:------:|:----|:-------------------|:-------|
| 1 | Comblement des zones blanches OSM | `source`, `osm_node_id` | OpenStreetMap |
| 2 | Topographie nationale (SRTM 30 m) | `elevation_m`, `topography_roughness_index` | Open-Elevation / SRTM |
| 3A | Continuité cyclable (cycleways OSM) | `infra_cyclable_km`, `infra_cyclable_pct` | OSM Overpass API |
| 3B | Sécurité — accidents cyclistes | `baac_accidents_cyclistes` | BAAC 2021-2023 (ONISR) |
| 4 | Multimodalité lourde (métro, tram, RER) | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | Flux GTFS nationaux |

**Stratégie** : traitement par lots avec requêtes HTTP asynchrones (`aiohttp`) et mise en cache locale. Rayon standard de **300 m** autour de chaque point de stationnement (standard *last-mile*).
        """
    )

# ── Section 5 — Statistiques descriptives ────────────────────────────────────
st.divider()
section(5, "Statistiques descriptives des sept métriques d'enrichissement")

left_stat, right_stat = st.columns([3, 2])

with left_stat:
    st.caption(
        "Statistiques calculées sur l'ensemble des stations disposant d'une valeur valide "
        "pour chaque métrique."
    )
    stat_rows = []
    for col_name, meta in METRICS.items():
        if col_name not in df.columns:
            continue
        s = df[col_name].dropna()
        stat_rows.append({
            "Métrique": meta["label"],
            "n":        f"{len(s):,}",
            "Moyenne":  round(s.mean(), 3),
            "Médiane":  round(s.median(), 3),
            "Éc. type": round(s.std(), 3),
            "Q25":      round(s.quantile(0.25), 3),
            "Q75":      round(s.quantile(0.75), 3),
            "Unité":    meta["unit"],
        })
    st.dataframe(
        pd.DataFrame(stat_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            k: st.column_config.NumberColumn(format="%.3f")
            for k in ["Moyenne", "Médiane", "Éc. type", "Q25", "Q75"]
        },
    )

with right_stat:
    st.caption(
        "Complétude : pourcentage de stations disposant d'une valeur valide "
        "pour chaque dimension d'enrichissement."
    )
    comp_df = completeness_report(df)
    fig_comp = px.bar(
        comp_df,
        x="Complétude (%)", y="Métrique",
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
    st.caption(
        "Figure 5.1. Taux de complétude par métrique d'enrichissement "
        "sur l'ensemble du corpus Gold Standard."
    )

# ── Section 6 — Définitions ───────────────────────────────────────────────────
st.divider()
section(6, "Glossaire — définitions et sources des métriques enrichies")

with st.expander("Consulter les définitions", expanded=False):
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
