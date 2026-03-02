"""
14_Flux_Villes.py — Visualisation des flux VLS collectés en temps réel via GBFS station_status.

Ce module lit les snapshots stockés dans data/status_snapshots/ par le script
scripts/collect_status.py et affiche les pseudo-flux inter-snapshots.

Si aucune donnée n'est encore collectée, il propose de lancer la collecte
et affiche un aperçu live (snapshot unique sans historique de flux).
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.styles import abstract_box, inject_css, section, sidebar_nav
from utils.gbfs_collector import (
    GBFSCollector,
    PRIORITY_SYSTEMS,
    compute_pseudo_flows,
)

st.set_page_config(
    page_title="Flux VLS Temps Réel - GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

_ROOT     = Path(__file__).parent.parent
_SNAP_DIR = _ROOT / "data" / "status_snapshots"

# ── Collecteur (lecture seule ici) ─────────────────────────────────────────────
@st.cache_resource
def get_collector() -> GBFSCollector:
    return GBFSCollector()

collector = get_collector()

# ── État des données collectées ───────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_available() -> pd.DataFrame:
    return collector.list_available()

avail = get_available()
_n_systems = len(avail)
_has_data  = _n_systems > 0

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("Flux VLS par Snapshot GBFS — Comparaison Nationale")
st.caption("Module transversal : collecte station_status en temps réel sur les grandes villes françaises")

if _has_data:
    _date_range = f"{avail['date_debut'].min()} → {avail['date_fin'].max()}"
    _n_snap_total = avail["n_files"].sum()
else:
    _date_range = "aucune donnée collectée"
    _n_snap_total = 0

abstract_box(
    "Ce module exploite les flux <b>GBFS station_status</b> — la couche temps-réel "
    "du standard General Bikeshare Feed Specification — pour reconstituer les "
    "<b>pseudo-flux de disponibilité</b> dans les grandes villes françaises. "
    "Contrairement aux données Vélomagg (courses individuelles issues de la TAM), "
    "les pseudo-flux sont inférés des variations de disponibilité entre snapshots "
    "consécutifs : &Delta;bikes &lt; 0 &rarr; départs estimés ; &Delta;bikes &gt; 0 "
    "&rarr; retours estimés. La résolution temporelle dépend de l'intervalle de collecte "
    "(recommandé : 60 s). Pour lancer la collecte, exécuter : "
    "<code>python scripts/collect_status.py</code>",
    findings=[
        (str(_n_systems), "réseaux avec données"),
        (str(_n_snap_total), "fichiers collectés"),
        (_date_range, "période couverte"),
        ("&ge; 20 villes", "réseaux cibles GBFS"),
    ],
)

sidebar_nav()
with st.sidebar:
    st.header("Paramètres")

    # Sélection du système
    all_sys_ids = PRIORITY_SYSTEMS
    collected_ids = avail["system_id"].tolist() if _has_data else []

    if _has_data:
        default_sys = collected_ids[0] if collected_ids else all_sys_ids[0]
    else:
        default_sys = "Paris"

    system_sel = st.selectbox(
        "Réseau à analyser",
        options=all_sys_ids,
        index=all_sys_ids.index(default_sys) if default_sys in all_sys_ids else 0,
        help="Seuls les réseaux avec données collectées affichent les flux historiques.",
    )

    multi_sel = st.multiselect(
        "Comparaison multi-réseaux",
        options=collected_ids if collected_ids else all_sys_ids,
        default=collected_ids[:5] if len(collected_ids) >= 2 else [],
        max_selections=8,
    )

    show_live = st.checkbox(
        "Afficher un snapshot live (appel API direct)",
        value=not _has_data,
        help="Récupère le status actuel sans historique. Utile pour tester avant collecte.",
    )
    st.divider()
    st.caption("R. Fossé & G. Pallares · 2025–2026")

# ── Démarrer la collecte ──────────────────────────────────────────────────────
if not _has_data:
    st.info(
        "Aucune donnée collectée. Pour commencer la collecte, ouvrez un terminal "
        "dans le répertoire du projet et exécutez :"
    )
    st.code(
        "# Tester quels réseaux sont compatibles station_status\n"
        "python scripts/test_status_feeds.py\n\n"
        "# Lancer la collecte continue (toutes les 60 s, 8 h)\n"
        "python scripts/collect_status.py --interval 60 --duration 28800\n\n"
        "# Collecte sur Paris + Lyon uniquement (3 snapshots de test)\n"
        "python scripts/collect_status.py --systems Paris lyon --interval 30 --max-iter 3",
        language="bash",
    )

# ── Section 1 — Inventaire des données collectées ─────────────────────────────
st.divider()
section(1, "Inventaire des Données Collectées")

if _has_data:
    cat = collector._catalog[["system_id", "city", "n_stations"]].copy()
    avail_enriched = avail.merge(cat, on="system_id", how="left")
    avail_enriched = avail_enriched.sort_values("n_files", ascending=False)

    fig_inv = px.bar(
        avail_enriched,
        x="system_id", y="n_files",
        color="n_stations",
        color_continuous_scale="Blues",
        text="date_debut",
        labels={"system_id": "Réseau", "n_files": "Journées collectées",
                "n_stations": "N stations"},
        height=320,
    )
    fig_inv.update_traces(textposition="outside", textfont=dict(size=9))
    fig_inv.update_layout(
        plot_bgcolor="white",
        xaxis_tickangle=-35,
        margin=dict(l=10, r=10, t=10, b=80),
        showlegend=True,
    )
    st.plotly_chart(fig_inv, use_container_width=True)
    st.caption(
        "**Figure 1.1.** Journées de données collectées par réseau. "
        "La couleur encode la taille du réseau (nombre de stations). "
        "Chaque journée correspond à un fichier Parquet dans data/status_snapshots/."
    )
    st.dataframe(
        avail_enriched[["system_id", "city", "n_stations", "n_files", "date_debut", "date_fin"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Lancez le script de collecte pour alimenter cette section.")

# ── Section 2 — Snapshot live ─────────────────────────────────────────────────
st.divider()
section(2, "Snapshot Temps Réel — Disponibilité Actuelle")

if show_live:
    with st.spinner(f"Récupération du snapshot live pour {system_sel}..."):
        targets = collector.get_target_systems()
        row_sel = targets[targets["system_id"] == system_sel]

        if row_sel.empty:
            st.warning(f"Système '{system_sel}' introuvable dans le catalogue.")
        else:
            row = row_sel.iloc[0]
            snap_live = collector.take_snapshot(str(row["system_id"]), str(row["gbfs_url"]))
            info_df   = collector.load_station_info(str(row["system_id"]), str(row["gbfs_url"]))

            if snap_live.empty:
                st.warning(
                    f"Le réseau **{system_sel}** n'expose pas de feed `station_status`. "
                    "Essayez `python scripts/test_status_feeds.py` pour identifier les réseaux compatibles."
                )
            else:
                if not info_df.empty:
                    snap_live = snap_live.merge(
                        info_df.rename(columns={"name": "station_name"}),
                        on="station_id", how="left",
                    )

                n_renting  = int(snap_live["is_renting"].sum())  if "is_renting" in snap_live.columns else "?"
                total_bikes = int(snap_live["num_bikes_available"].sum())
                total_docks = int(snap_live["num_docks_available"].sum())
                fill_rate   = total_bikes / max(total_bikes + total_docks, 1) * 100

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Stations actives", f"{len(snap_live)}")
                col_s2.metric("Vélos disponibles", f"{total_bikes:,}")
                col_s3.metric("Places libres", f"{total_docks:,}")
                col_s4.metric("Taux de remplissage", f"{fill_rate:.1f}%")

                # Distribution des disponibilités
                fig_live = px.histogram(
                    snap_live,
                    x="num_bikes_available",
                    nbins=30,
                    color_discrete_sequence=["#1A6FBF"],
                    labels={"num_bikes_available": "Vélos disponibles par station"},
                    height=300,
                )
                fig_live.update_layout(
                    plot_bgcolor="white",
                    xaxis_title="Vélos disponibles (snapshot actuel)",
                    yaxis_title="Stations",
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_live, use_container_width=True)

                if "lat" in snap_live.columns and snap_live["lat"].notna().any():
                    st.markdown("**Carte de disponibilité (snapshot actuel)**")
                    map_df = snap_live.dropna(subset=["lat", "lon"]).copy()
                    map_df["lat"] = map_df["lat"].astype(float)
                    map_df["lon"] = map_df["lon"].astype(float)
                    fig_map = px.scatter_mapbox(
                        map_df,
                        lat="lat", lon="lon",
                        size="num_bikes_available",
                        color="num_bikes_available",
                        color_continuous_scale="RdYlGn",
                        size_max=20,
                        zoom=12,
                        mapbox_style="carto-positron",
                        hover_name="station_name" if "station_name" in map_df.columns else "station_id",
                        hover_data={"num_bikes_available": True, "num_docks_available": True,
                                    "lat": False, "lon": False},
                        labels={"num_bikes_available": "Vélos dispo"},
                        height=500,
                    )
                    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.caption(
                        f"**Figure 2.1.** Disponibilité en temps réel — {system_sel} "
                        f"({datetime.now(timezone.utc).strftime('%H:%M UTC')}). "
                        "Vert = stations bien remplies, rouge = stations vides ou quasi-vides. "
                        "Taille des points proportionnelle au nombre de vélos disponibles."
                    )
                else:
                    st.info(
                        "Coordonnées GPS non chargées pour ce réseau. "
                        "Vérifiez que station_information.json est accessible."
                    )
else:
    st.info("Cochez 'Afficher un snapshot live' dans la barre latérale pour voir la disponibilité actuelle.")

# ── Section 3 — Flux historiques ──────────────────────────────────────────────
st.divider()
section(3, "Pseudo-Flux Horaires — Reconstruction depuis les Snapshots")

st.markdown(r"""
Les **pseudo-flux** sont estimés à partir de la variation de disponibilité entre snapshots consécutifs :

$$\hat{d}_i(t) = \max(0,\; b_i(t-1) - b_i(t)), \quad \hat{a}_i(t) = \max(0,\; b_i(t) - b_i(t-1))$$

où $\hat{d}$ est le nombre de **départs estimés** et $\hat{a}$ le nombre d'**arrivées estimées**
à la station $i$ entre les snapshots $t-1$ et $t$.

Ces estimations sont des **minorants** : si deux usagers partent et trois reviennent simultanément
entre deux snapshots, seul le flux net (+1) est observé.
""")

if _has_data and system_sel in collected_ids:
    with st.spinner(f"Chargement des flux pour {system_sel}..."):
        flows_df = collector.load_flows(system_sel)

    if not flows_df.empty:
        flows_df["hour"] = pd.to_datetime(flows_df["fetched_at"]).dt.hour

        # Agrégation horaire
        hourly_flows = flows_df.groupby("hour").agg(
            departures=("departures_est", "sum"),
            arrivals=("arrivals_est", "sum"),
            net_flow=("net_flow_est", "mean"),
            n_stations=("station_id", "nunique"),
        ).reset_index()

        fig_flows = go.Figure()
        fig_flows.add_trace(go.Bar(
            x=hourly_flows["hour"], y=hourly_flows["departures"],
            name="Départs estimés",
            marker_color="#C0392B", opacity=0.8,
        ))
        fig_flows.add_trace(go.Bar(
            x=hourly_flows["hour"], y=hourly_flows["arrivals"],
            name="Arrivées estimées",
            marker_color="#1E8449", opacity=0.8,
        ))
        fig_flows.add_trace(go.Scatter(
            x=hourly_flows["hour"], y=hourly_flows["net_flow"],
            name="Flux net moyen (par station)",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#D96B27", width=2.5),
        ))
        fig_flows.update_layout(
            plot_bgcolor="white",
            barmode="group",
            xaxis=dict(title="Heure", dtick=2),
            yaxis=dict(title="Pseudo-flux cumulés (stations)"),
            yaxis2=dict(title="Flux net moyen", overlaying="y", side="right",
                        showgrid=False),
            legend=dict(x=0.02, y=0.98),
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_flows, use_container_width=True)
        st.caption(
            f"**Figure 3.1.** Pseudo-flux horaires pour le réseau **{system_sel}** "
            f"({flows_df['fetched_at'].min()} → {flows_df['fetched_at'].max()}). "
            "Barres rouges = départs estimés, vertes = arrivées estimées. "
            "Ligne orange = flux net moyen par station (axe droit). "
            "Ces profils sont directement comparables à ceux de Montpellier (données TAM)."
        )

        # Top stations les plus actives
        top_stations = (
            flows_df.groupby("station_id")
            .agg(
                total_dep=("departures_est", "sum"),
                total_arr=("arrivals_est", "sum"),
                total_activity=("departures_est", lambda x: x.sum() + flows_df.loc[x.index, "arrivals_est"].sum()),
            )
            .sort_values("total_dep", ascending=False)
            .head(15)
            .reset_index()
        )

        if not top_stations.empty:
            # Enrichir avec les noms de stations si dispo
            info_df = collector._info_cache.get(system_sel, pd.DataFrame())
            if not info_df.empty:
                top_stations = top_stations.merge(
                    info_df[["station_id", "name"]], on="station_id", how="left"
                )
                top_stations["label"] = top_stations["name"].fillna(top_stations["station_id"])
            else:
                top_stations["label"] = top_stations["station_id"]

            top_stations = top_stations.sort_values("total_dep", ascending=True)
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(
                y=top_stations["label"], x=top_stations["total_dep"],
                orientation="h", name="Départs", marker_color="#C0392B", opacity=0.8,
            ))
            fig_top.add_trace(go.Bar(
                y=top_stations["label"], x=top_stations["total_arr"],
                orientation="h", name="Arrivées", marker_color="#1E8449", opacity=0.8,
            ))
            fig_top.update_layout(
                plot_bgcolor="white",
                barmode="group",
                xaxis_title="Pseudo-flux cumulés",
                height=max(350, len(top_stations) * 25),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(x=0.65, y=0.02),
            )
            st.plotly_chart(fig_top, use_container_width=True)
            st.caption(
                f"**Figure 3.2.** Top 15 stations par volume d'activité estimé ({system_sel}). "
                "Les stations à fort déséquilibre (barres rouge ≫ verte ou inverse) "
                "sont les candidates prioritaires à la redistribution."
            )
    else:
        st.info(f"Aucun flux calculable pour {system_sel} (moins de 2 snapshots).")
else:
    st.info(
        f"Données historiques indisponibles pour **{system_sel}**. "
        "Exécutez le script de collecte ou activez le snapshot live."
    )

# ── Section 4 — Comparaison multi-réseaux ─────────────────────────────────────
st.divider()
section(4, "Comparaison Multi-Réseaux — Profils de Flux Agrégés")

if len(multi_sel) >= 2:
    with st.spinner("Chargement des flux multi-réseaux..."):
        all_flows = []
        for sid in multi_sel:
            f = collector.load_flows(sid)
            if not f.empty:
                all_flows.append(f)

    if len(all_flows) >= 2:
        combined = pd.concat(all_flows, ignore_index=True)
        combined["hour"] = pd.to_datetime(combined["fetched_at"]).dt.hour

        hourly_by_sys = combined.groupby(["system_id", "hour"]).agg(
            departures=("departures_est", "sum"),
            arrivals=("arrivals_est", "sum"),
        ).reset_index()

        # Normaliser par nombre de stations pour comparaison équitable
        sys_sizes = {
            sid: max(combined[combined["system_id"] == sid]["station_id"].nunique(), 1)
            for sid in multi_sel
        }
        hourly_by_sys["dep_per_station"] = hourly_by_sys.apply(
            lambda r: r["departures"] / sys_sizes.get(r["system_id"], 1), axis=1
        )

        fig_comp = px.line(
            hourly_by_sys,
            x="hour", y="dep_per_station",
            color="system_id",
            markers=True,
            labels={
                "hour": "Heure",
                "dep_per_station": "Départs estimés / station",
                "system_id": "Réseau",
            },
            height=420,
        )
        fig_comp.update_layout(
            plot_bgcolor="white",
            xaxis=dict(dtick=2, title="Heure de la journée"),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption(
            "**Figure 4.1.** Profils horaires de départs estimés normalisés par station, "
            "pour chaque réseau sélectionné. La normalisation permet de comparer des réseaux "
            "de tailles très différentes. Des pics à des heures différentes indiquent des "
            "comportements de mobilité distincts selon les villes."
        )
    else:
        st.info("Données insuffisantes pour la comparaison (moins de 2 réseaux avec des flux).")
elif not _has_data:
    st.info(
        "La comparaison multi-réseaux sera disponible une fois des données collectées "
        "pour plusieurs systèmes."
    )
else:
    st.info("Sélectionnez au moins 2 réseaux dans la barre latérale pour activer la comparaison.")

# ── Section 5 — Guide de collecte ────────────────────────────────────────────
st.divider()
section(5, "Guide de Collecte — Mise en Place et Planification")

st.markdown("""
### Architecture de collecte

```
scripts/
├── test_status_feeds.py   ← Étape 1 : diagnostiquer les réseaux compatibles
└── collect_status.py      ← Étape 2 : lancer la collecte continue

data/status_snapshots/
├── feed_diagnostic.csv    ← Résultats du diagnostic
├── Paris/
│   ├── station_info.parquet    ← Infos statiques (nom, lat, lon, capacité)
│   ├── 2026-03-01.parquet      ← Snapshots du 01/03/2026
│   └── 2026-03-02.parquet      ← Snapshots du 02/03/2026
├── lyon/
│   └── ...
└── ...
```

### Schéma d'un snapshot (Parquet)

| Colonne | Type | Description |
|---|---|---|
| `fetched_at` | datetime (UTC) | Horodatage du snapshot |
| `system_id` | str | Identifiant du réseau |
| `station_id` | str | Identifiant de la station |
| `num_bikes_available` | int | Vélos disponibles |
| `num_docks_available` | int | Places libres |
| `is_renting` | bool | Station en service (location) |
| `is_returning` | bool | Station en service (retour) |

### Commandes recommandées
""")

col_cmd1, col_cmd2 = st.columns(2)

with col_cmd1:
    st.code(
        "# 1. Diagnostic des feeds compatibles\n"
        "python scripts/test_status_feeds.py\n\n"
        "# 2. Test rapide (3 snapshots)\n"
        "python scripts/collect_status.py \\\n"
        "  --systems Paris lyon toulouse \\\n"
        "  --interval 30 --max-iter 3",
        language="bash",
    )

with col_cmd2:
    st.code(
        "# 3. Collecte journalière (8h, toutes les 60s)\n"
        "python scripts/collect_status.py \\\n"
        "  --interval 60 --duration 28800\n\n"
        "# 4. Collecte longue en arrière-plan (Windows)\n"
        "start /B python scripts/collect_status.py \\\n"
        "  --interval 90 > logs/collect.log 2>&1",
        language="bash",
    )

st.caption(
    "**Tableau 5.1.** Pour obtenir des profils de flux comparables à Montpellier (TAM), "
    "une collecte minimale de **48 heures** à un intervalle de **60 secondes** est recommandée, "
    "couvrant au moins 2 jours ouvrés complets. "
    "L'inférence des flux est d'autant plus précise que l'intervalle est court. "
    "— **R. Fossé & G. Pallares · 2025–2026**"
)
