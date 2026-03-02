"""
14_Flux_Villes.py — Visualisation des flux VLS collectés en temps réel via GBFS station_status.

Ce module lit les snapshots stockés dans data/status_snapshots/ par le script
scripts/collect_status.py et affiche les pseudo-flux inter-snapshots.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("Flux VLS par Snapshot GBFS — Comparaison Nationale")
st.caption("Module transversal : collecte station_status en temps réel sur les grandes villes françaises")

# ── Collecteur ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_collector() -> GBFSCollector:
    return GBFSCollector()


@st.cache_data(ttl=60)
def get_available() -> pd.DataFrame:
    return get_collector().list_available()


@st.cache_resource
def get_collection_state() -> dict:
    """État partagé (mutable) du thread de collecte automatique."""
    return {
        "thread":          None,
        "running":         False,
        "started_at":      None,
        "n_iterations":    0,
        "last_collect_at": None,
        "last_error":      None,
        "interval":        60,
    }


def _bg_collect_loop(state: dict, collector: GBFSCollector) -> None:
    """Boucle de collecte exécutée dans un thread daemon."""
    targets = collector.get_target_systems()
    state["started_at"] = datetime.now(timezone.utc)
    while state["running"]:
        t0 = time.monotonic()
        try:
            collector.collect_and_save(targets)
            state["n_iterations"]    += 1
            state["last_collect_at"]  = datetime.now(timezone.utc)
            state["last_error"]       = None
        except Exception as exc:
            state["last_error"] = str(exc)
        # Attendre l'intervalle en vérifiant le flag toutes les 0.5 s
        sleep_end = time.monotonic() + max(0.0, state["interval"] - (time.monotonic() - t0))
        while state["running"] and time.monotonic() < sleep_end:
            time.sleep(0.5)
    state["running"] = False


def start_bg_collection(state: dict, collector: GBFSCollector, interval: int = 60) -> None:
    if state["running"]:
        return
    state.update(running=True, interval=interval, n_iterations=0, last_error=None)
    t = threading.Thread(
        target=_bg_collect_loop,
        args=(state, collector),
        daemon=True,
        name="gbfs-bg-collector",
    )
    state["thread"] = t
    t.start()


def stop_bg_collection(state: dict) -> None:
    state["running"] = False


@st.cache_data(ttl=300)
def load_raw_snapshots(system_id: str) -> pd.DataFrame:
    """Charge les snapshots bruts (avant flux) depuis data/status_snapshots/*/system_id.parquet."""
    if not _SNAP_DIR.exists():
        return pd.DataFrame()
    files = sorted(_SNAP_DIR.glob(f"*/{system_id}.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out["fetched_at"] = pd.to_datetime(out["fetched_at"])
    return out.sort_values("fetched_at")


try:
    collector     = get_collector()
    avail         = get_available()
    _n_systems    = len(avail)
    _has_data     = _n_systems > 0
    if _has_data:
        _date_range   = f"{avail['date_debut'].min()} → {avail['date_fin'].max()}"
        _n_snap_total = int(avail["n_files"].sum())
    else:
        _date_range   = "aucune donnée collectée"
        _n_snap_total = 0
except Exception as _exc:
    st.error(
        f"Impossible de charger le catalogue GBFS : {_exc}. "
        "Vérifiez que `data/gbfs_france/systems_catalog.csv` est bien présent."
    )
    sidebar_nav()
    st.stop()

abstract_box(
    "Ce module exploite les flux <b>GBFS station_status</b> — la couche temps-réel "
    "du standard General Bikeshare Feed Specification — pour reconstituer les "
    "<b>pseudo-flux de disponibilité</b> dans les grandes villes françaises. "
    "Contrairement aux données Vélomagg (courses individuelles issues de la TAM), "
    "les pseudo-flux sont inférés des variations de disponibilité entre snapshots "
    "consécutifs : &Delta;bikes &lt; 0 &rarr; départs estimés ; &Delta;bikes &gt; 0 "
    "&rarr; retours estimés. La résolution temporelle dépend de l'intervalle de collecte "
    "(recommandé : 60 s).",
    findings=[
        (str(_n_systems), "réseaux avec données"),
        (str(_n_snap_total), "fichiers collectés"),
        (_date_range, "période couverte"),
        (str(len(PRIORITY_SYSTEMS)), "réseaux cibles GBFS"),
    ],
)

sidebar_nav()
with st.sidebar:
    st.header("Paramètres")

    all_sys_ids   = PRIORITY_SYSTEMS
    collected_ids = avail["system_id"].tolist() if _has_data else []

    default_sys = collected_ids[0] if collected_ids else all_sys_ids[0]

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
        help="Récupère le status actuel sans historique.",
    )
    st.divider()

    # ── Collecte automatique en arrière-plan ──────────────────────────────────
    st.subheader("Collecte automatique")
    bg_state = get_collection_state()

    if bg_state["running"]:
        elapsed_h = (
            datetime.now(timezone.utc) - bg_state["started_at"]
        ).total_seconds() / 3600 if bg_state["started_at"] else 0.0
        st.success(f"En cours — {elapsed_h:.1f} h")
        st.metric("Itérations", bg_state["n_iterations"])
        if bg_state["last_collect_at"]:
            st.caption(
                f"Dernière collecte : "
                f"{bg_state['last_collect_at'].strftime('%H:%M:%S UTC')}"
            )
        if bg_state["last_error"]:
            st.error(f"Erreur : {bg_state['last_error'][:120]}")
        if st.button("Arrêter", use_container_width=True):
            stop_bg_collection(bg_state)
            st.rerun()
    else:
        bg_interval = st.number_input(
            "Intervalle (s)", min_value=30, max_value=300, value=60, step=10,
            help="Intervalle entre deux séries de snapshots.",
        )
        if st.button("Démarrer la collecte", use_container_width=True, type="primary"):
            start_bg_collection(bg_state, collector, interval=int(bg_interval))
            st.rerun()
        if bg_state["n_iterations"] > 0:
            st.info(f"Arrêtée après {bg_state['n_iterations']} itérations.")
        st.caption(
            "Le thread tourne tant que Streamlit est actif. "
            "Pour du h24 sans surveillance, utilisez le script PS1."
        )

    st.divider()
    if st.button("Rafraîchir les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("R. Fossé & G. Pallares · 2025–2026")

# ── Section 1 — Inventaire des données collectées ─────────────────────────────
st.divider()
section(1, "Inventaire des Données Collectées")

if _has_data:
    cat            = collector._catalog[["system_id", "city", "n_stations"]].copy()
    avail_enriched = avail.merge(cat, on="system_id", how="left")
    avail_enriched = avail_enriched.sort_values("n_files", ascending=False)

    total_stations = int(avail_enriched["n_stations"].sum())
    last_collect   = avail_enriched["date_fin"].max()
    jours_max      = int(avail_enriched["n_files"].max())
    coverage_pct   = round(_n_systems / len(PRIORITY_SYSTEMS) * 100)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Réseaux collectés", f"{_n_systems} / {len(PRIORITY_SYSTEMS)}",
              delta=f"{coverage_pct} % de couverture")
    m2.metric("Stations couvertes", f"{total_stations:,}")
    m3.metric("Fichiers Parquet", _n_snap_total)
    m4.metric("Dernière collecte", last_collect)

    # Graphique horizontal trié par taille de réseau (plus lisible)
    avail_sorted = avail_enriched.sort_values("n_stations", ascending=True)
    fig_inv = px.bar(
        avail_sorted,
        y="system_id",
        x="n_files",
        color="n_stations",
        color_continuous_scale="Blues",
        text="n_stations",
        orientation="h",
        labels={
            "system_id": "Réseau",
            "n_files":   "Journées collectées",
            "n_stations": "Stations",
        },
        height=max(300, len(avail_sorted) * 26),
    )
    fig_inv.update_traces(texttemplate="%{text} st.", textposition="outside", textfont=dict(size=9))
    fig_inv.update_layout(
        plot_bgcolor="white",
        coloraxis_colorbar=dict(title="Stations", thickness=12),
        margin=dict(l=10, r=90, t=20, b=20),
        xaxis_title="Journées collectées",
        yaxis_title="",
    )
    st.plotly_chart(fig_inv, use_container_width=True)
    st.caption(
        "**Figure 1.1.** Journées de données collectées par réseau (trié par taille). "
        "La couleur et l'annotation encodent la taille du réseau (nombre de stations)."
    )

    st.dataframe(
        avail_enriched[["system_id", "city", "n_stations", "n_files", "date_debut", "date_fin"]].rename(
            columns={
                "system_id":  "Réseau",
                "city":       "Ville",
                "n_stations": "Stations",
                "n_files":    "Journées",
                "date_debut": "Début",
                "date_fin":   "Fin",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Stations": st.column_config.NumberColumn(format="%d"),
            "Journées": st.column_config.ProgressColumn(
                format="%d j",
                min_value=0,
                max_value=jours_max,
            ),
        },
    )
else:
    st.info(
        "Aucune donnée collectée. Lancez la collecte avec :\n\n"
        "```\n.\\scripts\\run_collect.ps1 -Duration 28800\n```"
    )

# ── Section 2 — Snapshot live ─────────────────────────────────────────────────
st.divider()
section(2, "Snapshot Temps Réel — Disponibilité Actuelle")

if show_live:
    col_live_hd, col_live_btn = st.columns([6, 1])
    with col_live_btn:
        _do_refresh = st.button("Rafraîchir", key="btn_refresh_live")
    if _do_refresh:
        st.cache_data.clear()
        st.rerun()

    with st.spinner(f"Récupération du snapshot live pour {system_sel}..."):
        targets = collector.get_target_systems()
        row_sel = targets[targets["system_id"] == system_sel]

        if row_sel.empty:
            st.warning(f"Système '{system_sel}' introuvable dans le catalogue.")
        else:
            row       = row_sel.iloc[0]
            snap_ts   = datetime.now(timezone.utc)
            snap_live = collector.take_snapshot(str(row["system_id"]), str(row["gbfs_url"]))
            info_df   = collector.load_station_info(str(row["system_id"]), str(row["gbfs_url"]))

            if snap_live.empty:
                st.warning(
                    f"Le réseau **{system_sel}** n'expose pas de feed `station_status`. "
                    "Testez les réseaux compatibles avec `python scripts/test_status_feeds.py`."
                )
            else:
                with col_live_hd:
                    st.caption(
                        f"Snapshot pris à **{snap_ts.strftime('%H:%M:%S UTC')}** "
                        f"({snap_ts.strftime('%Y-%m-%d')})"
                    )

                if not info_df.empty:
                    snap_live = snap_live.merge(
                        info_df.rename(columns={"name": "station_name"}),
                        on="station_id", how="left",
                    )

                n_total     = len(snap_live)
                n_renting   = int(snap_live["is_renting"].sum()) if "is_renting" in snap_live.columns else n_total
                total_bikes = int(snap_live["num_bikes_available"].sum())
                total_docks = int(snap_live["num_docks_available"].sum())
                fill_rate   = total_bikes / max(total_bikes + total_docks, 1) * 100
                n_empty     = int((snap_live["num_bikes_available"] == 0).sum())
                n_full      = int((snap_live["num_docks_available"] == 0).sum())
                med_bikes   = float(snap_live["num_bikes_available"].median())
                q25 = float(snap_live["num_bikes_available"].quantile(0.25))
                q75 = float(snap_live["num_bikes_available"].quantile(0.75))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Stations actives", f"{n_renting} / {n_total}")
                c2.metric("Vélos disponibles", f"{total_bikes:,}")
                c3.metric("Places libres", f"{total_docks:,}")
                c4.metric("Taux de remplissage", f"{fill_rate:.1f} %")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Stations vides (0 vélo)", n_empty,
                          delta=f"{n_empty/n_total*100:.0f}% du réseau",
                          delta_color="inverse")
                c6.metric("Stations pleines (0 place)", n_full,
                          delta=f"{n_full/n_total*100:.0f}% du réseau",
                          delta_color="inverse")
                c7.metric("Médiane vélos / station", f"{med_bikes:.1f}")
                c8.metric("IQR disponibilité", f"[{q25:.0f} — {q75:.0f}]")

                col_hist, col_box = st.columns([3, 2])
                with col_hist:
                    fig_hist = px.histogram(
                        snap_live,
                        x="num_bikes_available",
                        nbins=30,
                        color_discrete_sequence=["#1A6FBF"],
                        labels={"num_bikes_available": "Vélos disponibles"},
                        height=300,
                    )
                    fig_hist.add_vline(
                        x=med_bikes, line_dash="dash", line_color="#D96B27",
                        annotation_text=f"Médiane {med_bikes:.1f}",
                        annotation_position="top right",
                    )
                    fig_hist.update_layout(
                        plot_bgcolor="white",
                        xaxis_title="Vélos disponibles par station",
                        yaxis_title="Nombre de stations",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col_box:
                    def _cat(row_s: pd.Series) -> str:
                        b = row_s["num_bikes_available"]
                        d = row_s["num_docks_available"]
                        if b == 0:   return "Vide"
                        if d == 0:   return "Pleine"
                        if b <= q25: return "Peu remplie"
                        if b >= q75: return "Bien remplie"
                        return "Équilibrée"

                    snap_live["statut"] = snap_live.apply(_cat, axis=1)
                    cat_counts = snap_live["statut"].value_counts().reset_index()
                    cat_counts.columns = ["Statut", "Stations"]
                    color_map = {
                        "Vide": "#C0392B", "Peu remplie": "#E67E22",
                        "Équilibrée": "#2980B9", "Bien remplie": "#27AE60", "Pleine": "#1A5276",
                    }
                    fig_pie = px.pie(
                        cat_counts, names="Statut", values="Stations",
                        color="Statut", color_discrete_map=color_map,
                        hole=0.45, height=300,
                    )
                    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
                    fig_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.caption(
                    f"**Figure 2.1.** Distribution de la disponibilité et répartition par statut — "
                    f"{system_sel} ({snap_ts.strftime('%H:%M UTC')}). "
                    "Seuils : vide = 0 vélo, pleine = 0 place, peu remplie = Q1, bien remplie ≥ Q3."
                )

                # Carte
                if "lat" in snap_live.columns and snap_live["lat"].notna().any():
                    map_df = snap_live.dropna(subset=["lat", "lon"]).copy()
                    map_df["lat"] = map_df["lat"].astype(float)
                    map_df["lon"] = map_df["lon"].astype(float)
                    map_df["pct_rempli"] = (
                        map_df["num_bikes_available"]
                        / (map_df["num_bikes_available"] + map_df["num_docks_available"]).replace(0, np.nan)
                        * 100
                    ).round(1)
                    fig_map = px.scatter_mapbox(
                        map_df,
                        lat="lat", lon="lon",
                        size="num_bikes_available",
                        color="pct_rempli",
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 100],
                        size_max=22,
                        zoom=12,
                        mapbox_style="carto-positron",
                        hover_name="station_name" if "station_name" in map_df.columns else "station_id",
                        hover_data={
                            "num_bikes_available":  True,
                            "num_docks_available":  True,
                            "pct_rempli":           True,
                            "lat": False, "lon": False,
                        },
                        labels={
                            "num_bikes_available": "Vélos dispo",
                            "num_docks_available": "Places libres",
                            "pct_rempli": "Remplissage (%)",
                        },
                        height=500,
                    )
                    fig_map.update_layout(
                        coloraxis_colorbar=dict(title="Rempli (%)", thickness=12),
                        margin=dict(l=0, r=0, t=0, b=0),
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.caption(
                        f"**Figure 2.2.** Carte de disponibilité en temps réel — {system_sel}. "
                        "La couleur encode le taux de remplissage (0 % = vide, 100 % = plein). "
                        "La taille des points est proportionnelle au nombre de vélos disponibles."
                    )
                else:
                    st.info("Coordonnées GPS non disponibles pour ce réseau.")

                # Stations critiques
                st.markdown("**Stations critiques (snapshot actuel)**")
                name_col = "station_name" if "station_name" in snap_live.columns else "station_id"
                snap_live["pct_rempli_st"] = (
                    snap_live["num_bikes_available"]
                    / (snap_live["num_bikes_available"] + snap_live["num_docks_available"]).replace(0, np.nan)
                    * 100
                ).round(1)
                display_cols = [name_col, "num_bikes_available", "num_docks_available", "pct_rempli_st"]

                col_crit1, col_crit2 = st.columns(2)
                _rename_crit = {
                    name_col: "Station",
                    "num_bikes_available": "Vélos dispo",
                    "num_docks_available": "Places libres",
                    "pct_rempli_st": "Remplissage (%)",
                }
                _cfg_crit = {
                    "Remplissage (%)": st.column_config.ProgressColumn(
                        format="%.0f %%", min_value=0, max_value=100
                    )
                }
                with col_crit1:
                    st.markdown("*Priorité redistribution entrante (stations vides)*")
                    st.dataframe(
                        snap_live.sort_values("num_bikes_available").head(8)[display_cols].rename(columns=_rename_crit),
                        use_container_width=True, hide_index=True, column_config=_cfg_crit,
                    )
                with col_crit2:
                    st.markdown("*Priorité redistribution sortante (stations saturées)*")
                    st.dataframe(
                        snap_live.sort_values("num_docks_available").head(8)[display_cols].rename(columns=_rename_crit),
                        use_container_width=True, hide_index=True, column_config=_cfg_crit,
                    )
                st.caption(
                    "**Tableau 2.1.** Stations nécessitant une intervention prioritaire de redistribution. "
                    "Gauche = stations à réapprovisionner. Droite = stations à décharger."
                )
else:
    st.info("Cochez 'Afficher un snapshot live' dans la barre latérale pour voir la disponibilité actuelle.")

# ── Section 3 — Flux historiques ──────────────────────────────────────────────
st.divider()
section(3, "Pseudo-Flux Horaires — Reconstruction depuis les Snapshots")

st.markdown(
    "Les **pseudo-flux** sont estimés à partir de la variation de disponibilité entre "
    "snapshots consécutifs. Pour la station _i_ entre les instants _t-1_ et _t_ :"
    "<br>&emsp;**Départs estimés** : max(0, b<sub>i</sub>(t-1) &minus; b<sub>i</sub>(t))"
    "<br>&emsp;**Arrivées estimées** : max(0, b<sub>i</sub>(t) &minus; b<sub>i</sub>(t-1))"
    "<br>Ces estimations sont des **minorants** : les flux simultanés s'annulent partiellement.",
    unsafe_allow_html=True,
)

if _has_data and system_sel in collected_ids:
    with st.spinner(f"Chargement des flux pour {system_sel}..."):
        flows_df = collector.load_flows(system_sel)

    if not flows_df.empty:
        flows_df["fetched_at"] = pd.to_datetime(flows_df["fetched_at"])
        flows_df["hour"] = flows_df["fetched_at"].dt.hour

        # Filtre optionnel par date (utile quand plusieurs jours de collecte)
        available_dates = sorted(flows_df["fetched_at"].dt.date.unique())
        if len(available_dates) > 1:
            sel_dates = st.multiselect(
                "Filtrer par date",
                options=available_dates,
                default=available_dates,
                format_func=lambda d: d.strftime("%d/%m/%Y"),
            )
            if sel_dates:
                flows_df = flows_df[flows_df["fetched_at"].dt.date.isin(sel_dates)]

        hourly_flows = flows_df.groupby("hour").agg(
            departures=("departures_est", "sum"),
            arrivals=("arrivals_est", "sum"),
            net_flow=("net_flow_est", "mean"),
            n_stations=("station_id", "nunique"),
        ).reset_index()
        hourly_flows["dep_per_station"] = (
            hourly_flows["departures"] / hourly_flows["n_stations"].replace(0, np.nan)
        ).round(2)

        peak_dep_h = int(hourly_flows["hour"].iloc[int(hourly_flows["departures"].to_numpy().argmax())])
        peak_arr_h = int(hourly_flows["hour"].iloc[int(hourly_flows["arrivals"].to_numpy().argmax())])
        total_dep  = int(hourly_flows["departures"].sum())
        total_arr  = int(hourly_flows["arrivals"].sum())
        imbalance  = abs(total_dep - total_arr) / max(total_dep + total_arr, 1) * 100

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Heure de pointe départs", f"{peak_dep_h:02d}h")
        s2.metric("Heure de pointe arrivées", f"{peak_arr_h:02d}h")
        s3.metric("Total départs estimés", f"{total_dep:,}")
        s4.metric("Déséquilibre global", f"{imbalance:.1f} %",
                  help="(|D−A| / (D+A)) × 100 — 0 % = réseau parfaitement équilibré")

        fig_flows = go.Figure()
        fig_flows.add_trace(go.Bar(
            x=hourly_flows["hour"], y=hourly_flows["departures"],
            name="Départs estimés", marker_color="#C0392B", opacity=0.85,
        ))
        fig_flows.add_trace(go.Bar(
            x=hourly_flows["hour"], y=hourly_flows["arrivals"],
            name="Arrivées estimées", marker_color="#1E8449", opacity=0.85,
        ))
        fig_flows.add_trace(go.Scatter(
            x=hourly_flows["hour"], y=hourly_flows["dep_per_station"],
            name="Départs / station", mode="lines+markers", yaxis="y2",
            line=dict(color="#D96B27", width=2.5), marker=dict(size=6),
        ))
        fig_flows.update_layout(
            plot_bgcolor="white",
            barmode="group",
            xaxis=dict(title="Heure de la journée", dtick=2, tickmode="linear"),
            yaxis=dict(title="Pseudo-flux cumulés"),
            yaxis2=dict(
                title="Départs / station",
                overlaying="y", side="right",
                showgrid=False, tickformat=".2f",
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
            height=420,
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig_flows, use_container_width=True)
        st.caption(
            f"**Figure 3.1.** Pseudo-flux horaires agrégés — {system_sel} "
            f"({flows_df['fetched_at'].min().strftime('%Y-%m-%d %H:%M')} → "
            f"{flows_df['fetched_at'].max().strftime('%Y-%m-%d %H:%M')}). "
            "Barres rouges = départs estimés, vertes = arrivées estimées. "
            "Ligne orange = départs normalisés par station (axe droit)."
        )

        # Top stations
        top_stations = (
            flows_df.groupby("station_id")
            .agg(total_dep=("departures_est", "sum"), total_arr=("arrivals_est", "sum"))
            .assign(imbalance=lambda d: d["total_dep"] - d["total_arr"])
            .sort_values("total_dep", ascending=False)
            .head(15)
            .reset_index()
        )

        if not top_stations.empty:
            info_cached = collector._info_cache.get(system_sel, pd.DataFrame())
            if not info_cached.empty and "name" in info_cached.columns:
                top_stations = top_stations.merge(
                    info_cached[["station_id", "name"]], on="station_id", how="left"
                )
                top_stations["label"] = top_stations["name"].fillna(top_stations["station_id"])
            else:
                top_stations["label"] = top_stations["station_id"]

            col_top, col_div = st.columns(2)

            with col_top:
                act_df = top_stations.sort_values("total_dep", ascending=True)
                fig_act = go.Figure()
                fig_act.add_trace(go.Bar(
                    y=act_df["label"], x=act_df["total_dep"],
                    orientation="h", name="Départs", marker_color="#C0392B", opacity=0.85,
                ))
                fig_act.add_trace(go.Bar(
                    y=act_df["label"], x=act_df["total_arr"],
                    orientation="h", name="Arrivées", marker_color="#1E8449", opacity=0.85,
                ))
                fig_act.update_layout(
                    plot_bgcolor="white", barmode="group",
                    xaxis_title="Pseudo-flux cumulés",
                    height=max(380, len(act_df) * 28),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text="Activité totale (Top 15)", font=dict(size=13)),
                    legend=dict(x=0.6, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
                )
                st.plotly_chart(fig_act, use_container_width=True)

            with col_div:
                div_df = (
                    top_stations.assign(abs_imbal=lambda d: d["imbalance"].abs())
                    .sort_values("abs_imbal", ascending=True)
                )
                colors_div = ["#C0392B" if v > 0 else "#1E8449" for v in div_df["imbalance"]]
                fig_div = go.Figure()
                fig_div.add_trace(go.Bar(
                    y=div_df["label"], x=div_df["imbalance"], orientation="h",
                    marker_color=colors_div, opacity=0.9,
                    hovertemplate="%{y}<br>Déséquilibre : %{x:+d}<extra></extra>",
                ))
                fig_div.add_vline(x=0, line_color="#333", line_width=1.5)
                fig_div.update_layout(
                    plot_bgcolor="white",
                    xaxis_title="Déséquilibre net (D − A)",
                    height=max(380, len(div_df) * 28),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title=dict(text="Déséquilibre (D − A)", font=dict(size=13)),
                    showlegend=False,
                )
                st.plotly_chart(fig_div, use_container_width=True)

            st.caption(
                f"**Figure 3.2.** Top 15 stations par activité totale ({system_sel}). "
                "Rouge = départs, vert = arrivées. Droite : déséquilibre net D−A — "
                "rouge = émetteur net, vert = attracteur net."
            )

        # Série temporelle brute — évolution de la disponibilité dans le temps
        with st.expander("Série temporelle brute — disponibilité d'une station"):
            raw_df = load_raw_snapshots(system_sel)
            if not raw_df.empty:
                # Trier les stations par variabilité décroissante (les plus actives en premier)
                station_var = (
                    raw_df.groupby("station_id")["num_bikes_available"]
                    .std()
                    .sort_values(ascending=False)
                )
                top_sids = list(station_var.index[:30])

                info_cached2 = collector._info_cache.get(system_sel, pd.DataFrame())
                if not info_cached2.empty and "name" in info_cached2.columns:
                    name_map = info_cached2.set_index("station_id")["name"].to_dict()
                    station_options = [name_map.get(sid, sid) for sid in top_sids]
                else:
                    station_options = top_sids
                sid_by_label = dict(zip(station_options, top_sids))

                sel_label = st.selectbox(
                    "Station (triées par variabilité décroissante — les plus actives d'abord)",
                    options=station_options,
                    key="sel_station_ts",
                )
                sel_station_id = sid_by_label[sel_label]

                st_raw = raw_df[raw_df["station_id"] == sel_station_id].copy()
                n_snaps   = len(st_raw)
                time_span = (st_raw["fetched_at"].max() - st_raw["fetched_at"].min()).total_seconds() / 3600
                st.caption(f"{n_snaps} snapshots sur {time_span:.1f} h")

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=st_raw["fetched_at"], y=st_raw["num_bikes_available"],
                    name="Vélos disponibles", mode="lines+markers",
                    line=dict(color="#1A6FBF", width=2), marker=dict(size=4),
                ))
                if "num_docks_available" in st_raw.columns:
                    fig_ts.add_trace(go.Scatter(
                        x=st_raw["fetched_at"], y=st_raw["num_docks_available"],
                        name="Places libres", mode="lines",
                        line=dict(color="#27AE60", width=2, dash="dot"),
                    ))
                    cap_series = st_raw["num_bikes_available"] + st_raw["num_docks_available"]
                else:
                    cap_series = st_raw["num_bikes_available"]
                cap_val = int(cap_series.max())
                if cap_val > 0:
                    fig_ts.add_hline(
                        y=cap_val, line_dash="dash", line_color="#AAA",
                        annotation_text=f"Capacité max estimée : {cap_val}",
                        annotation_position="bottom right",
                    )
                fig_ts.update_layout(
                    plot_bgcolor="white",
                    xaxis_title="Horodatage",
                    yaxis_title="Disponibilité",
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_ts, use_container_width=True)
                st.caption(
                    f"**Figure 3.3.** Série temporelle brute des snapshots — station {sel_label}. "
                    "Bleu = vélos disponibles, vert pointillé = places libres. "
                    "Les chutes soudaines suivies de remontées correspondent aux départs/retours réels. "
                    "La ligne grise indique la capacité maximale observée."
                )
            else:
                st.info("Aucun snapshot brut trouvé. Les données apparaîtront après la première collecte.")

        # Tableau heure par heure
        with st.expander("Tableau détaillé heure par heure"):
            tbl = hourly_flows.rename(columns={
                "hour": "Heure", "departures": "Départs", "arrivals": "Arrivées",
                "net_flow": "Flux net moy.", "n_stations": "Stations actives",
                "dep_per_station": "Départs / station",
            })
            st.dataframe(
                tbl, use_container_width=True, hide_index=True,
                column_config={
                    "Heure": st.column_config.NumberColumn(format="%02dh"),
                    "Départs": st.column_config.ProgressColumn(
                        format="%d", min_value=0, max_value=int(tbl["Départs"].max())),
                    "Arrivées": st.column_config.ProgressColumn(
                        format="%d", min_value=0, max_value=int(tbl["Départs"].max())),
                    "Flux net moy.": st.column_config.NumberColumn(format="%.2f"),
                    "Départs / station": st.column_config.NumberColumn(format="%.2f"),
                },
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
        combined["fetched_at"] = pd.to_datetime(combined["fetched_at"])
        combined["hour"] = combined["fetched_at"].dt.hour

        hourly_by_sys = combined.groupby(["system_id", "hour"]).agg(
            departures=("departures_est", "sum"),
            arrivals=("arrivals_est", "sum"),
        ).reset_index()

        sys_sizes = {
            sid: max(combined[combined["system_id"] == sid]["station_id"].nunique(), 1)
            for sid in multi_sel
        }
        hourly_by_sys["dep_per_station"] = hourly_by_sys.apply(
            lambda r: r["departures"] / sys_sizes.get(r["system_id"], 1), axis=1
        )
        hourly_by_sys["arr_per_station"] = hourly_by_sys.apply(
            lambda r: r["arrivals"] / sys_sizes.get(r["system_id"], 1), axis=1
        )

        # Classification automatique du profil de pic horaire
        def _classify_peak(sub: pd.DataFrame) -> str:
            morning = sub[(sub["hour"] >= 7) & (sub["hour"] <= 9)]["dep_per_station"].mean()
            evening = sub[(sub["hour"] >= 17) & (sub["hour"] <= 19)]["dep_per_station"].mean()
            noon    = sub[(sub["hour"] >= 11) & (sub["hour"] <= 13)]["dep_per_station"].mean()
            max_val = max(morning, evening, noon, 0.001)
            if max_val < 0.01:                           return "Faible activité"
            if morning >= evening * 1.3 and morning >= noon * 1.3: return "Pic matinal"
            if evening >= morning * 1.3 and evening >= noon * 1.3: return "Pic vespéral"
            if noon >= morning * 0.9 and noon >= evening * 0.9:    return "Pic méridien"
            return "Double pic (navetteurs)"

        summary_rows = []
        for sid in multi_sel:
            sub = hourly_by_sys[hourly_by_sys["system_id"] == sid]
            if sub.empty:
                continue
            peak_h   = int(sub["hour"].iloc[int(sub["dep_per_station"].to_numpy().argmax())])
            max_dep  = float(sub["dep_per_station"].max())
            mean_dep = float(sub["dep_per_station"].mean())
            total_d  = int(sub["departures"].sum())
            total_a  = int(sub["arrivals"].sum())
            disbal   = abs(total_d - total_a) / max(total_d + total_a, 1) * 100
            profil   = _classify_peak(sub)
            summary_rows.append({
                "Réseau":              sid,
                "Stations":            sys_sizes[sid],
                "Profil horaire":      profil,
                "Heure de pointe":     f"{peak_h:02d}h",
                "Max départs/station": round(max_dep, 2),
                "Moy départs/station": round(mean_dep, 2),
                "Total départs":       total_d,
                "Déséquilibre (%)":    round(disbal, 1),
            })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(
                df_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Total départs":       st.column_config.NumberColumn(format="%d"),
                    "Déséquilibre (%)":    st.column_config.NumberColumn(format="%.1f %%"),
                    "Max départs/station": st.column_config.NumberColumn(format="%.2f"),
                    "Moy départs/station": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption(
                "**Tableau 4.1.** Statistiques clés par réseau. "
                "Déséquilibre = |D−A|/(D+A) × 100. "
                "Profil horaire : classification automatique (matinal, vespéral, méridien, double pic)."
            )

        fig_comp = px.line(
            hourly_by_sys,
            x="hour", y="dep_per_station",
            color="system_id",
            markers=True,
            labels={
                "hour":            "Heure",
                "dep_per_station": "Départs estimés / station",
                "system_id":       "Réseau",
            },
            height=400,
        )
        fig_comp.update_layout(
            plot_bgcolor="white",
            xaxis=dict(dtick=2, title="Heure de la journée", tickmode="linear"),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption(
            "**Figure 4.1.** Profils horaires de départs estimés normalisés par station. "
            "La normalisation permet de comparer des réseaux de tailles très différentes. "
            "Des pics à des heures distinctes reflètent des rythmes de mobilité urbaine différents."
        )

        # Heatmap heure × réseau
        pivot = hourly_by_sys.pivot(index="system_id", columns="hour", values="dep_per_station").fillna(0)
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="Blues",
            labels={"x": "Heure", "y": "Réseau", "color": "Départs / station"},
            aspect="auto",
            height=max(250, len(pivot) * 38),
        )
        fig_heat.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title="Dép./st.", thickness=12),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "**Figure 4.2.** Carte de chaleur heure × réseau (départs / station). "
            "Les colonnes les plus sombres identifient les heures de pointe systémiques."
        )

        # Matrice de corrélation de Pearson entre profils horaires
        pivot_corr = hourly_by_sys.pivot(
            index="hour", columns="system_id", values="dep_per_station"
        ).fillna(0)
        if pivot_corr.shape[1] >= 2:
            corr_matrix = pivot_corr.corr(method="pearson").round(2)
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                text_auto=".2f",
                labels={"color": "r de Pearson"},
                height=max(300, len(corr_matrix) * 55),
            )
            fig_corr.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_colorbar=dict(title="r", thickness=12, len=0.8),
                font=dict(size=11),
            )
            fig_corr.update_traces(textfont=dict(size=12))
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption(
                "**Figure 4.3.** Matrice de corrélation de Pearson entre profils horaires des réseaux. "
                "r ≈ +1 = rythmes similaires. r ≈ −1 = comportements opposés. r ≈ 0 = profils indépendants."
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

st.divider()
st.caption(
    "**R. Fossé & G. Pallares · 2025–2026** — "
    "Données : GBFS station_status, collecte via `scripts/collect_status.py`. "
    "Pseudo-flux inférés des variations de disponibilité entre snapshots consécutifs."
)
