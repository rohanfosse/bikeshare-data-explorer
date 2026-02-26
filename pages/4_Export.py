"""
4_Export.py — Export des données Gold Standard pour les chercheurs.
Filtrage, prévisualisation et téléchargement du jeu de données.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Export des données — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Export des données — Gold Standard GBFS")
st.caption("Gold Standard GBFS · CESI BikeShare-ICT 2025-2026")

abstract_box(
    "Cette page permet aux chercheurs de filtrer, prévisualiser et télécharger "
    "le jeu de données Gold Standard GBFS selon leurs besoins analytiques. "
    "Les filtres portent sur les villes, les réseaux GBFS, la source des données "
    "et les plages de valeurs des métriques enrichies. "
    "L'export est disponible en CSV (UTF-8, séparateur virgule) "
    "ou en Parquet (Apache Arrow, recommandé pour les grands volumes). "
    "Les filtres appliqués ici n'affectent pas les autres pages du tableau de bord."
)

df = load_stations()

# ── Sidebar — filtres ─────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Filtres")

    all_cities = sorted(df["city"].unique())
    city_sel = st.multiselect(
        "Villes",
        options=all_cities,
        default=[],
        placeholder="Toutes les villes",
    )

    all_systems = sorted(df["system_id"].dropna().unique())
    system_sel = st.multiselect(
        "Réseaux GBFS",
        options=all_systems,
        default=[],
        placeholder="Tous les réseaux",
    )

    all_sources = sorted(df["source_label"].dropna().unique())
    source_sel = st.multiselect(
        "Source des données",
        options=all_sources,
        default=[],
        placeholder="Toutes les sources",
    )

    st.divider()
    st.markdown("**Filtres sur les métriques enrichies**")
    st.caption("Cochez pour activer un filtre de plage.")

    metric_filters: dict[str, tuple[float, float] | None] = {}
    for mkey, meta in METRICS.items():
        if mkey not in df.columns:
            continue
        s = df[mkey].dropna()
        if s.empty:
            continue
        active = st.checkbox(meta["label"], value=False, key=f"chk_{mkey}")
        if active:
            vmin, vmax = float(s.min()), float(s.max())
            lo, hi = st.slider(
                f"Plage — {meta['label']} ({meta['unit']})",
                min_value=vmin,
                max_value=vmax,
                value=(vmin, vmax),
                key=f"rng_{mkey}",
            )
            metric_filters[mkey] = (lo, hi)
        else:
            metric_filters[mkey] = None

    st.divider()
    st.markdown("**Colonnes à exporter**")
    all_cols      = list(df.columns)
    enriched_cols = [k for k in METRICS if k in df.columns]
    base_cols     = [c for c in all_cols if c not in enriched_cols]

    include_base     = st.checkbox("Colonnes de base (id, localisation, capacité…)", value=True)
    include_enriched = st.checkbox("Métriques enrichies (modules 2-4)", value=True)

# ── Application des filtres ───────────────────────────────────────────────────
dff = df.copy()
if city_sel:
    dff = dff[dff["city"].isin(city_sel)]
if system_sel:
    dff = dff[dff["system_id"].isin(system_sel)]
if source_sel:
    dff = dff[dff["source_label"].isin(source_sel)]

for mkey, rng in metric_filters.items():
    if rng is not None:
        dff = dff[(dff[mkey] >= rng[0]) & (dff[mkey] <= rng[1])]

cols_to_export: list[str] = []
if include_base:
    cols_to_export += base_cols
if include_enriched:
    cols_to_export += enriched_cols
if not cols_to_export:
    cols_to_export = all_cols

dff_export = dff[cols_to_export]

# ── Section 1 — Résumé de la sélection ───────────────────────────────────────
section(1, "Résumé de la sélection")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stations sélectionnées", f"{len(dff_export):,}")
m2.metric("Villes", f"{dff['city'].nunique()}")
m3.metric("Réseaux", f"{dff['system_id'].nunique()}")
m4.metric("Colonnes", f"{len(cols_to_export)}")

# ── Section 2 — Schéma des colonnes ──────────────────────────────────────────
st.divider()
section(2, "Schéma des colonnes exportées")

with st.expander("Afficher le schéma", expanded=False):
    schema_rows = []
    for col in cols_to_export:
        dtype   = str(dff_export[col].dtype)
        n_valid = int(dff_export[col].notna().sum())
        pct     = 100 * n_valid / len(dff_export) if len(dff_export) else 0
        desc    = METRICS[col]["description"] if col in METRICS else "Champ de base"
        schema_rows.append({
            "Colonne":     col,
            "Type":        dtype,
            "Valides":     f"{n_valid:,} ({pct:.1f} %)",
            "Description": desc,
        })
    st.dataframe(
        pd.DataFrame(schema_rows),
        use_container_width=True,
        hide_index=True,
    )

# ── Section 3 — Statistiques descriptives ─────────────────────────────────────
st.divider()
section(3, "Statistiques descriptives de la sélection")

with st.expander("Afficher les statistiques", expanded=False):
    metric_cols_present = [k for k in METRICS if k in dff_export.columns]
    if metric_cols_present:
        stat_rows = []
        for mkey in metric_cols_present:
            meta = METRICS[mkey]
            s    = dff_export[mkey].dropna()
            if s.empty:
                continue
            stat_rows.append({
                "Métrique": meta["label"],
                "n":        f"{len(s):,}",
                "Moyenne":  round(s.mean(), 3),
                "Médiane":  round(s.median(), 3),
                "Éc. type": round(s.std(), 3),
                "Min":      round(s.min(), 3),
                "Q25":      round(s.quantile(0.25), 3),
                "Q75":      round(s.quantile(0.75), 3),
                "Max":      round(s.max(), 3),
                "Unité":    meta["unit"],
            })
        st.dataframe(
            pd.DataFrame(stat_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                k: st.column_config.NumberColumn(format="%.3f")
                for k in ["Moyenne", "Médiane", "Éc. type", "Min", "Q25", "Q75", "Max"]
            },
        )
    else:
        st.info("Aucune métrique enrichie dans les colonnes sélectionnées.")

# ── Section 4 — Prévisualisation ──────────────────────────────────────────────
st.divider()
section(4, "Prévisualisation des données")

n_preview = st.slider("Nombre de lignes à afficher", 10, 200, 50, 10)
st.dataframe(dff_export.head(n_preview), use_container_width=True, hide_index=True)

# ── Section 5 — Téléchargement ────────────────────────────────────────────────
st.divider()
section(5, "Téléchargement")

st.caption(
    "Le fichier CSV est encodé en UTF-8 avec séparateur virgule. "
    "Le format Parquet (Apache Arrow) est recommandé pour les analyses "
    "Python ou R sur les grands volumes de données."
)

dl1, dl2 = st.columns(2)

with dl1:
    st.markdown("**Format CSV**")
    csv_bytes = dff_export.to_csv(index=False).encode("utf-8")
    size_kb   = len(csv_bytes) / 1024
    st.download_button(
        label=f"Télécharger en CSV ({size_kb:,.0f} Ko — {len(dff_export):,} lignes)",
        data=csv_bytes,
        file_name="gold_standard_gbfs_export.csv",
        mime="text/csv",
    )

with dl2:
    st.markdown("**Format Parquet**")
    buf = io.BytesIO()
    dff_export.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()
    size_kb_pq    = len(parquet_bytes) / 1024
    st.download_button(
        label=f"Télécharger en Parquet ({size_kb_pq:,.0f} Ko — {len(dff_export):,} lignes)",
        data=parquet_bytes,
        file_name="gold_standard_gbfs_export.parquet",
        mime="application/octet-stream",
    )

# ── Section 6 — Citation et reproductibilité ──────────────────────────────────
st.divider()
section(6, "Citation et reproductibilité")

with st.expander("Informations pour la citation", expanded=False):
    metric_keys_present = [k for k in METRICS if k in dff_export.columns]
    st.markdown(
        f"""
**Jeu de données** : Gold Standard GBFS — Micromobilité française
**Pipeline** : Notebook 27 — CESI BikeShare-ICT (2025-2026)
**Stations exportées** : {len(dff_export):,} / {len(df):,}
**Villes** : {dff["city"].nunique()}
**Réseaux** : {dff["system_id"].nunique()}
**Métriques enrichies présentes** : {", ".join(f"`{k}`" for k in metric_keys_present)}
**Rayon d'analyse** : 300 m autour de chaque station

**Sources primaires** :
- Infrastructure cyclable : OpenStreetMap / Overpass API
- Topographie : Open-Elevation (SRTM 30 m)
- Accidentologie : BAAC 2021-2023, ONISR
- Transports en commun : flux GTFS nationaux (transport.data.gouv.fr)
        """
    )
