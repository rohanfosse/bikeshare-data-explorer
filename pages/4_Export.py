"""
4_Export.py — Interface d'accès aux données Gold Standard GBFS (principes FAIR).
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
    page_title="Export FAIR — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Accès aux Données — Gold Standard GBFS")
st.caption("Infrastructure de Données Ouverte selon les Principes FAIR (Findable, Accessible, Interoperable, Reusable)")

abstract_box(
    "<b>Contribution méthodologique :</b> La mise à disposition du Gold Standard GBFS "
    "constitue une contribution académique autonome, indépendante des résultats analytiques.<br><br>"
    "Cette interface implémente les principes <em>FAIR</em> "
    "(<em>Findable, Accessible, Interoperable, Reusable</em>) pour la diffusion du corpus "
    "Gold Standard GBFS auprès de la communauté scientifique. "
    "Le jeu de données — 46 312 stations certifiées issues de 122 systèmes nationaux, "
    "enrichies selon cinq modules spatiaux — est mis à disposition dans deux formats "
    "interopérables : CSV (UTF-8) et Parquet (Apache Arrow). "
    "Les filtres disponibles permettent l'extraction de sous-corpus analytiquement cohérents "
    "pour des usages spécifiques : étude de cas urbaine, analyse sectorielle par métrique, "
    "ou extraction d'un réseau GBFS individuel. "
    "Les métadonnées de citation, incluant le protocole d'enrichissement et les sources primaires, "
    "sont générées automatiquement à partir de la sélection courante."
)

df = load_stations()

# ── Sidebar — filtres ─────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Filtres de Sélection")

    all_cities = sorted(df["city"].unique())
    city_sel = st.multiselect(
        "Agglomération(s)",
        options=all_cities,
        default=[],
        placeholder="Corpus national complet",
    )

    all_systems = sorted(df["system_id"].dropna().unique())
    system_sel = st.multiselect(
        "Réseau(x) GBFS",
        options=all_systems,
        default=[],
        placeholder="Tous les réseaux",
    )

    all_sources = sorted(df["source_label"].dropna().unique())
    source_sel = st.multiselect(
        "Source de données",
        options=all_sources,
        default=[],
        placeholder="Toutes les sources",
    )

    st.divider()
    st.markdown("**Filtres sur les dimensions d'enrichissement**")
    st.caption("Activez un filtre pour restreindre la sélection à une plage de valeurs.")

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
    st.markdown("**Sélection des colonnes à exporter**")
    all_cols      = list(df.columns)
    enriched_cols = [k for k in METRICS if k in df.columns]
    base_cols     = [c for c in all_cols if c not in enriched_cols]

    include_base     = st.checkbox("Colonnes de base (identifiants, localisation, capacité)", value=True)
    include_enriched = st.checkbox("Dimensions d'enrichissement (modules 2–4)", value=True)

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

# ── Section 1 — Résumé du sous-corpus ────────────────────────────────────────
section(1, "Résumé du Sous-Corpus Sélectionné — Volumétrie et Périmètre")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stations sélectionnées", f"{len(dff_export):,}")
m2.metric("Agglomérations", f"{dff['city'].nunique()}")
m3.metric("Réseaux GBFS", f"{dff['system_id'].nunique()}")
m4.metric("Colonnes exportées", f"{len(cols_to_export)}")

pct_corpus = 100 * len(dff_export) / len(df)
st.caption(
    f"Le sous-corpus sélectionné représente **{pct_corpus:.1f} %** du corpus Gold Standard complet "
    f"({len(df):,} stations). Les filtres appliqués sur cette page n'affectent pas les autres "
    "analyses du tableau de bord."
)

# ── Section 2 — Schéma des colonnes ──────────────────────────────────────────
st.divider()
section(2, "Schéma des Données — Dictionnaire de Variables et Taux de Complétude")

with st.expander("Afficher le dictionnaire de variables", expanded=False):
    st.markdown(r"""
    Le dictionnaire ci-dessous référence chaque colonne du sous-corpus exporté,
    son type de données, son taux de complétude sur la sélection courante,
    et sa description méthodologique. Les colonnes d'enrichissement présentent
    des taux de complétude inférieurs à 100 % en raison des contraintes géographiques
    des sources primaires (couverture BAAC, disponibilité GTFS locale, zone SRTM).
    """)
    schema_rows = []
    for col in cols_to_export:
        dtype   = str(dff_export[col].dtype)
        n_valid = int(dff_export[col].notna().sum())
        pct     = 100 * n_valid / len(dff_export) if len(dff_export) else 0
        desc    = METRICS[col]["description"] if col in METRICS else "Variable de base (identifiant, localisation ou métadonnée)"
        schema_rows.append({
            "Colonne":       col,
            "Type":          dtype,
            "Complétude":    f"{n_valid:,} / {len(dff_export):,} ({pct:.1f} %)",
            "Description":   desc,
        })
    st.dataframe(
        pd.DataFrame(schema_rows),
        use_container_width=True,
        hide_index=True,
    )

# ── Section 3 — Statistiques descriptives ─────────────────────────────────────
st.divider()
section(3, "Statistiques Descriptives du Sous-Corpus — Caractérisation Multivariée")

with st.expander("Afficher les statistiques du sous-corpus sélectionné", expanded=False):
    metric_cols_present = [k for k in METRICS if k in dff_export.columns]
    if metric_cols_present:
        stat_rows = []
        for mkey in metric_cols_present:
            meta = METRICS[mkey]
            s    = dff_export[mkey].dropna()
            if s.empty:
                continue
            stat_rows.append({
                "Dimension":  meta["label"],
                "n valides":  f"{len(s):,}",
                "Moyenne":    round(s.mean(), 3),
                "Médiane":    round(s.median(), 3),
                "Éc. type":   round(s.std(), 3),
                "Min":        round(s.min(), 3),
                "Q25":        round(s.quantile(0.25), 3),
                "Q75":        round(s.quantile(0.75), 3),
                "Max":        round(s.max(), 3),
                "Unité":      meta["unit"],
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
        st.info("Aucune dimension d'enrichissement présente dans les colonnes sélectionnées.")

# ── Section 4 — Prévisualisation ──────────────────────────────────────────────
st.divider()
section(4, "Prévisualisation Tabulaire — Aperçu des Premières Observations")

n_preview = st.slider("Nombre de lignes à afficher", 10, 200, 50, 10)
st.dataframe(dff_export.head(n_preview), use_container_width=True, hide_index=True)

# ── Section 5 — Téléchargement ────────────────────────────────────────────────
st.divider()
section(5, "Téléchargement — Formats CSV (UTF-8) et Parquet (Apache Arrow)")

st.markdown(r"""
Deux formats d'export sont proposés, conformément aux standards d'interopérabilité FAIR :

* **CSV (UTF-8, séparateur virgule)** : Compatible avec R (`read.csv`), Python (`pandas.read_csv`),
  Stata, SPSS et tout tableur standard. Recommandé pour les analyses exploratoires.
* **Parquet (Apache Arrow)** : Format colonnaire compressé, recommandé pour les pipelines
  analytiques sur grands volumes ($n > 10\,000$ lignes) en Python (`pyarrow`, `pandas`) ou R
  (`arrow`). Gain de taille typique : facteur 3 à 5 par rapport au CSV équivalent.
""")

dl1, dl2 = st.columns(2)

with dl1:
    st.markdown("**Format CSV — Interopérabilité universelle**")
    csv_bytes = dff_export.to_csv(index=False).encode("utf-8")
    size_kb   = len(csv_bytes) / 1024
    st.download_button(
        label=f"Télécharger en CSV ({size_kb:,.0f} Ko — {len(dff_export):,} observations)",
        data=csv_bytes,
        file_name="gold_standard_gbfs_export.csv",
        mime="text/csv",
    )

with dl2:
    st.markdown("**Format Parquet — Pipeline analytique haute performance**")
    buf = io.BytesIO()
    dff_export.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()
    size_kb_pq    = len(parquet_bytes) / 1024
    st.download_button(
        label=f"Télécharger en Parquet ({size_kb_pq:,.0f} Ko — {len(dff_export):,} observations)",
        data=parquet_bytes,
        file_name="gold_standard_gbfs_export.parquet",
        mime="application/octet-stream",
    )

# ── Section 6 — Métadonnées de citation ───────────────────────────────────────
st.divider()
section(6, "Métadonnées de Citation et de Reproductibilité")

with st.expander("Informations pour la citation scientifique", expanded=True):
    metric_keys_present = [k for k in METRICS if k in dff_export.columns]
    st.markdown(
        f"""
**Jeu de données** : Gold Standard GBFS — Micromobilité française
**Pipeline d'enrichissement** : Notebooks 20–27 · CESI BikeShare-ICT (2025-2026)
**Observations exportées** : {len(dff_export):,} / {len(df):,} (corpus complet)
**Agglomérations** : {dff["city"].nunique()} · **Réseaux GBFS** : {dff["system_id"].nunique()}
**Dimensions d'enrichissement présentes** : {", ".join(f"`{k}`" for k in metric_keys_present)}
**Rayon d'analyse spatiale** : 300 m autour de chaque point de stationnement

**Sources primaires de l'enrichissement spatial :**
- Infrastructure cyclable : OpenStreetMap / Overpass API (Module 3A)
- Topographie : Open-Elevation API — SRTM 30 m NASA (Module 2)
- Sinistralité cycliste : BAAC 2021–2023, ONISR / Ministère de l'Intérieur (Module 3B)
- Accessibilité multimodale : Point d'Accès National GTFS — transport.data.gouv.fr (Module 4)

**Protocole de reproductibilité :**
Les résultats sont reproductibles à partir du dépôt public (Notebook 27).
Le corpus Gold Standard a été constitué après application de la taxonomie d'audit en 5 classes
d'anomalies (A1–A5) documentée dans la Section 2 de la publication associée.
        """
    )
