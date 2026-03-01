"""
00_Gold_Standard.py - Ingénierie des données et audit multi-sources.
Présentation de l'hybridation des bases de données et de la correction des flux GBFS.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import (
    city_stats,
    completeness_report,
    load_stations,
    load_systems_catalog,
)
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Gold Standard - Audit et Hybridation",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Ingénierie des Données : Genèse du Gold Standard")
st.caption("Axe Préliminaire : De l'Open Data brut à l'infrastructure de recherche validée")

# ── Chargement des données (avant abstract pour valeurs dynamiques) ────────────
df      = load_stations()
catalog = load_systems_catalog()
cities  = city_stats(df)
compl   = completeness_report(df)

# ── Vue dock-based : station_type == 'docked_bike' ────────────────────────────
# Le Gold Standard Final classe chaque entrée via la colonne `station_type` :
# 'docked_bike' (VLS physique), 'free_floating' (A3), 'carsharing' (A1).
df_dock    = df[df["station_type"] == "docked_bike"].copy() if "station_type" in df.columns else df.copy()
cities_dock = city_stats(df_dock)

# ── Métriques clés calculées ───────────────────────────────────────────────────
n_ok         = int((catalog["status"] == "ok").sum()) if "status" in catalog.columns else len(catalog)
n_ff         = int((df["station_type"] == "free_floating").sum()) if "station_type" in df.columns else 0
n_carshare   = int((df["station_type"] == "carsharing").sum())    if "station_type" in df.columns else 0
n_dock       = len(df_dock)
n_dock_cities = df_dock["city"].nunique()
avg_compl    = compl["Complétude (%)"].mean() if not compl.empty else 0

abstract_box(
    "<b>Problématique méthodologique :</b> L'Open Data constitue-t-il un matériau de recherche prêt à l'emploi ?<br><br>"
    "La robustesse d'un modèle d'évaluation spatial (tel que l'IMD) dépend intégralement de la fiabilité "
    "de ses données d'entrée (paradigme de la lutte contre le <i>Garbage In, Garbage Out</i>). Cette section documente "
    "le pipeline d'audit massif réalisé sur les flux GBFS français et la stratégie d'hybridation "
    f"multi-sources (BAAC, Cerema, GTFS, INSEE) mise en œuvre pour constituer notre base de référence spatiale : "
    f"le <b>Gold Standard GBFS</b> - <b>{len(df):,} stations certifiées</b> issues de <b>{n_ok} systèmes</b> "
    f"({n_dock_cities} agglomérations), dont <b>{n_dock:,} stations dock-based</b> (VLS physique) et "
    f"{n_ff:,} points free-floating. Le corpus est enrichi selon six modules spatiaux (topographie SRTM, "
    "infrastructure OSM, accidentologie BAAC, multimodalité GTFS, profil socio-économique INSEE Filosofi, "
    "usage modal RP 2020)."
)

sidebar_nav()

# ── KPIs réels ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Stations Gold Standard", f"{len(df):,}")
k2.metric("Dont VLS dock-based", f"{n_dock:,}", f"{n_dock_cities} agglomérations")
k3.metric("Systèmes GBFS certifiés", f"{n_ok}")
k4.metric("Complétude moyenne d'enrichissement", f"{avg_compl:.1f} %")

# ── Section 1 : L'Illusion de l'Open Data et l'Audit GBFS ──────────────────────
st.divider()
section(1, "L'Illusion de l'Open Data : Taxonomie des Anomalies GBFS")

st.markdown(r"""
Le standard **GBFS** (*General Bikeshare Feed Specification*, v3.0) s'est imposé comme l'ontologie de
référence pour l'interopérabilité des données de micromobilité. En France, les données sont centralisées
par *transport.data.gouv.fr* et *MobilityData*. Si cette standardisation a catalysé le développement
d'applications *MaaS*, l'ingestion directe de ces données brutes dans des modèles de géographie
quantitative engendre des artefacts statistiques majeurs (*Romanillos et al., 2016 ; Fishman, 2016*).

L'audit systématique des systèmes GBFS français a mis en exergue une taxonomie de **5 classes d'anomalies
critiques (A1 à A5)**, désormais encodées dans la colonne `station_type` du Gold Standard Final :

* **A1 - Inclusion hors-domaine :** Présence de systèmes d'autopartage (ex. Citiz) encodés par erreur
  comme des flottes cyclables (14 systèmes affectés).
* **A2 - Capacité fictive (Placeholder) :** Valeur constante non nulle déclarée arbitrairement sur
  toutes les stations (ex. `pony_Nice` déclarant $c = 100$).
* **A3 - Le Biais de Surcapacité Structurelle (*Floating-Anchor*) :** L'anomalie la plus critique,
  inhérente aux flottes *free-floating* (cf. infra).
* **A4 - Aberrations géospatiales :** Coordonnées (Lat/Lon) permutées ou aberrantes générant des
  *bounding-boxes* à l'échelle continentale.
* **A5 - Hors périmètre :** Systèmes situés dans les DOM-TOM ou présentant un périmètre d'action
  macro-régional ($> 50\,000\,\text{km}^2$).

#### Zoom sur l'anomalie A3 : Le biais de la moyenne conditionnelle

L'anomalie la plus pernicieuse concerne l'hybridation des flottes *free-floating* s'attachant au mobilier
urbain. Pour éviter d'afficher des stations vides, le calcul du profil capacitaire se fait souvent par
**moyenne conditionnelle** (en excluant les stations à capacité nulle). Le biais se formalise ainsi :
""")

st.latex(r"""
\bar{c}_{\text{profil}} = \frac{\sum_{i : c_i > 0} c_i}{\#\{i : c_i > 0\}}
\quad \neq \quad
\bar{c}_{\text{réel}} = \frac{\sum_{i=1}^{N} c_i}{N}
""")

st.markdown(r"""
Ce biais mathématique classe à tort des milliers de vélos *free-floating* comme des stations d'ancrage
lourdes (*dock-based*), faussant intégralement l'analyse des densités urbaines et rendant caduque toute
comparaison inter-systèmes non auditée.
""")

# ── Section 2 : Protocole de Purge et Résultats ───────────────────────────────
st.divider()
section(2, "Protocole de Purge Algorithmique : Du Bruit au Signal Certifié")

st.markdown(r"""
Pour distiller ce bruit statistique, un pipeline de redressement séquentiel en **6 étapes** a été
implémenté :

1. **Exclusion sémantique :** Retrait des 14 systèmes d'autopartage Citiz identifiés lors de la phase
   d'audit.
2. **Reclassification A2 :** Passage forcé en *free-floating* pour les systèmes à capacité placeholder
   fictive (`pony_Nice`).
3. **Redressement A3 :** Recalcul systématique de $\bar{c}_{\text{réel}}$ et réassignation topologique
   (Dock / Semi-dock / FF).
4. **Géofiltre national :** Suppression des stations hors France métropolitaine
   (Box : $\varphi \in [41^\circ, 52^\circ]$, $\lambda \in [-6^\circ, 10^\circ]$).
5. **Filtre Topologique ($3\sigma$) :** Suppression des *outliers* GPS éloignés de plus de 3
   écarts-types du centroïde du système.
6. **Seuil de robustesse spatiale :** Exclusion des micro-réseaux ($N_{\min} < 20$ stations dock)
   incapables de soutenir une analyse maillée.

"""
)

st.markdown(
    f"**Bilan de la consolidation :** Le Gold Standard Final regroupe **{len(df):,} stations certifiées** "
    f"issues de **{n_ok} systèmes** couvrant **{df['city'].nunique()} agglomérations**, dont "
    f"**{n_dock:,} stations dock-based** (VLS physique, `station_type = docked_bike`), "
    f"**{n_ff:,} points free-floating** (mobilité légère, A3 corrigé) et "
    f"**{n_carshare:,} points d'autopartage** (A1, conservés pour analyse comparative)."
)

# Catalogue des systèmes
if not catalog.empty and "status" in catalog.columns:
    col_bar, col_stats = st.columns([3, 2])

    with col_bar:
        if "region" in catalog.columns and "n_stations" in catalog.columns:
            region_df = (
                catalog[catalog["status"] == "ok"]
                .groupby("region")["n_stations"]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={"sum": "n_stations", "count": "n_systemes"})
                .sort_values("n_stations", ascending=True)
            )
            fig_reg = px.bar(
                region_df,
                x="n_stations",
                y="region",
                orientation="h",
                color="n_stations",
                color_continuous_scale="Blues",
                text="n_systemes",
                labels={"n_stations": "Stations certifiées", "region": "Région", "n_systemes": "Systèmes"},
                height=max(320, len(region_df) * 28),
            )
            fig_reg.update_traces(
                texttemplate="%{text} syst.",
                textposition="outside",
            )
            fig_reg.update_layout(
                coloraxis_showscale=False,
                plot_bgcolor="white",
                margin=dict(l=10, r=80, t=10, b=10),
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            st.caption(
                "**Figure 2.1.** Distribution des stations Gold Standard certifiées par région "
                "administrative. Les étiquettes indiquent le nombre de systèmes GBFS certifiés par région. "
                "La concentration en Île-de-France et Occitanie reflète la présence des plus grands "
                "réseaux nationaux (Vélib', Vélomagg)."
            )

    with col_stats:
        # Status breakdown
        status_counts = catalog["status"].value_counts().reset_index()
        status_counts.columns = ["Statut", "Systèmes"]
        status_labels = {
            "ok": "Certifié (Gold Standard)",
            "too_small": "Exclu (micro-réseau)",
            "autopartage": "Exclu (autopartage)",
            "excluded": "Exclu (autre anomalie)",
            "dom_tom": "Hors périmètre (DOM-TOM)",
        }
        status_counts["Statut"] = status_counts["Statut"].map(
            lambda s: status_labels.get(s, s)
        )
        st.dataframe(
            status_counts,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"**Tableau 2.1.** Répartition des {len(catalog)} systèmes GBFS audités "
            "selon le verdict de l'audit (5 classes d'anomalies A1–A5)."
        )

        # Top cities by station count (dock-based uniquement)
        st.markdown("**Top 10 agglomérations - stations dock-based**")
        top_cities = cities_dock.head(10)[["city", "n_stations"]].rename(
            columns={"city": "Agglomération", "n_stations": "Stations"}
        )
        st.dataframe(top_cities, use_container_width=True, hide_index=True,
                     column_config={"Stations": st.column_config.ProgressColumn(
                         "Stations",
                         min_value=0,
                         max_value=int(cities_dock["n_stations"].max()),
                         format="%d",
                     )})

    # Catalog details (expandable)
    with st.expander(f"Catalogue complet des {len(catalog)} systèmes audités", expanded=False):
        display_cat = catalog.copy()
        if "gbfs_url" in display_cat.columns:
            display_cat = display_cat.drop(columns=["gbfs_url"])
        rename_cat = {
            "source":      "Source",
            "system_id":   "Identifiant",
            "title":       "Nom du système",
            "city":        "Agglomération",
            "region":      "Région",
            "department":  "Département",
            "n_stations":  "Stations (brut)",
            "status":      "Statut d'audit",
        }
        display_cat = display_cat.rename(columns={k: v for k, v in rename_cat.items() if k in display_cat.columns})
        st.dataframe(display_cat, use_container_width=True, hide_index=True)

# ── Section 3 : Avant / Après (Étude de Cas) ──────────────────────────────────
st.divider()
section(3, "Avant / Après : Preuve Empirique de l'Impact de l'Audit - Cas Bordeaux")

st.markdown(r"""
Pour saisir l'impact de ces corrections sur l'évaluation des politiques publiques, le cas de **Bordeaux**
(Anomalie A3 via l'opérateur *Pony*) est paradigmatique. Sans cet audit, 2 996 stations *Pony*
(capacité réelle de $0{,}03$ vélo par station) étaient comptabilisées comme de véritables stations
*dock-based*, hissant artificiellement l'agglomération au 2e rang national de l'IMD.
""")

col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
col_metrics1.metric("Bordeaux : Stations Dock (Avant)", "3 221")
col_metrics2.metric("Bordeaux : Stations Dock (Après)", "225", "-93 %", delta_color="inverse")
col_metrics3.metric("Bordeaux : Rang IMD", "2 → 14 (corrigé)")

st.info(
    "**Démonstration :** Une simple erreur de calcul asymptotique dans les flux ouverts suffit à hisser "
    "artificiellement une agglomération au 2e rang national. La correction ramène le réseau à sa stricte "
    "réalité physique (Rang 14), illustrant que la qualité de la donnée est éminemment politique."
)

st.markdown("<br>", unsafe_allow_html=True)

col_before, col_after = st.columns(2)

with col_before:
    st.error("**AVANT : GBFS Brut (Approche Naïve)**")
    st.markdown("""
    * **Nature :** Fichiers JSON éclatés par opérateur.
    * **Bruit statistique :** Fort (Stations doublons, erreurs de géocodage).
    * **Aberrations capacitives :** Prévalence du biais "A3" faussant les moyennes.
    * **Absence de contexte :** Un point GPS nu, sans environnement urbain.
    * **Agnosticisme social :** Aucune donnée sur la population desservie.
    """)
    st.code("""
    # Exemple d'un point GBFS brut
    {
      "station_id": "bordeaux_hub_12",
      "lat": 44.8377,
      "lon": -0.5791,
      "capacity": 999, // BIAIS MAJEUR (A3)
      "is_installed": true
    }
    """, language="json")

with col_after:
    st.success("**APRÈS : Gold Standard (Notre Contribution)**")
    st.markdown("""
    * **Nature :** Fichier structuré unique (`.parquet` / `.geojson`).
    * **Signal purifié :** Application des filtres et suppression des 5 anomalies.
    * **Redressement de l'offre :** Capacité recalculée $\\bar{c}_{\\text{réel}}$.
    * **Environnement 360° :** Scores de rugosité (MNT), accès au tramway (GTFS), et pistes (OSM).
    * **Dimension Sociale :** Enrichissement par les variables INSEE Filosofi/RP.
    """)
    st.code("""
    # Exemple d'un point Gold Standard enrichi
    {
      "station_id": "bordeaux_hub_12",
      "lat": 44.8377,
      "lon": -0.5791,
      "capacity_corrected": 14,     // CORRIGÉ
      "imd_topography_score": 0.85, // SRTM NASA
      "imd_safety_score": 0.42,     // BAAC
      "imd_transit_dist_m": 120,    // GTFS
      "insee_median_income": 22450  // FILOSOFI
    }
    """, language="json")

# ── Section 4 : Complétude de l'Enrichissement ────────────────────────────────
st.divider()
section(4, "Complétude de l'Enrichissement Spatial - Couverture par Module Méthodologique")

st.markdown(r"""
L'enrichissement spatial des {N_STATIONS} stations certifiées repose sur cinq modules indépendants
(Topographie, Infrastructure, Accidentologie, Multimodalité, Socio-Économique).
La complétude de chaque module - proportion de stations disposant d'une valeur valide - est conditionnée
par la couverture géographique de la source primaire et par les contraintes de l'algorithme de
*Spatial Join* (rayon de 300 m). Le tableau ci-dessous documente le taux de couverture observé pour
chaque dimension d'enrichissement sur le corpus complet.
""".replace("{N_STATIONS}", f"{len(df):,}"))

if not compl.empty:
    # Color-coded completeness bar chart
    compl_sorted = compl.sort_values("Complétude (%)", ascending=True).copy()
    compl_sorted["couleur"] = compl_sorted["Complétude (%)"].apply(
        lambda v: "#27ae60" if v >= 80 else ("#e67e22" if v >= 50 else "#e74c3c")
    )

    fig_compl = go.Figure(go.Bar(
        x=compl_sorted["Complétude (%)"],
        y=compl_sorted["Métrique"],
        orientation="h",
        marker_color=compl_sorted["couleur"].tolist(),
        text=compl_sorted["Complétude (%)"].apply(lambda v: f"{v:.1f} %"),
        textposition="outside",
    ))
    fig_compl.update_layout(
        height=max(280, len(compl_sorted) * 42),
        plot_bgcolor="white",
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(title="Taux de complétude (%)", range=[0, 115]),
        yaxis=dict(title=""),
        showlegend=False,
    )
    fig_compl.add_vline(x=80, line_dash="dash", line_color="#27ae60", opacity=0.5,
                        annotation_text="Seuil qualité (80 %)", annotation_position="top")
    st.plotly_chart(fig_compl, use_container_width=True)
    st.caption(
        f"**Figure 4.1.** Taux de complétude par dimension d'enrichissement spatial "
        f"sur les {len(df):,} stations Gold Standard. "
        "Vert : complétude $\\geq 80\\,\\%$ (qualité satisfaisante) ; "
        "Orange : $50\\,\\% \\leq$ complétude $< 80\\,\\%$ (couverture partielle) ; "
        "Rouge : complétude $< 50\\,\\%$ (contrainte de source primaire - couverture BAAC ou SRTM)."
    )

    # Completeness table
    with st.expander("Tableau détaillé de complétude", expanded=False):
        display_compl = compl.copy()
        display_compl["Complétude (%)"] = display_compl["Complétude (%)"].round(1)
        st.dataframe(
            display_compl,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Complétude (%)": st.column_config.ProgressColumn(
                    "Complétude (%)", min_value=0, max_value=100, format="%.1f %%"
                ),
                "Valides": st.column_config.NumberColumn(format="%d"),
                "Total":   st.column_config.NumberColumn(format="%d"),
            },
        )

# ── Section 5 : L'Hybridation Multi-Sources ────────────────────────────────────
st.divider()
section(5, "L'Hybridation Multi-Sources : Modéliser l'Environnement Cyclable à 360°")

st.markdown(r"""
Le GBFS indique *où* se trouve le vélo, mais demeure agnostique quant aux **déterminants
environnementaux et sociaux** qui conditionnent son usage. Le saut qualitatif du *Gold Standard* réside
dans l'enrichissement multidimensionnel (*Spatial Join*) des coordonnées avec des bases de données
institutionnelles. Six modules d'enrichissement ont été appliqués à l'ensemble des {N_STATIONS} stations
certifiées, couvrant les déterminants identifiés dans la littérature scientifique
(*Pucher et al., 2010 ; Parkin et al., 2008 ; Fishman, 2016*).
""".replace("{N_STATIONS}", f"{len(df):,}"))

donnees_sources = pd.DataFrame({
    "Dimension Modélisée": [
        "Infrastructure Primaire",
        "Sécurité Spatiale (S - IMD)",
        "Perméabilité Cyclable (I - IMD)",
        "Capillarité Multimodale (M - IMD)",
        "Friction Spatiale (T - IMD)",
        "Vulnérabilité Socio-Éco. (IES)",
        "Pratiques Réelles (Validation externe)",
    ],
    "Source de la donnée": [
        "GBFS transport.data.gouv.fr",
        "Base BAAC (ONISR) 2021–2023",
        "OpenStreetMap / Cerema",
        "Point d'Accès National (GTFS)",
        "NASA SRTM (30 m via Open-Topo-Data)",
        "INSEE (Filosofi & RP 2020)",
        "FUB Baromètre 2023 / INSEE EMP 2019",
    ],
    "Format / Nature": [
        "GeoJSON point",
        "Open Data (Accidents corporels)",
        "Réseau filaire (Lignes OSM)",
        "Schedules & Stops (Noeuds GTFS)",
        "Modèle Numérique de Terrain",
        "Carroyage Démographique 200 m",
        "Enquêtes déclaratives agrégées",
    ],
    "Variables intégrées et apport analytique": [
        "Coordonnées de vérité terrain, typologies certifiées et capacités redressées.",
        "Densité de clusters d'accidents corporels cyclistes dans un rayon de 300 m.",
        "Mesure continue de l'aménagement en site propre protégeant l'usager vulnérable.",
        "Distance isochrone aux pôles d'échanges lourds (Train, Tram, BHNS).",
        "Gradient altimétrique modélisant la barrière énergétique physiologique.",
        "Revenu médian, chômage, % de cadres, diplômés, sans voiture par carreau INSEE.",
        "Part modale vélo effective et score perçu du climat cyclable - validation externe de l'IMD.",
    ],
})

st.table(donnees_sources)

# Distribution des stations par ville (top 20) - dock-based uniquement
st.markdown("#### Couverture Géographique - Stations Dock-Based par Agglomération")
st.caption(
    "Filtrage appliqué : les flottes *free-floating* (A3 : bird, dott, pony, voi) et les "
    "systèmes d'autopartage (A1 : citiz) sont exclus. Seules les stations **dock-based** "
    "ou *semi-dock* certifiées sont comptabilisées, permettant une comparaison homogène "
    "de la capacité physique des réseaux VLS."
)
top20 = cities_dock.head(20).copy()
fig_cities = px.bar(
    top20,
    x="n_stations",
    y="city",
    orientation="h",
    color="n_stations",
    color_continuous_scale="Blues",
    text="n_stations",
    labels={"city": "Agglomération", "n_stations": "Stations dock certifiées"},
    height=480,
)
fig_cities.update_traces(texttemplate="%{x:,}", textposition="outside")
fig_cities.update_layout(
    coloraxis_showscale=False,
    plot_bgcolor="white",
    margin=dict(l=10, r=60, t=10, b=10),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_cities, use_container_width=True)
st.caption(
    "**Figure 5.1.** Top 20 agglomérations par nombre de stations VLS dock-based certifiées "
    "(hors flottes free-floating A3 et autopartage A1). Paris (Vélib') domine largement, "
    "suivi par Lyon et Toulouse. L'exclusion des opérateurs free-floating ramène Bordeaux "
    "de 9 920 stations brutes à 225 stations dock-based (*velo-TBM*), illustrant concrètement "
    "l'impact de la correction A3 sur l'évaluation comparative des réseaux VLS."
)

# ── Section 6 : Implication pour la Recherche ──────────────────────────────────
st.divider()
section(6, "Implication : L'Infrastructure de Données comme Objet de Recherche Autonome")

st.markdown(r"""
Dans le champ des études urbaines, le traitement des données est trop souvent relégué au rang de "détail
technique". Cette recherche prouve au contraire que **la qualité de la donnée est éminemment politique**.

En omettant de corriger les anomalies GBFS, un algorithme de planification publique conclurait à tort
qu'une agglomération est parfaitement couverte grâce à des capacités artificiellement gonflées, justifiant
potentiellement des réallocations de subventions inéquitables. Le cas Bordeaux illustre cette menace :
sans audit, l'agglomération se serait vue attribuer un rang IMD de 2e place nationale, conduisant à
une réallocation des ressources de mobilité au détriment de villes objectivement mieux équipées.

La mise à disposition de ce **Gold Standard au format `.parquet`** constitue donc une contribution
académique autonome à deux niveaux :

1. **Contribution méthodologique :** Un protocole d'audit reproductible et généralisable à tout corpus
   GBFS national ou international, documenté dans les Notebooks 20–21 du dépôt public.
2. **Contribution empirique :** Un jeu de données de {N_STATIONS} stations certifiées, enrichies selon
   six modules spatiaux (dont les données socio-économiques INSEE Filosofi - revenu médian, Gini,
   mobilité), prêt à supporter des modélisations complexes telles que la théorie des graphes,
   l'analyse temporelle des flux de micromobilité, ou la modélisation économétrique de
   l'équité spatiale - Indice d'Équité Sociale (IES, *cf.* page dédiée).
""".replace("{N_STATIONS}", f"{len(df):,}"))
