"""
00_Gold_Standard.py — Ingénierie des données et audit multi-sources.
Présentation de l'hybridation des bases de données et de la correction des flux GBFS.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Gold Standard — Audit et Hybridation",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Ingénierie des Données : Genèse du Gold Standard")
st.caption("Axe Préliminaire : De l'Open Data brut à l'infrastructure de recherche validée")

abstract_box(
    "<b>Problématique méthodologique :</b> L'Open Data constitue-t-il un matériau de recherche prêt à l'emploi ?<br><br>"
    "La robustesse d'un modèle d'évaluation spatial (tel que l'IMD) dépend intégralement de la fiabilité "
    "de ses données d'entrée (paradigme de la lutte contre le <i>Garbage In, Garbage Out</i>). Cette section documente "
    "le pipeline d'audit massif réalisé sur les flux GBFS français et la stratégie d'hybridation "
    "multi-sources (BAAC, Cerema, GTFS, INSEE) mise en œuvre pour constituer notre base de référence spatiale : le Gold Standard."
)

sidebar_nav()

# ── Section 1 : L'Illusion de l'Open Data et l'Audit GBFS ──────────────────────
st.divider()
section(1, "L'Illusion de l'Open Data : Taxonomie des Anomalies GBFS")

st.markdown(r"""
Le standard **GBFS** (*General Bikeshare Feed Specification*, v3.0) s'est imposé comme l'ontologie de référence
pour l'interopérabilité des données de micromobilité. En France, les données sont centralisées par
*transport.data.gouv.fr* et *MobilityData*. Si cette standardisation a catalysé le développement
d'applications *MaaS*, l'ingestion directe de ces données brutes dans des modèles de géographie
quantitative engendre des artefacts statistiques majeurs (*Romanillos et al., 2016 ; Fishman, 2016*).

L'audit systématique des 125 systèmes français a mis en exergue une taxonomie de **5 classes d'anomalies
critiques (A1 à A5)** :

* **A1 — Inclusion hors-domaine :** Présence de systèmes d'autopartage (ex. Citiz) encodés par erreur
  comme des flottes cyclables (14 systèmes affectés).
* **A2 — Capacité fictive (Placeholder) :** Valeur constante non nulle déclarée arbitrairement sur toutes
  les stations (ex. `pony_Nice` déclarant $c = 100$).
* **A3 — Le Biais de Surcapacité Structurelle (*Floating-Anchor*) :** L'anomalie la plus critique,
  inhérente aux flottes *free-floating* (cf. infra).
* **A4 — Aberrations géospatiales :** Coordonnées (Lat/Lon) permutées ou aberrantes générant des
  *bounding-boxes* à l'échelle continentale.
* **A5 — Hors périmètre :** Systèmes situés dans les DOM-TOM ou présentant un périmètre d'action
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

# ── Section 2 : Protocole de Purge et Filtrage ────────────────────────────────
st.divider()
section(2, "Protocole de Purge Algorithmique et Filtrage Spatial")

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

**Bilan de la consolidation :** Passage d'une base brute de **125 systèmes** à un jeu certifié de
**104 systèmes** (sur 62 agglomérations), regroupant **46 359 stations validées**.
""")

# ── Section 3 : Avant / Après (Étude de Cas) ──────────────────────────────────
st.divider()
section(3, "Avant / Après : Preuve Empirique de l'Impact de l'Audit")

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


# ── Section 4 : L'Hybridation Multi-Sources ────────────────────────────────────
st.divider()
section(4, "L'Hybridation Multi-Sources : Modéliser l'Environnement Cyclable")

st.markdown(r"""
Le GBFS indique *où* se trouve le vélo, mais demeure agnostique quant aux **déterminants
environnementaux et sociaux** qui conditionnent son usage. Le saut qualitatif du *Gold Standard* réside
dans l'enrichissement multidimensionnel (*Spatial Join*) des coordonnées avec des bases de données
institutionnelles. Six modules d'enrichissement ont été appliqués à l'ensemble des 46 359 stations
certifiées.
""")

donnees_sources = pd.DataFrame({
    "Dimension Modélisée": [
        "Infrastructure Primaire",
        "Sécurité Spatiale (S)",
        "Perméabilité Cyclable (I)",
        "Capillarité Multimodale (M)",
        "Friction Spatiale (T)",
        "Vulnérabilité Socio-Éco.",
        "Pratiques Réelles (Val.)",
    ],
    "Source de la donnée": [
        "GBFS transport.data.gouv.fr",
        "Base BAAC (ONISR)",
        "OpenStreetMap / Cerema",
        "Point d'Accès National (GTFS)",
        "NASA SRTM (30 m)",
        "INSEE (Filosofi & RP 2020)",
        "FUB (2023) / INSEE EMP (2019)",
    ],
    "Format / Nature": [
        "GeoJSON point",
        "Open Data (Accidents)",
        "Réseau filaire (Lignes)",
        "Schedules & Stops (Noeuds)",
        "Modèle Numérique (MNT)",
        "Carroyage Démographique",
        "Enquêtes déclaratives",
    ],
    "Variables intégrées et apport analytique": [
        "Coordonnées de vérité terrain, typologies certifiées et capacités redressées.",
        "Densité de clusters d'accidents corporels cyclistes dans un rayon de 300 m.",
        "Mesure continue de l'aménagement en site propre protégeant l'usager vulnérable.",
        "Distance isochrone aux pôles d'échanges lourds (Train, Tram, BHNS).",
        "Gradient altimétrique modélisant la barrière énergétique physiologique.",
        "Variables INSEE : Revenu médian, chômage, % de cadres, diplômés, sans voiture.",
        "Optimisation supervisée des pondérations du modèle IMD par convergence statistique.",
    ],
})

st.table(donnees_sources)

# ── Section 5 : Implication pour la Recherche ──────────────────────────────────
st.divider()
section(5, "Implication : L'Infrastructure de Données comme Objet de Recherche")

st.markdown(r"""
Dans le champ des études urbaines, le traitement des données est trop souvent relégué au rang de "détail
technique". Cette recherche prouve au contraire que **la qualité de la donnée est éminemment politique**.

En omettant de corriger les anomalies GBFS, un algorithme de planification publique conclurait à tort
qu'une agglomération est parfaitement couverte grâce à des capacités artificiellement gonflées, justifiant
potentiellement des réallocations de subventions inéquitables.

La mise à disposition de ce **Gold Standard au format `.parquet`** constitue donc une contribution
académique autonome. Elle offre aux futurs chercheurs et géomaticiens un "socle de vérité terrain" déjà
purgé de ses biais, prêt à supporter des modélisations complexes telles que la théorie des graphes,
l'analyse temporelle des flux de micromobilité, ou la modélisation économétrique de l'équité spatiale
(*cf.* Indice d'Équité Sociale — IES).
""")
