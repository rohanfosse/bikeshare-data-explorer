"""
00_Gold_Standard.py — Ingénierie des données et audit multi-sources.
Présentation de l'hybridation des bases de données et de la correction des flux GBFS.
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.styles import abstract_box, inject_css, section, sidebar_nav

# ── Configuration de la page ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Standard — Audit et Hybridation",
    page_icon="🛠️",
    layout="wide",
)
inject_css()

# ── En-tête de la page ─────────────────────────────────────────────────────────
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
section(1, "L'Illusion de l'Open Data : La nécessité de l'Audit GBFS")

st.markdown(r"""
Le standard **GBFS** (*General Bikeshare Feed Specification*) s'est imposé comme l'ontologie de référence pour la diffusion en temps réel de l'état des flottes de vélos en libre-service. Si cette standardisation a catalysé le développement d'applications de type *Mobility as a Service (MaaS)*, elle masque des **asymétries de qualité et des biais structurels profonds** d'un opérateur à l'autre.

L'ingestion directe de ces données brutes dans des modèles de géographie quantitative engendre des artefacts statistiques majeurs. L'audit systématique des 125 systèmes français a mis en exergue une taxonomie de **6 classes d'anomalies (A1 à A6)**, nécessitant un protocole de purge algorithmique strict :

1. **A1 — Redondance spatio-temporelle :** Multiplicité d'enregistrements d'une même entité induite par des latences de synchronisation des API.
2. **A2 — Artéfacts topologiques (Stations zombies) :** Entités déclarées actives dans l'architecture réseau (`station_information.json`) mais souffrant d'obsolescence physique sur le terrain.
3. **A3 — Le Biais de Surcapacité Structurelle (*Floating-Anchor*) :** L'anomalie la plus critique pour la modélisation. Sur les systèmes hybrides (vélos *free-floating* s'attachant au mobilier urbain), les opérateurs imputent arbitrairement des capacités virtuelles (ex. « 999 docks ») aux points d'ancrage. Non corrigé, ce biais génère une surestimation asymptotique de l'offre (supérieure à 90 % pour les métropoles de Bordeaux ou Lille), invalidant toute analyse spatiale de densité.
4. **A4 — Incohérence typologique :** Déficit de granularité dans la classification énergétique de la flotte (confusion mécanique vs. assistance électrique).
5. **A5 — Dérive géospatiale :** Aberrations de géocodage entraînant la projection de coordonnées hors des polygones administratifs (EPCI) de rattachement.
6. **A6 — Instabilité des clés primaires (UUID) :** Rupture de la continuité des identifiants au fil des itérations de l'API, prohibant toute analyse longitudinale des flux.

**Processus d'assainissement algorithmique :** L'implémentation de heuristiques de correction ciblées (notamment le redressement par moyenne conditionnelle pour purger l'anomalie A3, et le géofiltrage strict pour A5) a permis de distiller ce bruit statistique pour aboutir à une **base de vérité terrain certifiée (*Gold Standard*)** comprenant 46 359 stations validées, réparties sur 62 agglomérations.
""")

# ── Section 2 : L'Hybridation Multi-Sources ────────────────────────────────────
st.divider()
section(2, "L'Hybridation Multi-Sources : Modéliser l'Environnement Cyclable")

st.markdown(r"""
Bien que l'ontologie GBFS garantisse la localisation ponctuelle de l'offre matérielle, elle demeure agnostique quant aux déterminants environnementaux qui conditionnent la pratique cyclable. Le saut qualitatif de notre méthodologie – et le socle de l'IMD – réside dans la vectorisation et **l'enrichissement multidimensionnel (*Spatial Join* croisé)** de ces coordonnées avec six bases de données institutionnelles de référence.

Cette architecture de données hybride permet d'opérer une transition paradigmatique : il ne s'agit plus de mesurer un simple volume d'équipement (où sont les vélos ?), mais de **modéliser un système complexe** (le vélo est-il déployé dans un écosystème sécurisé, physiquement accessible et intégré aux autres modes de transport ?).
""")

# Tableau récapitulatif des sources
donnees_sources = pd.DataFrame({
    "Dimension Modélisée": [
        "Infrastructure Primaire (Offre)", 
        "Sécurité Spatiale (S)", 
        "Perméabilité Cyclable (I)", 
        "Capillarité Multimodale (M)", 
        "Friction Spatiale (T)", 
        "Vulnérabilité Socio-Économique",
        "Pratiques Comportementales"
    ],
    "Source de la donnée": [
        "APIs GBFS (Auditées)", 
        "Base BAAC (ONISR)", 
        "OpenStreetMap / Cerema", 
        "Point d'Accès National (GTFS)", 
        "NASA SRTM (30m)", 
        "INSEE (Dispositif Filosofi)",
        "FUB (2023) / INSEE EMP (2019)"
    ],
    "Format / Nature": [
        "GeoJSON point", 
        "Open Data Gouvernemental", 
        "Réseau filaire (Lignes)", 
        "Schedules & Stops", 
        "Modèle Numérique (MNT)", 
        "Carroyage (200m)",
        "Sondages & Enquêtes"
    ],
    "Intégration et Apport au Modèle Spatial": [
        "Coordonnées de vérité terrain et capacités ajustées post-correction.",
        "Densité de clusters d'accidents corporels à moins de 300m.",
        "Mesure de la continuité de l'aménagement en site propre.",
        "Distance aux pôles d'échanges lourds (Ferroviaire, BHNS, Tram).",
        "Gradient altimétrique pour modéliser la barrière énergétique.",
        "Revenus médians pour objectiver l'Indice d'Équité Sociale (IES).",
        "Convergence statistique validant l'efficience de l'IMD."
    ]
})

st.table(donnees_sources)

# ── Section 3 : Avant / Après ──────────────────────────────────────────────────
st.divider()
section(3, "Avant / Après : L'Impact Structurant de la Consolidation")

st.markdown(r"""
Pour saisir l'ampleur de la contribution de ce jeu de données (*Gold Standard*), il convient d'observer la métamorphose de l'information entre l'extraction initiale et l'architecture finale. Le fichier brut n'est qu'un inventaire logistique ; le fichier final est une véritable matrice de recherche socio-spatiale.
""")

col_before, col_after = st.columns(2)

with col_before:
    st.error("**AVANT : GBFS Brut (Approche Naïve)**")
    st.markdown("""
    * **Nature :** Fichiers JSON éclatés par opérateur.
    * **Bruit statistique :** Fort (Stations doublons, points géolocalisés en plein océan ou hors EPCI).
    * **Aberrations capacitives :** Prévalence du "999 vélos" pour les stations virtuelles, faussant totalement les moyennes.
    * **Absence de contexte :** Un point GPS nu. Impossible de savoir si la station se trouve sur une autoroute dangereuse ou au pied d'une gare.
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
    * **Signal purifié :** Application des filtres spatiaux et suppression des 6 anomalies.
    * **Redressement de l'offre :** Capacité recalculée par la moyenne conditionnelle locale.
    * **Environnement 360° :** Chaque station porte désormais son score de rugosité (MNT), sa distance au tramway (GTFS) et la qualité des pistes (OSM).
    * **Dimension Sociale :** Enrichissement par le revenu médian du carroyage INSEE.
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

st.info("**Conclusion méthodologique :** Les données sont intrinsèquement de *meilleure qualité* car elles reflètent la réalité physique du terrain, et elles sont *plus complètes* car elles intègrent les dimensions sécuritaires, topographiques et sociales indispensables à toute analyse d'équité.")


# ── Section 4 : Implication pour la Recherche ──────────────────────────────────
st.divider()
section(4, "Implication : L'Infrastructure de Données comme Objet de Recherche")

st.markdown("""
Dans le champ des études urbaines, le traitement des données est trop souvent relégué au rang de "détail technique". Cette recherche prouve au contraire que **la qualité de la donnée est éminemment politique**. 

En omettant de corriger les anomalies GBFS, un algorithme de planification publique conclurait à tort qu'une agglomération est parfaitement couverte grâce à des capacités artificiellement gonflées, justifiant potentiellement un arrêt des subventions pour l'aménagement cyclable de ce territoire. 

La mise à disposition de ce **Gold Standard au format `.parquet`** constitue donc une contribution académique autonome. Elle offre aux futurs chercheurs et géomaticiens un "socle de vérité terrain" déjà purgé de ses biais, prêt à supporter des modélisations complexes telles que la théorie des graphes ou l'analyse des flux de micromobilité.
""")