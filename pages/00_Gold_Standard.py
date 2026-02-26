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

# Tableau récapitulatif des sources (mis à jour avec un ton académique)
donnees_sources = pd.DataFrame({
    "Dimension Modélisée": [
        "Infrastructure Primaire (Offre)", 
        "Sécurité Spatiale (S)", 
        "Perméabilité Cyclable (I)", 
        "Capillarité Multimodale (M)", 
        "Friction Spatiale (T)", 
        "Vulnérabilité Socio-Économique",
        "Pratiques Comportementales (Validation)"
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
        "Réseau filaire (Lignes/Graphes)", 
        "Schedules & Stops (Noeuds)", 
        "Modèle Numérique de Terrain (MNT)", 
        "Carroyage Démographique (200m)",
        "Micro-données de sondage"
    ],
    "Intégration et Apport au Modèle Spatial": [
        "Coordonnées de vérité terrain et capacités ajustées post-correction (Le socle).",
        "Modélisation de l'exposition au risque via la densité de clusters d'accidents corporels (Rayon 300m).",
        "Mesure de la continuité de l'aménagement en site propre protégeant l'usager vulnérable.",
        "Calcul de la distance aux pôles d'échanges lourds (Ferroviaire, BHNS, Tram) pour évaluer la capacité de rabattement.",
        "Extraction du gradient altimétrique pour modéliser la friction spatiale et la barrière énergétique de l'usager.",
        "Analyse de la variance des revenus médians par quartier pour objectiver l'Indice d'Équité Sociale (IES).",
        "Analyse de convergence statistique pour valider l'efficience de l'IMD face au report modal réel."
    ]
})

st.table(donnees_sources)

# Tableau récapitulatif des sources
donnees_sources = pd.DataFrame({
    "Dimension Modélisée": [
        "Offre VLS (Le socle)", 
        "Sécurité Spatiale (S)", 
        "Infrastructure Continue (I)", 
        "Multimodalité (M)", 
        "Friction Spatiale (T)", 
        "Vulnérabilité Sociale",
        "Pratiques Réelles (Validation)"
    ],
    "Source de la donnée": [
        "APIs GBFS agrégées", 
        "Fichier BAAC (ONISR)", 
        "OpenStreetMap / Cerema", 
        "Point d'Accès National (GTFS)", 
        "NASA SRTM (30m)", 
        "INSEE (Filosofi)",
        "FUB / EMP 2019"
    ],
    "Type de Donnée": [
        "JSON/GeoJSON", 
        "Open Data Gouvernemental", 
        "Vecteur Spatial (Lignes)", 
        "Schedules / Stops", 
        "Modèle Numérique de Terrain (MNT)", 
        "Carroyage Socio-démographique",
        "Enquêtes / Statistiques"
    ],
    "Apport Scientifique pour le Modèle": [
        "Localisation exacte et capacité réelle des flottes.",
        "Cartographie des clusters d'accidents corporels cyclistes à moins de 300m.",
        "Mesure des aménagements en site propre protégeant l'usager vulnérable.",
        "Calcul de la distance aux pôles d'échanges (Train, Tram, BHNS) pour le 1er/dernier kilomètre.",
        "Calcul de la rugosité (dénivelé cumulé) mesurant l'effort physiologique requis.",
        "Revenu médian par quartier pour objectiver la présence de Déserts de Mobilité Sociale.",
        "Double validation de l'indice composite par le climat perçu et le report modal."
    ]
})

st.table(donnees_sources)

# ── Section 3 : Implication pour la Recherche ──────────────────────────────────
st.divider()
section(3, "Conclusion : L'Infrastructure de Recherche comme Contribution")

st.success("""
**Pourquoi cet effort d'ingénierie était-il indispensable ?**

Dans le champ des études urbaines, le traitement des données est trop souvent relégué au rang de "détail technique". Cette recherche prouve au contraire que **la donnée est éminemment politique**. 

En omettant de corriger les anomalies GBFS (notamment capacitives), un algorithme de planification publique conclurait à tort qu'une agglomération est parfaitement couverte, justifiant potentiellement un arrêt des subventions pour l'aménagement cyclable de ce territoire. 

La mise à disposition de ce **Gold Standard au format `.parquet`** constitue donc une contribution académique autonome. Elle offre aux futurs chercheurs et géomaticiens un "socle de vérité terrain" déjà purgé de ses biais, prêt à supporter des modélisations complexes telles que la théorie des graphes ou l'analyse des flux de micromobilité.
""")