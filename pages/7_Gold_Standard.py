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
Le format **GBFS** (*General Bikeshare Feed Specification*) s'est imposé comme le standard international pour la diffusion en temps réel de l'état des systèmes de vélos en libre-service. Si cette standardisation a facilité l'émergence d'applications de type *MaaS* (Mobility as a Service), elle dissimule des **hétérogénéités structurelles profondes** d'un opérateur à l'autre.

Une utilisation naïve de ces données brutes dans un algorithme de recherche induit des biais d'évaluation massifs. L'audit complet des 125 systèmes français a révélé une taxonomie de **6 classes d'anomalies (A1 à A6)** que nous avons dû purger algorithmiquement :

1. **A1 — Doublons spatio-temporels :** Enregistrements multiples d'une même station dus à des désynchronisations d'API.
2. **A2 — Stations fantômes (Zombies) :** Stations déclarées dans le fichier `station_information.json` mais physiquement inexistantes ou désactivées dans les faits.
3. **A3 — L'Illusion Capacitive (*Floating-Anchor*) :** L'anomalie la plus critique. Pour les systèmes hybrides (vélos *free-floating* pouvant s'attacher à du mobilier urbain générique), certains opérateurs déclarent arbitrairement une "capacité de 999" vélos par point d'ancrage. Non corrigé, ce biais surestime artificiellement la densité de réseaux comme ceux de Bordeaux ou de la métropole lilloise de plus de 90 %.
4. **A4 — Incohérence de typologie :** Flou entre vélos mécaniques et électriques.
5. **A5 — Dérive Géospatiale :** Coordonnées (Lat/Lon) projetées en dehors des limites administratives de l'agglomération (erreur de géocodage).
6. **A6 — Absence de standardisation des ID :** Rend impossible le suivi longitudinal d'une station.

**Bilan de la consolidation :** L'application de nos filtres de redressement (correction par la moyenne conditionnelle pour A3, géofiltres pour A5) a permis de passer d'un "bruit statistique" à un jeu de données certifié de **46 359 stations validées** sur 62 agglomérations.
""")

# ── Section 2 : L'Hybridation Multi-Sources ────────────────────────────────────
st.divider()
section(2, "L'Hybridation Multi-Sources : Modéliser l'Environnement Cyclable")

st.markdown(r"""
Si le GBFS permet de localiser le vélo, il ne dit rien de l'environnement dans lequel il évolue. L'innovation de notre démarche (et le cœur de l'IMD) repose sur **l'enrichissement spatial** (*Spatial Join*) des coordonnées des stations avec 6 bases de données institutionnelles indépendantes. 

Cette hybridation croisée permet de passer d'une vision "matérielle" (où sont les vélos ?) à une vision "écosystémique" (le vélo est-il sûr, utile et accessible ?).
""")

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