"""
app.py — Point d'entrée de l'application Streamlit.
Présentation du projet de recherche, de la méthodologie d'audit GBFS
et de la problématique de justice socio-spatiale.
"""
import streamlit as st

from utils.styles import abstract_box, inject_css, section, sidebar_nav

# ── Configuration de la page ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Atlas IMD — Justice Spatiale & Vélos en Libre-Service",
    page_icon="🚲",
    layout="wide",
)
inject_css()

# ── En-tête de l'application ───────────────────────────────────────────────────
st.title("Atlas de l'Indice de Mobilité Douce (IMD)")
st.caption("Évaluation de l'équité socio-spatiale des systèmes de vélos en libre-service en France")

abstract_box(
    "Bienvenue sur l'explorateur interactif du <b>Gold Standard Dataset</b> de la micromobilité française. "
    "Cet outil de recherche ouvert accompagne notre publication scientifique portant sur la justice socio-écologique. "
    "Il met à disposition de la communauté académique et des planificateurs urbains une interface de visualisation, "
    "d'analyse et d'export des données auditées issues de 125 systèmes de vélos partagés."
)

sidebar_nav()

# ── Section 1 : Contexte et Problématique ──────────────────────────────────────
st.divider()
section(1, "Contexte Politique et Problématique Scientifique")

st.markdown(r"""
La décarbonation des transports urbains constitue l'un des défis majeurs de la décennie. En France, sous l'impulsion de la Loi d'Orientation des Mobilités (LOM, 2019) et du Plan Vélo 2023–2027, le déploiement des Systèmes de Vélos en Libre-Service (SVLS) est devenu un axe central des politiques d'aménagement public. 

Cependant, la simple prolifération quantitative de ces flottes ne garantit ni l'efficacité multimodale, ni l'inclusion socio-spatiale. Face au risque d'une transition écologique à deux vitesses, cette recherche pose une question fondamentale : **dans quelle mesure les réseaux de vélos partagés actuels atténuent-ils ou aggravent-ils les fractures socio-spatiales préexistantes ?**
""")

# ── Section 2 : L'Urgence de l'Audit des Données Ouvertes (GBFS) ───────────────
st.divider()
section(2, "L'Urgence de l'Audit des Données Ouvertes (GBFS)")

st.markdown(r"""
La littérature académique s'appuie de manière croissante sur des flux de données ouverts au standard GBFS (*General Bikeshare Feed Specification*). Toutefois, nos travaux démontrent que l'utilisation naïve de ces données brutes est scientifiquement erronée.

Nous avons identifié et formalisé une taxonomie de **6 classes d'anomalies structurelles** (A1 à A6) inhérentes à ces flux. À titre d'exemple, l'anomalie *A3* (calcul de la moyenne conditionnelle des capacités pour les systèmes *floating-anchor*) engendre des biais de surestimation massifs, invalidant les classements de performance spatiale de plusieurs métropoles si elle n'est pas corrigée.

En purgeant rigoureusement les données de ces biais algorithmiques et structurels, nous avons construit un jeu de données de référence (*Gold Standard*), regroupant 46 359 stations validées sur 62 agglomérations. Ce socle fiabilisé constitue le prérequis indispensable à toute modélisation spatiale.
""")

# ── Section 3 : L'Indice de Mobilité Douce (IMD) ───────────────────────────────
st.divider()
section(3, "Vers une Mesure de l'Équité : L'Indice IMD et l'IES")

st.markdown(r"""
Afin d'évaluer objectivement l'offre cyclable, nous avons développé et calibré empiriquement l'**Indice de Mobilité Douce (IMD)**. Ce modèle mathématique composite dépasse le simple comptage capacitaire en intégrant :
* La couverture spatiale et la densité du maillage.
* L'hybridation des flottes (Multimodalité : *dock-based*, *semi-dock*, *free-floating*).
* La friction spatiale locale (Rugosité topographique issue des MNT).
* L'écosystème de risque (Accidentologie BAAC) et la continuité de l'infrastructure cyclable sécurisée.

La confrontation de cet indice d'offre avec la vulnérabilité socio-économique locale (via un modèle de régression Ridge) permet de générer un **Indice d'Équité Sociale (IES)**, révélant la présence de **« Déserts de Mobilité Sociale »** au sein des territoires urbains.
""")

# ── Section 4 : Navigation ─────────────────────────────────────────────────────
st.divider()
st.info(
    "**Parcours de Recherche (Navigation latérale) :**\n\n"
    "* **0_IMD :** Formulation mathématique formelle du modèle, décomposition des dimensions et classement national.\n"
    "* **1_Carte & 2_Villes :** Cartographie interactive des 46 359 stations et analyse de l'autocorrélation spatiale (indice global de Moran).\n"
    "* **3_Distributions :** Analyse statistique des disparités (démontrant notamment l'absence de corrélation significative entre l'échelle démographique et la performance cyclable, $r_s = -0{,}02$).\n"
    "* **6_Montpellier :** Étude de cas permettant une validation micro-locale de la friction spatiale et de l'intégration à l'écosystème de transport lourd (GTFS)."
)