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
La décarbonation des transports urbains constitue l'un des défis majeurs de la décennie. En France, sous l'impulsion de la Loi d'Orientation des Mobilités (LOM, 2019) et du Plan Vélo 2023–2027, le déploiement des Systèmes de Vélos en Libre-Service (SVLS) est devenu un axe central des politiques d'aménagement public. Cependant, la simple prolifération quantitative de ces flottes ne garantit ni l'efficacité multimodale, ni l'inclusion socio-spatiale. Face au risque d'une transition écologique à deux vitesses, cette recherche pose une question fondamentale : dans quelle mesure les réseaux de vélos partagés actuels atténuent-ils ou aggravent-ils les fractures socio-spatiales préexistantes ?

Pour y répondre, il apparaît impératif de s'affranchir du prisme purement capacitaire qui a jusqu'ici dominé l'évaluation des politiques cyclables. Historiquement, la littérature académique et les planificateurs urbains se sont appuyés sur des métriques volumétriques naïves (densité brute de stations, ratio de vélos par habitant) calculées à partir de flux de données ouverts (GBFS) rarement audités. Ce postulat, qui associe implicitement l'abondance de l'offre à son utilité sociale, masque des biais structurels majeurs : un réseau dense peut s'avérer inopérant s'il est déconnecté des pôles d'échanges multimodaux, ou inéquitable s'il exclut systématiquement les quartiers à forte vulnérabilité économique.

Afin de pallier ces lacunes méthodologiques, cet article propose une approche quantitative inédite, structurée autour de la constitution d'un jeu de données de référence (Gold Standard) expurgé des anomalies inhérentes à l'Open Data. À travers la calibration empirique d'un Indice de Mobilité Douce (IMD) – qui intègre la friction topographique, l'exposition au risque d'accidentologie, la continuité des infrastructures et l'hybridation multimodale –, nous modélisons objectivement la performance spatiale des réseaux français. La confrontation de cet indice physique aux disparités socio-économiques locales permet in fine d'introduire un Indice d'Équité Sociale (IES), outil de diagnostic spatial capable d'identifier formellement les « Déserts de Mobilité Sociale » et d'orienter vers une gouvernance cyclable plus juste.
""")

# ── Section 2 : L'Urgence de l'Audit des Données Ouvertes (GBFS) ───────────────
st.divider()
section(2, "L'Urgence de l'Audit des Données Ouvertes (GBFS)")

st.markdown(r"""
La littérature académique s'appuie de manière croissante sur des flux de données ouverts au standard GBFS (General Bikeshare Feed Specification). Toutefois, nos travaux démontrent que l'utilisation naïve de ces données brutes est scientifiquement erronée. Nous avons identifié et formalisé une taxonomie de 6 classes d'anomalies structurelles (A1 à A6) inhérentes à ces flux. À titre d'exemple, l'anomalie A3 (calcul de la moyenne conditionnelle des capacités pour les systèmes floating-anchor) engendre des biais de surestimation massifs, invalidant les classements de performance spatiale de plusieurs métropoles si elle n'est pas corrigée.

En purgeant rigoureusement les données de ces biais algorithmiques et structurels, nous avons construit un jeu de données de référence (Gold Standard), regroupant 46 359 stations validées sur 62 agglomérations. Ce socle fiabilisé constitue le prérequis indispensable à toute modélisation spatiale. Sans cette étape d'assainissement systématique, toute tentative d'évaluation de l'équité ou de l'intégration multimodale se heurterait à d'importants artefacts statistiques. Les surévaluations capacitives locales, induites par les anomalies de l'Open Data, dissimuleraient les véritables disparités de maillage et pourraient conduire les décideurs publics à formuler des diagnostics territoriaux biaisés. Par conséquent, l'ouverture et la documentation de ce Gold Standard dépassent la simple exigence de reproductibilité académique : elles fournissent à la communauté des chercheurs en géographie urbaine un nouveau cadre d'audit des flux de micromobilité. C'est exclusivement sur cette infrastructure de données validées que peut s'opérer la calibration de notre modèle d'évaluation, garantissant ainsi la robustesse des indicateurs qui en découlent, à l'instar de l'Indice de Mobilité Douce (IMD).
""")

# ── Section 3 : L'Indice de Mobilité Douce (IMD) ───────────────────────────────
st.divider()
section(3, "Modélisation Spatiale et Évaluation de la Justice Socio-Écologique")

st.markdown(r"""
Pour dépasser les limites inhérentes aux métriques volumétriques, nous avons conçu et calibré empiriquement un Indice de Mobilité Douce (IMD). Ce modèle mathématique composite évalue la performance structurelle des réseaux à travers l'agrégation de quatre dimensions environnementales décisives : l'exposition au risque (densité d'accidents), la continuité de l'infrastructure sécurisée, la friction spatiale (rugosité topographique) et l'hybridation multimodale (intégration spatiale aux réseaux de transports lourds). Afin d'éviter l'écueil des pondérations arbitraires, les paramètres de ce modèle ont été optimisés de manière supervisée et leur stabilité a été validée par des simulations de Monte Carlo, garantissant ainsi une forte corrélation avec les pratiques cyclables réelles mesurées sur le terrain.

Toutefois, une mesure d'offre purement physique demeure insuffisante pour statuer sur l'inclusivité d'une politique d'aménagement. Par conséquent, l'ultime étape de notre cadre analytique consiste à confronter la performance de ces réseaux à la vulnérabilité socio-économique des territoires. En modélisant l'offre attendue en fonction du revenu médian local via une régression pénalisée (modèle Ridge), nous introduisons l'Indice d'Équité Sociale (IES). Ce ratio évalue l'écart entre le déploiement physique de la flotte et le besoin social prédictible. Il permet de cartographier formellement les « Déserts de Mobilité Sociale » – des zones géographiques de relégation où se cumulent précarité financière et absence de report modal sécurisé. En isolant ces territoires de captivité automobile, cet article vise à fournir aux décideurs publics un outil de diagnostic spatial probant, indispensable pour réorienter les investissements vers une justice socio-écologique mesurable.
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