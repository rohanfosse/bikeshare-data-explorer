"""
app.py — Point d'entrée de l'application Streamlit.
Atlas de l'Indice de Mobilité Douce (IMD) — Gold Standard GBFS.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import city_stats, load_stations, load_systems_catalog
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Atlas IMD — Justice Spatiale & Vélos en Libre-Service",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── En-tête ────────────────────────────────────────────────────────────────────
st.title("Atlas de l'Indice de Mobilité Douce (IMD)")
st.caption(
    "Évaluation quantitative de l'équité socio-spatiale des systèmes de vélos en libre-service "
    "en France — Gold Standard GBFS · CESI BikeShare-ICT · 2025-2026"
)

# ── Chargement des données (avant abstract pour valeurs dynamiques) ───────────
df      = load_stations()
catalog = load_systems_catalog()
cities  = city_stats(df)

n_certified = int((catalog["status"] == "ok").sum()) if "status" in catalog.columns else len(catalog)
n_cities    = df["city"].nunique()

abstract_box(
    "<b>Résumé de recherche :</b> Cette plateforme constitue l'interface de diffusion des résultats "
    "d'une recherche en géographie quantitative portant sur l'équité socio-spatiale des systèmes de "
    f"vélos en libre-service (VLS) français. À partir d'un corpus de <b>{len(df):,} stations certifiées</b> "
    f"(Gold Standard GBFS, {n_certified} systèmes, {n_cities} agglomérations), deux indices composites sont calibrés "
    "empiriquement : l'<b>Indice de Mobilité Douce (IMD)</b>, qui évalue la qualité physique "
    "de l'environnement cyclable à travers quatre dimensions (sécurité, infrastructure, multimodalité, "
    "topographie), et l'<b>Indice d'Équité Sociale (IES)</b>, qui mesure l'écart entre l'offre "
    "observée et l'offre socialement attendue. Les résultats clés invalident deux hypothèses "
    "intuitives dominantes : <b>(1)</b> l'absence d'autocorrélation spatiale significative "
    "(Moran's $I = -0{,}023$, $p = 0{,}765$) réfute le déterminisme géographique — c'est "
    "la gouvernance locale, non la localisation, qui explique les disparités ; "
    "<b>(2)</b> la corrélation de Spearman nulle ($\\rho = +0{,}055$, $p = 0{,}677$) entre IMD "
    "et revenu médian (INSEE Filosofi, 59 agglomérations dock-based) réfute le déterminisme "
    "économique — la quasi-totalité de la qualité VLS relève de choix politiques locaux."
)

sidebar_nav()

# ── KPIs calculés depuis les données réelles ───────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
n_dock = int((df["station_type"] == "docked_bike").sum()) if "station_type" in df.columns else len(df)
k1.metric("Stations Gold Standard (total)",    f"{len(df):,}")
k2.metric("Dont stations dock-based (VLS)",    f"{n_dock:,}")
k3.metric("Systèmes GBFS certifiés",           f"{n_certified}")
k4.metric("Moran's I (autocorrélation spatiale)", "−0,023 (ns)")
k5.metric("ρ Spearman IMD × Revenu",           "+0,055 (ns)", "Indépendance totale")

# ── Section 1 : Contexte et Problématique ──────────────────────────────────────
st.divider()
section(1, "Contexte Politique et Problématique Scientifique")

st.markdown(r"""
La décarbonation des transports urbains constitue l'un des défis majeurs de la décennie. En France,
sous l'impulsion de la Loi d'Orientation des Mobilités (LOM, 2019) et du Plan Vélo 2023–2027,
le déploiement des Systèmes de Vélos en Libre-Service (SVLS) est devenu un axe central des
politiques d'aménagement public. Cependant, la simple prolifération quantitative de ces flottes
ne garantit ni l'efficacité multimodale, ni l'inclusion socio-spatiale.

Face au risque d'une transition écologique à deux vitesses, cette recherche pose une question
fondamentale : **dans quelle mesure les réseaux de vélos partagés actuels atténuent-ils ou
aggravent-ils les fractures socio-spatiales préexistantes ?**

Pour y répondre, il apparaît impératif de s'affranchir du prisme purement capacitaire qui a
jusqu'ici dominé l'évaluation des politiques cyclables. Historiquement, la littérature académique
et les planificateurs urbains se sont appuyés sur des métriques volumétriques naïves (densité brute
de stations, ratio de vélos par habitant) calculées à partir de flux de données ouverts (GBFS)
rarement audités. Ce postulat — qui associe implicitement l'abondance de l'offre à son utilité
sociale — masque des biais structurels majeurs : un réseau dense peut s'avérer inopérant s'il est
déconnecté des pôles d'échanges multimodaux, ou inéquitable s'il exclut systématiquement les
quartiers à forte vulnérabilité économique (*Médard de Chardon et al., 2017*).

Afin de pallier ces lacunes méthodologiques, cet article propose une approche quantitative inédite,
structurée autour de :
1. La constitution d'un jeu de données de référence — le **Gold Standard** — expurgé des anomalies
   inhérentes à l'Open Data GBFS.
2. La calibration empirique d'un **Indice de Mobilité Douce (IMD)** intégrant la friction spatiale,
   l'accidentologie, la continuité des infrastructures et l'hybridation multimodale.
3. L'introduction d'un **Indice d'Équité Sociale (IES)** permettant de cartographier formellement
   les "Déserts de Mobilité Sociale" et d'orienter vers une gouvernance cyclable plus juste.
""")

# ── Section 2 : L'Urgence de l'Audit GBFS ─────────────────────────────────────
st.divider()
section(2, "L'Urgence de l'Audit des Données Ouvertes (GBFS) : Du Bruit au Signal")

st.markdown(r"""
La littérature académique s'appuie de manière croissante sur des flux de données ouverts au standard
GBFS (*General Bikeshare Feed Specification*, v3.0). Toutefois, nos travaux démontrent que
l'utilisation naïve de ces données brutes est **scientifiquement erronée**. L'audit systématique
des {N_CATALOG} systèmes français a révélé une taxonomie de 5 classes d'anomalies structurelles (A1 à A5).

L'anomalie A3 — le *biais de la moyenne conditionnelle* inhérent aux flottes *free-floating* — est
la plus pernicieuse : elle engendre des surestimations massives de capacité
($\bar{c}_{\text{profil}} \gg \bar{c}_{\text{réel}}$), invalidant les classements de performance
de plusieurs métropoles. À titre d'illustration, Bordeaux passait du rang 2 au rang 14 national
après correction de ce seul biais, ce qui aurait pu conduire à des réallocations de subventions
publiques erronées de plusieurs millions d'euros.

En purgeant rigoureusement les données de ces biais algorithmiques, nous avons constitué un
**Gold Standard** de {N_STATIONS} stations validées sur {N_CITIES} agglomérations — socle indispensable à toute
modélisation spatiale robuste. Ce jeu de données est mis à disposition de la communauté
scientifique via l'interface d'export de cette plateforme (formats CSV et Parquet, principes FAIR).
""".replace("{N_CATALOG}", str(len(catalog))).replace("{N_STATIONS}", f"{len(df):,}").replace("{N_CITIES}", str(n_cities)))

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**Bilan de l'Audit GBFS**")
    audit_rows = [
        {"Étape": "Systèmes GBFS bruts disponibles",   "Valeur": "125"},
        {"Étape": "Exclusion A1 (Autopartage Citiz)",    "Valeur": "−14"},
        {"Étape": "Exclusion A4 & A5 (Géo / Périmètre)", "Valeur": "−7"},
        {"Étape": "Micro-réseaux exclus (< 20 stations)","Valeur": "−20 (approx.)"},
        {"Étape": "Systèmes Gold Standard certifiés",   "Valeur": str(n_certified)},
        {"Étape": "Stations Gold Standard certifiées",  "Valeur": f"{len(df):,}"},
        {"Étape": "Agglomérations couvertes",           "Valeur": str(df['city'].nunique())},
    ]
    st.table(pd.DataFrame(audit_rows))

with col_r:
    st.markdown("**Anomalies GBFS — Taxonomie A1–A5**")
    anomaly_rows = [
        {"Classe": "A1", "Nature": "Inclusion hors-domaine (autopartage)", "Impact": "Biais de classification"},
        {"Classe": "A2", "Nature": "Capacité fictive (placeholder)",        "Impact": "Surestimation capacitaire"},
        {"Classe": "A3", "Nature": "Biais floating-anchor (moyenne cond.)", "Impact": "Surestimation massive IMD"},
        {"Classe": "A4", "Nature": "Aberrations géospatiales (Lat/Lon)",    "Impact": "Biais topologique"},
        {"Classe": "A5", "Nature": "Hors périmètre (DOM-TOM / macro-rég.)", "Impact": "Artefacts de distribution"},
    ]
    st.table(pd.DataFrame(anomaly_rows))

# ── Section 3 : Architecture analytique ───────────────────────────────────────
st.divider()
section(3, "Architecture Analytique — Cinq Axes de Recherche Complémentaires")

st.markdown(r"""
La recherche est structurée en cinq axes analytiques complémentaires, progressant de l'ingénierie
des données vers la modélisation spatiale, puis vers l'évaluation de la justice sociale.
""")

axes = [
    {
        "Axe":         "Axe Prél.",
        "Page":        "Gold Standard",
        "Question":    "L'Open Data GBFS est-il un matériau de recherche fiable ?",
        "Méthode":     "Audit multi-systèmes, taxonomie A1–A5, pipeline de purge en 6 étapes",
        "Résultat clé":f"{len(df):,} stations certifiées — Bordeaux : rang 2 → 14 après correction",
    },
    {
        "Axe":         "Axe 1",
        "Page":        "IMD",
        "Question":    "La qualité cyclable se réduit-elle au volume de stations ?",
        "Méthode":     "Indice composite 4D (S, I, M, T), optimisation supervisée, Monte Carlo $N=10\\,000$",
        "Résultat clé":"Top 10 stable dans 89 % des simulations — $w_M^* = 0{,}578$",
    },
    {
        "Axe":         "Axe 2",
        "Page":        "IES",
        "Question":    "L'offre cyclable est-elle équitablement distribuée socialement ?",
        "Méthode":     "Modèle Ridge ($\\lambda$ par CV), $\\text{IES}_i = \\text{IMD}_{\\text{obs}} / \\widehat{\\text{IMD}}(R_m)$",
        "Résultat clé":"$R^2 = 0{,}28$ — 72 % de la variance IMD relèvent de la gouvernance locale",
    },
    {
        "Axe":         "Axe 3",
        "Page":        "Villes",
        "Question":    "Les disparités inter-urbaines sont-elles géographiques ou politiques ?",
        "Méthode":     "Indice global de Moran (autocorrélation spatiale), analyse comparative",
        "Résultat clé":"Moran's $I = -0{,}023$ ($p = 0{,}765$) — déterminisme géographique invalidé",
    },
    {
        "Axe":         "Axe 4",
        "Page":        "Distributions",
        "Question":    "La taille d'une agglomération prédit-elle sa performance cyclable ?",
        "Méthode":     "Corrélation de Spearman, boîtes à encoches, matrice de corrélation",
        "Résultat clé":"$r_s = -0{,}02$ (hors Paris) — aucune corrélation taille–performance",
    },
    {
        "Axe":         "Axe 5",
        "Page":        "Montpellier",
        "Question":    "Les modèles nationaux se valident-ils à l'échelle micro-locale ?",
        "Méthode":     "Théorie des graphes (Louvain, PageRank), GTFS, IES intra-urbain par quartier",
        "Résultat clé":"Structure bimodale Commuter confirmée — fracture socio-spatiale cartographiée",
    },
]
st.table(pd.DataFrame(axes))
st.caption(
    "**Tableau 3.1.** Architecture des cinq axes de recherche. "
    "Chaque axe correspond à une page dédiée dans la barre de navigation latérale. "
    "Les pages *Carte*, *France* et *Export* constituent des modules transversaux "
    "(visualisation spatiale, validation multi-sources et diffusion FAIR des données)."
)

# ── Section 4 : Résultats clés ────────────────────────────────────────────────
st.divider()
section(4, "Résultats Clés — Invalidation de Deux Hypothèses Intuitives Majeures")

st.markdown(r"""
Deux résultats contre-intuitifs structurent l'ensemble de la contribution :

#### 4.1. L'Absence de Déterminisme Géographique (Moran's $I = -0{,}023$, $p = 0{,}765$)

L'hypothèse implicitement dominante en géographie urbaine postule qu'une agglomération
"bien située" — c'est-à-dire bénéficiant d'une forte densité et d'une tradition de mobilité
douce — est condamnée à la performance cyclable, tandis que les villes périphériques seraient
structurellement pénalisées. L'indice de Moran appliqué aux scores IMD des {N_CITIES} agglomérations
**réfute formellement cette hypothèse** : les villes performantes et sous-performantes ne forment
pas de clusters territoriaux cohérents. Bordeaux, Rennes et Grenoble, très éloignées
géographiquement, peuvent atteindre des scores comparables, quand des villes voisines présentent
des écarts considérables. **Les choix de gouvernance locale — politique tarifaire, densification
ciblée, intégration multimodale — priment sur le déterminisme géographique.**

#### 4.2. L'Absence de Déterminisme Économique ($R^2_{\text{Ridge}} = 0{,}28$)

L'hypothèse symétrique suggère que les agglomérations les plus riches auraient
"les moyens de leurs ambitions cyclables". Le modèle Ridge réfute cette vision déterministe :
le revenu médian n'explique que **28 % de la variance de l'IMD**. Des agglomérations à revenu
modeste atteignent d'excellents scores IMD grâce à des politiques tarifaires inclusives,
une planification stratégique des stations en pôles multimodaux et des partenariats actifs avec
les opérateurs GBFS. À l'inverse, des agglomérations aisées présentent des scores décevants
par sous-investissement dans la continuité des infrastructures. **La gouvernance locale prime
sur le capital économique.**

Ces deux résultats convergent vers une conclusion politique forte : toute agglomération,
quelle que soit sa situation géographique ou économique, dispose d'une **marge d'action
significative** pour améliorer l'équité et la qualité de son environnement cyclable.
""".replace("{N_CITIES}", str(n_cities)))

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.metric("Indice de Moran (IMD spatial)", "I = −0,023", "p = 0,765 — non significatif", delta_color="off")
    st.caption("Absence d'autocorrélation spatiale — les disparités ne sont pas géographiquement déterminées.")
with col_r2:
    st.metric("R² Ridge (revenu médian → IMD)", "0,28", "72 % expliqués par la gouvernance", delta_color="off")
    st.caption("Absence de déterminisme économique — la qualité cyclable est un choix politique.")

# ── Section 5 : Guide de Navigation ───────────────────────────────────────────
st.divider()
section(5, "Guide de Navigation — Parcours de Recherche et Modules Analytiques")

st.markdown(r"""
La plateforme est organisée en modules thématiques accessibles depuis la barre de navigation
latérale. Chaque module correspond à un axe de recherche ou à un outil transversal.
""")

nav_rows = [
    {"Module": "Gold Standard",       "Axe":      "Prél.",  "Contenu principal": f"Taxonomie des anomalies GBFS (A1–A5), pipeline de purge en 6 étapes, complétude de l'enrichissement, catalogue des {n_certified} systèmes certifiés."},
    {"Module": "IMD",                 "Axe":      "1",      "Contenu principal": "Formulation mathématique, poids optimaux, Monte Carlo ($N=10\\,000$), classement national, décomposition (S, I, M, T), validation FUB/EMP."},
    {"Module": "IES",                 "Axe":      "2",      "Contenu principal": "Formalisation de l'IES, modèle Ridge, matrice de justice cyclable (4 quadrants), validation empirique IMD × part modale réelle."},
    {"Module": "Carte",               "Axe":      "Trans.", "Contenu principal": f"Visualisation WebGL des {len(df):,} stations (pydeck), filtrage par dimension d'enrichissement, heatmap de densité et couches thématiques."},
    {"Module": "Villes",              "Axe":      "3",      "Contenu principal": "Classement univarié, nuage infra × sinistralité (Moran's $I$), profil radar multi-dimensionnel comparatif."},
    {"Module": "Distributions",       "Axe":      "4",      "Contenu principal": "Histogrammes, boîtes à moustaches à encoches, matrice de corrélation de Spearman ($\\rho$), scatter matriciel stratifié."},
    {"Module": "France",              "Axe":      "Trans.", "Contenu principal": "Triangulation FUB 2023, EMP 2019, éco-compteurs, BAAC et Cerema — indicateurs nationaux de la mobilité cyclable."},
    {"Module": "Montpellier",         "Axe":      "5",      "Contenu principal": "Topologie Louvain, déséquilibres source/puits, vulnérabilité structurelle $V_i$, intégration GTFS tramway, fracture socio-spatiale IES intra-urbain."},
    {"Module": "Export",              "Axe":      "FAIR",   "Contenu principal": "Accès libre au Gold Standard (CSV UTF-8 / Parquet), filtres multi-critères, dictionnaire de variables, métadonnées de citation."},
]
st.table(pd.DataFrame(nav_rows))
st.caption(
    "**Tableau 5.1.** Guide de navigation par module analytique. "
    "Les modules 'Trans.' sont transversaux et ne sont pas rattachés à un axe de recherche unique. "
    "Le module 'FAIR' implémente les principes *Findable, Accessible, Interoperable, Reusable* "
    "pour la diffusion académique du corpus Gold Standard."
)
