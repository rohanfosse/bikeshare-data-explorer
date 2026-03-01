"""
9_References.py — Références bibliographiques, sources de données et citation.
Atlas IMD — Justice Spatiale & Vélos en Libre-Service · CESI BikeShare-ICT · 2025–2026
"""
from __future__ import annotations

import streamlit as st

from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Références — Atlas IMD",
    layout="wide",
)
inject_css()

st.title("Références et Sources")
st.caption(
    "Bibliographie académique, sources de données institutionnelles, "
    "stack technique et modalités de citation — Gold Standard GBFS · CESI BikeShare-ICT · 2025–2026"
)

abstract_box(
    "Cette page recense l'ensemble des références mobilisées dans la recherche : "
    "les travaux académiques fondateurs sur l'équité des systèmes de vélos en libre-service, "
    "les sources de données institutionnelles ayant alimenté le Gold Standard GBFS, "
    "et le stack technique open-source utilisé pour la construction des indices IMD et IES. "
    "Les modalités de citation de cette plateforme et du corpus Gold Standard sont précisées en fin de page."
)

sidebar_nav()

# ── Section 1 : Références académiques ─────────────────────────────────────────
st.divider()
section(1, "Références Académiques")

st.markdown("""
Les références ci-dessous correspondent aux travaux cités dans le texte de la plateforme.
Elles sont présentées par ordre alphabétique du premier auteur.
""")

refs = [
    {
        "Référence": "Bachand-Marleau et al. (2012)",
        "Citation complète": (
            "Bachand-Marleau J., Lee B.H.Y. & El-Geneidy A.M. (2012). "
            "Better Understanding of Factors Influencing Likelihood of Using Shared Bicycle Systems "
            "and Frequency of Use. *Transportation Research Record*, 2314(1), 66–71. "
            "https://doi.org/10.3141/2314-09"
        ),
        "Pertinence": "Facteurs d'adoption des SVLS, accessibilité multimodale",
    },
    {
        "Référence": "Bai & Jiao (2020)",
        "Citation complète": (
            "Bai S. & Jiao J. (2020). Dockless E-scooter usage patterns and the influence "
            "of built environment factors. *Cities*, 98, 102521. "
            "https://doi.org/10.1016/j.cities.2019.102521"
        ),
        "Pertinence": "Systèmes free-floating — dynamiques spatiales structurellement différentes des dock-based",
    },
    {
        "Référence": "Médard de Chardon et al. (2017)",
        "Citation complète": (
            "Médard de Chardon C., Caruso G. & Thomas I. (2017). "
            "Bicycle sharing system 'success' determinants. "
            "*Transportation Research Part A: Policy and Practice*, 100, 202–214. "
            "https://doi.org/10.1016/j.tra.2017.04.020"
        ),
        "Pertinence": (
            "Référence fondatrice sur les déterminants de performance des SVLS. "
            "R²_Ridge ≈ 0,28 sur panel international (contexte anglo-saxon). "
            "Biais structurels liés à l'utilisation naïve des données GBFS."
        ),
    },
    {
        "Référence": "Moran (1950)",
        "Citation complète": (
            "Moran P.A.P. (1950). Notes on Continuous Stochastic Phenomena. "
            "*Biometrika*, 37(1/2), 17–23. "
            "https://doi.org/10.2307/2332142"
        ),
        "Pertinence": "Indice de Moran — mesure d'autocorrélation spatiale globale (I = −0,023, p = 0,765)",
    },
    {
        "Référence": "Riley et al. (1999)",
        "Citation complète": (
            "Riley S.J., DeGloria S.D. & Elliot R. (1999). "
            "A Terrain Ruggedness Index That Quantifies Topographic Heterogeneity. "
            "*Intermountain Journal of Sciences*, 5(1–4), 23–27."
        ),
        "Pertinence": "Terrain Ruggedness Index (TRI) — composante T de l'IMD, données SRTM 30 m",
    },
    {
        "Référence": "Spearman (1904)",
        "Citation complète": (
            "Spearman C. (1904). The Proof and Measurement of Association between Two Things. "
            "*The American Journal of Psychology*, 15(1), 72–101. "
            "https://doi.org/10.2307/1412159"
        ),
        "Pertinence": "Corrélation de rang — tests non paramétriques (ρ IMD × revenu = +0,055, p = 0,677)",
    },
]

for r in refs:
    with st.expander(r["Référence"]):
        st.markdown(r["Citation complète"])
        st.caption(f"**Pertinence dans cette recherche :** {r['Pertinence']}")

# ── Section 2 : Sources de données institutionnelles ───────────────────────────
st.divider()
section(2, "Sources de Données Institutionnelles")

st.markdown("""
Le Gold Standard GBFS est construit par hybridation de sept modules de données ouvertes.
Chaque source est documentée ci-dessous avec son millésime, sa granularité et les variables mobilisées.
""")

import pandas as pd

sources = [
    {
        "Source": "GBFS — MobilityData",
        "Millésime": "2024–2025",
        "Granularité": "Station",
        "Variables mobilisées": "Capacité (num_docks_available), lat/lon, station_type",
        "URL / Référence": "https://github.com/MobilityData/gbfs",
        "Rôle dans l'étude": "Corpus principal — 46 312 stations certifiées, 122 systèmes français",
    },
    {
        "Source": "INSEE Filosofi",
        "Millésime": "2019",
        "Granularité": "Carreau 200 m (INSPIRE)",
        "Variables mobilisées": "Revenu médian/UC, indice de Gini, taux de pauvreté",
        "URL / Référence": "https://www.insee.fr/fr/statistiques/4507787",
        "Rôle dans l'étude": "IES — déterminisme économique (ρ Spearman, modèle Ridge)",
    },
    {
        "Source": "INSEE Recensement de la Population (RP)",
        "Millésime": "2020",
        "Granularité": "Carreau 200 m",
        "Variables mobilisées": "Part ménages sans voiture, part modale vélo",
        "URL / Référence": "https://www.insee.fr/fr/statistiques/5650714",
        "Rôle dans l'étude": "Composante I (Infrastructure) — demande cyclable de proximité",
    },
    {
        "Source": "GTFS national (SNCF / RATP / Métropoles)",
        "Millésime": "2024",
        "Granularité": "Arrêt de transport en commun",
        "Variables mobilisées": "Géolocalisation des arrêts, lignes desservies",
        "URL / Référence": "https://transport.data.gouv.fr/",
        "Rôle dans l'étude": "Composante M (Multimodalité) — accessibilité à 300 m (poids w_M* = 0,578)",
    },
    {
        "Source": "BAAC — ONISR (Min. Intérieur)",
        "Millésime": "2020–2023",
        "Granularité": "Accident individuel géolocalisé",
        "Variables mobilisées": "Localisation, type d'usager, gravité",
        "URL / Référence": "https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere/",
        "Rôle dans l'étude": "Composante S (Sécurité) — densité d'accidents cyclables dans un rayon de 300 m",
    },
    {
        "Source": "SRTM — NASA / USGS",
        "Millésime": "2000 (30 m GSD)",
        "Granularité": "Pixel raster 30 m",
        "Variables mobilisées": "Altitude (MNT), TRI (Terrain Ruggedness Index)",
        "URL / Référence": "https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm",
        "Rôle dans l'étude": "Composante T (Topographie) — friction altimétrique (poids w_T* = 0,096)",
    },
    {
        "Source": "FUB — Baromètre des villes cyclables",
        "Millésime": "2023",
        "Granularité": "Agglomération",
        "Variables mobilisées": "Note globale perçue (/6), sous-scores thématiques",
        "URL / Référence": "https://www.fub.fr/velo-ville/barometre-villes-cyclables/resultats-2023",
        "Rôle dans l'étude": "Validation externe de l'IMD (validité de face, corrélation Spearman positive)",
    },
    {
        "Source": "EMP — Enquête Mobilité des Personnes (SDES)",
        "Millésime": "2019",
        "Granularité": "Agglomération / Ménage",
        "Variables mobilisées": "Part modale vélo, nombre de déplacements/jour",
        "URL / Référence": "https://www.statistiques.developpement-durable.gouv.fr/enquete-nationale-transports-et-deplacements-entd-2008",
        "Rôle dans l'étude": "Validation externe de l'IMD (validité prédictive — lien offre/usage observé)",
    },
]

df_sources = pd.DataFrame(sources)
st.dataframe(
    df_sources[["Source", "Millésime", "Granularité", "Variables mobilisées", "Rôle dans l'étude"]],
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "**Tableau 2.1.** Sources de données institutionnelles mobilisées pour la construction "
    "du Gold Standard GBFS et des indices IMD / IES. Toutes les sources sont librement accessibles "
    "sous licences ouvertes (Etalab / ODbL / CC-BY)."
)

st.markdown("#### Textes réglementaires et plans de référence")
reglementaire = [
    ("Loi d'Orientation des Mobilités (LOM)", "2019",
     "Décentralisation de la gouvernance des mobilités douces aux métropoles françaises.",
     "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000039666574"),
    ("Plan Vélo et Marche 2023–2027", "2023",
     "Programme national d'investissement cyclable (250 M€/an). Référence de politique publique.",
     "https://www.ecologie.gouv.fr/politiques-publiques/plan-velo-marche-2023-2027"),
    ("Règlement Européen sur les Données d'Accessibilité (MMTIS)", "2017",
     "Cadre réglementaire pour les données GTFS/NeTEx en Europe.",
     "https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32017R1926"),
]
for name, year, desc, url in reglementaire:
    st.markdown(f"- **{name}** ({year}) — {desc} [Lien]({url})")

# ── Section 3 : Stack technique ────────────────────────────────────────────────
st.divider()
section(3, "Stack Technique et Logiciels")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Plateforme et visualisation")
    stack_viz = [
        ("Streamlit", "≥ 1.32", "Interface web interactive, multipage"),
        ("Plotly", "≥ 5.18", "Graphiques interactifs (barres, scatter, heatmap, radar)"),
        ("pydeck", "≥ 0.8", "Visualisation WebGL des 46 312 stations (ScatterplotLayer)"),
        ("pandas", "≥ 2.0", "Manipulation du corpus Gold Standard (Parquet / CSV)"),
        ("pyarrow", "≥ 14.0", "Lecture / écriture Parquet Apache Arrow"),
    ]
    st.dataframe(
        pd.DataFrame(stack_viz, columns=["Bibliothèque", "Version min.", "Rôle"]),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.markdown("#### Calcul scientifique et modélisation")
    stack_sci = [
        ("NumPy", "≥ 1.26", "Calculs vectorisés, algèbre linéaire (OLS, IMD)"),
        ("SciPy", "≥ 1.11", "Tests statistiques (Spearman, Shapiro-Wilk, Mann-Whitney)"),
        ("scikit-learn", "≥ 1.3", "Modèle Ridge (IES), optimisation différentielle, Monte Carlo"),
        ("libpysal / esda", "optionnel", "Indice de Moran (autocorrélation spatiale)"),
        ("networkx", "optionnel", "Graphes de stations Montpellier (Louvain, PageRank)"),
    ]
    st.dataframe(
        pd.DataFrame(stack_sci, columns=["Bibliothèque", "Version min.", "Rôle"]),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "**Tableau 3.1.** Stack technique open-source. Toutes les bibliothèques sont disponibles "
    "sous licences libres (MIT, BSD-3, Apache-2.0). La reproductibilité complète est assurée "
    "par le fichier `requirements.txt` inclus dans le dépôt."
)

st.markdown("""
#### Formats de données
- **Parquet (Apache Arrow)** — format colonnaire optimisé, compression ZSTD, ~3–5× plus compact que CSV
- **CSV UTF-8** — format universel, séparateur virgule, compatible Excel / R / QGIS
- **GeoJSON** — export géospatial optionnel (coordonnées WGS-84 EPSG:4326)
- **GBFS v3.0** — format source (JSON temps réel, MobilityData)
""")

# ── Section 4 : Comment citer ce travail ───────────────────────────────────────
st.divider()
section(4, "Comment Citer ce Travail")

st.markdown("""
Si vous utilisez la plateforme, les indices IMD/IES ou le corpus Gold Standard dans vos travaux,
merci d'utiliser l'une des formes de citation ci-dessous.
""")

tab_apa, tab_bib, tab_data = st.tabs(["Format APA", "Format BibTeX", "Citation des données"])

with tab_apa:
    st.markdown("""
**Plateforme et indices IMD/IES :**

> CESI BikeShare-ICT. (2026). *Atlas de l'Indice de Mobilité Douce (IMD) :
> Évaluation quantitative de l'équité socio-spatiale des systèmes de vélos en libre-service en France.*
> Tableau de bord Streamlit. CESI École d'ingénieurs. https://github.com/cesi/bikeshare-data-explorer

**Corpus Gold Standard GBFS :**

> CESI BikeShare-ICT. (2026). *Gold Standard GBFS France — 46 312 stations certifiées,
> 122 systèmes, enrichissement multi-sources* [Ensemble de données].
> CESI École d'ingénieurs. https://doi.org/10.xxxx/gold-standard-gbfs-france-2026
""")

with tab_bib:
    st.code(
        """\
@software{cesi_atlas_imd_2026,
  author       = {{CESI BikeShare-ICT}},
  title        = {{Atlas de l'Indice de Mobilité Douce (IMD) :
                   Équité socio-spatiale des vélos en libre-service français}},
  year         = {2026},
  publisher    = {CESI École d'ingénieurs},
  url          = {https://github.com/cesi/bikeshare-data-explorer},
  note         = {Plateforme Streamlit — Gold Standard GBFS · 46 312 stations · 122 systèmes}
}

@dataset{cesi_gold_standard_gbfs_2026,
  author       = {{CESI BikeShare-ICT}},
  title        = {{Gold Standard GBFS France — 46 312 stations certifiées}},
  year         = {2026},
  publisher    = {CESI École d'ingénieurs},
  doi          = {10.xxxx/gold-standard-gbfs-france-2026},
  url          = {https://github.com/cesi/bikeshare-data-explorer},
  note         = {Fichier Parquet Apache Arrow, enrichissement multi-sources
                   (INSEE, GTFS, BAAC, SRTM, Filosofi)}
}""",
        language="bibtex",
    )

with tab_data:
    st.markdown("""
**Principes FAIR** (*Findable, Accessible, Interoperable, Reusable*) appliqués au corpus :

| Principe | Implémentation |
|---|---|
| **Findable** | DOI permanent attribué au dataset · Métadonnées Dublin Core complètes |
| **Accessible** | Téléchargement libre CSV / Parquet depuis le module Export |
| **Interoperable** | Format Parquet (Apache Arrow) + CSV UTF-8 · WGS-84 EPSG:4326 |
| **Reusable** | Licence CC BY 4.0 · Dictionnaire de variables complet · Pipeline reproducible |

Le corpus est disponible en téléchargement direct depuis la page [Export](pages/4_Export.py).
Un dictionnaire complet des 40+ variables est inclus dans cette même page.
""")

# ── Section 5 : Remerciements et affiliations ──────────────────────────────────
st.divider()
section(5, "Affiliations et Contexte de Recherche")

st.markdown("""
**Institution :** CESI École d'ingénieurs — Programme BikeShare-ICT 2025–2026

**Contexte :** Ce travail s'inscrit dans le cadre d'une recherche appliquée en géographie quantitative
et en sciences des données, portant sur l'évaluation de l'équité socio-spatiale des politiques
de mobilité douce en France. La plateforme constitue l'interface de diffusion des résultats ;
le pipeline de traitement (construction du Gold Standard, calibration IMD/IES) est entièrement
open-source.

**Données sensibles :** Aucune donnée personnelle n'est collectée ni stockée par cette plateforme.
Les données INSEE Filosofi sont utilisées à l'échelle agrégée (carreau 200 m minimum),
conformément aux conditions d'utilisation INSEE.

**Accès au code source :** Le code source complet est disponible sur le dépôt GitHub du projet.
Les scripts de construction du Gold Standard, de calcul de l'IMD et de l'IES sont documentés
et reproductibles.
""")
