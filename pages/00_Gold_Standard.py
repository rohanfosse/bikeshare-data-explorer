"""
00_Gold_Standard.py - Ingénierie des données et audit multi-sources.
Présentation de l'hybridation des bases de données et de la correction des flux GBFS.
"""
import sys
from pathlib import Path

import numpy as np
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
    "de ses données d'entrée (paradigme <i>Garbage In, Garbage Out</i>). Cette page documente "
    "le pipeline d'audit massif réalisé sur les flux GBFS français et la stratégie d'hybridation "
    f"multi-sources mise en œuvre pour constituer le <b>Gold Standard GBFS</b> : "
    f"<b>{len(df):,} stations certifiées</b> issues de <b>{n_ok} systèmes</b> "
    f"({n_dock_cities} agglomérations), dont {n_dock:,} stations dock-based (VLS) "
    f"et {n_ff:,} points free-floating. "
    "Résultat clé : Bordeaux passait du rang 2 au rang 14 national après correction de l'anomalie A3 "
    "- illustrant l'enjeu d'un audit rigoureux avant toute modélisation spatiale.",
    findings=[
        (f"{len(df):,}", "stations certifiées"),
        (str(n_ok), "systèmes Gold Standard"),
        (f"{n_dock:,}", "stations dock-based VLS"),
        (str(n_dock_cities), "agglomérations"),
        (f"{avg_compl:.0f} %", "complétude enrichissement"),
    ],
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
        # Status breakdown → donut
        status_counts = catalog["status"].value_counts().reset_index()
        status_counts.columns = ["Statut", "Systèmes"]
        status_labels = {
            "ok":           "Certifié (Gold Standard)",
            "too_small":    "Exclu (micro-réseau)",
            "autopartage":  "Exclu (autopartage A1)",
            "excluded":     "Exclu (autre anomalie)",
            "dom_tom":      "Hors périmètre (DOM-TOM)",
        }
        _status_colors = {
            "Certifié (Gold Standard)":    "#27ae60",
            "Exclu (micro-réseau)":        "#e67e22",
            "Exclu (autopartage A1)":      "#c0392b",
            "Exclu (autre anomalie)":      "#8e44ad",
            "Hors périmètre (DOM-TOM)":    "#95a5a6",
        }
        status_counts["Statut"] = status_counts["Statut"].map(
            lambda s: status_labels.get(s, s)
        )
        fig_donut = go.Figure(go.Pie(
            labels=status_counts["Statut"],
            values=status_counts["Systèmes"],
            hole=0.5,
            marker_colors=[
                _status_colors.get(s, "#bdc3c7") for s in status_counts["Statut"]
            ],
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value} systèmes (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            height=280,
            margin=dict(l=5, r=5, t=10, b=5),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{len(catalog)}</b><br>systèmes",
                x=0.5, y=0.5, font_size=13, showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption(
            f"**Figure 2.2.** Répartition des {len(catalog)} systèmes GBFS audités "
            "selon le verdict de l'audit. Vert = certifié Gold Standard ; "
            "les autres couleurs encodent les 5 classes d'anomalies A1–A5."
        )

    # Distribution de capacité par station_type (démontre le biais A3)
    if "station_type" in df.columns and "capacity" in df.columns:
        st.markdown("#### Distribution de Capacité par Type — Preuve Visuelle du Biais A3")
        _cap_df = df[df["capacity"].notna() & (df["capacity"] > 0)].copy()
        _cap_df["capacity_clipped"] = _cap_df["capacity"].clip(upper=150)
        _type_labels = {
            "docked_bike":   "Dock-based VLS",
            "free_floating": "Free-floating (A3)",
            "carsharing":    "Autopartage (A1)",
        }
        _cap_df["Type"] = _cap_df["station_type"].map(
            lambda t: _type_labels.get(t, str(t))
        )
        _type_colors = {
            "Dock-based VLS":       "#1A6FBF",
            "Free-floating (A3)":   "#c0392b",
            "Autopartage (A1)":     "#8e44ad",
        }
        fig_cap = px.violin(
            _cap_df,
            x="Type",
            y="capacity_clipped",
            color="Type",
            color_discrete_map=_type_colors,
            box=True,
            points=False,
            labels={
                "Type":             "Type de station",
                "capacity_clipped": "Capacité déclarée (bornes, plafonnée à 150)",
            },
            height=340,
        )
        fig_cap.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor="#eee"),
        )
        st.plotly_chart(fig_cap, use_container_width=True)

        # Statistiques synthétiques par type
        _cap_stats = (
            _cap_df.groupby("Type")["capacity"]
            .agg(["median", "mean", "max", "count"])
            .reset_index()
            .rename(columns={
                "Type":   "Type de station",
                "median": "Capacité médiane",
                "mean":   "Capacité moyenne",
                "max":    "Capacité max observée",
                "count":  "Stations",
            })
        )
        st.dataframe(
            _cap_stats.style.format({
                "Capacité médiane": "{:.0f}",
                "Capacité moyenne": "{:.1f}",
                "Capacité max observée": "{:.0f}",
                "Stations": "{:,}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "**Figure 2.3 & Tableau 2.2.** Distribution des capacités déclarées par type de station "
            "(plafonnée à 150 bornes pour la lisibilité). "
            "Le biais A3 se lit directement : les stations *free-floating* présentent une "
            "capacité médiane disproportionnée — artefact du calcul par **moyenne conditionnelle** "
            "(exclusion des stations à capacité nulle). "
            "Après correction, seules les stations *dock-based* conservent une capacité physiquement "
            "interprétable comme nombre de bornes de stationnement réelles."
        )

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

# Graphique multi-villes : stations brutes (GBFS) vs certifiées (Gold Standard)
st.markdown("#### Impact de l'Audit sur la Taille des Réseaux — Comparaison Multi-Villes")
st.markdown(
    "Le catalogue GBFS déclare pour chaque système un comptage brut de stations "
    "(`n_stations` GBFS), tandis que le Gold Standard Final ne retient que les "
    "stations **dock-based certifiées**. L'écart révèle l'ampleur de la contamination "
    "A3 dans chaque agglomération."
)

if not catalog.empty and {"city", "n_stations", "status"}.issubset(catalog.columns):
    # Stations brutes par ville (catalogue GBFS)
    _raw = (
        catalog.groupby("city")["n_stations"]
        .sum().reset_index()
        .rename(columns={"n_stations": "GBFS brut"})
    )
    # Stations certifiées (Gold Standard, dock-based)
    _cert = cities_dock[["city", "n_stations"]].rename(columns={"n_stations": "Gold Standard"})

    _compare = _raw.merge(_cert, on="city", how="inner")
    _compare = _compare[_compare["GBFS brut"] > 0].copy()
    _compare["Réduction (%)"] = (
        100 * (1 - _compare["Gold Standard"] / _compare["GBFS brut"])
    ).round(1)
    # Tri par réduction décroissante, top 15 villes avec le plus grand écart
    _compare = _compare.sort_values("Réduction (%)", ascending=False).head(15)

    _before_after = pd.melt(
        _compare,
        id_vars=["city", "Réduction (%)"],
        value_vars=["GBFS brut", "Gold Standard"],
        var_name="Phase",
        value_name="Stations",
    )

    fig_ba = px.bar(
        _before_after,
        x="Stations",
        y="city",
        color="Phase",
        barmode="group",
        orientation="h",
        color_discrete_map={"GBFS brut": "#e74c3c", "Gold Standard": "#1A6FBF"},
        text="Stations",
        labels={"city": "Agglomération", "Stations": "Stations", "Phase": ""},
        height=max(360, len(_compare) * 42),
        custom_data=["Réduction (%)"],
    )
    fig_ba.update_traces(
        texttemplate="%{x:,}",
        textposition="outside",
    )
    fig_ba.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=80, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
        xaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    st.plotly_chart(fig_ba, use_container_width=True)

    # Tableau récapitulatif
    _tbl_ba = _compare[["city", "GBFS brut", "Gold Standard", "Réduction (%)"]].rename(
        columns={"city": "Agglomération"}
    )
    st.dataframe(
        _tbl_ba.style.format({
            "GBFS brut": "{:,.0f}",
            "Gold Standard": "{:,.0f}",
            "Réduction (%)": "{:.1f} %",
        }).background_gradient(subset=["Réduction (%)"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**Figure 3.1 & Tableau 3.1.** Comparaison du nombre de stations GBFS brutes "
        "versus stations dock-based certifiées (Gold Standard) par agglomération. "
        "Les 15 agglomérations présentant le plus fort taux de réduction sont affichées. "
        "**Bordeaux** passe de plus de 3 000 stations brutes (Pony free-floating inclus) "
        "à 225 stations dock-based — soit une réduction de plus de 93 % — illustrant "
        "le risque systémique d'une ingestion directe des flux GBFS non audités."
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
    col_compl_bar, col_compl_kpi = st.columns([3, 1])

    # Color-coded completeness bar chart
    compl_sorted = compl.sort_values("Complétude (%)", ascending=True).copy()
    compl_sorted["couleur"] = compl_sorted["Complétude (%)"].apply(
        lambda v: "#27ae60" if v >= 80 else ("#e67e22" if v >= 50 else "#e74c3c")
    )

    with col_compl_bar:
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

    with col_compl_kpi:
        _n_green  = int((compl["Complétude (%)"] >= 80).sum())
        _n_orange = int(((compl["Complétude (%)"] >= 50) & (compl["Complétude (%)"] < 80)).sum())
        _n_red    = int((compl["Complétude (%)"] < 50).sum())
        st.metric("Métriques ≥ 80 %", f"{_n_green} / {len(compl)}", "Qualité satisfaisante")
        st.metric("Métriques 50–80 %", f"{_n_orange} / {len(compl)}", "Couverture partielle",
                  delta_color="off")
        st.metric("Métriques < 50 %", f"{_n_red} / {len(compl)}", "Contrainte source primaire",
                  delta_color="inverse")

    # ── Heatmap de complétude par ville ──────────────────────────────────────
    st.markdown("#### Heatmap de Complétude par Agglomération — Couverture Spatiale des Modules")
    st.markdown(
        "La heatmap ci-dessous croise les **agglomérations dock-based** (axe vertical, top 25) "
        "et les **dimensions d'enrichissement** (axe horizontal), révélant quelles villes "
        "bénéficient d'une couverture complète et lesquelles présentent des lacunes structurelles "
        "(zones blanches = donnée absente)."
    )

    from utils.data_loader import METRICS as _METRICS_DEF

    _hm_cities = cities_dock.head(25)["city"].tolist()
    _hm_metrics = list(_METRICS_DEF.keys())
    _hm_labels  = [_METRICS_DEF[k]["label"] for k in _hm_metrics]

    # Calcul du taux de complétude ville × métrique
    _hm_df_dock = df_dock[df_dock["city"].isin(_hm_cities)]
    _hm_matrix  = []
    for _city in _hm_cities:
        _city_df = _hm_df_dock[_hm_df_dock["city"] == _city]
        _row = []
        for _col in _hm_metrics:
            if _col in _city_df.columns and len(_city_df) > 0:
                _pct = 100 * _city_df[_col].notna().mean()
            else:
                _pct = 0.0
            _row.append(round(_pct, 1))
        _hm_matrix.append(_row)

    _hm_z    = np.array(_hm_matrix)
    _hm_text = [[f"{v:.0f} %" for v in row] for row in _hm_matrix]

    fig_hm = go.Figure(go.Heatmap(
        z=_hm_z,
        x=_hm_labels,
        y=_hm_cities,
        text=_hm_text,
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorscale=[
            [0.0,  "#e74c3c"],
            [0.5,  "#e67e22"],
            [0.8,  "#f9e79f"],
            [1.0,  "#27ae60"],
        ],
        zmin=0, zmax=100,
        showscale=True,
        colorbar=dict(
            title="Complétude (%)",
            thickness=12,
            tickvals=[0, 50, 80, 100],
            ticktext=["0 %", "50 %", "80 %", "100 %"],
        ),
        hovertemplate="Ville : %{y}<br>Métrique : %{x}<br>Complétude : %{z:.0f} %<extra></extra>",
    ))
    fig_hm.update_layout(
        height=max(400, len(_hm_cities) * 22),
        margin=dict(l=10, r=20, t=10, b=120),
        xaxis=dict(tickangle=-40, tickfont=dict(size=9), side="bottom"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(
        f"**Figure 4.2.** Heatmap de complétude des {len(_hm_metrics)} dimensions d'enrichissement "
        f"pour les {len(_hm_cities)} plus grandes agglomérations dock-based. "
        "Vert = couverture complète (100 %) ; Rouge = donnée absente (0 %). "
        "Les lacunes de couverture BAAC et SRTM sont visibles sur les petites agglomérations "
        "périphériques, confirmant la dépendance à la densité urbaine des sources primaires."
    )

    # Completeness table (dépliable)
    with st.expander("Tableau détaillé de complétude (toutes métriques)", expanded=False):
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
    "certifiées sont comptabilisées, permettant une comparaison homogène "
    "de la capacité physique des réseaux VLS."
)

tab_bar, tab_map, tab_scatter = st.tabs([
    "Classement par agglomération",
    "Carte nationale des stations",
    "Capacité × Infrastructure cyclable",
])

with tab_bar:
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
        "**Figure 5.1.** Top 20 agglomérations par nombre de stations VLS dock-based certifiées. "
        "Paris (Vélib') domine largement, suivi par Lyon et Toulouse. "
        "L'exclusion des opérateurs free-floating ramène Bordeaux "
        "de 9 920 stations brutes à 225 stations dock-based, illustrant concrètement "
        "l'impact de la correction A3 sur l'évaluation comparative des réseaux VLS."
    )

with tab_map:
    st.markdown(
        "Carte interactive de **toutes les stations certifiées**, colorées par type. "
        "Un échantillon aléatoire de 5 000 points est affiché pour la performance."
    )
    _map_df = df.dropna(subset=["lat", "lon"]).copy()
    # Sous-échantillonnage pour la performance
    if len(_map_df) > 5000:
        _map_df = _map_df.sample(5000, random_state=42)

    _type_labels_map = {
        "docked_bike":   "Dock-based VLS",
        "free_floating": "Free-floating (A3)",
        "carsharing":    "Autopartage (A1)",
    }
    _map_df["Type"] = _map_df["station_type"].map(
        lambda t: _type_labels_map.get(str(t), str(t))
    ) if "station_type" in _map_df.columns else "Dock-based VLS"

    _map_type_colors = {
        "Dock-based VLS":       "#1A6FBF",
        "Free-floating (A3)":   "#e74c3c",
        "Autopartage (A1)":     "#8e44ad",
    }
    fig_map_gs = px.scatter_mapbox(
        _map_df,
        lat="lat", lon="lon",
        color="Type",
        color_discrete_map=_map_type_colors,
        hover_name="city" if "city" in _map_df.columns else None,
        hover_data={
            "capacity": True,
            "lat": False,
            "lon": False,
            "Type": False,
            "source_label": True,
        },
        mapbox_style="carto-positron",
        zoom=4.8,
        center={"lat": 46.8, "lon": 2.3},
        size_max=8,
        opacity=0.65,
        height=500,
        labels={"Type": "Type de station", "capacity": "Capacité", "source_label": "Source"},
    )
    fig_map_gs.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_map_gs, use_container_width=True)
    st.caption(
        f"**Figure 5.2.** Carte nationale des stations du Gold Standard "
        f"(échantillon de {min(5000, len(df)):,} points sur {len(df):,}). "
        "Bleu = stations dock-based certifiées (VLS physique) ; "
        "Rouge = flottes free-floating (A3 corrigé) ; "
        "Violet = systèmes d'autopartage (A1, conservés pour analyse comparative). "
        "La densité bleue en Île-de-France traduit la présence de Vélib' ; "
        "les clusters rouges à Bordeaux/Nice illustrent la prévalence de l'anomalie A3."
    )

with tab_scatter:
    st.markdown(r"""
    La relation entre la **capacité moyenne des stations** (proxy de l'offre physique) et le
    **taux d'infrastructure cyclable** (proxy de la demande latente) détermine si les réseaux
    VLS sont déployés là où l'environnement les soutient — ou au contraire dans des zones
    peu propices à l'usage cyclable.
    """)
    _sc_df = cities_dock.dropna(subset=["infra_cyclable_pct", "capacity"]).copy()
    if not _sc_df.empty:
        # Taille encodée par nombre de stations
        _sc_rho = float(
            pd.Series(_sc_df["capacity"].values).rank()
            .corr(pd.Series(_sc_df["infra_cyclable_pct"].values).rank())
        )
        # Ligne OLS manuelle
        _sc_x  = _sc_df["infra_cyclable_pct"].values
        _sc_y  = _sc_df["capacity"].values
        _sc_c  = np.polyfit(_sc_x, _sc_y, 1)
        _sc_xr = np.linspace(_sc_x.min(), _sc_x.max(), 100)

        fig_sc = px.scatter(
            _sc_df,
            x="infra_cyclable_pct",
            y="capacity",
            size="n_stations",
            color="n_stations",
            color_continuous_scale="Blues",
            hover_name="city",
            size_max=30,
            labels={
                "infra_cyclable_pct": "Infrastructure cyclable moyenne (% dans buffer 300 m)",
                "capacity":           "Capacité moyenne par station (bornes)",
                "n_stations":         "Stations dock",
            },
            height=420,
            opacity=0.85,
        )
        # Annoter top 5 villes par n_stations
        _top5_cities = _sc_df.nlargest(5, "n_stations")
        for _, _r in _top5_cities.iterrows():
            fig_sc.add_annotation(
                x=_r["infra_cyclable_pct"], y=_r["capacity"],
                text=str(_r["city"]),
                showarrow=True, arrowhead=2, arrowwidth=1,
                arrowcolor="#333", ax=25, ay=-18,
                font=dict(size=9),
            )
        fig_sc.add_trace(go.Scatter(
            x=_sc_xr, y=np.polyval(_sc_c, _sc_xr),
            mode="lines",
            line=dict(color="#1A2332", dash="dash", width=2),
            name=f"Tendance OLS (ρ = {_sc_rho:+.2f})",
            showlegend=True,
        ))
        fig_sc.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
            coloraxis_showscale=False,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            f"**Figure 5.3.** Infrastructure cyclable moyenne (axe x) versus "
            f"capacité moyenne par station (axe y). Taille = nombre de stations. "
            f"ρ de Spearman = {_sc_rho:+.2f}. "
            "Les 5 agglomérations les plus grandes sont annotées. "
            "Une corrélation positive suggère que les villes dotées d'un environnement "
            "cyclable dense ont tendance à concevoir des stations à plus haute capacité, "
            "cohérent avec une planification intégrée offre/demande."
        )
    else:
        st.info("Données insuffisantes pour le scatter capacité × infrastructure.")

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
2. **Contribution empirique :** Un jeu de données enrichi selon six modules spatiaux
   (dont les données socio-économiques INSEE Filosofi - revenu médian, Gini, mobilité),
   prêt à supporter des modélisations complexes telles que la théorie des graphes,
   l'analyse temporelle des flux de micromobilité, ou la modélisation économétrique de
   l'équité spatiale - Indice d'Équité Sociale (IES, *cf.* page dédiée).
""")

# ── Synthèse visuelle du pipeline ─────────────────────────────────────────────
st.markdown("#### Synthèse du Pipeline d'Ingénierie — De la Source Brute au Gold Standard")

# Métriques dynamiques pour le pipeline
_n_sources       = 6   # modules d'enrichissement
_n_anomaly_types = 5   # classes A1–A5
_pct_dock        = round(100 * n_dock / max(len(df), 1), 1)
_avg_cap_dock    = round(float(df_dock["capacity"].median()), 1) if "capacity" in df_dock.columns else "n.d."

col_pipe1, col_pipe2, col_pipe3, col_pipe4 = st.columns(4)
col_pipe1.metric("Étapes du pipeline", "6", "Audit + Purge + Enrichissement")
col_pipe2.metric("Classes d'anomalies traitées", str(_n_anomaly_types), "A1 → A5")
col_pipe3.metric("Modules d'enrichissement spatial", str(_n_sources), "Topo · Sécu · Infra · TC · Socio · Val.")
col_pipe4.metric("Part dock-based finale", f"{_pct_dock:.1f} %",
                 f"Capacité médiane : {_avg_cap_dock} bornes")

# Diagramme de flux textuel du pipeline
st.markdown("""
```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PIPELINE D'INGÉNIERIE DU GOLD STANDARD GBFS                    │
├─────────────┬──────────────────────────────────────┬────────────────────────┤
│  ÉTAPE      │  OPÉRATION                           │  RÉSULTAT              │
├─────────────┼──────────────────────────────────────┼────────────────────────┤
│ 1. Collecte │ Agrégation GBFS (MobilityData + OSM) │ ~122 systèmes bruts    │
│ 2. Audit    │ Détection anomalies A1–A5            │ Classification typée   │
│ 3. Purge    │ Exclusion sémantique + géofiltre      │ Réduction ~-40 %       │
│ 4. Redress. │ Recalc. capacité réelle A3            │ Fin biais FF-anchor    │
│ 5. Spatial  │ Spatial Join 6 sources (300 m buffer) │ +11 métriques/station  │
│ 6. Certif.  │ Seuil robustesse ≥ 20 stations dock  │ Gold Standard Final    │
└─────────────┴──────────────────────────────────────┴────────────────────────┘
```
""")

# Distribution géographique finale (treemap par région)
if not catalog.empty and {"region", "n_stations", "status"}.issubset(catalog.columns):
    _treemap_df = (
        catalog[catalog["status"] == "ok"]
        .groupby(["region", "city"])
        .agg(n_stations=("n_stations", "sum"), n_systems=("status", "count"))
        .reset_index()
    )
    if not _treemap_df.empty:
        _treemap_df["label"] = _treemap_df.apply(
            lambda r: f"{r['city']}<br>{r['n_stations']:,} stations", axis=1
        )
        fig_tree = px.treemap(
            _treemap_df,
            path=["region", "city"],
            values="n_stations",
            color="n_stations",
            color_continuous_scale="Blues",
            hover_data={"n_stations": True, "n_systems": True},
            labels={"n_stations": "Stations", "n_systems": "Systèmes"},
            height=420,
        )
        fig_tree.update_traces(
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Stations : %{value:,}<extra></extra>",
        )
        fig_tree.update_layout(
            margin=dict(l=5, r=5, t=10, b=5),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        st.caption(
            "**Figure 6.1.** Treemap des stations Gold Standard certifiées par région et agglomération. "
            "La surface de chaque cellule est proportionnelle au nombre de stations certifiées. "
            "Paris (Vélib') et l'Île-de-France dominent la surface nationale, "
            "illustrant la forte concentration géographique de l'offre VLS française."
        )
