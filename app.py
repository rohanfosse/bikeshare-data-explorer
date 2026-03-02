"""
app.py - Point d'entrée de l'application Streamlit.
Atlas de l'Indice de Mobilité Douce (IMD) - Gold Standard GBFS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import city_stats, compute_imd_cities, load_stations, load_systems_catalog
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Atlas IMD - Justice Spatiale & Vélos en Libre-Service",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Chargement des données (avant abstract pour valeurs dynamiques) ─────────────
df      = load_stations()
catalog = load_systems_catalog()
cities  = city_stats(df)
imd_df  = compute_imd_cities(df)

n_certified = int((catalog["status"] == "ok").sum()) if "status" in catalog.columns else len(catalog)
n_cities    = df["city"].nunique()
n_dock      = int((df["station_type"] == "docked_bike").sum()) if "station_type" in df.columns else len(df)

# IMD rankings dynamiques
_imd_ranked = imd_df.sort_values("IMD", ascending=False).reset_index(drop=True)
_top_city   = _imd_ranked.iloc[0]["city"] if len(_imd_ranked) > 0 else "-"
_top_imd    = float(_imd_ranked.iloc[0]["IMD"]) if len(_imd_ranked) > 0 else 0.0
_n_imd      = len(_imd_ranked)
_imd_median = float(_imd_ranked["IMD"].median())

_mmm_row    = _imd_ranked[_imd_ranked["city"] == "Montpellier"]
_mmm_rank   = int(_mmm_row.index[0]) + 1 if not _mmm_row.empty else "?"
_mmm_imd    = float(_mmm_row["IMD"].iloc[0]) if not _mmm_row.empty else 0.0

# Calcul OLS R² et Spearman IMD ~ Revenu (données réelles)
_R2_real = float("nan")
_rho_real = float("nan")
_pval_real = float("nan")
_n_filosofi = 0

if "revenu_median_uc" in imd_df.columns:
    _tmp = imd_df.dropna(subset=["revenu_median_uc", "IMD"]).copy()
    _n_filosofi = len(_tmp)
    if _n_filosofi >= 5:
        _xr = _tmp["revenu_median_uc"].values.astype(float)
        _yr = _tmp["IMD"].values.astype(float)
        _c  = np.polyfit(_xr, _yr, 1)
        _SS_res = float(np.sum((_yr - np.polyval(_c, _xr)) ** 2))
        _SS_tot = float(np.sum((_yr - _yr.mean()) ** 2))
        _R2_real = 1.0 - _SS_res / _SS_tot
        _rho_real = float(pd.Series(_xr).rank().corr(pd.Series(_yr).rank()))
        # p-value approchée
        try:
            from scipy.stats import spearmanr as _sp
            _, _pval_real = _sp(_xr, _yr)
            _pval_real = float(_pval_real)
        except ImportError:
            _n = _n_filosofi
            _t = _rho_real * np.sqrt((_n - 2) / max(1e-10, 1 - _rho_real ** 2))
            _z = abs(_t)
            _phi = 0.5 * (1 + np.sign(_t) * (1 - np.exp(-0.717 * _z - 0.416 * _z ** 2)))
            _pval_real = float(max(0.0, min(1.0, 2 * (1 - _phi))))

_rho_str  = f"{_rho_real:+.3f}" if not np.isnan(_rho_real) else "n.d."
_pval_str = (f"{_pval_real:.3f}" if (not np.isnan(_pval_real) and _pval_real >= 0.001)
             else ("< 0,001" if not np.isnan(_pval_real) else "n.d."))
_R2_str   = f"{_R2_real:.4f}" if not np.isnan(_R2_real) else "n.d."

# ── En-tête ────────────────────────────────────────────────────────────────────
st.title("Atlas de l'Indice de Mobilité Douce (IMD)")
st.caption(
    "Évaluation quantitative de l'équité socio-spatiale des systèmes de vélos en libre-service "
    "en France - Gold Standard GBFS · R. Fossé & G. Pallares · 2025–2026"
)

abstract_box(
    "<b>Résumé :</b> Cette plateforme présente les résultats d'une recherche en géographie quantitative "
    "sur l'équité socio-spatiale des réseaux de vélos en libre-service (VLS) en France. "
    f"Le corpus de référence - le <b>Gold Standard GBFS</b> - regroupe <b>{len(df):,} stations certifiées</b> "
    f"issues de {n_certified} systèmes couvrant {n_cities} agglomérations, après audit et purge systématique "
    "des anomalies des flux GBFS ouverts.<br><br>"
    "Deux indices composites sont calibrés empiriquement sur ce corpus : "
    "l'<b>IMD (Indice de Mobilité Douce)</b> évalue la qualité de l'environnement cyclable "
    "selon quatre dimensions - sécurité, infrastructure, multimodalité et topographie - "
    f"et place <b>{_top_city} en tête</b> du classement national (IMD&nbsp;=&nbsp;{_top_imd:.1f}/100, "
    f"médiane&nbsp;=&nbsp;{_imd_median:.1f}/100, {_n_imd} agglomérations). "
    "L'<b>IES (Indice d'Équité Sociale)</b> mesure l'écart entre l'offre observée et l'offre "
    "attendue au regard du revenu local.<br><br>"
    "Deux résultats contre-intuitifs structurent la contribution : "
    "<b>(1)</b> l'indice de Moran appliqué aux scores IMD est <i>I</i>&nbsp;=&nbsp;&minus;0,023 "
    "(<i>p</i>&nbsp;=&nbsp;0,765, non significatif) - le déterminisme géographique est réfuté ; "
    f"<b>(2)</b> la corrélation de rang entre IMD et revenu médian est "
    f"<i>&rho;</i>&nbsp;=&nbsp;{_rho_str} (<i>p</i>&nbsp;=&nbsp;{_pval_str}, "
    f"<i>R</i>²&nbsp;=&nbsp;{_R2_str}) - le déterminisme économique est également réfuté. "
    "La qualité d'un réseau VLS relève avant tout de choix de gouvernance locale."
)

sidebar_nav()

# ── KPIs calculés depuis les données réelles ───────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Stations Gold Standard",       f"{len(df):,}")
k2.metric("Dont dock-based VLS",          f"{n_dock:,}")
k3.metric("Systèmes GBFS certifiés",      f"{n_certified}")
k4.metric(f"IMD #1 - {_top_city}",        f"{_top_imd:.1f} / 100")
k5.metric("Moran's I (spatial)",          "−0,023", "p = 0,765 - non sign.")
k6.metric("ρ Spearman IMD × Revenu",      _rho_str, f"p = {_pval_str} - non sign.")

# ── Pipeline de recherche (vue synthétique) ─────────────────────────────────────
st.markdown(
    f"""
    <div style="display:flex; gap:0; align-items:stretch; margin:1.4rem 0 0.3rem 0;">
      <div style="flex:1; background:#e8edf5; border:1px solid #c0cfe0; border-radius:6px;
                  padding:0.75rem 0.5rem; text-align:center;">
        <div style="font-weight:700; font-size:0.82rem; color:#1A2332;">GBFS Bruts</div>
        <div style="font-size:0.72rem; color:#4a6a88; margin-top:0.28rem;">
          125 systèmes<br>~60&nbsp;000 points bruts</div>
      </div>
      <div style="display:flex; align-items:center; padding:0 0.45rem;
                  color:#5a8abf; font-size:1.4rem; flex-shrink:0;">&#8594;</div>
      <div style="flex:1; background:#fef3cd; border:1px solid #e8d460; border-radius:6px;
                  padding:0.75rem 0.5rem; text-align:center;">
        <div style="font-weight:700; font-size:0.82rem; color:#856404;">Audit A1–A5</div>
        <div style="font-size:0.72rem; color:#856404; margin-top:0.28rem;">
          Taxonomie<br>5&nbsp;classes d'anomalies</div>
      </div>
      <div style="display:flex; align-items:center; padding:0 0.45rem;
                  color:#5a8abf; font-size:1.4rem; flex-shrink:0;">&#8594;</div>
      <div style="flex:1.2; background:#1A6FBF; border:1px solid #1A6FBF; border-radius:6px;
                  padding:0.75rem 0.5rem; text-align:center;">
        <div style="font-weight:700; font-size:0.82rem; color:white;">Gold Standard</div>
        <div style="font-size:0.72rem; color:#b8d8f4; margin-top:0.28rem;">
          <b style="color:white">{len(df):,}</b>&nbsp;stations<br>{n_certified}&nbsp;systèmes certifiés</div>
      </div>
      <div style="display:flex; align-items:center; padding:0 0.45rem;
                  color:#5a8abf; font-size:1.4rem; flex-shrink:0;">&#8594;</div>
      <div style="flex:1; background:#155a9c; border:1px solid #155a9c; border-radius:6px;
                  padding:0.75rem 0.5rem; text-align:center;">
        <div style="font-weight:700; font-size:0.82rem; color:white;">IMD &middot; IES</div>
        <div style="font-size:0.72rem; color:#a8c8e8; margin-top:0.28rem;">
          4&nbsp;composantes<br>{_n_imd}&nbsp;agglomérations</div>
      </div>
      <div style="display:flex; align-items:center; padding:0 0.45rem;
                  color:#5a8abf; font-size:1.4rem; flex-shrink:0;">&#8594;</div>
      <div style="flex:1; background:#0d3d6b; border:1px solid #0d3d6b; border-radius:6px;
                  padding:0.75rem 0.5rem; text-align:center;">
        <div style="font-weight:700; font-size:0.82rem; color:white;">Justice Spatiale</div>
        <div style="font-size:0.72rem; color:#90b8d8; margin-top:0.28rem;">
          Gouvernance<br>&#62;&nbsp;Géographie</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "**Figure 0.1.** Pipeline de recherche en quatre étapes : "
    "de la donnée GBFS brute à l'atlas de justice spatiale (IMD, IES)."
)

# ── Section 1 : Contexte et Problématique ──────────────────────────────────────
st.divider()
section(1, "Contexte Politique et Problématique Scientifique")

st.markdown(rf"""
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
rarement audités. Ce postulat - qui associe implicitement l'abondance de l'offre à son utilité
sociale - masque des biais structurels majeurs : un réseau dense peut s'avérer inopérant s'il est
déconnecté des pôles d'échanges multimodaux, ou inéquitable s'il exclut systématiquement les
quartiers à forte vulnérabilité économique (*Médard de Chardon et al., 2017*).

Afin de pallier ces lacunes méthodologiques, cet article propose une approche quantitative inédite,
structurée autour de :

1. La constitution d'un jeu de données de référence - le **Gold Standard** - expurgé des anomalies
   inhérentes à l'Open Data GBFS.
2. La calibration empirique d'un **Indice de Mobilité Douce (IMD)** intégrant la friction spatiale,
   l'accidentologie, la continuité des infrastructures et l'hybridation multimodale.
3. L'introduction d'un **Indice d'Équité Sociale (IES)** permettant de cartographier formellement
   les "Déserts de Mobilité Sociale" et d'orienter vers une gouvernance cyclable plus juste.
4. Une **analyse topographique** (Terrain Ruggedness Index, SRTM 30 m) et des distances
   inter-stations vol d'oiseau (haversine) pour décomposer la contrainte géographique.
""")

# ── Section 2 : L'Urgence de l'Audit GBFS ─────────────────────────────────────
st.divider()
section(2, "L'Urgence de l'Audit des Données Ouvertes (GBFS) : Du Bruit au Signal")

st.markdown(rf"""
La littérature académique s'appuie de manière croissante sur des flux de données ouverts au standard
GBFS (*General Bikeshare Feed Specification*, v3.0). Toutefois, nos travaux démontrent que
l'utilisation naïve de ces données brutes est **scientifiquement erronée**. L'audit systématique
des {len(catalog)} systèmes français a révélé une taxonomie de 5 classes d'anomalies structurelles (A1 à A5).

L'anomalie A3 - le *biais de la moyenne conditionnelle* inhérent aux flottes *free-floating* - est
la plus pernicieuse : elle engendre des surestimations massives de capacité
($\bar{{c}}_{{\text{{profil}}}} \gg \bar{{c}}_{{\text{{réel}}}}$), invalidant les classements de performance
de plusieurs métropoles. À titre d'illustration, Bordeaux passait du rang 2 au rang 14 national
après correction de ce seul biais, ce qui aurait pu conduire à des réallocations de subventions
publiques erronées de plusieurs millions d'euros.

En purgeant rigoureusement les données de ces biais algorithmiques, nous avons constitué un
**Gold Standard** de {len(df):,} stations validées sur {n_cities} agglomérations - socle indispensable à toute
modélisation spatiale robuste. Ce jeu de données est mis à disposition de la communauté
scientifique via l'interface d'export de cette plateforme (formats CSV et Parquet, principes FAIR).
""")

col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**Bilan de l'Audit GBFS**")
    audit_rows = [
        {"Étape": "Systèmes GBFS bruts disponibles",    "Valeur": "125"},
        {"Étape": "Exclusion A1 (Autopartage Citiz)",    "Valeur": "−14"},
        {"Étape": "Exclusion A4 & A5 (Géo / Périmètre)","Valeur": "−7"},
        {"Étape": "Micro-réseaux exclus (< 20 stations)","Valeur": "−20 (approx.)"},
        {"Étape": "Systèmes Gold Standard certifiés",    "Valeur": str(n_certified)},
        {"Étape": "Stations Gold Standard certifiées",   "Valeur": f"{len(df):,}"},
        {"Étape": "Agglomérations couvertes",            "Valeur": str(df["city"].nunique())},
        {"Étape": "Dont stations dock-based VLS",        "Valeur": f"{n_dock:,}"},
    ]
    st.table(pd.DataFrame(audit_rows))

with col_r:
    st.markdown("**Anomalies GBFS - Taxonomie A1–A5**")
    anomaly_rows = [
        {"Classe": "A1", "Nature": "Inclusion hors-domaine (autopartage)",  "Impact": "Biais de classification"},
        {"Classe": "A2", "Nature": "Capacité fictive (placeholder)",          "Impact": "Surestimation capacitaire"},
        {"Classe": "A3", "Nature": "Biais floating-anchor (moyenne cond.)",  "Impact": "Surestimation massive IMD"},
        {"Classe": "A4", "Nature": "Aberrations géospatiales (Lat/Lon)",     "Impact": "Biais topologique"},
        {"Classe": "A5", "Nature": "Hors périmètre (DOM-TOM / macro-rég.)", "Impact": "Artefacts de distribution"},
    ]
    st.table(pd.DataFrame(anomaly_rows))

# ── Section 3 : Architecture analytique ───────────────────────────────────────
st.divider()
section(3, "Architecture Analytique - Six Axes de Recherche Complémentaires")

st.markdown(r"""
La recherche est structurée en six axes analytiques complémentaires, progressant de l'ingénierie
des données vers la modélisation spatiale, puis vers l'évaluation de la justice sociale et
de la contrainte topographique.
""")

axes = [
    {
        "Axe":          "Axe Prél.",
        "Page":         "Gold Standard",
        "Question":     "L'Open Data GBFS est-il un matériau de recherche fiable ?",
        "Méthode":      "Audit multi-systèmes, taxonomie A1–A5, pipeline de purge en 6 étapes",
        "Résultat clé": f"{len(df):,} stations certifiées - Bordeaux : rang 2 → 14 après correction",
    },
    {
        "Axe":          "Axe 1",
        "Page":         "IMD",
        "Question":     "La qualité cyclable se réduit-elle au volume de stations ?",
        "Méthode":      "Indice composite 4D (S, I, M, T), optimisation supervisée, Monte Carlo N = 10 000",
        "Résultat clé": f"Top 10 stable dans 89 % des simulations - w_M* = 0,578 - #1 : {_top_city} (IMD = {_top_imd:.1f}/100)",
    },
    {
        "Axe":          "Axe 2",
        "Page":         "IES",
        "Question":     "L'offre cyclable est-elle équitablement distribuée socialement ?",
        "Méthode":      "Modèle Ridge (lambda par CV), IES = IMD_obs / IMD_prédit(R_m)",
        "Résultat clé": f"ρ = {_rho_str} (p = {_pval_str}, n.s.) - R² = {_R2_str} - Gouvernance > Économie",
    },
    {
        "Axe":          "Axe 3",
        "Page":         "Villes",
        "Question":     "Les disparités inter-urbaines sont-elles géographiques ou politiques ?",
        "Méthode":      "Indice global de Moran (autocorrélation spatiale), analyse comparative",
        "Résultat clé": "Moran's I = −0,023 (p = 0,765) - déterminisme géographique invalidé",
    },
    {
        "Axe":          "Axe 4",
        "Page":         "Distributions",
        "Question":     "La taille d'une agglomération prédit-elle sa performance cyclable ?",
        "Méthode":      "Corrélation de Spearman, boîtes à encoches, matrice de corrélation",
        "Résultat clé": "r_s = −0,02 (hors Paris) - aucune corrélation taille–performance",
    },
    {
        "Axe":          "Axe 5",
        "Page":         "Topographie",
        "Question":     "Le relief contraint-il structurellement la qualité des réseaux VLS ?",
        "Méthode":      "TRI (Riley 1999, SRTM 30 m), haversine inter-stations, OLS TRI ~ composante T",
        "Résultat clé": "Laon (TRI = 11,1) > Montpellier (TRI = 3,9) > Tarbes (TRI = 0,35) - T validé",
    },
    {
        "Axe":          "Axe 6",
        "Page":         "Montpellier",
        "Question":     "Les modèles nationaux se valident-ils à l'échelle micro-locale ?",
        "Méthode":      "Théorie des graphes (Louvain, PageRank), GTFS, IES intra-urbain par quartier",
        "Résultat clé": "Structure bimodale Commuter confirmée - fracture socio-spatiale cartographiée",
    },
]
st.table(pd.DataFrame(axes))
st.caption(
    "**Tableau 3.1.** Architecture des six axes de recherche. "
    "Chaque axe correspond à une page dédiée dans la barre de navigation latérale. "
    "Les pages *Carte*, *France* et *Export* constituent des modules transversaux "
    "(visualisation spatiale, validation multi-sources et diffusion FAIR des données)."
)

# Donut IMD + texte explicatif
_col_donut, _col_expl = st.columns([1, 2])
with _col_donut:
    _fig_donut = go.Figure(go.Pie(
        labels=["Multimodalité (M)", "Infrastructure (I)", "Sécurité (S)", "Topographie (T)"],
        values=[57.8, 18.4, 14.2, 9.6],
        hole=0.55,
        marker=dict(colors=["#1A6FBF", "#2196F3", "#64B5F6", "#BBDEFB"],
                    line=dict(color="white", width=1.5)),
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>w* = %{value} %<extra></extra>",
    ))
    _fig_donut.add_annotation(
        text="IMD<br><b>4 axes</b>", x=0.5, y=0.5, showarrow=False,
        font=dict(size=12, color="#1A2332"),
    )
    _fig_donut.update_layout(
        height=280, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(size=9), orientation="v", x=1.02, y=0.5),
    )
    st.plotly_chart(_fig_donut, use_container_width=True, config={"displayModeBar": False})
    st.caption("**Figure 3.1.** Poids optimaux des quatre composantes IMD (calibration par évolution différentielle, Monte Carlo N = 10 000).")

with _col_expl:
    st.markdown(rf"""
La **composante M - Multimodalité** (w* = 57,8 %) domine le modèle avec une marge nette.
Ce résultat, obtenu par optimisation supervisée, valide empiriquement que la proximité
aux arrêts de transports en commun (GTFS, rayon 300 m) est le déterminant le plus fort
de la qualité cyclable - loin devant le simple linéaire d'infrastructure.

La **composante I - Infrastructure** (18,4 %) reflète la continuité des aménagements
cyclables (pistes, bandes, voies vertes) dans un rayon normalisé.
La **composante S - Sécurité** (14,2 %) est calibrée à partir des données BAAC 2020–2023
(densité d'accidents cyclables géolocalisés). Enfin, la **composante T - Topographie**
(9,6 %) est la seule contrainte purement géographique du modèle : elle est validée par
la corrélation du TRI SRTM 30 m avec les scores observés.

L'absence de forte colinéarité entre composantes (matrice Spearman disponible en page
*Distributions*) valide la construction de l'indice comme somme pondérée non redondante.
""")

# ── Section 4 : Résultats clés ────────────────────────────────────────────────
st.divider()
section(4, "Résultats Clés - Invalidation de Deux Hypothèses Intuitives Majeures")

st.markdown(rf"""
Deux résultats contre-intuitifs structurent l'ensemble de la contribution :

#### 4.1. L'Absence de Déterminisme Géographique (Moran's $I = -0{{,}}023$, $p = 0{{,}}765$)

L'hypothèse implicitement dominante en géographie urbaine postule qu'une agglomération
"bien située" - c'est-à-dire bénéficiant d'une forte densité et d'une tradition de mobilité
douce - est condamnée à la performance cyclable, tandis que les villes périphériques seraient
structurellement pénalisées. L'indice de Moran appliqué aux scores IMD des {_n_imd} agglomérations
**réfute formellement cette hypothèse** : les villes performantes et sous-performantes ne forment
pas de clusters territoriaux cohérents. **{_top_city}** (#1, IMD = {_top_imd:.1f}/100) et Montpellier
(#{ _mmm_rank}, IMD = {_mmm_imd:.1f}/100), géographiquement éloignés, atteignent des scores comparables,
quand des villes voisines présentent des écarts considérables. **Les choix de gouvernance locale
priment sur le déterminisme géographique.**

#### 4.2. L'Absence de Déterminisme Économique ($\rho = {_rho_str}$, $p = {_pval_str}$, $R^2 = {_R2_str}$)

L'hypothèse symétrique suggère que les agglomérations les plus riches auraient
"les moyens de leurs ambitions cyclables". Le test empirique sur le panel Gold Standard
dock-based ({_n_filosofi} agglomérations françaises, données INSEE Filosofi) **réfute cette vision
déterministe** avec une force inattendue : la corrélation de Spearman entre IMD et revenu
médian/UC est $\rho = {_rho_str}$ ($p = {_pval_str}$, **statistiquement nulle**),
et le revenu médian n'explique que **$R^2 = {_R2_str}$** de la variance de l'IMD - soit moins
de 1 %. Des agglomérations à revenu modeste atteignent d'excellents scores IMD grâce à des
politiques tarifaires inclusives et une planification stratégique ; à l'inverse, des agglomérations
aisées sous-investissent dans la continuité des infrastructures. **La gouvernance locale prime
sur le capital économique.**

*Note comparative :* La littérature internationale (*Médard de Chardon et al., 2017*) rapporte
un $R^2_\text{{Ridge}} \approx 0{{,}}28$ sur des panels plus larges et des contextes anglo-saxons.
L'écart avec notre $R^2 = {_R2_str}$ s'explique par la spécificité française : la décentralisation
de la LOM (2019) confère aux métropoles une autonomie quasi-totale dans le déploiement des SVLS,
découplant davantage la qualité cyclable du niveau de revenu communal.
""")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.metric("Indice de Moran (IMD spatial)", "I = −0,023", "p = 0,765 - non significatif", delta_color="off")
    st.caption(
        "Absence d'autocorrélation spatiale globale (permutation $n = 999$). "
        "Les disparités de qualité VLS ne sont pas géographiquement déterminées."
    )
with col_r2:
    st.metric(f"R² OLS (revenu médian → IMD, n = {_n_filosofi})", _R2_str,
              f"ρ = {_rho_str}, p = {_pval_str} - non significatif", delta_color="off")
    st.caption(
        "Absence de déterminisme économique (données réelles Gold Standard + INSEE Filosofi). "
        "La qualité cyclable est un choix de gouvernance locale, non une fatalité économique."
    )

# Graphiques résultats : classement + scatter
_col_rank, _col_scat = st.columns(2)

with _col_rank:
    _top10 = _imd_ranked.head(10).copy()
    _fig_rank = go.Figure(go.Bar(
        x=_top10["IMD"],
        y=_top10["city"],
        orientation="h",
        marker_color=["#e74c3c" if c == "Montpellier" else "#1A6FBF" for c in _top10["city"]],
        text=_top10["IMD"].round(1).astype(str),
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>IMD = %{x:.1f} / 100<extra></extra>",
    ))
    _fig_rank.update_layout(
        title=dict(text=f"Classement national IMD - Top 10 sur {_n_imd} agglomérations",
                   font=dict(size=11), x=0),
        height=370, margin=dict(l=10, r=55, t=38, b=30),
        xaxis=dict(range=[0, 110], title="IMD / 100", gridcolor="#e8edf3", tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafd",
    )
    st.plotly_chart(_fig_rank, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 4.1.** Classement IMD national (top 10). "
        f"Montpellier (rang #{_mmm_rank}, rouge) - médiane nationale : {_imd_median:.1f}/100."
    )

with _col_scat:
    if _n_filosofi >= 5:
        _xfit = np.linspace(
            float(_tmp["revenu_median_uc"].min()),
            float(_tmp["revenu_median_uc"].max()),
            100,
        )
        _yfit = np.polyval(_c, _xfit)
        _highlight = {"Montpellier", _top_city, "Paris", "Lyon", "Marseille", "Bordeaux", "Rennes"}
        _lab_mask   = _tmp["city"].isin(_highlight)
        _fig_scat = go.Figure()
        _fig_scat.add_trace(go.Scatter(
            x=_tmp["revenu_median_uc"], y=_tmp["IMD"],
            mode="markers",
            marker=dict(
                color=["#e74c3c" if c == "Montpellier" else "#1A6FBF" for c in _tmp["city"]],
                size=7, opacity=0.68,
            ),
            text=_tmp["city"],
            hovertemplate="<b>%{text}</b><br>%{x:,.0f} €/UC &nbsp;·&nbsp; IMD %{y:.1f}<extra></extra>",
            showlegend=False,
        ))
        _fig_scat.add_trace(go.Scatter(
            x=_tmp.loc[_lab_mask, "revenu_median_uc"],
            y=_tmp.loc[_lab_mask, "IMD"],
            mode="markers+text",
            marker=dict(
                color=["#e74c3c" if c == "Montpellier" else "#1565C0"
                       for c in _tmp.loc[_lab_mask, "city"]],
                size=9, opacity=1,
            ),
            text=_tmp.loc[_lab_mask, "city"],
            textposition="top center",
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate="<b>%{text}</b><br>%{x:,.0f} €/UC &nbsp;·&nbsp; IMD %{y:.1f}<extra></extra>",
        ))
        _fig_scat.add_trace(go.Scatter(
            x=_xfit, y=_yfit,
            mode="lines",
            line=dict(color="#e74c3c", dash="dash", width=1.5),
            name=f"OLS (R² = {_R2_str})",
        ))
        _fig_scat.update_layout(
            title=dict(
                text=f"IMD × Revenu médian/UC - ρ = {_rho_str}, p = {_pval_str}",
                font=dict(size=11), x=0,
            ),
            height=370, margin=dict(l=10, r=15, t=38, b=45),
            xaxis=dict(title="Revenu médian/UC (€, INSEE Filosofi 2019)",
                       gridcolor="#e8edf3", tickformat=",.0f", tickfont=dict(size=9)),
            yaxis=dict(title="IMD / 100", gridcolor="#e8edf3", tickfont=dict(size=10)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafd",
            legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.85)",
                        x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )
        st.plotly_chart(_fig_scat, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"**Figure 4.2.** Corrélation IMD × revenu médian/UC "
            f"({_n_filosofi} agglomérations dock-based, INSEE Filosofi 2019). "
            f"La droite OLS (tirets) est quasi-horizontale - "
            f"R² = {_R2_str} (< 1 % de variance expliquée)."
        )

# ── Section 5 : Guide de Navigation ───────────────────────────────────────────
st.divider()
section(5, "Guide de Navigation - Parcours de Recherche et Modules Analytiques")

st.markdown(r"""
La plateforme est organisée en modules thématiques accessibles depuis la barre de navigation
latérale. Chaque module correspond à un axe de recherche ou à un outil transversal.
""")

nav_rows = [
    {"Module": "Gold Standard",  "Axe": "Prél.", "Contenu principal": f"Taxonomie des anomalies GBFS (A1–A5), pipeline de purge en 6 étapes, complétude de l'enrichissement, catalogue des {n_certified} systèmes certifiés."},
    {"Module": "IMD",            "Axe": "1",     "Contenu principal": f"Formulation mathématique, poids optimaux (w_M* = 0,578), Monte Carlo N = 10 000, classement national ({_top_city} #1), décomposition (S, I, M, T), validation FUB/EMP."},
    {"Module": "IES",            "Axe": "2",     "Contenu principal": f"Formalisation de l'IES, modèle Ridge, matrice de justice cyclable (4 quadrants), OLS formel (R² = {_R2_str}), bootstrap CI (N = 2 000), Mann-Whitney U."},
    {"Module": "Carte",          "Axe": "Trans.","Contenu principal": f"Visualisation WebGL des {len(df):,} stations (pydeck), filtrage par dimension d'enrichissement, distribution empirique, classement par agglomération."},
    {"Module": "Villes",         "Axe": "3",     "Contenu principal": "Classement univarié, nuage infra × sinistralité (Moran's I), profil radar multi-dimensionnel comparatif."},
    {"Module": "Distributions",  "Axe": "4",     "Contenu principal": "Histogrammes, boîtes à moustaches à encoches, matrice de corrélation Spearman, statistiques de forme (γ₁, γ₂, Shapiro-Wilk)."},
    {"Module": "Topographie",    "Axe": "5",     "Contenu principal": "Terrain Ruggedness Index (Riley 1999, SRTM 30 m), classement national TRI, distances vol d'oiseau (haversine), validation IMD composante T."},
    {"Module": "France",         "Axe": "Trans.","Contenu principal": "Triangulation FUB 2023, EMP 2019, éco-compteurs, BAAC et Cerema - indicateurs nationaux de la mobilité cyclable."},
    {"Module": "Montpellier",    "Axe": "6",     "Contenu principal": "Topologie Louvain, déséquilibres source/puits, vulnérabilité structurelle V_i, intégration GTFS tramway, fracture socio-spatiale IES intra-urbain."},
    {"Module": "Export",         "Axe": "FAIR",  "Contenu principal": "Accès libre au Gold Standard (CSV UTF-8 / Parquet), filtres multi-critères, dictionnaire de variables, métadonnées de citation."},
]
st.table(pd.DataFrame(nav_rows))
st.caption(
    "**Tableau 5.1.** Guide de navigation par module analytique. "
    "Les modules 'Trans.' sont transversaux et ne sont pas rattachés à un axe de recherche unique. "
    "Le module 'FAIR' implémente les principes *Findable, Accessible, Interoperable, Reusable* "
    "pour la diffusion académique du corpus Gold Standard."
)

# ── Section 6 : Données et Reproductibilité ───────────────────────────────────
st.divider()
section(6, "Données, Sources et Reproductibilité")

st.markdown(rf"""
**Corpus principal.** Le Gold Standard GBFS comprend {len(df):,} stations issues de {n_certified} systèmes
certifiés, enrichies selon cinq modules spatiaux dans un rayon normalisé de 300 m à partir de :
INSEE RP 2019 (carreaux 200 m), GTFS national (SNCF/RATP/Métropoles), IGN BD TOPO (POI),
SRTM 30 m (NASA/USGS - topographie), INSEE Filosofi 2019 (socio-économique).

**Reproductibilité.** L'ensemble du pipeline de construction du Gold Standard, du calcul de l'IMD
et de l'IES est implémenté en Python open-source (pandas, numpy, scipy, scikit-learn). Le fichier
`stations_gold_standard_final.parquet` est téléchargeable depuis le module Export au format Parquet
(Apache Arrow) ou CSV UTF-8, accompagné d'un dictionnaire de variables complet.

**Limites et biais résiduels.** *(1)* Le TRI SRTM 30 m présente une précision altimétrique de
±16 m en zone urbaine plane. *(2)* Les données Filosofi sont interpolées à l'échelle de carreaux
200 m - la résolution peut introduire un biais d'agrégation (MAUP). *(3)* L'absence de données
GBFS certifiées pour certaines agglomérations crée un biais de sélection : les agglomérations
analysées sont vraisemblablement plus matures dans leur politique VLS que la moyenne nationale.
*(4)* Le panel dock-based ({_n_imd} agglomérations avec IMD) exclut les systèmes *free-floating*
purs, dont les dynamiques spatiales diffèrent structurellement (*Bai & Jiao, 2020*).

| Source | Millésime | Granularité | Variables utilisées |
|---|---|---|---|
| GBFS (MobilityData) | 2024–2025 | Station | Capacité, lat/lon, type |
| INSEE Filosofi | 2019 | Carreau 200 m | Revenu médian, Gini, équipements |
| INSEE RP | 2020 | Carreau 200 m | Part ménages sans voiture, part vélo |
| GTFS national | 2024 | Arrêt | Accessibilité multimodale 300 m |
| BAAC (ONISR) | 2020–2023 | Accident | Sinistralité cyclable (S) |
| SRTM | 2000 (30 m) | Pixel | Altitude, TRI (T) |
| FUB Baromètre | 2023 | Agglomération | Climat perçu (validation externe) |
| EMP | 2019 | Agglomération | Part modale vélo (validation externe) |
""")
