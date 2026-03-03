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

col_l, col_r = st.columns([3, 2])
with col_l:
    st.markdown("**Pipeline de purge GBFS — Entonnoir de certification**")
    _funnel_stages = [
        ("GBFS bruts disponibles",         125,        "#95a5a6"),
        ("Après exclusion A1 (autopartage)", 111,       "#e67e22"),
        ("Après exclusion A4/A5 (géo/DOM)", 104,        "#f1c40f"),
        ("Après exclusion micro-réseaux",   84,         "#3498db"),
        (f"Gold Standard certifiés",        n_certified,"#27ae60"),
    ]
    fig_funnel = go.Figure(go.Funnel(
        y=[s[0] for s in _funnel_stages],
        x=[s[1] for s in _funnel_stages],
        marker_color=[s[2] for s in _funnel_stages],
        textinfo="value+percent initial",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x} systèmes (%{percentInitial:.0%})<extra></extra>",
        connector=dict(line=dict(color="#ccc", width=1)),
    ))
    fig_funnel.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_funnel, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 2.1.** Entonnoir de certification GBFS : de 125 systèmes bruts à "
        f"{n_certified} systèmes Gold Standard ({len(df):,} stations, {n_dock:,} dock-based)."
    )

with col_r:
    st.markdown("**Taxonomie des anomalies A1–A5**")
    _anomaly_df = pd.DataFrame([
        {"Cl.": "A1", "Nature": "Autopartage inclus",         "Gravité": "Haute",    "Nb": 14},
        {"Cl.": "A2", "Nature": "Capacité fictive (placeholder)", "Gravité": "Modérée", "Nb": "n.d."},
        {"Cl.": "A3", "Nature": "Biais floating-anchor",      "Gravité": "Critique", "Nb": "n.d."},
        {"Cl.": "A4", "Nature": "Aberrations géospatiales",   "Gravité": "Haute",    "Nb": 5},
        {"Cl.": "A5", "Nature": "Hors périmètre (DOM/rég.)",  "Gravité": "Modérée",  "Nb": 2},
    ])
    st.dataframe(
        _anomaly_df.style.apply(
            lambda col: [
                "background-color:#fde8e8;color:#c0392b" if v == "Critique"
                else "background-color:#fef3cd;color:#856404" if v == "Haute"
                else "" for v in col
            ], subset=["Gravité"]
        ),
        use_container_width=True, hide_index=True,
    )
    st.markdown(
        "<small>**A3** (biais *floating-anchor*) : l'anomalie la plus pernicieuse — "
        "Bordeaux passe du **rang 2 au rang 14** après correction.</small>",
        unsafe_allow_html=True,
    )

# ── Section 3 : Architecture analytique ───────────────────────────────────────
st.divider()
section(3, "Architecture Analytique - Six Axes de Recherche Complémentaires")

st.markdown(r"""
La recherche est structurée en six axes analytiques complémentaires, progressant de l'ingénierie
des données vers la modélisation spatiale, puis vers l'évaluation de la justice sociale et
de la contrainte topographique.
""")

_axes_cards = [
    {
        "badge": "Axe Prél.", "page": "Gold Standard", "color": "#e67e22",
        "question": "L'Open Data GBFS est-il un matériau de recherche fiable ?",
        "methode":  "Audit multi-systèmes · Taxonomie A1–A5 · Pipeline de purge en 6 étapes",
        "result":   f"✓ {len(df):,} stations certifiées · Bordeaux : rang 2 → 14 après correction A3",
    },
    {
        "badge": "Axe 1", "page": "IMD", "color": "#1A6FBF",
        "question": "La qualité cyclable se réduit-elle au volume de stations ?",
        "methode":  "Indice composite 4D (S, I, M, T) · Évolution différentielle · Monte Carlo N = 10 000",
        "result":   f"✓ Top 10 stable 89 % des simulations · w_M* = 0,578 · #1 : {_top_city} (IMD = {_top_imd:.1f}/100)",
    },
    {
        "badge": "Axe 2", "page": "IES", "color": "#27ae60",
        "question": "L'offre cyclable est-elle équitablement distribuée socialement ?",
        "methode":  "Modèle Ridge (λ par CV) · IES = IMD_obs / IMD_prédit(Revenu) · Bootstrap CI N = 2 000",
        "result":   f"✓ ρ = {_rho_str} (p = {_pval_str}, n.s.) · R² = {_R2_str} · Gouvernance > Économie",
    },
    {
        "badge": "Axe 3", "page": "Villes", "color": "#8e44ad",
        "question": "Les disparités inter-urbaines sont-elles géographiques ou politiques ?",
        "methode":  "Indice global de Moran (autocorrélation spatiale) · Diagramme de Moran",
        "result":   "✓ Moran's I = −0,023 (p = 0,765, n.s.) · Déterminisme géographique invalidé",
    },
    {
        "badge": "Axe 4", "page": "Distributions", "color": "#16a085",
        "question": "La taille d'une agglomération prédit-elle sa performance cyclable ?",
        "methode":  "Spearman · Boîtes à encoches · Matrice de corrélation inter-dimensions",
        "result":   "✓ ρ_s = −0,02 (hors Paris) · Aucune corrélation taille–performance",
    },
    {
        "badge": "Axe 5", "page": "Topographie", "color": "#2c3e50",
        "question": "Le relief contraint-il structurellement la qualité des réseaux VLS ?",
        "methode":  "TRI (Riley 1999, SRTM 30 m) · Haversine inter-stations · Score d'effort cyclable",
        "result":   "✓ Brest/Saint-Étienne > Montpellier > Calais · Effort VAE −60 % · Composante T validée",
    },
    {
        "badge": "Axe 6", "page": "Montpellier", "color": "#c0392b",
        "question": "Les modèles nationaux se valident-ils à l'échelle micro-locale ?",
        "methode":  "Graphes Louvain · PageRank · GTFS · IES intra-urbain par quartier (iris)",
        "result":   "✓ Structure bimodale Commuter confirmée · Fracture socio-spatiale cartographiée",
    },
]

# Grille 2 colonnes
_ax_c1, _ax_c2 = st.columns(2)
for _i, _ax in enumerate(_axes_cards):
    _col = _ax_c1 if _i % 2 == 0 else _ax_c2
    with _col:
        st.markdown(
            f"<div style='border-left:4px solid {_ax['color']};background:#f8f9fa;"
            f"border-radius:6px;padding:10px 14px;margin-bottom:10px;'>"
            f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:4px'>"
            f"<span style='background:{_ax['color']};color:white;font-size:0.72rem;"
            f"font-weight:700;padding:1px 7px;border-radius:3px'>{_ax['badge']}</span>"
            f"<span style='font-size:0.78rem;color:#555;font-weight:600'>{_ax['page']}</span>"
            f"</div>"
            f"<div style='font-size:0.83rem;font-weight:600;color:#1A2332;margin-bottom:3px'>"
            f"{_ax['question']}</div>"
            f"<div style='font-size:0.75rem;color:#666;margin-bottom:3px'>"
            f"<em>{_ax['methode']}</em></div>"
            f"<div style='font-size:0.78rem;color:{_ax['color']};font-weight:500'>"
            f"{_ax['result']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
st.caption(
    "**Figure 3.2.** Architecture des six axes de recherche. "
    "Chaque carte correspond à une page dédiée dans la barre de navigation latérale. "
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
    _n_rank_show = st.slider(
        "Agglomérations à afficher (classement IMD)",
        min_value=10, max_value=min(40, _n_imd), value=min(20, _n_imd), step=5,
        key="home_rank_slider",
    )
    _top_n = _imd_ranked.head(_n_rank_show).copy()
    _top_n["_pctile"] = ((_n_imd - np.arange(len(_top_n))) / _n_imd * 100).round(0).astype(int)

    _fig_rank = go.Figure()
    _fig_rank.add_trace(go.Bar(
        x=_top_n["IMD"],
        y=_top_n["city"],
        orientation="h",
        marker_color=[
            "#e74c3c" if c == "Montpellier"
            else "#f1c40f" if c == _top_city
            else "#1A6FBF"
            for c in _top_n["city"]
        ],
        text=_top_n["IMD"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
        textfont=dict(size=10),
        customdata=_top_n[["_pctile"]].values,
        hovertemplate="<b>%{y}</b><br>IMD = %{x:.1f} / 100<br>Percentile : top %{customdata[0]:.0f} %<extra></extra>",
    ))
    # Ligne médiane
    _fig_rank.add_vline(
        x=_imd_median, line_dash="dot", line_color="#aaa", line_width=1.5,
        annotation_text=f"Médiane {_imd_median:.1f}", annotation_position="top right",
        annotation_font_size=9,
    )
    _fig_rank.update_layout(
        title=dict(text=f"Classement national IMD — top {_n_rank_show} / {_n_imd} agglomérations",
                   font=dict(size=11), x=0),
        height=max(340, _n_rank_show * 20),
        margin=dict(l=10, r=65, t=38, b=30),
        xaxis=dict(range=[0, 112], title="IMD / 100", gridcolor="#e8edf3", tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafd",
    )
    st.plotly_chart(_fig_rank, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 4.1.** Classement IMD national (top {_n_rank_show}). "
        f"Montpellier (rang #{_mmm_rank}, rouge) · médiane nationale : {_imd_median:.1f}/100. "
        "Jaune = agglomération #1. Ligne pointillée = médiane."
    )

    # Violin distribution nationale
    _fig_violin = go.Figure()
    _fig_violin.add_trace(go.Violin(
        x=_imd_ranked["IMD"],
        name="Distribution nationale",
        line_color="#1A6FBF",
        fillcolor="rgba(26,111,191,0.15)",
        meanline_visible=True,
        orientation="h",
        side="positive",
        width=1.8,
        points="all",
        pointpos=-0.9,
        marker=dict(size=4, color="#1A6FBF", opacity=0.4),
        hovertemplate="IMD %{x:.1f}<extra></extra>",
    ))
    if not _mmm_row.empty:
        _fig_violin.add_vline(
            x=float(_mmm_imd), line_dash="dash", line_color="#e74c3c", line_width=2,
            annotation_text=f"Montpellier {_mmm_imd:.1f}",
            annotation_font=dict(size=9, color="#e74c3c"),
            annotation_position="top left",
        )
    _fig_violin.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="IMD / 100", gridcolor="#e8edf3", range=[0, 105]),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafd",
        showlegend=False,
        title=dict(text="Distribution des scores IMD (toutes agglomérations)", font_size=10, x=0),
    )
    st.plotly_chart(_fig_violin, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"**Figure 4.2.** Violin plot de la distribution nationale des scores IMD "
        f"({_n_imd} agglomérations dock-based). Points individuels = chaque agglomération. "
        f"Ligne pointillée rouge = Montpellier (rang #{_mmm_rank})."
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

_nav_cards = [
    {
        "module": "Gold Standard", "axe": "Axe Prél.", "color": "#e67e22",
        "file": "pages/00_Gold_Standard.py",
        "desc": f"Taxonomie A1–A5, pipeline de purge en 6 étapes, complétude de l'enrichissement, "
                f"catalogue des {n_certified} systèmes certifiés ({len(df):,} stations).",
    },
    {
        "module": "IMD", "axe": "Axe 1", "color": "#1A6FBF",
        "file": "pages/0_IMD.py",
        "desc": f"Formulation 4D (S, I, M, T), poids w_M* = 0,578, Monte Carlo N = 10 000, "
                f"classement national ({_top_city} #1, IMD = {_top_imd:.1f}/100), validation FUB/EMP.",
    },
    {
        "module": "IES", "axe": "Axe 2", "color": "#27ae60",
        "file": "pages/7_IES.py",
        "desc": f"Indice d'Équité Sociale, modèle Ridge, 4 régimes cyclables, "
                f"R² = {_R2_str}, bootstrap CI N = 2 000, Mann-Whitney U.",
    },
    {
        "module": "Carte", "axe": "Transversal", "color": "#16a085",
        "file": "pages/1_Carte.py",
        "desc": f"Visualisation WebGL des {len(df):,} stations (pydeck), filtrage "
                "par dimension d'enrichissement, distribution empirique, classement par agglomération.",
    },
    {
        "module": "Villes", "axe": "Axe 3", "color": "#8e44ad",
        "file": "pages/2_Villes.py",
        "desc": "Classement univarié, scatter infra × sinistralité, heatmap agglomérations × dimensions, "
                "profil radar comparatif, diagramme de Moran, carte nationale de la qualité cyclable.",
    },
    {
        "module": "Distributions", "axe": "Axe 4", "color": "#2c3e50",
        "file": "pages/3_Distributions.py",
        "desc": "Histogrammes, boîtes à encoches, matrice de corrélation Spearman, "
                "statistiques de forme (γ₁, γ₂, Shapiro-Wilk), tests formels de normalité.",
    },
    {
        "module": "Topographie", "axe": "Axe 5", "color": "#566573",
        "file": "pages/8_Topographie.py",
        "desc": "TRI (SRTM 30 m), classement national de rugosité, score d'effort cyclable, "
                "simulateur VAE, profil altimétrique par agglomération, distances haversine.",
    },
    {
        "module": "Mobilité France", "axe": "Transversal", "color": "#c0392b",
        "file": "pages/5_Mobilite_France.py",
        "desc": "Triangulation FUB 2023, EMP 2019, éco-compteurs, BAAC et Cerema — "
                "indicateurs nationaux de la mobilité cyclable et validation externe.",
    },
    {
        "module": "Montpellier", "axe": "Axe 6", "color": "#e74c3c",
        "file": "pages/6_Montpellier.py",
        "desc": "Graphes Louvain, déséquilibres source/puits, vulnérabilité V_i, "
                "intégration GTFS tramway, super-spreaders, fracture socio-spatiale IES intra-urbaine.",
    },
    {
        "module": "Export", "axe": "FAIR", "color": "#7f8c8d",
        "file": "pages/4_Export.py",
        "desc": "Gold Standard en accès libre (CSV / Parquet), filtres multi-critères, "
                "dictionnaire de variables complet, métadonnées FAIR pour citation académique.",
    },
]

_nav_c1, _nav_c2 = st.columns(2)
for _ni, _nc in enumerate(_nav_cards):
    _col = _nav_c1 if _ni % 2 == 0 else _nav_c2
    with _col:
        st.markdown(
            f"<div style='border:1px solid #dde3ea;border-radius:6px;padding:10px 14px;"
            f"margin-bottom:4px;background:#fafbfc;border-top:3px solid {_nc['color']}'>"
            f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:5px'>"
            f"<span style='font-weight:700;font-size:0.9rem;color:#1A2332'>{_nc['module']}</span>"
            f"<span style='margin-left:auto;background:{_nc['color']}20;color:{_nc['color']};"
            f"font-size:0.68rem;font-weight:600;padding:1px 7px;border-radius:3px'>{_nc['axe']}</span>"
            f"</div>"
            f"<div style='font-size:0.77rem;color:#555;line-height:1.5;margin-bottom:6px'>"
            f"{_nc['desc']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.page_link(_nc["file"], label=f"Ouvrir {_nc['module']}", use_container_width=True)

st.caption(
    "**Figure 5.1.** Guide de navigation par module analytique. "
    "Les modules 'Transversal' ne sont pas rattachés à un axe unique. "
    "Le module 'FAIR' implémente les principes *Findable, Accessible, Interoperable, Reusable*."
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
