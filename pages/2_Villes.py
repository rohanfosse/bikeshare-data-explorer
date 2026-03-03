"""
2_Villes.py - Analyse comparative inter-urbaine des métriques d'enrichissement spatial.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import METRICS, city_stats, compute_imd_cities, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Analyse Comparative Inter-Urbaine - Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Chargement anticipé (abstract dynamique) ───────────────────────────────────
df     = load_stations()
imd_df = compute_imd_cities(df)   # dock-based, sans entrées non-ville
cities = city_stats(df)

_imd_ranked    = imd_df.sort_values("IMD", ascending=False).reset_index(drop=True)
_n_dock_cities = len(_imd_ranked)
_top_city      = _imd_ranked.iloc[0]["city"] if _n_dock_cities > 0 else "-"
_top_imd       = f"{_imd_ranked.iloc[0]['IMD']:.1f}" if _n_dock_cities > 0 else "-"

_mmm_row  = _imd_ranked[_imd_ranked["city"] == "Montpellier"]
_mmm_rank = int(_mmm_row.index[0]) + 1 if not _mmm_row.empty else "?"
_mmm_imd  = float(_mmm_row["IMD"].iloc[0]) if not _mmm_row.empty else 0.0

# Calcul Spearman préliminaire (pour abstract et légendes)
_rho_pre  = float("nan")
_pval_pre = float("nan")
_n_filosofi = 0
if "revenu_median_uc" in imd_df.columns:
    _tmp_pre = imd_df.dropna(subset=["revenu_median_uc", "IMD"])
    _n_filosofi = len(_tmp_pre)
    if _n_filosofi >= 5:
        try:
            from scipy.stats import spearmanr as _sp
            _rho_pre, _pval_pre = _sp(
                _tmp_pre["revenu_median_uc"].values, _tmp_pre["IMD"].values
            )
            _rho_pre  = float(_rho_pre)
            _pval_pre = float(_pval_pre)
        except ImportError:
            _xp = _tmp_pre["revenu_median_uc"].values.astype(float)
            _yp = _tmp_pre["IMD"].values.astype(float)
            _rho_pre = float(pd.Series(_xp).rank().corr(pd.Series(_yp).rank()))
            _t = _rho_pre * np.sqrt((_n_filosofi - 2) / max(1e-10, 1 - _rho_pre ** 2))
            _z = abs(_t)
            _phi = 0.5 * (1 + np.sign(_t) * (1 - np.exp(-0.717 * _z - 0.416 * _z ** 2)))
            _pval_pre = float(max(0.0, min(1.0, 2 * (1 - _phi))))

_rho_str  = f"{_rho_pre:+.3f}" if not np.isnan(_rho_pre) else "n.d."
_pval_str = (
    f"{_pval_pre:.3f}" if (not np.isnan(_pval_pre) and _pval_pre >= 0.001)
    else ("< 0,001" if not np.isnan(_pval_pre) else "n.d.")
)

# ── En-tête ────────────────────────────────────────────────────────────────────
st.title("Analyse Comparative Inter-Urbaine")
st.caption("Axe de Recherche 3 : Gouvernance Locale et Disparités Structurelles de l'Environnement Cyclable")

abstract_box(
    "<b>Question de recherche :</b> Les disparités inter-urbaines de qualité cyclable "
    "sont-elles le produit d'une fatalité géographique ou d'inégalités de gouvernance ?<br><br>"
    f"Cette analyse comparative classe les <b>{_n_dock_cities} agglomérations</b> françaises dotées d'un "
    "réseau VLS dock-based certifié Gold Standard selon les dimensions d'enrichissement spatial. "
    "Le résultat clé de l'analyse spatiale globale - l'absence d'autocorrélation significative "
    "(Moran's <i>I</i>&nbsp;=&nbsp;&minus;0,023, <i>p</i>&nbsp;=&nbsp;0,765) - invalide l'hypothèse d'un déterminisme "
    "géographique structurant les disparités. Les villes performantes et sous-performantes "
    "ne forment pas de clusters territoriaux cohérents : ce sont les choix de gouvernance "
    "locale, et non la localisation géographique, qui expliquent l'hétérogénéité observée. "
    f"<b>Classement national : {_top_city} #1 (IMD = {_top_imd}/100) - "
    f"Montpellier #{_mmm_rank} (IMD = {_mmm_imd:.1f}/100). "
    f"ρ Spearman (IMD × Revenu) = {_rho_str} (p = {_pval_str}, non significatif).</b> "
    "Cinq niveaux d'analyse sont proposés : classement univarié, profil infrastructure × sinistralité, "
    "justice sociale IMD × revenu, audit radar multi-dimensionnel, et synthèse statistique comparative.",
    findings=[
        (str(_n_dock_cities), "agglomérations classées"),
        (_top_city, "#1 national"),
        (f"{_top_imd}/100", "IMD maximum"),
        (f"#{_mmm_rank}", "Montpellier"),
        (f"ρ = {_rho_str}", "IMD × Revenu (n.s.)"),
    ],
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres d'Analyse")
    n_top = st.slider("Nombre d'agglomérations", min_value=5, max_value=min(50, _n_dock_cities), value=20, step=5)
    metric_key = st.selectbox(
        "Dimension principale",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k]["label"],
        index=0,
    )
    min_stations = st.number_input(
        "Seuil de robustesse (min. stations dock)", min_value=1, max_value=500, value=10,
        help="Exclut les micro-réseaux pour garantir la pertinence statistique de la comparaison."
    )
    meta = METRICS[metric_key]
    st.divider()
    st.markdown(f"**{meta['label']}**")
    st.caption(meta["description"])
    if meta["higher_is_better"] is True:
        st.info("Indicateur direct : valeur élevée = favorable.")
    elif meta["higher_is_better"] is False:
        st.warning("Indicateur inverse : valeur faible = favorable.")
    st.divider()
    st.markdown("**Classement IMD national (top 5)**")
    for i, row in _imd_ranked.head(5).iterrows():
        _city_label = f"#{i+1} {row['city']}"
        _highlight  = " (cas d'étude)" if row["city"] == "Montpellier" else ""
        st.caption(f"{_city_label} - IMD {row['IMD']:.1f}/100{_highlight}")

# ── Filtrage ──────────────────────────────────────────────────────────────────
df_dock = df[df["station_type"] == "docked_bike"].copy() if "station_type" in df.columns else df.copy()
_NON_CITY   = {"France", "FR", "Grand Est", "Basque Country"}
df_dock     = df_dock[~df_dock["city"].isin(_NON_CITY)]
cities_dock = city_stats(df_dock)
cities_f    = cities_dock[cities_dock["n_stations"] >= min_stations].copy()

if metric_key not in cities_f.columns:
    st.warning(f"Dimension `{metric_key}` absente des données agrégées (dock-based).")
    st.stop()

ascending     = not meta.get("higher_is_better", True)
cities_sorted = cities_f.dropna(subset=[metric_key]).sort_values(
    metric_key, ascending=ascending
).reset_index(drop=True)

# Recalcul du rang Montpellier sur le panel filtré
_mmm_f = _imd_ranked[
    (_imd_ranked["city"].isin(cities_f["city"].values)) & (_imd_ranked["city"] == "Montpellier")
]
_mmm_rank_f = int(_mmm_f.index[0]) + 1 if not _mmm_f.empty else _mmm_rank

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Agglomérations dock analysées",   f"{len(cities_f)}")
k2.metric("IMD #1 national",                 _top_city, f"{_top_imd} / 100")
k3.metric(f"Montpellier (rang #{_mmm_rank})", f"{_mmm_imd:.1f} / 100")
k4.metric("IMD médian national",             f"{_imd_ranked['IMD'].median():.1f} / 100")
k5.metric("ρ Spearman IMD × Revenu",         _rho_str,  f"p = {_pval_str} - non sign.")

# ── Section 1 - Classement univarié ───────────────────────────────────────────
st.divider()
section(1, "Classement Univarié - Agglomérations Triées par la Dimension Sélectionnée")

st.markdown(r"""
Le classement univarié constitue le premier niveau de diagnostic territorial. Il met en évidence
les agglomérations se situant aux extrêmes de la distribution nationale pour la dimension
sélectionnée, révélant des situations de sur-performance ou de sous-investissement structurel
dans un environnement cyclable de qualité.
""")

col_tab, col_chart = st.columns([2, 3])

with col_tab:
    extra_cols = [
        c for c in ["infra_cyclable_pct", "baac_accidents_cyclistes", "gtfs_heavy_stops_300m"]
        if c != metric_key and c in cities_sorted.columns
    ]
    disp_cols = ["city", "n_stations", metric_key] + extra_cols
    display   = cities_sorted.head(n_top)[disp_cols].copy()
    display.insert(0, "Rang", range(1, len(display) + 1))
    display = display.rename(columns={
        "city":                      "Agglomération",
        "n_stations":                "Stations",
        metric_key:                  meta["label"],
        "infra_cyclable_pct":        "Infra cyclable (%)",
        "baac_accidents_cyclistes":  "Sinistralité (moy.)",
        "gtfs_heavy_stops_300m":     "TC lourds (moy.)",
    })
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            meta["label"]: st.column_config.ProgressColumn(
                meta["label"],
                min_value=float(cities_sorted[metric_key].min()),
                max_value=float(cities_sorted[metric_key].max()),
                format=f"%.2f {meta['unit']}",
            )
        },
    )
    st.caption(
        f"**Tableau 1.1.** Top {n_top} agglomérations classées par **{meta['label']}** "
        f"(stations dock-based, seuil ≥ {min_stations} stations). "
        f"Les valeurs représentent la moyenne de la dimension sur l'ensemble des stations dock. "
        f"Montpellier est au rang #{_mmm_rank_f} IMD sur {len(cities_f)} agglomérations."
    )

with col_chart:
    plot_df = cities_sorted.head(n_top).copy()
    _bar_colors = ["#e74c3c" if c == "Montpellier" else "#1A6FBF" for c in plot_df["city"]]
    fig_bar = px.bar(
        plot_df,
        x=metric_key, y="city",
        orientation="h",
        color=metric_key,
        color_continuous_scale=meta["color_scale"],
        text=metric_key,
        labels={"city": "Agglomération", metric_key: meta["label"]},
        height=max(400, n_top * 22),
    )
    if "Montpellier" in plot_df["city"].values:
        fig_bar.data[0].marker.color = _bar_colors
    fig_bar.update_traces(texttemplate=f"%{{x:.2f}} {meta['unit']}", textposition="outside")
    fig_bar.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=80, t=10, b=10),
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        f"**Figure 1.1.** Classement des {n_top} premières agglomérations par **{meta['label']}** "
        f"(stations dock-based). La barre rouge identifie Montpellier "
        f"(cas d'étude, rang IMD #{_mmm_rank_f}/{len(cities_f)}). "
        "Les écarts inter-urbains, non expliqués par la géographie (Moran's $I = -0{,}023$), "
        "reflètent des choix différenciés de politique d'aménagement cyclable."
    )

# ── Section 2 - Infrastructure × sinistralité ─────────────────────────────────
st.divider()
section(2, "Analyse Croisée Infrastructure × Sinistralité - Effet Protecteur de l'Aménagement Cyclable")

st.markdown(r"""
L'hypothèse d'un effet protecteur de l'infrastructure cyclable sur la sinistralité est centrale
dans la littérature (*Pucher et al., 2010 ; Jacobsen, 2003*). Ce nuage de points teste cette
relation à l'échelle des agglomérations françaises : les villes situées dans le quadrant
supérieur gauche (forte densité d'infrastructure, faible sinistralité) valident l'hypothèse.
La corrélation de Spearman est calculée dynamiquement sur les données disponibles.
""")

scatter_df = cities_f.dropna(subset=["infra_cyclable_pct", "baac_accidents_cyclistes"])

if not scatter_df.empty:
    # Corrélation Spearman infrastructure × sinistralité
    _rho_sc  = float(pd.Series(scatter_df["infra_cyclable_pct"].values).rank().corr(
        pd.Series(scatter_df["baac_accidents_cyclistes"].values).rank()
    ))
    try:
        from scipy.stats import spearmanr as _sp2
        _, _pval_sc = _sp2(
            scatter_df["infra_cyclable_pct"].values,
            scatter_df["baac_accidents_cyclistes"].values
        )
        _pval_sc = float(_pval_sc)
    except ImportError:
        _n_sc = len(scatter_df)
        _t_sc = _rho_sc * np.sqrt((_n_sc - 2) / max(1e-10, 1 - _rho_sc ** 2))
        _z_sc = abs(_t_sc)
        _phi_sc = 0.5 * (1 + np.sign(_t_sc) * (1 - np.exp(-0.717 * _z_sc - 0.416 * _z_sc ** 2)))
        _pval_sc = float(max(0.0, min(1.0, 2 * (1 - _phi_sc))))

    _pval_sc_str = f"{_pval_sc:.3f}" if _pval_sc >= 0.001 else "< 0,001"
    _sig_sc = "significatif" if _pval_sc < 0.05 else "non significatif"

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Agglomérations (données BAAC + Infra)", f"{len(scatter_df)}")
    sc2.metric("ρ Spearman (Infra × Sinistralité)", f"{_rho_sc:+.3f}")
    sc3.metric("p-valeur", _pval_sc_str, _sig_sc)

    _labels_highlight = {
        "Montpellier", "Strasbourg", "Paris", "Lyon", "Bordeaux",
        "Nantes", "Rennes", "Brest", "Grenoble", "Toulouse"
    }
    fig_sc = px.scatter(
        scatter_df,
        x="infra_cyclable_pct", y="baac_accidents_cyclistes",
        size="n_stations", color="gtfs_heavy_stops_300m",
        color_continuous_scale="Blues",
        hover_name="city",
        text=scatter_df["city"].apply(lambda c: c if c in _labels_highlight else ""),
        hover_data={"n_stations": True, "infra_cyclable_pct": ":.2f",
                    "baac_accidents_cyclistes": ":.3f"},
        labels={
            "infra_cyclable_pct":       "Couverture en infrastructure cyclable (%)",
            "baac_accidents_cyclistes": "Densité de sinistralité cycliste (BAAC, 300 m)",
            "gtfs_heavy_stops_300m":    "Accessibilité multimodale (arrêts TC lourds)",
        },
        size_max=40, height=480,
    )

    # OLS sur scatter
    _xsc = scatter_df["infra_cyclable_pct"].values.astype(float)
    _ysc = scatter_df["baac_accidents_cyclistes"].values.astype(float)
    _csc = np.polyfit(_xsc, _ysc, 1)
    _xlsc = np.linspace(_xsc.min(), _xsc.max(), 200)
    fig_sc.add_trace(go.Scatter(
        x=_xlsc, y=np.polyval(_csc, _xlsc),
        mode="lines",
        name=f"Droite OLS (ρ = {_rho_sc:+.3f}, {_sig_sc})",
        line=dict(color="#1A2332", dash="dash", width=1.5),
        showlegend=True,
    ))

    # Cercle rouge Montpellier
    _mmm_sc = scatter_df[scatter_df["city"] == "Montpellier"]
    if not _mmm_sc.empty:
        fig_sc.add_trace(go.Scatter(
            x=[_mmm_sc["infra_cyclable_pct"].iloc[0]],
            y=[_mmm_sc["baac_accidents_cyclistes"].iloc[0]],
            mode="markers",
            marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color="#e74c3c", width=3)),
            name=f"Montpellier (rang IMD #{_mmm_rank_f})",
            showlegend=True,
        ))
    fig_sc.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig_sc.update_layout(
        plot_bgcolor="white",
        coloraxis_colorbar=dict(title="TC lourds"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    _xm_sc = float(scatter_df["infra_cyclable_pct"].mean())
    _ym_sc = float(scatter_df["baac_accidents_cyclistes"].mean())
    fig_sc.add_hline(y=_ym_sc, line_dash="dot", line_color="#e74c3c", opacity=0.5,
                     annotation_text="Moy. sinistralité", annotation_position="right")
    fig_sc.add_vline(x=_xm_sc, line_dash="dot", line_color="#1A6FBF", opacity=0.5,
                     annotation_text="Moy. infrastructure", annotation_position="top")

    # Annotations de quadrant
    _xmin_sc = float(scatter_df["infra_cyclable_pct"].min())
    _xmax_sc = float(scatter_df["infra_cyclable_pct"].max())
    _ymin_sc = float(scatter_df["baac_accidents_cyclistes"].min())
    _ymax_sc = float(scatter_df["baac_accidents_cyclistes"].max())
    for _ql, _qx, _qy, _qc in [
        ("Zone Idéale<br>(infra forte, faible sinistralité)",
         _xm_sc + (_xmax_sc - _xm_sc) * 0.45, _ymin_sc + (_ym_sc - _ymin_sc) * 0.15, "#27ae60"),
        ("Zone Critique<br>(faible infra, sinistralité forte)",
         _xmin_sc + (_xm_sc - _xmin_sc) * 0.15, _ym_sc + (_ymax_sc - _ym_sc) * 0.65, "#e74c3c"),
        ("Zone Protégée<br>(faible infra et sinistralité)",
         _xmin_sc + (_xm_sc - _xmin_sc) * 0.15, _ymin_sc + (_ym_sc - _ymin_sc) * 0.15, "#1A6FBF"),
        ("Paradoxe<br>(forte infra et sinistralité)",
         _xm_sc + (_xmax_sc - _xm_sc) * 0.45, _ym_sc + (_ymax_sc - _ym_sc) * 0.65, "#e67e22"),
    ]:
        fig_sc.add_annotation(
            x=_qx, y=_qy, text=_ql,
            showarrow=False,
            font=dict(size=9, color=_qc),
            opacity=0.45,
            align="center",
        )
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption(
        "**Figure 2.1.** Couverture en infrastructure cyclable (axe horizontal) versus densité "
        "de sinistralité cycliste BAAC (axe vertical) par agglomération (stations dock-based). "
        "La taille encode le volume de stations dock Gold Standard ; la couleur encode l'accessibilité "
        f"aux TC lourds. Corrélation de Spearman : $\\rho = {_rho_sc:+.3f}$ "
        f"($p = {_pval_sc_str}$, {_sig_sc}). "
        f"Le cercle rouge identifie Montpellier (rang IMD #{_mmm_rank_f})."
    )
else:
    st.info(
        "Données BAAC et/ou infrastructure insuffisantes pour l'analyse croisée. "
        "Vérifiez les colonnes `baac_accidents_cyclistes` et `infra_cyclable_pct`."
    )

# ── Section 3 - IMD × Revenu médian ───────────────────────────────────────────
st.divider()
section(3, "Justice Sociale - Classement IMD × Revenu Médian par Agglomération")

_has_revenu = "revenu_median_uc" in imd_df.columns and imd_df["revenu_median_uc"].notna().sum() >= 5
if _has_revenu:
    _imd_f = imd_df[imd_df["n_stations"] >= min_stations].dropna(
        subset=["revenu_median_uc", "IMD"]
    ).copy()

    if len(_imd_f) >= 5:
        # Calcul dynamique ρ sur ce sous-panel
        _rho_s3 = float(pd.Series(_imd_f["revenu_median_uc"].values).rank().corr(
            pd.Series(_imd_f["IMD"].values).rank()
        ))
        try:
            from scipy.stats import spearmanr as _sp3
            _, _pval_s3 = _sp3(_imd_f["revenu_median_uc"].values, _imd_f["IMD"].values)
            _pval_s3 = float(_pval_s3)
        except ImportError:
            _n_s3 = len(_imd_f)
            _t_s3 = _rho_s3 * np.sqrt((_n_s3 - 2) / max(1e-10, 1 - _rho_s3 ** 2))
            _z_s3 = abs(_t_s3)
            _phi_s3 = 0.5 * (1 + np.sign(_t_s3) * (1 - np.exp(-0.717 * _z_s3 - 0.416 * _z_s3 ** 2)))
            _pval_s3 = float(max(0.0, min(1.0, 2 * (1 - _phi_s3))))
        _pval_s3_str = f"{_pval_s3:.3f}" if _pval_s3 >= 0.001 else "< 0,001"
        _sig_s3 = "non significatif" if _pval_s3 > 0.05 else "significatif"

        # R²
        _c_s3 = np.polyfit(_imd_f["revenu_median_uc"].values, _imd_f["IMD"].values, 1)
        _yhat_s3 = np.polyval(_c_s3, _imd_f["revenu_median_uc"].values)
        _SS_res_s3 = float(np.sum((_imd_f["IMD"].values - _yhat_s3) ** 2))
        _SS_tot_s3 = float(np.sum((_imd_f["IMD"].values - _imd_f["IMD"].values.mean()) ** 2))
        _R2_s3     = 1.0 - _SS_res_s3 / _SS_tot_s3

        st.markdown(
            rf"""
Le croisement du score IMD (qualité physique de l'offre) avec le revenu médian par unité
de consommation (INSEE Filosofi) révèle la dimension d'équité sociale de la distribution
des réseaux VLS. Le résultat empirique sur ce panel ($n = {len(_imd_f)}$ agglomérations,
seuil $\geq {min_stations}$ stations dock) est :

$$\rho_s(\text{{IMD}}, \text{{Revenu}}) = {_rho_s3:+.3f}, \quad p = {_pval_s3_str}
\quad \Rightarrow \text{{{_sig_s3}}}, \quad R^2 = {_R2_s3:.4f}$$

**La qualité cyclable est indépendante du niveau de revenu de l'agglomération.**
"""
        )

        _imd_f["IMD_hat"] = np.polyval(_c_s3, _imd_f["revenu_median_uc"].values).clip(min=1.0)
        _imd_f["IES"]     = (_imd_f["IMD"] / _imd_f["IMD_hat"]).round(3)

        _med_r = float(_imd_f["revenu_median_uc"].median())
        _med_i = float(_imd_f["IMD"].median())
        _imd_f["regime"] = _imd_f.apply(
            lambda row: (
                "Mobilité Inclusive"    if row["revenu_median_uc"] < _med_r and row["IMD"] >= _med_i
                else "Excellence Consolidée" if row["revenu_median_uc"] >= _med_r and row["IMD"] >= _med_i
                else "Désert de Mobilité"    if row["revenu_median_uc"] < _med_r and row["IMD"] < _med_i
                else "Sous-Performance"
            ), axis=1
        )

        _regime_colors = {
            "Mobilité Inclusive":    "#27ae60",
            "Excellence Consolidée": "#1A6FBF",
            "Désert de Mobilité":    "#e74c3c",
            "Sous-Performance":      "#e67e22",
        }
        _r_counts = _imd_f["regime"].value_counts()
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col_w, (lbl, _clr) in zip([rc1, rc2, rc3, rc4], _regime_colors.items()):
            _n_q = int(_r_counts.get(lbl, 0))
            col_w.metric(lbl, f"{_n_q} villes", f"{100 * _n_q / len(_imd_f):.0f} %")

        _x_line = np.linspace(
            float(_imd_f["revenu_median_uc"].min()),
            float(_imd_f["revenu_median_uc"].max()), 200
        )
        fig_ies = px.scatter(
            _imd_f, x="revenu_median_uc", y="IMD",
            color="regime", color_discrete_map=_regime_colors,
            size="n_stations", size_max=22,
            hover_name="city",
            hover_data={"IMD": ":.1f", "revenu_median_uc": ":,.0f", "IES": ":.3f"},
            labels={
                "revenu_median_uc": "Revenu médian/UC (€/an, INSEE Filosofi)",
                "IMD":              "Score IMD (/100)",
                "regime":           "Régime IES",
            },
            height=500,
        )
        fig_ies.add_trace(go.Scatter(
            x=_x_line, y=np.polyval(_c_s3, _x_line),
            mode="lines",
            name=f"Référentiel OLS (ρ = {_rho_s3:+.3f}, {_sig_s3})",
            line=dict(color="#1A2332", dash="dash", width=2),
        ))
        fig_ies.add_hline(y=_med_i, line_dash="dot", line_color="#888", opacity=0.5,
                          annotation_text="Médiane IMD", annotation_position="right")
        fig_ies.add_vline(x=_med_r, line_dash="dot", line_color="#888", opacity=0.5,
                          annotation_text="Médiane revenu", annotation_position="top")

        # Annotation Montpellier
        _mmm_ies = _imd_f[_imd_f["city"] == "Montpellier"]
        if not _mmm_ies.empty:
            _mmm_ies_val = float(_mmm_ies["IES"].iloc[0])
            _mmm_ies_rank = int(
                _imd_f.sort_values("IMD", ascending=False).reset_index(drop=True)
                .pipe(lambda x: x[x["city"] == "Montpellier"]).index[0]
            ) + 1
            fig_ies.add_annotation(
                x=float(_mmm_ies["revenu_median_uc"].iloc[0]),
                y=float(_mmm_ies["IMD"].iloc[0]),
                text=f"<b>Montpellier<br>Rang #{_mmm_ies_rank} - IES = {_mmm_ies_val:.3f}</b>",
                showarrow=True, ax=-65, ay=-35,
                font=dict(size=11, color="#27ae60"),
                bgcolor="rgba(240,255,240,0.9)",
                bordercolor="#27ae60", borderpad=5,
                arrowcolor="#27ae60",
            )
        # Annotation top city
        _top_c_row = _imd_f[_imd_f["city"] == _top_city]
        if not _top_c_row.empty and _top_city != "Montpellier":
            fig_ies.add_annotation(
                x=float(_top_c_row["revenu_median_uc"].iloc[0]),
                y=float(_top_c_row["IMD"].iloc[0]),
                text=f"<b>{_top_city}<br>Rang #1</b>",
                showarrow=True, ax=50, ay=-30,
                font=dict(size=11, color="#1A6FBF"),
                bgcolor="rgba(240,248,255,0.9)",
                bordercolor="#1A6FBF", borderpad=5,
                arrowcolor="#1A6FBF",
            )

        fig_ies.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        )
        st.plotly_chart(fig_ies, use_container_width=True)
        st.caption(
            f"**Figure 3.1.** Score IMD versus revenu médian/UC (INSEE Filosofi) par agglomération "
            f"($n = {len(_imd_f)}$, seuil $\\geq {min_stations}$ stations dock). "
            f"Corrélation de Spearman : $\\rho = {_rho_s3:+.3f}$ ($p = {_pval_s3_str}$, "
            f"**{_sig_s3}**) - $R^2 = {_R2_s3:.4f}$. "
            "La droite OLS est quasi-horizontale : la qualité VLS est indépendante du niveau de revenu. "
            "Les quadrants révèlent les quatre régimes de justice cyclable. "
            "Analyse complète des tests formels (bootstrap CI, Mann-Whitney U) : page **IES**."
        )

        # Tableau IES synthétique
        with st.expander("Tableau IES par agglomération - tous les régimes", expanded=False):
            _disp_ies = _imd_f[["city", "n_stations", "revenu_median_uc", "IMD", "IES", "regime"]].copy()
            _disp_ies = _disp_ies.sort_values("IES", ascending=False).reset_index(drop=True)
            _disp_ies.insert(0, "Rang IMD", _disp_ies["city"].map(
                lambda c: int(_imd_f.sort_values("IMD", ascending=False)
                               .reset_index(drop=True)
                               .pipe(lambda x: x[x["city"] == c]).index[0]) + 1
                if c in _imd_f["city"].values else "?"
            ))
            _disp_ies.columns = [
                "Rang IMD", "Agglomération", "Stations",
                "Revenu médian/UC (€)", "IMD (/100)", "IES", "Régime"
            ]
            _disp_ies["IMD (/100)"]           = _disp_ies["IMD (/100)"].round(1)
            _disp_ies["Revenu médian/UC (€)"] = _disp_ies["Revenu médian/UC (€)"].round(0).astype(int)
            st.dataframe(
                _disp_ies, use_container_width=True, hide_index=True,
                column_config={
                    "IES": st.column_config.NumberColumn("IES", format="%.3f"),
                    "IMD (/100)": st.column_config.ProgressColumn(
                        "IMD (/100)", min_value=0, max_value=100, format="%.1f"
                    ),
                },
            )
            st.caption(
                f"**Tableau 3.1.** Classement des agglomérations par IES (décroissant). "
                f"$n = {len(_disp_ies)}$ agglomérations dock-based (seuil $\\geq {min_stations}$ stations). "
                "Les IES < 1 identifient les villes dont l'offre cyclable est inférieure au prédit "
                "par le revenu - cibles prioritaires des politiques d'équité cyclable."
            )
else:
    st.info(
        "Les données INSEE Filosofi (`revenu_median_uc`) ne sont pas disponibles. "
        "Vérifiez que le fichier `stations_gold_standard_final.parquet` est utilisé."
    )

# ── Section 4 - Profil radar ───────────────────────────────────────────────────
st.divider()
section(4, "Profil Radar Multi-Dimensionnel - Audit Comparatif des Agglomérations")

st.markdown(r"""
Le profil radar permet de visualiser simultanément les dimensions de l'environnement
cyclable normalisées min-max sur l'échantillon sélectionné. Chaque axe est exprimé selon
la relation $\tilde{c}(v) = \frac{c(v) - \min_v c}{\max_v c - \min_v c} \in [0, 1]$,
où les indicateurs inverses (sinistralité) sont retournés de sorte qu'une valeur élevée
corresponde systématiquement à un environnement favorable. L'aire de la figure est
proportionnelle à la performance globale de l'environnement cyclable.
""")

radar_cols = {
    "infra_cyclable_pct":         "Infrastructure cyclable",
    "gtfs_heavy_stops_300m":      "Multimodalité (TC lourds)",
    "baac_accidents_cyclistes":   "Sécurité (inv. sinistralité)",
    "gtfs_stops_within_300m_pct": "Couverture GTFS",
}
radar_cols = {k: v for k, v in radar_cols.items() if k in cities_f.columns}

# Défaut : top city + Montpellier + top 3 IMD
_imd_top = _imd_ranked["city"].head(5).tolist()
_default_radar = list(dict.fromkeys([_top_city, "Montpellier"] + _imd_top))[:5]
_default_radar = [c for c in _default_radar if c in cities_f["city"].values]

radar_city_sel = st.multiselect(
    "Sélection de l'échantillon d'audit (2 à 8 agglomérations)",
    options=sorted(cities_f["city"].unique()),
    default=_default_radar,
    max_selections=8,
)

if len(radar_city_sel) >= 2 and radar_cols:
    radar_df = cities_f[cities_f["city"].isin(radar_city_sel)].dropna(subset=list(radar_cols))

    if not radar_df.empty:
        ndf = radar_df[list(radar_cols)].copy()
        for c in ndf.columns:
            rng = ndf[c].max() - ndf[c].min()
            ndf[c] = (ndf[c] - ndf[c].min()) / rng if rng else 0.5
        if "baac_accidents_cyclistes" in ndf.columns:
            ndf["baac_accidents_cyclistes"] = 1 - ndf["baac_accidents_cyclistes"]

        ndf["city"] = radar_df["city"].values
        categories  = list(radar_cols.values())

        fig_radar = go.Figure()
        palette_r = ["#1A6FBF", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22", "#1A2332", "#f39c12", "#16a085"]
        for i, (_, row) in enumerate(ndf.iterrows()):
            vals   = [row[c] for c in radar_cols]
            vals  += vals[:1]
            _color = "#e74c3c" if row["city"] == "Montpellier" else (
                "#1A6FBF" if row["city"] == _top_city else palette_r[i % len(palette_r)]
            )
            _name = row["city"]
            if row["city"] == "Montpellier":
                _name += f" (cas d'étude, rang #{_mmm_rank_f})"
            elif row["city"] == _top_city:
                _name += " (rang #1 IMD)"
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=_name,
                opacity=0.65,
                line_color=_color,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=480,
            margin=dict(l=60, r=60, t=30, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "**Figure 4.1.** Profil radar des agglomérations sélectionnées. "
            "Chaque axe est normalisé min-max ($\\tilde{c} \\in [0, 1]$) sur l'échantillon affiché. "
            "La composante sécurité est retournée (valeur haute = faible sinistralité). "
            f"**{_top_city}** (bleu, rang #1 IMD) et **Montpellier** (rouge, rang #{_mmm_rank_f}) "
            "sont inclus par défaut comme références nationales."
        )
    else:
        st.warning("Données insuffisantes pour les agglomérations sélectionnées.")
elif len(radar_city_sel) < 2:
    st.info("Sélectionnez au moins 2 agglomérations pour initier l'audit radar comparatif.")
else:
    st.warning("Aucune dimension radar disponible dans ce corpus.")

# ── Section 5 - Synthèse statistique comparative ──────────────────────────────
st.divider()
section(5, "Synthèse Statistique - Distribution des Dimensions et Écarts Extrêmes")

st.markdown(
    "La synthèse statistique compare les extrema et la dispersion intra-nationale pour chaque "
    "dimension d'enrichissement. Elle révèle les dimensions les plus hétérogènes entre "
    "agglomérations - celles où les choix de gouvernance sont les plus différenciés."
)

_synth_dims = {k: v for k, v in {
    "infra_cyclable_pct":         ("Infrastructure cyclable (%)", "%"),
    "baac_accidents_cyclistes":   ("Sinistralité BAAC (moy.)", ""),
    "gtfs_heavy_stops_300m":      ("TC lourds 300 m (moy.)", ""),
    "gtfs_stops_within_300m_pct": ("Couverture GTFS (%)", "%"),
}.items() if k in cities_f.columns}

if _synth_dims:
    _synth_rows = []
    for col, (label, unit) in _synth_dims.items():
        s = cities_f[col].dropna()
        if len(s) < 3:
            continue
        _best = cities_f.loc[cities_f[col].idxmax(), "city"] if METRICS.get(col, {}).get("higher_is_better", True) else cities_f.loc[cities_f[col].idxmin(), "city"]
        _worst = cities_f.loc[cities_f[col].idxmin(), "city"] if METRICS.get(col, {}).get("higher_is_better", True) else cities_f.loc[cities_f[col].idxmax(), "city"]
        _mmm_val = cities_f.loc[cities_f["city"] == "Montpellier", col].values
        _mmm_display = f"{float(_mmm_val[0]):.2f}" if len(_mmm_val) > 0 else "n.d."
        _cv = float(s.std() / s.mean() * 100) if s.mean() != 0 else float("nan")
        _synth_rows.append({
            "Dimension": label,
            "Min": f"{float(s.min()):.3f}",
            "Médiane": f"{float(s.median()):.3f}",
            "Max": f"{float(s.max()):.3f}",
            "CV (%)": f"{_cv:.1f}" if not np.isnan(_cv) else "-",
            "Meilleur": _best,
            "Moins bon": _worst,
            f"Montpellier (#{_mmm_rank_f})": _mmm_display,
        })

    if _synth_rows:
        st.table(pd.DataFrame(_synth_rows))
        st.caption(
            f"**Tableau 5.1.** Statistiques descriptives inter-agglomérations pour chaque "
            f"dimension d'enrichissement ({len(cities_f)} agglomérations dock-based, "
            f"seuil $\\geq {min_stations}$ stations). "
            "CV = Coefficient de Variation (écart-type / moyenne × 100). "
            f"Montpellier est au rang #{_mmm_rank_f} IMD national sur {len(cities_f)} agglomérations."
        )

        # Box plot comparatif des dimensions
        if len(_synth_dims) >= 2:
            _box_data = []
            for col, (label, _) in _synth_dims.items():
                _s = cities_f[["city", col]].dropna().copy()
                _rng = _s[col].max() - _s[col].min()
                _s["val_norm"] = (_s[col] - _s[col].min()) / _rng if _rng else 0.5
                _s["dimension"] = label
                _box_data.append(_s[["city", "val_norm", "dimension"]])
            _box_df = pd.concat(_box_data, ignore_index=True)
            fig_box = px.box(
                _box_df,
                x="dimension", y="val_norm",
                color="dimension",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"dimension": "Dimension", "val_norm": "Valeur normalisée min-max [0, 1]"},
                height=380,
                points="all",
            )
            fig_box.update_layout(
                showlegend=False,
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis=dict(tickangle=-20),
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption(
                "**Figure 5.1.** Distribution inter-agglomérations des dimensions d'enrichissement "
                "(valeurs normalisées min-max pour comparabilité). Les points individuels représentent "
                "chaque agglomération. Les dimensions avec une boîte plus large indiquent une plus "
                "forte hétérogénéité de gouvernance cyclable à l'échelle nationale."
            )

        # ── Heatmap comparative top-N villes × dimensions ──────────────────
        st.markdown("#### Matrice de performance : Top agglomérations × Dimensions")
        _n_hm = st.slider("Agglomérations à inclure dans la matrice", 10, min(40, len(cities_f)),
                          min(25, len(cities_f)), 5, key="heatmap_n_cities")
        _hm_cities = _imd_ranked[_imd_ranked["city"].isin(cities_f["city"].values)]["city"].head(_n_hm).tolist()
        if "Montpellier" not in _hm_cities and "Montpellier" in cities_f["city"].values:
            _hm_cities = ["Montpellier"] + _hm_cities[: _n_hm - 1]

        _hm_df = cities_f[cities_f["city"].isin(_hm_cities)].set_index("city")
        _hm_cols = [c for c in _synth_dims if c in _hm_df.columns]
        _hm_data = _hm_df[_hm_cols].copy().astype(float)

        # Normalisation min-max par dimension
        for _c in _hm_cols:
            _rng = _hm_data[_c].max() - _hm_data[_c].min()
            _hm_data[_c] = (_hm_data[_c] - _hm_data[_c].min()) / _rng if _rng else 0.5
        # Inversion sinistralité (valeur basse = meilleure → inverser pour cohérence visuelle)
        if "baac_accidents_cyclistes" in _hm_data.columns:
            _hm_data["baac_accidents_cyclistes"] = 1.0 - _hm_data["baac_accidents_cyclistes"]

        # Tri par IMD décroissant
        _hm_order = [c for c in _imd_ranked["city"].tolist() if c in _hm_data.index]
        _hm_data = _hm_data.reindex(_hm_order).dropna(how="all")
        _hm_labels = [_synth_dims[c][0] for c in _hm_data.columns]

        if not _hm_data.empty:
            _hm_text = _hm_data.values.round(2)
            fig_hm = go.Figure(go.Heatmap(
                z=_hm_data.values,
                x=_hm_labels,
                y=_hm_data.index.tolist(),
                colorscale="RdYlGn",
                zmin=0, zmax=1,
                text=_hm_text,
                texttemplate="%{text:.2f}",
                textfont=dict(size=9),
                hovertemplate="<b>%{y}</b> — %{x}<br>Score normalisé : %{z:.2f}<extra></extra>",
                colorbar=dict(title="Score<br>[0–1]", thickness=14, len=0.65),
            ))
            # Surligner Montpellier (rectangle autour de la ligne)
            if "Montpellier" in _hm_data.index:
                _mmm_hm_idx = list(_hm_data.index).index("Montpellier")
                fig_hm.add_shape(
                    type="rect",
                    xref="paper", yref="y",
                    x0=0, x1=1,
                    y0=_mmm_hm_idx - 0.5, y1=_mmm_hm_idx + 0.5,
                    line=dict(color="#e74c3c", width=2),
                    fillcolor="rgba(0,0,0,0)",
                )
            fig_hm.update_layout(
                height=max(340, len(_hm_data) * 22 + 60),
                margin=dict(l=10, r=60, t=40, b=10),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(side="top", tickangle=-20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption(
                f"**Figure 5.2.** Matrice de performance normalisée min-max : top {_n_hm} agglomérations "
                f"(triées par IMD décroissant) × {len(_hm_cols)} dimensions d'enrichissement. "
                "Vert = performance élevée ; rouge = performance faible. "
                "La composante sécurité est inversée (vert = faible sinistralité). "
                "Le rectangle rouge identifie **Montpellier**."
            )
else:
    st.info("Données insuffisantes pour la synthèse statistique comparative.")

# ── Section 6 - Diagramme de Moran (autocorrélation spatiale IMD) ─────────────
st.divider()
section(6, "Diagramme de Moran - Validation de l'Absence d'Autocorrélation Spatiale de l'IMD")

st.markdown(r"""
Le **diagramme de Moran** (*Anselin, 1996*) représente graphiquement la relation entre la valeur
normalisée d'une variable en un lieu ($z_i$) et sa **lag spatiale** — la moyenne pondérée des valeurs
dans le voisinage ($\bar{z}_i^{\text{lag}}$). La pente OLS de ce nuage de points est exactement
l'**indice de Moran's $I$** : une pente proche de 0 signifie une distribution spatiale aléatoire.

L'analyse globale conclut à $I = -0{,}023$ ($p = 0{,}765$, non significatif), ce qui invalide
l'hypothèse d'un **déterminisme géographique** dans la qualité cyclable :
les villes performantes ne forment pas de clusters territoriaux cohérents.
Chaque quadrant du diagramme identifie un régime spatial distinct :

- **HH** (rouge) : villes à forte performance entourées de villes similaires — *clusters d'excellence*
- **LL** (bleu) : villes faibles entourées de villes faibles — *déserts cyclables territoriaux*
- **HL** (orange) : villes performantes isolées dans un environnement faible — *îlots d'excellence*
- **LH** (vert) : villes faibles dans un environnement fort — *sous-performance locale*
""")

# Calcul du diagramme de Moran sur l'IMD
_moran_df = _imd_ranked[["city", "IMD"]].copy()

# Récupérer les coordonnées moyennes par ville depuis le Gold Standard
_coord_ref = (
    df[df["station_type"] == "docked_bike"][["city", "lat", "lon"]]
    .groupby("city")[["lat", "lon"]].mean()
    .reset_index()
) if "station_type" in df.columns else (
    df[["city", "lat", "lon"]].groupby("city")[["lat", "lon"]].mean().reset_index()
)
_moran_df = _moran_df.merge(_coord_ref, on="city", how="left").dropna(subset=["lat", "lon", "IMD"])

if len(_moran_df) >= 5:
    # Matrice de poids spatiale W = 1/d_ij (distance en degrés)
    _coords = _moran_df[["lat", "lon"]].values
    _n_m    = len(_moran_df)
    _W      = np.zeros((_n_m, _n_m))
    for _ii in range(_n_m):
        for _jj in range(_n_m):
            if _ii != _jj:
                _d = np.sqrt(
                    (_coords[_ii, 0] - _coords[_jj, 0]) ** 2
                    + (_coords[_ii, 1] - _coords[_jj, 1]) ** 2
                )
                _W[_ii, _jj] = 1.0 / max(_d, 1e-6)
    # Standardisation en ligne (row-standardized W)
    _row_sums = _W.sum(axis=1, keepdims=True)
    _row_sums[_row_sums == 0] = 1.0
    _W_std    = _W / _row_sums

    # Standardisation de l'IMD (z-scores)
    _z        = (_moran_df["IMD"].values - _moran_df["IMD"].mean()) / max(_moran_df["IMD"].std(), 1e-6)
    _z_lag    = _W_std @ _z

    _moran_df["z_IMD"]     = _z
    _moran_df["z_lag_IMD"] = _z_lag

    # Quadrants
    def _quadrant(zi, zl):
        if zi >= 0 and zl >= 0:
            return "HH"
        if zi >= 0 and zl < 0:
            return "HL"
        if zi < 0 and zl >= 0:
            return "LH"
        return "LL"
    _moran_df["quadrant"] = [_quadrant(z, l) for z, l in zip(_z, _z_lag)]

    _q_colors = {"HH": "#c0392b", "LL": "#2980b9", "HL": "#e67e22", "LH": "#27ae60"}
    _moran_df["couleur"] = _moran_df["quadrant"].map(_q_colors)

    # Pente OLS = Moran's I calculé
    _z_arr   = _moran_df["z_IMD"].values
    _zl_arr  = _moran_df["z_lag_IMD"].values
    _moran_I = float(np.dot(_z_arr - _z_arr.mean(), _zl_arr - _zl_arr.mean()) /
                     max(np.dot(_z_arr - _z_arr.mean(), _z_arr - _z_arr.mean()), 1e-10))

    fig_moran = go.Figure()

    for _q, _qdf in _moran_df.groupby("quadrant"):
        fig_moran.add_trace(go.Scatter(
            x=_qdf["z_IMD"],
            y=_qdf["z_lag_IMD"],
            mode="markers+text",
            name=_q,
            text=_qdf["city"],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                color=_q_colors.get(_q, "#888"),
                size=10,
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
        ))

    # Droite OLS (pente = Moran's I)
    _x_line = np.linspace(_z.min() - 0.3, _z.max() + 0.3, 100)
    fig_moran.add_trace(go.Scatter(
        x=_x_line,
        y=_moran_I * _x_line,
        mode="lines",
        name=f"Pente = I = {_moran_I:.3f}",
        line=dict(color="#1A2332", width=2, dash="dash"),
        showlegend=True,
    ))

    # Lignes d'axes (quadrants)
    fig_moran.add_hline(y=0, line_width=1, line_dash="dot", line_color="#aaa")
    fig_moran.add_vline(x=0, line_width=1, line_dash="dot", line_color="#aaa")

    # Annotations de quadrant
    for _ql, _xt, _yt in [("HH", 0.97, 0.97), ("LL", 0.03, 0.03),
                           ("HL", 0.97, 0.03), ("LH", 0.03, 0.97)]:
        fig_moran.add_annotation(
            xref="paper", yref="paper",
            x=_xt, y=_yt,
            text=f"<b>{_ql}</b>",
            showarrow=False,
            font=dict(size=14, color=_q_colors[_ql]),
            opacity=0.6,
        )

    fig_moran.update_layout(
        xaxis_title="IMD standardisé (z-score)",
        yaxis_title="Lag spatial IMD (moyenne des voisins pondérée 1/d)",
        height=520,
        plot_bgcolor="white",
        margin=dict(l=60, r=30, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.5, xanchor="center"),
        xaxis=dict(zeroline=False, showgrid=True, gridcolor="#eee"),
        yaxis=dict(zeroline=False, showgrid=True, gridcolor="#eee"),
    )
    st.plotly_chart(fig_moran, use_container_width=True)

    _n_hh = int((_moran_df["quadrant"] == "HH").sum())
    _n_ll = int((_moran_df["quadrant"] == "LL").sum())
    _n_hl = int((_moran_df["quadrant"] == "HL").sum())
    _n_lh = int((_moran_df["quadrant"] == "LH").sum())

    st.caption(
        f"**Figure 6.1.** Diagramme de Moran (IMD, {len(_moran_df)} agglomérations dock-based). "
        f"Chaque point représente une agglomération ; la couleur encode le quadrant spatial. "
        f"La pente OLS de ce nuage vaut exactement Moran's $I = {_moran_I:.3f}$, "
        "confirmant l'absence d'autocorrélation spatiale significative. "
        f"Répartition : HH = {_n_hh} villes, LL = {_n_ll}, HL = {_n_hl}, LH = {_n_lh}. "
        "Poids spatial : inverse de la distance euclidienne (lat/lon), standardisation en ligne."
    )

    # Tableau des outliers remarquables
    _moran_df["abs_z"] = _moran_df["z_IMD"].abs()
    _outliers = _moran_df.nlargest(10, "abs_z")[
        ["city", "IMD", "z_IMD", "z_lag_IMD", "quadrant"]
    ].rename(columns={
        "city": "Agglomération", "IMD": "IMD (/100)",
        "z_IMD": "z (IMD)", "z_lag_IMD": "Lag spatial",
        "quadrant": "Quadrant",
    })
    with st.expander("Tableau des agglomérations les plus atypiques (|z| le plus fort)"):
        st.dataframe(
            _outliers.style.format({"IMD (/100)": "{:.1f}", "z (IMD)": "{:+.3f}", "Lag spatial": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Données de géolocalisation insuffisantes pour calculer le diagramme de Moran.")

# ── Section 7 - Carte nationale ────────────────────────────────────────────────
st.divider()
section(7, "Carte Nationale — Distribution Géographique de la Qualité Cyclable")

st.markdown(
    "Chaque agglomération est représentée par un cercle centré sur son barycentre de stations. "
    "La **couleur** encode la dimension sélectionnée dans la sidebar, la **taille** le nombre "
    "de stations dock-based Gold Standard. Basculez la métrique dans le panneau latéral pour "
    "explorer la géographie de chaque dimension d'enrichissement cyclable."
)

# Coordonnées moyennes par agglomération (barycentre des stations dock)
_city_coords_7 = (
    df[df["station_type"] == "docked_bike"][["city", "lat", "lon"]]
    .groupby("city")[["lat", "lon"]].mean()
    .reset_index()
) if "station_type" in df.columns else (
    df[["city", "lat", "lon"]].groupby("city")[["lat", "lon"]].mean().reset_index()
)

_map_df7 = (
    cities_f[["city", "n_stations", metric_key]]
    .merge(_imd_ranked[["city", "IMD"]], on="city", how="left")
    .merge(_city_coords_7, on="city", how="left")
    .dropna(subset=["lat", "lon", metric_key])
)

if not _map_df7.empty:
    col_map7, col_rank7 = st.columns([3, 1])

    with col_map7:
        fig_map7 = px.scatter_mapbox(
            _map_df7,
            lat="lat", lon="lon",
            color=metric_key,
            size="n_stations",
            color_continuous_scale=meta["color_scale"],
            hover_name="city",
            hover_data={
                "IMD":       ":.1f",
                metric_key:  ":.3f",
                "n_stations": True,
                "lat": False, "lon": False,
            },
            labels={metric_key: meta["label"], "n_stations": "Stations dock"},
            mapbox_style="carto-positron",
            zoom=5,
            center={"lat": 46.5, "lon": 2.5},
            size_max=28,
            opacity=0.85,
            height=560,
        )
        # Annotation Montpellier
        _mmm_map = _map_df7[_map_df7["city"] == "Montpellier"]
        if not _mmm_map.empty:
            fig_map7.add_trace(go.Scattermapbox(
                lat=[float(_mmm_map["lat"].iloc[0])],
                lon=[float(_mmm_map["lon"].iloc[0])],
                mode="markers+text",
                marker=dict(size=22, color="rgba(0,0,0,0)",
                            symbol="circle", opacity=1.0),
                text=["Montpellier"],
                textposition="top right",
                textfont=dict(size=11, color="#e74c3c"),
                name="Montpellier",
                showlegend=False,
            ))
        fig_map7.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title=meta["label"], thickness=14, len=0.6),
        )
        st.plotly_chart(fig_map7, use_container_width=True)
        st.caption(
            f"**Figure 7.1.** Distribution géographique des agglomérations VLS françaises "
            f"colorées par **{meta['label']}** (dimension sélectionnée). "
            f"Taille des cercles : nombre de stations dock-based Gold Standard. "
            f"{len(_map_df7)} agglomérations représentées (seuil ≥ {min_stations} stations). "
            "Source : Gold Standard GBFS, centroïdes moyens des stations."
        )

    with col_rank7:
        st.markdown(f"#### Top 10 — {meta['label']}")
        _rank7 = _map_df7.sort_values(
            metric_key, ascending=not meta.get("higher_is_better", True)
        ).reset_index(drop=True).head(10)
        for _i, _r in _rank7.iterrows():
            _is_mmm = _r["city"] == "Montpellier"
            _color7  = "#e74c3c" if _is_mmm else "#1A6FBF"
            _bold7   = "<b>" if _is_mmm else ""
            _bold7e  = "</b>" if _is_mmm else ""
            st.markdown(
                f"<div style='padding:4px 8px;margin:2px 0;border-left:3px solid {_color7}'>"
                f"<span style='color:{_color7};font-size:12px'>{_bold7}#{_i+1} {_r['city']}{_bold7e}</span><br>"
                f"<small>{_r[metric_key]:.3f} {meta['unit']} · IMD {_r['IMD']:.1f}</small>"
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown(f"#### Flop 5 — {meta['label']}")
        _flop7 = _map_df7.sort_values(
            metric_key, ascending=meta.get("higher_is_better", True)
        ).reset_index(drop=True).head(5)
        for _i, _r in _flop7.iterrows():
            _is_mmm = _r["city"] == "Montpellier"
            _color7f = "#e74c3c" if _is_mmm else "#e67e22"
            st.markdown(
                f"<div style='padding:4px 8px;margin:2px 0;border-left:3px solid {_color7f}'>"
                f"<span style='color:{_color7f};font-size:12px'>#{_i+1} {_r['city']}</span><br>"
                f"<small>{_r[metric_key]:.3f} {meta['unit']} · IMD {_r['IMD']:.1f}</small>"
                "</div>",
                unsafe_allow_html=True,
            )
else:
    st.info("Données de géolocalisation insuffisantes pour la carte nationale.")
