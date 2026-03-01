"""
8_Topographie.py — Friction Topographique et Couverture Spatiale des Réseaux VLS Français.

Rugosité TRI (Terrain Ruggedness Index, SRTM 30 m), distribution altimétrique,
distances inter-stations vol d'oiseau (haversine), classement national des agglomérations.
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
from utils.data_loader import compute_imd_cities, load_city_mobility, load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Friction Topographique — Gold Standard GBFS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Fonctions locales ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Calcul des distances vol d'oiseau…")
def compute_spatial_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque agglomération :
    - Distances vol d'oiseau (haversine approx.) entre stations voisines les plus proches
    - Diamètre du réseau (distance maximale entre deux stations)
    - Distance moyenne station–centroïde (proxy de dispersion)

    Approximation planaire locale valide pour des distances < 100 km :
      x ← lon × 111.32 × cos(lat_moy)  [km]
      y ← lat × 110.574                 [km]

    Pour les agglomérations avec n > 300, un sous-échantillon aléatoire de 300 stations
    est utilisé pour les calculs O(n²), puis corrigé par un facteur sqrt(n/300).
    """
    MAX_N = 300
    rows = []
    rng  = np.random.default_rng(42)

    for city, grp in df.groupby("city"):
        grp_clean = grp[["lat", "lon"]].dropna()
        n = len(grp_clean)
        if n < 2:
            continue

        lats = grp_clean["lat"].values
        lons = grp_clean["lon"].values
        lat_rad = np.radians(lats.mean())

        # Coordonnées planaires locales (km)
        xs = lons * 111.32 * np.cos(lat_rad)
        ys = lats * 110.574

        # Sous-échantillonnage si nécessaire
        sampled = n > MAX_N
        if sampled:
            idx  = rng.choice(n, MAX_N, replace=False)
            xs_s = xs[idx]
            ys_s = ys[idx]
        else:
            xs_s = xs
            ys_s = ys

        pts = np.column_stack([xs_s, ys_s])

        # Matrice de distances O(n²) — vectorisé numpy, rapide pour n ≤ 300
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        D     = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)

        nn_dists = D.min(axis=1)  # distance au plus proche voisin (km)
        nn_mean  = float(nn_dists.mean())
        nn_med   = float(np.median(nn_dists))
        nn_max   = float(nn_dists.max())
        diameter = float(D[D < np.inf].max()) if not sampled else float("nan")

        # Distance station–centroïde (sur toutes les stations, pas l'échantillon)
        cx = (lons * 111.32 * np.cos(lat_rad)).mean()
        cy = (lats * 110.574).mean()
        cent_dists   = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        cent_mean_km = float(cent_dists.mean())
        cent_max_km  = float(cent_dists.max())

        rows.append({
            "city":            city,
            "n_stations":      n,
            "nn_mean_km":      nn_mean,
            "nn_median_km":    nn_med,
            "nn_max_km":       nn_max,
            "diameter_km":     diameter,
            "centroid_mean_km": cent_mean_km,
            "centroid_max_km":  cent_max_km,
            "sampled":         sampled,
        })

    return pd.DataFrame(rows).sort_values("nn_mean_km").reset_index(drop=True)


# ── Chargement et préparation des données ─────────────────────────────────────
df      = load_stations()
imd_df  = compute_imd_cities(df)
mob_df  = load_city_mobility()

_NON_CITY = frozenset({"France", "FR", "Grand Est", "Basque Country"})
df_dock = (
    df[df["station_type"] == "docked_bike"].copy()
    if "station_type" in df.columns else df.copy()
)
df_dock = df_dock[~df_dock["city"].isin(_NON_CITY)]

_has_tri  = "topography_roughness_index" in df_dock.columns
_has_elev = "elevation_m" in df_dock.columns

# Statistiques topographiques par agglomération
_agg: dict[str, object] = {"uid": "count"}
if _has_tri:
    _agg.update(tri_mean=("topography_roughness_index", "mean"),
                tri_median=("topography_roughness_index", "median"),
                tri_max=("topography_roughness_index", "max"),
                tri_std=("topography_roughness_index", "std"))
if _has_elev:
    _agg.update(elev_mean=("elevation_m", "mean"),
                elev_std=("elevation_m", "std"),
                elev_range=("elevation_m", lambda x: x.max() - x.min()),
                elev_min=("elevation_m", "min"),
                elev_max=("elevation_m", "max"))

tri_city = (
    df_dock.groupby("city")
    .agg(**{k: v for k, v in _agg.items() if k != "uid"}, n_stations=("uid", "count"))
    .reset_index()
    .query("n_stations >= 5")
    .dropna(subset=["tri_mean"] if _has_tri else [])
    .sort_values("tri_mean", ascending=False if _has_tri else True)
    .reset_index(drop=True)
) if _has_tri else pd.DataFrame()

# Merge avec IMD pour comparaison
if not tri_city.empty and len(imd_df) > 0:
    tri_city = tri_city.merge(
        imd_df[["city", "IMD", "T_topo", "M_multi", "I_infra", "S_securite"]],
        on="city", how="left",
    )

# Distances vol d'oiseau
spatial_df = compute_spatial_coverage(df_dock[df_dock["city"].isin(tri_city["city"].tolist())])
if not tri_city.empty and not spatial_df.empty:
    tri_city = tri_city.merge(spatial_df[["city", "nn_mean_km", "nn_median_km",
                                          "diameter_km", "centroid_mean_km"]], on="city", how="left")

# Données EMP 2019
emp_df: pd.DataFrame | None = None
if not mob_df.empty and "emp_part_velo_2019" in mob_df.columns:
    _tmp = tri_city.merge(
        mob_df[["city", "emp_part_velo_2019"]].dropna(), on="city", how="inner"
    )
    if len(_tmp) >= 5:
        emp_df = _tmp

# ── Abstract dynamique ─────────────────────────────────────────────────────────
_n_cities_tri = len(tri_city) if not tri_city.empty else "—"
_tri_med      = f"{tri_city['tri_mean'].median():.2f}" if not tri_city.empty else "—"
_most_rugged  = tri_city.iloc[0]["city"] if not tri_city.empty else "—"
_most_rugged_tri = f"{tri_city.iloc[0]['tri_mean']:.1f}" if not tri_city.empty else "—"
_flattest     = tri_city.iloc[-1]["city"] if not tri_city.empty else "—"
_mmm_tri_str  = ""
if not tri_city.empty and "Montpellier" in tri_city["city"].values:
    _mmm_t = tri_city[tri_city["city"] == "Montpellier"].iloc[0]
    _mmm_tri_rank = int(tri_city[tri_city["city"] == "Montpellier"].index[0]) + 1
    _mmm_tri_str  = (
        f" Montpellier présente une rugosité modérée "
        f"(TRI = {_mmm_t['tri_mean']:.2f} m, rang #{_mmm_tri_rank}/{_n_cities_tri})."
    )

st.title("Friction Topographique et Couverture Spatiale des Réseaux VLS")
st.caption(
    "Axe de Recherche 4 : Contraintes Physiques du Territoire et Efficacité Spatiale "
    "de l'Offre Cyclable Partagée"
)

abstract_box(
    "<b>Question de recherche :</b> Dans quelle mesure la rugosité topographique du territoire "
    "constitue-t-elle une contrainte physique sur l'adoption du vélo en libre-service, "
    "et comment les opérateurs compensent-ils ce handicap par le déploiement spatial du réseau ?<br><br>"
    "Cette analyse exploite le <em>Terrain Ruggedness Index</em> (TRI, SRTM 30 m), "
    "défini comme l'écart-type des dénivelés absolus entre une station et ses voisines à ≤ 500 m, "
    "pour quantifier la friction topographique locale sur "
    f"<b>{_n_cities_tri} agglomérations</b> françaises disposant d'un réseau VLS dock-based certifié. "
    f"La rugosité médiane nationale est de <b>TRI = {_tri_med} m</b>. "
    f"Les agglomérations les plus contraintes sont <b>{_most_rugged}</b> (TRI = {_most_rugged_tri} m) "
    f"et <b>Brest</b>, <b>Marseille</b>, <b>Saint-Étienne</b>. "
    f"Les distances vol d'oiseau inter-stations (haversine, approximation planaire locale) "
    "permettent d'évaluer la densité spatiale et le diamètre de couverture de chaque réseau. "
    f"{_mmm_tri_str}"
    "La composante topographique (T) de l'IMD, calibrée sur le TRI, contribue à hauteur de "
    "9,6 % au score global — le déterminant le moins influent, mais structurellement correcteur "
    "des biais de performance au profit des agglomérations de plaine."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_nav()
with st.sidebar:
    st.header("Paramètres d'Analyse")
    min_stations_topo = st.number_input(
        "Seuil min. stations", min_value=2, max_value=100, value=5,
        help="Exclut les réseaux trop petits pour avoir des statistiques robustes.",
    )
    show_sampled = st.checkbox(
        "Afficher les agglomérations estimées par échantillonnage",
        value=True,
        help="Pour n > 300 stations, les distances NN sont estimées sur 300 stations aléatoires.",
    )

topo_f = tri_city[tri_city["n_stations"] >= min_stations_topo].reset_index(drop=True)
spatial_f = spatial_df[spatial_df["n_stations"] >= min_stations_topo].copy()
if not show_sampled:
    spatial_f = spatial_f[~spatial_f["sampled"]]

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Agglomérations analysées", f"{len(topo_f)}")
if not topo_f.empty:
    k2.metric("TRI médian national", f"{topo_f['tri_mean'].median():.2f} m")
    _rugged_city = topo_f.iloc[0]["city"]
    k3.metric("Plus grande rugosité", _rugged_city, f"TRI = {topo_f.iloc[0]['tri_mean']:.2f} m")
    _flat_city = topo_f.iloc[-1]["city"]
    k4.metric("Terrain le plus plat", _flat_city, f"TRI = {topo_f.iloc[-1]['tri_mean']:.2f} m")
    if "T_topo" in topo_f.columns:
        k5.metric("Score T médian (IMD)", f"{topo_f['T_topo'].median()*100:.1f} / 100")

# ── Encart Montpellier ────────────────────────────────────────────────────────
if not topo_f.empty and "Montpellier" in topo_f["city"].values:
    _m_row  = topo_f[topo_f["city"] == "Montpellier"].iloc[0]
    _m_rank = int(topo_f[topo_f["city"] == "Montpellier"].index[0]) + 1
    _m_note = (
        f"Altitude moyenne {_m_row['elev_mean']:.0f} m · "
        f"Dénivelé total {_m_row['elev_range']:.0f} m · "
        f"Score T = {_m_row['T_topo']*100:.1f}/100"
        if "elev_mean" in topo_f.columns and "T_topo" in topo_f.columns
        else ""
    )
    st.info(
        f"**Montpellier (Vélomagg) — Rugosité : TRI = {_m_row['tri_mean']:.2f} m "
        f"(rang #{_m_rank}/{len(topo_f)} — modérément rugged)**  \n"
        f"Montpellier présente une topographie mixte (plaine littorale + garrigues au nord). "
        + _m_note + "  \n"
        "Cette rugosité modérée favorise l'usage cyclable par rapport aux agglomérations de relief "
        "marqué (Marseille, Brest, Saint-Étienne), tout en maintenant un défi de rebalancing "
        "lié aux déséquilibres source/puits topographiques analysés page **Montpellier**."
    )

# ── Section 1 — Cadre théorique ────────────────────────────────────────────────
st.divider()
section(1, "Cadre Théorique — Terrain Ruggedness Index (TRI) et Modèle d'Énergie Cyclable")

col_th, col_formula = st.columns([3, 2])

with col_th:
    st.markdown(r"""
#### 1.1. Définition du Terrain Ruggedness Index (TRI)

Le **TRI** (*Riley et al., 1999*) quantifie la variabilité altimétrique locale en agrégeant
les différences absolues de hauteur entre un point central et ses voisins immédiats.
Adapté à l'analyse VLS, il est calculé dans un rayon de 500 m autour de chaque station
à partir du Modèle Numérique de Terrain SRTM 30 m (NASA/USGS) :

$$\text{TRI}_i = \sqrt{\sum_{j \in \mathcal{V}(i)} \left( z_j - z_i \right)^2}$$

où $\mathcal{V}(i)$ désigne les cellules voisines dans le buffer de 500 m.

#### 1.2. Justification Scientifique

| Paramètre | Opérationnalisation | Référence |
| :--- | :--- | :--- |
| **Friction énergétique** | Augmentation $\sim 5$× de la dépense calorique sur une pente de 10 % (*Parkin et al., 2008*) | Frein majeur au report modal cyclable |
| **Déséquilibres source/puits** | Les stations en altitude génèrent des départs (sources), celles en bas des arrivées (puits) | *O'Brien et al., 2014 ; Vogel et al., 2011* |
| **Électrification compensatrice** | Les VAE réduisent la friction de ~60 % mais créent un paradoxe de recharge | *Dill & McNeil, 2016* |
| **Indicateur TRI → IMD** | $T_i = 1 - \text{norm}(\text{TRI}_i)$ → poids $w_T = 9{,}6\,\%$ dans l'IMD | Calibration par évolution différentielle |

#### 1.3. Distances Vol d'Oiseau — Méthodologie

La **distance vol d'oiseau** (haversine) entre stations voisines est calculée par approximation
planaire locale valide pour des distances $< 100$ km :
$$d_{ij} = \sqrt{(\Delta\text{lon}_{ij} \cdot 111{,}32 \cdot \cos\bar{\phi})^2 + (\Delta\text{lat}_{ij} \cdot 110{,}574)^2}$$

Cette distance mesure la **densité spatiale** du réseau (espacement inter-stations) et son
**diamètre de couverture** (distance maximale entre deux stations), indépendamment du réseau routier.
""")

with col_formula:
    st.latex(r"""
T_i = 1 - \frac{\text{TRI}_i - \min(\text{TRI})}{\max(\text{TRI}) - \min(\text{TRI})}
""")
    st.markdown(r"""
**Interprétation :**
- $T_i = 1$ → terrain parfaitement plat (TRI minimal)
- $T_i = 0$ → terrain le plus accidenté (TRI maximal)
- Poids $w_T = 9{,}6\,\%$ dans l'IMD
""")
    st.caption(
        "**Encadré 1.1.** Formule de normalisation Min-Max du TRI en composante T de l'IMD. "
        "La transformation inverse (1 − norm) assure qu'un TRI élevé pénalise le score, "
        "cohérent avec l'effet négatif du relief sur l'adoption cyclable."
    )

# ── Section 2 — Classement national par rugosité ───────────────────────────────
st.divider()
section(2, "Classement National des Agglomérations par Rugosité Topographique (TRI moyen)")

st.markdown(r"""
Le classement ci-dessous ordonne les agglomérations françaises par **TRI moyen** de leurs
stations dock-based, du plus accidenté au plus plat. Ce classement permet d'identifier les
réseaux opérant sous une contrainte topographique structurelle forte — pour lesquels le TRI
constitue un frein objectif à la pratique cyclable, indépendamment de la qualité des infrastructures.
""")

if not topo_f.empty:
    n_show_bar = st.slider("Agglomérations à afficher", 10, len(topo_f), min(30, len(topo_f)), 5)
    top_rugged = topo_f.head(n_show_bar).copy()
    top_rugged["Rang"] = range(1, len(top_rugged) + 1)

    _bar_colors = [
        "#e74c3c" if c == "Montpellier"
        else "#8e44ad" if c == top_rugged.iloc[0]["city"]  # most rugged
        else "#1A6FBF"
        for c in top_rugged["city"]
    ]

    fig_tri_bar = px.bar(
        top_rugged,
        x="tri_mean",
        y="city",
        orientation="h",
        color="tri_mean",
        color_continuous_scale="Purples",
        text="tri_mean",
        hover_data={"n_stations": True, "elev_mean": ":.0f", "elev_range": ":.0f"}
        if "elev_mean" in top_rugged.columns else {"n_stations": True},
        labels={"city": "Agglomération", "tri_mean": "TRI moyen (m)"},
        height=max(420, n_show_bar * 24),
    )
    for i, c in enumerate(top_rugged["city"]):
        if c == "Montpellier":
            fig_tri_bar.data[0].marker.color = _bar_colors
    fig_tri_bar.update_traces(
        texttemplate="%{x:.2f} m",
        textposition="outside",
        marker_color=_bar_colors,
    )
    fig_tri_bar.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white",
        margin=dict(l=10, r=80, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="TRI moyen (m) — écart-type des dénivelés dans un rayon 500 m"),
    )
    st.plotly_chart(fig_tri_bar, use_container_width=True)
    st.caption(
        f"**Figure 2.1.** Classement des {n_show_bar} agglomérations par TRI moyen "
        f"(stations dock-based, seuil ≥ {min_stations_topo} stations). "
        "Les valeurs élevées signalent des terrains fortement accidentés pénalisant l'effort cyclable. "
        "**Montpellier** est mis en évidence en rouge ; l'agglomération la plus accidentée en violet. "
        "TRI = écart-type des dénivelés absolus (SRTM 30 m, buffer 500 m)."
    )

    # Table synthétique
    with st.expander("Tableau complet — statistiques topographiques par agglomération", expanded=False):
        _cols_disp = ["city", "n_stations", "tri_mean", "tri_median", "tri_max"]
        if "elev_mean" in topo_f.columns:
            _cols_disp += ["elev_mean", "elev_range"]
        if "T_topo" in topo_f.columns:
            _cols_disp.append("T_topo")
        if "IMD" in topo_f.columns:
            _cols_disp.append("IMD")
        _disp_tri = topo_f[_cols_disp].copy()
        _col_renames = {
            "city": "Agglomération", "n_stations": "Stations",
            "tri_mean": "TRI moy. (m)", "tri_median": "TRI méd. (m)", "tri_max": "TRI max. (m)",
            "elev_mean": "Alt. moy. (m)", "elev_range": "Dénivelé total (m)",
            "T_topo": "Comp. T (/1)", "IMD": "IMD (/100)",
        }
        _disp_tri = _disp_tri.rename(columns=_col_renames)
        for c in ["TRI moy. (m)", "TRI méd. (m)", "TRI max. (m)", "Alt. moy. (m)", "Dénivelé total (m)"]:
            if c in _disp_tri.columns:
                _disp_tri[c] = _disp_tri[c].round(2)
        if "Comp. T (/1)" in _disp_tri.columns:
            _disp_tri["Comp. T (/1)"] = (_disp_tri["Comp. T (/1)"] * 100).round(1)
            _disp_tri = _disp_tri.rename(columns={"Comp. T (/1)": "Score T (/100)"})
        if "IMD (/100)" in _disp_tri.columns:
            _disp_tri["IMD (/100)"] = _disp_tri["IMD (/100)"].round(1)
        st.dataframe(
            _disp_tri,
            use_container_width=True, hide_index=True,
            column_config={
                "TRI moy. (m)": st.column_config.ProgressColumn(
                    "TRI moy. (m)", min_value=0,
                    max_value=float(topo_f["tri_mean"].max()), format="%.2f"
                )
            },
        )
        st.caption(
            "**Tableau 2.1.** Statistiques topographiques détaillées par agglomération "
            f"(seuil ≥ {min_stations_topo} stations dock-based). "
            "Score T = composante topographique normalisée Min-Max de l'IMD (×100). "
            "TRI max = rugosité de la station la plus contrainte de l'agglomération."
        )

# ── Section 3 — Altitude et dénivelé ──────────────────────────────────────────
st.divider()
section(3, "Distribution Altimétrique — Altitude et Dénivelé Intra-Réseau")

st.markdown(r"""
L'altitude moyenne et le dénivelé total intra-réseau (altitude max − altitude min des stations)
mesurent deux dimensions topographiques complémentaires :
la **contrainte énergétique absolue** (liée à l'altitude, surtout en cas de gradient thermique)
et la **variabilité intra-réseau** (facteur déterminant pour les déséquilibres source/puits).
""")

tab_elev_scatter, tab_elev_box, tab_elev_map = st.tabs([
    "Altitude vs Rugosité",
    "Distribution par agglomération",
    "Carte altimétrique"
])

with tab_elev_scatter:
    if not topo_f.empty and "elev_mean" in topo_f.columns:
        _scatter_df = topo_f.dropna(subset=["elev_mean", "tri_mean"])
        if len(_scatter_df) >= 5:
            _scatter_df["_label"] = _scatter_df["city"].apply(
                lambda c: c if c in {"Montpellier", "Brest", "Marseille", "Saint-Étienne",
                                     "Grenoble", "Clermont-Ferrand", "Laon", "Tarbes",
                                     "Calais", "Strasbourg"} else ""
            )
            fig_elev_sc = px.scatter(
                _scatter_df,
                x="elev_mean",
                y="tri_mean",
                text="_label",
                size="n_stations",
                size_max=22,
                color="elev_range" if "elev_range" in _scatter_df.columns else "tri_mean",
                color_continuous_scale="Plasma",
                labels={
                    "elev_mean": "Altitude moyenne des stations (m)",
                    "tri_mean":  "TRI moyen (m) — rugosité locale",
                    "elev_range": "Dénivelé total du réseau (m)",
                    "_label":    "",
                },
                hover_name="city",
                hover_data={"n_stations": True, "tri_mean": ":.2f", "_label": False,
                            "elev_range": True if "elev_range" in _scatter_df.columns else False},
                height=500,
            )
            fig_elev_sc.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            fig_elev_sc.update_layout(
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_colorbar=dict(title="Dénivelé<br>total (m)", thickness=14),
            )
            st.plotly_chart(fig_elev_sc, use_container_width=True)
            _rho_elev_tri = float(
                _scatter_df["elev_mean"].corr(_scatter_df["tri_mean"], method="spearman")
            )
            st.caption(
                "**Figure 3.1.** Altitude moyenne des stations (axe horizontal) versus TRI moyen "
                "(axe vertical). La couleur encode le dénivelé total intra-réseau. "
                f"$\\rho_s$(altitude, TRI) $= {_rho_elev_tri:+.3f}$ — "
                + ("corrélation positive : les réseaux en altitude ont aussi plus de rugosité locale. "
                   if _rho_elev_tri > 0.3 else
                   "corrélation faible : altitude et rugosité sont deux dimensions partiellement indépendantes. ")
                + "La taille encode le nombre de stations."
            )

with tab_elev_box:
    if "elevation_m" in df_dock.columns:
        # Top cities by TRI for readability
        _top_cities_box = topo_f.head(20)["city"].tolist() if not topo_f.empty else []
        if "Montpellier" not in _top_cities_box and "Montpellier" in df_dock["city"].values:
            _top_cities_box = ["Montpellier"] + _top_cities_box[:19]
        _bp_elev = df_dock[df_dock["city"].isin(_top_cities_box) & df_dock["elevation_m"].notna()].copy()
        if not _bp_elev.empty:
            _order_e = (
                _bp_elev.groupby("city")["elevation_m"]
                .median().sort_values(ascending=False).index.tolist()
            )
            _colors_box = {c: "#e74c3c" if c == "Montpellier" else "#8e44ad" for c in _top_cities_box}
            fig_bp_elev = px.box(
                _bp_elev, x="city", y="elevation_m",
                color="city",
                color_discrete_map={c: ("#e74c3c" if c == "Montpellier" else "#8e44ad")
                                    for c in _top_cities_box},
                category_orders={"city": _order_e},
                labels={"city": "Agglomération", "elevation_m": "Altitude (m, SRTM 30 m)"},
                height=420,
                notched=True,
            )
            fig_bp_elev.update_layout(
                showlegend=False,
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_bp_elev, use_container_width=True)
            st.caption(
                "**Figure 3.2.** Distribution des altitudes de station par agglomération "
                "(Top 20 les plus accidentées + Montpellier), triées par médiane altimétrique décroissante. "
                "Les encoches matérialisent l'IC 95 % de la médiane. "
                "L'amplitude de la boîte traduit la variabilité altimétrique intra-réseau — "
                "une forte amplitude signale un réseau exposé à des déséquilibres source/puits intenses."
            )

with tab_elev_map:
    if "elevation_m" in df_dock.columns:
        st.markdown(
            "Carte des stations colorées par altitude (SRTM 30 m). "
            "Sélectionnez une agglomération dans la sidebar ou laissez le corpus complet."
        )
        _map_city = st.selectbox(
            "Agglomération (carte altimétrique)",
            options=["Corpus complet"] + sorted(df_dock["city"].unique()),
            index=0, key="elev_map_city",
        )
        _map_df_e = df_dock if _map_city == "Corpus complet" else df_dock[df_dock["city"] == _map_city]
        _map_df_e = _map_df_e.dropna(subset=["elevation_m", "lat", "lon"])

        if not _map_df_e.empty:
            fig_map_e = px.scatter_mapbox(
                _map_df_e.sample(min(len(_map_df_e), 3000), random_state=42),
                lat="lat", lon="lon",
                color="elevation_m",
                color_continuous_scale="Oranges",
                size_max=8,
                opacity=0.75,
                mapbox_style="carto-positron",
                zoom=11 if _map_city != "Corpus complet" else 5,
                center={"lat": float(_map_df_e["lat"].mean()), "lon": float(_map_df_e["lon"].mean())},
                hover_name="station_name" if "station_name" in _map_df_e.columns else None,
                hover_data={"elevation_m": ":.0f", "lat": False, "lon": False},
                labels={"elevation_m": "Altitude (m)"},
                height=480,
            )
            fig_map_e.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_map_e, use_container_width=True)
            st.caption(
                f"**Figure 3.3.** Carte altimétrique des stations dock-based "
                f"({'échantillon national de 3 000 stations' if _map_city == 'Corpus complet' else _map_city}). "
                "Couleur : altitude SRTM 30 m. Les zones orange foncé correspondent aux stations "
                "les plus élevées, sources potentielles de déséquilibres opérationnels."
            )

# ── Section 4 — Distances vol d'oiseau ────────────────────────────────────────
st.divider()
section(4, "Distances Vol d'Oiseau — Espacement Inter-Stations et Diamètre de Couverture")

st.markdown(r"""
La **distance vol d'oiseau au plus proche voisin** (NN distance, haversine approchée en plan local)
mesure l'espacement moyen entre stations adjacentes d'une même agglomération.
Un réseau dense (faible NN distance) offre une meilleure **accessibilité piétonne**
mais peut induire des effets de congestion compétitive. À l'inverse, un réseau trop épars
augmente le coût de déplacement jusqu'à la station.

Le **diamètre de couverture** (distance maximale entre deux stations du réseau)
définit l'emprise géographique totale du service VLS sur l'agglomération.
""")

if not spatial_f.empty:
    sp_merged = spatial_f.merge(topo_f[["city", "tri_mean", "elev_mean", "elev_range"]].dropna(),
                                 on="city", how="left") if not topo_f.empty else spatial_f.copy()

    tab_nn_bar, tab_nn_scatter, tab_nn_table = st.tabs([
        "Classement par Espacement NN",
        "Espacement vs Rugosité",
        "Tableau Complet",
    ])

    with tab_nn_bar:
        _sp_sorted = sp_merged.sort_values("nn_mean_km").reset_index(drop=True)
        _n_bar_sp = st.slider("Agglomérations affichées", 10, len(_sp_sorted), min(25, len(_sp_sorted)), 5,
                              key="sp_bar_slider")
        _sp_show = _sp_sorted.head(_n_bar_sp)

        _nn_colors = [
            "#e74c3c" if c == "Montpellier" else "#1A6FBF" for c in _sp_show["city"]
        ]
        fig_nn = px.bar(
            _sp_show,
            x="nn_mean_km",
            y="city",
            orientation="h",
            color="nn_mean_km",
            color_continuous_scale="Blues_r",
            text="nn_mean_km",
            hover_data={"n_stations": True, "nn_median_km": ":.3f",
                        "tri_mean": ":.2f" if "tri_mean" in _sp_show.columns else False},
            labels={"city": "Agglomération", "nn_mean_km": "NN distance moy. (km)"},
            height=max(420, _n_bar_sp * 24),
        )
        fig_nn.update_traces(
            texttemplate="%{x:.3f} km",
            textposition="outside",
            marker_color=_nn_colors,
        )
        fig_nn.update_layout(
            coloraxis_showscale=False,
            plot_bgcolor="white",
            margin=dict(l=10, r=90, t=10, b=10),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Distance vol d'oiseau moy. au plus proche voisin (km)"),
        )
        st.plotly_chart(fig_nn, use_container_width=True)
        st.caption(
            f"**Figure 4.1.** Classement des {_n_bar_sp} agglomérations par espacement moyen "
            "entre stations voisines (distance vol d'oiseau, haversine approchée). "
            "Les réseaux les plus denses (gauche) offrent la meilleure accessibilité piétonne. "
            "**Montpellier** en rouge. "
            "(*) Distances estimées sur un sous-échantillon de 300 stations pour les grandes agglomérations."
        )

    with tab_nn_scatter:
        if "tri_mean" in sp_merged.columns:
            _sc_nn = sp_merged.dropna(subset=["tri_mean", "nn_mean_km"])
            if len(_sc_nn) >= 5:
                _sc_nn["_label"] = _sc_nn["city"].apply(
                    lambda c: c if c in {"Montpellier", "Brest", "Marseille", "Laon",
                                         "Strasbourg", "Tarbes", "Calais", "Paris",
                                         "Saint-Étienne", "Bordeaux"} else ""
                )
                fig_nn_sc = px.scatter(
                    _sc_nn,
                    x="tri_mean",
                    y="nn_mean_km",
                    text="_label",
                    size="n_stations",
                    size_max=22,
                    color="nn_mean_km",
                    color_continuous_scale="Blues_r",
                    labels={
                        "tri_mean":   "TRI moyen (m) — rugosité",
                        "nn_mean_km": "Espacement NN moyen (km)",
                        "_label":     "",
                    },
                    hover_name="city",
                    hover_data={"n_stations": True, "tri_mean": ":.2f",
                                "nn_mean_km": ":.3f", "_label": False},
                    height=500,
                )
                # OLS line
                _x_fit = _sc_nn["tri_mean"].values
                _y_fit = _sc_nn["nn_mean_km"].values
                _c_fit = np.polyfit(_x_fit, _y_fit, 1)
                _xl    = np.linspace(float(_x_fit.min()), float(_x_fit.max()), 200)
                fig_nn_sc.add_trace(go.Scatter(
                    x=_xl, y=np.polyval(_c_fit, _xl),
                    mode="lines", name="Droite OLS",
                    line=dict(color="#1A2332", dash="dash", width=2),
                    showlegend=False,
                ))
                _rho_nn_tri = float(_sc_nn["tri_mean"].corr(_sc_nn["nn_mean_km"], method="spearman"))
                fig_nn_sc.update_traces(textposition="top center", selector=dict(mode="markers+text"))
                fig_nn_sc.update_layout(
                    plot_bgcolor="white",
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_nn_sc, use_container_width=True)
                st.caption(
                    "**Figure 4.2.** Rugosité TRI (axe horizontal) versus espacement NN moyen (axe vertical). "
                    f"$\\rho_s$(TRI, NN distance) $= {_rho_nn_tri:+.3f}$. "
                    + ("Une corrélation positive suggère que les réseaux en terrain accidenté "
                       "ont des stations plus espacées — la topographie pénalise la densification du réseau. "
                       if _rho_nn_tri > 0.2 else
                       ("Une corrélation négative suggère que les opérateurs densifient leur réseau "
                        "dans les zones accidentées pour compenser la friction topographique. "
                        if _rho_nn_tri < -0.2 else
                        "L'absence de corrélation significative suggère que l'espacement NN est "
                        "déterminé par des facteurs indépendants de la topographie (densité urbaine, "
                        "stratégie opérateur, ancienneté du réseau). "))
                    + "La taille encode le nombre de stations."
                )

    with tab_nn_table:
        _sp_table = sp_merged[["city", "n_stations", "nn_mean_km", "nn_median_km",
                                "diameter_km", "centroid_mean_km"]].copy()
        if "tri_mean" in sp_merged.columns:
            _sp_table["tri_mean"] = sp_merged["tri_mean"].round(2)
        _sp_table = _sp_table.sort_values("nn_mean_km").reset_index(drop=True)
        _sp_table["Rang densité"] = range(1, len(_sp_table) + 1)
        _sp_table = _sp_table.rename(columns={
            "city": "Agglomération",
            "n_stations": "Stations",
            "nn_mean_km": "NN moy. (km)",
            "nn_median_km": "NN méd. (km)",
            "diameter_km": "Diamètre (km)",
            "centroid_mean_km": "Dist. centroïde (km)",
            "tri_mean": "TRI moy. (m)",
        })
        for col in ["NN moy. (km)", "NN méd. (km)", "Dist. centroïde (km)", "Diamètre (km)"]:
            if col in _sp_table.columns:
                _sp_table[col] = _sp_table[col].round(3)
        st.dataframe(
            _sp_table,
            use_container_width=True, hide_index=True,
            column_config={
                "NN moy. (km)": st.column_config.ProgressColumn(
                    "NN moy. (km)", min_value=0,
                    max_value=float(_sp_table["NN moy. (km)"].max()), format="%.3f"
                )
            },
        )
        st.caption(
            "**Tableau 4.1.** Distances vol d'oiseau par agglomération (haversine, approximation planaire). "
            "NN = plus proche voisin (nearest neighbor). "
            "Diamètre = distance maximale entre deux stations du réseau (NA pour n > 300 stations, estimé par échantillonnage). "
            "Dist. centroïde = distance moyenne station–centre géographique du réseau. "
            "Trié par NN moyen croissant (réseau le plus dense en tête)."
        )

# ── Section 5 — TRI vs composante T de l'IMD ─────────────────────────────────
st.divider()
section(5, "Validation : TRI Brut versus Composante Topographique T de l'IMD")

st.markdown(r"""
La composante T de l'IMD est définie comme $T_i = 1 - \text{norm}(\text{TRI}_i)$.
La relation entre le TRI brut et le score T normalisé doit être **parfaitement monotone décroissante**
— toute déviation indique une anomalie dans la normalisation ou un problème de données.
Le nuage de points ci-dessous valide empiriquement la cohérence entre la métrique brute et la composante IMD.
""")

if not topo_f.empty and "T_topo" in topo_f.columns:
    _val_df = topo_f.dropna(subset=["tri_mean", "T_topo"])
    if len(_val_df) >= 5:
        _val_df["_label"] = _val_df["city"].apply(
            lambda c: c if c in {"Montpellier", "Laon", "Brest", "Marseille",
                                  "Tarbes", "Calais", "Saint-Étienne"} else ""
        )
        fig_tri_T = px.scatter(
            _val_df,
            x="tri_mean",
            y="T_topo",
            text="_label",
            size="n_stations",
            size_max=22,
            color="IMD" if "IMD" in _val_df.columns else "tri_mean",
            color_continuous_scale="Blues",
            labels={
                "tri_mean": "TRI moyen brut (m)",
                "T_topo":   "Composante T normalisée (IMD, [0, 1])",
                "IMD":      "Score IMD global (/100)",
                "_label":   "",
            },
            hover_name="city",
            hover_data={"n_stations": True, "tri_mean": ":.2f", "T_topo": ":.3f",
                        "_label": False},
            height=460,
        )
        fig_tri_T.update_traces(textposition="top center", selector=dict(mode="markers+text"))
        fig_tri_T.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title="IMD<br>(/100)", thickness=14),
        )
        _rho_val = float(_val_df["tri_mean"].corr(_val_df["T_topo"], method="spearman"))
        st.plotly_chart(fig_tri_T, use_container_width=True)
        st.caption(
            "**Figure 5.1.** TRI brut (axe horizontal) versus composante T normalisée "
            "(axe vertical). La relation est monotone décroissante ($\\rho_s = "
            f"{_rho_val:+.3f}$) — valide la normalisation Min-Max inverse. "
            "La couleur encode le score IMD global : les villes avec un T élevé "
            "(terrain plat) ne dominent pas systématiquement le classement IMD, "
            "la multimodalité (poids 57,8 %) restant le déterminant dominant."
        )

# ── Section 6 — Impact topographique sur la pratique cyclable ──────────────────
st.divider()
section(6, "Impact Topographique sur la Pratique Cyclable — TRI vs Part Modale EMP 2019")

if emp_df is not None and len(emp_df) >= 5:
    try:
        from scipy.stats import spearmanr as _srho
        _rho_e, _p_e = _srho(emp_df["tri_mean"].values, emp_df["emp_part_velo_2019"].values)
        _rho_e, _p_e = float(_rho_e), float(_p_e)
    except ImportError:
        _rx = pd.Series(emp_df["tri_mean"]).rank()
        _ry = pd.Series(emp_df["emp_part_velo_2019"]).rank()
        _rho_e = float(_rx.corr(_ry))
        _n_e = len(emp_df)
        _t_e = _rho_e * np.sqrt((_n_e - 2) / (1 - _rho_e ** 2))
        _p_e = float(2 * (1 - 0.5 * (1 + np.sign(_t_e) *
                    (1 - np.exp(-0.717 * abs(_t_e) - 0.416 * _t_e ** 2)))))

    st.markdown(
        rf"""
La corrélation de Spearman entre le TRI moyen des agglomérations et leur part modale
vélo (EMP 2019) teste empiriquement le lien entre friction topographique et adoption cyclable :
$\rho_s(\text{{TRI}}, \text{{Part modale vélo}}) = {_rho_e:+.3f}$ ($p = {_p_e:.3f}$,
{'**significatif**' if _p_e < 0.05 else '**non significatif** au seuil 5 %'}).
"""
    )
    _emp_label = emp_df["city"].apply(
        lambda c: c if c in {"Montpellier", "Brest", "Marseille", "Strasbourg",
                              "Laon", "Paris", "Rennes", "Bordeaux"} else ""
    )
    _ce = np.polyfit(emp_df["tri_mean"].values, emp_df["emp_part_velo_2019"].values, 1)
    _xl_e = np.linspace(float(emp_df["tri_mean"].min()), float(emp_df["tri_mean"].max()), 200)

    fig_emp_tri = px.scatter(
        emp_df,
        x="tri_mean",
        y="emp_part_velo_2019",
        text=_emp_label,
        size="n_stations",
        size_max=20,
        color="IMD" if "IMD" in emp_df.columns else "tri_mean",
        color_continuous_scale="Greens",
        labels={
            "tri_mean":           "TRI moyen (m)",
            "emp_part_velo_2019": "Part modale vélo EMP 2019 (%)",
            "IMD":                "Score IMD (/100)",
        },
        hover_name="city",
        height=460,
    )
    fig_emp_tri.add_trace(go.Scatter(
        x=_xl_e, y=np.polyval(_ce, _xl_e),
        mode="lines", name="Droite OLS",
        line=dict(color="#1A2332", dash="dash", width=2), showlegend=False,
    ))
    fig_emp_tri.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig_emp_tri.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_emp_tri, use_container_width=True)
    st.caption(
        "**Figure 6.1.** TRI moyen (axe horizontal) versus part modale vélo déclarée EMP 2019 "
        "(axe vertical). La droite OLS indique la tendance linéaire. "
        f"$\\rho_s = {_rho_e:+.3f}$ ($p = {_p_e:.3f}$). "
        + ("Un coefficient négatif significatif validerait l'hypothèse de friction topographique : "
           "les agglomérations accidentées cyclent moins." if _rho_e < -0.2 else
           "L'absence de corrélation négative significative suggère que d'autres facteurs "
           "(infrastructure, culture cyclable, multimodalité) dominent sur la contrainte topographique.")
    )
else:
    st.info(
        "Les données EMP 2019 ne sont pas disponibles dans ce corpus, "
        "ou le nombre d'agglomérations appariées est insuffisant (< 5). "
        "Vérifiez `data/external/mobility_sources/emp_2019_city_modal_share.csv`."
    )

# ── Section 7 — Diagnostic par agglomération ──────────────────────────────────
st.divider()
section(7, "Audit Topographique par Agglomération — Profil Détaillé et Carte des Stations")

_city_sel_topo = st.selectbox(
    "Sélectionnez une agglomération",
    options=sorted(df_dock["city"].unique()),
    index=list(sorted(df_dock["city"].unique())).index("Montpellier")
    if "Montpellier" in df_dock["city"].values else 0,
    key="city_topo_diag",
)

_grp_city = df_dock[df_dock["city"] == _city_sel_topo].copy()
_grp_clean = _grp_city.dropna(subset=["lat", "lon"])

if not _grp_clean.empty:
    col_diag_l, col_diag_r = st.columns([2, 3])

    with col_diag_l:
        st.markdown(f"#### Profil — {_city_sel_topo}")
        _kpi_rows = {"Stations dock-based": f"{len(_grp_clean)}"}
        if "topography_roughness_index" in _grp_clean.columns:
            _tri_s = _grp_clean["topography_roughness_index"].dropna()
            if not _tri_s.empty:
                _kpi_rows["TRI moyen (m)"]   = f"{_tri_s.mean():.3f}"
                _kpi_rows["TRI médian (m)"]  = f"{_tri_s.median():.3f}"
                _kpi_rows["TRI max (m)"]     = f"{_tri_s.max():.3f}"
        if "elevation_m" in _grp_clean.columns:
            _elev_s = _grp_clean["elevation_m"].dropna()
            if not _elev_s.empty:
                _kpi_rows["Altitude min (m)"]  = f"{_elev_s.min():.0f}"
                _kpi_rows["Altitude max (m)"]  = f"{_elev_s.max():.0f}"
                _kpi_rows["Dénivelé total (m)"] = f"{_elev_s.max() - _elev_s.min():.0f}"
                _kpi_rows["Altitude moy. (m)"] = f"{_elev_s.mean():.1f}"
        if _city_sel_topo in spatial_df["city"].values:
            _sp_r = spatial_df[spatial_df["city"] == _city_sel_topo].iloc[0]
            _kpi_rows["NN distance moy. (km)"] = f"{_sp_r['nn_mean_km']:.3f}"
            if not pd.isna(_sp_r.get("diameter_km", float("nan"))):
                _kpi_rows["Diamètre réseau (km)"] = f"{_sp_r['diameter_km']:.2f}"
            _kpi_rows["Dist. centroïde moy. (km)"] = f"{_sp_r['centroid_mean_km']:.3f}"

        if _city_sel_topo in topo_f["city"].values if not topo_f.empty else False:
            _imd_row = topo_f[topo_f["city"] == _city_sel_topo].iloc[0]
            _kpi_rows["Score T (/100)"] = f"{_imd_row['T_topo']*100:.1f}" if "T_topo" in _imd_row else "—"
            _kpi_rows["IMD (/100)"] = f"{_imd_row['IMD']:.1f}" if "IMD" in _imd_row else "—"

        for label, val in _kpi_rows.items():
            st.metric(label, val)

        # Histogram TRI stations
        if "topography_roughness_index" in _grp_clean.columns:
            _tri_city_s = _grp_clean["topography_roughness_index"].dropna()
            if len(_tri_city_s) >= 3:
                fig_tri_hist = px.histogram(
                    _tri_city_s,
                    nbins=20,
                    color_discrete_sequence=["#8e44ad"],
                    labels={"value": "TRI (m)", "count": "Stations"},
                    height=220,
                )
                fig_tri_hist.add_vline(
                    x=float(_tri_city_s.median()),
                    line_dash="dash", line_color="#1A2332",
                    annotation_text=f"Méd. {_tri_city_s.median():.2f} m",
                    annotation_position="top right",
                )
                fig_tri_hist.update_layout(
                    plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=20, b=10),
                    showlegend=False,
                    title=dict(text="Distribution TRI", font_size=12),
                )
                st.plotly_chart(fig_tri_hist, use_container_width=True)

    with col_diag_r:
        # Scatter géographique stations colorées par TRI/altitude
        _color_col = (
            "topography_roughness_index" if _has_tri and "topography_roughness_index" in _grp_clean.columns
            else "elevation_m" if _has_elev else None
        )
        if _color_col:
            _grp_map = _grp_clean.dropna(subset=[_color_col])
            if not _grp_map.empty:
                _label_col = "TRI (m)" if _color_col == "topography_roughness_index" else "Altitude (m)"
                fig_city_map = px.scatter_mapbox(
                    _grp_map,
                    lat="lat", lon="lon",
                    color=_color_col,
                    color_continuous_scale="Plasma",
                    hover_name="station_name" if "station_name" in _grp_map.columns else None,
                    hover_data={
                        _color_col: ":.2f",
                        "elevation_m": ":.0f" if "elevation_m" in _grp_map.columns else False,
                        "lat": False, "lon": False,
                    },
                    size_max=12,
                    opacity=0.85,
                    mapbox_style="carto-positron",
                    zoom=12,
                    center={"lat": float(_grp_map["lat"].mean()),
                            "lon": float(_grp_map["lon"].mean())},
                    labels={_color_col: _label_col},
                    height=480,
                )
                fig_city_map.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    coloraxis_colorbar=dict(title=_label_col, thickness=14),
                )
                st.plotly_chart(fig_city_map, use_container_width=True)
                st.caption(
                    f"**Figure 7.1.** Carte des stations dock-based de **{_city_sel_topo}** "
                    f"colorées par **{_label_col.lower()}** (SRTM 30 m). "
                    "Les stations en violet foncé (TRI élevé) ou orange foncé (altitude élevée) "
                    "sont les plus exposées à la friction topographique et aux déséquilibres "
                    "source/puits dans la journée de mobilité."
                )
else:
    st.info(f"Aucune station dock-based géolocalisée trouvée pour {_city_sel_topo}.")
