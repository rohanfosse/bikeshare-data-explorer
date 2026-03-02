"""
11_Physique.py — Physique statistique et thermodynamique appliquées aux réseaux VLS.
Concepts : entropie de Boltzmann, distribution de Boltzmann, loi de puissance, percolation,
modèle de gravité newtonien.
"""
from __future__ import annotations

import sys
from pathlib import Path
from math import radians, cos

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, optimize
import networkx as nx
from scipy.spatial import cKDTree, ConvexHull
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Physique Statistique - Réseaux VLS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Données ──────────────────────────────────────────────────────────────────
_NON_CITY = {"France", "FR", "Grand Est", "Basque Country"}

@st.cache_data(ttl=3600)
def load_dock_stations():
    df = load_stations()
    df = (
        df
        .query('station_type == "docked_bike"')
        .dropna(subset=["capacity", "lat", "lon", "city"])
        .query("capacity > 0")
        .loc[lambda x: ~x["city"].isin(_NON_CITY)]
        .copy()
    )
    return df

@st.cache_data(ttl=3600)
def compute_city_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("city")
        .agg(
            n_stations=("capacity", "count"),
            total_cap=("capacity", "sum"),
            mean_cap=("capacity", "mean"),
            std_cap=("capacity", "std"),
            lat_mean=("lat", "mean"),
            lon_mean=("lon", "mean"),
        )
        .query("n_stations >= 5")
        .reset_index()
    )

@st.cache_data(ttl=3600)
def compute_entropy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, grp in df.groupby("city"):
        if len(grp) < 5:
            continue
        caps = grp["capacity"].values.astype(float)
        total = caps.sum()
        p = caps / total
        p = p[p > 0]
        S = float(-np.sum(p * np.log(p)))
        S_max = float(np.log(len(caps)))
        H = S / S_max if S_max > 0 else 0.0
        cv = float(caps.std() / caps.mean()) if caps.mean() > 0 else 0.0
        T_star = float(caps.mean() / caps.max())
        rows.append({
            "city": city,
            "n": len(grp),
            "S_brute": S,
            "H_norm": H,
            "T_star": T_star,
            "cv": cv,
            "total_cap": float(caps.sum()),
            "mean_cap": float(caps.mean()),
        })
    return pd.DataFrame(rows).sort_values("H_norm", ascending=False)

@st.cache_data(ttl=3600)
def compute_city_area(df: pd.DataFrame) -> pd.DataFrame:
    areas = []
    for city, grp in df.groupby("city"):
        if len(grp) < 5:
            continue
        lats = grp["lat"].values
        lons = grp["lon"].values
        lat_m = lats.mean()
        x = lons * 111.0 * cos(radians(lat_m))
        y = lats * 111.0
        pts = np.column_stack([x, y])
        try:
            hull = ConvexHull(pts)
            area = float(hull.volume)
        except Exception:
            area = float("nan")
        areas.append({"city": city, "area_km2": area, "n": len(grp)})
    return pd.DataFrame(areas)

df_all = load_dock_stations()
city_agg = compute_city_agg(df_all)
df_ent = compute_entropy_table(df_all)

_n_stations = len(df_all)
_n_cities = df_all["city"].nunique()
_H_med = df_ent["H_norm"].median()
_lambda = 1.0 / df_all["capacity"].mean()

st.title("Physique Statistique et Thermodynamique des Réseaux VLS")
st.caption("Exploration transversale : analogies formelles entre physique statistique et mobilité urbaine")

abstract_box(
    "<b>Cadre théorique :</b> La physique statistique fournit un ensemble de formalismes "
    "mathématiques directement applicables aux systèmes complexes que constituent les réseaux "
    "de vélos en libre-service. L'entropie de Boltzmann quantifie le désordre de la distribution "
    "des capacités ; la loi de puissance révèle l'invariance d'échelle de la topologie ; "
    "le modèle de gravité newtonien prédit les flux inter-urbains ; la percolation identifie "
    "le seuil critique de connectivité piétonne. Ces analogies ne sont pas métaphoriques : "
    "elles appartiennent au corpus de la <i>physique urbaine</i> (Batty, 2013 ; Bettencourt, 2021).",
    findings=[
        (f"{_n_stations:,}", "stations dock-based"),
        (f"{_n_cities}", "agglomérations"),
        (f"H = {_H_med:.3f}", "entropie médiane"),
        (f"&lambda; = {_lambda:.3f}", "temp. Boltzmann (empl⁻¹)"),
    ],
)

sidebar_nav()
with st.sidebar:
    st.header("Paramètres")
    city_sel_perc = st.selectbox(
        "Ville pour la percolation",
        options=sorted(df_all["city"].unique()),
        index=sorted(df_all["city"].unique()).index("Montpellier")
        if "Montpellier" in df_all["city"].unique() else 0,
    )
    n_cities_grav = st.slider("Villes pour le modèle de gravité", 15, 40, 25, 5)
    st.divider()
    st.caption("R. Fossé & G. Pallares · 2025–2026")

# ── Section 1 — Entropie de Boltzmann ────────────────────────────────────────
st.divider()
section(1, "Entropie de Boltzmann — Désordre Thermique des Réseaux VLS")

st.markdown(r"""
En thermodynamique statistique, l'**entropie de Boltzmann** mesure le nombre de micro-états
compatibles avec un macro-état donné : $S = k_B \ln \Omega$. La formulation de Shannon (1948),
formellement identique, s'applique à toute distribution de probabilité :

$$S = -\sum_i p_i \ln p_i, \qquad H = \frac{S}{\ln N} \in [0,\, 1]$$

Ici, $p_i = c_i / C_{\text{tot}}$ est la fraction de capacité de la station $i$ dans son réseau.
$H \to 1$ : toutes les stations ont des capacités équivalentes (*réseau isotherme*).
$H \to 0$ : une station concentre toute la capacité (*état fondamental, réseau ordonné*).
""")

col1, col2 = st.columns(2)

with col1:
    fig_hist_H = px.histogram(
        df_ent, x="H_norm", nbins=25,
        labels={"H_norm": "Entropie normalisée H"},
        color_discrete_sequence=["#1A6FBF"],
        height=360,
    )
    fig_hist_H.add_vline(
        x=float(_H_med), line_dash="dash", line_color="#D96B27",
        annotation_text=f"Médiane H = {_H_med:.3f}",
        annotation_position="top left",
    )
    fig_hist_H.update_layout(
        plot_bgcolor="white",
        xaxis_title="Entropie normalisée H",
        yaxis_title="Nombre de villes",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_hist_H, use_container_width=True)
    st.caption(
        "**Figure 1.1.** Distribution de l'entropie de Shannon normalisée $H$ sur "
        f"{len(df_ent)} agglomérations françaises (dock-based, &ge;5 stations). "
        f"Médiane H&thinsp;=&thinsp;{_H_med:.3f} : les réseaux VLS français sont "
        "tendanciellement proches de la distribution uniforme (équilibre thermodynamique)."
    )

with col2:
    fig_ent_cv = px.scatter(
        df_ent, x="cv", y="H_norm",
        size="n", color="total_cap",
        hover_name="city",
        color_continuous_scale="Blues",
        labels={
            "cv": "Coefficient de variation (σ/μ)",
            "H_norm": "Entropie normalisée H",
            "total_cap": "Capacité totale",
            "n": "N stations",
        },
        height=360,
    )
    fig_ent_cv.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_ent_cv, use_container_width=True)
    st.caption(
        "**Figure 1.2.** Entropie normalisée en fonction du coefficient de variation (CV = &sigma;/&mu;) "
        "des capacités par ville. La taille des points encode le nombre de stations, "
        "la couleur la capacité totale. La relation H–CV est négative par construction : "
        "un réseau plus hétérogène ($\\uparrow$ CV) est plus ordonné ($\\downarrow$ H)."
    )

# Diagramme Entropie–Température
st.markdown(r"""
### 1.1 Diagramme Entropie–Température

Par analogie avec les diagrammes d'état thermodynamiques, on définit la **température réduite**
d'un réseau comme le rapport entre la capacité moyenne et la capacité maximale :
$T^* = \langle c \rangle / c_{\max}$. Un réseau *chaud* ($T^* \to 1$) est homogène ;
un réseau *froid* ($T^* \to 0$) est dominé par une grande station-hub.
""")

fig_HT = px.scatter(
    df_ent, x="T_star", y="H_norm",
    size="total_cap", color="cv",
    hover_name="city",
    color_continuous_scale="RdYlBu_r",
    labels={
        "T_star": "Température réduite T* = ⟨c⟩ / c_max",
        "H_norm": "Entropie normalisée H",
        "cv": "Coeff. variation",
        "total_cap": "Capacité totale",
    },
    height=420,
)
fig_HT.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))

# Quadrants
fig_HT.add_hline(y=0.85, line_dash="dot", line_color="#888888", opacity=0.6)
fig_HT.add_vline(x=0.55, line_dash="dot", line_color="#888888", opacity=0.6)
fig_HT.add_annotation(x=0.25, y=0.92, text="Uniforme & hiérarchisé", showarrow=False,
                       font=dict(color="#555555", size=10))
fig_HT.add_annotation(x=0.75, y=0.92, text="Uniforme & homogène", showarrow=False,
                       font=dict(color="#1E8449", size=10))
fig_HT.add_annotation(x=0.25, y=0.70, text="Hiérarchisé & froid", showarrow=False,
                       font=dict(color="#C0392B", size=10))
fig_HT.add_annotation(x=0.75, y=0.70, text="Dispersé & chaud", showarrow=False,
                       font=dict(color="#D96B27", size=10))
st.plotly_chart(fig_HT, use_container_width=True)
st.caption(
    "**Figure 1.3.** Diagramme Entropie–Température analogue à un diagramme d'état thermodynamique. "
    "Taille &prop; capacité totale, couleur = coefficient de variation. "
    "Les quadrants délimitent quatre régimes thermodynamiques : "
    "les réseaux matures (Paris, Strasbourg, Montpellier) se positionnent généralement "
    "dans la zone *uniforme & homogène* (H élevé, T* modéré à élevé)."
)

# ── Section 2 — Distribution de Boltzmann ────────────────────────────────────
st.divider()
section(2, "Distribution de Boltzmann — Modèle Statistique des Capacités")

st.markdown(r"""
À l'équilibre thermique, la probabilité qu'une particule ait une énergie $E$ suit
la **distribution de Boltzmann** :

$$P(E) \propto e^{-E/k_BT}$$

En posant $E \equiv c$ (capacité), la distribution de Boltzmann prévoit une loi exponentielle
décroissante $P(c) = \lambda e^{-\lambda c}$, avec $\lambda = 1/\langle c \rangle$ — paramètre
directement relié à la *température thermodynamique* du réseau.

Si la distribution réelle s'en écarte (lognormale, gamma, loi de puissance), cela indique
que des **contraintes supplémentaires** (urbanisme, budget, géographie) brisent l'équilibre
statistique pur.
""")

caps_all = df_all["capacity"].values.astype(float)
mean_c = float(caps_all.mean())
lam_fit = 1.0 / mean_c
_, scale_exp = stats.expon.fit(caps_all, floc=0)
lam_exp = 1.0 / scale_exp
s_ln, _, scale_ln = stats.lognorm.fit(caps_all, floc=0)
mu_ln = float(np.log(scale_ln))
a_gam, _, scale_gam = stats.gamma.fit(caps_all, floc=0)

x_fit = np.linspace(1, 80, 300)

fig_bolt = go.Figure()
# Histogramme
hist_vals, hist_edges = np.histogram(caps_all[caps_all <= 80], bins=60, density=True)
bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
fig_bolt.add_trace(go.Bar(
    x=bin_centers, y=hist_vals, name="Données observées",
    marker_color="#1A6FBF", opacity=0.65,
))
fig_bolt.add_trace(go.Scatter(
    x=x_fit, y=stats.expon.pdf(x_fit, 0, scale_exp),
    name=f"Boltzmann exponentiel (λ = {lam_exp:.3f})",
    line=dict(color="#D96B27", width=2.5),
))
fig_bolt.add_trace(go.Scatter(
    x=x_fit, y=stats.lognorm.pdf(x_fit, s_ln, 0, scale_ln),
    name=f"Lognormal (μ_ln = {mu_ln:.2f}, σ = {s_ln:.2f})",
    line=dict(color="#1E8449", width=2.5, dash="dash"),
))
fig_bolt.add_trace(go.Scatter(
    x=x_fit, y=stats.gamma.pdf(x_fit, a_gam, 0, scale_gam),
    name=f"Gamma (α = {a_gam:.2f})",
    line=dict(color="#C0392B", width=2.5, dash="dot"),
))
fig_bolt.update_layout(
    plot_bgcolor="white",
    xaxis_title="Capacité (emplacements par station)",
    yaxis_title="Densité de probabilité",
    legend=dict(x=0.55, y=0.95),
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_bolt, use_container_width=True)

ks_exp = stats.kstest(caps_all[caps_all <= 150], "expon", args=(0, scale_exp))
ks_ln  = stats.kstest(caps_all[caps_all <= 150], "lognorm", args=(s_ln, 0, scale_ln))

col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Température thermodynamique T*", f"{mean_c:.1f} empl", help="T* = 1/λ = ⟨c⟩")
col_b2.metric("KS test Boltzmann (p-value)", f"{ks_exp.pvalue:.4f}")
col_b3.metric("KS test Lognormal (p-value)", f"{ks_ln.pvalue:.4f}")

st.caption(
    "**Figure 2.1.** Ajustement de la distribution nationale des capacités (dock-based, c&thinsp;&le;&thinsp;80) "
    f"par trois modèles. Le modèle de Boltzmann (exponentiel, &lambda;&thinsp;=&thinsp;{lam_exp:.3f}&thinsp;empl<sup>-1</sup>) "
    "prédit la distribution d'un système à l'équilibre thermodynamique pur. "
    f"Le test de Kolmogorov-Smirnov rejette l'exponentiel (p&thinsp;=&thinsp;{ks_exp.pvalue:.4f}) "
    f"mais aussi le lognormal (p&thinsp;=&thinsp;{ks_ln.pvalue:.4f}), "
    "indiquant que des contraintes urbanistiques et budgétaires structurent la distribution "
    "au-delà de l'équilibre statistique maximal.",
    unsafe_allow_html=True,
)

# ── Section 3 — Loi de puissance ─────────────────────────────────────────────
st.divider()
section(3, "Loi de Puissance — Invariance d'Échelle et Topologie Scale-Free")

st.markdown(r"""
Les **réseaux scale-free** (Barabási & Albert, 1999) exhibent une distribution de degré
en loi de puissance $P(k) \sim k^{-\gamma}$, avec $\gamma \in [2, 3]$ pour la plupart
des réseaux réels. Ce comportement est analogue aux **systèmes critiques** en physique
(transitions de phase du second ordre, phénomènes d'invariance d'échelle).

La distribution des capacités et la distribution de la taille des réseaux VLS sont testées
via leur *CCDF* (*Complementary CDF*) en échelle log-log. Une droite en log-log valide
la loi de puissance ; la pente donne l'exposant $\alpha$.
""")

# CCDF des capacités
caps_s = np.sort(caps_all)[::-1]
n_total = len(caps_s)
ccdf_caps = np.arange(1, n_total + 1) / n_total

# CCDF des tailles de réseau
ns_sorted = np.sort(city_agg["n_stations"].values)[::-1]
ccdf_cities = np.arange(1, len(ns_sorted) + 1) / len(ns_sorted)

# Fit loi de puissance sur les capacités (CCDF : slope = -(alpha-1))
mask_pw = caps_s >= 10
log_c = np.log10(caps_s[mask_pw])
log_p = np.log10(ccdf_caps[mask_pw])
slope_c, intercept_c, r_c, *_ = stats.linregress(log_c, log_p)
alpha_c = 1 - slope_c

# Fit sur tailles de réseaux
mask_n = ns_sorted >= 8
log_n = np.log10(ns_sorted[mask_n])
log_pn = np.log10(ccdf_cities[mask_n])
slope_n, intercept_n, r_n, *_ = stats.linregress(log_n, log_pn)
alpha_n = 1 - slope_n

col_pw1, col_pw2 = st.columns(2)

with col_pw1:
    fig_pw1 = go.Figure()
    fig_pw1.add_trace(go.Scatter(
        x=caps_s, y=ccdf_caps, mode="markers",
        marker=dict(color="#1A6FBF", size=3, opacity=0.4),
        name="Données observées",
    ))
    x_fit_c = np.logspace(1, np.log10(caps_s.max()), 80)
    fig_pw1.add_trace(go.Scatter(
        x=x_fit_c,
        y=10**intercept_c * x_fit_c**slope_c,
        mode="lines",
        name=f"Loi de puissance α = {alpha_c:.2f}",
        line=dict(color="#D96B27", width=2.5),
    ))
    fig_pw1.update_layout(
        xaxis_type="log", yaxis_type="log",
        plot_bgcolor="white",
        xaxis_title="Capacité c (emplacements)",
        yaxis_title="P(C ≥ c) — CCDF",
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(x=0.35, y=0.95),
    )
    st.plotly_chart(fig_pw1, use_container_width=True)
    st.caption(
        f"**Figure 3.1.** CCDF des capacités en log-log. "
        f"Exposant estimé &alpha;&thinsp;=&thinsp;{alpha_c:.2f} "
        f"({'dans' if 2 <= alpha_c <= 3 else 'hors'} la zone scale-free typique [2, 3]). "
        "Une droite parfaite en log-log serait caractéristique d'un réseau scale-free pur.",
        unsafe_allow_html=True,
    )

with col_pw2:
    fig_pw2 = go.Figure()
    fig_pw2.add_trace(go.Scatter(
        x=ns_sorted, y=ccdf_cities, mode="markers",
        marker=dict(color="#1A6FBF", size=8, opacity=0.7),
        text=city_agg.nlargest(len(ns_sorted), "n_stations")["city"].values,
        hovertemplate="%{text}<br>N = %{x}<br>CCDF = %{y:.3f}",
        name="Villes",
    ))
    x_fit_n = np.logspace(np.log10(8), np.log10(ns_sorted.max()), 60)
    fig_pw2.add_trace(go.Scatter(
        x=x_fit_n,
        y=10**intercept_n * x_fit_n**slope_n,
        mode="lines",
        name=f"Loi de puissance α = {alpha_n:.2f}",
        line=dict(color="#D96B27", width=2.5),
    ))
    fig_pw2.update_layout(
        xaxis_type="log", yaxis_type="log",
        plot_bgcolor="white",
        xaxis_title="Nombre de stations par ville",
        yaxis_title="P(N ≥ n) — CCDF",
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(x=0.35, y=0.95),
    )
    st.plotly_chart(fig_pw2, use_container_width=True)
    st.caption(
        f"**Figure 3.2.** CCDF des tailles de réseau par agglomération. "
        f"Exposant &alpha;&thinsp;=&thinsp;{alpha_n:.2f} — "
        "analogue à la loi de Zipf pour la taille des villes (Zipf, 1949). "
        "La queue lourde confirme que quelques métropoles concentrent la majorité des stations.",
        unsafe_allow_html=True,
    )

# ── Section 4 — Modèle de gravité ─────────────────────────────────────────────
st.divider()
section(4, "Modèle de Gravité Newtonien — Attractivité Inter-Urbaine")

st.markdown(r"""
La **loi de Newton** $F = G m_1 m_2 / d^2$ a été transposée à la mobilité urbaine
par Reilly (1929) et formalisée par Zipf (1946) :

$$T_{ij} = K \,\frac{M_i^{\,\alpha} \cdot M_j^{\,\beta}}{d_{ij}^{\,\gamma}}$$

Ici, la **masse** est la capacité totale du réseau VLS (proxy de la demande), et $d_{ij}$
la distance haversine inter-ville. La **force gravitationnelle** $F_{ij}$ identifie
les paires de villes candidates à un réseau de vélos inter-communal ou à une politique
de régulation commune.
""")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

@st.cache_data(ttl=3600)
def compute_gravity(city_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    from itertools import combinations
    cities_top = city_df.nlargest(top_n, "total_cap").copy()
    rows = []
    for i, j in combinations(cities_top.index, 2):
        ri, rj = cities_top.loc[i], cities_top.loc[j]
        d = haversine_km(ri["lat_mean"], ri["lon_mean"], rj["lat_mean"], rj["lon_mean"])
        if d < 1:
            continue
        F = (ri["total_cap"] * rj["total_cap"]) / d**2
        rows.append({"city_i": ri["city"], "city_j": rj["city"],
                     "dist_km": d, "F_grav": F,
                     "mass_i": ri["total_cap"], "mass_j": rj["total_cap"]})
    return pd.DataFrame(rows).sort_values("F_grav", ascending=False)

df_grav = compute_gravity(city_agg, n_cities_grav)

col_g1, col_g2 = st.columns([1.3, 1])

with col_g1:
    top12 = df_grav.head(12).copy()
    top12["paire"] = top12["city_i"] + " — " + top12["city_j"]
    top12 = top12.sort_values("F_grav", ascending=True)
    fig_grav_bar = px.bar(
        top12, x="F_grav", y="paire", orientation="h",
        color="dist_km", color_continuous_scale="Blues_r",
        labels={"F_grav": "Force gravitationnelle F (u.a.)", "paire": "",
                "dist_km": "Distance (km)"},
        height=380,
    )
    fig_grav_bar.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_grav_bar, use_container_width=True)
    st.caption(
        "**Figure 4.1.** Top 12 paires d'agglomérations par force gravitationnelle VLS "
        f"(sur les {n_cities_grav} villes avec la plus grande capacité totale). "
        "La couleur encode la distance inter-urbaine : les paires proches et massives "
        "sont candidates à une gouvernance intercommunale du vélo en libre-service."
    )

with col_g2:
    # F vs distance en log-log
    log_d = np.log10(df_grav["dist_km"])
    log_F = np.log10(df_grav["F_grav"])
    slope_g, intercept_g, r_g, p_g, _ = stats.linregress(log_d, log_F)

    fig_grav_log = go.Figure()
    fig_grav_log.add_trace(go.Scatter(
        x=df_grav["dist_km"], y=df_grav["F_grav"],
        mode="markers",
        marker=dict(color="#1A6FBF", size=5, opacity=0.5),
        hovertemplate="%{customdata[0]} — %{customdata[1]}<br>d = %{x:.0f} km<br>F = %{y:.0f}",
        customdata=df_grav[["city_i", "city_j"]].values,
        name="Paires observées",
    ))
    d_range = np.logspace(np.log10(df_grav["dist_km"].min()),
                          np.log10(df_grav["dist_km"].max()), 80)
    fig_grav_log.add_trace(go.Scatter(
        x=d_range, y=10**intercept_g * d_range**slope_g,
        mode="lines",
        name=f"F ∝ d<sup>{slope_g:.2f}</sup>  (R²={r_g**2:.3f})",
        line=dict(color="#D96B27", width=2.5),
    ))
    fig_grav_log.update_layout(
        xaxis_type="log", yaxis_type="log",
        plot_bgcolor="white",
        xaxis_title="Distance inter-urbaine (km)",
        yaxis_title="Force gravitationnelle (u.a.)",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(x=0.02, y=0.10),
    )
    st.plotly_chart(fig_grav_log, use_container_width=True)
    st.caption(
        f"**Figure 4.2.** Force gravitationnelle en fonction de la distance (log-log). "
        f"Exposant observé : {slope_g:.2f} (Newton prédit −2,00 ; "
        f"R²&thinsp;=&thinsp;{r_g**2:.3f}). L'écart à −2 reflète les asymétries "
        "de taille des réseaux, non pris en compte dans la formulation simple.",
        unsafe_allow_html=True,
    )

# ── Section 5 — Percolation ───────────────────────────────────────────────────
st.divider()
section(5, "Percolation — Seuil Critique de Connectivité Piétonne")

st.markdown(r"""
La **théorie de la percolation** (Broadbent & Hammersley, 1957) étudie l'apparition
d'un *cluster géant* dans un réseau aléatoire. Il existe un **seuil critique** $r_c$
au-dessus duquel la composante géante émerge soudainement — analogue exact
d'une **transition de phase du second ordre** (universalité, exposants critiques).

**Application :** deux stations sont *connectées* si leur distance est $d \le r$.
On fait varier $r$ de 50 m à 2 km et on observe l'émergence du cluster géant.
Le seuil $r_c$ définit la **distance piétonne critique** à partir de laquelle
le réseau devient effectivement continu pour un usager à pied.

La *dérivée* de la fraction géante — analogue à la **chaleur spécifique** en physique —
présente un maximum au point critique, signature universelle des transitions de phase.
""")

@st.cache_data(ttl=3600)
def compute_percolation(city_name: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grp = df[df["city"].str.contains(city_name, na=False, regex=False)].copy()
    if len(grp) < 5:
        return np.array([]), np.array([]), np.array([])
    coords = grp[["lat", "lon"]].values
    n = len(coords)
    radii_km = np.linspace(0.05, 2.0, 60)
    giant_fracs, derivs, n_comps = [], [], []
    tree = cKDTree(coords)
    for r_km in radii_km:
        r_deg = r_km / 111.0
        G = nx.Graph()
        G.add_nodes_from(range(n))
        pairs = tree.query_pairs(r_deg)
        G.add_edges_from(pairs)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        gf = len(comps[0]) / n if comps else 0.0
        giant_fracs.append(gf)
        n_comps.append(len(comps))
    giant_fracs = np.array(giant_fracs)
    derivs = np.gradient(giant_fracs, radii_km)
    return radii_km, giant_fracs, derivs

with st.spinner(f"Calcul de la percolation pour {city_sel_perc}..."):
    radii_km, gf_arr, deriv_arr = compute_percolation(city_sel_perc, df_all)

if len(radii_km) > 0:
    r_c_idx = int(np.argmax(deriv_arr))
    r_c = radii_km[r_c_idx]

    fig_perc = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Fraction de la composante géante P(r)",
            "Susceptibilité dP/dr — analogue à la chaleur spécifique",
        ],
    )
    fig_perc.add_trace(go.Scatter(
        x=radii_km * 1000, y=gf_arr, mode="lines",
        name="Fraction géante", line=dict(color="#1A6FBF", width=2.5),
    ), row=1, col=1)
    fig_perc.add_vline(x=r_c * 1000, line_dash="dash", line_color="#D96B27",
                       annotation_text=f"r_c = {r_c*1000:.0f} m",
                       annotation_position="top right",
                       row=1, col=1)
    fig_perc.add_hline(y=0.5, line_dash="dot", line_color="#888888", row=1, col=1)

    fig_perc.add_trace(go.Scatter(
        x=radii_km * 1000, y=deriv_arr, mode="lines",
        name="Susceptibilité dP/dr", line=dict(color="#C0392B", width=2.5),
    ), row=1, col=2)
    fig_perc.add_vline(x=r_c * 1000, line_dash="dash", line_color="#D96B27",
                       annotation_text=f"r_c = {r_c*1000:.0f} m",
                       row=1, col=2)

    fig_perc.update_xaxes(title_text="Rayon de voisinage r (m)")
    fig_perc.update_yaxes(title_text="P_giant", row=1, col=1)
    fig_perc.update_yaxes(title_text="dP_giant/dr", row=1, col=2)
    fig_perc.update_layout(
        plot_bgcolor="white", height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_perc, use_container_width=True)

    col_pc1, col_pc2, col_pc3 = st.columns(3)
    col_pc1.metric("Seuil de percolation r_c", f"{r_c*1000:.0f} m")
    col_pc2.metric("Fraction géante à r_c", f"{gf_arr[r_c_idx]:.1%}")
    n_grp = len(df_all[df_all["city"].str.contains(city_sel_perc, na=False, regex=False)])
    col_pc3.metric("Stations analysées", f"{n_grp}")

    st.caption(
        f"**Figure 5.1–5.2.** Transition de percolation pour le réseau VLS de **{city_sel_perc}**. "
        f"Le seuil critique r<sub>c</sub>&thinsp;=&thinsp;{r_c*1000:.0f}&thinsp;m correspond à la distance piétonne "
        f"à partir de laquelle {gf_arr[r_c_idx]:.1%} des stations sont connectées en un seul cluster. "
        "Le pic de susceptibilité (dérivée maximale) est la signature universelle d'une transition "
        "de phase du second ordre — analogue à la divergence de la chaleur spécifique au point de Curie "
        "en physique des matériaux magnétiques.",
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Données insuffisantes pour {city_sel_perc} (moins de 5 stations).")

st.divider()
st.caption(
    "**Sources.** Boltzmann L. (1877). *Über die Beziehung zwischen dem zweiten Hauptsatze der mechanischen "
    "Wärmetheorie.* Shannon C.E. (1948). *A Mathematical Theory of Communication.* "
    "Barabási A.L. & Albert R. (1999). *Emergence of Scaling in Random Networks.* Science. "
    "Broadbent S.R. & Hammersley J.M. (1957). *Percolation processes.* Proc. Cambridge Phil. Soc. "
    "Batty M. (2013). *The New Science of Cities.* MIT Press. — "
    "**R. Fossé & G. Pallares · 2025–2026**"
)
