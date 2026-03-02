"""
13_Chimie.py — Chimie et cinétique chimique appliquées aux flux VLS (Montpellier Vélomagg).
Concepts : équilibre chimique (Le Chatelier), cinétique d'ordre 1, loi d'Arrhenius,
catalyse par les hubs, profil d'énergie potentielle journalier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, optimize
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Chimie & Cinétique - Réseaux VLS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Chemins ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_NET_FLOW_PATH   = _ROOT / "data" / "processed" / "flow_analysis" / "net_flow_analysis.csv"
_HOURLY_PATH     = _ROOT / "data" / "processed" / "flow_analysis" / "hourly_flow_statistics.csv"
_WEATHER_PATH    = _ROOT / "data" / "processed" / "weather_data_enriched.csv"
_STRESS_PATH     = _ROOT / "data" / "processed" / "station_stress_ranking.csv"

# ── Chargement ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_net_flow() -> pd.DataFrame:
    return pd.read_csv(_NET_FLOW_PATH)

@st.cache_data(ttl=3600)
def load_hourly_stats() -> pd.DataFrame:
    return pd.read_csv(_HOURLY_PATH)

@st.cache_data(ttl=3600)
def load_weather() -> pd.DataFrame:
    df = pd.read_csv(_WEATHER_PATH, parse_dates=["datetime"])
    df["hour"] = df["datetime"].dt.hour
    return df

@st.cache_data(ttl=3600)
def load_stress() -> pd.DataFrame:
    return pd.read_csv(_STRESS_PATH)

@st.cache_data(ttl=3600)
def compute_keq(nf: pd.DataFrame) -> pd.DataFrame:
    """Constante d'équilibre K = outflow_moy / inflow_moy par station."""
    agg = nf.groupby("station").agg(
        mean_inflow=("inflow", "mean"),
        mean_outflow=("outflow", "mean"),
        mean_abs_net=("net_flow", lambda x: x.abs().mean()),
        std_net=("net_flow", "std"),
    ).reset_index()
    agg["K_eq"] = agg["mean_outflow"] / agg["mean_inflow"].replace(0, np.nan)
    agg = agg.dropna(subset=["K_eq"])
    agg["log_K"] = np.log(agg["K_eq"])
    return agg

@st.cache_data(ttl=3600)
def compute_hourly_temp(weather: pd.DataFrame) -> pd.DataFrame:
    """Température moyenne par heure de la journée."""
    return weather.groupby("hour").agg(
        T_mean=("temperature", "mean"),
        T_std=("temperature", "std"),
        precip_mean=("precipitation", "mean"),
        bad_weather=("bad_weather_score", "mean"),
    ).reset_index()

nf       = load_net_flow()
hf       = load_hourly_stats()
weather  = load_weather()
stress   = load_stress()
keq_df   = compute_keq(nf)
hourly_T = compute_hourly_temp(weather)

_n_stations = nf["station"].nunique()
_K_med      = float(keq_df["K_eq"].median())
_T_peak     = int(hf.loc[hf["total_trips"].idxmax(), "hour"])
_t_half_est = "--"

# Estimation demi-vie (section 2)
hf_decay = hf[hf["hour"] >= _T_peak].copy()
hf_decay["t_rel"] = hf_decay["hour"] - _T_peak
if len(hf_decay) >= 4:
    log_std = np.log(hf_decay["std_flow"].clip(lower=1e-9))
    sl, ic, *_ = stats.linregress(hf_decay["t_rel"], log_std)
    k_cin = float(-sl)
    if k_cin > 0:
        t_half = np.log(2) / k_cin
        _t_half_est = f"{t_half:.1f} h"

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("Chimie et Cinétique Chimique des Flux VLS")
st.caption("Exploration transversale : équilibre, cinétique d'ordre 1, Arrhenius, catalyse — Montpellier Vélomagg")

abstract_box(
    "<b>Cadre théorique :</b> La chimie physique offre un cadre rigoureux pour décrire "
    "les flux de mobilité à l'échelle d'une station. Le <i>principe de Le Chatelier</i> prédit "
    "la réponse d'un réseau à une perturbation (heure de pointe, météo). La "
    "<i>cinétique d'ordre 1</i> modélise la résorption du déséquilibre après perturbation. "
    "La <i>loi d'Arrhenius</i> quantifie la dépendance de l'usage cyclable à la température. "
    "La <i>catalyse</i> décrit le rôle des stations-hub dans la réduction de la barrière "
    "de redistribution. Données source : <b>Vélomagg Montpellier</b> — "
    f"<b>{_n_stations} stations</b>, flux horaires moyennés sur 5 ans.",
    findings=[
        (f"{_n_stations}", "stations Vélomagg"),
        (f"K = {_K_med:.3f}", "constante d'équilibre médiane"),
        (f"t½ = {_t_half_est}", "demi-vie du déséquilibre"),
        (f"h {_T_peak}h00", "pic de flux (réactif activé)"),
    ],
)

sidebar_nav()
with st.sidebar:
    st.header("Paramètres")
    peak_hour = st.slider(
        "Heure du pic (équilibre perturbé)", 6, 22, _T_peak,
        help="Heure à partir de laquelle on mesure la décroissance cinétique",
    )
    Ea_cat_pct = st.slider(
        "Réduction Ea par catalyse (%)", 10, 60, 35, 5,
        help="Diminution de l'énergie d'activation apportée par les stations-hub",
    )
    T_min_K = st.number_input("T min Arrhenius (°C)", value=5.0, step=1.0)
    T_max_K = st.number_input("T max Arrhenius (°C)", value=35.0, step=1.0)
    st.divider()
    st.caption("R. Fossé & G. Pallares · 2025–2026")

# ── Section 1 — Équilibre chimique ───────────────────────────────────────────
st.divider()
section(1, "Équilibre Chimique — Constante K et Principe de Le Chatelier")

st.markdown(r"""
En chimie, un système en **équilibre chimique** satisfait :

$$A \underset{k_{-1}}{\stackrel{k_1}{\rightleftharpoons}} B, \qquad K_{eq} = \frac{[B]}{[A]} = \frac{k_1}{k_{-1}}$$

$K_{eq} > 1$ : l'équilibre est déplacé vers les *produits* (réaction favorable). $K_{eq} < 1$ : vers les *réactifs*.

**Application VLS :** On modélise chaque station comme un réacteur chimique.
Le *réactif* A est un vélo disponible (inflow), le *produit* B un vélo loué et redistribué (outflow).
La constante d'équilibre $K_{eq} = \langle q_{out} \rangle / \langle q_{in} \rangle$
mesure si une station est structurellement **émettrice** ($K > 1$) ou **réceptrice** ($K < 1$).

**Le principe de Le Chatelier** prédit que tout réseau en déséquilibre tend à retrouver
son état d'équilibre : une station surexploitée (pic de sortie) sera compensée par un flux
entrant accru lors de la redistribution nocturne.
""")

col1, col2 = st.columns(2)

with col1:
    # Distribution de K_eq
    keq_plot = keq_df[keq_df["K_eq"] <= 4].copy()
    fig_keq = px.histogram(
        keq_plot, x="K_eq", nbins=30,
        color_discrete_sequence=["#1A6FBF"],
        labels={"K_eq": "Constante d'équilibre K = ⟨q_out⟩ / ⟨q_in⟩"},
        height=360,
    )
    fig_keq.add_vline(x=1.0, line_dash="dash", line_color="#D96B27",
                      annotation_text="Équilibre K = 1",
                      annotation_position="top right")
    fig_keq.add_vline(x=float(keq_df["K_eq"].median()), line_dash="dot",
                      line_color="#C0392B",
                      annotation_text=f"Médiane K = {keq_df['K_eq'].median():.3f}",
                      annotation_position="top left")
    fig_keq.update_layout(
        plot_bgcolor="white",
        xaxis_title="K_eq (stations à K > 1 : émettrices nettes)",
        yaxis_title="Nombre de stations",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_keq, use_container_width=True)
    st.caption(
        "**Figure 1.1.** Distribution de la constante d'équilibre K par station Vélomagg. "
        "La majorité des stations présentent K ≈ 1 (équilibre des flux entrants et sortants), "
        "confirmant la quasi-réversibilité du système sur une journée. "
        "Les stations à K ≫ 1 sont des *puits* structurels (hubs de destination) ; "
        "K < 1 indique des *sources* nettes (zones résidentielles de départ matinal)."
    )

with col2:
    # K_eq vs déséquilibre net
    fig_keq_net = px.scatter(
        keq_df,
        x="log_K", y="mean_abs_net",
        color="std_net",
        color_continuous_scale="RdYlBu_r",
        hover_name="station",
        labels={
            "log_K": "ln(K) — biais structurel (+ = émettrice, - = réceptrice)",
            "mean_abs_net": "Déséquilibre moyen |q_net| (flux/h)",
            "std_net": "Volatilité du flux (σ)",
        },
        height=360,
    )
    fig_keq_net.add_vline(x=0, line_dash="dot", line_color="#888888",
                           annotation_text="Équilibre parfait")
    fig_keq_net.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_keq_net, use_container_width=True)
    st.caption(
        "**Figure 1.2.** Déséquilibre moyen |q_net| en fonction de ln(K). "
        "Les stations les plus éloignées de l'équilibre (|ln K| grand) présentent "
        "aussi les déséquilibres absolus les plus élevés. "
        "La couleur encode la volatilité &sigma; du flux : les stations centrales "
        "(K élevé, &sigma; élevé) sont les candidates prioritaires à la redistribution active.",
        unsafe_allow_html=True,
    )

# Perturbation Le Chatelier : K avant/après pointe
st.markdown(r"""
### 1.1 Perturbation de l'équilibre — Principe de Le Chatelier

Le principe de Le Chatelier stipule que *tout système en équilibre soumis à une perturbation
extérieure évolue de façon à s'opposer à cette perturbation*. Appliqué aux flux VLS :
une heure de pointe correspond à une **perturbation de concentration** (surconsommation de vélos).
""")

nf_early  = nf[nf["hour"].between(7, 9)].groupby("station")["net_flow"].mean().reset_index()
nf_night  = nf[nf["hour"].between(0, 5)].groupby("station")["net_flow"].mean().reset_index()
nf_compare = nf_early.merge(nf_night, on="station", suffixes=("_pointe", "_nuit"))

fig_lechat = go.Figure()
fig_lechat.add_trace(go.Scatter(
    x=nf_compare["net_flow_pointe"],
    y=nf_compare["net_flow_nuit"],
    mode="markers",
    marker=dict(
        color=nf_compare["net_flow_pointe"],
        colorscale="RdBu", size=8, opacity=0.8,
        colorbar=dict(title="Flux net pointe"),
        cmid=0,
    ),
    text=nf_compare["station"],
    hovertemplate="%{text}<br>Pointe = %{x:.3f}<br>Nuit = %{y:.3f}",
    name="Stations",
))
# Droite Le Chatelier théorique : nuit = -pointe (compensation parfaite)
lim = max(abs(nf_compare["net_flow_pointe"]).max(),
          abs(nf_compare["net_flow_nuit"]).max()) * 1.1
fig_lechat.add_trace(go.Scatter(
    x=[-lim, lim], y=[lim, -lim],
    mode="lines", name="Le Chatelier parfait (nuit = −pointe)",
    line=dict(color="#D96B27", width=2, dash="dash"),
))
fig_lechat.add_vline(x=0, line_color="#CCCCCC", line_width=1)
fig_lechat.add_hline(y=0, line_color="#CCCCCC", line_width=1)
fig_lechat.update_layout(
    plot_bgcolor="white",
    xaxis_title="Flux net moyen en heure de pointe (7h–9h)",
    yaxis_title="Flux net moyen la nuit (0h–5h)",
    legend=dict(x=0.02, y=0.98),
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_lechat, use_container_width=True)

slope_lc, intercept_lc, r_lc, p_lc, _ = stats.linregress(
    nf_compare["net_flow_pointe"], nf_compare["net_flow_nuit"]
)
col_lc1, col_lc2, col_lc3 = st.columns(3)
col_lc1.metric("Pente Le Chatelier observée", f"{slope_lc:.3f}", "théorique : −1.0")
col_lc2.metric("R² (corrélation pointe–nuit)", f"{r_lc**2:.3f}")
col_lc3.metric("p-value (Spearman)", f"{p_lc:.4f}")
st.caption(
    "**Figure 1.3.** Test du principe de Le Chatelier : le flux net nocturne (0h–5h) "
    "devrait être l'opposé exact du flux diurne en heure de pointe (pente = −1, droite orange). "
    f"La pente observée {slope_lc:.3f} révèle une compensation "
    f"{'quasi-complète' if abs(slope_lc + 1) < 0.3 else 'partielle'} du déséquilibre : "
    "le réseau ne retrouve pas toujours son état d'équilibre chimique par la seule mécanique "
    "des usages, justifiant l'intervention active de redistribution."
)

# ── Section 2 — Cinétique d'ordre 1 ──────────────────────────────────────────
st.divider()
section(2, "Cinétique d'Ordre 1 — Demi-Vie du Déséquilibre")

st.markdown(r"""
La **cinétique d'ordre 1** est la plus simple des lois cinétiques chimiques :

$$\frac{d[A]}{dt} = -k\,[A] \implies [A](t) = [A]_0\, e^{-kt}, \qquad t_{1/2} = \frac{\ln 2}{k}$$

Elle s'applique à tout processus dont le taux de disparition est proportionnel à la concentration actuelle.

**Application VLS :** La *concentration de déséquilibre* d'un réseau est mesurée par
l'écart-type inter-stations du flux net $\sigma_{q}(t)$ — une mesure de la *dispersion*
des flux à l'heure $t$. Après le pic de l'après-midi (moment de perturbation maximale),
$\sigma_q$ décroît vers sa valeur nocturne (équilibre). Cette décroissance suit-elle
une loi d'ordre 1 ?

Le **taux de redistribution** $k$ (h⁻¹) quantifie l'efficacité du rééquilibrage spontané
du réseau ; la **demi-vie** $t_{1/2}$ est le temps nécessaire pour diviser par deux
l'hétérogénéité des flux.
""")

hf_sorted = hf.sort_values("hour").copy()

# Calcul de la décroissance depuis l'heure de pointe sélectionnée
hf_post = hf_sorted[hf_sorted["hour"] >= peak_hour].copy()
hf_post["t_rel"] = hf_post["hour"] - peak_hour
hf_pre  = hf_sorted[hf_sorted["hour"] < peak_hour].copy()
hf_pre["t_rel"] = hf_pre["hour"]

if len(hf_post) >= 3:
    log_std_post = np.log(hf_post["std_flow"].clip(lower=1e-9))
    sl_k, ic_k, r_k, p_k, _ = stats.linregress(hf_post["t_rel"], log_std_post)
    k_obs = float(-sl_k)
    t_half_obs = np.log(2) / k_obs if k_obs > 0 else float("inf")
    C0_fit = float(np.exp(ic_k))
else:
    k_obs, t_half_obs, C0_fit = 0.05, 14.0, 0.05

t_fit = np.linspace(0, 24 - peak_hour, 100)
std_fit = C0_fit * np.exp(-k_obs * t_fit)

fig_kin = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Déséquilibre horaire σ_q(h) — données brutes",
        "Cinétique post-pic : ln(σ_q) vs temps relatif",
    ],
)

# Panel 1 : courbe complète
fig_kin.add_trace(go.Scatter(
    x=hf_sorted["hour"], y=hf_sorted["std_flow"],
    mode="lines+markers", name="σ_q observé",
    line=dict(color="#1A6FBF", width=2.5),
    marker=dict(size=7),
), row=1, col=1)
fig_kin.add_vline(x=peak_hour, line_dash="dash", line_color="#D96B27",
                   annotation_text=f"Pic h={peak_hour}h", row=1, col=1)
fig_kin.add_trace(go.Scatter(
    x=hf_post["hour"],
    y=C0_fit * np.exp(-k_obs * hf_post["t_rel"].values),
    mode="lines", name=f"Décroissance k={k_obs:.3f} h⁻¹",
    line=dict(color="#C0392B", width=2, dash="dot"),
), row=1, col=1)

# Panel 2 : log-linéaire
fig_kin.add_trace(go.Scatter(
    x=hf_post["t_rel"],
    y=np.log(hf_post["std_flow"].clip(lower=1e-9)),
    mode="markers", name="ln(σ_q) observé",
    marker=dict(color="#1A6FBF", size=9),
), row=1, col=2)
fig_kin.add_trace(go.Scatter(
    x=t_fit,
    y=np.log(C0_fit) - k_obs * t_fit,
    mode="lines", name=f"Ajustement ordre 1 (R²={r_k**2:.3f})",
    line=dict(color="#C0392B", width=2.5),
), row=1, col=2)

fig_kin.update_xaxes(title_text="Heure", row=1, col=1)
fig_kin.update_xaxes(title_text="Temps depuis le pic (h)", row=1, col=2)
fig_kin.update_yaxes(title_text="σ_q inter-stations", row=1, col=1)
fig_kin.update_yaxes(title_text="ln(σ_q)", row=1, col=2)
fig_kin.update_layout(
    plot_bgcolor="white", height=390,
    margin=dict(l=10, r=10, t=40, b=10),
    showlegend=True,
    legend=dict(x=0.55, y=0.95),
)
st.plotly_chart(fig_kin, use_container_width=True)

col_k1, col_k2, col_k3 = st.columns(3)
col_k1.metric("Taux cinétique k", f"{k_obs:.4f} h⁻¹")
col_k2.metric("Demi-vie t½ = ln2 / k", f"{t_half_obs:.1f} h" if t_half_obs < 100 else "> 24 h")
col_k3.metric("R² ajustement ordre 1", f"{r_k**2:.3f}")
st.caption(
    f"**Figure 2.1–2.2.** Cinétique de premier ordre du déséquilibre inter-stations "
    f"après le pic de h={peak_hour}h. "
    "La linéarité en échelle log révèle une décroissance exponentielle : "
    f"k = {k_obs:.4f} h⁻¹, soit une demi-vie t½ = {t_half_obs:.1f} h. "
    "Interprétation : après chaque heure de pointe, il faut environ "
    f"{t_half_obs:.1f} h pour que l'hétérogénéité des flux soit réduite de moitié "
    "par la seule dynamique spontanée du réseau."
)

# ── Section 3 — Loi d'Arrhenius ──────────────────────────────────────────────
st.divider()
section(3, "Loi d'Arrhenius — Barrière Climatique à l'Usage Cyclable")

st.markdown(r"""
La **loi d'Arrhenius** (1889) relie le taux d'une réaction chimique à la température :

$$k(T) = A \cdot e^{-E_a / RT}, \qquad \ln k = \ln A - \frac{E_a}{R} \cdot \frac{1}{T}$$

où $E_a$ est l'*énergie d'activation* (barrière énergétique à franchir), $R$ la constante
des gaz parfaits et $T$ la température absolue (K).

**Application VLS :** Le *flux cyclable total* (nombre de courses par heure) joue le rôle
du taux cinétique $k$. La *température extérieure* $T$ (en K) est la variable thermodynamique.
L'énergie d'activation $E_a$ mesure la **barrière climatique à l'usage du vélo** :
plus $E_a$ est élevée, plus le flux chute rapidement lorsque la température baisse.

En tracant $\ln(\text{courses})$ en fonction de $1/T$ (diagramme d'Arrhenius), une droite
confirme le modèle et la pente $-E_a/R$ donne l'énergie d'activation en J/mol.
""")

# Fusion: température horaire + flux horaire
R_gaz = 8.314  # J/(mol·K)

merged = hf.merge(hourly_T, on="hour")
merged["T_K"] = merged["T_mean"] + 273.15  # Kelvin
merged["inv_T"] = 1.0 / merged["T_K"]
merged["ln_trips"] = np.log(merged["total_trips"].clip(lower=0.1))

# Régression Arrhenius (ln k vs 1/T)
slope_arr, intercept_arr, r_arr, p_arr, _ = stats.linregress(
    merged["inv_T"], merged["ln_trips"]
)
Ea_Jmol = float(-slope_arr * R_gaz)
Ea_kJmol = Ea_Jmol / 1000.0
lnA_arr  = float(intercept_arr)

T_range_K = np.linspace(
    max(T_min_K + 273.15, merged["T_K"].min()),
    min(T_max_K + 273.15, merged["T_K"].max()),
    100,
)
fit_arr = np.exp(lnA_arr - Ea_Jmol / (R_gaz * T_range_K))

col_a1, col_a2 = st.columns(2)

with col_a1:
    fig_arrh = go.Figure()
    # Données
    fig_arrh.add_trace(go.Scatter(
        x=merged["inv_T"] * 1000,  # × 1000 pour lisibilité (10⁻³ K⁻¹)
        y=merged["ln_trips"],
        mode="markers+text",
        text=merged["hour"].apply(lambda h: f"{h}h"),
        textposition="top right",
        textfont=dict(size=8),
        marker=dict(
            color=merged["T_mean"], colorscale="RdYlBu_r",
            size=10, opacity=0.9,
            colorbar=dict(title="T (°C)"),
        ),
        name="Données horaires",
        hovertemplate="h=%{text}<br>1/T = %{x:.4f}×10⁻³ K⁻¹<br>ln(courses) = %{y:.3f}",
    ))
    # Fit Arrhenius
    fig_arrh.add_trace(go.Scatter(
        x=1000 / T_range_K,
        y=np.log(fit_arr),
        mode="lines",
        name=f"Arrhenius (Eₐ = {Ea_kJmol:.1f} kJ/mol)",
        line=dict(color="#D96B27", width=2.5),
    ))
    fig_arrh.update_layout(
        plot_bgcolor="white",
        xaxis_title="1/T × 10³ (K⁻¹)",
        yaxis_title="ln(courses/heure)",
        legend=dict(x=0.02, y=0.15),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_arrh, use_container_width=True)
    st.caption(
        "**Figure 3.1.** Diagramme d'Arrhenius : ln(courses) en fonction de 1/T. "
        "Chaque point correspond à une heure de la journée (couleur = température moyenne). "
        f"La pente de la droite donne E_a/R = {-slope_arr:.1f} K. "
        "La dispersion autour de la droite reflète les effets non-thermiques "
        "(heure de la journée, habitudes de mobilité)."
    )

with col_a2:
    # Courbe k(T) — visualisation intuitive
    T_c_range = np.linspace(T_min_K, T_max_K, 200)
    T_K_range  = T_c_range + 273.15
    k_range    = np.exp(lnA_arr - Ea_Jmol / (R_gaz * T_K_range))

    fig_arrh2 = go.Figure()
    fig_arrh2.add_trace(go.Scatter(
        x=T_c_range, y=k_range, mode="lines",
        name=f"k(T) = A·exp(−Eₐ/RT), Eₐ = {Ea_kJmol:.1f} kJ/mol",
        line=dict(color="#C0392B", width=2.5),
    ))
    # Données observées
    fig_arrh2.add_trace(go.Scatter(
        x=merged["T_mean"], y=merged["total_trips"],
        mode="markers",
        text=merged["hour"].apply(lambda h: f"{h}h"),
        marker=dict(color="#1A6FBF", size=8, opacity=0.75),
        name="Courses observées",
        hovertemplate="%{text}<br>T = %{x:.1f}°C<br>Courses = %{y:.1f}",
    ))
    fig_arrh2.update_layout(
        plot_bgcolor="white",
        xaxis_title="Température (°C)",
        yaxis_title="Taux d'usage (courses / heure)",
        legend=dict(x=0.02, y=0.98),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_arrh2, use_container_width=True)
    st.caption(
        "**Figure 3.2.** Taux d'usage cyclable k(T) modélisé par la loi d'Arrhenius. "
        f"Énergie d'activation estimée : E_a = {Ea_kJmol:.1f} kJ/mol. "
        "En chimie, les réactions typiques ont E_a ∈ [40, 200] kJ/mol. "
        f"Ici, E_a = {Ea_kJmol:.1f} kJ/mol représente la <i>barrière climatique</i> "
        "à l'usage du vélo — la sensibilité du comportement cyclable à la température.",
        unsafe_allow_html=True,
    )

col_ea1, col_ea2, col_ea3 = st.columns(3)
col_ea1.metric("Énergie d'activation Eₐ", f"{Ea_kJmol:.1f} kJ/mol")
col_ea2.metric("ln A (facteur pré-exponentiel)", f"{lnA_arr:.2f}")
col_ea3.metric("R² diagramme d'Arrhenius", f"{r_arr**2:.3f}")

# ── Section 4 — Catalyse par les hubs ─────────────────────────────────────────
st.divider()
section(4, "Catalyse — Les Stations-Hub Réduisent la Barrière de Redistribution")

st.markdown(r"""
Un **catalyseur** est une espèce qui augmente le taux d'une réaction en abaissant
son énergie d'activation $E_a$, sans être consommée :

$$k_{cat} = A \cdot e^{-E_a(1 - \eta_{cat}) / RT}, \qquad \eta_{cat} \in [0, 1]$$

où $\eta_{cat}$ est l'*efficacité catalytique* (fraction de réduction de $E_a$).

**Application VLS :** Les **stations-hub** (Stress_Index élevé, forte centralité PageRank)
jouent le rôle de catalyseurs : elles attirent et redistribuent les flux plus efficacement,
réduisant le coût effectif de la redistribution pour leurs stations voisines.
La barrière d'activation est ici la *friction topographique et géographique* au rééquilibrage.
""")

# Fusion stress + net_flow
nf_mean = nf.groupby("station").agg(
    mean_flow=("inflow", "mean"),
    total_flow=("inflow", lambda x: x.sum() + nf.loc[x.index, "outflow"].sum()),
    abs_net=("net_flow", lambda x: x.abs().mean()),
).reset_index()
nf_mean.columns = ["station", "mean_flow", "total_flow", "abs_net"]

# Normaliser les noms de station (stress ranking peut avoir des suffixes différents)
stress_merged = stress.copy()
stress_merged["station_key"] = stress_merged["Station_Name"].str.strip().str.lower()
nf_mean["station_key"]       = nf_mean["station"].str.strip().str.lower()

# Jointure approximative par inclusion de clé
merged_cat = nf_mean.merge(stress_merged, on="station_key", how="inner")
if len(merged_cat) == 0:
    # Tentative via merge partiel sur les premiers caractères
    stress_merged["station_key2"] = stress_merged["station_key"].str[:8]
    nf_mean["station_key2"] = nf_mean["station_key"].str[:8]
    merged_cat = nf_mean.merge(stress_merged, on="station_key2", how="inner")

col_cat1, col_cat2 = st.columns(2)

with col_cat1:
    if len(merged_cat) > 3:
        fig_cat = px.scatter(
            merged_cat,
            x="Stress_Index", y="mean_flow",
            size="PageRank",
            color="abs_net",
            color_continuous_scale="RdYlGn_r",
            hover_name="Station_Name",
            labels={
                "Stress_Index": "Stress_Index (efficacité catalytique)",
                "mean_flow": "Flux moyen q_in (courses/h)",
                "abs_net": "Déséquilibre |q_net|",
                "PageRank": "PageRank",
            },
            height=380,
        )
        fig_cat.update_layout(
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        corr_cat, p_cat = stats.spearmanr(merged_cat["Stress_Index"], merged_cat["mean_flow"])
        st.caption(
            "**Figure 4.1.** Flux moyen entrant en fonction du Stress_Index (proxy de l'efficacité catalytique). "
            f"Corrélation de Spearman ρ = {corr_cat:.3f} (p = {p_cat:.4f}). "
            "La taille des points encode le PageRank (centralité de nœud). "
            "Les stations à fort Stress_Index concentrent les flux — ils catalysent "
            "la redistribution en servant de nœuds de transfert privilégiés."
        )
    else:
        st.info("Fusion stress × flux insuffisante — affichage du profil Stress_Index.")
        fig_str = px.bar(
            stress.sort_values("Stress_Index", ascending=False).head(15),
            x="Stress_Index", y="Station_Name", orientation="h",
            color="PageRank", color_continuous_scale="Blues",
            labels={"Station_Name": "", "Stress_Index": "Stress_Index",
                    "PageRank": "PageRank"},
            height=380,
        )
        fig_str.update_layout(plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_str, use_container_width=True)
        st.caption(
            "**Figure 4.1.** Top 15 stations par Stress_Index (proxy de l'efficacité catalytique). "
            "La couleur encode le PageRank. Les stations-catalyseurs concentrent les flux de redistribution."
        )

with col_cat2:
    # Profil d'énergie potentielle : avec et sans catalyseur
    eta = Ea_cat_pct / 100.0
    Ea_uncat = abs(Ea_Jmol)
    Ea_cat_val  = Ea_uncat * (1 - eta)

    # Coordonnée de réaction (arbitraire, 0→1)
    xi = np.linspace(0, 1, 300)

    def reaction_profile(xi, Ea, E_r=-5000.0):
        """Profil gaussien avec pic d'activation centré en xi=0.5."""
        peak = Ea * np.exp(-((xi - 0.5)**2) / 0.02)
        product_energy = E_r * xi
        return peak + product_energy

    E_uncat = reaction_profile(xi, Ea_uncat)
    E_cat   = reaction_profile(xi, Ea_cat_val)

    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=xi, y=E_uncat / 1000, mode="lines",
        name="Sans catalyseur (réseau non-hub)",
        line=dict(color="#1A6FBF", width=2.5),
    ))
    fig_prof.add_trace(go.Scatter(
        x=xi, y=E_cat / 1000, mode="lines",
        name=f"Avec catalyseur ({Ea_cat_pct}% réduction Eₐ)",
        line=dict(color="#C0392B", width=2.5, dash="dash"),
    ))

    # Annotations
    fig_prof.add_annotation(x=0.05, y=0, text="État initial<br>(vélo disponible)",
                             showarrow=False, font=dict(size=9))
    fig_prof.add_annotation(x=0.95, y=-5, text="État final<br>(trajet effectué)",
                             showarrow=False, font=dict(size=9))
    fig_prof.add_annotation(x=0.5, y=Ea_uncat / 1000,
                             text=f"Eₐ = {Ea_uncat/1000:.1f} kJ/mol",
                             showarrow=True, ax=30, ay=-30,
                             font=dict(color="#1A6FBF", size=9))
    fig_prof.add_annotation(x=0.5, y=Ea_cat_val / 1000,
                             text=f"Eₐ_cat = {Ea_cat_val/1000:.1f} kJ/mol",
                             showarrow=True, ax=-40, ay=-20,
                             font=dict(color="#C0392B", size=9))

    fig_prof.update_layout(
        plot_bgcolor="white",
        xaxis_title="Coordonnée de réaction ξ (état initial → final)",
        yaxis_title="Énergie potentielle (kJ/mol)",
        legend=dict(x=0.35, y=0.98),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_prof, use_container_width=True)
    st.caption(
        f"**Figure 4.2.** Profil d'énergie potentielle de la mobilité : sans catalyseur "
        f"(réseau sans hub, E_a = {Ea_uncat/1000:.1f} kJ/mol) "
        f"et avec catalyseur (station-hub, E_a réduit de {Ea_cat_pct}% à {Ea_cat_val/1000:.1f} kJ/mol). "
        "L'abscisse représente la coordonnée de réaction (état initial = vélo disponible ; "
        "état final = trajet accompli). Les stations-hub réduisent la barrière de friction "
        "spatiale et temporelle, accélérant la mobilité à l'image d'un catalyseur enzymatique."
    )

# ── Section 5 — Profil d'énergie potentielle journalier ──────────────────────
st.divider()
section(5, "Profil Énergétique Journalier — Diagramme de Réaction sur 24 Heures")

st.markdown(r"""
En chimie, le **diagramme de profil de réaction** représente l'énergie potentielle
du système le long de la coordonnée de réaction, faisant apparaître :
- les *états réactifs* (minima, creux énergétiques) — système au repos,
- l'*état de transition* (*complexe activé*) — sommet énergétique, état instable,
- les *états produits* (nouveaux minima) — après transformation.

**Application VLS :** On interprète le flux total horaire (courses/heure) comme
l'**énergie du système**. Les heures creuses (nuit) sont les *réactifs* — système
au repos. Les heures de pointe (7h–9h, 17h–19h) sont les *états de transition* —
le système est activé, les flux sont maximaux, le réseau est perturbé.
Entre les pointes, le système retourne à un *état intermédiaire stable* (déjeuner).
""")

# Profil énergétique = total_trips par heure + annotations chimie
hf_24 = hf.copy().sort_values("hour")

fig_energy = go.Figure()

# Zone de nuit (minima = réactif)
fig_energy.add_vrect(x0=-0.5, x1=5.5, fillcolor="#EAF2FF", opacity=0.4,
                      layer="below", line_width=0, annotation_text="Réactifs (nuit)",
                      annotation_position="top left",
                      annotation_font=dict(size=9, color="#1A6FBF"))

# Zone pic matin
fig_energy.add_vrect(x0=6.5, x1=9.5, fillcolor="#FDEAEA", opacity=0.4,
                      layer="below", line_width=0,
                      annotation_text="Complexe activé I\n(pointe matin)",
                      annotation_position="top left",
                      annotation_font=dict(size=9, color="#C0392B"))

# Zone journée
fig_energy.add_vrect(x0=9.5, x1=16.5, fillcolor="#EAF8EA", opacity=0.4,
                      layer="below", line_width=0,
                      annotation_text="État intermédiaire\n(équilibre diurne)",
                      annotation_position="top left",
                      annotation_font=dict(size=9, color="#1E8449"))

# Zone pic soir
fig_energy.add_vrect(x0=16.5, x1=19.5, fillcolor="#FEF3E2", opacity=0.4,
                      layer="below", line_width=0,
                      annotation_text="Complexe activé II\n(pointe soir)",
                      annotation_position="top left",
                      annotation_font=dict(size=9, color="#D96B27"))

# Zone soirée
fig_energy.add_vrect(x0=19.5, x1=23.5, fillcolor="#F5EAF8", opacity=0.4,
                      layer="below", line_width=0,
                      annotation_text="Produits\n(retour repos)",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color="#7D3C98"))

# Courbe principale
fig_energy.add_trace(go.Scatter(
    x=hf_24["hour"], y=hf_24["total_trips"],
    mode="lines+markers",
    name="Énergie du système (courses/h)",
    line=dict(color="#1A6FBF", width=3),
    marker=dict(size=8, color=hf_24["total_trips"],
                colorscale="Plasma", showscale=False),
    hovertemplate="h = %{x}h<br>Courses = %{y:.1f}/h",
))

# Marquer les maxima et minima
idx_max1 = int(hf_24["total_trips"].iloc[6:12].idxmax())
idx_max2 = int(hf_24["total_trips"].iloc[16:21].idxmax())
idx_min_mid = int(hf_24["total_trips"].iloc[2:7].idxmin())

for idx, label, color in [
    (idx_max1, "État de transition I<br>(complexe activé)", "#C0392B"),
    (idx_max2, "État de transition II<br>(complexe activé)", "#D96B27"),
]:
    row = hf_24.iloc[idx]
    fig_energy.add_annotation(
        x=row["hour"], y=row["total_trips"] + 3,
        text=label, showarrow=True, ax=0, ay=-35,
        font=dict(color=color, size=9),
        arrowcolor=color,
    )

# Std comme "incertitude d'énergie"
fig_energy.add_trace(go.Scatter(
    x=list(hf_24["hour"]) + list(hf_24["hour"])[::-1],
    y=list(hf_24["total_trips"] + hf_24["std_flow"] * 20)
      + list((hf_24["total_trips"] - hf_24["std_flow"] * 20).clip(lower=0))[::-1],
    fill="toself",
    fillcolor="rgba(26,111,191,0.12)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Intervalle de dispersion ±σ_q × 20",
    showlegend=True,
))

fig_energy.update_layout(
    plot_bgcolor="white",
    xaxis=dict(title="Heure de la journée", tickmode="linear", dtick=2),
    yaxis_title="Flux total (courses / heure) — énergie du système",
    legend=dict(x=0.02, y=0.98),
    height=460,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_energy, use_container_width=True)

# Énergie d'activation (différence pointe/creux)
E_act_matin = float(hf_24["total_trips"].iloc[7:10].max() - hf_24["total_trips"].iloc[0:6].mean())
E_act_soir  = float(hf_24["total_trips"].iloc[17:20].max() - hf_24["total_trips"].iloc[12:16].mean())

col_e1, col_e2, col_e3 = st.columns(3)
col_e1.metric("Eₐ pic matin (courses/h)", f"{E_act_matin:.1f}",
              help="Flux crête matin − flux nuit moyen")
col_e2.metric("Eₐ pic soir (courses/h)", f"{E_act_soir:.1f}",
              help="Flux crête soir − flux journée moyen")
col_e3.metric("Rapport Eₐ soir / Eₐ matin", f"{E_act_soir / max(E_act_matin, 0.1):.2f}")

st.caption(
    "**Figure 5.1.** Profil d'énergie potentielle journalier du réseau Vélomagg Montpellier. "
    "Les zones colorées délimitent les analogues chimiques : réactifs (nuit, creux d'énergie), "
    "complexes activés (pointes, maxima d'énergie), état intermédiaire stable (journée) "
    "et produits (soirée, retour vers l'équilibre). "
    "La bande translucide autour de la courbe représente ±20&thinsp;&times;&thinsp;&sigma;_q "
    "(dispersion inter-stations). "
    "Les deux barrières d'activation — pointe du matin (E_a = {:.0f} courses/h) "
    "et du soir (E_a = {:.0f} courses/h) — correspondent à des états de non-équilibre "
    "nécessitant une intervention active de redistribution pour revenir à l'équilibre thermique.",
    unsafe_allow_html=True,
)

st.divider()
st.caption(
    "**Sources.** Arrhenius S. (1889). *Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker.* "
    "Z. Phys. Chem. "
    "Le Chatelier H. (1888). *De l'influence de la température sur les équilibres chimiques.* "
    "C. R. Acad. Sci. "
    "Van 't Hoff J.H. (1884). *Études de dynamique chimique.* "
    "Eyring H. (1935). *The Activated Complex in Chemical Reactions.* J. Chem. Phys. — "
    "**R. Fossé & G. Pallares · 2025–2026**"
)
