"""
12_Biologie.py — Biologie appliquée aux réseaux VLS : allométrie, biodiversité, épidémiologie,
croissance logistique, théorie des niches écologiques.
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
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_stations
from utils.styles import abstract_box, inject_css, section, sidebar_nav

st.set_page_config(
    page_title="Biologie - Réseaux VLS",
    page_icon=None,
    layout="wide",
)
inject_css()

# ── Données ──────────────────────────────────────────────────────────────────
_NON_CITY = {"France", "FR", "Grand Est", "Basque Country"}

@st.cache_data(ttl=3600)
def load_dock_stations() -> pd.DataFrame:
    df = load_stations()
    return (
        df
        .query('station_type == "docked_bike"')
        .dropna(subset=["capacity", "lat", "lon", "city"])
        .query("capacity > 0")
        .loc[lambda x: ~x["city"].isin(_NON_CITY)]
        .copy()
    )

@st.cache_data(ttl=3600)
def compute_city_agg(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
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
    # Population proxy : approximation par capacité totale + population INSEE si disponible
    return agg

@st.cache_data(ttl=3600)
def compute_biodiversity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Indices de diversité par ville : Shannon, Simpson, richesse, équitabilité de Pielou."""
    rows = []
    for city, grp in df.groupby("city"):
        if len(grp) < 5:
            continue
        caps = grp["capacity"].values.astype(float)
        # Classes de capacité (catégories = "espèces" en écologie)
        bins = [0, 5, 10, 15, 20, 30, 50, 200]
        labels = ["1-5", "6-10", "11-15", "16-20", "21-30", "31-50", ">50"]
        cat = pd.cut(caps, bins=bins, labels=labels)
        counts = cat.value_counts()
        counts = counts[counts > 0]
        total = counts.sum()
        p = counts / total

        # Shannon H'
        H_shannon = float(-np.sum(p * np.log(p)))
        # Richesse S (nombre de classes occupées)
        S = len(p)
        # Équitabilité de Pielou J = H' / ln(S)
        J_pielou = H_shannon / np.log(S) if S > 1 else 0.0
        # Index de Simpson D = 1 - sum(n*(n-1)/N*(N-1))
        n_arr = counts.values.astype(float)
        N = n_arr.sum()
        D_simpson = float(1 - np.sum(n_arr * (n_arr - 1)) / (N * (N - 1))) if N > 1 else 0.0

        rows.append({
            "city": city,
            "n": len(grp),
            "richesse_S": S,
            "H_shannon": H_shannon,
            "J_pielou": J_pielou,
            "D_simpson": D_simpson,
            "total_cap": float(caps.sum()),
            "mean_cap": float(caps.mean()),
        })
    return pd.DataFrame(rows).sort_values("H_shannon", ascending=False)

df_all = load_dock_stations()
city_agg = compute_city_agg(df_all)
df_bio = compute_biodiversity_metrics(df_all)

_n_stations = len(df_all)
_n_cities = df_all["city"].nunique()
_H_bio_med = df_bio["H_shannon"].median()

st.title("Biologie et Écologie des Réseaux de Vélos en Libre-Service")
st.caption("Exploration transversale : allométrie, biodiversité, épidémiologie, croissance logistique")

abstract_box(
    "<b>Cadre théorique :</b> La biologie théorique fournit des modèles mathématiques "
    "applicables aux systèmes urbains. L'<i>allométrie</i> de West, Brown & Enquist (1997) "
    "prédit que les grandeurs biologiques varient en loi de puissance avec la masse — "
    "Bettencourt (2007) a étendu ce résultat aux villes. La <i>théorie de la biodiversité</i> "
    "(indice de Shannon, Simpson, Pielou) caractérise la diversité des écosystèmes ; "
    "appliquée aux réseaux VLS, elle mesure l'hétérogénéité des classes de capacité. "
    "Le <i>modèle logistique</i> de Verhulst (1838) et les <i>modèles épidémiques</i> SIR "
    "complètent ce cadre pour décrire la croissance et la propagation des usages de mobilité douce.",
    findings=[
        (f"{_n_stations:,}", "stations dock-based"),
        (f"{_n_cities}", "agglomérations"),
        (f"H' = {_H_bio_med:.3f}", "diversité médiane"),
        ("7 classes", "de capacité"),
    ],
)

sidebar_nav()
with st.sidebar:
    st.header("Paramètres")
    top_n_allo = st.slider("Nombre de villes (allométrie)", 20, 59, 40, 5)
    city_sir = st.selectbox(
        "Ville pour le modèle SIR",
        options=sorted(df_all["city"].unique()),
        index=sorted(df_all["city"].unique()).index("Montpellier")
        if "Montpellier" in df_all["city"].unique() else 0,
    )
    st.divider()
    st.caption("R. Fossé & G. Pallares · 2025–2026")

# ── Section 1 — Biodiversité écologique ──────────────────────────────────────
st.divider()
section(1, "Biodiversité Écologique — Diversité des Classes de Capacité")

st.markdown(r"""
En **écologie**, la diversité d'un écosystème est quantifiée par des indices qui mesurent
à la fois la *richesse* (nombre d'espèces) et l'*équitabilité* (répartition des abondances).

Appliqués aux réseaux VLS, on définit les **classes de capacité** comme des *espèces fonctionnelles* :
petites stations (1–5 emplacements), moyennes (6–20), grandes (> 20).

- **Indice de Shannon** $H' = -\sum_i p_i \ln p_i$ : diversité globale
- **Équitabilité de Pielou** $J = H' / \ln S$ : uniformité de la distribution entre classes
- **Indice de Simpson** $D = 1 - \sum_i p_i^2$ : probabilité que deux stations tirées au sort soient de classes différentes
""")

col1, col2 = st.columns(2)

with col1:
    # Top 20 villes par indice de Shannon
    top20_h = df_bio.head(20).sort_values("H_shannon", ascending=True)
    fig_shannon = px.bar(
        top20_h, x="H_shannon", y="city",
        orientation="h",
        color="J_pielou",
        color_continuous_scale="Greens",
        labels={"H_shannon": "Indice de Shannon H'", "city": "",
                "J_pielou": "Équitabilité J"},
        height=420,
    )
    fig_shannon.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_shannon, use_container_width=True)
    st.caption(
        "**Figure 1.1.** Top 20 agglomérations par indice de Shannon H' (diversité des classes de capacité). "
        "La couleur encode l'équitabilité de Pielou J : une valeur proche de 1 indique "
        "une répartition uniforme entre classes — réseau *maximalement diversifié*. "
        "H' élevé correspond à une palette variée de types de stations dans le réseau."
    )

with col2:
    # Diagramme de Whittaker (J vs richesse S)
    fig_whit = px.scatter(
        df_bio, x="richesse_S", y="J_pielou",
        size="n", color="D_simpson",
        hover_name="city",
        color_continuous_scale="Viridis",
        labels={
            "richesse_S": "Richesse S (nombre de classes occupées)",
            "J_pielou": "Équitabilité de Pielou J",
            "D_simpson": "Index Simpson D",
            "n": "N stations",
        },
        height=420,
    )
    fig_whit.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_whit, use_container_width=True)
    st.caption(
        "**Figure 1.2.** Diagramme de diversité : équitabilité J en fonction de la richesse S. "
        "Taille &prop; nombre de stations, couleur = index de Simpson D. "
        "Un réseau avec S&thinsp;=&thinsp;7 et J&thinsp;&approx;&thinsp;1 serait *l'écosystème VLS le plus diversifié*. "
        "La plupart des réseaux n'occupent que 4–6 classes sur 7, révélant des *niches vides*.",
        unsafe_allow_html=True,
    )

# Radar des indices — top 6 villes
top6_cities = df_bio.nlargest(6, "total_cap")["city"].tolist()
radar_df = df_bio[df_bio["city"].isin(top6_cities)].copy()

cols_radar = ["H_shannon", "J_pielou", "D_simpson"]
norms = {
    "H_shannon": df_bio["H_shannon"].max(),
    "J_pielou": 1.0,
    "D_simpson": 1.0,
}
labels_radar = ["Shannon H'", "Pielou J", "Simpson D"]

fig_radar = go.Figure()
for _, row in radar_df.iterrows():
    vals = [row[c] / norms[c] for c in cols_radar]
    vals_closed = vals + [vals[0]]
    labels_closed = labels_radar + [labels_radar[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed, theta=labels_closed,
        fill="toself", name=row["city"], opacity=0.7,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    height=350,
    margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig_radar, use_container_width=True)
st.caption(
    "**Figure 1.3.** Graphique radar des indices de biodiversité normalisés "
    "pour les 6 agglomérations à plus grande capacité totale. "
    "Chaque axe est normalisé par le maximum national. "
    "Un réseau *idéalement diversifié* occuperait tout le disque."
)

# ── Section 2 — Allométrie ────────────────────────────────────────────────────
st.divider()
section(2, "Allométrie Urbaine — Loi de Puissance Taille-Réseau")

st.markdown(r"""
L'**allométrie** (du grec *allos* = autre, *metron* = mesure) désigne les relations
de type $Y = a \cdot X^b$ entre une grandeur biologique $Y$ et la taille $X$ d'un organisme.
West, Brown & Enquist (1997) ont montré que des grandeurs comme le métabolisme de base,
la fréquence cardiaque ou la durée de vie suivent des lois de puissance avec la masse :
$b = 3/4$, $b = -1/4$, $b = 1/4$ respectivement.

**Bettencourt et al. (2007)** ont étendu ce résultat aux villes : le PIB, les crimes,
les brevets varient comme $\text{Population}^{\beta}$ avec $\beta > 1$ (*supra-linéaire*,
économies d'agglomération) et les infrastructures comme $\beta < 1$ (*sous-linéaire*,
efficacité d'échelle).

**Question :** La capacité totale d'un réseau VLS suit-elle une loi allométrique
avec le nombre de stations (taille du réseau) ? L'exposant $\beta$ révèle si les
grandes villes ont des stations proportionnellement plus grandes (*supra-linéaire*)
ou investissent dans la quantité plutôt que la capacité individuelle (*sous-linéaire*).
""")

allo_df = city_agg.nlargest(top_n_allo, "total_cap").copy()

log_n = np.log10(allo_df["n_stations"])
log_cap = np.log10(allo_df["total_cap"])
slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(log_n, log_cap)

log_n_mc = np.log10(allo_df["n_stations"])
log_mc = np.log10(allo_df["mean_cap"])
slope_mc, intercept_mc, r_mc, p_mc, _ = stats.linregress(log_n_mc, log_mc)

col_a1, col_a2 = st.columns(2)

with col_a1:
    # C_tot vs N (loi allométrique principale)
    fig_allo1 = px.scatter(
        allo_df,
        x="n_stations", y="total_cap",
        log_x=True, log_y=True,
        text="city",
        size="mean_cap",
        color="std_cap",
        color_continuous_scale="RdYlGn_r",
        labels={
            "n_stations": "Nombre de stations N",
            "total_cap": "Capacité totale C (emplacements)",
            "std_cap": "Écart-type capacité",
            "mean_cap": "Capacité moyenne",
        },
        height=420,
    )
    # Courbe de fit
    n_range = np.logspace(np.log10(allo_df["n_stations"].min()),
                          np.log10(allo_df["n_stations"].max()), 80)
    fig_allo1.add_trace(go.Scatter(
        x=n_range, y=10**intercept_a * n_range**slope_a,
        mode="lines",
        name=f"C ∝ N<sup>{slope_a:.3f}</sup>  (R²={r_a**2:.3f})",
        line=dict(color="#D96B27", width=2.5),
    ))
    # Ligne de référence linéaire (β=1)
    C_ref = 10**intercept_a * n_range**1.0 * (10**intercept_a * allo_df["n_stations"].mean()**slope_a) / (10**intercept_a * allo_df["n_stations"].mean()**1.0)
    fig_allo1.add_trace(go.Scatter(
        x=n_range, y=10**intercept_a * n_range**1.0,
        mode="lines", name="Référence linéaire (β=1)",
        line=dict(color="#888888", width=1.5, dash="dot"),
    ))
    fig_allo1.update_traces(textposition="top center", selector=dict(mode="markers+text"),
                             textfont=dict(size=8))
    fig_allo1.update_layout(
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_allo1, use_container_width=True)
    st.caption(
        f"**Figure 2.1.** Allométrie VLS : capacité totale C en fonction du nombre de stations N (log-log). "
        f"Exposant allométrique &beta;&thinsp;=&thinsp;{slope_a:.3f} "
        f"({'supra-linéaire : grandes villes ont des stations proportionnellement plus grandes' if slope_a > 1 else 'sous-linéaire : efficacité dechelle, grandes villes investissent dans le nombre'}). "
        f"R²&thinsp;=&thinsp;{r_a**2:.3f}. "
        "La ligne pointillée indique la linéarité stricte (β=1). "
        "Interprétation biologique : β&thinsp;>&thinsp;1 est analogue au métabolisme cardiaque (supra-linéaire) "
        "des mammifères de grande taille.",
        unsafe_allow_html=True,
    )

with col_a2:
    # Capacité moyenne vs N (allométrie de la capacité individuelle)
    fig_allo2 = px.scatter(
        allo_df,
        x="n_stations", y="mean_cap",
        log_x=True, log_y=True,
        text="city",
        color="total_cap",
        color_continuous_scale="Blues",
        labels={
            "n_stations": "Nombre de stations N",
            "mean_cap": "Capacité moyenne ⟨c⟩ (empl/station)",
            "total_cap": "Capacité totale",
        },
        height=420,
    )
    n_range2 = np.logspace(np.log10(allo_df["n_stations"].min()),
                           np.log10(allo_df["n_stations"].max()), 80)
    fig_allo2.add_trace(go.Scatter(
        x=n_range2, y=10**intercept_mc * n_range2**slope_mc,
        mode="lines",
        name=f"⟨c⟩ ∝ N<sup>{slope_mc:.3f}</sup>  (R²={r_mc**2:.3f})",
        line=dict(color="#D96B27", width=2.5),
    ))
    fig_allo2.add_hline(y=float(allo_df["mean_cap"].mean()), line_dash="dot",
                        line_color="#888888",
                        annotation_text=f"Moyenne nationale = {allo_df['mean_cap'].mean():.1f}")
    fig_allo2.update_traces(textposition="top center", selector=dict(mode="markers+text"),
                             textfont=dict(size=8))
    fig_allo2.update_layout(
        plot_bgcolor="white",
        showlegend=True, legend=dict(x=0.02, y=0.98),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_allo2, use_container_width=True)
    st.caption(
        f"**Figure 2.2.** Allométrie de la capacité moyenne : &langle;c&rangle; en fonction de N (log-log). "
        f"Exposant &beta;'&thinsp;=&thinsp;{slope_mc:.3f} "
        f"({'positif : les grands réseaux ont des stations individuellement plus grandes' if slope_mc > 0 else 'négatif : les grands réseaux densifient sans agrandir les stations'}). "
        "En biologie, l'analogue est la relation entre la fréquence respiratoire et la taille corporelle "
        "(sous-linéaire, β&thinsp;=&thinsp;−1/4 chez Kleiber).",
        unsafe_allow_html=True,
    )

col_ma1, col_ma2, col_ma3 = st.columns(3)
col_ma1.metric("Exposant allométrique β (C_tot vs N)", f"{slope_a:.3f}",
               delta="supra-linéaire" if slope_a > 1 else "sous-linéaire")
col_ma2.metric("Exposant allométrique β' (⟨c⟩ vs N)", f"{slope_mc:.3f}")
col_ma3.metric("R² allométrie principale", f"{r_a**2:.3f}")

# ── Section 3 — Croissance logistique ────────────────────────────────────────
st.divider()
section(3, "Modèle de Croissance Logistique — Saturation des Réseaux VLS")

st.markdown(r"""
Le **modèle logistique de Verhulst** (1838) décrit la croissance d'une population
limitée par une capacité de charge $K$ (*carrying capacity*) :

$$\frac{dN}{dt} = r\, N\!\left(1 - \frac{N}{K}\right)$$

La solution est la courbe sigmoïde : $N(t) = \frac{K}{1 + e^{-r(t - t_0)}}$.

**Application :** La *densité de stations* par rapport à la surface de la ville
peut être modélisée comme une population en croissance. On ajuste le modèle logistique
sur la relation entre capacité totale et nombre de stations pour estimer la *capacité
de charge* d'un réseau VLS — i.e., la saturation maximale au-delà de laquelle
ajouter des stations n'augmente plus la capacité totale proportionnellement.
""")

# Modèle logistique : capacité totale en fonction du nombre de stations
def logistic(x, K, r, x0):
    return K / (1 + np.exp(-r * (x - x0)))

allo_sorted = allo_df.sort_values("n_stations").copy()
x_data = allo_sorted["n_stations"].values.astype(float)
y_data = allo_sorted["total_cap"].values.astype(float)

K0 = float(y_data.max() * 1.5)
r0 = 0.05
x0_0 = float(x_data.mean())

try:
    popt_log, _ = optimize.curve_fit(logistic, x_data, y_data,
                                      p0=[K0, r0, x0_0],
                                      maxfev=10000,
                                      bounds=([0, 0, 0], [np.inf, 10, np.inf]))
    K_log, r_log, x0_log = popt_log
    fit_ok = True
except Exception:
    fit_ok = False
    K_log, r_log, x0_log = K0, r0, x0_0

# Comparaison logistique vs linéaire
x_fit_log = np.linspace(0, x_data.max() * 1.5, 200)

fig_log = go.Figure()
fig_log.add_trace(go.Scatter(
    x=allo_sorted["n_stations"], y=allo_sorted["total_cap"],
    mode="markers+text",
    text=allo_sorted["city"],
    textposition="top center",
    textfont=dict(size=7),
    marker=dict(color="#1A6FBF", size=8, opacity=0.75),
    name="Villes observées",
    hovertemplate="%{text}<br>N = %{x:.0f} stations<br>C = %{y:.0f} empl",
))
if fit_ok:
    fig_log.add_trace(go.Scatter(
        x=x_fit_log, y=logistic(x_fit_log, K_log, r_log, x0_log),
        mode="lines",
        name=f"Modèle logistique (K={K_log:.0f}, r={r_log:.3f})",
        line=dict(color="#C0392B", width=2.5),
    ))
    fig_log.add_hline(y=K_log, line_dash="dash", line_color="#888888",
                      annotation_text=f"Capacité de charge K = {K_log:.0f} empl",
                      annotation_position="bottom right")

fig_log.add_trace(go.Scatter(
    x=x_fit_log, y=np.polyval(np.polyfit(x_data, y_data, 1), x_fit_log),
    mode="lines", name="Régression linéaire",
    line=dict(color="#D96B27", width=1.5, dash="dot"),
))

fig_log.update_layout(
    plot_bgcolor="white",
    xaxis_title="Nombre de stations N",
    yaxis_title="Capacité totale C (emplacements)",
    legend=dict(x=0.02, y=0.98),
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_log, use_container_width=True)

if fit_ok:
    col_l1, col_l2, col_l3 = st.columns(3)
    col_l1.metric("Capacité de charge K", f"{K_log:,.0f} empl",
                  help="Saturation maximale estimée du réseau national")
    col_l2.metric("Taux de croissance r", f"{r_log:.4f}")
    col_l3.metric("Point d'inflexion N₀", f"{x0_log:.0f} stations")

st.caption(
    "**Figure 3.1.** Ajustement du modèle de Verhulst sur la relation capacité totale – nombre de stations. "
    "La courbe sigmoïde prédit une **saturation** de la capacité totale au-delà d'une certaine taille de réseau : "
    "les grands réseaux atteignent une borne supérieure K (analogie avec la capacité de charge d'un écosystème). "
    "Le point d'inflexion N₀ correspond au réseau à croissance maximale — stade de l'expansion rapide. "
    "En biologie, cette saturation décrit le plafonnement des populations par les ressources disponibles."
)

# ── Section 4 — Modèle épidémique SIR ────────────────────────────────────────
st.divider()
section(4, "Modèle Épidémique SIR — Diffusion Spatiale de l'Usage VLS")

st.markdown(r"""
Le **modèle SIR** (Kermack & McKendrick, 1927) est le formalisme de base de l'épidémiologie :

$$\frac{dS}{dt} = -\beta \frac{SI}{N}, \quad \frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I, \quad \frac{dR}{dt} = \gamma I$$

- $S$ : *Susceptibles* — stations voisines sans vélos disponibles
- $I$ : *Infectés* — stations saturées ou vides (en déséquilibre)
- $R$ : *Rétablis* — stations rééquilibrées par redistribution

**Application spatiale :** On interprète chaque station Vélomagg comme un nœud.
Une station *saturée* (taux de remplissage élevé, proxy : grande capacité relative)
"contamine" ses voisines par la demande en créant un flux de déséquilibre.
Le taux de *guérison* $\gamma$ représente la capacité de redistribution de l'opérateur.

**Simulation :** On initialise quelques stations *infectées* (grandes stations centrales)
et on propage le déséquilibre selon une distance de contamination $d_c$.
""")

@st.cache_data(ttl=3600)
def run_sir_simulation(city_name: str, df: pd.DataFrame,
                        beta: float, gamma: float,
                        d_c_m: float, t_max: int) -> pd.DataFrame:
    from scipy.spatial import cKDTree
    grp = df[df["city"].str.contains(city_name, na=False, regex=False)].copy()
    if len(grp) < 5:
        return pd.DataFrame()
    grp = grp.reset_index(drop=True)
    n = len(grp)
    coords = grp[["lat", "lon"]].values
    caps = grp["capacity"].values.astype(float)
    tree = cKDTree(coords)

    # Matrice d'adjacence
    d_c_deg = d_c_m / 1000.0 / 111.0
    pairs = list(tree.query_pairs(d_c_deg))
    adj = {i: [] for i in range(n)}
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)

    # Initialisation : les stations de grande capacité sont "infectées"
    cap_thresh = np.percentile(caps, 75)
    state = np.array(["S"] * n, dtype=object)
    state[caps >= cap_thresh] = "I"

    results = []
    for t in range(t_max):
        S_count = int((state == "S").sum())
        I_count = int((state == "I").sum())
        R_count = int((state == "R").sum())
        results.append({"t": t, "S": S_count, "I": I_count, "R": R_count, "N": n})

        new_state = state.copy()
        for i in range(n):
            if state[i] == "S":
                neighbors_I = sum(1 for j in adj[i] if state[j] == "I")
                p_infect = 1 - (1 - beta)**neighbors_I
                if np.random.random() < p_infect:
                    new_state[i] = "I"
            elif state[i] == "I":
                if np.random.random() < gamma:
                    new_state[i] = "R"
        state = new_state
        if I_count == 0:
            break

    return pd.DataFrame(results)

st.markdown(f"**Simulation SIR pour {city_sir}**")
col_sir1, col_sir2, col_sir3 = st.columns(3)
beta_sir  = col_sir1.slider("Taux de contamination β", 0.05, 0.8, 0.3, 0.05,
                             help="Probabilité de déséquilibre par station voisine saturée")
gamma_sir = col_sir2.slider("Taux de rééquilibrage γ", 0.05, 0.5, 0.15, 0.05,
                             help="Probabilité de redistribution par l'opérateur")
d_c_sir   = col_sir3.slider("Distance de contamination (m)", 100, 800, 350, 50,
                             help="Rayon de propagation du déséquilibre")

with st.spinner("Simulation SIR en cours..."):
    np.random.seed(42)
    sir_df = run_sir_simulation(city_sir, df_all, beta_sir, gamma_sir, d_c_sir, t_max=60)

if not sir_df.empty:
    R0_sir = beta_sir / gamma_sir
    peak_I = int(sir_df["I"].max())
    peak_t = int(sir_df.loc[sir_df["I"].idxmax(), "t"])

    fig_sir = go.Figure()
    fig_sir.add_trace(go.Scatter(x=sir_df["t"], y=sir_df["S"], mode="lines",
                                  name="S — Stations équilibrées (susceptibles)",
                                  line=dict(color="#1A6FBF", width=2.5)))
    fig_sir.add_trace(go.Scatter(x=sir_df["t"], y=sir_df["I"], mode="lines",
                                  name="I — Stations en déséquilibre (infectées)",
                                  line=dict(color="#C0392B", width=2.5)))
    fig_sir.add_trace(go.Scatter(x=sir_df["t"], y=sir_df["R"], mode="lines",
                                  name="R — Stations rééquilibrées (rétablies)",
                                  line=dict(color="#1E8449", width=2.5)))
    fig_sir.add_vline(x=peak_t, line_dash="dash", line_color="#D96B27",
                      annotation_text=f"Pic I = {peak_I} stations (t={peak_t})")
    fig_sir.update_layout(
        plot_bgcolor="white",
        xaxis_title="Temps (pas de redistribution)",
        yaxis_title="Nombre de stations",
        legend=dict(x=0.55, y=0.95),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_sir, use_container_width=True)

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Nombre de reproduction R₀", f"{R0_sir:.2f}",
                  delta="épidémie" if R0_sir > 1 else "extinction",
                  delta_color="inverse")
    col_r2.metric("Pic de déséquilibre", f"{peak_I} stations", f"t={peak_t}")
    col_r3.metric("Taux d'attaque final", f"{sir_df.iloc[-1]['R'] / sir_df.iloc[0]['N']:.1%}")

    st.caption(
        f"**Figure 4.1.** Simulation SIR du déséquilibre de disponibilité VLS à **{city_sir}**. "
        f"R₀&thinsp;=&thinsp;&beta;/&gamma;&thinsp;=&thinsp;{R0_sir:.2f} : "
        f"{'le déséquilibre se propage (R₀ > 1)' if R0_sir > 1 else 'le déséquilibre est résorbé (R₀ < 1)'}. "
        f"Paramètres : &beta;&thinsp;=&thinsp;{beta_sir}, &gamma;&thinsp;=&thinsp;{gamma_sir}, "
        f"d_c&thinsp;=&thinsp;{d_c_sir}&thinsp;m. "
        "En épidémiologie, R₀ est le nombre moyen de personnes contaminées par un individu infecté. "
        "Ici, il mesure la propagation moyenne du déséquilibre de disponibilité entre stations voisines.",
        unsafe_allow_html=True,
    )
else:
    st.warning(f"Données insuffisantes pour {city_sir}.")

# ── Section 5 — Théorie des niches écologiques ────────────────────────────────
st.divider()
section(5, "Niches Écologiques — Spécialisation Fonctionnelle des Stations")

st.markdown(r"""
En écologie, le concept de **niche écologique** (Hutchinson, 1957) décrit l'hypervolume
multidimensionnel des conditions dans lesquelles une espèce peut survivre et se reproduire.
Deux espèces occupant la même niche entrent en *compétition interspécifique* (principe d'exclusion
compétitive de Gause).

**Application :** Les stations VLS peuvent être caractérisées par leur *niche fonctionnelle*
multidimensionnelle : capacité, accessibilité TC, infrastructure cyclable, sinistralité.
Des stations similaires sur toutes ces dimensions sont en *compétition fonctionnelle* —
leur co-occurrence dans un réseau est sous-optimale (redondance de couverture).

On projette les stations en 2D via une PCA pour visualiser la structure des niches fonctionnelles.
""")

_NICHE_COLS = ["capacity", "infra_cyclable_pct", "baac_accidents_cyclistes",
               "gtfs_heavy_stops_300m"]
available_cols = [c for c in _NICHE_COLS if c in df_all.columns]

if len(available_cols) >= 2:
    niche_df = df_all[available_cols + ["city"]].dropna().copy()
    # Standardisation
    from scipy.stats import zscore
    X = niche_df[available_cols].values.astype(float)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # PCA manuelle
    cov = np.cov(X_std.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    PC1 = X_std @ eigvecs[:, 0]
    PC2 = X_std @ eigvecs[:, 1]
    var1 = eigvals[0] / eigvals.sum() * 100
    var2 = eigvals[1] / eigvals.sum() * 100

    niche_df["PC1"] = PC1
    niche_df["PC2"] = PC2

    top_cities_niche = niche_df["city"].value_counts().nlargest(8).index.tolist()
    niche_top = niche_df[niche_df["city"].isin(top_cities_niche)].copy()
    niche_sample = niche_top.sample(min(2000, len(niche_top)), random_state=42)

    fig_pca = px.scatter(
        niche_sample, x="PC1", y="PC2",
        color="city", opacity=0.55, size_max=6,
        labels={
            "PC1": f"PC1 ({var1:.1f}% de variance)",
            "PC2": f"PC2 ({var2:.1f}% de variance)",
        },
        height=420,
    )
    fig_pca.update_traces(marker=dict(size=4))
    fig_pca.update_layout(
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(title="Agglomération"),
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption(
        f"**Figure 5.1.** Espace des niches fonctionnelles VLS (ACP, {len(available_cols)} dimensions). "
        f"PC1 explique {var1:.1f}% de la variance, PC2 {var2:.1f}%. "
        "Chaque point est une station ; la couleur indique l'agglomération. "
        "Le chevauchement entre agglomérations indique des stations aux niches fonctionnelles similaires — "
        "analogue à la compétition interspécifique en écologie. "
        "Des clusters bien séparés révèlent des *stratégies écologiques* différentes "
        "(réseau multimodal dense vs réseau de quartier à faible intensité TC)."
    )
else:
    st.info("Colonnes de niche fonctionnelle insuffisantes dans le dataset.")

st.divider()
st.caption(
    "**Sources.** Verhulst P.F. (1838). *Notice sur la loi que la population suit dans son accroissement.* "
    "Kermack W.O. & McKendrick A.G. (1927). *A Contribution to the Mathematical Theory of Epidemics.* "
    "West G., Brown J. & Enquist B. (1997). *A General Model for the Origin of Allometric Scaling Laws in Biology.* Science. "
    "Bettencourt L. et al. (2007). *Growth, Innovation, Scaling, and the Pace of Life in Cities.* PNAS. "
    "Hutchinson G.E. (1957). *Concluding Remarks.* Cold Spring Harbor Symposia. "
    "Shannon C.E. (1948). *A Mathematical Theory of Communication.* — "
    "**R. Fossé & G. Pallares · 2025–2026**"
)
