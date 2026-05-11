"""
00b_Audit_Mondial.py - Restitution de l'audit GBFS a l'echelle mondiale.

Presente la transferabilite de la taxonomie A1-A5 documentee sur les
123 systemes francais a un corpus de 1254 systemes non-francais issus
du catalogue canonique MobilityData (47 pays). Inclut la formalisation
empirique d'une 6e classe candidate (A6 zero-capacity dock).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.styles import abstract_box, inject_css, section, sidebar_nav

ROOT = Path(__file__).parent.parent
E5_DIR = ROOT / "papers" / "01_gold_standard" / "experiments" / "e5_europe"
E6_DIR = ROOT / "papers" / "01_gold_standard" / "experiments" / "e6_freefloating"

st.set_page_config(
    page_title="Audit GBFS mondial",
    page_icon=None,
    layout="wide",
)
inject_css()

st.title("Audit GBFS a l'echelle mondiale")
st.caption(
    "Audit exhaustif des 1509 systemes du catalogue canonique "
    "MobilityData (48 pays), avec formalisation de deux classes "
    "candidates (A6, A7) decouvertes lors de l'extension geographique"
)


# ─── Chargement des artefacts d'audit ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def _load_audit() -> tuple[pd.DataFrame, dict]:
    csv = E5_DIR / "massive_audit_results.csv"
    summary = E5_DIR / "massive_audit_summary.json"
    df = pd.read_csv(csv, dtype={"stations": "Int64"})
    # The CSV stores Python booleans which pandas reads as object/bool mix;
    # cast each flag column to clean Python bool with NaN as False.
    for col in ("a1_cars", "a2_placeholder", "a3_overcap_flag",
                "a4_perim_flag", "a5_macro_flag", "any_anomaly", "reachable"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: v is True or v == "True"
            )
    js = json.loads(summary.read_text(encoding="utf-8"))
    return df, js


@st.cache_data(ttl=3600)
def _load_freefloating_patterns() -> pd.DataFrame:
    f = E6_DIR / "freefloating_per_brand.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()


df_audit, audit_summary = _load_audit()
df_ff_brand = _load_freefloating_patterns()


# ─── Indicateurs synthese ──────────────────────────────────────────────────────
n_audited = audit_summary["total_audited"]
n_reachable = audit_summary["reachable"]
n_with_data = audit_summary["with_stations"]
n_flagged = audit_summary["flagged_anomalies"]
n_countries = sum(
    1 for v in audit_summary["by_country"].values() if v["audited"] > 0
)
n_a1 = sum(
    v["anomalies"].get("a1_cars", 0)
    for v in audit_summary["by_country"].values()
)
n_a2 = sum(
    v["anomalies"].get("a2_placeholder", 0)
    for v in audit_summary["by_country"].values()
)
n_a3 = sum(
    v["anomalies"].get("a3_overcap_flag", 0)
    for v in audit_summary["by_country"].values()
)
n_a4 = sum(
    v["anomalies"].get("a4_perim_flag", 0)
    for v in audit_summary["by_country"].values()
)
n_a5 = sum(
    v["anomalies"].get("a5_macro_flag", 0)
    for v in audit_summary["by_country"].values()
)


abstract_box(
    "<b>Question posee :</b> les cinq classes d'anomalie A1 a A5 "
    "documentees sur le corpus francais sont-elles des artefacts de "
    "publication francais, ou capturent-elles des features structurelles "
    "qui se transferent a d'autres juridictions ?<br><br>"
    "Le protocole d'audit a ete applique sans changement methodologique "
    f"aux {n_audited:,} systemes GBFS listes dans le catalogue canonique "
    "MobilityData, soit l'inventaire mondial de reference au moment de "
    "la requete. Seuls les perimetres geographiques par pays ont ete "
    "recalibres. Les seuils statistiques (<i>&sigma;</i><sub>max</sub>=3, "
    "<i>N</i><sub>min</sub>=20, &tau;<sub>A3</sub>=5) sont conserves "
    "identiques.<br><br>"
    "<b>Resultat global :</b> "
    f"{n_flagged} systemes (sur {n_with_data} avec donnees exploitables) "
    f"sont flagues par au moins une classe A1 a A5, repartis sur "
    f"{n_countries} pays. Deux hotspots inversement structures emergent : "
    "la Tchequie (un seul operateur, nextbike, fait 100 % du corpus "
    "national et propage A2/A3) et la Suisse (15 operateurs differents "
    "declenchent les 5 classes simultanement). L'extension a aussi mis "
    "en evidence deux classes candidates : A6 (dock a capacite nulle, "
    "14 systemes en queue extreme) et A7 (champ capacite structurellement "
    "nul, 215 systemes / 70 176 stations supplementaires invisibles a "
    "A1-A6, dominees par Dott a l'international).",
    findings=[
        (f"{n_audited:,}", "systemes audites (catalogue mondial)"),
        (f"{n_countries}", "pays couverts"),
        (f"{n_flagged}", "systemes flagues A1-A5"),
        ("215 / 70 176", "A7 systemes / stations non couverts par A1-A6"),
        ("nextbike (CZ), Pony (FR), Dott (global)", "operateurs-hotspots"),
    ],
)

sidebar_nav()

# ─── KPI ───────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Systemes audites", f"{n_audited:,}",
          f"{n_reachable:,} reachables (HTTP 200)")
k2.metric("Avec donnees exploitables", f"{n_with_data:,}",
          f"{100 * n_with_data / n_audited:.1f} % du total")
k3.metric("Flagues A1 a A5", f"{n_flagged}",
          f"{100 * n_flagged / n_with_data:.1f} % des exploitables")
k4.metric("Pays couverts", f"{n_countries}",
          "europeens, americains, asiatiques")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 : Cadre methodologique
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(1, "Le cadre methodologique transfere au corpus mondial")

st.markdown(
    "Le pipeline d'audit utilise sur le corpus francais (123 systemes, "
    "46 307 stations certifiees) repose sur cinq classes d'anomalie "
    "formalisees a partir de la dimensionnalite ISO/IEC 25012 enrichie "
    "par retro-ingenierie sur les flottes free-floating francaises. "
    "Le tableau 1.1 rappelle les classes et leur incidence sur les deux "
    "corpus."
)

tax = pd.DataFrame(
    {
        "Classe": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        "Nom": [
            "Inclusion hors-domaine",
            "Capacite placeholder",
            "Sur-capacite structurelle",
            "Erreur geospatiale",
            "Couverture hors perimetre",
            "Dock a capacite nulle (candidate)",
            "Champ capacite nul (candidate)",
        ],
        "Signature": [
            "Autopartage publie comme VLS",
            "c_i constant non nul (ex. c=100)",
            "Moyenne conditionnelle sur free-floating",
            "Coordonnees transposees ou hors pays",
            "Aire systeme >50 000 km^2 ou outre-mer",
            ">= seuil de stations c=0 sur dock-based",
            ">= 50 % des stations declarent c = NaN",
        ],
        "Incidence FR": ["14 syst.", "3 syst.", "8 syst.",
                         "3.8 % stns", "5 syst.", "0 syst.",
                         "19 syst. (free-floating)"],
        "Incidence mondiale": [
            f"{n_a1} syst.",
            f"{n_a2} syst.",
            f"{n_a3} syst.",
            f"{n_a4} syst.",
            f"{n_a5} syst.",
            "14 syst. (A6-soft)",
            "215 syst. / 70 176 stations",
        ],
        "Origine": [
            "Corpus FR",
            "Corpus FR",
            "Corpus FR",
            "Corpus FR",
            "Corpus FR",
            "Audit mondial",
            "Audit mondial (paradoxe italien)",
        ],
    }
)
st.dataframe(
    tax, hide_index=True, use_container_width=True,
    column_config={
        "Classe": st.column_config.TextColumn(width="small"),
        "Nom": st.column_config.TextColumn(width="medium"),
        "Signature": st.column_config.TextColumn(width="large"),
    },
)
st.caption(
    "Tableau 1.1 - Taxonomie des anomalies. A1 a A5 sont issues du "
    "corpus francais et sont restees inchangees lors de l'extension. "
    "A6 et A7 ont ete decouvertes uniquement grace a l'audit mondial."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 : Vue d'ensemble par pays
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(2, "Repartition des anomalies par pays")

st.markdown(
    "La figure 2.1 cartographie le taux de detection (systemes flagues / "
    "systemes audites avec donnees) par pays. Les pays ou le taux depasse "
    "20 % sont consideres comme des hotspots. La Tchequie atteint 55.6 %, "
    "la Suisse 32.7 %, l'Allemagne 13.1 % apres correction du bug A1 "
    "documente en section 5."
)

# Construction d'un dataframe pays
country_rows = []
for code, val in audit_summary["by_country"].items():
    if val["audited"] >= 5:
        a = val.get("anomalies", {})
        country_rows.append({
            "Pays": code,
            "Audites": val["audited"],
            "Reachables": val["reachable"],
            "Flagues": val["flagged"],
            "Taux flag (%)": (
                100 * val["flagged"] / val["reachable"]
                if val["reachable"] else 0
            ),
            "A1 cars": a.get("a1_cars", 0),
            "A2 placeholder": a.get("a2_placeholder", 0),
            "A3 over-cap": a.get("a3_overcap_flag", 0),
            "A4 perim": a.get("a4_perim_flag", 0),
            "A5 macro": a.get("a5_macro_flag", 0),
        })
df_country = pd.DataFrame(country_rows).sort_values(
    "Taux flag (%)", ascending=False
)

# Bar chart pays vs taux de flag
top12 = df_country.head(12)
fig2 = go.Figure()
classes = [
    ("A1 cars", "#C0392B"),
    ("A2 placeholder", "#E67E22"),
    ("A3 over-cap", "#F1C40F"),
    ("A4 perim", "#16A085"),
    ("A5 macro", "#2980B9"),
]
for cname, color in classes:
    fig2.add_trace(go.Bar(
        name=cname, x=top12["Pays"], y=top12[cname],
        marker_color=color,
    ))
fig2.update_layout(
    barmode="stack",
    title="Decompte de classes d'anomalie par pays (top 12 audites)",
    yaxis_title="Nombre de systemes flagues",
    xaxis_title="Code pays ISO",
    height=420, margin=dict(t=50, b=40, l=40, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "Figure 2.1 - Decompte des systemes flagues par classe et par pays. "
    "Les barres sont empilees ; un meme systeme peut etre flague sous "
    "plusieurs classes."
)

st.dataframe(
    df_country, hide_index=True, use_container_width=True,
    column_config={
        "Taux flag (%)": st.column_config.NumberColumn(
            "Taux flag (%)", format="%.1f %%"
        ),
    },
)
st.caption(
    "Tableau 2.1 - Detail par pays (audites >= 5 systemes). "
    "Sources : MobilityData canonical catalogue + ingestion automatique "
    "le 2026-05 sur le pipeline `massive_audit.py`."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 : Tchequie - un seul operateur drive 25 anomalies
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(3, "Tchequie : un operateur unique propage l'anti-pattern")

st.markdown(
    "Sur les 45 systemes GBFS audites en Tchequie, **les 45 appartiennent "
    "a un meme operateur, nextbike**. 25 deploiements declenchent A2 "
    "(capacite placeholder, 16 systemes) ou A3 (sur-capacite structurelle, "
    "12 systemes), avec 3 deploiements doublement flagues. Les 25 systemes "
    "publient ensemble 1606 entrees de station_information qui, sous la "
    "semantique declaree du standard, seraient comptabilisees comme des "
    "docks physiques. C'est le miroir exact du cas francais : Pony "
    "concentre l'A2/A3 free-floating en France, nextbike concentre "
    "l'A2/A3 en Tchequie."
)

cz_rows = df_audit[df_audit["country"] == "CZ"].copy()
cz_flagged = cz_rows[cz_rows["any_anomaly"]].copy()
cz_flagged["A2"] = cz_flagged["a2_placeholder"]
cz_flagged["A3"] = cz_flagged["a3_overcap_flag"]
cz_flagged["ratio A3"] = pd.to_numeric(
    cz_flagged["a3_overcap_ratio"], errors="coerce"
).round(2)
display_cz = cz_flagged[["name", "stations", "ratio A3", "A2", "A3"]].rename(
    columns={"name": "Systeme nextbike", "stations": "Stations declarees"}
)
display_cz["A2"] = display_cz["A2"].map({True: "x", False: ""})
display_cz["A3"] = display_cz["A3"].map({True: "x", False: ""})
st.dataframe(
    display_cz.sort_values("Stations declarees", ascending=False),
    hide_index=True, use_container_width=True, height=420,
)
st.caption(
    "Tableau 3.1 - 25 deploiements nextbike Tchequie declenchant A2 et/ou "
    "A3. La colonne 'ratio A3' est le ratio capacite-profil / capacite-"
    "moyenne ; un ratio >5 indique une moyenne conditionnelle sur des "
    "stations vides. La colonne 'Stations declarees' est l'effectif "
    "publie dans station_information, qui surevalue le parc physique "
    "reel."
)

cz_total = int(cz_rows["stations"].fillna(0).sum())
cz_flag_stations = int(cz_flagged["stations"].fillna(0).sum())
k1, k2, k3 = st.columns(3)
k1.metric("Systemes audites en CZ", "45", "tous nextbike")
k2.metric("Systemes flagues", "25", f"{25/45*100:.1f} % du corpus national")
k3.metric("Stations affectees", f"{cz_flag_stations:,}",
          f"sur {cz_total:,} declarees en CZ")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 : Suisse - fragmentation a 15 operateurs
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(4, "Suisse : 15 operateurs, 5 classes simultanees")

st.markdown(
    "La Suisse est le seul pays de l'audit (avec la France) ou les cinq "
    "classes A1-A5 sont detectees simultanement : 8 A1, 3 A2, 3 A3, 1 A4 "
    "et 5 A5 sur 16 systemes flagues parmi 49 audites. L'enjeu structurel "
    "est inverse de la Tchequie : la-bas, un operateur unique propage un "
    "anti-pattern ; ici, **15 operateurs differents declenchent les 5 "
    "classes**. Le pattern suisse demontre que la meme taxonomie capture "
    "les anomalies sous deux configurations organisationnelles opposees "
    "(monopole, fragmentation), ce qui en renforce la robustesse."
)

ch_rows = df_audit[df_audit["country"] == "CH"].copy()
ch_flagged = ch_rows[ch_rows["any_anomaly"]].copy()


def _flags_for_row(r: pd.Series) -> str:
    bits = []
    if r.get("a1_cars"): bits.append("A1")
    if r.get("a2_placeholder"): bits.append("A2")
    if r.get("a3_overcap_flag"): bits.append("A3")
    if r.get("a4_perim_flag"): bits.append("A4")
    if r.get("a5_macro_flag"): bits.append("A5")
    return ", ".join(bits)


ch_flagged["Classes detectees"] = ch_flagged.apply(_flags_for_row, axis=1)
display_ch = ch_flagged[
    ["name", "stations", "vehicle_form_factors", "Classes detectees"]
].rename(columns={
    "name": "Systeme suisse",
    "stations": "Stations",
    "vehicle_form_factors": "Vehicle types declares",
})
st.dataframe(
    display_ch.sort_values("Stations", ascending=False),
    hide_index=True, use_container_width=True, height=460,
)
st.caption(
    "Tableau 4.1 - 16 systemes suisses flagues sur 49 audites, par classe "
    "detectee et type de vehicule declare dans vehicle_types. "
    "Trois familles emergent : (i) operateurs car-sharing nationaux "
    "(2EM, Mobility, MyBuxi, edrive, Car-ship, QuickRent), (ii) flottes "
    "mixtes velo + trottinette (Voi, Bolt, nextbike CH), (iii) "
    "agregateurs nationaux (sharedmobility.ch, Velospot)."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 : Allemagne - l'audit s'audite
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(5, "Allemagne : auditer le code de l'audit")

st.markdown(
    "L'execution initiale de l'audit mondial a flague 64 systemes "
    "allemands sur 251, dont 58 en classe A1. Une inspection manuelle a "
    "revele que le detecteur A1 reposait sur un test de sous-chaine "
    "`'car' in form_factor`, ce qui faisait remonter `cargo_bicycle` "
    "(velo cargo) comme un car-sharing par erreur. Apres correction du "
    "detecteur (test d'appartenance stricte sur l'enum GBFS v3), "
    "**21 systemes allemands restent flagues A1**, et ce sont des marques "
    "de car-sharing authentiques : Conficars, Stadtmobil, TeilAuto, Ford "
    "Carsharing, Flinkster. L'epreuve illustre deux points : les "
    "pipelines d'audit ouverts necessitent eux-memes un audit, et le "
    "signal sous-jacent reste reel (21 vrais car-sharing sont encore "
    "etiquetes BSS sur les portails nationaux)."
)

# Stats DE avant / apres correction
fix_data = pd.DataFrame({
    "Etape": [
        "Audit initial (bug substring)",
        "Apres correction (membership strict)",
        "Reduction des faux positifs",
    ],
    "Total flagues DE": [64, 33, "-31"],
    "A1 cars": [58, 21, "-37 faux positifs cargo_bicycle"],
})
st.dataframe(fix_data, hide_index=True, use_container_width=True)
st.caption(
    "Tableau 5.1 - Impact du fix substring sur le compte allemand. "
    "37 faux positifs A1 etaient des operateurs de velos cargo "
    "(`cargo_bicycle` matchait `'car' in form_factor`). Apres "
    "correction, le signal de fond reste : 21 vrais car-sharing en DE."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 : A6 zero-capacity dock - calibration empirique
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(6, "A6 : decouverte d'une 6e classe et calibration empirique")

st.markdown(
    "Au-dela des classes A1 a A5 transferees du corpus francais, l'audit "
    "mondial fait emerger un pattern non couvert : une fraction non "
    "triviale de stations declare `capacity = 0` dans station_information. "
    "Puisque la specification GBFS reserve station_information pour "
    "l'infrastructure physique, une station non virtuelle a capacite "
    "nulle est semantiquement incoherente. La classe candidate A6 "
    "(zero-capacity dock) est formalisee sur la base de la distribution "
    "empirique de ce taux sur le corpus."
)

# Distribution buckets (precomputee)
buckets_df = pd.DataFrame({
    "Bucket": [
        "exactement 0 %",
        "0+ a 0.5 %",
        "0.5 a 1 %",
        "1 a 2 %",
        "2 a 5 %",
        "5 a 10 %",
        ">= 10 %",
    ],
    "Systemes": [508, 9, 8, 4, 6, 2, 1],
    "Part (%)": [94.4, 1.7, 1.5, 0.7, 1.1, 0.4, 0.2],
})
fig6 = px.bar(
    buckets_df, x="Bucket", y="Systemes",
    title="Distribution du taux de stations declarant capacity = 0 (n = 538 systemes mondiaux audites >= 20 stations)",
    color="Part (%)", color_continuous_scale="Blues",
    text="Systemes",
)
fig6.update_layout(
    height=380, margin=dict(t=50, b=40, l=40, r=20),
    xaxis_title="Taux de stations declarant c = 0", yaxis_title="Nombre de systemes",
)
fig6.update_traces(textposition="outside")
st.plotly_chart(fig6, use_container_width=True)
st.caption(
    "Figure 6.1 - Distribution empirique du taux r_A6 sur les 538 "
    "systemes mondiaux audites avec au moins 20 stations. La masse de "
    "94.4 % a 0 % constitue le baseline dock-based ; le reste forme une "
    "queue lourde. C'est sur cette queue que les deux seuils A6-soft et "
    "A6-hard sont calibres."
)

st.markdown(
    "**Calibration des seuils.** Plutot que de fixer le seuil par "
    "convention, on l'ancre sur la distribution :"
)
seuils = pd.DataFrame({
    "Seuil": ["A6-soft", "A6-hard"],
    "Definition": [
        "95e percentile empirique",
        "mean + 3 sigma (queue normale)",
    ],
    "Valeur": ["0.15 %", "approx. 2.2 %"],
    "Systemes flagues (monde)": [30, 8],
    "Usage": [
        "Flag pour consommateurs avals",
        "Anomalie operateur high-confidence",
    ],
})
st.dataframe(seuils, hide_index=True, use_container_width=True)
st.caption(
    "Tableau 6.1 - Deux seuils derivees de la distribution empirique. "
    "France dock-based : 0 / 65 systemes franchissent l'un ou l'autre seuil "
    "(controle negatif corpus-level)."
)

st.markdown("**Systemes en queue extreme (A6-hard).**")
a6_top = pd.DataFrame(audit_summary.get("a6_zero_capacity_top", []))
if not a6_top.empty:
    a6_top = a6_top.rename(columns={
        "country": "Pays",
        "name": "Systeme",
        "stations": "Stations",
        "c0_pct": "Taux c=0 (%)",
    })
    st.dataframe(
        a6_top.head(15), hide_index=True, use_container_width=True,
        column_config={
            "Taux c=0 (%)": st.column_config.NumberColumn(
                "Taux c=0 (%)", format="%.2f %%"
            ),
        },
    )
    st.caption(
        "Tableau 6.2 - Top 15 systemes mondiaux par taux declare de stations "
        "a capacite nulle. PBSC HQ (Canada, 18.2 %) et Beryl Greater "
        "Manchester (UK, 10.5 %) sont les cas extremes. Citi Bike NYC "
        "(2.4 %) confirme que le pattern affecte aussi les grands "
        "operateurs etablis."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 : Paradoxe italien & A7 candidate (champ capacite nul)
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(7, "Paradoxe italien : 70 000 stations cachees par une lacune de la taxonomie")

st.markdown(
    "L'Italie illustre le risque oppose : un audit apparemment **propre** "
    "(0 systeme flague sur 34 audites) qui masque en fait un gros volume "
    "d'anomalies. Sur les 25 systemes italiens avec donnees exploitables, "
    "**17 sont des deploiements Dott** dont *Dott Milan* avec ses "
    "**11 199 stations declarees**. Chaque deploiement Dott publie "
    "`capacity = NaN` sur 100 % des entrees `station_information`. La "
    "taxonomie A1-A6 telle qu'elle etait formulee n'attrape pas ce pattern :"
)

st.markdown(
    "- **A2** exige une valeur constante non nulle (NaN ne match pas)\n"
    "- **A3** exige des capacites numeriques pour calculer le ratio "
    "capacite-profil (NaN bloque le calcul)\n"
    "- **A4/A5** sont geospatiales\n"
    "- **A6** teste c = 0 (et non c = NaN)\n\n"
    "La generalisation de ce probe a l'ensemble du corpus mondial "
    "documente une 7e classe candidate (A7) : **215 systemes mondiaux** "
    "avec >= 50 % de capacite NaN, totalisant **70 176 stations** que "
    "A1-A6 manquait. Dott concentre 141 de ces 215 systemes."
)

# ─── Distribution rho_A7 (bimodale) ─────────────────────────────────────────────
df_a7 = df_audit[
    df_audit["reachable"] & (df_audit["stations"].fillna(0) >= 20)
].copy()
df_a7["nan_pct"] = pd.to_numeric(df_a7["capacity_nan_pct"], errors="coerce")
df_a7 = df_a7.dropna(subset=["nan_pct"])

buckets_a7 = pd.DataFrame({
    "Bucket de capacite NaN": [
        "0 %", "0+ a 10 %", "10 a 50 %", "50 a 90 %", "90 a 100 %", "exactement 100 %"
    ],
    "Systemes": [
        int(((df_a7["nan_pct"] == 0)).sum()),
        int(((df_a7["nan_pct"] > 0) & (df_a7["nan_pct"] < 10)).sum()),
        int(((df_a7["nan_pct"] >= 10) & (df_a7["nan_pct"] < 50)).sum()),
        int(((df_a7["nan_pct"] >= 50) & (df_a7["nan_pct"] < 90)).sum()),
        int(((df_a7["nan_pct"] >= 90) & (df_a7["nan_pct"] < 100)).sum()),
        int(((df_a7["nan_pct"] == 100)).sum()),
    ],
})

fig7 = px.bar(
    buckets_a7, x="Bucket de capacite NaN", y="Systemes",
    title="Figure 7.1 - Distribution du taux NaN sur la capacite (n = "
          f"{len(df_a7)} systemes globaux >= 20 stations)",
    color="Systemes", color_continuous_scale="Reds",
    text="Systemes",
)
fig7.update_layout(
    height=380, margin=dict(t=50, b=40, l=40, r=20),
    yaxis_title="Nombre de systemes",
)
fig7.update_traces(textposition="outside")
st.plotly_chart(fig7, use_container_width=True)
st.caption(
    "Figure 7.1 - Distribution bimodale du taux de stations declarant "
    "`capacity = NaN`. La grande masse a 0 % (systemes qui remplissent "
    "le champ) et la masse secondaire a 100 % (systemes type Dott qui "
    "ne le remplissent jamais) justifient le seuil tau_A7 = 0.5 ; tout "
    "seuil entre 1 % et 99 % donne le meme ensemble flague."
)

# ─── Top operateurs A7 ─────────────────────────────────────────────────────────
from collections import Counter as _Counter
a7_rows = df_a7[
    (df_a7["nan_pct"] >= 50) & (~df_a7["any_anomaly"])
].copy()
op_a7 = _Counter(
    (str(r["name"]).split()[0].lower() if pd.notna(r["name"]) else "?")
    for _, r in a7_rows.iterrows()
)
df_op_a7 = pd.DataFrame(
    op_a7.most_common(15),
    columns=["Operateur", "Systemes flagues A7 et non A1-A6"],
)
df_op_a7["Stations affectees"] = [
    int(a7_rows[a7_rows["name"].str.lower().str.startswith(op)]["stations"].fillna(0).sum())
    for op in df_op_a7["Operateur"]
]
st.markdown("**Top 15 operateurs propageant A7 a l'international.**")
st.dataframe(df_op_a7, hide_index=True, use_container_width=True)
st.caption(
    "Tableau 7.1 - Operateurs derriere les systemes A7 invisibles a A1-A6. "
    "Dott domine massivement (141 systemes), suivi de nextbike (13) et "
    "Bird (10). Ces operateurs ne remplissent simplement pas le champ "
    "`capacity` dans `station_information`."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 : Operateurs transnationaux - portabilite des anti-patterns
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(8, "Portabilite operationnelle des anti-patterns")

st.markdown(
    "Trois operateurs majeurs du marche micromobilite operent dans "
    "plusieurs pays simultanement, ce qui permet de tester si leurs "
    "anti-patterns sont specifiques d'une juridiction ou portables. "
    "Le tableau 8.1 montre que la convention de publication suit "
    "l'operateur, pas le pays."
)

df_a = df_audit[df_audit["reachable"]].copy()
df_a["op"] = df_a["name"].astype(str).str.split().str[0].str.lower()
ops_focus = ["dott", "nextbike", "bird", "pony", "voi", "bolt", "lime", "tier"]
rows_op = []
for op in ops_focus:
    subset = df_a[df_a["op"] == op]
    if len(subset) == 0:
        continue
    countries = sorted(subset["country"].unique().tolist())
    stations = int(subset["stations"].fillna(0).sum())
    a7_count = int(((pd.to_numeric(subset["capacity_nan_pct"], errors="coerce") >= 50)
                    & ~subset["any_anomaly"]).sum())
    flagged = int(subset["any_anomaly"].sum())
    rows_op.append({
        "Operateur": op,
        "Pays presents": ", ".join(countries),
        "Nb pays": len(countries),
        "Systemes audites": len(subset),
        "Stations totales": stations,
        "Flagues A1-A5": flagged,
        "Flagues A7 seul": a7_count,
    })
df_op_overview = pd.DataFrame(rows_op).sort_values("Stations totales", ascending=False)
st.dataframe(df_op_overview, hide_index=True, use_container_width=True)
st.caption(
    "Tableau 8.1 - Empreinte mondiale des principaux operateurs "
    "micromobilite. Dott et Bird transportent leur convention "
    "`capacity = NaN` d'un pays a l'autre. nextbike alterne entre "
    "placeholders (A2) et profils-flotte (A3) selon le pays. Pony "
    "reste essentiellement franco-francais avec quelques deploiements "
    "etrangers."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9 : Corrections methodologiques apportees a l'audit
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(9, "Corrections methodologiques apportees pendant l'audit")

st.markdown(
    "L'audit a large echelle a permis d'identifier trois failles dans le "
    "pipeline initial et de les corriger. La transparence sur ces "
    "corrections est elle-meme un finding du protocole : un audit "
    "ouvert s'audite. Le tableau 9.1 trace les trois corrections "
    "principales et leur impact sur le nombre total de systemes "
    "flagues."
)

fixes = pd.DataFrame({
    "Correction": [
        "A4 (geospatial) : seuil a 2 niveaux",
        "A1 (cars) : substring vers membership stricte",
        "A5 (perimetre) : bbox vers convex hull",
    ],
    "Probleme detecte": [
        "Bbox lat/lon trop large pour certains pays (PL, FI), faux positifs"
        " sur 1 station hors-perimetre",
        "Test 'car' in form_factor catchait 'cargo_bicycle' (velo cargo)",
        "Bbox rectangulaire surestime l'aire des reseaux etires "
        "(Suisse 99000 km^2 vs surface reelle 41000 km^2)",
    ],
    "Solution": [
        "Flag A4 seulement si >= 5 % des stations ET >= 5 absolu hors perimetre",
        "Test d'appartenance exact sur l'enum GBFS v3",
        "Convex hull projete en equirectangulaire local (scipy.spatial)",
    ],
    "Impact": [
        "Total: 226 -> 215 systemes flagues",
        "DE: 58 -> 21 systemes A1 (37 cargo_bicycle elimines)",
        "CH A5: 5 -> 1 (Suisse 99000 -> 41000 km^2)",
    ],
})
st.dataframe(fixes, hide_index=True, use_container_width=True)
st.caption(
    "Tableau 9.1 - Trois corrections methodologiques appliquees au "
    "pipeline initial pendant l'audit a grande echelle. Le total "
    "global de systemes flagues evolue de 226 (initial) a 173 (apres "
    "A1 fix) puis 166 (apres A5 hull) puis 204 (incluant les 38 systemes "
    "FR du catalogue MobilityData)."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10 : Free-floating francais - fragmentation semantique
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(10, "Free-floating francais : six semantiques pour un meme champ")

st.markdown(
    "La caracterisation interne du sous-ensemble free-floating (39 235 "
    "stations sur les 46 307 du Gold Standard, soit 84.7 % du corpus "
    "certifie) documente que **le meme champ GBFS "
    "station_information.capacity est populated selon six semantiques "
    "mutuellement incompatibles** par les quatre principaux operateurs."
)

if not df_ff_brand.empty:
    df_show = df_ff_brand.rename(columns={
        "brand": "Marque",
        "n_systems": "Systemes",
        "n_stations": "Stations",
        "n_cities": "Villes",
        "patterns": "Patterns observes",
    })
    st.dataframe(df_show, hide_index=True, use_container_width=True)
    st.caption(
        "Tableau 7.1 - Patterns d'usage de la capacite par marque "
        "operateur sur le sous-ensemble free-floating francais. "
        "Le champ `capacity` est utilise comme : (i) valeur nulle "
        "systematique (Dott, Bird), (ii) placeholder constant "
        "(Pony Nice c=100), (iii) ratio par vehicule (Pony Paris "
        "c~=1.6), (iv) estimateur de profil de flotte (Pony Bordeaux "
        "c=15), (v) estimateur de profil reduit (Voi)."
    )

st.markdown(
    "**Controle positif : Beryl London.** L'operateur britannique Beryl "
    "publie zero entree dans station_information et place tous ses "
    "vehicules dans free_bike_status, ce qui est le pattern canonique "
    "GBFS v2/v3 pour les flottes free-floating. Le fait que cet usage "
    "correct existe demontre que la fragmentation observee en France "
    "n'est pas une fatalite du standard mais une convention propagee "
    "faute de couche d'audit entre publication et usage academique."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 : Synthese
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
section(11, "Synthese et portee scientifique")

st.markdown(
    "Trois conclusions empiriques se degagent de l'audit a large echelle."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        "**1. La taxonomie transfere.**<br>"
        "Les classes A1 a A5 documentees sur les 123 systemes francais "
        "detectent des anomalies structurelles equivalentes sur "
        f"{n_flagged} systemes non-francais a travers {n_countries} pays, "
        "sans recalibration methodologique. Le protocole n'est pas "
        "specifiquement adapte au marche francais.",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        "**2. Les hotspots sont operateur-driven.**<br>"
        "Tchequie et France illustrent la version monopole "
        "(un operateur unique propage l'anti-pattern) ; Suisse illustre "
        "la version fragmentee (15 operateurs declenchent toutes les "
        "classes). L'agregation par pays est descriptive, "
        "l'agregation par operateur est explicative.",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        "**3. Le corpus etendu enrichit la taxonomie.**<br>"
        "La 6e classe candidate A6 (zero-capacity dock) n'aurait pas "
        "pu emerger de l'audit francais seul (incidence FR = 0). "
        "L'extension geographique est ainsi un mecanisme de "
        "decouverte methodologique en plus d'une validation de "
        "transferabilite.",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    "**Pour aller plus loin.** Les protocoles complets des six "
    "experiences de validation (E1 inter-rater, E2 sensibilite des seuils, "
    "E3 stabilite temporelle 12 mois, E4 extension dynamique, "
    "E5 generalisation europeenne et mondiale, E6 audit free-floating "
    "natif) sont documentes dans "
    "`papers/01_gold_standard/experiments/README.md`. Le pipeline "
    "d'audit mondial est reproductible via "
    "`papers/01_gold_standard/experiments/e5_europe/massive_audit.py`."
)

st.caption(
    "R. Fosse & G. Pallares, 2025-2026 - Programme BikeShare-ICT, "
    "CESI LINEACT. Donnees : MobilityData canonical catalogue "
    "(2026-05) + transport.data.gouv.fr."
)
