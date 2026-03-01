"""
styles.py - Mise en page académique partagée pour toutes les pages du tableau de bord.
Thème : panneau de navigation sombre, contenu blanc épuré, style article de recherche.
"""
from __future__ import annotations

import inspect
import pathlib

import streamlit as st

# ── Navigation groupée par catégorie ──────────────────────────────────────────
# Chaque entrée : (fichier, label court)
# Les groupes sont rendus avec un séparateur visuel dans la sidebar.

_NAV_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Vue d'ensemble", [
        ("app.py",                     "Introduction"),
    ]),
    ("Axes de recherche", [
        ("pages/00_Gold_Standard.py",  "Gold Standard"),
        ("pages/0_IMD.py",             "IMD — Mobilité Douce"),
        ("pages/7_IES.py",             "IES — Équité Sociale"),
        ("pages/2_Villes.py",          "Villes — Comparaison"),
        ("pages/3_Distributions.py",   "Distributions"),
        ("pages/8_Topographie.py",     "Topographie"),
        ("pages/6_Montpellier.py",     "Montpellier"),
    ]),
    ("Modules transversaux", [
        ("pages/1_Carte.py",           "Carte des stations"),
        ("pages/5_Mobilite_France.py", "France — Validation"),
    ]),
    ("Données et références", [
        ("pages/4_Export.py",          "Export FAIR"),
        ("pages/9_References.py",      "Références"),
    ]),
]


# ── CSS global ─────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ═══════════════════════════════════════════════
   CONTENU PRINCIPAL
═══════════════════════════════════════════════ */

.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 2.5rem !important;
}

/* Titre principal */
h1 {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #1A2332 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.2rem !important;
    border-bottom: 2px solid #1A6FBF !important;
    padding-bottom: 0.35rem !important;
}

/* Sous-titres de section */
h2 {
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    color: #1A2332 !important;
    border-bottom: 1px solid #e8edf3 !important;
    padding-bottom: 0.2rem !important;
    margin-top: 0.3rem !important;
}

/* En-têtes de sous-section numérotés */
h3 {
    font-size: 0.93rem !important;
    font-weight: 600 !important;
    color: #1A6FBF !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    margin-top: 0.5rem !important;
}

/* Séparateurs */
hr {
    border: none !important;
    border-top: 1px solid #e8edf3 !important;
    margin: 1.1rem 0 !important;
}

/* ═══════════════════════════════════════════════
   PANNEAU DE NAVIGATION (SOMBRE)
═══════════════════════════════════════════════ */

[data-testid="stSidebar"] {
    background-color: #1B2635 !important;
}

/* Texte courant sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    color: #7a9bb8 !important;
    font-size: 0.8rem !important;
}

/* En-têtes sidebar */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #c2d6e8 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    border-bottom: 1px solid #2a3f58 !important;
    padding-bottom: 0.25rem !important;
    margin-bottom: 0.4rem !important;
    font-weight: 600 !important;
}

/* Dividers sidebar */
[data-testid="stSidebar"] hr {
    border-top-color: #2a3f58 !important;
    margin: 0.7rem 0 !important;
}

/* Labels widgets sidebar */
[data-testid="stSidebar"] label {
    color: #8aadc6 !important;
    font-size: 0.79rem !important;
}

/* Liens de navigation (st.page_link) */
[data-testid="stSidebar"] [data-testid="stPageLink"] a,
[data-testid="stSidebar"] [data-testid="stPageLink"] button {
    color: #7a9bb8 !important;
    font-size: 0.82rem !important;
    padding: 0.28rem 0.55rem !important;
    border-radius: 4px !important;
    border-left: 2px solid transparent !important;
    transition: all 0.14s ease !important;
    text-decoration: none !important;
    background: transparent !important;
    display: block;
    width: 100%;
    text-align: left;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover,
[data-testid="stSidebar"] [data-testid="stPageLink"] button:hover {
    color: #5ab4e8 !important;
    background: rgba(74, 159, 223, 0.1) !important;
    border-left-color: #4A9FDF !important;
}

/* Page active dans la navigation (st.page_link natif) */
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current],
[data-testid="stSidebar"] [data-testid="stPageLink"] button[aria-current] {
    color: #e0eaf4 !important;
    background: rgba(74, 159, 223, 0.18) !important;
    border-left-color: #5ab4e8 !important;
    font-weight: 600 !important;
}

/* Masquer la navigation automatique de Streamlit */
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
[data-testid="stSidebarNavLink"] {
    display: none !important;
}

/* ═══════════════════════════════════════════════
   CARTES DE MÉTRIQUES
═══════════════════════════════════════════════ */

[data-testid="metric-container"] {
    border: 1px solid #e4ecf3 !important;
    border-radius: 6px !important;
    padding: 0.55rem 0.9rem !important;
    background: #f8fafd !important;
    box-shadow: 0 1px 4px rgba(26, 35, 50, 0.05) !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1A2332 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #6b8aaa !important;
}
</style>
"""


def inject_css() -> None:
    """Injecte le CSS global dans la page courante."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_nav() -> None:
    """
    Affiche le panneau de navigation sombre avec le branding du projet.
    Détecte automatiquement la page courante et la met en évidence.
    À appeler une seule fois par page, après inject_css().
    """
    # Détection automatique de la page courante via le frame d'appel
    try:
        caller_file = pathlib.Path(inspect.stack()[1].filename).name
    except Exception:
        caller_file = ""

    with st.sidebar:
        # Branding projet
        st.markdown(
            """
            <div style="
                padding: 0.9rem 0.6rem 0.8rem;
                margin-bottom: 0.3rem;
                border-bottom: 1px solid #2a3f58;
            ">
                <div style="
                    font-size: 0.62rem;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: #4A9FDF;
                    font-weight: 700;
                ">CESI · BikeShare-ICT · 2025-2026</div>
                <div style="
                    font-size: 1.0rem;
                    font-weight: 700;
                    color: #e0eaf4;
                    margin-top: 0.3rem;
                    line-height: 1.2;
                ">Gold Standard GBFS</div>
                <div style="
                    font-size: 0.73rem;
                    color: #4a6a88;
                    margin-top: 0.2rem;
                ">Micromobilité française</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation groupée
        for group_label, entries in _NAV_GROUPS:
            st.markdown(
                f"<div style='font-size:0.60rem; text-transform:uppercase; "
                f"letter-spacing:0.13em; color:#3a5a78; font-weight:600; "
                f"margin: 0.75rem 0 0.25rem 0.3rem;'>{group_label}</div>",
                unsafe_allow_html=True,
            )
            for path, label in entries:
                nav_name = pathlib.Path(path).name
                is_active = caller_file == nav_name
                if is_active:
                    # Page courante : indicateur visuel non-cliquable
                    st.markdown(
                        f"<div style='"
                        f"color:#e0eaf4; font-weight:600; font-size:0.82rem;"
                        f"padding:0.28rem 0.55rem; border-radius:4px;"
                        f"border-left:2px solid #5ab4e8;"
                        f"background:rgba(74,159,223,0.18);"
                        f"margin-bottom:0.05rem;"
                        f"'>{label}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.page_link(path, label=label)

        st.divider()


def abstract_box(text: str) -> None:
    """
    Boîte résumé avec bordure bleue à gauche - style abstract d'article.
    """
    st.markdown(
        f"""
        <div style="
            border-left: 3px solid #1A6FBF;
            background: #f4f8fc;
            padding: 0.85rem 1.3rem;
            border-radius: 0 5px 5px 0;
            margin: 0.4rem 0 1.4rem 0;
            font-size: 0.91rem;
            line-height: 1.65;
            color: #2c3e50;
        ">
            <span style="font-weight:600; color:#1A6FBF; font-size:0.72rem;
                         text-transform:uppercase; letter-spacing:0.08em;">
                Résumé
            </span><br/>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(number: int | str, title: str) -> None:
    """En-tête de section numérotée - style article de recherche."""
    st.markdown(f"### {number}. {title}")