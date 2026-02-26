"""
styles.py — Mise en page académique partagée pour toutes les pages du tableau de bord.
Thème inspiré des publications de recherche : sobre, structuré, sans émoji.
"""
from __future__ import annotations

import streamlit as st

# ── Navigation ─────────────────────────────────────────────────────────────────

_NAV: list[tuple[str, str]] = [
    ("app.py",                     "Introduction"),
    ("pages/0_IMD.py",             "Indice de Mobilité Douce"),
    ("pages/1_Carte.py",           "Carte des stations"),
    ("pages/2_Villes.py",          "Comparaison des villes"),
    ("pages/3_Distributions.py",   "Distributions statistiques"),
    ("pages/5_Mobilite_France.py", "Mobilité nationale"),
    ("pages/6_Montpellier.py",     "Montpellier — Vélomagg"),
    ("pages/4_Export.py",          "Export des données"),
]

# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Academic Research Theme ──────────────────────────────────────────── */

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

h1 {
    font-size: 1.60rem !important;
    font-weight: 700 !important;
    color: #1A2332 !important;
    border-bottom: 2px solid #1A6FBF;
    padding-bottom: 0.35rem !important;
    margin-bottom: 0.6rem !important;
}

h2 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #1A2332 !important;
}

h3 {
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    color: #1A6FBF !important;
}

[data-testid="metric-container"] {
    border: 1px solid #dde3ec !important;
    border-radius: 5px !important;
    padding: 0.5rem 0.85rem !important;
    background-color: #f9fbfd !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #1A2332 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #5a7a99 !important;
}

[data-testid="stSidebar"] {
    background-color: #f4f8fc !important;
    border-right: 1px solid #dde3ec !important;
}

hr {
    border-color: #dde3ec !important;
    margin: 1.2rem 0 !important;
}
</style>
"""


def inject_css() -> None:
    """Injecte le CSS académique dans la page courante."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_nav() -> None:
    """Affiche le bloc de navigation dans la sidebar."""
    with st.sidebar:
        st.markdown("## Navigation")
        for path, label in _NAV:
            st.page_link(path, label=label)
        st.divider()


def abstract_box(text: str) -> None:
    """
    Affiche une boîte résumé avec bordure bleue à gauche.
    Style inspiré des abstracts d'articles scientifiques.
    """
    st.markdown(
        f"""<div style="
            border-left: 4px solid #1A6FBF;
            background-color: #f4f8fc;
            padding: 0.85rem 1.4rem;
            border-radius: 0 5px 5px 0;
            margin: 0.4rem 0 1.4rem 0;
            font-size: 0.93rem;
            line-height: 1.65;
            color: #2c3e50;
        "><strong>Résumé.</strong> {text}</div>""",
        unsafe_allow_html=True,
    )


def section(number: int | str, title: str) -> None:
    """Affiche un en-tête de section numérotée (style article)."""
    st.markdown(f"### {number}. {title}")
