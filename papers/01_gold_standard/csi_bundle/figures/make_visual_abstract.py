"""
Generate the Patterns visual abstract for the GBFS France Audit Catalogue paper.

Output: fig00_visual_abstract.pdf  (and .png for journal preview)

Layout (single-pane, ~square):
  - Top:    Raw input (142 GBFS feeds, ~67k stations)
  - Middle: 6-step audit pipeline + 5 anomaly classes
  - Bottom: Certified Audit Catalogue (46,307 stations / 97 cities / 5 enrichment sources)
  - Side:   Headline result (Bordeaux rank 2 -> 14)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent

PRIMARY = "#1A6FBF"
DARK    = "#0F4C81"
ACCENT  = "#C0392B"
GREEN   = "#2E7D32"
GREY    = "#5A6470"
LIGHT   = "#F2F4F7"
BORDER  = "#C7CCD1"

def rounded_box(ax, x, y, w, h, fc=LIGHT, ec=BORDER, lw=1.0, rad=0.018):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    ))

def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, zorder=3,
    ))

def text(ax, x, y, s, ha="center", va="center", fs=8.5, color="black",
         weight="normal", style="normal"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
            fontweight=weight, fontstyle=style, zorder=5)

def make_figure():
    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("equal"); ax.axis("off")

    # Title
    text(ax, 50, 97.0, "GBFS France Audit Catalogue", fs=15.5, weight="bold", color=DARK)
    text(ax, 50, 93.7, "An auditable, versioned reference dataset for the 123 French bike-sharing feeds",
         fs=9.6, color=GREY, style="italic")

    # ---- TOP : raw input
    rounded_box(ax, 6, 80.5, 88, 8.5, fc="#FEF3E2", ec="#E0A030")
    text(ax, 50, 86.6, "Raw input  ·  transport.data.gouv.fr  +  MobilityData",
         fs=10.2, weight="bold", color="#8A5A00")
    text(ax, 50, 83.0,
         "142 candidate GBFS feeds  ·  ~67,000 stations  ·  6 operator families  ·  no semantic audit",
         fs=9.3, color="#4A3000")

    arrow(ax, 50, 80.0, 50, 76.7, color=DARK, lw=2.0)

    # ---- MIDDLE-TOP : 5 anomaly classes (compact)
    text(ax, 50, 74.7, "Five recurring anomaly classes (A1–A5)", fs=10.5, weight="bold", color=DARK)
    anomalies = [
        ("A1", "Out-of-domain", "Citiz car-sharing\nlabelled as BSS", "14 systems"),
        ("A2", "Placeholder", "Constant c = 100\non all stations", "3 systems"),
        ("A3", "Over-capacity", "Conditional averaging\non free-floating", "8 systems"),
        ("A4", "Geospatial", "Transposed coords\nor > 3σ outliers", "3.8 % stns"),
        ("A5", "Perimeter", "Overseas; rural\n> 50,000 km²", "5 systems"),
    ]
    x0, y0, w, h, gap = 5.5, 60.5, 17.2, 11.5, 1.6
    for i, (code, name, sig, inc) in enumerate(anomalies):
        x = x0 + i * (w + gap)
        rounded_box(ax, x, y0, w, h, fc=LIGHT, ec=BORDER)
        rounded_box(ax, x, y0 + h - 2.7, w, 2.7, fc=ACCENT, ec=ACCENT, rad=0.012)
        text(ax, x + w/2, y0 + h - 1.35, f"{code}  —  {name}",
             fs=9.0, weight="bold", color="white")
        text(ax, x + w/2, y0 + 5.7, sig, fs=7.6, color="black")
        text(ax, x + w/2, y0 + 1.7, inc, fs=7.6, color=DARK, weight="bold")

    arrow(ax, 50, 60.2, 50, 56.8, color=DARK, lw=2.0)

    # ---- MIDDLE : 6-step pipeline (horizontal)
    rounded_box(ax, 4, 47.0, 92, 9.5, fc="#E8F1FB", ec=PRIMARY, lw=1.4)
    text(ax, 50, 54.6, "Six-step purging pipeline  —  idempotent  ·  reversible  ·  logged",
         fs=10.0, weight="bold", color=DARK)
    steps = [
        ("S1", "carsharing\nrelabel"),
        ("S2", "placeholder\ndetect"),
        ("S3", "capacity\nrecompute"),
        ("S4", "geofilter\nφ, λ"),
        ("S5", "topological\nσ-clip"),
        ("S6", "N_min\nthreshold"),
    ]
    sx, sw, sgap = 9.0, 12.5, 1.6
    sy, sh = 48.0, 5.0
    for i, (code, lbl) in enumerate(steps):
        x = sx + i * (sw + sgap)
        rounded_box(ax, x, sy, sw, sh, fc="white", ec=PRIMARY, lw=1.1)
        text(ax, x + sw/2, sy + sh - 1.4, code, fs=9.0, weight="bold", color=PRIMARY)
        text(ax, x + sw/2, sy + 1.5, lbl, fs=7.4, color="black")
        if i < 5:
            arrow(ax, x + sw + 0.1, sy + sh/2, x + sw + sgap - 0.2, sy + sh/2,
                  color=PRIMARY, lw=1.1)

    # ---- MIDDLE-BOTTOM : 5 enrichment sources
    arrow(ax, 50, 46.7, 50, 43.4, color=DARK, lw=2.0)
    text(ax, 50, 41.8, "Multi-source hybridisation  —  spatial joins  ·  administrative aggregation",
         fs=9.6, weight="bold", color=DARK)
    sources = [
        ("INSEE\nFilosofi", "Income, Gini"),
        ("BAAC\nCerema", "Crash density"),
        ("BD TOPO", "Cycle linear"),
        ("Open\nStreetMap", "Facilities"),
        ("FUB\nBarometer", "Perception"),
    ]
    sxh, swh, sgaph = 6.0, 17.0, 1.6
    syh, shh = 33.0, 6.5
    for i, (lbl, var) in enumerate(sources):
        x = sxh + i * (swh + sgaph)
        rounded_box(ax, x, syh, swh, shh, fc="#EAF5EC", ec=GREEN, lw=1.0)
        text(ax, x + swh/2, syh + shh - 1.7, lbl, fs=8.0, weight="bold", color=GREEN)
        text(ax, x + swh/2, syh + 1.7, var, fs=7.5, color="black")

    arrow(ax, 50, 32.7, 50, 29.0, color=DARK, lw=2.0)

    # ---- BOTTOM : Audit Catalogue output
    rounded_box(ax, 6, 11.5, 60, 16.5, fc="#E8F1FB", ec=PRIMARY, lw=1.5)
    text(ax, 36, 25.0, "GBFS France Audit Catalogue  v1.0", fs=12.0, weight="bold", color=DARK)
    text(ax, 36, 22.0, "DOI 10.5281/zenodo.20125460  ·  ODbL  ·  Croissant + DCAT-AP + JSON Schema",
         fs=7.9, color=GREY, style="italic")

    # Stats columns
    stats = [
        ("46,307", "certified\nstations"),
        ("123", "audited\nsystems"),
        ("97", "cities\ncovered"),
        ("5,442", "dock-based\n(fully audited)"),
        ("30.9 %", "raw stations\nreclassified"),
    ]
    sx2, sw2 = 7.5, 11.5
    for i, (val, lbl) in enumerate(stats):
        x = sx2 + i * sw2
        text(ax, x + sw2/2, 17.5, val, fs=11.5, weight="bold", color=PRIMARY)
        text(ax, x + sw2/2, 14.0, lbl, fs=7.6, color="black")

    # Right-side callout : headline result
    rounded_box(ax, 69, 11.5, 25, 16.5, fc="#FCE9E6", ec=ACCENT, lw=1.5)
    text(ax, 81.5, 25.5, "Headline result", fs=9.6, weight="bold", color=ACCENT)
    text(ax, 81.5, 22.5, "Bordeaux  —  one A3 fix", fs=9.2, weight="bold", color="black")
    text(ax, 81.5, 19.7, "9,921 raw entries  →  225", fs=8.0, color=GREY)
    text(ax, 81.5, 17.7, "dock-based stations", fs=8.0, color=GREY)
    text(ax, 81.5, 14.5, "National rank", fs=8.4, color="black")
    text(ax, 81.5, 12.7, "2  →  14", fs=14.0, weight="bold", color=ACCENT)

    # Bottom strip : value claim
    text(ax, 50, 7.0,
         "Syntactic standardisation is necessary but not sufficient.",
         fs=9.6, weight="bold", color=DARK, style="italic")
    text(ax, 50, 4.5,
         "An open mobility feed is a starting point for research, not a substitute for one.",
         fs=8.8, color=GREY)
    text(ax, 50, 2.0,
         "Fossé & Pallares  ·  CESI LINEACT, Montpellier  ·  2026",
         fs=7.4, color=GREY, style="italic")

    plt.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig00_visual_abstract.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig00_visual_abstract.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", OUT / "fig00_visual_abstract.pdf")
    print("Wrote", OUT / "fig00_visual_abstract.png")


if __name__ == "__main__":
    make_figure()
