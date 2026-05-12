# IMD/IES paper — Overleaf bundle

Self-contained build directory for the paper *Cycling quality, local
income, and spatial equity in French bike-sharing systems*.

## Contents

```
overleaf/
├── main.tex          — paper source (46 pages, 19 experiments)
├── references.bib    — bibliography (66 entries)
├── main.pdf          — pre-compiled PDF for sanity check
├── figures/          — all 24 figures referenced by main.tex
└── README.md         — this file
```

## Build

pdfLaTeX + BibTeX, standard 4-pass:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Most striking results

- **France ranks #1 in Europe for GBFS deployments (255 systems) but
  #18 for cycling modal share (4 %)**, with the Netherlands at the
  symmetric position. Cross-country ρ(modal share, ECF index) = +0.79,
  vs ρ(N systems, modal share) = +0.46 — confirming the IMD's central
  thesis at continental scale.
- **80.8 % of the IMD variance is *within* cities**, not between them.
  The Paris within-city spread (17.5–83.1) is wider than the
  inter-city dispersion of the panel mean.
- **Seven positive-deviant cities** share a single signature:
  multimodality = 4× the panel mean. Lyon and Saumur are negative
  deviants.
- **Counterfactual joint M+I uplift** → +18 IMD points median across
  the panel, implying a +44 % eco-counter increase under the E3
  elasticity.
- **Concurrent validity triangulated**: $R^2 = 0.79$ on 25 cities
  with IMD + Cerema km/km² + log(Cerema km), with neither variable
  redundant.

## Figure inventory (24 figures)

Core (6) :
fig01_weights · fig02_volume_vs_imd · fig03_imd_vs_income ·
fig04_equity_quadrant · fig05_top10_components · fig06_ies_ranking

Validation core (8) :
e2b_top10_freq_loo · e7_sobol_panel · e9_desert_posterior ·
e10_imd_ci · e11_radius_sweep · e12_leverage · e13_archetypes ·
summary_scorecard

Validation extension (6) :
e15_predictive_power · e16_components_vs_fub · e17_pseudo_flow_vs_imd ·
e18_within_vs_between · e19_deviants · e20_counterfactual

Extension wave new (4) :
e21_imd_vs_life_expectancy · e22_precarity_ranking ·
e23_concurrent_validity · e24_european_panel

## Companion repository

<https://github.com/rohanfosse/bikeshare-data-explorer>
