# IMD/IES paper — Overleaf bundle

Self-contained build directory for the paper *Cycling quality, local
income, and spatial equity in French bike-sharing systems*.

## Contents

```
overleaf/
├── main.tex          — paper source (39 pages, 13-experiment programme)
├── references.bib    — bibliography (66 entries)
├── main.pdf          — pre-compiled PDF for sanity check
├── figures/          — all 17 figures referenced by main.tex
└── README.md         — this file
```

## Build

Set the compiler to **pdfLaTeX** and the bibliography to **BibTeX** in
Overleaf, then click *Recompile*. The build sequence is the standard

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## What changed in this version

- **Lyon-removed weight vector** added to Table 2 alongside the
  in-sample optimum, with the Top-10 reordering quantified
  (Kendall τ = 0.83, 9/10 overlap).
- **Bayesian IES prior sensitivity** sweep over τ ∈ {0.1, 1, 10}:
  reports four prior-invariant high-confidence deserts (Amiens,
  Lyon, Nancy, Saumur) alongside the nine-city τ = 1 reference list.
- **Montpellier case study** quantified: 83 % of stations have
  ≥ 1 heavy GTFS stop within 300 m, 95th-percentile mean.
- **E17 pseudo-flow** demoted to methodological probe with four
  documented interpretations of the negative correlation.
- **Honest scorecard labels**: seven clean / two qualified / three
  substitutes for deferred pre-registered protocols.

## Figure inventory

| File | Source experiment | Description |
|---|---|---|
| fig01_weights.pdf | -- | Calibrated IMD weights (softmax optimum) |
| fig02_volume_vs_imd.pdf | -- | Raw station count vs IMD |
| fig03_imd_vs_income.pdf | -- | IMD vs median income per consumption unit |
| fig04_equity_quadrant.pdf | -- | Median-split equity quadrants |
| fig05_top10_components.pdf | -- | Top-10 component decomposition |
| fig06_ies_ranking.pdf | -- | Spatial Equity Index ranking |
| e2b_top10_freq_loo.pdf | E2 | Top-10 retention under LOO bootstrap |
| e7_sobol_panel.pdf | E7 | Saltelli/Jansen variance decomposition |
| e9_desert_posterior.pdf | E9 | Bayesian IES desert probabilities |
| e10_imd_ci.pdf | E10 | Per-city bootstrap 95% CI on IMD score |
| e11_radius_sweep.pdf | E11 | Parametric buffer-radius sweep |
| e12_leverage.pdf | E12 | Cook-style calibration leverage |
| e13_archetypes.pdf | E13 | k-means typology at k=4 |
| e15_predictive_power.pdf | E15 | Metric tournament LOO predictive power |
| e16_components_vs_fub.pdf | E16 | FUB well-being decomposition by IMD component |
| e17_pseudo_flow_vs_imd.pdf | E17 | GBFS pseudo-flow vs IMD |
| summary_scorecard.pdf | -- | Validation scorecard (clean/qualified/substitute) |

## Companion repository

Full data, code and experiment scripts are released at
<https://github.com/rohanfosse/bikeshare-data-explorer>.
The experiment scripts that generate the figures above live under
`papers/02_imd/experiments/`.
