# IMD-Bayesian paper — Overleaf bundle

Self-contained build directory for the paper *A Bayesian
station-level Soft Mobility Index for the audited French
bike-sharing panel*.

## Contents

```
overleaf/
├── main.tex          — paper source (12 pages, IMD-3 Bayesian)
├── references.bib    — bibliography
├── main.pdf          — pre-compiled PDF
├── figures/          — 3 figures (weights posterior, top-20
│                       ranking with CrI, within-city stations)
└── README.md         — this file
```

## Build

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Headline findings

- Posterior on multimodality weight: $w_M = 0.82$
  ($95\,\%$ CrI $[0.70, 0.89]$), with $P(w_M \text{ dominates})
  = 1.00$ on 12 000 MCMC samples.
- Both reference slopes credibly positive:
  $\beta_{\text{FUB}} = +0.55$ ([0.28, 0.83]),
  $\beta_{\text{EMP}} = +0.69$ ([0.48, 0.90]).
- $\rho(\text{IMD}, \text{income}) = +0.04$ ($p = 0.76$):
  cycling-environment quality is statistically independent of
  local income.
- $80.8\,\%$ of station-level IMD variance is within cities
  rather than between -- the indicator should operate at
  station granularity.
- Strasbourg, Montpellier and Paris lead the panel with
  CI-overlapping posterior medians; Strasbourg's posterior
  median is $20.5$ (CrI $[16.0, 30.0]$).

## Companion repository

<https://github.com/rohanfosse/bikeshare-data-explorer>
