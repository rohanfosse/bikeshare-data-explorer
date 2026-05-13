# Paper 04 — A National Cycling-Environment Composite Indicator for French Communes

**Authors:** R. Fossé & G. Pallares (CESI LINEACT, 2025–2026)
**Status:** v1 draft, self-contained
**Submission target:** Transportation Research Part D / Transport Policy / Journal of Transport Geography

## Pitch

First commune-resolution real-time public-data composite indicator of cycling-environment quality for France, validated on five independent references, adding +18 pts R² over the standard Cerema infrastructure indicator, extended to all 34,858 French communes, with a national equity index (IES) identifying 362 priority "double-penalty" communes for the Plan Vélo.

## Headline results

| # | Result | Source |
|---|---|---|
| 1 | IMD-4 (Bayesian M+I+T+D simplex) wins 4/5 tournament references | B14 |
| 2 | +18 pts R² over Cerema infrastructure baseline (Cerema-residual regression) | B16 |
| 3 | Out-of-sample ρ=+0.42 in 5-fold CV (in-sample +0.62) | B18 |
| 4 | National panel on 34,858 communes: ρ=+0.41 overall, ρ=+0.55 on 42 cities pop≥100k | B20 |
| 5 | National IES: ρ(IMD,income)=+0.001 in urban subset (pop≥5k) — IMD ≠ wealth proxy | B21 |
| 6 | 362 cycling-poverty deserts identified, 6 of top-15 in La Réunion | B21 |
| 7 | Regional decomposition: no political signal, geographic-demographic gradient only | B22 |

## File structure

```
papers/04_imd_national/
├── imd_national.tex         # main paper (build from project root)
├── README.md                # this file
├── figures/                 # local copies of B14-B22 PDFs
└── overleaf/
    ├── main.tex             # same content, graphics path adjusted for Overleaf
    ├── references.bib       # bibliography copy
    └── figures/             # PDFs flat for Overleaf upload
```

## Build

```bash
cd papers/04_imd_national
pdflatex imd_national && bibtex imd_national && pdflatex imd_national && pdflatex imd_national
```

For Overleaf, zip the `overleaf/` folder and upload as a project.

## Underlying experiments

All B14–B22 experiments live in `papers/03_imd_bayesian/experiments/` and produce
their outputs into `papers/03_imd_bayesian/experiments/outputs/`. The paper
reads the figures from that folder (relative path in `\graphicspath`).

## Relationship to other papers in this repo

- **paper 01 (`papers/01_gold_standard/`)** — the 46k-station GBFS Gold Standard inventory, prerequisite for the calibration panel.
- **paper 02 (`papers/02_imd/`)** — exploratory 53-page draft, superseded.
- **paper 03 (`papers/03_imd_bayesian/`)** — station-level Bayesian methodology paper; paper 04 takes its weights as given and pushes the application to the national scale.
- **paper 04 (this one)** — standalone national-application paper, intended to be the first paper of the series to be published.

## Status notes

- v1 is figure-complete and prose-complete but lacks inline `\citep` references.
- Section 5 (regional political decomposition) is descriptive only — included to defuse the "you're measuring partisan colour" critique.
- The topography component T is set to its panel z-score mean (zero) at national application; integrating BD ALTI 25m nationally is documented as follow-up work in Section 6.
