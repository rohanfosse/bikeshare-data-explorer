# Research papers -- BikeShare-ICT

This folder gathers the scientific articles produced within the
**BikeShare-ICT** research programme (CESI LINEACT, 2025--2026)
and everything that supports them: shared bibliography, figure
generator, and the empirical validation experiments referenced by
the papers' validation roadmaps.

## Layout

```text
papers/
|-- README.md                                  -- this file
|-- references.bib                             -- shared bibliography (IMD paper)
|-- tools/
|   `-- generate_figures.py                    -- matplotlib figures for both papers
|-- 01_gold_standard/
|   |-- gold_standard.tex                      -- paper I (audit & infrastructure)
|   |-- SCITEPRESS.sty                         -- conference style
|   |-- references.bib                         -- local bib (Gold Standard)
|   |-- figures/                               -- generated PDFs
|   `-- experiments/                           -- E1/E2/E4 validation pilots
|       |-- README.md
|       |-- _paths.py
|       |-- e1_irr/                            -- inter-rater reliability
|       |-- e2_sensitivity/                    -- sigma_max sweep
|       `-- e4_dynamic_pilot/                  -- A6 candidate detection
`-- 02_imd/
    |-- imd.tex                                -- paper II (IMD composite index)
    `-- figures/                               -- generated PDFs
```

`papers/references.bib` is the legacy shared bibliography used by
the IMD paper (`\addbibresource{../references.bib}`). The Gold
Standard paper has its own local bibliography
`papers/01_gold_standard/references.bib` (with extra entries on
data-quality tooling, GTFS audit precedents, ISO/IEC 25012,
Croissant and DCAT-AP) referenced via `\bibliography{references}`.

## Papers

### 01 -- Gold Standard GBFS (English)

**Working title:** *Building a Gold Standard for Bike-Sharing
Systems: GBFS auditing, multi-source hybridisation and
reproducibility conditions for quantitative urban mobility
research.*

**Status:** working paper, v0.1.

**Main claim.** Raw GBFS feeds, despite their nominal
standardisation, are not a ready-to-use research artefact. The
paper documents a taxonomy of anomaly classes (A1--A5), a
reproducible audit pipeline, and the contextual enrichment that
produces the *Gold Standard GBFS France* (~46,000 stations, 122
systems, 57 urban areas) -- positioned as a research commons for
the soft mobility community. The paper closes with a six-experiment
validation roadmap, of which E1, E2 and E4 already have
preliminary pilots in [`01_gold_standard/experiments/`](01_gold_standard/experiments/).

### 02 -- Indice de Mobilite Douce (IMD) (French)

**Working title:** *Au-dela du prisme capacitaire :
l'Indice de Mobilite Douce (IMD) pour evaluer la justice spatiale
et la qualite des reseaux cyclables partages en France.*

**Status:** working paper, v0.3.

**Main claim.** Infrastructure volume does not predict cycling
quality. The paper proposes a four-component composite index
(Multimodality, Infrastructure, Safety, Topography) calibrated
empirically, together with an Indice d'Equite Sociale (IES) that
identifies *social mobility deserts*.

## Compilation

```bash
# Regenerate the figures (matplotlib -> PDF, both papers)
python papers/tools/generate_figures.py

# Compile Gold Standard (BibTeX, unsrt)
cd papers/01_gold_standard
pdflatex gold_standard.tex
bibtex   gold_standard
pdflatex gold_standard.tex
pdflatex gold_standard.tex

# Compile IMD (biblatex + biber)
cd ../02_imd
pdflatex imd.tex
biber    imd
pdflatex imd.tex
pdflatex imd.tex
```

## Validation experiments

Each experiment under
[`01_gold_standard/experiments/`](01_gold_standard/experiments/)
is self-contained and writes its outputs to its own `outputs/`
folder. See [`01_gold_standard/experiments/README.md`](01_gold_standard/experiments/README.md)
for the full list and the current headline results.

### Overleaf (free tier)

The free Overleaf plan caps each compile at ~1 minute. The
preambles have been trimmed (no `tcolorbox[most]`, no unused
`tikz`/`amsthm`/`mathtools`/`multirow`/`longtable`), so a fresh
first pass should fit. If it still times out:

1. Switch the Overleaf compiler menu to **Recompile from scratch**
   *only when needed*. The default **Fast** mode keeps `.aux` and
   `.bbl` between runs and skips `biber` when the bibliography is
   unchanged -- this alone saves ~5 s.
2. If `biber` is the bottleneck, compile once locally and upload
   the resulting `.bbl` next to the `.tex`; Overleaf will reuse it.
3. Make sure the `figures/` folder of each paper is uploaded with
   the matching `.tex`.

## Authors

R. Fosse & G. Pallares -- CESI LINEACT, BikeShare-ICT programme
(2025--2026).
