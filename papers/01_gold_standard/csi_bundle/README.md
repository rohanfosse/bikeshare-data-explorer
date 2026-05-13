# CSI submission bundle — Gold Standard GBFS Audit

Ready-to-upload Elsevier *Computer Standards & Interfaces* manuscript.

## Files

- `manuscript.tex` — main file (same content as `../gold_standard_csi.tex`)
- `references.bib` — bibliography
- `figures/` — all referenced figures (PDFs + the `fig00_visual_abstract.png` graphical abstract)

## CSI compliance checklist

- [x] Abstract ≤ 250 words (208 words)
- [x] Highlights 3–5 bullets, each ≤ 85 characters (5 items)
- [x] Keywords 1–7 (7 keywords)
- [x] Numbered sections (1, 1.1, ...)
- [x] Numbered references [n] (bibstyle `unsrt`)
- [x] CRediT authorship contribution statement
- [x] Declaration of competing interests
- [x] Declaration of generative AI use
- [x] Funding statement
- [x] Data and code availability statement (Section 5)
- [x] Graphical abstract embedded
- [x] Single-column A4

## Build

```bash
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

PDF builds at 20 pages, ~1 MB.

## Submission steps

1. Compress this folder: `zip -r csi_submission.zip .` (excluding *.aux, *.log, *.out, etc.)
2. Upload via Elsevier Editorial Manager (https://www.editorialmanager.com/csi/)
3. Suggested cover letter pitch: see `../README.md` or the lead author for details.
4. Recommended reviewer fields: open mobility data, GBFS / GTFS validation, data quality engineering, FAIR data principles.
