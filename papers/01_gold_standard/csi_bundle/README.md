# CSI submission bundle — Gold Standard GBFS Audit

Ready-to-upload Elsevier *Computer Standards & Interfaces* manuscript.

## Files

- `manuscript.tex` — main file (same content as `../gold_standard_csi.tex`)
- `references.bib` — bibliography
- `figures/` — all referenced figures (PDFs + the `fig00_visual_abstract.png` graphical abstract)

## CSI compliance checklist

- [x] Title length 17 words (CSI typical 10-20)
- [x] Abstract ≤ 250 words (208 words)
- [x] Highlights 3–5 bullets, each ≤ 85 characters (5 items)
- [x] Keywords 1–7 (7 keywords)
- [x] Numbered sections (1, 1.1, ...)
- [x] Numbered references [n] (bibstyle `unsrt`)
- [x] CRediT authorship contribution statement
- [x] Declaration of competing interests
- [x] Declaration of generative AI use
- [x] Funding statement
- [x] Data and code availability statement
- [x] Single-column A4
- [x] 29 unique citations (target: 25-45 for this length)
- [x] No unpublished-paper citations in main reference list
- [x] All 7 figures (graphical abstract + 6 in-paper) used
- [ ] **Graphical abstract**: submit `figures/fig00_visual_abstract.png`
      as a SEPARATE file via Editorial Manager (NOT embedded in
      manuscript). The relevant block is commented out at the top of
      `manuscript.tex`.

## Build

```bash
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

PDF builds at 22 pages, ~0.7 MB.

## Submission steps

1. Compress this folder: `zip -r csi_submission.zip .` (excluding build artefacts such as `.aux`, `.log`, `.out`).
2. Upload via Elsevier Editorial Manager: <https://www.editorialmanager.com/csi/>.
3. Upload `figures/fig00_visual_abstract.png` as a SEPARATE graphical-abstract file (not part of the manuscript).
4. Suggested cover letter pitch: see `../README.md` or contact the lead author.
5. Recommended reviewer fields: open mobility data, GBFS / GTFS validation, data quality engineering, FAIR data principles.
