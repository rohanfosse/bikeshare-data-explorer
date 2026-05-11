# Experiments for the Gold Standard paper

This folder hosts the empirical validation experiments referenced
by the validation roadmap (Section 8 of
[`../gold_standard.tex`](../gold_standard.tex)). Each experiment
is self-contained: a runnable script, optional supporting code,
its own `outputs/` folder, and a short README when extra
explanation is needed.

```text
experiments/
|-- _paths.py                       shared repo-root + outputs helper
|-- README.md                       this file
|-- e1_irr/                         E1 -- inter-rater reliability of A1--A5
|   |-- README.md                   human-annotator protocol
|   |-- annotation_guide.md         A1--A5 definitions for annotators
|   |-- AUTO_ANNOTATION_NOTE.md     limits of the automated surrogate
|   |-- sample_stations.py          stratified 500-station sampler
|   |-- score_kappa.py              Cohen + Fleiss kappa scorer (humans)
|   |-- auto_annotators.py          three orthogonal automated classifiers
|   |-- coverage_analysis.py        complementary-coverage report
|   `-- outputs/                    generated samples + per-run reports
|-- e2_sensitivity/                 E2 -- sigma_max sensitivity (single-axis)
|   |-- run.py
|   `-- outputs/report.json
`-- e4_dynamic_pilot/               E4 -- dynamic A6 candidate detection
    |-- run.py
    `-- outputs/{report.json, indicators.csv}
```

## How to run

All scripts use `_paths.py` to locate the repository root via the
`.git` / `CITATION.cff` marker, so they are robust to relocation.
Run from the repository root:

```bash
# E1 (automated surrogate)
python papers/01_gold_standard/experiments/e1_irr/sample_stations.py
python papers/01_gold_standard/experiments/e1_irr/auto_annotators.py
python papers/01_gold_standard/experiments/e1_irr/coverage_analysis.py

# E2 (sigma_max single-axis sweep)
python papers/01_gold_standard/experiments/e2_sensitivity/run.py

# E4 (2-day dynamic pilot on station_status snapshots)
python papers/01_gold_standard/experiments/e4_dynamic_pilot/run.py
```

Each run rewrites the corresponding `outputs/` files; outputs are
deterministic (seed 42 throughout).

## Headline results (current snapshot)

| Experiment | Threshold              | Observed                                     | Verdict     |
|------------|------------------------|----------------------------------------------|-------------|
| E1 (surrogate) | union recall >= 0.90 | recall 0.97, precision 0.92               | passes      |
| E2 (sigma_max) | Top-10 churn = 0      | churn 0 across [2.0, 4.0], tau >= 0.92    | passes      |
| E4 (2-day) | A6 reclass >= 5%         | 12.9% of monitored stations                | passes      |
| E3, E5, E6 | -                         | not runnable locally yet                   | open        |

E1 is documented as a surrogate; the human-annotator version is
the next priority. See `e1_irr/AUTO_ANNOTATION_NOTE.md` for the
intellectual-honesty caveats.
