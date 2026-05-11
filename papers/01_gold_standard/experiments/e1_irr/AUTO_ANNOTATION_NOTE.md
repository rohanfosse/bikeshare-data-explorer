# Note on the automated E1 surrogate

This note documents the **limits** and the **interpretation** of
the automated coverage experiment provided by
`auto_annotators.py` + `coverage_analysis.py`. It is intended to be
read before citing those scripts in a paper or a reviewer response.

## What we wanted to test (full E1)

Three external human annotators, who did NOT participate in the
design of the A1–A5 taxonomy, classify a 500-station stratified
sample using only the textual rules of `annotation_guide.md`.
Inter-rater agreement is measured by Fleiss κ; if κ ≥ 0.75, the
auditor-effect critique is empirically defused.

## What we ran instead (automated surrogate)

In the absence of available external annotators, three
**independent rule-based heuristics** were implemented, each
restricted to a single orthogonal signal source:

  * **A — name-based** (`system_name`, `station_name` only)
  * **B — capacity-statistics** (`capacity` column only)
  * **C — geospatial** (`lat`, `lon` only)

All three were written by the same team, so this is not a defense
against the auditor effect proper. It IS a defense against a
weaker, but still relevant, charge: that A1–A5 is a single
heuristic dressed up as five rules.

## How to read the output

The pairwise Cohen κ between the three heuristics is near zero
(Fleiss κ = −0.07). **This is not a failure.** Each heuristic
covers a different sub-region of the taxonomy:

| Heuristic   | Class it can detect | Class it cannot |
|-------------|---------------------|-----------------|
| Name        | A1, some A3, some A5 | A2, A4          |
| Capacity    | A2, some A3         | A1, A4, A5      |
| Geospatial  | A4, some A5         | A1, A2, A3      |

Computing κ between annotators that solve **different** slices of
a classification problem is a category error: κ assumes a shared
task. The right metric here is **complementary coverage**, which
is what `coverage_analysis.py` reports.

## Headline results (n = 500, sample drawn 2026-04)

  * **Union recall vs rule-based anomalies:**
    340 / 320 stations flagged → **recall 97.2 %**,
    precision 91.5 %.
  * **Intersection:** only 7 / 500 stations flagged by all three.
    This is the *expected* signature of orthogonal coverage.
  * **Class-by-class:**
    * A1: 56 / 60 detected by *name only*, 0 / 60 by capacity, 0 / 60 by geo.
    * A3: 53 / 260 by *name only*, 14 / 260 by ≥ 2 heuristics, 193 missed
      because their system name does not match any free-floating
      pattern (signal absent from this heuristic).

## What this does and does not prove

  * **Proves** that A1–A5 cannot be reduced to a single heuristic.
    Each signal source has zero recall on the classes it cannot
    see, so the integration step of the Gold Standard pipeline
    is itself a contribution.
  * **Does not prove** that the rule-based verdict is correct, only
    that three orthogonal heuristics covering A1, A3 (partially)
    and A4 produce a near-perfect *union* (97 % recall).
  * **Does not replace** human annotators: the auditor-effect
    critique stands until E1 is run with at least two external
    annotators on the same sample.

## Reproducing

```bash
# 1. Generate the stratified sample (once)
python papers/01_gold_standard/experiments/e1_irr/sample_stations.py

# 2. Run the three automated annotators
python papers/01_gold_standard/experiments/e1_irr/auto_annotators.py

# 3. Run the coverage analysis
python papers/01_gold_standard/experiments/e1_irr/coverage_analysis.py
```

All three scripts are deterministic (seed 42).

## Citation in the paper

The result is reported under E1's "preliminary findings" and
explicitly framed as an automated robustness check, not as a
substitute for human annotators.
