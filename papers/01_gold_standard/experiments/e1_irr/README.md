# E1 — Inter-Rater Reliability of the A1–A5 Taxonomy

Implementation package of experiment **E1** of the Gold Standard
validation roadmap (Section 8 of the paper). This directory contains
everything an external annotator needs to participate, plus the
scoring scripts that turn three independent annotations into
publishable Cohen's κ and Fleiss κ values.

## What E1 does

The Gold Standard relies on a rule-based taxonomy of five anomaly
classes (A1–A5). The risk is the **auditor effect**: the suspicion
that the rules were designed *around* the very systems they
identify. E1 defuses that risk by asking three annotators who did
**not** participate in the design of the protocol to classify the
same stratified sample of 500 stations from the textual definitions
alone.

The experiment is considered successful if:

- Fleiss κ ≥ 0.75 across the five classes
- the rule-based classifier reaches a recall ≥ 0.90 on A1–A3
  against the adjudicated labels.

A pilot with two annotators on n=120 stations has been completed
(Cohen's κ = 0.81 [0.73, 0.89], recall = 0.94); the full run with
three annotators on the 500-station sample is being executed.

## Workflow

```text
1. Researcher runs   python sample_stations.py
   →  outputs       e1_sample_v1.csv             (stratified sample)
                    e1_answer_key_v1.csv         (researcher-only)

2. Annotators each   fill annotator_NN.csv from e1_sample_v1.csv

3. Researcher runs   python score_kappa.py
                       --annotators annotator_01.csv
                                    annotator_02.csv
                                    annotator_03.csv
                       --answer-key e1_answer_key_v1.csv
   →  outputs       e1_kappa_report.json         (final metrics)
                    e1_disagreements.csv         (for adjudication)
```

## Files

| File                       | Audience       | Purpose                                  |
|----------------------------|----------------|------------------------------------------|
| `README.md`                | everyone       | this protocol                            |
| `annotation_guide.md`      | annotators     | textual A1–A5 definitions + 5 examples   |
| `sample_stations.py`       | researchers    | stratified-sample generator              |
| `score_kappa.py`           | researchers    | Cohen's κ + Fleiss κ + recall            |
| `annotator_template.csv`   | annotators     | empty CSV produced by `sample_stations`  |

## Stratification

The sample is drawn proportionally to declared `station_type` and
**over-sampled** for systems flagged A2 or A3 by the rule-based
audit, so that the pilot covers the most contested classes. The
seed is fixed (`numpy=42`, `python=42`) so that the same researcher
re-running the sampler obtains the same 500 stations.

## What annotators see / do not see

Annotators see, per station: `station_id`, `system_id`,
`station_name`, `city`, `lat`, `lon`, `capacity`,
`is_virtual_station` (boolean) and the raw GBFS `vehicle_type` if
present. They **do not** see: the rule-based verdict, the
`station_type` of the Gold Standard, or any field derived from the
hybridisation pipeline. Annotators classify each station as one of:

```text
ok    A1    A2    A3    A4    A5
```

Multiple labels are allowed (an A2 anomaly can also be A3, for
instance); annotators are asked to enter the *most specific*
applicable class and, optionally, secondary classes separated by
`+`.

## Adjudication

Disagreements between annotators are resolved by a fourth blind
reviewer (no access to the verdict, full GBFS payload available).
The adjudicated labels become the **ground truth** against which
the rule-based classifier is scored for precision and recall.

## Compensation

External annotators are compensated by the research programme.
Allow approximately three to five hours of annotation work for the
500-station sample.

## Citation

If you use this package, please cite the companion paper:

> Fossé R., Pallares G. (2026). *Building a Gold Standard for
> Bike-Sharing Systems: GBFS Auditing, Multi-Source Hybridisation
> and Reproducibility Conditions for Quantitative Urban Mobility
> Research.* SCITEPRESS, working paper.
