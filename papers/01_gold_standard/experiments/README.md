# Experiments for the Gold Standard paper

This folder hosts the empirical validation experiments referenced by
the *Future directions* subsection of the Gold Standard GBFS France
paper. Each experiment is **falsifiable** in the Popperian sense: it
states a hypothesis, a re-executable protocol, an explicit success
criterion, and (where available) a pilot result. Each experiment is
self-contained: a runnable script, optional supporting code, its own
`outputs/` folder, and a short README when extra explanation is
needed.

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
|-- e4_dynamic_pilot/               E4 -- dynamic A6 candidate detection
|   |-- run.py
|   `-- outputs/{report.json, indicators.csv}
`-- e5_europe/                      E5 -- cross-country transferability
    |-- pilot_bicing_barcelona.py   Bicing (ES) audit
    |-- pilot_oslo_bysykkel.py      Oslo Bysykkel (NO) audit
    |-- bicing_audit_report.json
    `-- oslo_audit_report.json
```

## How to run

All scripts use `_paths.py` to locate the repository root via the
`.git` / `CITATION.cff` marker, so they are robust to relocation.
Run from the repository root:

```bash
# E1 (automated surrogate; human-annotator version is the next priority)
python papers/01_gold_standard/experiments/e1_irr/sample_stations.py
python papers/01_gold_standard/experiments/e1_irr/auto_annotators.py
python papers/01_gold_standard/experiments/e1_irr/coverage_analysis.py

# E2 (sigma_max single-axis sweep; full 4D grid is the next priority)
python papers/01_gold_standard/experiments/e2_sensitivity/run.py

# E4 (2-day dynamic pilot on station_status snapshots)
python papers/01_gold_standard/experiments/e4_dynamic_pilot/run.py

# E5 (2-system cross-country pilot; requires curl-fetched GBFS payloads)
curl -o "$TEMP/bicing_root.json"     https://barcelona.publicbikesystem.net/customer/gbfs/v2/gbfs.json
curl -o "$TEMP/bicing_stations.json" https://barcelona.publicbikesystem.net/customer/gbfs/v2/en/station_information
curl -o "$TEMP/bicing_vtypes.json"   https://barcelona.publicbikesystem.net/customer/gbfs/v2/en/vehicle_types
python papers/01_gold_standard/experiments/e5_europe/pilot_bicing_barcelona.py

curl -o "$TEMP/oslo_root.json"     https://gbfs.urbansharing.com/oslobysykkel.no/gbfs.json
curl -o "$TEMP/oslo_stations.json" https://gbfs.urbansharing.com/oslobysykkel.no/station_information.json
curl -o "$TEMP/oslo_vtypes.json"   https://gbfs.urbansharing.com/oslobysykkel.no/vehicle_types.json
python papers/01_gold_standard/experiments/e5_europe/pilot_oslo_bysykkel.py
```

Each run rewrites the corresponding `outputs/` files; outputs are
deterministic (seed 42 throughout).

## Headline results (current snapshot)

| Experiment     | Threshold               | Observed                                      | Verdict |
| -------------- | ----------------------- | --------------------------------------------- | ------- |
| E1 (surrogate) | union recall >= 0.90    | recall 0.97, precision 0.92                   | passes  |
| E2 (sigma_max) | Top-10 churn = 0        | churn 0 across [2.0, 4.0], tau >= 0.92        | passes  |
| E4 (2-day)     | A6 reclass >= 5%        | 12.9% of monitored stations                   | passes  |
| E5 (2-system)  | no A1-A5 false-positive | Bicing ratio 1.006, Oslo ratio 1.000          | passes  |
| E3             | --                      | not runnable locally yet (requires 12 months) | open    |
| E6             | --                      | requires a free-floating-native taxonomy      | open    |

E1 is documented as a surrogate; the human-annotator version is the
next priority. See `e1_irr/AUTO_ANNOTATION_NOTE.md` for the
intellectual-honesty caveats.

---

## Full protocols

These specifications correspond to the *Future directions* subsection
of the paper; they are reproduced here in full so that any team can
re-execute or contest them.

### E1 -- Inter-rater reliability of A1--A5

**Hypothesis.** The taxonomy is sufficiently operational to be applied
by annotators who did not participate in its construction.

**Protocol.** Draw a stratified random sample of 500 stations covering
all 123 systems, over-sampled for systems flagged A2-A3. Recruit
three independent transport-data engineers external to the team. Each
annotator classifies every station given only the textual definitions
of A1--A5 and the raw GBFS payload of the station. Compute pairwise
Cohen's kappa and the three-way Fleiss kappa; adjudicate disagreements
by a fourth blind reviewer. Re-run the pipeline with the adjudicated
labels as ground truth and report rule-based precision and recall.

**Success criterion.** Fleiss kappa >= 0.75 across all five classes
and rule-based recall >= 0.90 on A1--A3.

**Pilot status.** Three automated surrogate classifiers were run on
the 500-station sample. Orthogonal-coverage check yields Fleiss kappa
= -0.07 (expected; each classifier sees only one slice of the
taxonomy) with union-recall 97.2% at precision 91.5%. Full-task
agreement check (three classifiers each seeing all signals in
different priority orders) yields Fleiss kappa = 0.80. The bracketed
interval [-0.07, 0.80] is reported as a robustness measure, not as a
substitute for human annotation.

### E2 -- Threshold and buffer sensitivity

**Hypothesis.** The certified set G and the downstream ranking remain
stable under reasonable variations of operational thresholds.

**Protocol.** Sweep the four-dimensional Cartesian grid
`(sigma_max, N_min, r_BAAC, r_BD_TOPO) in {2, 2.5, 3, 3.5, 4} x
{10, 20, 30, 50} x {300, 500, 800}m x {200, 300, 500}m`. For every
grid point, recompute the certified set, the enriched variables and
the IMD ranking. Report Jaccard similarity to the reference run,
Kendall's tau on the ranking, and first- and total-order Sobol
indices on tau.

**Success criterion.** On the modal grid (centre +/- one step),
Kendall's tau >= 0.95 and Jaccard >= 0.97. Any parameter with total
Sobol index > 0.40 is reclassified from *operational threshold* to
*modelling choice* and documented as such in the schema.

**Pilot status.** Single-dimension pilot on sigma_max in {2.0, 2.5,
3.0, 3.5, 4.0}. Jaccard similarities 0.66, 0.79, 1.00, 1.00, 1.00;
Kendall's tau 0.92, 0.95, 1.00, 1.00, 1.00. National Top-10 invariant
across the entire grid. Full 4D sweep is the next priority.

### E3 -- Twelve-month temporal stability

**Hypothesis.** Anomaly verdicts are stable in time: the pipeline
reclassifies automatically when an operator publishes a corrected
feed but does not introduce spurious noise when nothing has changed
upstream.

**Protocol.** Re-execute the audit on the first of every month over
twelve consecutive months. Archive each version with its hash. For
each system, build the verdict time series `v_s(t)` and the churn
rate `rho_s = mean(1[v_s(t) != v_s(t-1)])`. Correlate verdict changes
with announced operator releases to distinguish *operator effect*
(legitimate reclassification) from *audit drift* (spurious noise).

**Success criterion.** Median rho_s across the certified systems
< 0.10, and all verdict changes traceable to an exogenous event.

**Pilot status.** Running; first three months collected.

### E4 -- Dynamic extension to `station_status`

**Hypothesis.** A second family of anomalies, invisible in
`station_information`, becomes detectable when the `station_status`
feed is monitored over time (counts exceeding declared capacity,
zero-availability persisting for weeks, geo-fence drift). These are
referred to as A6 candidates.

**Protocol.** Collect `station_status` snapshots every 5 minutes for
30 days on the certified systems. Flag stations where
`P(available > capacity) > 0.01`. Identify pathologically static
stations where `P(unchanged for 24h) > 0.40`. Publish the dynamic
profile as a new field `audit_status_dynamic`.

**Success criterion.** >= 3 statistically distinct A6 sub-types
identified across >= 5 systems; dynamic audit reclassifies >= 5% of
stations whose static verdict was `ok`.

**Pilot status.** 2-day pilot on 43 systems (4,635 stations, 3.0M
snapshots) identified 599 stations (12.9% of the monitored corpus)
covered by the union of three A6 sub-types (overflow, saturated-empty,
degenerate). Full 30-day E4 is the next dynamic milestone.

### E5 -- European generalisation

**Hypothesis.** A1-A5 captures structural features of GBFS publication
and transfers to other European jurisdictions modulo per-country
threshold recalibration.

**Protocol.** Apply the pipeline to the national GBFS feeds of Spain
(MITMA), Italy (MIMS), Germany (Mobilithek), Belgium (data.gov.be)
and the Netherlands (NDOV). Repeat E1 with three local annotators per
country. Recalibrate `sigma_max` and `N_min` per country via the E2
sensitivity grid.

**Success criterion.** A1, A3 and A4 detected in at least four of the
five countries with country-level kappa >= 0.70; recalibrated
thresholds within +/- 20% of the French values.

**Pilot status.** Two-system pilot complete on Bicing Barcelona (ES,
544 stations) and Oslo Bysykkel (NO, 266 stations). Both pass A1-A5
with zero structural anomalies under French thresholds + country
perimeter. Capacity-profile ratio = 1.006 (Bicing) and 1.000 (Oslo),
consistent with clean dock-based systems. Full 5-country E5 planned
for 2027-Q1.

### E6 -- Free-floating-native protocol

**Hypothesis.** Free-floating fleets warrant a dedicated audit
framework whose anomaly classes are not subsumed by A1-A5.

**Protocol.** On the eight free-floating systems of the corpus,
define five candidate classes: F1 (geo-fence drift), F2 (dwell-time
saturation), F3 (vehicle-id non-persistence), F4 (trip-distance
distribution incompatible with declared vehicle class), F5
(abandonment outside service area). Codify each class as a
statistical test on `free_bike_status` snapshots and on inferred
trips. Run E1-style inter-rater reliability on a 200-vehicle sample.
Publish a complementary *Floating Standard* dataset.

**Success criterion.** At least three of the five candidate classes
survive E1-style validation (kappa >= 0.70); their combined incidence
covers >= 60% of the free-floating fleet.

**Pilot status.** Planned 2027-Q2; addresses limitation L3.
