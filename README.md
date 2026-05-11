# bikeshare-data-explorer

[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Data licence: ODbL-1.0](https://img.shields.io/badge/data-ODbL--1.0-orange.svg)](https://opendatacommons.org/licenses/odbl/1-0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20125460.svg)](https://doi.org/10.5281/zenodo.20125460)

A reproducible audit pipeline for the GBFS bike-sharing standard,
shipping two complementary artefacts. The local product is the
**GBFS France Audit Catalogue** (46,307 station records from the 123
GBFS systems inventoried on `transport.data.gouv.fr`, of which 5,442
are dock-based stations fully validated, 39,235 are free-floating
anchors typed and annotated, and 1,630 are car-sharing entries
relabelled). Each station is enriched with twelve derived variables
from five reference sources (INSEE Filosofi, BAAC, BD TOPO,
OpenStreetMap, FUB Cycling Barometer). The same audit pipeline is
then applied at world scale to the **1,509 GBFS systems of the
MobilityData canonical catalogue** (48 countries), flagging 204
systems under A1-A5 and surfacing two additional candidate classes
(A6, A7) that affect another 215 systems and 70,176 stations.

> Produced by the BikeShare-ICT research programme (CESI LINEACT,
> 2025-2026). The release was previously circulated under the
> working title *Gold Standard GBFS France*; that scope description
> overstated coverage of the free-floating subset and has been
> revised to *Audit Catalogue* in v1.0.

## Scope and what is actually audited

The release is honest about what it audits and what it does not.

| Subset                   | Stations | Status                                                       |
| ------------------------ | -------: | ------------------------------------------------------------ |
| Dock-based               |    5,442 | Fully audited against A1-A5 with reversible exclusion log    |
| Free-floating            |   39,235 | Typed and annotated, not rigorously audited at station level |
| Car-sharing (Citiz)      |    1,630 | Relabelled `carsharing`, excluded from bike-sharing use      |
| **Total certified rows** |   46,307 | All carry a typed `station_type` enum for explicit filtering |

The dock-based subset spans 64 cities and 65 GBFS systems; the
free-floating subset spans 35 cities and 41 GBFS systems; the
union covers 97 cities and 123 systems out of the 142 candidates
inventoried on the national portal (14 fail technical ingestion;
5 are excluded as out-of-perimeter under class A5).

## Why this exists

GBFS is mandated in France under article L.1115-1 of the transport
code. The implicit assumption is that a GBFS feed constitutes a
ready-to-use research dataset. A systematic audit of the 123 French
systems documents that 22 systems exhibit at least one of five
recurring anomaly classes (A1 out-of-domain inclusion, A2
placeholder capacity, A3 structural over-capacity on free-floating
fleets, A4 geospatial error, A5 out-of-perimeter coverage). The
audit pipeline was then applied to the 1,509 systems of the
MobilityData canonical catalogue, which expanded the picture to two
new candidate classes (A6 zero-capacity dock, A7 null-capacity
field) and confirmed that the anti-patterns are global rather than
French-specific (nextbike drives the Czech hotspot exactly as Pony
drives the French free-floating hotspot, and Dott propagates
`capacity = NaN` across all of its international deployments). On the
raw corpus, 30.9 % of stations (95 % bootstrap CI [30.5, 31.3]) are
reassigned, removed or reclassified by a semantically-aware audit.
The single A3 reclassification of the *Pony* free-floating fleet
moves the Bordeaux agglomeration from rank 2 to rank 14 in the
national soft-mobility ranking computed in the companion IMD paper,
which quantifies the cost of bypassing the audit. This repository
ships the data product, the deterministic pipeline that produced it,
and the validation experiments that support it.

## Installation

```bash
git clone https://github.com/rohanfosse/bikeshare-data-explorer.git
cd bikeshare-data-explorer
pip install -r requirements.txt
```

Python >= 3.10 is required.

## Quickstart

```python
from utils.data_loader import (
    load_stations, city_stats, compute_imd_cities,
)

# Load the 46,307 certified rows (Parquet, ~8 MB on disk).
gs = load_stations()
print(gs.shape)                          # (46307, 30)

# Filter to the fully-audited dock-based subset.
dock = gs[gs.station_type == "docked_bike"]
print(dock.shape)                        # (5442, 30)

# Per-city aggregates on the dock-based subset.
by_city = city_stats(dock)
print(by_city.head())

# Composite Indice de Mobilite Douce (IMD) at city level.
imd = compute_imd_cities(gs)
print(imd[["city", "IMD"]].sort_values("IMD", ascending=False).head(10))
```

A Streamlit dashboard ships alongside the library:

```bash
streamlit run app.py
```

### Three reuse patterns

| Pattern | Snippet | Outcome |
| :--- | :--- | :--- |
| 1. Filter dock-based stations of one city | `gs[(gs.city=="Paris") & (gs.station_type=="docked_bike")]` | 1,507 rows for Paris |
| 2. Spatial-equity analysis with INSEE Filosofi | `gs.groupby("city").agg(n=("uid","size"), income=("revenu_median_uc","median"))` | rho ~ +0.39 Spearman |
| 3. Identify mobility deserts | Q1 income and zero heavy-transit stops within 300 m | 1,041 stations (19.1 % of dock-based) |

## Global audit (MobilityData canonical catalogue)

The audit pipeline is also applied to the entire world inventory of
GBFS feeds maintained by MobilityData. Headline numbers below;
per-system results in
[`papers/01_gold_standard/experiments/e5_europe/massive_audit_results.csv`](papers/01_gold_standard/experiments/e5_europe/massive_audit_results.csv)
and a country-level aggregation in
[`papers/01_gold_standard/experiments/e5_europe/massive_audit_summary.json`](papers/01_gold_standard/experiments/e5_europe/massive_audit_summary.json).

| Metric                                    | Value                                                |
| ----------------------------------------- | ---------------------------------------------------- |
| Systems in the MobilityData catalogue     | 1,509                                                |
| Reachable                                 | 1,421                                                |
| Publish `station_information`             | 917                                                  |
| **Flagged by at least one A1-A5 class**   | **204**                                              |
| Additional systems caught by A6 candidate | 14                                                   |
| Additional systems caught by A7 candidate | 215                                                  |
| Countries covered                         | 48                                                   |
| Hotspot countries                         | CZ (nextbike), CH (15 operators), DE (21 car-sharing) |

Side finding: the French national portal
`transport.data.gouv.fr` indexes only 123 of the 255 French entries
listed by MobilityData, so the regulatory pipeline is less complete
than the international catalogue. Reproduce the audit with:

```bash
python papers/01_gold_standard/experiments/e5_europe/massive_audit.py
```

## Repository layout

```text
bikeshare-data-explorer/
|-- utils/                          ingestion + audit pipeline (Python library)
|   |-- data_loader.py              load and enrich the audited catalogue
|   `-- gbfs_collector.py           threaded GBFS crawler + pseudo-flow helper
|-- scripts/                        operational scripts (status snapshots)
|-- data/
|   |-- stations_gold_standard_final.parquet   the certified dataset (46,307 rows)
|   |-- gold_standard.croissant.json           Croissant JSON-LD manifest
|   `-- ...
|-- papers/
|   `-- 01_gold_standard/
|       |-- gold_standard.tex                 SCITEPRESS conference draft
|       |-- gold_standard_patterns.tex        Patterns (Cell Press) draft
|       |-- gold_standard_scidata.tex         Scientific Data (Nature) draft
|       |-- figures/                          7 figures including the visual abstract
|       |-- references.bib
|       `-- experiments/
|           |-- baselines/          comparison against 3 audit strategies
|           |-- e1_irr/             rule-based robustness on a 500-station sample
|           |-- e2_sensitivity/     sigma_max sweep
|           |-- e4_dynamic_pilot/   2-day station_status anomaly pilot
|           |-- e5_europe/          cross-country transferability (Bicing, Oslo)
|           `-- engineering_benchmark/
|-- tests/                          pytest test suite (21 tests, < 10 s)
|-- app.py                          Streamlit dashboard
|-- CITATION.cff                    citation metadata (GitHub picks it up)
|-- LICENSE                         MIT (code), ODbL-1.0 (data)
`-- requirements.txt
```

Note that the dataset filename (`stations_gold_standard_final.parquet`)
and the Croissant manifest filename retain the legacy `gold_standard`
slug for DOI continuity. The conceptual rename to *Audit Catalogue*
applies to the project framing, the paper drafts, and any new artefact.

## Audit pipeline

The certified catalogue is built from raw GBFS feeds by a six-step
purging pipeline plus a five-module contextual enrichment, against
the A1-A5 anomaly taxonomy formalised in the companion paper.

### Purging steps

1. `S1` Relabel car-sharing systems (A1).
2. `S2` Detect placeholder capacities (zero variance on non-zero `c`, A2).
3. `S3` Recompute actual capacity and reassign topologically (A3).
4. `S4` Apply national geofilter (A4 perimeter).
5. `S5` Drop topological outliers beyond `sigma_max = 3` (A4 outliers).
6. `S6` Label systems with fewer than `N_min = 20` dock-based stations.

### Enrichment modules

| Module | Axis                    | Columns produced                                      | Source                 |
| :----: | :---------------------- | :---------------------------------------------------- | :--------------------- |
|   1    | OSM white-zone backfill | `source`, `osm_node_id`                               | OpenStreetMap          |
|   2    | National topography     | `elevation_m`, `topography_roughness_index`           | SRTM 30 m              |
|   3A   | Cycling continuity      | `infra_cyclable_km`, `infra_cyclable_pct`             | OSM Overpass API       |
|   3B   | Cyclist safety          | `baac_accidents_cyclistes`                            | BAAC 2021-2023 (ONISR) |
|   4    | Heavy multimodality     | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | French GTFS feeds      |
|   5    | Socio-economic context  | `revenu_median_uc`, `gini_revenu`, `revenu_d1`, ...   | INSEE Filosofi         |

Idempotence is enforced by a unit test executed in continuous
integration. Excluded stations are preserved in
`rejected_stations.parquet` together with the exclusion motive.

## Reproducing the audit

```bash
# Regenerate the figures used by the papers
python papers/01_gold_standard/figures/make_visual_abstract.py

# Run the validation pilots (each writes its own outputs/)
python papers/01_gold_standard/experiments/baselines/run.py
python papers/01_gold_standard/experiments/e1_irr/sample_stations.py
python papers/01_gold_standard/experiments/e1_irr/auto_annotators.py
python papers/01_gold_standard/experiments/e1_irr/full_task_classifiers.py
python papers/01_gold_standard/experiments/e1_irr/coverage_analysis.py
python papers/01_gold_standard/experiments/e2_sensitivity/run.py
python papers/01_gold_standard/experiments/e4_dynamic_pilot/run.py
python papers/01_gold_standard/experiments/engineering_benchmark/run.py

# E5 (cross-country): fetch the foreign GBFS payloads first
curl -o "$TEMP/bicing_root.json"     https://barcelona.publicbikesystem.net/customer/gbfs/v2/gbfs.json
curl -o "$TEMP/bicing_stations.json" https://barcelona.publicbikesystem.net/customer/gbfs/v2/en/station_information
curl -o "$TEMP/bicing_vtypes.json"   https://barcelona.publicbikesystem.net/customer/gbfs/v2/en/vehicle_types
python papers/01_gold_standard/experiments/e5_europe/pilot_bicing_barcelona.py
```

All scripts are deterministic (seed = 42) and write to `outputs/`
subfolders. The full audit report is regenerated automatically on
every release.

## Testing

```bash
pytest tests/ -v
```

The 21-test suite (< 10 s) covers schema invariants, idempotence of
the load path, sanity of the IMD aggregate, pseudo-flow correctness
on synthetic snapshots, and baseline reproducibility.

## Papers

Three concurrent drafts target three submission tracks; the SCITEPRESS
version is the working paper, while the two others are journal-format
adaptations of the same content.

| Draft                                                | Target                   | Words | Status             |
| ---------------------------------------------------- | ------------------------ | ----: | ------------------ |
| `papers/01_gold_standard/gold_standard.tex`          | SCITEPRESS (conference)  | 9,125 | Working paper      |
| `papers/01_gold_standard/gold_standard_patterns.tex` | Patterns (Cell Press)    | 6,131 | Draft v1           |
| `papers/01_gold_standard/gold_standard_scidata.tex`  | Scientific Data (Nature) | 4,153 | Recommended target |

The Scientific Data draft is currently the most natural fit:
the contribution is a versioned, audited reference dataset shipped
with technical validation, which is the Data Descriptor format.

## Validation roadmap

Each known limitation of the v1.0 release is paired with a
falsifiable experiment whose protocol, success criterion and pilot
status are documented under
[`papers/01_gold_standard/experiments/`](papers/01_gold_standard/experiments/).
Four of the six experiments include a completed pilot; the remaining
priorities are the human inter-rater reliability of E1, the full
4-dimensional threshold sweep of E2, the twelve-month temporal
stability of E3, and the free-floating-native protocol of E6.

## Citation

```bibtex
@software{bikeshare_data_explorer_2026,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {bikeshare-data-explorer: a reproducible pipeline for
             the GBFS France Audit Catalogue},
  year    = {2026},
  url     = {https://github.com/rohanfosse/bikeshare-data-explorer},
  version = {1.0.0},
  doi     = {10.5281/zenodo.20125460}
}
```

The GitHub *Cite this repository* button uses
[`CITATION.cff`](CITATION.cff) and exports to BibTeX, RIS or
CSL-JSON. The dataset is archived on Zenodo under concept DOI
[`10.5281/zenodo.20125460`](https://doi.org/10.5281/zenodo.20125460);
each tagged release receives its own version DOI.

## Licences

- Code: MIT (see [LICENSE](LICENSE))
- Dataset: Open Database Licence v1.0 (ODbL-1.0), full metadata in
  [`data/gold_standard.croissant.json`](data/gold_standard.croissant.json)

## Authors

- **Rohan Foss\'e**, CESI Engineering School, Montpellier, France
  ([ORCID 0009-0002-2195-0198](https://orcid.org/0009-0002-2195-0198))
- **Ga\"el Pallares**, CESI LINEACT, Montpellier, France
  ([ORCID 0009-0002-8680-604X](https://orcid.org/0009-0002-8680-604X))

BikeShare-ICT research programme, 2025-2026.

## Acknowledgements

The authors thank the producers of the open data on which this
research depends: GBFS operators, MobilityData, OpenStreetMap,
Cerema, ONISR, INSEE, FUB and the French GTFS community.
