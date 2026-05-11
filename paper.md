---
title: 'bikeshare-data-explorer: a reproducible pipeline to audit French GBFS feeds and produce the Gold Standard GBFS France dataset'
tags:
  - Python
  - GBFS
  - bike-sharing
  - data quality
  - audit
  - FAIR
  - reproducibility
  - soft mobility
  - urban mobility
authors:
  - name: Rohan Fossé
    orcid: 0009-0002-2195-0198
    corresponding: true
    affiliation: 1
  - name: Gaël Pallares
    orcid: 0009-0002-8680-604X
    affiliation: 2
affiliations:
  - name: CESI, CESI Engineering School, Montpellier, France
    index: 1
  - name: CESI, CESI LINEACT, Montpellier, France
    index: 2
date: 11 May 2026
bibliography: paper.bib
---

# Summary

`bikeshare-data-explorer` is an open-source Python package that ingests
all French open feeds published under the General Bikeshare Feed
Specification (GBFS) [@OMF2023], audits them against a taxonomy of
five anomaly classes (A1–A5), enriches the certified stations with
five public reference sources (INSEE Filosofi, BAAC, BD TOPO,
OpenStreetMap, FUB Cycling Barometer), and produces the **Gold
Standard GBFS France** dataset: 46,307 certified stations across 57
urban areas, distributed as a versioned Apache Parquet file with an
audit report, a JSON Schema and a Croissant JSON-LD manifest
[@Croissant2024]. A companion Streamlit application exposes the
dataset through an interactive cartographic and analytical interface.

The package is designed for two audiences. Mobility researchers can
load the dataset and run downstream analyses (composite indices,
demand modelling, spatial-equity studies) in a few lines of Python.
Data engineers can re-execute the full audit pipeline end-to-end and
modify it for other countries or other shared-mobility modes.

# Statement of need

The GBFS specification has become the *de facto* ontology for
shared-bike fleets and is now mandated in France under article L.
1115-1 of the transport code. The implicit promise of this
standardisation is that a GBFS feed constitutes a *ready-to-use*
research dataset. This promise is misleading. A systematic audit of
the 122 French GBFS systems documented in `bikeshare-data-explorer`
reveals five recurring semantic anomalies that pass schema-level
validation but invalidate downstream cross-city comparisons: 14
car-sharing systems advertised as bike-sharing fleets, three systems
declaring placeholder capacities, eight free-floating fleets reported
as dock-based stations, geospatial errors on $\sim 4\%$ of stations
and five systems whose perimeter is incompatible with urban analysis.
The cumulative effect of these anomalies is to reassign, remove or
reclassify roughly $31\%$ of the raw corpus.

While the GBFS community provides a syntactic validator
[@MobilityData2024Validator] modelled on the canonical GTFS one, no
publicly available tool addresses the semantic-validity problem at
the scale of an entire country. Mobility researchers therefore
typically rebuild a private cleaning pipeline for each study, which
weakens reproducibility and prevents cross-study comparison.
General-purpose data-validation frameworks such as Great Expectations
or Frictionless Data [@FrictionlessData2023] would catch many of the
geospatial errors but cannot detect the more subtle anomalies (A2
placeholder capacity, A3 floating-anchor over-capacity) without
domain-specific rules.

`bikeshare-data-explorer` fills that gap. It packages (i) a
versioned, FAIR-aligned reference dataset, (ii) the deterministic
pipeline that produced it, (iii) a documented taxonomy of GBFS
anomalies (A1–A5) and (iv) a falsifiable validation roadmap of six
experiments [@Fosse2026audit], three of which are already
implemented as runnable pilots in `papers/01_gold_standard/experiments/`.

# Functionality

The pipeline is organised in three layers, each importable as a
standalone Python module.

**Ingestion** (`utils.gbfs_collector`) provides an asynchronous
`GBFSCollector` class that crawls the national `transport.data.gouv.fr`
catalogue plus the MobilityData global index, discovers the
`station_information` and `station_status` feed URLs, fetches them
with `aiohttp` and persists snapshots as Parquet files under
`data/status_snapshots/<system>/<date>.parquet`. The `compute_pseudo_flows`
helper turns consecutive snapshots into per-station inflow and outflow
estimates, supporting downstream demand-modelling work.

**Auditing and enrichment** (`utils.data_loader`) loads the
certified station file, applies the six-step purging algorithm
described in [@Fosse2026audit], performs spatial joins with five
public reference sources, and exposes ready-to-query DataFrames via
`load_stations`, `city_stats`, `completeness_report`,
`compute_imd_cities` and ~20 thematic helpers. All loaders are
cached with `streamlit.cache_data` (TTL 1h) and are pure when called
outside the Streamlit runtime, so the package is usable as a plain
library.

**Validation experiments**
(`papers/01_gold_standard/experiments/`) ship runnable pilots of
the validation roadmap: an inter-rater reliability check based on
three orthogonal and three full-task classifiers (`e1_irr/`); a
single-axis $\sigma_{\max}$ sensitivity sweep on the certified set
(`e2_sensitivity/`); a 2-day dynamic A6-candidate detection pilot
on `station_status` snapshots (`e4_dynamic_pilot/`); a comparative
benchmark of the Gold Standard against three naive baselines
(`baselines/`); and an engineering footprint benchmark
(`engineering_benchmark/`). Each pilot writes a JSON report under
its own `outputs/` directory.

The Gold Standard parquet weighs 7.8 MB on disk, loads in 0.23 s
into a `pandas` DataFrame with a memory peak of 15.9 MB, and serves
typical research queries in 3–22 ms on a commodity laptop.

# Comparison with existing software

Three families of tools share part of the design space.
**GBFS-validator** [@MobilityData2024Validator] is the closest
counterpart but is purely syntactic — it does not address A2 or A3
anomalies and does not produce an enriched dataset.
**Frictionless Data** [@FrictionlessData2023] and **Great
Expectations** are general-purpose declarative data-validation
frameworks; they can implement A4 geofiltering but have no built-in
knowledge of GBFS semantics. **`bikesharingdata`** and similar
ad-hoc Python packages on PyPI provide GBFS download helpers without
auditing. `bikeshare-data-explorer` is, to our knowledge, the first
open-source package to publish a country-scale audited GBFS dataset
together with the deterministic pipeline that produced it and a
public validation roadmap.

# Acknowledgements

The authors thank the producers of open data that made this software
possible: GBFS operators, MobilityData, OpenStreetMap, Cerema, ONISR,
INSEE, FUB and the French GTFS community.

# References
