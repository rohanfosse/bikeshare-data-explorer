# bikeshare-data-explorer

[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Data licence: ODbL-1.0](https://img.shields.io/badge/data-ODbL--1.0-orange.svg)](https://opendatacommons.org/licenses/odbl/1-0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20125460.svg)](https://doi.org/10.5281/zenodo.20125460)

A reproducible audit pipeline for French bike-sharing GBFS feeds and
the **Gold Standard GBFS France** dataset it produces: 46,307
certified stations across 57 urban areas, enriched with five public
reference sources (INSEE Filosofi, BAAC, BD~TOPO, OpenStreetMap, FUB
Cycling Barometer) and shipped with an audit report, a JSON schema
and a Croissant manifest.

> Produced by the BikeShare-ICT research programme (CESI LINEACT,
> 2025-2026). Companion working paper: *Auditing Open GBFS Feeds at
> Country Scale: The Gold Standard GBFS France Protocol and Dataset*.

## Why this exists

GBFS, the *de facto* standard for shared-bike fleets, is now mandated
in France under article L.1115-1 of the transport code. The implicit
promise of this standardisation is that a GBFS feed constitutes a
*ready-to-use* research dataset. A systematic audit of the 122
French systems shows that this is misleading: nearly one third of
the raw corpus is reassigned, removed or reclassified by a
semantically-aware audit, and a single anomaly drops the Bordeaux
agglomeration from rank 2 to rank 14 in the national soft-mobility
ranking. This repository ships the dataset, the deterministic
pipeline that produced it, and the validation experiments that
support it.

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

# Load the 46,307 certified stations (Parquet, ~8 MB).
gs = load_stations()
print(gs.shape)                       # (46307, 30)

# Per-city aggregates.
by_city = city_stats(gs)
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
|:---|:---|:---|
| 1. Filter dock-based stations of one city | `gs[(gs.city=="Paris") & (gs.station_type=="docked_bike")]` | 1,507 rows for Paris |
| 2. Spatial-equity analysis with INSEE Filosofi | `gs.groupby("city").agg(n=("uid","size"), income=("revenu_median_uc","median"))` | rho ~ +0.39 Spearman |
| 3. Identify mobility deserts | low-income quartile and zero heavy-transit stops within 300 m | 1,041 stations (19.1% of dock-based) |

## Repository layout

```text
bikeshare-data-explorer/
|-- utils/                          ingestion + audit pipeline (Python library)
|   |-- data_loader.py              load and enrich the Gold Standard
|   `-- gbfs_collector.py           async GBFS crawler + pseudo-flow helper
|-- scripts/                        operational scripts (status snapshots)
|-- data/
|   |-- stations_gold_standard_final.parquet   the certified dataset (46,307 rows)
|   |-- gold_standard.croissant.json           Croissant JSON-LD manifest
|   `-- ...
|-- papers/
|   |-- 01_gold_standard/           working paper + figures + experiments
|   |   `-- experiments/
|   |       |-- baselines/          comparison vs 3 naive baselines
|   |       |-- e1_irr/             inter-rater reliability surrogates
|   |       |-- e2_sensitivity/     sigma_max sweep
|   |       |-- e4_dynamic_pilot/   2-day A6 detection on status feeds
|   |       `-- engineering_benchmark/
|   |-- 02_imd/                     companion paper on the IMD composite
|   `-- tools/generate_figures.py
|-- tests/                          pytest test suite (21 tests, < 10 s)
|-- app.py                          Streamlit dashboard
|-- paper.md / paper.bib            JOSS submission
|-- CITATION.cff                    citation metadata (GitHub picks it up)
|-- LICENSE                         MIT (code), ODbL-1.0 (data)
`-- requirements.txt
```

## Enrichment pipeline

The Gold Standard is built from raw GBFS feeds by a five-module
contextual enrichment, then audited against an A1-A5 anomaly
taxonomy formalised in the companion paper:

| Module | Axis                       | Columns produced                          | Source                  |
|:------:|:---------------------------|:------------------------------------------|:------------------------|
| 1      | OSM white-zone backfill    | `source`, `osm_node_id`                   | OpenStreetMap           |
| 2      | National topography        | `elevation_m`, `topography_roughness_index` | SRTM 30 m             |
| 3A     | Cycling continuity         | `infra_cyclable_km`, `infra_cyclable_pct` | OSM Overpass API        |
| 3B     | Cyclist safety             | `baac_accidents_cyclistes`                | BAAC 2021-2023 (ONISR)  |
| 4      | Heavy multimodality        | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | French GTFS feeds |
| 5      | Socio-economic context     | `revenu_median_uc`, `gini_revenu`, ...    | INSEE Filosofi          |

## Reproducing the audit

```bash
# Regenerate the seven figures used by the companion paper
python papers/tools/generate_figures.py

# Run the validation pilots (each writes its own outputs/)
python papers/01_gold_standard/experiments/baselines/run.py
python papers/01_gold_standard/experiments/e1_irr/sample_stations.py
python papers/01_gold_standard/experiments/e1_irr/auto_annotators.py
python papers/01_gold_standard/experiments/e1_irr/full_task_classifiers.py
python papers/01_gold_standard/experiments/e1_irr/coverage_analysis.py
python papers/01_gold_standard/experiments/e2_sensitivity/run.py
python papers/01_gold_standard/experiments/e4_dynamic_pilot/run.py
python papers/01_gold_standard/experiments/engineering_benchmark/run.py
```

All scripts are deterministic (seed = 42) and write to `outputs/`
subfolders.

## Testing

```bash
pytest tests/ -v
```

The 21-test suite (< 10 s) covers schema invariants, idempotence of
the load path, sanity of the IMD aggregate, pseudo-flow correctness
on synthetic snapshots, and baseline reproducibility.

## Contributing

Contributions are welcome on:

- New audit rules (extending the A1-A5 taxonomy)
- European generalisation (Spain MITMA, Italy MIMS, Germany Mobilithek)
- Dynamic A6 detection refinements
- Bug fixes and additional tests

Please open an issue first to discuss substantial changes. The
maintainers follow the GitHub flow: fork, branch, PR. All new code
must come with at least one passing test.

## Citation

If you use this software or the Gold Standard dataset, please cite:

```bibtex
@software{bikeshare_data_explorer_2026,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {bikeshare-data-explorer: a reproducible pipeline for
             French GBFS feeds},
  year    = {2026},
  url     = {https://github.com/rohanfosse/bikeshare-data-explorer},
  version = {1.0.0}
}
```

A GitHub "Cite this repository" button is generated from
[`CITATION.cff`](CITATION.cff) and produces BibTeX / RIS / CSL-JSON
on demand. The dataset is archived on Zenodo under concept DOI
[`10.5281/zenodo.20125460`](https://doi.org/10.5281/zenodo.20125460);
each tagged release receives its own version DOI.

## Licences

- Code: MIT (see [LICENSE](LICENSE))
- Dataset: Open Database Licence v1.0 (ODbL-1.0), see
  [`data/gold_standard.croissant.json`](data/gold_standard.croissant.json)
  for the full metadata

## Authors

- **Rohan Fosse** -- CESI Engineering School, Montpellier, France
  ([ORCID 0009-0002-2195-0198](https://orcid.org/0009-0002-2195-0198))
- **Gael Pallares** -- CESI LINEACT, Montpellier, France
  ([ORCID 0009-0002-8680-604X](https://orcid.org/0009-0002-8680-604X))

Programme BikeShare-ICT, 2025-2026.

## Acknowledgements

The authors thank the producers of the open data that made this
research possible: GBFS operators, MobilityData, OpenStreetMap,
Cerema, ONISR, INSEE, FUB and the French GTFS community.
