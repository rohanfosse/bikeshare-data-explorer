---
license: odbl
language:
  - en
  - fr
pretty_name: GBFS Audit Catalogue
size_categories:
  - 10K<n<100K
task_categories:
  - tabular-classification
  - tabular-regression
tags:
  - bike-sharing
  - micromobility
  - gbfs
  - data-quality
  - audit
  - smart-city
  - urban-mobility
  - reproducibility
  - FAIR
  - france
configs:
  - config_name: default
    data_files:
      - split: train
        path: stations_gold_standard_final.parquet
dataset_info:
  features:
    - name: uid
      dtype: string
    - name: system_id
      dtype: string
    - name: city
      dtype: string
    - name: lat
      dtype: float64
    - name: lon
      dtype: float64
    - name: station_type
      dtype: string
    - name: capacity_raw
      dtype: float64
    - name: capacity_audited
      dtype: float64
    - name: flag_A1
      dtype: bool
    - name: flag_A2
      dtype: bool
    - name: flag_A3
      dtype: bool
    - name: flag_A4
      dtype: bool
    - name: flag_A5
      dtype: bool
    - name: flag_A6
      dtype: bool
    - name: flag_A7
      dtype: bool
    - name: operator_name
      dtype: string
    - name: audit_confidence
      dtype: string
  splits:
    - name: train
      num_examples: 46307
---

# GBFS Audit Catalogue

A reproducible audit of 1,509 GBFS bike-sharing feeds worldwide, with the resulting 46-column reference dataset for 46,307 certified stations across 123 French operators.

## TL;DR

```python
from datasets import load_dataset
gs = load_dataset("rohanfosse/gbfs-audit-catalogue", split="train").to_pandas()

# High-confidence dock-based stations
clean = gs[(gs.station_type == "docked_bike") & (gs.audit_confidence == "high")]
print(len(clean))  # 5,402

# Per-operator anomaly rates
gs.groupby("operator_name").agg(
    n=("uid", "size"),
    A3_rate=("flag_A3", "mean"),
    A7_rate=("flag_A7", "mean"),
).sort_values("n", ascending=False).head(10)
```

## What is this?

The General Bikeshare Feed Specification (GBFS) is the open standard that French bike-sharing operators must publish on `transport.data.gouv.fr` under the 2019 Mobility Orientation Law (LOM). The standard guarantees syntactic interoperability but **not** semantic consistency: identical fields carry mutually incompatible meanings across operators.

This dataset is the output of a systematic audit of the 1,509 GBFS systems catalogued by MobilityData worldwide. Seven recurring anomaly classes (A1–A7) are detected at the row level; 30.9% of the raw French stations are reclassified, removed, or relabelled. The remaining 46,307 stations are released here with per-row anomaly flags and contextual enrichment so that researchers can reuse the audited data without rerunning the pipeline.

## Schema (46 columns)

| Group | Columns | Source |
|---|---|---|
| Identifiers | `uid`, `station_id`, `system_id`, `system_name`, `source` | GBFS |
| Spatial | `lat`, `lon`, `city`, `commune_name`, `code_commune`, `region_id` | GBFS + INSEE COG |
| Station description | `station_name`, `address`, `capacity`, `n_stations_system` | GBFS |
| **Audit pipeline (11)** | `station_type`, `capacity_raw`, `capacity_audited`, `flag_A1`–`flag_A7`, `operator_name`, `audit_confidence`, `fetched_at` | This work |
| **Network geometry (5)** | `dist_to_nearest_station_m`, `n_stations_within_500m`, `n_stations_within_1km`, `nearest_system_dist_m`, `catchment_density_per_km2` | KNN on this work |
| Topography | `elevation_m`, `topography_roughness_index` | IGN BD ALTI |
| Cycling infrastructure | `infra_cyclable_km`, `infra_cyclable_pct` | BD TOPO 300 m buffer |
| Safety | `baac_accidents_cyclistes` | ONISR BAAC 500 m, 5 yr |
| Multimodal access | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | National GTFS aggregation |
| Socio-economy | `revenu_median_uc`, `gini_revenu`, `revenu_d1`, `ecart_interquar`, `part_menages_voit0` | INSEE Filosofi |
| Modal share | `part_velo_travail` | INSEE Recensement |

## The seven anomaly classes

| Class | Name | Signature | FR systems | Global systems |
|---|---|---|---|---|
| A1 | Out-of-domain inclusion | car-sharing advertised as BSS | 14 | 46 |
| A2 | Placeholder capacity | constant non-zero c across stations | 3 | 48 |
| A3 | Structural over-capacity | conditional averaging on free-floating | 8 | 33 |
| A4 | Geospatial error | transposed coords or >3σ outliers | 3.8% stations | 81 |
| A5 | Out-of-perimeter | system area >50,000 km² or overseas | 5 | 17 |
| A6 | Zero-capacity dock | ≥1% stations with c=0 | 0 | 14 |
| A7 | Null capacity field | ≥50% stations with c=NaN | 19 (FF) | 215 |

## Provenance

- **Raw source**: 142 candidate GBFS feeds from `transport.data.gouv.fr` (FR national portal) + the MobilityData canonical catalogue (1,509 systems globally).
- **Audit code**: Python, MIT licence, [github.com/rohanfosse/bikeshare-data-explorer](https://github.com/rohanfosse/bikeshare-data-explorer).
- **Reference Zenodo deposit**: [10.5281/zenodo.20125460](https://doi.org/10.5281/zenodo.20125460) (concept DOI, current release 1.0.0).

## Citation

If you use the catalogue in your research, please cite both the paper and the Zenodo deposit:

```bibtex
@article{Fosse2026gbfs,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {Auditing {GBFS} bike-sharing feeds at country and global scale:
             A reproducible anomaly taxonomy for open mobility data},
  journal = {Computer Standards \& Interfaces},
  year    = {2026},
  note    = {Under review}
}

@dataset{Fosse2026gbfsdata,
  author    = {Foss\'e, Rohan and Pallares, Ga\"el},
  title     = {{GBFS Audit Catalogue} v1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20125460}
}
```

## Licence

- **Data**: Open Data Commons Open Database License (ODbL) v1.0 — share-alike, attribution required.
- **Code**: MIT.

## Issues, contributions, contact

GitHub issues for bugs, schema requests or new audit classes:
<https://github.com/rohanfosse/bikeshare-data-explorer/issues>

Lead contact: Rohan Fossé (`rfosse@cesi.fr`), CESI LINEACT, Montpellier, France.
