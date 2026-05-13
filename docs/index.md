# GBFS Audit Catalogue

**A reproducible audit of 1,509 open bike-sharing feeds across 48 countries.**

The General Bikeshare Feed Specification (GBFS) is the open standard that municipal bike-sharing operators publish on national open-data portals. The standard guarantees syntactic interoperability — but not semantic consistency. The audit reported here exposes **seven recurring anomaly classes (A1–A7)** that together reclassify **30.9 %** of the raw French stations and flag 215 systems worldwide that the v1.0 taxonomy would otherwise miss.

The result is the **GBFS Audit Catalogue v1.0**, a 46-column reference dataset for 46,307 certified stations across 123 French operators, with per-row anomaly flags and contextual enrichment from INSEE, BAAC, BD TOPO, BD ALTI and the national GTFS aggregator.

---

## Quick start

### Load via Hugging Face Datasets

```python
from datasets import load_dataset
gs = load_dataset("rohanfosse/gbfs-audit-catalogue", split="train").to_pandas()
```

### Load via Zenodo (no auth)

```python
import pandas as pd
gs = pd.read_parquet(
    "https://zenodo.org/records/20125460/files/stations_gold_standard_final.parquet"
)
```

### Inspect the audit

```python
# High-confidence dock-based stations
clean = gs[(gs.station_type == "docked_bike") & (gs.audit_confidence == "high")]
print(len(clean))  # 5,402

# Operator-driven anomaly hotspots
gs.groupby("operator_name").agg(
    n=("uid", "size"),
    A3_rate=("flag_A3", "mean"),
    A7_rate=("flag_A7", "mean"),
).sort_values("n", ascending=False).head(10)
```

---

## Resources

| Resource | URL |
|---|---|
| **Paper (CSI 2026, under review)** | preprint coming |
| **Zenodo deposit (DOI)** | [10.5281/zenodo.20125460](https://doi.org/10.5281/zenodo.20125460) |
| **Hugging Face Datasets** | [rohanfosse/gbfs-audit-catalogue](https://huggingface.co/datasets/rohanfosse/gbfs-audit-catalogue) |
| **Interactive dashboard** | [gbfs-audit.streamlit.app](https://gbfs-audit.streamlit.app) |
| **Source code & audit pipeline** | [github.com/rohanfosse/bikeshare-data-explorer](https://github.com/rohanfosse/bikeshare-data-explorer) |
| **Companion notebook (8 recipes)** | [catalogue_recipes.ipynb](https://github.com/rohanfosse/bikeshare-data-explorer/blob/main/papers/01_gold_standard/notebooks/catalogue_recipes.ipynb) |

---

## The seven anomaly classes

| Class | Name | Signature | FR | Global |
|---|---|---|---|---|
| A1 | Out-of-domain inclusion | car-sharing labelled as BSS | 14 | 46 |
| A2 | Placeholder capacity | constant non-zero `c` on all stations | 3 | 48 |
| A3 | Structural over-capacity | conditional averaging on free-floating | 8 | 33 |
| A4 | Geospatial error | transposed coords or >3σ outliers | 3.8 % stns | 81 |
| A5 | Out-of-perimeter | system area >50,000 km² or overseas | 5 | 17 |
| A6 | Zero-capacity dock | ≥1 % stations declare `c = 0` | 0 | 14 |
| A7 | Null capacity field | ≥50 % stations declare `c = NaN` | 19 (FF) | 215 |

---

## Schema (46 columns at a glance)

- **5 identifiers** — `uid`, `station_id`, `system_id`, `system_name`, `source`
- **5 spatial / admin** — `lat`, `lon`, `city`, `commune_name`, `code_commune`, `region_id`
- **4 station description** — `station_name`, `address`, `capacity`, `n_stations_system`
- **11 audit pipeline** — `station_type`, `capacity_raw`, `capacity_audited`, `flag_A1`–`flag_A7`, `operator_name`, `audit_confidence`, `fetched_at`
- **5 network geometry** — KNN distances, density within buffers
- **2 topography** — `elevation_m`, `topography_roughness_index`
- **2 cycling infrastructure** — `infra_cyclable_km`, `infra_cyclable_pct`
- **1 safety** — `baac_accidents_cyclistes`
- **2 multimodal access** — `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct`
- **5 socio-economic context** — `revenu_median_uc`, `gini_revenu`, `revenu_d1`, `ecart_interquar`, `part_menages_voit0`
- **1 modal share** — `part_velo_travail`

The full machine-readable schema ships as a JSON Schema, a Frictionless Data Package descriptor, a DCAT-AP record and a Croissant manifest in the Zenodo deposit.

---

## Citation

```bibtex
@article{Fosse2026gbfs,
  author  = {Foss\'e, Rohan and Pallares, Ga\"el},
  title   = {Auditing GBFS bike-sharing feeds at country and global scale:
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

---

## Licence

- **Data**: Open Data Commons Open Database License (ODbL) v1.0
- **Code**: MIT

## Contact

Rohan Fossé — `rfosse@cesi.fr` — CESI LINEACT, Montpellier, France.

Issues and contributions on [GitHub](https://github.com/rohanfosse/bikeshare-data-explorer/issues).
