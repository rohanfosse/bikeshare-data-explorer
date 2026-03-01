# Tableau de bord de la micromobilité française

Tableau de bord interactif pour l'exploration et l'analyse comparative
du jeu de données Gold Standard GBFS, produit dans le cadre du projet de
recherche BikeShare-ICT (CESI, 2025-2026).

## Contexte

Les flux GBFS (*General Bikeshare Feed Specification*) agrègent en temps réel
les données de disponibilité des systèmes de vélos en libre-service.
Ce projet exploite une extraction statique de **46 000+ stations** issues
des réseaux français référencés par MobilityData et complétés par OpenStreetMap.

Ces stations brutes ont été enrichies selon un pipeline en cinq modules
(voir `notebooks/27_gold_standard_enrichment.ipynb`) afin de produire
le jeu de données dit « Gold Standard » : chaque station y est caractérisée
par des métriques spatiales et contextuelles calculées dans un rayon de **300 m**,
seuil standard pour l'analyse du dernier kilomètre.

## Pipeline d'enrichissement

| Module | Axe | Colonnes produites | Source |
|:------:|:----|:-------------------|:-------|
| 1 | Comblement des zones blanches OSM | `source`, `osm_node_id` | OpenStreetMap |
| 2 | Topographie nationale (SRTM 30 m) | `elevation_m`, `topography_roughness_index` | Open-Elevation / SRTM |
| 3A | Continuité cyclable (cycleways) | `infra_cyclable_km`, `infra_cyclable_pct` | OSM Overpass API |
| 3B | Sécurité - accidents cyclistes | `baac_accidents_cyclistes` | BAAC 2021-2023 (ONISR) |
| 4 | Multimodalité lourde | `gtfs_heavy_stops_300m`, `gtfs_stops_within_300m_pct` | Flux GTFS nationaux |

Le traitement des 46 000 stations repose sur du *batch processing* avec
requêtes HTTP asynchrones (`aiohttp`) et mise en cache locale pour limiter
la charge sur les API externes.
