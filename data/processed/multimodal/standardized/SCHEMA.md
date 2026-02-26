
# MULTIMODAL TRANSPORT DATABASE SCHEMA
Generated: 2025-12-16 10:11:13
Version: 1.0

## Tables Overview

1. bike_stations.csv         - Master bike station registry
2. tram_stations.csv         - Master tram stop registry
3. park_and_ride.csv         - Park & Ride facilities
4. bike_tram_proximity.csv   - Distance matrix (bike ↔ tram)
5. tram_frequencies.csv      - Service frequency by line/period
6. integration_zones.csv     - Multimodal integration zones

## Table Schemas

### 1. bike_stations
Primary Key: station_id
Columns:
  - station_id (PK)         : Unique identifier (BK001, BK002, ...)
  - station_name            : Clean station name
  - station_name_raw        : Original name from data source
  - latitude                : Decimal degrees
  - longitude               : Decimal degrees
  - data_source             : Origin of data
  - last_updated            : Timestamp
  - data_version            : Schema version

### 2. tram_stations
Primary Key: tram_id
Columns:
  - tram_id (PK)            : Unique identifier (TR0001, TR0002, ...)
  - tram_name               : Station name
  - latitude                : Decimal degrees
  - longitude               : Decimal degrees
  - tram_line               : Line number(s)
  - network                 : Transport network (TAM)
  - operator                : Operating company
  - shelter                 : Has shelter (yes/no/unknown)
  - bench                   : Has bench (yes/no/unknown)
  - osm_id                  : OpenStreetMap node ID
  - source                  : Data source
  - data_retrieved          : Retrieval timestamp
  - last_updated            : Timestamp
  - data_version            : Schema version

### 3. park_and_ride
Primary Key: pr_id
Columns:
  - pr_id (PK)              : Unique identifier (PR001, PR002, ...)
  - name                    : P+R facility name
  - latitude                : Decimal degrees
  - longitude               : Decimal degrees
  - capacity                : Number of parking spaces
  - tram_line               : Connected tram line(s)
  - address                 : Full address
  - accessible              : Wheelchair accessible (boolean)
  - charging_stations       : Has EV charging (boolean)
  - bike_parking            : Has bike parking (boolean)
  - zone                    : Geographic zone
  - data_created            : Creation timestamp
  - last_updated            : Timestamp
  - data_version            : Schema version

### 4. bike_tram_proximity
Foreign Keys: station_id → bike_stations, tram_id → tram_stations
Columns:
  - station_id (FK)         : References bike_stations.station_id
  - tram_id (FK)            : References tram_stations.tram_id
  - station_name            : Bike station name (denormalized for convenience)
  - tram_name               : Tram stop name (denormalized for convenience)
  - station_latitude        : Bike station coordinates
  - station_longitude       : Bike station coordinates
  - tram_latitude           : Tram stop coordinates
  - tram_longitude          : Tram stop coordinates
  - tram_line               : Tram line number
  - distance_km             : Distance in kilometers
  - distance_m              : Distance in meters
  - walkable_5min           : Within 5-minute walk (~400m)
  - walkable_10min          : Within 10-minute walk (~800m)
  - last_updated            : Timestamp
  - data_version            : Schema version

### 5. tram_frequencies
Primary Key: frequency_id
Columns:
  - frequency_id (PK)       : Unique identifier (FREQ001, ...)
  - tram_line               : Line number
  - time_period             : Period (Peak Hours/Off-Peak/Evening/Weekend)
  - frequency_minutes       : Minutes between services
  - services_per_hour       : Services per hour
  - source                  : Data source
  - data_created            : Creation timestamp
  - last_updated            : Timestamp
  - data_version            : Schema version

### 6. integration_zones
Primary Key: zone_id
Columns:
  - zone_id (PK)            : Unique identifier (MZ001, MZ002, ...)
  - zone_name               : Descriptive name
  - latitude                : Center latitude
  - longitude               : Center longitude
  - radius_km               : Radius in kilometers
  - tram_lines              : Tram lines serving zone
  - bike_stations_count     : Number of bike stations
  - pr_nearby               : Nearby P+R facility ID (FK to park_and_ride)
  - description             : Zone description
  - data_created            : Creation timestamp
  - last_updated            : Timestamp
  - data_version            : Schema version

## Relationships

bike_stations (1) ----< (N) bike_tram_proximity
tram_stations (1) ----< (N) bike_tram_proximity
park_and_ride (1) ----< (N) integration_zones [via pr_nearby]
tram_frequencies (N) ----< (1) tram_line [logical relationship]

## Usage Examples

### Join bike and tram stations via proximity:
```sql
SELECT
    b.station_name,
    t.tram_name,
    p.distance_m,
    p.walkable_5min
FROM bike_tram_proximity p
JOIN bike_stations b ON p.station_id = b.station_id
JOIN tram_stations t ON p.tram_id = t.tram_id
WHERE p.walkable_5min = true
```

### Find all bike stations near a specific tram line:
```sql
SELECT DISTINCT
    b.*
FROM bike_stations b
JOIN bike_tram_proximity p ON b.station_id = p.station_id
WHERE p.tram_line = '1' AND p.walkable_10min = true
```

### Get P+R with nearby bike stations:
```sql
SELECT
    pr.name as pr_name,
    pr.capacity,
    COUNT(DISTINCT p.station_id) as nearby_bike_stations
FROM park_and_ride pr
JOIN integration_zones z ON pr.pr_id = z.pr_nearby
JOIN bike_tram_proximity p ON z.zone_id = p.zone_id  -- conceptual
GROUP BY pr.pr_id, pr.name, pr.capacity
```

## Notes

- All coordinates are in WGS84 (EPSG:4326) decimal degrees
- Distances calculated using Haversine formula
- Walking distance assumptions: 5 km/h average speed
- Data version tracks schema changes (currently 1.0)
- last_updated tracks when data was last refreshed
