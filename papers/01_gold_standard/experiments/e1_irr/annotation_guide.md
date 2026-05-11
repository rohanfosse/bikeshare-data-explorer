# Annotation Guide — E1 Inter-Rater Reliability of A1–A5

This guide is meant for the three external annotators who will
classify the 500-station sample. It describes the five anomaly
classes A1–A5 in plain language, gives worked examples and lists
the disambiguation rules to apply when several classes seem to
overlap.

## How to annotate

Open `e1_sample_v1.csv` in your spreadsheet editor of choice. Each
row is one station. Fill the `label` column with one of the six
values below; if several apply, enter the most specific one first
and the secondary ones after a `+` (e.g. `A2+A3`).

```text
ok   the station is a genuine bike-sharing station with no anomaly
A1   the station is not a bike-sharing station (car-sharing, scooters, ...)
A2   the declared capacity is a placeholder (constant unrealistic value)
A3   the station is a free-floating anchor, not a physical dock
A4   the GPS coordinates are inconsistent with the rest of the system
A5   the station is outside the research perimeter (overseas, rural macro)
```

## A1 — Out-of-domain inclusion

A1 captures the situation where a non-BSS service is published on
the GBFS portal. The signal is the **`vehicle_type` field** or
the **system name**: if the system is *Citiz*, *Yego*, *Tier*,
*Lime*, *Voi*, *Free2move* or anything that suggests car-sharing,
scooter-sharing, mopeds or e-scooters, the verdict is A1. The
station might still have a legitimate `station_id`, a coherent
GPS position and a sensible capacity — none of that disqualifies
the A1 verdict, since the question is **the nature of the
vehicle**, not the quality of the station record.

*Example.* A station of `citiz_nantes_atlantique` with
`vehicle_type = car` and `capacity = 2` is A1.

## A2 — Placeholder capacity

A2 captures the situation where the operator declares the same
non-zero capacity value on every single station of the system,
regardless of the physical anchoring point. The signal is the
**zero variance** of `capacity` across the system, together with
a **non-zero** value (a system of `c = 0` everywhere is a
different problem). Typical placeholder values are 100, 50 or 10.

*Example.* A *Pony* station in Nice with `capacity = 100`, while
all the other Pony Nice stations also declare `capacity = 100`,
is A2.

*Disambiguation with A3.* A2 is **structural** (the operator
publishes a fictitious constant); A3 is **statistical** (the
operator estimates a capacity from non-empty stations). In
practice, A2 is detected at the **system level** (one verdict for
all stations of a system), whereas A3 is detected at the
**station level** within a free-floating fleet. When in doubt,
prefer A2 if the variance is exactly zero across the whole
system, A3 otherwise.

## A3 — Structural over-capacity (floating-anchor)

A3 captures the situation where a free-floating fleet (Pony,
Cykleo, …) advertises virtual stations in GBFS, the term
"station" designating in this context the GPS position of an
anchoring point currently occupied by a vehicle. The signal is
the **`is_virtual_station: true`** flag (when present) or, in its
absence, a `station_id` that includes a per-vehicle suffix
(`pony_33_a91`, `cykleo_freefloat_002`).

*Example.* `pony_33_a91` in Bordeaux with
`is_virtual_station = true`, `capacity = 12` and a `station_name`
of `Bordeaux #a91` is A3.

## A4 — Geospatial error

A4 captures three sub-types:

- **Transposed coordinates.** `lat` and `lon` have been swapped.
  Easy to spot: the station claims to sit somewhere in
  $\varphi \in [-6, 10]$ and $\lambda \in [41, 52]$ (the
  geofilter mirror).
- **Out of perimeter.** The station sits outside the national
  bounding box altogether (typically projected onto the Greenwich
  meridian, the equator or `(0, 0)`).
- **Topological outlier.** The station sits within the national
  perimeter but is isolated more than three standard deviations
  from the centroid of its own system. Visually, on a map of the
  system, the station appears as a lone dot far from the dense
  cluster of legitimate stations.

*Example.* A *V'Lille* station with `lat = 50.6`, `lon = 3.05` is
fine; a *V'Lille* station with `lat = 3.05`, `lon = 50.6` is A4
(transposed).

## A5 — Out of perimeter (overseas, macro-regional)

A5 captures systems whose geographical coverage is incompatible
with urban analysis, either because they are located **outside
metropolitan France** (Saint-Denis-de-La-Réunion, Cayenne,
Pointe-à-Pitre, …) or because they declare an **operating surface
exceeding 50,000 km²** (rural micromobility schemes such as
Basque Country or Grand Est). Note that A5 is a property of the
**system**, not of an individual station: if you flag a station
A5, it is because the whole system is out of perimeter.

## Disambiguation flowchart

```text
Is the vehicle a bike (or e-bike, cargo bike)?   no  -->  A1
                                                  |
                                                 yes
                                                  |
Is the system outside metropolitan France or >50000 km^2 ?
                                                  yes -->  A5
                                                  |
                                                  no
                                                  |
Is capacity exactly constant across the system?  yes -->  A2
                                                  |
                                                  no
                                                  |
Is is_virtual_station = true (or vehicle-suffix station_id)?
                                                  yes -->  A3
                                                  |
                                                  no
                                                  |
Are the coordinates inconsistent with the system?
                                                  yes -->  A4
                                                  |
                                                  no
                                                  |
                                                  ok
```

## Notes

- Take your time. The 500-station sample should require about
  three to five hours of work in total.
- When a station leaves you genuinely unsure, write `?` in the
  `label` column rather than guessing — the adjudicator will
  resolve the case with full payload access.
- Do **not** consult the rest of the corpus when annotating a
  given station: each row is to be classified on its own
  evidence.
- Do **not** look at any companion paper, the audit report, or
  the rule-based verdict during the annotation phase.
