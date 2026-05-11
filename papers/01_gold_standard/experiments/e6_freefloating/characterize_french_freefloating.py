"""
Quick characterization of the French free-floating subset of the
GBFS France Audit Catalogue, by operator. Goal: document that the
four major operators (Pony, Dott, Bird, Voi) use the GBFS
`capacity` field in four incompatible ways, and quantify the
prevalence of each pattern.

This is a partial step towards the full E6 free-floating-native
audit (planned 2027-Q2); it does not yet propose a complete F1-F5
taxonomy, but it characterizes the inputs that such a taxonomy
would have to address.

Outputs:
  - freefloating_per_operator.csv      per-operator capacity profile
  - freefloating_capacity_patterns.json categorized pattern per operator
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = Path(__file__).parent.parent.parent.parent.parent / "data" / "stations_gold_standard_final.parquet"


def operator_brand(system_id: str) -> str:
    """Extract operator brand from system_id."""
    sid = (system_id or "").lower()
    for brand in ["pony", "dott", "bird", "voi", "tier", "lime", "bolt", "wind", "cykleo"]:
        if brand in sid:
            return brand
    return "other"


def classify_capacity_pattern(cap_count: int, cap_unique: int,
                              cap_nan_pct: float, cap_mean: float) -> str:
    """Classify how an operator reports capacity on free-floating anchors."""
    if cap_nan_pct > 0.95:
        return "NaN (no capacity reported)"
    if cap_unique == 1 and cap_mean > 50:
        return "Single placeholder value"
    if cap_mean < 2:
        return "Per-vehicle ratio (<2 bikes/station)"
    if cap_unique <= 5 and cap_mean < 10:
        return "Small fleet-profile estimator"
    if cap_mean > 10:
        return "Capacity-profile estimator (conditional-averaging)"
    return "Mixed/unclear pattern"


def main() -> None:
    df = pd.read_parquet(DATA)
    ff = df[df["station_type"] == "free_floating"].copy()
    ff["brand"] = ff["system_id"].astype(str).map(operator_brand)

    rows = []
    for (brand, system_id), grp in ff.groupby(["brand", "system_id"]):
        cap = grp["capacity"]
        n_total = len(grp)
        n_nan = int(cap.isna().sum())
        cap_nz = cap.dropna()
        cap_unique = int(cap_nz.nunique()) if len(cap_nz) else 0
        cap_mean = float(cap_nz.mean()) if len(cap_nz) else 0.0
        cap_std = float(cap_nz.std()) if len(cap_nz) > 1 else 0.0
        nan_pct = n_nan / n_total if n_total else 0.0

        pattern = classify_capacity_pattern(n_total, cap_unique, nan_pct, cap_mean)
        rows.append({
            "brand": brand,
            "system_id": system_id,
            "n_stations": n_total,
            "n_cities": grp["city"].nunique(),
            "cap_nan_pct": round(nan_pct * 100, 1),
            "cap_unique_nonzero": cap_unique,
            "cap_mean": round(cap_mean, 2),
            "cap_std": round(cap_std, 2),
            "pattern": pattern,
        })

    res = pd.DataFrame(rows).sort_values(["brand", "n_stations"], ascending=[True, False])
    res.to_csv(HERE / "freefloating_per_operator.csv", index=False)

    # Pattern summary at brand level
    brand_summary = res.groupby("brand").agg(
        n_systems=("system_id", "nunique"),
        n_stations=("n_stations", "sum"),
        n_cities=("n_cities", "sum"),
        patterns=("pattern", lambda s: ", ".join(sorted(set(s)))),
    ).reset_index()
    brand_summary.to_csv(HERE / "freefloating_per_brand.csv", index=False)

    summary = {
        "total_freefloating": int(len(ff)),
        "brands_observed": sorted(ff["brand"].unique().tolist()),
        "per_brand": brand_summary.to_dict(orient="records"),
        "capacity_use_patterns_observed": sorted(res["pattern"].unique().tolist()),
        "key_findings": {
            "Pony": (res[res.brand == "pony"][["system_id", "n_stations", "cap_mean", "pattern"]]
                    .to_dict(orient="records")),
            "Dott": (res[res.brand == "dott"][["system_id", "n_stations", "cap_nan_pct", "pattern"]]
                    .to_dict(orient="records")),
            "Bird": (res[res.brand == "bird"][["system_id", "n_stations", "cap_nan_pct", "pattern"]]
                    .to_dict(orient="records")),
            "Voi": (res[res.brand == "voi"][["system_id", "n_stations", "cap_mean", "pattern"]]
                    .to_dict(orient="records")),
        },
    }
    (HERE / "freefloating_capacity_patterns.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
