"""Deep-dive on the Czech Republic hotspot of A1-A5 anomalies.

Question: among the 1254 non-French GBFS systems, Czech Republic
concentrates 12 A3 + 16 A2 detections (25 systems flagged out of 45).
What operator(s) cause this? Is it a single-operator anti-pattern
(like Pony in France) or a market-wide issue?
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "massive_audit_results.csv"


def operator_brand(name: str) -> str:
    """Extract operator brand from system name."""
    if not name:
        return "unknown"
    first = name.split()[0].lower()
    return re.sub(r"[^a-z]", "", first) or "unknown"


def main() -> None:
    with SRC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cz = [r for r in rows if r["country"] == "CZ"]
    cz_flagged = [r for r in cz if r["any_anomaly"] == "True"]
    cz_a2 = [r for r in cz_flagged if r["a2_placeholder"] == "True"]
    cz_a3 = [r for r in cz_flagged if r["a3_overcap_flag"] == "True"]
    cz_both = [r for r in cz_flagged
               if r["a2_placeholder"] == "True" and r["a3_overcap_flag"] == "True"]

    operators = Counter(operator_brand(r["name"]) for r in cz_flagged)
    operators_cz_all = Counter(operator_brand(r["name"]) for r in cz)

    summary: dict = {
        "country": "CZ",
        "audited": len(cz),
        "reachable": sum(1 for r in cz if r["reachable"] == "True"),
        "flagged": len(cz_flagged),
        "a2_placeholder": len(cz_a2),
        "a3_overcap": len(cz_a3),
        "both_a2_and_a3": len(cz_both),
        "operators_among_flagged": dict(operators.most_common()),
        "operators_among_all_cz": dict(operators_cz_all.most_common()),
        "flagged_systems": [
            {
                "name": r["name"],
                "system_id": r["system_id"],
                "stations": int(r["stations"]) if r["stations"] else 0,
                "a2": r["a2_placeholder"] == "True",
                "a3": r["a3_overcap_flag"] == "True",
                "a3_ratio": float(r["a3_overcap_ratio"]) if r["a3_overcap_ratio"] else None,
                "vehicle_form_factors": r["vehicle_form_factors"],
            }
            for r in cz_flagged
        ],
    }
    (HERE / "deep_dive_czech.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Console output (ASCII-safe to avoid Windows cp1252 issues)
    print("=" * 60)
    print(f"Czech Republic deep-dive: {len(cz)} audited, {len(cz_flagged)} flagged")
    print(f"  A2 placeholder: {len(cz_a2)}")
    print(f"  A3 over-capacity: {len(cz_a3)}")
    print(f"  Both A2+A3: {len(cz_both)}")
    print()
    print("Operators among flagged CZ systems:")
    for op, n in operators.most_common():
        print(f"  {op:20s} {n}")
    print()
    print("Operators among ALL CZ systems (for context):")
    for op, n in operators_cz_all.most_common():
        print(f"  {op:20s} {n}")
    print()
    print(f"Total Czech stations affected: "
          f"{sum(int(r['stations']) if r['stations'] else 0 for r in cz_flagged)}")


if __name__ == "__main__":
    main()
