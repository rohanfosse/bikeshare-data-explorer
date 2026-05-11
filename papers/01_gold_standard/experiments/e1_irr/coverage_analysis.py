"""Coverage analysis: are the three orthogonal heuristics complementary?

Rather than measuring inter-annotator kappa (which assumes the
three annotators try to solve the same task), this script measures
the *complementary coverage* of three independent heuristics that
look at orthogonal signal sources:

    A : name-based       (system_name, station_name)
    B : capacity-stats   (capacity column only)
    C : geospatial       (lat, lon only)

For each anomaly class A1..A5, the script reports:
  - the per-heuristic precision / recall against the rule-based
    verdict (used here as a proxy for ground truth)
  - the union coverage (at least one heuristic flags it)
  - the intersection (all heuristics agree)

If the three heuristics are complementary, we expect:
  - low pairwise kappa
  - HIGH union coverage on the union of A1..A5
  - LOW intersection (because each heuristic sees a different class)

This pattern is what justifies the integration step of the Gold
Standard pipeline: no single signal is sufficient.

Usage:
    python papers/01_gold_standard/experiments/e1_irr/coverage_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir  # noqa: E402

DIR: Final[Path] = outputs_dir(__file__)
LABELS: Final[tuple[str, ...]] = ("ok", "A1", "A2", "A3", "A4", "A5")


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DIR / name, encoding="utf-8")


def _precision_recall(
    pred: pd.Series, truth: pd.Series, target: str
) -> dict[str, float]:
    tp = int(((pred == target) & (truth == target)).sum())
    fp = int(((pred == target) & (truth != target)).sum())
    fn = int(((pred != target) & (truth == target)).sum())
    support = int((truth == target).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
    }


def main() -> None:
    a = _load("auto_annotator_A_name.csv").rename(columns={"label": "A"})
    b = _load("auto_annotator_B_capacity.csv").rename(columns={"label": "B"})
    c = _load("auto_annotator_C_geospatial.csv").rename(columns={"label": "C"})
    key = _load("e1_answer_key_v1.csv")
    df = (a.merge(b, on="row_id").merge(c, on="row_id")
            .merge(key[["row_id", "rule_based_label"]], on="row_id"))

    print("=" * 60)
    print("COMPLEMENTARY-COVERAGE ANALYSIS (n = 500)")
    print("=" * 60)
    print()
    print("Distribution of labels by source:")
    print("-" * 60)
    for col, name in [("A", "Name-based   "),
                      ("B", "Capacity-stat"),
                      ("C", "Geospatial   "),
                      ("rule_based_label", "Rule-based   ")]:
        counts = df[col].value_counts().reindex(LABELS, fill_value=0)
        print(f"  {name}:  " + "  ".join(
            f"{k}={v:>3}" for k, v in counts.items()
        ))
    print()

    # Per-heuristic, per-class precision/recall against rule-based.
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    print("Per-heuristic precision / recall against the rule-based verdict:")
    print("-" * 60)
    print(f"  {'Class':<6}{'Source':<14}"
          f"{'Precision':>10}{'Recall':>10}{'TP':>5}{'FP':>5}{'FN':>5}"
          f"{'Sup':>5}")
    for cls in ("A1", "A2", "A3", "A4", "A5"):
        metrics[cls] = {}
        for col, name in [("A", "name"),
                          ("B", "capacity"),
                          ("C", "geospatial")]:
            row = _precision_recall(df[col], df["rule_based_label"], cls)
            metrics[cls][name] = row
            print(f"  {cls:<6}{name:<14}"
                  f"{row['precision']!s:>10}{row['recall']!s:>10}"
                  f"{row['tp']:>5}{row['fp']:>5}{row['fn']:>5}"
                  f"{row['support']:>5}")
        print()

    # Union / intersection of heuristic verdicts (per row).
    print("Coverage of the union vs intersection of heuristics:")
    print("-" * 60)
    union_anomaly = (
        (df["A"] != "ok") | (df["B"] != "ok") | (df["C"] != "ok")
    )
    intersect_anomaly = (
        (df["A"] != "ok") & (df["B"] != "ok") & (df["C"] != "ok")
    )
    rule_anomaly = df["rule_based_label"] != "ok"
    print(f"  Stations flagged by rule-based pipeline: "
          f"{int(rule_anomaly.sum()):>3} / 500")
    print(f"  Stations flagged by UNION of heuristics: "
          f"{int(union_anomaly.sum()):>3} / 500")
    print(f"  Stations flagged by ALL three heuristics: "
          f"{int(intersect_anomaly.sum()):>3} / 500")
    print()
    # Recall of the union vs rule-based ground truth.
    tp_u = int((union_anomaly & rule_anomaly).sum())
    recall_u = tp_u / int(rule_anomaly.sum()) if rule_anomaly.sum() else 0
    print(f"  Union recall against rule-based anomalies: "
          f"{recall_u:.2%}")
    fp_u = int((union_anomaly & ~rule_anomaly).sum())
    precision_u = tp_u / (tp_u + fp_u) if (tp_u + fp_u) else 0
    print(f"  Union precision against rule-based anomalies: "
          f"{precision_u:.2%}")
    print()

    # Per-class: which single heuristic is responsible?
    print("Class-by-class coverage source:")
    print("-" * 60)
    for cls in ("A1", "A2", "A3", "A4", "A5"):
        truth_mask = df["rule_based_label"] == cls
        n = int(truth_mask.sum())
        if n == 0:
            print(f"  {cls}: no rule-based instances in the sample.")
            continue
        only_a = int((df[truth_mask].apply(
            lambda r: r["A"] == cls
            and r["B"] != cls and r["C"] != cls,
            axis=1)).sum())
        only_b = int((df[truth_mask].apply(
            lambda r: r["B"] == cls
            and r["A"] != cls and r["C"] != cls,
            axis=1)).sum())
        only_c = int((df[truth_mask].apply(
            lambda r: r["C"] == cls
            and r["A"] != cls and r["B"] != cls,
            axis=1)).sum())
        multi = int((df[truth_mask].apply(
            lambda r: (
                [r["A"] == cls, r["B"] == cls, r["C"] == cls].count(True) > 1
            ),
            axis=1)).sum())
        none = n - only_a - only_b - only_c - multi
        print(f"  {cls} (n={n}): "
              f"name only={only_a:>3}, capacity only={only_b:>3}, "
              f"geo only={only_c:>3}, multi={multi:>3}, missed={none:>3}")

    # Save a JSON report for paper-side reference.
    report = {
        "n_stations": 500,
        "per_class_metrics": metrics,
        "union": {
            "n_flagged": int(union_anomaly.sum()),
            "recall": round(recall_u, 3),
            "precision": round(precision_u, 3),
        },
        "intersection": {
            "n_flagged": int(intersect_anomaly.sum()),
        },
    }
    out = DIR / "e1_coverage_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
