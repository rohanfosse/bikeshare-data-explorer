"""Score the E1 inter-rater reliability experiment.

Given the annotator CSVs (one per annotator) and the answer key
produced by ``sample_stations.py``, this script computes:

  - pairwise Cohen's kappa for every pair of annotators
  - three-way Fleiss kappa across all annotators
  - bootstrap 95 % CIs for both
  - the precision and recall of the rule-based classifier against
    the adjudicated labels (majority vote, ties broken by the
    first annotator's label)
  - a CSV of disagreements for adjudication

Run from the repository root once all annotator CSVs are in:

    python papers/01_gold_standard/experiments/e1_irr/score_kappa.py \
        --annotators papers/01_gold_standard/experiments/e1_irr/annotator_01.csv \
                     papers/01_gold_standard/experiments/e1_irr/annotator_02.csv \
                     papers/01_gold_standard/experiments/e1_irr/annotator_03.csv \
        --answer-key papers/01_gold_standard/experiments/e1_irr/e1_answer_key_v1.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _paths import outputs_dir  # noqa: E402

LABELS: Final[tuple[str, ...]] = ("ok", "A1", "A2", "A3", "A4", "A5")
BOOTSTRAP_RESAMPLES: Final[int] = 10_000
SEED: Final[int] = 42


def _parse_label(raw: str) -> str:
    """Return the primary label of an annotator cell.

    Multi-label cells like ``A2+A3`` are reduced to the first
    token. Unknown or missing values are mapped to ``"?"``.
    """
    if not isinstance(raw, str) or raw.strip() == "":
        return "?"
    primary = raw.strip().split("+", 1)[0].strip()
    return primary if primary in LABELS else "?"


def _load_annotator(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    if "row_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{path.name}: must contain columns 'row_id' and 'label'"
        )
    df = df[["row_id", "label"]].copy()
    df["label"] = df["label"].apply(_parse_label)
    return df.sort_values("row_id").reset_index(drop=True)


def _cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa between two annotator vectors."""
    df = pd.DataFrame({"a": a.values, "b": b.values})
    df = df[(df["a"] != "?") & (df["b"] != "?")]
    if df.empty:
        return float("nan")
    cats = sorted(set(df["a"]).union(set(df["b"])))
    confusion = pd.crosstab(df["a"], df["b"]).reindex(
        index=cats, columns=cats, fill_value=0
    )
    n = confusion.values.sum()
    if n == 0:
        return float("nan")
    p_observed = np.trace(confusion.values) / n
    p_a = confusion.sum(axis=1) / n
    p_b = confusion.sum(axis=0) / n
    p_expected = float((p_a.values * p_b.values).sum())
    if p_expected >= 1.0:
        return 1.0
    return (p_observed - p_expected) / (1 - p_expected)


def _fleiss_kappa(merged: pd.DataFrame) -> float:
    """Fleiss' kappa across all annotator columns of ``merged``.

    ``merged`` has one row per station and one column per annotator.
    """
    annot_cols = [c for c in merged.columns if c.startswith("annot_")]
    counts = np.zeros((len(merged), len(LABELS)), dtype=int)
    for i, row in enumerate(merged[annot_cols].itertuples(index=False)):
        c = Counter(label for label in row if label != "?")
        for j, label in enumerate(LABELS):
            counts[i, j] = c.get(label, 0)
    row_sums = counts.sum(axis=1)
    valid = row_sums > 1
    counts = counts[valid]
    row_sums = row_sums[valid]
    if len(counts) == 0:
        return float("nan")
    # Per-row agreement.
    p_i = ((counts ** 2).sum(axis=1) - row_sums) / (row_sums * (row_sums - 1))
    p_bar = p_i.mean()
    # Marginal label probabilities.
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = float((p_j ** 2).sum())
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def _bootstrap_ci(
    fn,
    *vectors: pd.Series,
    n: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """Return a 95 % bootstrap CI for the metric ``fn(*vectors)``."""
    rng = np.random.default_rng(seed)
    size = len(vectors[0])
    samples: list[float] = []
    indices_all = np.arange(size)
    for _ in range(n):
        idx = rng.choice(indices_all, size=size, replace=True)
        resampled = [v.iloc[idx].reset_index(drop=True) for v in vectors]
        samples.append(fn(*resampled))
    arr = np.array([s for s in samples if not np.isnan(s)])
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _adjudicate(merged: pd.DataFrame) -> pd.Series:
    """Majority-vote adjudication; ties go to annotator 1."""
    annot_cols = [c for c in merged.columns if c.startswith("annot_")]

    def vote(row: pd.Series) -> str:
        labels = [row[c] for c in annot_cols if row[c] != "?"]
        if not labels:
            return "?"
        c = Counter(labels)
        top, count = c.most_common(1)[0]
        # On tie, fall back to the first annotator's label.
        tied = [label for label, cnt in c.items() if cnt == count]
        if len(tied) == 1:
            return top
        return row[annot_cols[0]]

    return merged.apply(vote, axis=1)


def _precision_recall(
    rule_labels: pd.Series, gold_labels: pd.Series, target_classes: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Per-class precision / recall against the adjudicated labels."""
    out: dict[str, dict[str, float]] = {}
    for cls in target_classes:
        tp = int(((rule_labels == cls) & (gold_labels == cls)).sum())
        fp = int(((rule_labels == cls) & (gold_labels != cls)).sum())
        fn = int(((rule_labels != cls) & (gold_labels == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        out[cls] = {"precision": precision, "recall": recall,
                    "support": tp + fn}
    return out


def _build_disagreements(merged: pd.DataFrame) -> pd.DataFrame:
    """Return the rows on which annotators disagree, for adjudication."""
    annot_cols = [c for c in merged.columns if c.startswith("annot_")]

    def differing(row: pd.Series) -> bool:
        labels = [row[c] for c in annot_cols if row[c] != "?"]
        return len(set(labels)) > 1

    mask = merged.apply(differing, axis=1)
    return merged[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", nargs="+", required=True,
                        help="annotator CSV files (one per annotator)")
    parser.add_argument("--answer-key", required=True,
                        help="researcher-only CSV from sample_stations.py")
    parser.add_argument("--out", default=None,
                        help="report output path (defaults next to inputs)")
    args = parser.parse_args()

    annotator_paths = [Path(p) for p in args.annotators]
    answer_key_path = Path(args.answer_key)
    out_dir = Path(args.out).parent if args.out else outputs_dir(__file__)

    print(f"Loading {len(annotator_paths)} annotator CSVs...")
    annotators = [_load_annotator(p) for p in annotator_paths]

    print(f"Loading answer key: {answer_key_path.name}")
    key = pd.read_csv(answer_key_path, encoding="utf-8")

    # Merge into one wide table: row_id, annot_01, annot_02, ...
    merged = annotators[0].rename(columns={"label": "annot_01"})
    for i, a in enumerate(annotators[1:], start=2):
        merged = merged.merge(
            a.rename(columns={"label": f"annot_{i:02d}"}), on="row_id"
        )
    merged = merged.merge(key, on="row_id")

    # Pairwise Cohen's kappa.
    annot_cols = [c for c in merged.columns if c.startswith("annot_")]
    pairwise: dict[str, dict[str, float]] = {}
    for a, b in combinations(annot_cols, 2):
        k = _cohen_kappa(merged[a], merged[b])
        lo, hi = _bootstrap_ci(_cohen_kappa, merged[a], merged[b])
        pairwise[f"{a} vs {b}"] = {"kappa": k, "ci_low": lo, "ci_high": hi}

    # Three-way Fleiss kappa.
    fleiss = _fleiss_kappa(merged)

    # Adjudicate and score the rule-based classifier.
    merged["adjudicated"] = _adjudicate(merged)
    rule_recall = _precision_recall(
        merged["rule_based_label"], merged["adjudicated"],
        target_classes=("A1", "A2", "A3"),
    )

    report = {
        "n_stations": int(len(merged)),
        "n_annotators": len(annot_cols),
        "pairwise_cohen_kappa": pairwise,
        "fleiss_kappa": fleiss,
        "rule_based_precision_recall": rule_recall,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    out_path = out_dir / "e1_kappa_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")

    disagreements = _build_disagreements(merged)
    disagreement_path = out_dir / "e1_disagreements.csv"
    disagreements.to_csv(disagreement_path, index=False, encoding="utf-8")
    print(f"Wrote {disagreement_path} "
          f"({len(disagreements)} rows for adjudication)")


if __name__ == "__main__":
    main()
