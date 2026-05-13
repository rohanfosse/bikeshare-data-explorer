"""Reorganize imd_trd.tex by moving subsection blocks to new sections.

Reads imd_trd.tex, parses subsection blocks by their \\label, applies a
declarative reorganization plan, and writes the result back. Idempotent
on labels: subsections keep their \\label{sec:val-X} so cross-references
remain valid.
"""
from __future__ import annotations

import re
from pathlib import Path

TEX_PATH = Path(__file__).resolve().parents[1] / "imd_trd.tex"

# Move plan: (label, target_section_id)
# target_section_id is one of: 'findings_extended', 'continental',
# 'self_correction', or 'robustness' (default; no move).
MOVE_PLAN = {
    # E18-E20: extended findings on the French panel
    "sec:val-within":       "findings_extended",
    "sec:val-deviants":     "findings_extended",
    "sec:val-counterfactual": "findings_extended",
    # E24-E26: continental scale
    "sec:val-european":     "continental",
    "sec:val-eu-stations":  "continental",
    "sec:val-imd-lite":     "continental",
    # E29-E30-E31: data-quality self-correction (order: E29, E31, E30)
    "sec:val-osm-gtfs":     "self_correction",
    "sec:val-gtfs-audit":   "self_correction",
    "sec:val-a6-rerun":     "self_correction",
}

# Order within each target section (label list)
ORDER = {
    "findings_extended": [
        "sec:val-within",
        "sec:val-deviants",
        "sec:val-counterfactual",
    ],
    "continental": [
        "sec:val-european",
        "sec:val-eu-stations",
        "sec:val-imd-lite",
    ],
    "self_correction": [
        "sec:val-osm-gtfs",
        "sec:val-gtfs-audit",
        "sec:val-a6-rerun",
    ],
}

# Section banners to insert
BANNER_BEFORE = "% =========================================================================\n"
BANNER_AFTER = "\n% =========================================================================\n"


def _section_header(title: str, label: str) -> str:
    return (
        BANNER_BEFORE +
        f"\\section{{{title}}}\n"
        f"\\label{{{label}}}\n" +
        BANNER_AFTER.lstrip()
    )


NEW_FINDINGS_INTRO = r"""
The descriptive results of Section~\ref{sec:results} answered the
volume- and income-related research questions on the supply side
and the well-being question through the metric tournament. We now
deepen the French-panel picture along three axes that were not
visible in the headline ranking: how much of the IMD variance is
hidden \emph{inside} cities rather than between them
(Section~\ref{sec:val-within-city}); which cities outperform their
socio-economic profile and what their component signature looks
like (Section~\ref{sec:val-deviants}); and how much IMD uplift the
panel could harvest under a targeted multimodal+infrastructure
intervention (Section~\ref{sec:val-counterfactual}). All three
extensions consume the same calibrated weights and the same Gold
Standard data as Section~\ref{sec:results} and report substantive
findings rather than robustness tests.
"""

NEW_CONTINENTAL_INTRO = r"""
The French-panel findings of Sections~\ref{sec:results} and
\ref{sec:findings-extended} establish the IMD as a context-aware
composite indicator that doubles the well-being predictive power of
the volumetric default. A natural next question is whether the
French structural diagnosis -- few stations per system, low
multimodality -- is a French specificity or a deviation from a
broader European norm. We answer the question on three converging
data sources: the MobilityData global GBFS catalogue
(Section~\ref{sec:val-european}), a live station-level fetch of
ten major European systems (Section~\ref{sec:val-eu-stations}),
and an OSM-based recomputation of the IMD multimodality component
on the same European cities (Section~\ref{sec:val-imd-lite}). The
three sources converge on a single answer: French bike-sharing is
not a European median, it is a continental outlier on three
distinct supply-side dimensions.
"""

NEW_SELF_CORRECTION_INTRO = r"""
The continental comparison of Section~\ref{sec:continental} ran
the multimodality count through OSM Overpass while the French
panel used the national GTFS feeds ingested by the Gold Standard
pipeline of \citet{Fosse2026gold}. We must therefore close the
methodological loop: do OSM and GTFS agree on the French panel
where both can be measured? If they disagree, which is the right
source, and what does the disagreement imply for the desert lists
and the Lyon-leverage finding made earlier in the paper?

The answer is that OSM and GTFS systematically disagree on three
to six French metropolises, that the disagreement is feed-level
(an entire municipal GTFS feed misses the tramway or metro
network), and that the disagreement \emph{retroactively corrects}
several of the paper's earlier findings. We treat this as the
self-correction layer of the paper. The diagnostic protocol
extends the data-quality programme of the companion Gold Standard
paper \citep{Fosse2026gold} from a station-level taxonomy
(A1--A5) to a feed-level taxonomy (G1--G5); the A6 patch on the
worst-affected cities reorders the Top-10 and reshapes the
mobility-desert shortlist.
"""

NEW_ROBUSTNESS_INTRO = r"""
The findings of Sections~\ref{sec:results},
\ref{sec:findings-extended} and \ref{sec:continental} rest on a
supervised calibration, on a Bayesian equity diagnostic, and on
a fixed spatial-aggregation geometry. We now report the
pre-registered validation programme that tested each of these
ingredients independently. Twelve experiments are executed; the
thirteenth (a longitudinal synthetic-control identification of
the effect of infrastructure openings) requires historical GBFS
archives still to be ingested and is listed under outstanding
tests. The data-quality experiments E29--E31 that retroactively
self-correct three earlier findings have been promoted to
Section~\ref{sec:self-correction}; the present section reports
the remaining twelve plus two honest nulls (E17 GBFS pseudo-flow,
E21 département life expectancy) and one external-triangulation
test (E22+E23).
"""


def _extract_subsection_block(text: str, label: str) -> tuple[int, int]:
    """Return (start, end) char indices of the subsection block carrying label.

    The block begins at the line containing \\subsection{...} preceding
    the \\label{label} line, and ends just before the next
    \\subsection{...} or \\section{...} header.
    """
    label_pat = re.escape(label)
    m = re.search(rf"\\label\{{{label_pat}\}}", text)
    if not m:
        raise ValueError(f"Label not found: {label}")
    # Find the \subsection line before the label
    start = text.rfind("\\subsection", 0, m.start())
    if start == -1:
        raise ValueError(f"No \\subsection preceding label {label}")
    # End: next \subsection or \section
    after = m.end()
    nxt = re.search(r"\n\\subsection\{|\n\\section\{", text[after:])
    end = after + nxt.start() + 1 if nxt else len(text)
    return start, end


def _split_blocks(text: str) -> tuple[str, dict[str, str], dict[str, tuple[int, int]]]:
    """Return (text_with_blanks, label_to_block, label_to_pos)."""
    blocks: dict[str, str] = {}
    positions: dict[str, tuple[int, int]] = {}
    # Capture in document order so we can later splice cleanly
    for label in MOVE_PLAN:
        start, end = _extract_subsection_block(text, label)
        blocks[label] = text[start:end]
        positions[label] = (start, end)
    return text, blocks, positions


def _splice_removal(text: str, positions: dict[str, tuple[int, int]]) -> str:
    """Remove the moved blocks. Returns new text with cuts."""
    spans = sorted(positions.values(), key=lambda p: p[0], reverse=True)
    out = text
    for s, e in spans:
        out = out[:s] + out[e:]
    return out


def reorganize() -> None:
    text = TEX_PATH.read_text(encoding="utf-8")
    text, blocks, positions = _split_blocks(text)

    print(f"Extracted {len(blocks)} subsection blocks for relocation:")
    for label in blocks:
        b = blocks[label]
        title_m = re.search(r"\\subsection\{([^}]+)\}", b)
        title = title_m.group(1) if title_m else "?"
        print(f"  [{label}] -> {MOVE_PLAN[label]}  ({title!r}, {len(b)} chars)")

    # Remove blocks from original positions
    text2 = _splice_removal(text, positions)

    # Build the three new sections to insert before the (former) Robustness section
    findings_extended = (
        _section_header("Extended findings on the French panel",
                        "sec:findings-extended") +
        NEW_FINDINGS_INTRO + "\n" +
        "\n".join(blocks[lbl] for lbl in ORDER["findings_extended"])
    )
    continental = (
        _section_header("Continental-scale findings",
                        "sec:continental") +
        NEW_CONTINENTAL_INTRO + "\n" +
        "\n".join(blocks[lbl] for lbl in ORDER["continental"])
    )
    self_correction = (
        _section_header("Data-quality audit and self-correction",
                        "sec:self-correction") +
        NEW_SELF_CORRECTION_INTRO + "\n" +
        "\n".join(blocks[lbl] for lbl in ORDER["self_correction"])
    )

    # Inject all three before \section{Robustness and validation}
    rob_pat = re.search(r"\n% =+\n\\section\{Robustness and validation\}", text2)
    if not rob_pat:
        # fall back: just look for the section command
        rob_pat = re.search(r"\\section\{Robustness and validation\}", text2)
        if not rob_pat:
            raise RuntimeError("Robustness section header not found")
        insert_at = rob_pat.start()
    else:
        insert_at = rob_pat.start() + 1  # after the leading newline

    insertion = findings_extended + "\n\n" + continental + "\n\n" + self_correction + "\n\n"
    text3 = text2[:insert_at] + insertion + text2[insert_at:]

    # Rename the (now shrunk) Robustness section and replace its intro paragraph
    # Robustness intro replacement
    rob_section_pat = re.compile(
        r"\\section\{Robustness and validation\}.*?\\subsection\{Out-of-sample",
        re.DOTALL,
    )
    new_rob_section = (
        "\\section{Robustness programme}\n"
        "\\label{sec:validation}\n"
        + NEW_ROBUSTNESS_INTRO + "\n"
        "\\subsection{Out-of-sample"
    )
    # Use lambda to avoid backslash-escape interpretation in the replacement.
    text4, n_sub = rob_section_pat.subn(lambda m: new_rob_section, text3, count=1)
    if n_sub == 0:
        print("WARNING: Robustness intro not replaced (regex miss)")
    else:
        print("Robustness section intro replaced.")

    TEX_PATH.write_text(text4, encoding="utf-8")
    print(f"Wrote {TEX_PATH}")
    print(f"Final length: {len(text4):,} chars (was {len(text):,})")


if __name__ == "__main__":
    reorganize()
