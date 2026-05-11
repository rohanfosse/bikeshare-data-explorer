"""Shared path helpers for the experiment scripts.

Locating the repository root and the script's own output folder
should be robust to where the experiment lives in the directory
tree. We walk up from the calling script until we find ``.git`` or
``CITATION.cff``; this lets every experiment script be moved
without breaking its imports.

Usage from any experiment script:

    from _paths import repo_root, outputs_dir
    ROOT = repo_root()
    OUT = outputs_dir(__file__)
"""
from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | str | None = None) -> Path:
    """Walk up until we find a repo marker (.git or CITATION.cff)."""
    here = Path(start).resolve() if start else Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists() or (candidate / "CITATION.cff").exists():
            return candidate
    raise RuntimeError(f"Cannot find repo root walking up from {here}")


def outputs_dir(script_file: str | Path) -> Path:
    """Return ``<script_dir>/outputs/``, creating it if missing."""
    path = Path(script_file).resolve().parent / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path
