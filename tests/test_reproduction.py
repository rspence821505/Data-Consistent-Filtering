"""
Level 1 regression tests.

* The published numerical results are byte-for-byte those recorded in
  results/published/MANIFEST.json.
* Every figure of the paper is redrawn from them pixel-identically to the PNG
  used in the manuscript (figures/paper/).  Pixel identity holds with the
  pinned environment (environment.yml: matplotlib 3.11.0, freetype 2.14.3).

Run:  python -m pytest tests/test_reproduction.py -v     (~15 s)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.figures import FIGURES, render  # noqa: E402
from src.figures.compare import compare_png, describe  # noqa: E402
from src.paths import PAPER_FIGURES_DIR, PUBLISHED_RESULTS_DIR  # noqa: E402


def test_published_results_match_manifest():
    manifest = json.loads((PUBLISHED_RESULTS_DIR / "MANIFEST.json").read_text())
    for relative, info in manifest["files"].items():
        path = PUBLISHED_RESULTS_DIR / relative
        assert path.exists(), f"missing published file {relative}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], (
            f"{relative} differs from MANIFEST.json"
        )


@pytest.mark.parametrize("number", sorted(FIGURES))
def test_figure_is_pixel_identical_to_paper(number, tmp_path):
    written, summary = render(number, PUBLISHED_RESULTS_DIR, tmp_path)
    assert summary, "every figure reports a numerical summary"
    png = [path for path in written if path.suffix == ".png"][-1]
    reference = PAPER_FIGURES_DIR / png.name
    result = compare_png(png, reference)
    assert result["identical"], f"Figure {number}: {describe(result)}"
