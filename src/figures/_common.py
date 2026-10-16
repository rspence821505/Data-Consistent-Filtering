"""
Shared plumbing for the figure modules: loading processed results, isolating
matplotlib state per figure, and writing the output files.

Each figure module exposes ``build(results_root) -> (figure, summary)`` and is
driven by ``scripts/reproduce_figures.py``.  ``summary`` is the numerical
report the original notebook printed next to the figure, so the numbers quoted
in the paper can be checked against the same data the figure is drawn from.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; never needs a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.paths import lorenz96_dir, require, shallow_water_dir  # noqa: E402

STYLE_SHEET = Path(__file__).with_name("publication_style.mplstyle")

_EXTRACT_HINT = (
    "Processed results are created by scripts/extract_results.py from the raw "
    "outputs of scripts/run_all_experiments.py (Level 2)."
)


def load_lorenz96(results_root: Path, filename: str):
    """Open one processed Lorenz-96 archive (allow_pickle=False)."""
    path = require(lorenz96_dir(results_root) / filename, _EXTRACT_HINT)
    return np.load(path, allow_pickle=False)


def shallow_water_root(results_root: Path) -> Path:
    return require(shallow_water_dir(results_root), _EXTRACT_HINT)


@contextlib.contextmanager
def figure_style(style_sheet: Path | None = None):
    """Matplotlib state for one figure, restored afterwards.

    Starts from matplotlib's defaults (plus the publication style sheet the
    Lorenz-96 notebooks loaded with ``plt.style.use``), so no figure can inherit
    settings from one drawn before it.
    """
    font_log = logging.getLogger("matplotlib.font_manager")
    level = font_log.level
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        matplotlib.use("Agg")
        if style_sheet is not None:
            plt.style.use(str(style_sheet))
        with warnings.catch_warnings():
            # A few font lookups (e.g. bold weights requested from fonts that
            # have none) fall back to the regular weight, and matplotlib logs a
            # "findfont" line each time -- exactly as it did when the paper's
            # figures were made.  Silence the noise; rendering is unchanged.
            warnings.simplefilter("ignore")
            font_log.setLevel(logging.ERROR)
            try:
                yield
            finally:
                font_log.setLevel(level)


class Report:
    """Collects the printed numerical summary of a figure."""

    def __init__(self):
        self._buffer = io.StringIO()

    def __call__(self, *args, **kwargs):
        print(*args, file=self._buffer, **kwargs)

    def text(self) -> str:
        return self._buffer.getvalue()
