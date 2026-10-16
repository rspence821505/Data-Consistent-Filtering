"""
Level 1: regenerate the figures of the paper from processed numerical results.

One module per figure, numbered as in the paper.  Each exposes

* ``OUTPUT`` -- the file name used by ``\\includegraphics`` in the manuscript
  (``OUTPUTS`` when the original script also wrote a PDF, in save order),
* ``SAVE_KWARGS`` -- the ``savefig`` arguments the original code used,
* ``STYLE`` -- the matplotlib style sheet the original notebook loaded, if any,
* ``build(results_root, save)`` -- draws the figure, calls ``save(fig)`` at the
  point where the original code saved it, and returns the printed summary.

No module reruns an experiment; each reads files under
``<results_root>/lorenz96/`` or ``<results_root>/shallow_water/`` only.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt

from src.figures._common import figure_style
from src.paths import PUBLISHED_RESULTS_DIR

#: paper figure number -> module name
FIGURES = {
    1: "fig1_calibration_ribbons",
    2: "fig2_coverage_reliability",
    3: "fig3_rmse_by_ensemble_size",
    4: "fig4_hierarchical_bias_variance",
    5: "fig5_n_kappa_surfaces",
    6: "fig6_inlet_setup",
    7: "fig7_inlet_trajectories",
    8: "fig8_inlet_endpoint_error",
}


def figure_module(number: int):
    return importlib.import_module(f"src.figures.{FIGURES[int(number)]}")


def outputs(module):
    return tuple(getattr(module, "OUTPUTS", (module.OUTPUT,)))


def results_for(module, results_root: Path) -> Path:
    """The result set a figure is drawn from.

    The shallow-water figures (``PUBLISHED_ONLY``) always use the published run
    outputs: that experiment is reproduced at Level 1 only.
    """
    return PUBLISHED_RESULTS_DIR if getattr(module, "PUBLISHED_ONLY", False) else Path(results_root)


def render(number: int, results_root: Path, out_dir: Path):
    """Draw one figure into ``out_dir``; return (written paths, summary text)."""
    module = figure_module(number)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig):
        for name in outputs(module):
            path = out_dir / name
            fig.savefig(path, **module.SAVE_KWARGS)
            written.append(path)

    with figure_style(module.STYLE):
        try:
            summary = module.build(results_for(module, results_root), save)
        finally:
            plt.close("all")
    if not written:
        raise RuntimeError(f"Figure {number} was built but never saved.")
    return written, summary
