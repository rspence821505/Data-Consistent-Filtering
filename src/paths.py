"""
Repository paths shared by the processing, figure and verification code.

Two *result sets* are recognised:

``published``
    ``results/published/`` -- the processed numerical results behind the
    figures in the paper.  Committed to the repository and never written by any
    script except ``scripts/extract_results.py --target published``, which is
    how they were created.

``regenerated``
    ``results/regenerated/`` -- where rerunning the experiments (Level 2) puts
    its raw checkpoints (``raw/``) and the processed files reduced from them.
    Ignored by git.

A result set can also be given as an explicit directory with the same layout.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
PUBLISHED_RESULTS_DIR = RESULTS_DIR / "published"
REGENERATED_RESULTS_DIR = RESULTS_DIR / "regenerated"

DATA_DIR = PROJECT_ROOT / "data"
SHALLOW_WATER_MESH_H5 = DATA_DIR / "shallow_water" / "Ideal_Inlet.h5"

FIGURES_DIR = PROJECT_ROOT / "figures"
PAPER_FIGURES_DIR = FIGURES_DIR / "paper"
GENERATED_FIGURES_DIR = FIGURES_DIR / "generated"

RESULT_SETS = {
    "published": PUBLISHED_RESULTS_DIR,
    "regenerated": REGENERATED_RESULTS_DIR,
}


def resolve_results_dir(results: str | Path) -> Path:
    """Map ``"published"``, ``"regenerated"`` or a directory to a result-set root."""
    key = str(results)
    if key in RESULT_SETS:
        return RESULT_SETS[key]
    path = Path(results).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{results!r} is neither a known result set {sorted(RESULT_SETS)} "
            "nor an existing directory."
        )
    return path


def lorenz96_dir(results_root: Path) -> Path:
    return Path(results_root) / "lorenz96"


def shallow_water_dir(results_root: Path) -> Path:
    return Path(results_root) / "shallow_water"


def require(path: Path, hint: str = "") -> Path:
    """Return ``path`` or raise a FileNotFoundError that says how to create it."""
    path = Path(path)
    if not path.exists():
        message = f"Required input not found: {path}"
        if hint:
            message += f"\n{hint}"
        raise FileNotFoundError(message)
    return path
