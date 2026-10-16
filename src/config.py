"""
Configuration management for the project.

Paths only: no experimental parameter is set here.  Every scientific setting
lives in the YAML files under ``experiments/lorenz96/``.

Raw outputs of the Lorenz-96 experiment drivers (Level 2 in the README) are
written under ``results/regenerated/raw/lorenz96/`` so that rerunning an
experiment can never overwrite the published numerical results in
``results/published/``.  Set the environment variable ``DCF_RAW_RESULTS_DIR``
to send them somewhere else, for example a scratch file system.
"""

import os
from pathlib import Path


# Find project root dynamically
# This file is in src/config.py, so project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "lorenz96"
BASELINE_EXPERIMENT_DIR = EXPERIMENTS_DIR
DIAGNOSTICS_EXPERIMENT_DIR = EXPERIMENTS_DIR

RAW_RESULTS_DIR = Path(
    os.environ.get(
        "DCF_RAW_RESULTS_DIR",
        PROJECT_ROOT / "results" / "regenerated" / "raw" / "lorenz96",
    )
).resolve()
BASELINE_RESULTS_DIR = RAW_RESULTS_DIR / "baseline"
DIAGNOSTICS_RESULTS_DIR = RAW_RESULTS_DIR / "diagnostics"
SENSITIVITY_RESULTS_DIR = RAW_RESULTS_DIR / "sensitivity"
