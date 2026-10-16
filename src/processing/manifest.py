"""
SHA-256 manifest of the published results (``results/published/MANIFEST.json``).

``scripts/verify_results.py`` and ``tests/test_reproduction.py`` check every
published file against it, so an accidental change to the numbers behind the
paper is caught.  It is rewritten only by
``scripts/extract_results.py --target published``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROVENANCE = {
    "lorenz96/calibration_N10.npz":
        "Figures 1-2. Reduced by src/processing/lorenz96.py (extract_calibration) "
        "from the 100 per-seed shards written by src/experiments/baseline.py with "
        "EXTENDED_FILTER_ORDER (original cache: results/baseline_extended/seeds).",
    "lorenz96/coverage_ranks_N10.npz":
        "Figure 2. Reduced (extract_coverage) from the 100 per-seed shards of "
        "src/experiments/coverage.py (original cache: results/diagnostics/coverage_v1).",
    "lorenz96/ensemble_size_sweep.npz":
        "Figure 3. Reduced (extract_ensemble_sweep) from the per-size shards of "
        "src/experiments/baseline.py at N in {5,10,15,20,25,30} (original cache: "
        "results/sensitivity/ensemble_size/N_*).",
    "lorenz96/hierarchical_bias_variance.npz":
        "Figure 4. Reduced (extract_hierarchical) from the 585 (truth seed, filter) "
        "checkpoints of src/experiments/bias_variance_mc.py --hierarchical (original "
        "cache: results/diagnostics/bias_variance_analysis_100x100, 1.2 GB).",
    "lorenz96/n_kappa_bootstrap.npz":
        "Figure 5. Reduced (extract_n_kappa) from the 1500 cells of "
        "src/experiments/n_kappa_bias_variance.py run grid_aae8ed2c (708 MB); "
        "bootstrap exactly as in the original analysis notebook.",
    "lorenz96/restart_law_2fbc05cb06f87eaa.npz":
        "Level 2 input (optional). The restart forecast law used by the (N, kappa) "
        "run, calibrated by src/experiments/restart_vs_cycled.py (run full_907c3cfc). "
        "Recalibrating from scratch reproduces the same identifier.",
}
RUN_PROVENANCE = (
    "Figures 6-8, Table 2. Per-window diagnostics of one shallow-water run, written "
    "on TACC Vista by the FEniCSx solver that produced the experiment (not part of "
    "this repository; the experiment is reproduced from these outputs). Each file "
    "records its run configuration under 'config'."
)
TABLE_PROVENANCE = (
    "Figures 6-8, Table 2. Built from shallow_water/runs/ by the scripts ported to "
    "src/processing/shallow_water.py (the original files, generated under Python 3.11; "
    "rebuilding under Python >= 3.12 changes some means in the last bit)."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path) -> Path:
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(root).as_posix()
        entry = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if rel in PROVENANCE:
            entry["provenance"] = PROVENANCE[rel]
        elif rel.startswith("shallow_water/runs/"):
            entry["provenance"] = RUN_PROVENANCE
        elif rel.startswith("shallow_water/"):
            entry["provenance"] = TABLE_PROVENANCE
        files[rel] = entry
    manifest = {
        "description": "Processed numerical results behind the figures of 'A "
                       "Data-Consistent Approach to Ensemble Filtering'. Checked by "
                       "scripts/verify_results.py and tests/test_reproduction.py.",
        "files": files,
    }
    path = root / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
