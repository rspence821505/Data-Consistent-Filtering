#!/usr/bin/env python
"""
Reduce raw experiment outputs to the processed numerical results that the
figure scripts read -- the interface between Level 2 and Level 1.

``run_all_experiments.py`` calls this automatically; run it by hand to
re-reduce existing Lorenz-96 checkpoints::

    python scripts/extract_results.py                       # everything available
    python scripts/extract_results.py --only calibration coverage
    python scripts/extract_results.py --shallow-water       # rebuild the inlet tables

Lorenz-96 checkpoints are read from ``results/regenerated/raw/lorenz96/`` (or
``--raw-dir``) and reduced into ``results/regenerated/lorenz96/``.

The shallow-water experiment is reproduced from its saved run outputs only (it
is not rerun).  ``--shallow-water`` rebuilds the inlet tables -- the regime map
behind Figure 6 and Table 2 -- from the published per-run JSON files into
``results/regenerated/shallow_water/``, where ``verify_results.py`` compares
them with the published tables.

The published results in ``results/published/`` are never written unless
``--target published --i-know-this-overwrites-published`` is given; that is how
they were created from the original checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_RESULTS_DIR  # noqa: E402
from src.paths import PUBLISHED_RESULTS_DIR, REGENERATED_RESULTS_DIR  # noqa: E402
from src.processing import lorenz96, shallow_water  # noqa: E402
from src.processing.manifest import write_manifest  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_RESULTS_DIR,
                        help=f"Raw Lorenz-96 checkpoints (default: {RAW_RESULTS_DIR}).")
    parser.add_argument("--target", default="regenerated",
                        help="Result set to write: 'regenerated' (default) or a directory.")
    parser.add_argument("--only", nargs="+", choices=list(lorenz96.EXTRACTORS),
                        default=None, help="Lorenz-96 reductions to run (default: all "
                        "whose raw outputs exist).")
    parser.add_argument("--shallow-water", action="store_true",
                        help="Only rebuild the shallow-water tables from the published "
                             "run outputs.")
    parser.add_argument("--i-know-this-overwrites-published", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.target == "regenerated":
        target = REGENERATED_RESULTS_DIR
    elif args.target == "published":
        target = PUBLISHED_RESULTS_DIR
    else:
        target = Path(args.target).expanduser().resolve()
    publishing = target.resolve() == PUBLISHED_RESULTS_DIR.resolve()
    if publishing and not args.i_know_this_overwrites_published:
        parser.error("refusing to overwrite results/published/.")
    target.mkdir(parents=True, exist_ok=True)

    if args.shallow_water:
        if publishing:
            parser.error("the published inlet tables are the original files and are "
                         "not rebuilt in place.")
        shallow_water.build_all(PUBLISHED_RESULTS_DIR / "shallow_water" / "runs",
                                target / "shallow_water")
        return 0

    raw = Path(args.raw_dir)
    selected = args.only
    if selected is None:
        # Everything whose raw outputs are present.
        present = {
            "calibration": (raw / "baseline_extended" / "seeds").is_dir(),
            "coverage": (raw / "diagnostics" / "coverage_v1" / "seeds").is_dir(),
            "ensemble_sweep": (raw / "sensitivity" / "ensemble_size").is_dir(),
            "hierarchical": (raw / "diagnostics" / "bias_variance_analysis_100x100").is_dir(),
            "n_kappa": (raw / "diagnostics" / "n_kappa_bias_variance").is_dir(),
        }
        selected = [name for name, ok in present.items() if ok]
        missing = [name for name, ok in present.items() if not ok]
        if missing:
            print(f"[lorenz96] no raw outputs for {missing} under {raw}; skipped")
    if selected:
        lorenz96.extract_all(raw, target / "lorenz96", only=selected)

    if publishing:
        print(f"rewrote {write_manifest(PUBLISHED_RESULTS_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
