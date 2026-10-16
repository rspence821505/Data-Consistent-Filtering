#!/usr/bin/env python
"""
Level 1: regenerate the figures of "A Data-Consistent Approach to Ensemble
Filtering" from saved numerical results.

No experiment is rerun.  By default the figures are drawn from the published
numerical results in ``results/published/`` and written to
``figures/generated/published/``; ``--results regenerated`` draws them from the
outputs of ``scripts/run_all_experiments.py`` instead.

Examples
--------
    python scripts/reproduce_figures.py                    # all eight figures
    python scripts/reproduce_figures.py --figures 1 4 7    # a subset
    python scripts/reproduce_figures.py --compare          # also diff against figures/paper/
    python scripts/reproduce_figures.py --results regenerated
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.figures import FIGURES, figure_module, outputs, render  # noqa: E402
from src.figures.compare import compare_png, describe  # noqa: E402
from src.paths import (  # noqa: E402
    GENERATED_FIGURES_DIR,
    PAPER_FIGURES_DIR,
    PUBLISHED_RESULTS_DIR,
    resolve_results_dir,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None,
    )
    parser.add_argument(
        "--results", default="published",
        help="Result set to plot: 'published' (default), 'regenerated', or a "
             "directory with the same layout.",
    )
    parser.add_argument(
        "--figures", type=int, nargs="+", default=sorted(FIGURES),
        choices=sorted(FIGURES), metavar="N",
        help="Paper figure numbers to draw (default: all, 1-8).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: figures/generated/<result set>/).",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare every PNG with the corresponding figure of the paper in "
             "figures/paper/ and exit non-zero if any differs.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Do not print the numerical summaries.",
    )
    args = parser.parse_args(argv)

    results_root = resolve_results_dir(args.results)
    label = args.results if args.results in ("published", "regenerated") else Path(results_root).name
    out_dir = args.out_dir or (GENERATED_FIGURES_DIR / label)
    out_dir = Path(out_dir).resolve()

    print(f"Result set : {results_root}")
    print(f"Output dir : {out_dir}\n")

    summaries = []
    failures = []
    mismatches = []
    for number in args.figures:
        module = figure_module(number)
        t0 = time.perf_counter()
        print(f"Figure {number}: {module.TITLE}")
        if getattr(module, "PUBLISHED_ONLY", False) and Path(results_root) != PUBLISHED_RESULTS_DIR:
            print("  (drawn from results/published/: the shallow-water experiment is "
                  "reproduced from its saved run outputs only)")
        try:
            written, summary = render(number, results_root, out_dir)
        except FileNotFoundError as exc:
            print(f"  SKIPPED -- {exc}\n")
            failures.append(number)
            continue
        for path in written:
            print(f"  wrote {path.relative_to(out_dir.parent) if out_dir.parent in path.parents else path}")
        print(f"  ({time.perf_counter() - t0:.1f} s)")
        if args.compare:
            png = [p for p in written if p.suffix == ".png"][-1]
            reference = PAPER_FIGURES_DIR / png.name
            if reference.exists():
                result = compare_png(png, reference)
                print(f"  compare: {describe(result)}")
                if not result["identical"]:
                    mismatches.append(number)
            else:
                print(f"  compare: no reference image {reference}")
        if summary and not args.quiet:
            print("  " + summary.rstrip().replace("\n", "\n  "))
        print()
        summary_file = out_dir / "summaries" / f"figure_{number}.txt"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(
            f"Figure {number}: {', '.join(outputs(module))}\n"
            f"Result set: {results_root}\n\n{summary}"
        )
        summaries.append(summary_file)

    if summaries:
        print(f"Numerical summaries written to {summaries[0].parent}/")
    if failures:
        print(f"\nFigures not produced (missing inputs): {failures}")
    if args.compare:
        if mismatches:
            print(f"Figures that are not pixel-identical to the paper: {mismatches}")
        else:
            print("All compared figures are pixel-identical to the paper's PNGs.")
    return 1 if failures or (args.compare and mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
