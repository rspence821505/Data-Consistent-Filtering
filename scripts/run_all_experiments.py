#!/usr/bin/env python
"""
Level 2: rerun the numerical experiments behind the paper from scratch.

Each experiment is launched exactly as it was for the paper (same drivers,
configuration files, seeds and settings; see ``experiments/lorenz96/``).  Raw,
resumable checkpoints go to ``results/regenerated/raw/lorenz96/`` and are then
reduced by ``scripts/extract_results.py`` to the processed files in
``results/regenerated/lorenz96/`` that the figure scripts read::

    python scripts/run_all_experiments.py                 # every Lorenz-96 experiment
    python scripts/run_all_experiments.py --experiments baseline coverage
    python scripts/run_all_experiments.py --list          # what each one is, and its cost
    python scripts/reproduce_figures.py --results regenerated
    python scripts/verify_results.py                      # compare with the published numbers

Nothing here writes to ``results/published/``.  Every driver is resumable: an
interrupted run continues where it stopped when the same command is repeated.

The shallow-water experiment (Figures 6-8, Table 2) is not part of Level 2: it
is reproduced from its saved run outputs in ``results/published/shallow_water/``
(see the README).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DIAGNOSTICS_RESULTS_DIR, RAW_RESULTS_DIR  # noqa: E402
from src.paths import PUBLISHED_RESULTS_DIR, REGENERATED_RESULTS_DIR  # noqa: E402

PUBLISHED_RESTART_LAW = PUBLISHED_RESULTS_DIR / "lorenz96" / "restart_law_2fbc05cb06f87eaa.npz"


def _python_c(code: str):
    return [sys.executable, "-c", code]


def _python_m(module: str, *args: str):
    return [sys.executable, "-m", module, *args]


# name -> (paper figures, description, measured wall clock, command builder, extractors)
EXPERIMENTS = {
    "baseline": dict(
        figures="1, 2",
        description="Lorenz-96, N = 10: Seq-EnKF, Seq-ETKF, 4D-EnKF, 4D-ETKF and "
                    "QPCA-EnDCF on 100 seeds x 1000 windows.",
        runtime="~4 min",
        command=lambda j: _python_c(
            "from src.experiments.baseline import EXTENDED_FILTER_ORDER, "
            "EXTENDED_RESULTS_DIR, load_config, resolve_seeds, run_experiment; "
            "c = load_config(); "
            f"run_experiment(c, seeds=resolve_seeds(c, 100), n_jobs={j}, "
            "out_dir=EXTENDED_RESULTS_DIR, filters=EXTENDED_FILTER_ORDER)"
        ),
        extract=["calibration"],
    ),
    "coverage": dict(
        figures="2",
        description="Common-endpoint rank counts of the same 100 x 1000 runs "
                    "(re-run with the extra diagnostic; trajectories are bit-identical).",
        runtime="~4 min",
        command=lambda j: _python_m("src.experiments.coverage", "--n-seeds", "100",
                                    "--n-jobs", str(j)),
        extract=["coverage"],
    ),
    "ensemble-sweep": dict(
        figures="3",
        description="The baseline experiment repeated for N in {5, 10, 15, 20, 25, 30}.",
        runtime="~1-1.5 h",
        command=lambda j: _python_c(
            "from src.experiments.baseline import EXTENDED_FILTER_ORDER, "
            "config_at_ensemble_size, ensemble_results_dir, load_config, "
            "resolve_seeds, run_experiment; "
            "c = load_config(); seeds = resolve_seeds(c, 100); "
            f"[run_experiment(config_at_ensemble_size(c, n), seeds=seeds, n_jobs={j}, "
            "out_dir=ensemble_results_dir(n), filters=EXTENDED_FILTER_ORDER) "
            "for n in (5, 10, 15, 20, 25, 30)]"
        ),
        extract=["ensemble_sweep"],
    ),
    "hierarchical": dict(
        figures="4",
        description="Hierarchical bias-variance decomposition: 117 truth seeds drawn "
                    "(the first 100 complete ones are used) x 100 realizations x "
                    "1000 windows x 5 filters.",
        runtime="~1 h",
        command=lambda j: _python_m(
            "src.experiments.bias_variance_mc", "--hierarchical",
            "--n-truth-seeds", "117", "--truth-seed-base", "20260728",
            "--n-mc", "100", "--n-windows", "1000", "--n-jobs", str(j),
            "--results-dir", str(DIAGNOSTICS_RESULTS_DIR / "bias_variance_analysis_100x100"),
        ),
        extract=["hierarchical"],
    ),
    "n-kappa": dict(
        figures="5",
        description="QPCA-EnDCF over N in {5, 10, 20, 40} x kappa in {1, 2, 4, 8} "
                    "(kappa <= N - 1), cycled and independent-restart regimes, 50 truth "
                    "seeds x 100 realizations x 100 evaluated windows.",
        runtime="~1-2 h",
        command=lambda j: _python_m(
            "src.experiments.n_kappa_bias_variance", "--n-kappa", "--label", "grid",
            "--n-jobs", str(j),
        ),
        extract=["n_kappa"],
    ),
}


def list_experiments() -> None:
    print("Lorenz-96 experiments (Level 2).  Wall-clock times for --n-jobs 10 on a "
          "15-core Apple M5 Pro (measured for baseline and coverage, projected from "
          "the drivers' --dry-run estimators otherwise); they scale roughly inversely "
          "with the number of workers.\n")
    for name, spec in EXPERIMENTS.items():
        print(f"  {name:<15} Figure(s) {spec['figures']:<8} {spec['runtime']:<9} "
              f"{spec['description']}")
    print("\n  (The shallow-water experiment behind Figures 6-8 is not rerun; its figures "
          "are drawn from the saved run outputs in results/published/shallow_water/.)")
    print(f"\nRaw checkpoints: {RAW_RESULTS_DIR}")
    print(f"Processed results: {REGENERATED_RESULTS_DIR / 'lorenz96'}")


def seed_published_restart_law() -> Path:
    """Place the published restart law where the (N, kappa) driver looks for it.

    The (N, kappa) experiment needs a fixed restart forecast law.  By default it
    calibrates one from scratch under the published specification (seconds); this
    copies the published law into the driver's calibration cache instead, so the
    experiment starts from bit-identical inputs.
    """
    from src.experiments import n_kappa_bias_variance as nk
    from src.utils.config import load_experiment_config

    config = load_experiment_config(nk.DEFAULT_CONFIG_PATH)
    settings = nk.settings_from_config(config)
    target = nk._calibration_cache_path(config, settings, nk.DEFAULT_RESULTS_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PUBLISHED_RESTART_LAW, target)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rerun the Lorenz-96 experiments of the paper (Level 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --list to see every experiment, its figures and its cost.",
    )
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS),
                        default=list(EXPERIMENTS), metavar="NAME",
                        help=f"Subset to run (default: all of {', '.join(EXPERIMENTS)}).")
    parser.add_argument("--n-jobs", type=int, default=10,
                        help="Worker processes per experiment (default 10).")
    parser.add_argument("--list", action="store_true", help="Describe the experiments and exit.")
    parser.add_argument("--no-extract", action="store_true",
                        help="Only run the drivers; skip reducing their output.")
    parser.add_argument("--reuse-published-restart-law", action="store_true",
                        help="n-kappa: start from the published restart law instead of "
                             "recalibrating it (the recalibrated law has the same "
                             "identifier and agrees to ~1e-13).")
    args = parser.parse_args(argv)

    if args.list:
        list_experiments()
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"Raw checkpoints -> {RAW_RESULTS_DIR}\n")

    failed = []
    to_extract = []
    for name in args.experiments:
        spec = EXPERIMENTS[name]
        if name == "n-kappa" and args.reuse_published_restart_law:
            print(f"[n-kappa] using the published restart law -> {seed_published_restart_law()}")
        print(f"=== {name} (Figure(s) {spec['figures']}; expected {spec['runtime']}) ===",
              flush=True)
        t0 = time.perf_counter()
        status = subprocess.call(spec["command"](args.n_jobs), cwd=PROJECT_ROOT, env=env)
        minutes = (time.perf_counter() - t0) / 60
        if status != 0:
            print(f"=== {name}: FAILED (exit {status}) after {minutes:.1f} min ===\n")
            failed.append(name)
            continue
        print(f"=== {name}: finished in {minutes:.1f} min ===\n", flush=True)
        to_extract += spec["extract"]

    if to_extract and not args.no_extract:
        from src.processing import lorenz96

        lorenz96.extract_all(RAW_RESULTS_DIR, REGENERATED_RESULTS_DIR / "lorenz96",
                             only=to_extract)
        print("\nNext: python scripts/reproduce_figures.py --results regenerated --figures "
              + " ".join(sorted({f.strip() for n in args.experiments if n not in failed
                                 for f in EXPERIMENTS[n]["figures"].split(",")})))
        print("      python scripts/verify_results.py")
    if failed:
        print(f"\nFailed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
