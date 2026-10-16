#!/usr/bin/env python
"""
Compare regenerated numerical results with the published ones.

Three checks, each optional:

``--integrity`` (always run)
    Every file in ``results/published/`` still has the SHA-256 recorded in
    ``results/published/MANIFEST.json``.

default
    Every processed file present in ``results/regenerated/`` is compared with its
    published counterpart and classified as

    * IDENTICAL       -- bit-for-bit equal;
    * ROUNDOFF        -- equal to a relative tolerance of 1e-9 (floating-point
                         reassociation, e.g. a different BLAS build);
    * DIFFERENT       -- larger differences.  The experiments are chaotic: a
                         roundoff-level change in one window grows until the
                         trajectories decorrelate, after which the regenerated
                         numbers are a new sample of the same random experiment.
                         Each figure-level statistic is then compared through a
                         paired z-score across seeds (|z| < 4: consistent with
                         sampling variability; |z| >= 4: a meaningful discrepancy).

``--spot-check``
    Re-runs a small subset of every Lorenz-96 experiment in a temporary directory
    (one seed or cell each, ~5 minutes in total) and compares it with the
    published per-seed / per-cell values.  Nothing is written to ``results/``.

Examples
--------
    python scripts/verify_results.py                  # integrity + regenerated vs published
    python scripts/verify_results.py --spot-check     # integrity + quick re-runs
"""

from __future__ import annotations

# Single-threaded BLAS, as in every experiment driver; must precede numpy.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PUBLISHED_RESULTS_DIR, resolve_results_dir  # noqa: E402
from src.processing.manifest import sha256  # noqa: E402

RTOL, ATOL = 1e-9, 1e-12
Z_FLAG = 4.0


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, status: str, name: str, detail: str = ""):
        self.rows.append((status, name, detail))
        print(f"  [{status:<9}] {name}" + (f" -- {detail}" if detail else ""), flush=True)

    @property
    def failed(self):
        return [r for r in self.rows if r[0] in ("FAIL", "DIFFERENT", "MISSING")]


# --------------------------------------------------------------------------- #
# Integrity of the published results
# --------------------------------------------------------------------------- #
def check_integrity(checks: Checks) -> None:
    manifest_path = PUBLISHED_RESULTS_DIR / "MANIFEST.json"
    if not manifest_path.exists():
        checks.add("FAIL", "published manifest", f"{manifest_path} not found")
        return
    manifest = json.loads(manifest_path.read_text())
    bad = []
    for rel, info in manifest["files"].items():
        path = PUBLISHED_RESULTS_DIR / rel
        if not path.exists() or sha256(path) != info["sha256"]:
            bad.append(rel)
    on_disk = {
        p.relative_to(PUBLISHED_RESULTS_DIR).as_posix()
        for p in PUBLISHED_RESULTS_DIR.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
    }
    extra = sorted(on_disk - set(manifest["files"]))
    if bad:
        checks.add("FAIL", "published results integrity",
                   f"{len(bad)} file(s) changed or missing, e.g. {bad[:3]}")
    else:
        checks.add("PASS", "published results integrity",
                   f"{len(manifest['files'])} files match MANIFEST.json"
                   + (f"; {len(extra)} unlisted file(s): {extra[:3]}" if extra else ""))


# --------------------------------------------------------------------------- #
# Array comparison helpers
# --------------------------------------------------------------------------- #
def classify(pub: np.ndarray, reg: np.ndarray):
    pub, reg = np.asarray(pub), np.asarray(reg)
    if pub.shape != reg.shape:
        return "DIFFERENT", f"shape {reg.shape} vs published {pub.shape}"
    if np.array_equal(pub, reg, equal_nan=pub.dtype.kind == "f"):
        return "IDENTICAL", f"{pub.size} values"
    if pub.dtype.kind in "fc" and np.allclose(reg, pub, rtol=RTOL, atol=ATOL, equal_nan=True):
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.nanmax(np.abs(reg - pub) / np.maximum(np.abs(pub), ATOL))
        return "ROUNDOFF", f"max relative difference {rel:.1e}"
    finite = np.isfinite(pub) & np.isfinite(reg) if pub.dtype.kind == "f" else np.ones(pub.shape, bool)
    diff = np.abs(reg.astype(float) - pub.astype(float))[finite]
    return "DIFFERENT", (f"max |difference| {diff.max():.3g}, "
                         f"{np.mean(diff > 0):.1%} of values differ")


def paired_z(pub_per_seed: np.ndarray, reg_per_seed: np.ndarray) -> float:
    d = np.asarray(reg_per_seed, float) - np.asarray(pub_per_seed, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan")
    se = d.std(ddof=1) / math.sqrt(d.size)
    return 0.0 if se == 0 else float(d.mean() / se)


def unpaired_z(pub_values: np.ndarray, reg_values: np.ndarray) -> float:
    a, b = np.asarray(pub_values, float), np.asarray(reg_values, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return 0.0 if se == 0 else float((b.mean() - a.mean()) / se)


def report_z(checks: Checks, name: str, zs: dict) -> None:
    worst = max(zs.items(), key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1)
    status = "CONSISTENT" if all(abs(z) < Z_FLAG for z in zs.values() if np.isfinite(z)) else "FAIL"
    checks.add(status, name, f"largest paired |z| = {abs(worst[1]):.2f} ({worst[0]}); "
                             f"flag threshold {Z_FLAG}")


# --------------------------------------------------------------------------- #
# Regenerated vs published: Lorenz-96
# --------------------------------------------------------------------------- #
def compare_lorenz96(pub_root: Path, reg_root: Path, checks: Checks) -> None:
    from src.processing.lorenz96 import BURN_IN, FILENAMES

    pub_dir, reg_dir = pub_root / "lorenz96", reg_root / "lorenz96"
    for key, filename in FILENAMES.items():
        reg_path = reg_dir / filename
        if not reg_path.exists():
            checks.add("SKIP", f"lorenz96/{filename}", "not regenerated")
            continue
        pub = np.load(pub_dir / filename, allow_pickle=False)
        reg = np.load(reg_path, allow_pickle=False)
        print(f"\nlorenz96/{filename}")
        for array in pub.files:
            if array == "meta":
                continue
            if array not in reg.files:
                checks.add("MISSING", f"  {array}")
                continue
            status, detail = classify(pub[array], reg[array])
            checks.rows.append((status if status != "DIFFERENT" else "INFO", f"{filename}:{array}", detail))
            print(f"  [{status:<9}] {array} -- {detail}")
            if status != "DIFFERENT":
                continue
            # Sampling-level comparison of the statistics each figure draws.
            filters = [str(f) for f in pub["filters"]] if "filters" in pub.files else []
            if key == "calibration" and array in ("rmse", "spread"):
                zs = {}
                for i, f in enumerate(filters):
                    p = np.median(pub[array][i][:, BURN_IN:], axis=1)
                    r = np.median(reg[array][i][:, BURN_IN:], axis=1)
                    zs[f] = paired_z(p, r)
                report_z(checks, f"    settled per-seed median {array}", zs)
            elif key == "coverage":
                zs = {}
                for i, f in enumerate(filters):
                    def outside(block):
                        block = block[:, BURN_IN:, :]
                        return (block[..., 0] + block[..., -1]).sum(axis=1) / block.sum(axis=(1, 2))
                    zs[f] = paired_z(outside(pub[array][i]), outside(reg[array][i]))
                report_z(checks, "    settled outside-ensemble rate per seed", zs)
            elif key == "ensemble_sweep" and array == "per_seed_settled_median_rmse":
                zs = {f"{f}, N={n}": paired_z(pub[array][i, j], reg[array][i, j])
                      for i, f in enumerate(filters)
                      for j, n in enumerate(pub["ensemble_sizes"])}
                report_z(checks, "    per-seed settled median RMSE", zs)
            elif key == "hierarchical" and array.endswith("_trace"):
                same_seeds = np.array_equal(pub["truth_seeds"], reg["truth_seeds"])
                zs = {}
                for i, f in enumerate(filters):
                    p = pub[array][i][:, 1:].mean(axis=1)
                    r = reg[array][i][:, 1:].mean(axis=1)
                    zs[f] = paired_z(p, r) if same_seeds else unpaired_z(p, r)
                report_z(checks, f"    full-run mean {array} per truth seed "
                                 f"({'paired' if same_seeds else 'unpaired: different truth seeds'})", zs)
            elif key == "n_kappa" and array == "cell_plugin_stats":
                same_seeds = np.array_equal(pub["truth_seeds"], reg["truth_seeds"])
                zs = {}
                for g, regime in enumerate(pub["regimes"]):
                    for p_idx, (N, k) in enumerate(pub["design_points"]):
                        for s, stat in enumerate(("mse", "bias2")):
                            a = pub[array][g, p_idx, :, s]
                            b = reg[array][g, p_idx, :, s]
                            zs[f"{regime} N={N} kappa={k} {stat}"] = (
                                paired_z(a, b) if same_seeds else unpaired_z(a, b))
                report_z(checks, "    per-cell plug-in MSE and squared bias", zs)


# --------------------------------------------------------------------------- #
# Regenerated vs published: shallow water
# --------------------------------------------------------------------------- #
def _numeric_csv_compare(pub_path: Path, reg_path: Path, ignore=("source",)):
    pub = list(csv.DictReader(open(pub_path)))
    reg = list(csv.DictReader(open(reg_path)))
    if len(pub) != len(reg):
        return "DIFFERENT", f"{len(reg)} rows vs published {len(pub)}"
    worst, mismatched = 0.0, 0
    for rp, rr in zip(pub, reg):
        for key, vp in rp.items():
            if key in ignore:
                continue
            vr = rr.get(key)
            if vp == vr:
                continue
            try:
                a, b = float(vp), float(vr)
            except (TypeError, ValueError):
                mismatched += 1
                continue
            if math.isnan(a) and math.isnan(b):
                continue
            rel = abs(a - b) / max(abs(a), ATOL)
            worst = max(worst, rel)
            if rel > RTOL:
                mismatched += 1
    if mismatched:
        return "DIFFERENT", f"{mismatched} values differ beyond rel {RTOL:g} (max rel {worst:.2g})"
    if worst == 0.0:
        return "IDENTICAL", f"{len(pub)} rows"
    return "ROUNDOFF", f"max relative difference {worst:.1e}"


def _numeric_json_compare(pub_path: Path, reg_path: Path):
    pub, reg = json.loads(pub_path.read_text()), json.loads(reg_path.read_text())
    if pub.keys() != reg.keys():
        return "DIFFERENT", "different entries"
    worst = 0.0
    for key in pub:
        for field, a in pub[key].items():
            b = reg[key].get(field)
            if a == b or (isinstance(a, float) and isinstance(b, float)
                          and math.isnan(a) and math.isnan(b)):
                continue
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                return "DIFFERENT", f"{key}.{field}: {a!r} vs {b!r}"
            worst = max(worst, abs(a - b) / max(abs(a), ATOL))
    if worst == 0.0:
        return "IDENTICAL", f"{len(pub)} entries"
    if worst <= RTOL:
        return "ROUNDOFF", f"max relative difference {worst:.1e}"
    return "DIFFERENT", f"max relative difference {worst:.2g}"


def compare_shallow_water(pub_root: Path, reg_root: Path, checks: Checks) -> None:
    """Inlet tables rebuilt from the published run outputs vs the published tables.

    The shallow-water experiment is reproduced from its saved outputs only, so
    the check is the reduction from runs to tables (extract_results.py
    --shallow-water), not a rerun.
    """
    reg_dir = reg_root / "shallow_water"
    names = ("scale_matching_trajectories_v3.csv", "scale_matching_regime_map_v3.csv",
             "all_filter_shelf_gamma_E_v3.csv", "all_filter_per_window_v3.json")
    if not any((reg_dir / name).exists() for name in names):
        checks.add("SKIP", "shallow_water tables",
                   "not rebuilt (python scripts/extract_results.py --shallow-water)")
        return
    print("\nshallow_water/ (tables rebuilt from the published run outputs)")
    for name in names:
        if not (reg_dir / name).exists():
            checks.add("SKIP", f"shallow_water/{name}", "not rebuilt")
            continue
        compare = _numeric_json_compare if name.endswith(".json") else _numeric_csv_compare
        status, detail = compare(pub_root / "shallow_water" / name, reg_dir / name)
        checks.add(status, f"shallow_water/{name}", detail)


# --------------------------------------------------------------------------- #
# Spot checks: rerun a slice of every Lorenz-96 experiment
# --------------------------------------------------------------------------- #
def spot_check(checks: Checks) -> None:
    from threadpoolctl import threadpool_limits

    from src.experiments import n_kappa_bias_variance as nk
    from src.experiments.baseline import (
        EXTENDED_FILTER_ORDER,
        config_at_ensemble_size,
        load_config,
        run_seed,
    )
    from src.experiments.bias_variance_mc import (
        FILTER_NAMES,
        _run_pair,
        generate_truth_batch,
    )
    from src.experiments.restart_vs_cycled import _config_for_windows, load_calibration
    from src.processing.lorenz96 import BURN_IN, hierarchical_config, window_axis
    from src.utils.config import load_experiment_config

    pub = PUBLISHED_RESULTS_DIR / "lorenz96"
    calib = np.load(pub / "calibration_N10.npz", allow_pickle=False)
    cover = np.load(pub / "coverage_ranks_N10.npz", allow_pickle=False)
    sweep = np.load(pub / "ensemble_size_sweep.npz", allow_pickle=False)
    hier = np.load(pub / "hierarchical_bias_variance.npz", allow_pickle=False)
    nkap = np.load(pub / "n_kappa_bootstrap.npz", allow_pickle=False)

    config = load_config()
    window_len = config["assimilation"]["window_length"]
    n_windows = config["assimilation"]["n_windows"]
    filters = list(EXTENDED_FILTER_ORDER)
    seed_index, seed = 0, int(calib["seeds"][0])

    with threadpool_limits(limits=1):
        # (a, b) Figures 1-2: one seed of the N = 10 experiment, with ranks.
        print(f"\nFigures 1-2: seed {seed}, all five filters, 1000 windows")
        t0 = time.perf_counter()
        results = run_seed(seed, config, filters=filters, collect_window_ranks=True)
        for i, label in enumerate(filters):
            for array in ("rmse", "spread"):
                status, detail = classify(calib[array][i, seed_index],
                                          window_axis(results[label][array], window_len, n_windows))
                checks.add(status, f"Fig 1-2 {label} {array}", detail)
            status, detail = classify(cover["rank_counts"][i, seed_index],
                                      results[label]["rank_counts_by_window"].astype(np.int32))
            checks.add(status, f"Fig 2 {label} rank counts", detail)
        print(f"  ({time.perf_counter() - t0:.0f} s)")

        # (c) Figure 3: the same seed at N = 5.
        print(f"\nFigure 3: seed {seed} at N = 5")
        t0 = time.perf_counter()
        j = list(sweep["ensemble_sizes"]).index(5)
        results = run_seed(seed, config_at_ensemble_size(config, 5), filters=filters)
        for i, label in enumerate(filters):
            matrix = window_axis(results[label]["rmse"], window_len, n_windows)[None, :]
            value = np.median(matrix[:, BURN_IN:], axis=1)
            status, detail = classify(sweep["per_seed_settled_median_rmse"][i, j, seed_index:seed_index + 1], value)
            checks.add(status, f"Fig 3 {label} settled median RMSE (N=5)", detail)
        print(f"  ({time.perf_counter() - t0:.0f} s)")

    # (d) Figure 4: one truth seed of the hierarchical experiment.
    hconfig = hierarchical_config()
    truth_seed = int(hier["truth_seeds"][0])
    print(f"\nFigure 4: truth seed {truth_seed}, five filters x 100 realizations x 1000 windows")
    t0 = time.perf_counter()
    truth_data = generate_truth_batch(hconfig, [truth_seed])
    spinup = int(hconfig["bias_variance"].get("spinup_windows", 0))
    with tempfile.TemporaryDirectory() as tmp:
        for name in FILTER_NAMES:
            _seed, _name, _sec, error = _run_pair(
                hconfig, truth_seed, name, *truth_data[truth_seed], Path(tmp), spinup)
            if error:
                checks.add("FAIL", f"Fig 4 {name}", error)
                continue
            slug = name.lower().replace("-", "_")
            with np.load(Path(tmp) / f"{slug}_truth_seed_{truth_seed}.npz") as data:
                i = list(hier["filters"]).index(name)
                for key in ("bias_squared_trace", "variance_trace", "mse_trace"):
                    status, detail = classify(hier[key][i, 0], data[key])
                    checks.add(status, f"Fig 4 {name} {key}", detail)
    print(f"  ({time.perf_counter() - t0:.0f} s)")

    # (e) Figure 5: two cells of the (N, kappa) experiment, with the published law.
    nk_config = load_experiment_config(nk.DEFAULT_CONFIG_PATH)
    settings = nk.settings_from_config(nk_config)
    law = load_calibration(pub / "restart_law_2fbc05cb06f87eaa.npz")["common"]
    meta = json.loads(str(nkap["meta"]))
    with tempfile.TemporaryDirectory() as tmp:
        results_dir, digest, _scientific = nk.resolve_results_dir(
            nk_config, settings, law.identifier, Path(tmp))
        checks.add("PASS" if digest == meta["config_hash"] else "FAIL",
                   "Fig 5 configuration hash",
                   f"{digest[:12]} (published {meta['config_hash'][:12]})")
        cell_seed = int(nkap["truth_seeds"][0])
        point = (10, 1)
        p_idx = [tuple(map(int, dp)) for dp in nkap["design_points"]].index(point)
        print(f"\nFigure 5: truth seed {cell_seed}, (N, kappa) = {point}, both regimes")
        t0 = time.perf_counter()
        evaluation_config = _config_for_windows(nk_config, settings.total_cycled_windows)
        truth_data = generate_truth_batch(evaluation_config, [cell_seed])
        for g, regime in enumerate(nkap["regimes"]):
            out = nk._cell_task(evaluation_config, settings, str(regime), point[0], point[1],
                                cell_seed, *truth_data[cell_seed], law, results_dir, digest)
            if out[-1]:
                checks.add("FAIL", f"Fig 5 {regime} cell", out[-1])
                continue
            path = nk.cell_checkpoint_path(results_dir, str(regime), point[0], point[1], cell_seed)
            with np.load(path, allow_pickle=False) as data:
                gram = data["error_gram"]
            ones = np.ones(gram.shape[0])
            stats = np.array([ones @ np.diag(gram) / gram.shape[0],
                              ones @ gram @ ones / gram.shape[0] ** 2])
            status, detail = classify(nkap["cell_plugin_stats"][g, p_idx, 0], stats)
            checks.add(status, f"Fig 5 {regime} cell plug-in MSE and squared bias", detail)
        print(f"  ({time.perf_counter() - t0:.0f} s)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None,
    )
    parser.add_argument("--results", default="regenerated",
                        help="Result set to compare with the published one "
                             "(default: regenerated).")
    parser.add_argument("--spot-check", action="store_true",
                        help="Re-run one seed/cell of every Lorenz-96 experiment and "
                             "compare it with the published values (~5 min).")
    args = parser.parse_args(argv)

    checks = Checks()
    print("Integrity of results/published/")
    check_integrity(checks)

    if args.spot_check:
        spot_check(checks)
    else:
        reg_root = resolve_results_dir(args.results)
        print(f"\nComparing {reg_root} with {PUBLISHED_RESULTS_DIR}")
        compare_lorenz96(PUBLISHED_RESULTS_DIR, reg_root, checks)
        compare_shallow_water(PUBLISHED_RESULTS_DIR, reg_root, checks)

    counts = {}
    for status, _name, _detail in checks.rows:
        counts[status] = counts.get(status, 0) + 1
    print("\nSummary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if checks.failed:
        print("Checks that need attention:")
        for status, name, detail in checks.failed:
            print(f"  [{status}] {name} -- {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
