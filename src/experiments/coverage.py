"""
Common-endpoint rank diagnostics for the post-burn-in coverage analysis.

Why this exists as its own cache
--------------------------------
The baseline shards store `rank_hist`: rank counts accumulated over a filter's
*whole run* at its *native* cadence. That is enough for a rank histogram of the
full run and nothing else. It cannot answer the three questions notebook 01's
coverage figure asks, because

  * it has already been summed over time, so burn-in cannot be excluded;
  * it is accumulated at each filter's own verification times, so a sequential
    filter contributes L times as many ranks as a 4D filter and the two are not
    comparable window by window;
  * it has no time axis at all, so no temporal outside-ensemble rate exists.

The fix is `rank_counts_by_window`, shape ``(W, N+1)``: one row per assimilation
window endpoint, each row the rank counts over the n state components at that
endpoint, so each row sums to n. At W=1000, N=10 that is 44 kB per filter-seed
against ~21 MB for the ensemble history, so it is cheap to keep and there is no
reason to store ensembles.

This module writes those counts to a *separate versioned cache* rather than
extending the baseline shards. The baseline cache is reused on the strength of
its filename alone, so adding a field to it would make every existing shard
silently look complete while missing the new diagnostic. A separate directory
plus recorded metadata means a shard is only reused when it demonstrably holds
what this analysis needs, and no baseline shard is touched or rewritten.

Reproducibility
---------------
`run_seed` is called with the same seed, config and filter set as the baseline
run, so the trajectories are bit-identical to the ones already cached -- the
extra diagnostic is read off the same states, not off a re-randomised run. This
is checked in `tests/test_coverage_ranks.py`.

Usage
-----
    python -m src.experiments.coverage --n-seeds 100 --n-jobs -2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DIAGNOSTICS_RESULTS_DIR
from src.experiments.baseline import (
    EXTENDED_FILTER_ORDER,
    SHARD_DIR_NAME,
    load_config,
    resolve_seeds,
    run_seed,
)

# Bump when the stored quantity or either convention changes. The directory name
# carries it, so an old cache is never silently reinterpreted -- it is simply a
# different directory that this version does not read.
COVERAGE_VERSION = 1
COVERAGE_RESULTS_DIR = DIAGNOSTICS_RESULTS_DIR / f"coverage_v{COVERAGE_VERSION}"

DIAGNOSTIC = "rank_counts_by_window"

# Recorded in every shard and checked before reuse. Free text, but it is the
# text a future reader will compare against their own assumptions.
RANK_CONVENTION = "zero-based: rank = number of members strictly below the truth, 0..N"
ENDPOINT_CONVENTION = (
    "assimilation-window endpoints only; 4D filters at every native window end, "
    "sequential filters at zero-based observation indices L-1, 2L-1, ..., WL-1"
)
BURN_IN_NOTE = "no burn-in applied here; the analysis slices it off"


def coverage_shard_path(seed: int, out_dir=COVERAGE_RESULTS_DIR) -> Path:
    return Path(out_dir) / SHARD_DIR_NAME / f"seed_{seed}.npz"


def expected_metadata(config: dict, filters) -> dict:
    """The metadata a shard must match to be reusable for this analysis."""
    return {
        "version": int(COVERAGE_VERSION),
        "ensemble_size": int(config["assimilation"]["ensemble_size"]),
        "n_windows": int(config["assimilation"]["n_windows"]),
        "window_length": int(config["assimilation"]["window_length"]),
        "state_dim": int(config["model"]["n"]),
        "filters": list(filters),
        "rank_convention": RANK_CONVENTION,
        "endpoint_convention": ENDPOINT_CONVENTION,
        "burn_in": BURN_IN_NOTE,
    }


def _save_shard(path: Path, seed: int, per_filter: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"rank__{label}": counts for label, counts in per_filter.items()}
    payload["meta__seed"] = np.int64(seed)
    for key in ("version", "ensemble_size", "n_windows", "window_length", "state_dim"):
        payload[f"meta__{key}"] = np.int64(meta[key])
    payload["meta__filters"] = np.array(meta["filters"], dtype="U32")
    for key in ("rank_convention", "endpoint_convention", "burn_in"):
        payload[f"meta__{key}"] = np.array(meta[key], dtype="U256")
    tmp = path.parent / f"{path.stem}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)  # atomic: an interrupted run never leaves a half shard


def load_shard(path: Path, seed: int, meta: dict, filters):
    """
    Return ``{label: (W, N+1) int array}``, or None when the shard is unusable.

    Unusable covers missing file, unreadable file, absent diagnostic, a filter
    that was not requested when the shard was written, and any metadata
    mismatch. Returning None rather than raising is what lets `run_coverage`
    treat a stale shard exactly like a missing one and recompute it.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["meta__seed"]) != int(seed):
                return None
            for key in ("version", "ensemble_size", "n_windows", "window_length",
                        "state_dim"):
                if int(data[f"meta__{key}"]) != int(meta[key]):
                    return None
            for key in ("rank_convention", "endpoint_convention", "burn_in"):
                if str(data[f"meta__{key}"]) != meta[key]:
                    return None
            stored = set(data["meta__filters"].tolist())
            if not set(filters) <= stored:
                return None
            out = {}
            for label in filters:
                key = f"rank__{label}"
                if key not in data.files:
                    return None
                counts = data[key]
                if counts.shape != (meta["n_windows"], meta["ensemble_size"] + 1):
                    return None
                out[label] = counts
    except (OSError, ValueError, KeyError):
        return None
    return out


def _run_and_save(seed: int, config: dict, out_dir: Path, filters, force: bool) -> str:
    meta = expected_metadata(config, filters)
    path = coverage_shard_path(seed, out_dir)
    if not force and load_shard(path, seed, meta, filters) is not None:
        return f"skip {seed}"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        results = run_seed(seed, config, filters=list(filters),
                           collect_window_ranks=True)
    else:
        with threadpool_limits(limits=1):
            results = run_seed(seed, config, filters=list(filters),
                               collect_window_ranks=True)
    per_filter = {}
    for label in filters:
        counts = np.asarray(results[label][DIAGNOSTIC], dtype=np.int32)
        validate_counts(counts, meta, where=f"seed {seed}, {label}")
        per_filter[label] = counts
    _save_shard(path, seed, per_filter, meta)
    return f"ran {seed}"


def validate_counts(counts, meta: dict, where: str = "") -> None:
    """Shape, non-negativity and the row-sum identity, on every path."""
    want = (meta["n_windows"], meta["ensemble_size"] + 1)
    if counts.shape != want:
        raise ValueError(f"{where}: rank counts are {counts.shape}, expected {want}")
    if (counts < 0).any():
        raise ValueError(f"{where}: negative rank counts")
    rows = counts.sum(axis=1)
    if not (rows == meta["state_dim"]).all():
        bad = int(np.argmax(rows != meta["state_dim"]))
        raise ValueError(
            f"{where}: row {bad} sums to {int(rows[bad])}, expected "
            f"{meta['state_dim']} (one rank per state component)"
        )


def run_coverage(
    config: dict,
    seeds=None,
    filters=None,
    n_jobs: int = -2,
    out_dir=COVERAGE_RESULTS_DIR,
    force: bool = False,
    verbose: int = 5,
) -> dict:
    """
    Return ``{label: (n_seeds, W, N+1) int array}`` of common-endpoint ranks.

    Seeds whose shard is missing *or fails validation* are recomputed; a shard
    is never trusted because it exists, only because it says what it holds and
    that matches this config and filter set.
    """
    filters = list(EXTENDED_FILTER_ORDER if filters is None else filters)
    seeds = list(resolve_seeds(config) if seeds is None else seeds)
    out_dir = Path(out_dir)
    meta = expected_metadata(config, filters)

    pending = [
        s for s in seeds
        if force or load_shard(coverage_shard_path(s, out_dir), s, meta, filters) is None
    ]
    print(
        f"{len(seeds)} seeds requested; {len(pending)} to compute, "
        f"{len(seeds) - len(pending)} valid in {out_dir / SHARD_DIR_NAME}"
    )
    if pending:
        Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_run_and_save)(s, config, out_dir, filters, force) for s in pending
        )

    stacked = {label: [] for label in filters}
    for s in seeds:
        shard = load_shard(coverage_shard_path(s, out_dir), s, meta, filters)
        if shard is None:
            raise RuntimeError(f"seed {s}: shard still invalid after recomputation")
        for label in filters:
            stacked[label].append(shard[label])
    return {label: np.stack(v) for label, v in stacked.items()}


# --------------------------------------------------------------------------
# Coverage algebra (shared by the notebook and the tests)
# --------------------------------------------------------------------------
def central_intervals(n_ensemble: int):
    """
    Attainable symmetric central intervals, as ``(q, nominal, lo_bin, hi_bin)``.

    Dropping q members from each side leaves the rank bins q+1 .. N-q-1, so the
    interval holds N-2q-1 of the N+1 equiprobable bins. q = 0 is the ensemble
    min-max interval, whose nominal coverage is (N-1)/(N+1) -- 9/11 at N = 10,
    not 10/11: the truth also falls outside whenever it ties the extremes'
    side of the partition, and there are N+1 bins for N members.
    """
    out = []
    for q in range((n_ensemble - 1) // 2 + 1):
        lo, hi = q + 1, n_ensemble - q - 1
        width = hi - lo + 1
        if width <= 0:
            break
        out.append((q, width / (n_ensemble + 1), lo, hi))
    return out


def coverage_from_counts(counts, lo_bin: int, hi_bin: int):
    """
    Empirical coverage of the central interval ``[lo_bin, hi_bin]``.

    `counts` is (..., n_windows, N+1); the reduction is over the last two axes,
    so a (n_seeds, W, N+1) block returns one coverage per seed.
    """
    counts = np.asarray(counts)
    inside = counts[..., lo_bin:hi_bin + 1].sum(axis=(-2, -1))
    total = counts.sum(axis=(-2, -1))
    return inside / total


def tail_rates(counts):
    """``(lower, upper)`` miss rates: truth below the ensemble, and above it."""
    counts = np.asarray(counts)
    total = counts.sum(axis=(-2, -1))
    return (counts[..., 0].sum(axis=-1) / total,
            counts[..., -1].sum(axis=-1) / total)


def outside_rate_by_window(counts, state_dim: int):
    """Per-window fraction of components whose truth is outside the ensemble."""
    counts = np.asarray(counts)
    return (counts[..., 0] + counts[..., -1]) / state_dim


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    seeds = resolve_seeds(config, args.n_seeds)
    ranks = run_coverage(config, seeds=seeds, n_jobs=args.n_jobs, force=args.force)
    n = config["model"]["n"]
    for label, block in ranks.items():
        print(f"{label:<14} {block.shape}  rows sum to n: "
              f"{bool((block.sum(axis=2) == n).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
