"""Contract checks for the common-endpoint rank diagnostic and its cache.

The post-burn-in coverage figure in notebook 01 rests on three things that are
cheap to pin and expensive to notice the loss of:

  1. The rank convention. Ranks are zero-based -- the number of members
     strictly below the truth -- so they run 0..N over N+1 bins, and the
     attainable central intervals follow from that, not from N.
  2. The endpoint convention. Sequential filters verify L times per window and
     4D filters once, so only every L-th sequential verification may enter the
     diagnostic or the two are not comparable window by window.
  3. Cache validation. A shard written before this diagnostic existed must be
     rejected rather than silently treated as complete, because the baseline
     cache is otherwise reused on the strength of its filename alone.

Run:  python tests/test_coverage_ranks.py   (or: python -m pytest tests/test_coverage_ranks.py -v)
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.baseline import (  # noqa: E402
    EXTENDED_FILTER_ORDER,
    EXTENDED_RESULTS_DIR,
    METRICS,
    load_config,
    resolve_seeds,
    run_seed,
    shard_path,
)
from src.experiments.coverage import (  # noqa: E402
    central_intervals,
    coverage_from_counts,
    coverage_shard_path,
    expected_metadata,
    load_shard,
    outside_rate_by_window,
    tail_rates,
    validate_counts,
    _save_shard,
)

N_ENS = 10
N_STATE = 40
N_WIN = 50


def _counts_from_ranks(ranks, n_ensemble=N_ENS, state_dim=N_STATE, n_windows=N_WIN):
    """(n_windows, N+1) counts placing every component at the given rank(s)."""
    ranks = np.broadcast_to(np.asarray(ranks), (n_windows, state_dim))
    out = np.zeros((n_windows, n_ensemble + 1), dtype=np.int64)
    for w in range(n_windows):
        out[w] = np.bincount(ranks[w], minlength=n_ensemble + 1)
    return out


# --------------------------------------------------------------------------
# 1. Coverage algebra
# --------------------------------------------------------------------------
def test_uniform_ranks_give_the_nominal_central_coverages():
    """Exchangeable ranks put every filter exactly on the diagonal.

    40 components do not divide 11 bins, so no single window can be uniform.
    Eleven windows can be: each contributes 3 per bin plus 7 extras, and
    rotating the extras by 7 (coprime with 11) gives every bin exactly 7 of
    them, so the aggregate is 40 per bin and the coverages are exact.
    """
    n_bins = N_ENS + 1
    base, extra = divmod(N_STATE, n_bins)
    counts = np.full((n_bins, n_bins), base, dtype=np.int64)
    for w in range(n_bins):
        for j in range(extra):
            counts[w, (w * extra + j) % n_bins] += 1

    assert (counts.sum(axis=1) == N_STATE).all()
    assert (counts.sum(axis=0) == counts.sum() // n_bins).all(), "not exactly uniform"

    for q, nominal, lo, hi in central_intervals(N_ENS):
        empirical = coverage_from_counts(counts, lo, hi)
        assert abs(empirical - nominal) < 1e-12, (
            f"q={q}: empirical {empirical:.6f} vs nominal {nominal:.6f}"
        )


def test_central_interval_formula_and_the_min_max_special_case():
    """(N-2q-1)/(N+1), and the outermost interval is 9/11 at N=10, not 10/11."""
    intervals = central_intervals(N_ENS)
    nominals = [round(nom, 6) for _, nom, _, _ in intervals]
    assert nominals == [round(v / 11, 6) for v in (9, 7, 5, 3, 1)], nominals
    q0, nom0, lo0, hi0 = intervals[0]
    assert (q0, lo0, hi0) == (0, 1, N_ENS - 1)
    assert abs(nom0 - 9 / 11) < 1e-12, "ensemble min-max must be 9/11, not 10/11"


def test_coverage_formula_generalises_to_another_ensemble_size():
    """Nothing may be hard-coded to N = 10."""
    for n in (4, 7, 20):
        intervals = central_intervals(n)
        for q, nominal, lo, hi in intervals:
            assert lo == q + 1 and hi == n - q - 1
            assert abs(nominal - (n - 2 * q - 1) / (n + 1)) < 1e-12
        assert abs(intervals[0][1] - (n - 1) / (n + 1)) < 1e-12
        # Uniform ranks reproduce the nominal coverage at every size.
        counts = np.tile(np.full(n + 1, 3), (7, 1))
        for q, nominal, lo, hi in intervals:
            assert abs(coverage_from_counts(counts, lo, hi) - nominal) < 1e-12


def test_extreme_only_ranks_give_zero_coverage_and_a_full_outside_rate():
    half = N_STATE // 2
    counts = _counts_from_ranks([0] * half + [N_ENS] * (N_STATE - half))
    _, _, lo, hi = central_intervals(N_ENS)[0]
    assert coverage_from_counts(counts, lo, hi) == 0.0
    assert (outside_rate_by_window(counts, N_STATE) == 1.0).all()
    lower, upper = tail_rates(counts)
    assert abs(lower - 0.5) < 1e-12 and abs(upper - 0.5) < 1e-12


def test_lower_only_extremes_land_on_the_lower_side():
    """A one-sided miss must not be reported as symmetric underdispersion."""
    counts = _counts_from_ranks([0] * N_STATE)
    lower, upper = tail_rates(counts)
    assert lower == 1.0 and upper == 0.0
    counts = _counts_from_ranks([N_ENS] * N_STATE)
    lower, upper = tail_rates(counts)
    assert lower == 0.0 and upper == 1.0


def test_validate_counts_rejects_bad_rows():
    meta = {"n_windows": N_WIN, "ensemble_size": N_ENS, "state_dim": N_STATE}
    good = _counts_from_ranks([3] * N_STATE)
    validate_counts(good, meta)                      # must not raise
    bad = good.copy()
    bad[7, 3] -= 1                                   # row no longer sums to n
    try:
        validate_counts(bad, meta)
    except ValueError as exc:
        assert "sums to" in str(exc)
    else:
        raise AssertionError("a short row must be rejected")
    try:
        validate_counts(good[:, :-1], meta)
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("a wrong-shaped block must be rejected")


# --------------------------------------------------------------------------
# 2. The endpoint convention
# --------------------------------------------------------------------------
def test_sequential_endpoints_are_the_window_axis_indices():
    """(k+1) % L == 0 selects L-1, 2L-1, ..., WL-1 -- what `window_axis` keeps."""
    for L, W in ((5, 1000), (3, 17), (1, 9)):
        K = W * L
        selected = [k for k in range(K) if (k + 1) % L == 0]
        assert selected == [(w + 1) * L - 1 for w in range(W)]
        # `window_axis` subsamples a per-observation series as series[L-1::L].
        assert selected == list(np.arange(K)[L - 1::L])
        assert len(selected) == W


def test_endpoint_diagnostic_matches_the_native_histogram_for_4d_filters():
    """A 4D filter verifies once per window, so the two must agree exactly.

    The same run also shows the sequential filter contributing L times as many
    ranks to `rank_hist` as to the endpoint diagnostic, which is the whole
    reason the endpoint diagnostic exists.
    """
    config = copy.deepcopy(load_config())
    config["assimilation"]["n_windows"] = 12          # keep the test quick
    config["assimilation"]["total_observations"] = 12 * config["assimilation"]["window_length"]
    config["derived"]["total_observations"] = config["assimilation"]["total_observations"]
    W = config["assimilation"]["n_windows"]
    L = config["assimilation"]["window_length"]
    n = config["model"]["n"]

    results = run_seed(42, config, filters=["Seq-ETKF", "4D-ETKF"],
                       collect_window_ranks=True)

    for label in ("Seq-ETKF", "4D-ETKF"):
        counts = results[label]["rank_counts_by_window"]
        assert counts.shape == (W, config["assimilation"]["ensemble_size"] + 1)
        assert (counts.sum(axis=1) == n).all()
        assert (counts >= 0).all()

    four_d = results["4D-ETKF"]
    assert np.array_equal(four_d["rank_counts_by_window"].sum(axis=0),
                          four_d["rank_hist"].astype(np.int64))
    seq = results["Seq-ETKF"]
    assert seq["rank_counts_by_window"].sum() == W * n
    assert seq["rank_hist"].sum() == W * L * n
    # The endpoint subset can never exceed the full-run aggregate.
    assert (seq["rank_counts_by_window"].sum(axis=0)
            <= seq["rank_hist"].astype(np.int64)).all()


def test_collect_window_ranks_does_not_disturb_the_other_metrics():
    """The flag adds a diagnostic; it must not move a single existing number."""
    config = copy.deepcopy(load_config())
    config["assimilation"]["n_windows"] = 8
    config["assimilation"]["total_observations"] = 8 * config["assimilation"]["window_length"]
    config["derived"]["total_observations"] = config["assimilation"]["total_observations"]
    labels = ["Seq-EnKF", "4D-ETKF", "QPCA-EnDCF"]

    plain = run_seed(7, config, filters=labels)
    with_ranks = run_seed(7, config, filters=labels, collect_window_ranks=True)
    for label in labels:
        for metric in METRICS:
            assert np.array_equal(plain[label][metric], with_ranks[label][metric]), (
                f"{label}/{metric} changed when the diagnostic was switched on"
            )
        assert "rank_counts_by_window" not in plain[label]


# --------------------------------------------------------------------------
# 3. Cache safety
# --------------------------------------------------------------------------
def test_cache_rejects_shards_that_lack_the_new_diagnostic():
    """A baseline shard must never look complete to the coverage analysis."""
    config = load_config()
    meta = expected_metadata(config, EXTENDED_FILTER_ORDER)
    baseline = shard_path(resolve_seeds(config, 1)[0], EXTENDED_RESULTS_DIR)
    if baseline.exists():
        assert load_shard(baseline, resolve_seeds(config, 1)[0], meta,
                          EXTENDED_FILTER_ORDER) is None, (
            "a baseline shard was accepted as a coverage shard"
        )
    assert load_shard(Path("/nonexistent/seed_1.npz"), 1, meta,
                      EXTENDED_FILTER_ORDER) is None


def test_cache_rejects_metadata_mismatches_and_accepts_a_good_shard():
    config = load_config()
    meta = expected_metadata(config, EXTENDED_FILTER_ORDER)
    small = dict(meta, n_windows=4)
    counts = _counts_from_ranks([3] * N_STATE, n_windows=4)
    with tempfile.TemporaryDirectory() as tmp:
        path = coverage_shard_path(11, tmp)
        _save_shard(path, 11, {label: counts for label in EXTENDED_FILTER_ORDER}, small)

        assert load_shard(path, 11, small, EXTENDED_FILTER_ORDER) is not None
        assert load_shard(path, 12, small, EXTENDED_FILTER_ORDER) is None   # wrong seed
        assert load_shard(path, 11, meta, EXTENDED_FILTER_ORDER) is None    # wrong W
        assert load_shard(path, 11, dict(small, ensemble_size=20),
                          EXTENDED_FILTER_ORDER) is None
        assert load_shard(path, 11, dict(small, rank_convention="one-based"),
                          EXTENDED_FILTER_ORDER) is None
        assert load_shard(path, 11, small, EXTENDED_FILTER_ORDER + ["Nope"]) is None


def test_existing_baseline_outputs_remain_readable():
    """The patched runner must not have broken the shards already on disk."""
    config = load_config()
    seed = resolve_seeds(config, 1)[0]
    path = shard_path(seed, EXTENDED_RESULTS_DIR)
    if not path.exists():
        return
    with np.load(path) as data:
        for label in EXTENDED_FILTER_ORDER:
            for metric in METRICS:
                assert f"{label}__{metric}" in data.files, f"{label}__{metric} missing"
            hist = data[f"{label}__rank_hist"]
            assert hist.shape == (config["assimilation"]["ensemble_size"] + 1,)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
    print("\nall passed" if not failures else f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
