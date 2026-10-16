"""Correctness checks for the independent-restart versus cycled experiment.

The checks that matter here are structural rather than numerical: the whole
comparison is only meaningful if the restart regime really does discard its
ensemble between windows, the cycled regime really does carry it, both regimes
see the same random numbers wherever they can, and the calibration never sees an
evaluation trajectory.  Each of those is asserted directly rather than inferred
from an aggregate number looking plausible.

Run:  python -m pytest tests/test_restart_vs_cycled.py -v
  or: python tests/test_restart_vs_cycled.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import bias_variance_mc as bv  # noqa: E402
from src.experiments import restart_vs_cycled as rvc  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _config() -> dict:
    """The Lorenz-96 configuration the experiment runs on, minus the horizon."""
    return {
        "model": {"n": 40, "F": 8.0, "dt": 0.01, "spinup_steps": 0},
        "observations": {
            "m": 20,
            "observe_every": 2,
            "noise_std": 1.5,
            "steps_per_obs": 10,
        },
        "assimilation": {"window_length": 5, "n_windows": 20, "ensemble_size": 10},
        "filters": {"qpca_endcf": {"k": 1, "localization_radius": None}},
        "restart_vs_cycled": {},
    }


def _tiny_settings(**overrides) -> rvc.ExperimentSettings:
    base = dict(
        ensemble_sizes=(5, 10),
        n_mc=4,
        n_windows=4,
        window_stride=2,
        spinup_windows=2,
        n_calibration_seeds=2,
        n_evaluation_seeds=2,
        calibration_ensemble_size=10,
        calibration_n_mc=3,
        calibration_n_windows=8,
        calibration_spinup_windows=2,
        label="test",
    )
    base.update(overrides)
    return rvc.settings_from_config(_config(), **base)


def _random_distribution(n: int = 40, seed: int = 7) -> rvc.RestartDistribution:
    """A structured (non-isotropic) restart law, built from a random factor."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    B = rng.standard_normal((n, n)) / np.sqrt(n)
    return rvc.build_restart_distribution(
        A @ A.T + 0.1 * np.eye(n),
        0.5 * (B @ B.T) + 0.2 * np.eye(n),
        name="test-law",
        n_samples=1000,
        eigenvalue_floor=1e-6,
        shrinkage=0.0,
    )


# --------------------------------------------------------------------------- #
# 1-4: restart ensemble generation
# --------------------------------------------------------------------------- #


def test_restart_ensemble_generation_is_reproducible():
    """The same (r, w, N) keys must give bit-identical ensembles, every time."""
    distribution = _random_distribution()
    truth_state = np.linspace(-3.0, 3.0, 40)

    def draw():
        return distribution.draw(
            truth_state,
            10,
            6,
            rvc.restart_mean_generator(12345, 7),
            rvc.restart_anomaly_generator(12345, 7, 10),
        )

    assert np.array_equal(draw(), draw())

    # A different window, realization count aside, must give a different draw.
    other = distribution.draw(
        truth_state,
        10,
        6,
        rvc.restart_mean_generator(12345, 8),
        rvc.restart_anomaly_generator(12345, 8, 10),
    )
    assert not np.allclose(draw(), other)


def test_shared_mean_error_and_anomalies_are_separated():
    """x = truth + b + eps~, with b shared by every member of a realization.

    The ensemble mean must therefore be exactly ``truth + b`` -- no
    within-ensemble sampling noise leaks into the shared mean error -- and the
    shared error must be *reused* across ensemble sizes, because its stream is
    keyed on (r, w) only.
    """
    distribution = _random_distribution()
    truth_state = np.zeros(40)
    M = 5

    means = {}
    for N in (5, 10, 40):
        block = distribution.draw(
            truth_state,
            N,
            M,
            rvc.restart_mean_generator(99, 3),
            rvc.restart_anomaly_generator(99, 3, N),
        )
        members = block.T.reshape(M, N, 40)
        means[N] = members.mean(axis=1)

    # The shared mean error does not depend on N by construction.
    assert np.allclose(means[5], means[10], atol=1e-12)
    assert np.allclose(means[5], means[40], atol=1e-12)

    # And it is not the zero vector -- i.e. the forecast mean is not an oracle.
    assert np.linalg.norm(means[10]) > 1.0


def test_per_window_mean_error_is_shared_across_realizations():
    """The companion design must fix the forecast distribution at the window.

    Under ``per_window`` every realization sees the same ``b``, so the ensemble
    means coincide exactly and the realization-to-realization variability comes
    only from the within-ensemble anomalies -- which is what makes the N^-1
    question well posed.  Realizations must still differ, or a deterministic
    filter such as QPCA-EnDCF would collapse to a single trajectory.
    """
    distribution = _random_distribution()
    truth_state = np.zeros(40)
    M, N = 6, 8

    shared = distribution.draw(
        truth_state, N, M,
        rvc.restart_mean_generator(3, 1),
        rvc.restart_anomaly_generator(3, 1, N),
        shared_mean_across_realizations=True,
    ).T.reshape(M, N, 40)
    means = shared.mean(axis=1)
    assert np.allclose(means - means[0], 0.0, atol=1e-12)
    assert np.linalg.norm(means[0]) > 1.0  # not an oracle mean either

    # Realizations are still distinct through their anomalies.
    anomalies = shared - means[:, None, :]
    assert not np.allclose(anomalies[0], anomalies[1])

    per_realization = distribution.draw(
        truth_state, N, M,
        rvc.restart_mean_generator(3, 1),
        rvc.restart_anomaly_generator(3, 1, N),
        shared_mean_across_realizations=False,
    ).T.reshape(M, N, 40)
    spread = per_realization.mean(axis=1)
    assert not np.allclose(spread - spread[0], 0.0)

    # The mode is a scientifically relevant setting and must key the checkpoint.
    baseline = rvc.config_hash(rvc.scientific_config(_config(), _tiny_settings()))
    other = rvc.config_hash(
        rvc.scientific_config(_config(), _tiny_settings(mean_error_mode="per_window"))
    )
    assert baseline != other
    with pytest.raises(ValueError, match="mean_error_mode"):
        _tiny_settings(mean_error_mode="nonsense")


def test_restart_anomalies_sum_to_zero_after_centering():
    distribution = _random_distribution()
    truth_state = np.full(40, 2.0)
    M, N = 8, 7
    block = distribution.draw(
        truth_state,
        N,
        M,
        rvc.restart_mean_generator(5, 5),
        rvc.restart_anomaly_generator(5, 5, N),
    )
    members = block.T.reshape(M, N, 40)
    anomalies = members - members.mean(axis=1, keepdims=True)
    # Exactly centered, not merely centered in expectation.
    assert np.abs(anomalies.sum(axis=1)).max() < 1e-11


def test_empirical_restart_covariance_approaches_the_target():
    """A large synthetic draw must reproduce both target covariances."""
    distribution = _random_distribution()
    report = rvc.validate_restart_draws(distribution, ensemble_size=10, n_draws=20000)

    assert abs(report["trace_ratio_mean"] - 1.0) < 0.05
    assert abs(report["trace_ratio_within"] - 1.0) < 0.05
    # Structure, not just scale.
    assert report["relative_frobenius_mean"] < 0.10
    assert report["relative_frobenius_within"] < 0.10
    assert report["max_abs_anomaly_sum"] < 1e-9


# --------------------------------------------------------------------------- #
# 5: seed hygiene
# --------------------------------------------------------------------------- #


def test_calibration_and_evaluation_seeds_are_disjoint():
    calibration, evaluation = rvc.make_seed_sets(10, 50)
    assert len(calibration) == 10 and len(evaluation) == 50
    assert not set(calibration) & set(evaluation)
    assert len(set(calibration)) == len(calibration)
    assert len(set(evaluation)) == len(evaluation)

    settings = _tiny_settings(n_calibration_seeds=4, n_evaluation_seeds=6)
    assert not set(settings.calibration_seeds) & set(settings.evaluation_seeds)

    # An explicit overlapping pair must be rejected, not silently deduplicated.
    with pytest.raises(ValueError, match="overlap"):
        _tiny_settings(calibration_seeds=(1, 2, 3), evaluation_seeds=(3, 4, 5))


# --------------------------------------------------------------------------- #
# 6-8: regime semantics and matched random numbers
# --------------------------------------------------------------------------- #


def _instrumented_windows(monkeypatch):
    """Record the (input, output) ensemble block of every window."""
    record = []
    original = rvc.run_one_window

    def wrapper(filter_name, block, *args, **kwargs):
        entering = block.copy()
        out_block, means, diverged = original(filter_name, block, *args, **kwargs)
        record.append((entering, out_block.copy()))
        return out_block, means, diverged

    monkeypatch.setattr(rvc, "run_one_window", wrapper)
    return record


def _short_problem(n_windows: int = 12):
    config = _config()
    config["assimilation"]["n_windows"] = n_windows
    truth, observations, H = bv.generate_truth_batch(config, [4242])[4242]
    return config, truth, observations, H


def test_restart_regime_never_reuses_an_analysis_ensemble(monkeypatch):
    """Each evaluated restart window must start from a fresh draw, not the last analysis."""
    record = _instrumented_windows(monkeypatch)
    config, truth, observations, H = _short_problem()
    context = rvc.build_window_context(config, 1)
    distribution = _random_distribution()
    windows = [2, 4, 6]
    M, N, seed = 3, 5, 4242

    rvc.run_restart_cell(
        "QPCA-EnDCF", truth, observations, H, context, distribution,
        seed, N, M, windows,
    )

    assert len(record) == len(windows)
    L = context.L
    for index, w in enumerate(windows):
        entering, _leaving = record[index]
        expected = distribution.draw(
            truth[w * L],
            N,
            M,
            rvc.restart_mean_generator(seed, w),
            rvc.restart_anomaly_generator(seed, w, N),
        )
        assert np.array_equal(entering, expected), f"window {w} did not start fresh"

    for index in range(len(windows) - 1):
        leaving = record[index][1]
        entering_next = record[index + 1][0]
        assert not np.allclose(leaving, entering_next), (
            "the analysis ensemble of one evaluated restart window reached the next"
        )


def test_cycled_regime_reuses_the_previous_analysis_ensemble(monkeypatch):
    record = _instrumented_windows(monkeypatch)
    config, truth, observations, H = _short_problem()
    context = rvc.build_window_context(config, 1)
    distribution = _random_distribution()
    windows = [4, 6]

    rvc.run_cycled_cell(
        "QPCA-EnDCF", truth, observations, H, context, distribution,
        4242, 5, 3, windows,
    )

    # Every window from 0 to the last evaluated one is integrated.
    assert len(record) == max(windows) + 1
    for index in range(len(record) - 1):
        leaving = record[index][1]
        entering_next = record[index + 1][0]
        assert np.array_equal(leaving, entering_next), (
            f"window {index + 1} did not inherit window {index}'s analysis ensemble"
        )


def test_divergence_is_recorded_and_never_silently_recovered():
    """A realization that blows up mid-window stays marked for that window.

    ``_sanitize`` zeroes a diverged realization's columns so LAPACK is not handed
    NaNs on behalf of the healthy realizations. A zero Lorenz-96 state is finite
    and relaxes toward F, so a per-observation-time divergence test would find
    the realization "recovered" and record a fabricated trajectory as valid.
    Seq-EnKF is the case that matters: it is the only filter that sanitizes
    inside the window, because it is the only one that analyses inside it.
    """
    config, truth, observations, H = _short_problem()
    context = rvc.build_window_context(config, 1)
    M, N = 4, 5

    for name in ("Seq-EnKF", "4D-EnKF", "QPCA-EnDCF"):
        rng = np.random.default_rng(0)
        block = truth[0][:, None] + 0.5 * rng.standard_normal((40, M * N))
        block[:, 1 * N] = np.nan  # realization 1, member 0

        _out, means, diverged = rvc.run_one_window(
            name, block.copy(), observations[:context.L], H, context, M, N,
            rvc.observation_perturbation_generator(name, 1, 0, N)
            if rvc.requires_observation_perturbations(name)
            else None,
        )
        assert diverged.tolist() == [False, True, False, False], f"{name}: {diverged}"
        assert np.isnan(means[1]).all(), f"{name} reported a value for a diverged run"
        assert np.isfinite(means[[0, 2, 3]]).all(), (
            f"{name}: a healthy realization was contaminated by the diverged one"
        )

    # summarize_cell must then drop that realization from the Gram and say so.
    errors = np.zeros((M, 3, 40))
    diverged_grid = np.zeros((M, 3), dtype=bool)
    diverged_grid[1, 2] = True
    errors[1, 2] = np.nan
    summary = rvc.summarize_cell(errors, diverged_grid)
    assert int(summary["n_valid_realizations"]) == M - 1
    assert summary["error_gram"].shape == (M - 1, M - 1)
    assert np.isfinite(summary["mse_trace"]).all()


def test_matched_random_numbers_are_preserved_across_filters(monkeypatch):
    """All filters must face identical restart ensembles at matched (r, w, t, N)."""
    config, truth, observations, H = _short_problem()
    context = rvc.build_window_context(config, 1)
    distribution = _random_distribution()
    windows = [2, 4]

    entering_by_filter = {}
    for name in ("Seq-EnKF", "4D-EnKF", "QPCA-EnDCF"):
        record = _instrumented_windows(monkeypatch)
        rvc.run_restart_cell(
            name, truth, observations, H, context, distribution, 4242, 5, 3, windows
        )
        entering_by_filter[name] = [pair[0] for pair in record]

    reference = entering_by_filter["Seq-EnKF"]
    for name, blocks in entering_by_filter.items():
        for index, block in enumerate(blocks):
            assert np.array_equal(block, reference[index]), (
                f"{name} saw a different restart ensemble at evaluated window {index}"
            )

    # Observation perturbations are keyed by filter (so adding a filter does not
    # repaint another) but NOT by regime (so restart and cycled are paired).
    a = rvc.observation_perturbation_generator("Seq-EnKF", 4242, 3, 10).standard_normal(8)
    b = rvc.observation_perturbation_generator("Seq-EnKF", 4242, 3, 10).standard_normal(8)
    c = rvc.observation_perturbation_generator("4D-EnKF", 4242, 3, 10).standard_normal(8)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


# --------------------------------------------------------------------------- #
# 9-10: validation of inputs
# --------------------------------------------------------------------------- #


def test_unsupported_filters_are_rejected_at_construction():
    """The cycled decomposition's filter set must not leak into this experiment.

    ``bias_variance_mc.FILTER_NAMES`` carries five filters; ``run_one_window``
    implements three. Inheriting that tuple would redefine the experiment and
    change its configuration hash, so the default is pinned and anything
    unsupported is refused up front rather than failing inside a worker.
    """
    assert rvc.ExperimentSettings.__dataclass_fields__["filter_names"].default == (
        rvc.RESTART_SUPPORTED_FILTERS
    )
    with pytest.raises(ValueError, match="supports"):
        _tiny_settings(filter_names=("Seq-EnKF", "Seq-ETKF"))
    with pytest.raises(ValueError, match="supports"):
        _tiny_settings(filter_names=bv.FILTER_NAMES)


def test_invalid_kappa_is_rejected():
    # kappa must satisfy 1 <= kappa <= min(N - 1, d).
    assert rvc.validate_kappa(1, 5, 100) == 1
    assert rvc.validate_kappa(4, 5, 100) == 4
    for bad in (5, 6, 40):
        with pytest.raises(ValueError, match="rank at most"):
            rvc.validate_kappa(bad, 5, 100)
    with pytest.raises(ValueError):
        rvc.validate_kappa(0, 10, 100)
    # Rejected at construction time too, for every size in the sweep.
    with pytest.raises(ValueError, match="rank at most"):
        _tiny_settings(kappa=6, ensemble_sizes=(5, 10))


def test_covariance_stabilization_and_rejection():
    n = 12
    rng = np.random.default_rng(3)
    A = rng.standard_normal((n, n))
    good = A @ A.T

    # A small negative eigenvalue (roundoff, or a finite-sample artefact) is
    # stabilized, and the stabilized matrix is a valid, symmetric, PSD law.
    values, vectors = np.linalg.eigh(good)
    values[0] = -1e-12
    perturbed = (vectors * values) @ vectors.T
    P, factor, raw, stabilized, rank = rvc._stabilize_covariance(
        perturbed, 1e-6, 0.0, "test"
    )
    assert raw.min() < 0.0, "the test input should carry a negative eigenvalue"
    assert stabilized.min() > 0.0
    assert np.allclose(P, P.T)
    assert np.allclose(factor @ factor.T, P, atol=1e-10)
    assert np.linalg.eigvalsh(P).min() > 0.0
    assert 1.0 <= rank <= n

    # A covariance with no positive eigenvalue cannot be sampled from, and is
    # rejected rather than floored into something meaningless.
    with pytest.raises(ValueError, match="no positive eigenvalue"):
        rvc._stabilize_covariance(np.zeros((n, n)), 1e-6, 0.0, "test")
    with pytest.raises(ValueError, match="no positive eigenvalue"):
        rvc._stabilize_covariance(-good, 1e-6, 0.0, "test")

    # Non-finite entries are a failure, not something to shrink away.
    broken = good.copy()
    broken[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        rvc._stabilize_covariance(broken, 1e-6, 0.0, "test")

    # Shrinkage moves the spectrum toward the isotropic target.
    _P, _f, _raw, shrunk, _r = rvc._stabilize_covariance(good, 1e-6, 0.5, "test")
    spread = lambda v: v.max() / v.min()  # noqa: E731
    assert spread(shrunk) < spread(np.linalg.eigvalsh(good))


# --------------------------------------------------------------------------- #
# 11: configuration hashing
# --------------------------------------------------------------------------- #


def test_config_hash_tracks_scientifically_relevant_changes():
    config = _config()
    settings = _tiny_settings()
    baseline = rvc.config_hash(rvc.scientific_config(config, settings))

    # Presentational / scheduling changes must NOT invalidate a checkpoint.
    assert (
        rvc.config_hash(rvc.scientific_config(config, _tiny_settings(label="other")))
        == baseline
    )

    # Scientific changes must.
    changed_settings = {
        "kappa": 2,
        "n_mc": 8,
        "n_windows": 5,
        "window_stride": 3,
        "spinup_windows": 4,
        "ensemble_sizes": (5, 10, 20),
        "n_evaluation_seeds": 3,
        "calibration_n_windows": 9,
        "calibration_ensemble_size": 20,
        "eigenvalue_floor": 1e-4,
        "shrinkage": 0.1,
        "restart_mode": "filter_marginal",
    }
    for key, value in changed_settings.items():
        digest = rvc.config_hash(
            rvc.scientific_config(config, _tiny_settings(**{key: value}))
        )
        assert digest != baseline, f"changing {key} did not change the config hash"

    for section, key, value in (
        ("observations", "noise_std", 2.0),
        ("model", "F", 8.5),
        ("assimilation", "window_length", 4),
    ):
        altered = _config()
        altered[section][key] = value
        assert rvc.config_hash(rvc.scientific_config(altered, settings)) != baseline

    # And it is stable across processes: no reliance on Python's hash().
    assert rvc.config_hash(rvc.scientific_config(_config(), _tiny_settings())) == baseline


# --------------------------------------------------------------------------- #
# 12: the published hierarchical path is untouched
# --------------------------------------------------------------------------- #


def test_existing_hierarchical_behaviour_is_unchanged():
    """The published N=10 path must be unaffected by this module.

    Two things could quietly break it: a change to the initial-ensemble
    construction the published run used, and this module disturbing the legacy
    global RNG that the published kernels draw from.  Both are checked.
    """
    config = {
        "experiment": {"seeds": [42]},
        "model": {"n": 40, "F": 8.0, "dt": 0.01, "spinup_steps": 0},
        "observations": {
            "m": 20,
            "observe_every": 2,
            "noise_std": 1.5,
            "steps_per_obs": 10,
        },
        "assimilation": {"window_length": 5, "n_windows": 3, "ensemble_size": 10},
        "bias_variance": {"n_monte_carlo": 3, "spinup_windows": 0},
        "filters": {"qpca_endcf": {"k": 1}},
    }
    truth, observations, H = bv.generate_truth_and_observations(config, 42)

    # (a) The initial ensemble is still truth_0 + 0.5 xi_0 + 0.5 xi_j drawn from
    #     the legacy global stream, member by member, in that order.
    np.random.seed(10000)
    expected = np.empty((40, 10))
    xi_0 = np.random.randn(40)
    for j in range(10):
        expected[:, j] = truth[0] + 0.5 * xi_0 + 0.5 * np.random.randn(40)
    np.random.seed(10000)
    assert np.array_equal(bv._initial_ensemble(truth[0], 40, 10), expected)

    # (b) Running this module's machinery in between does not perturb it.
    before = {
        name: bv.run_filter_batched(name, config, truth, observations, H, range(3))[0]
        for name in bv.FILTER_NAMES
    }

    distribution = _random_distribution()
    context = rvc.build_window_context(config, 1)
    rvc.run_restart_cell(
        "Seq-EnKF", truth, observations, H, context, distribution, 42, 5, 2, [0, 1]
    )
    rvc.validate_restart_draws(distribution, 10, n_draws=200)

    for name in bv.FILTER_NAMES:
        after = bv.run_filter_batched(name, config, truth, observations, H, range(3))[0]
        assert np.array_equal(before[name], after), (
            f"{name}'s published path changed after exercising restart_vs_cycled"
        )


# --------------------------------------------------------------------------- #
# 13-14: end-to-end smoke and the decomposition identity
# --------------------------------------------------------------------------- #


def test_smoke_experiment_completes_with_finite_outputs(tmp_path):
    config = _config()
    settings = _tiny_settings()
    outcome = rvc.run_restart_vs_cycled(
        config, settings, n_jobs=1, results_root=tmp_path, resume=False, verbose=0
    )

    cells = outcome["cells"]
    expected = [
        (regime, name, int(N), int(seed))
        for regime in rvc.REGIMES
        for name in settings.filter_names
        for N in settings.ensemble_sizes
        for seed in settings.evaluation_seeds
    ]
    assert set(cells) == set(expected), "some (regime, filter, N, seed) cell is missing"

    for key, cell in cells.items():
        for metric in ("mse_trace", "variance_trace", "bias_squared_trace"):
            values = cell[metric]
            assert np.isfinite(values).all(), f"{key}: non-finite {metric}"
        assert (cell["mse_trace"] >= 0).all()
        assert (cell["variance_trace"] >= -1e-10).all()
        assert int(cell["n_valid_realizations"]) == settings.n_mc

    report = rvc.complete_truth_seeds(cells, settings)
    assert len(report["common"]) == len(settings.evaluation_seeds)

    # The manifest and calibration artefacts exist and round-trip.
    results_dir = outcome["results_dir"]
    assert (results_dir / "manifest.json").exists()
    reloaded = rvc.load_calibration(results_dir / "calibration" / "restart_distribution.npz")
    assert reloaded["common"].identifier == outcome["calibration"]["common"].identifier


def test_decomposition_satisfies_the_plugin_mse_identity():
    """MSE = Bias^2 + Variance must hold numerically, window by window."""
    rng = np.random.default_rng(11)
    M, W, n = 32, 9, 40
    errors = rng.standard_normal((M, W, n)) * 1.7 + 0.4
    diverged = np.zeros((M, W), dtype=bool)

    summary = rvc.summarize_cell(errors, diverged)
    identity = summary["bias_squared_trace"] + summary["variance_trace"]
    assert np.allclose(summary["mse_trace"], identity, rtol=1e-12, atol=1e-12)

    # The same identity holds for the quantities the bootstrap derives from the
    # error Gram, which is what makes the Gram a sufficient statistic.
    gram = summary["error_gram"]
    counts = np.full(M, 1)
    mse = counts @ np.diag(gram) / M
    bias2 = counts @ gram @ counts / M**2
    assert np.isclose(mse, summary["mse_trace"].mean(), rtol=1e-12)
    assert np.isclose(bias2, summary["bias_squared_trace"].mean(), rtol=1e-12)
    assert np.isclose(mse - bias2, summary["variance_trace"].mean(), rtol=1e-10)

    # The debiased estimator is the plug-in minus Var/(M-1), retained as-is --
    # small negative values are meaningful and are not clipped away.
    assert np.allclose(
        summary["bias_squared_debiased_trace"],
        summary["bias_squared_trace"] - summary["variance_trace"] / (M - 1),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# 15-16: inference
# --------------------------------------------------------------------------- #


def _synthetic_cells(settings, gram_by_cell):
    cells = {}
    for (regime, name, N), gram in gram_by_cell.items():
        for seed in settings.evaluation_seeds:
            cells[(regime, name, int(N), int(seed))] = {
                "error_gram": gram,
                "error_gram_stride2": gram,
                "n_realizations": np.asarray(settings.n_mc),
                "n_valid_realizations": np.asarray(settings.n_mc),
            }
    return cells


def test_bootstrap_resamples_seeds_and_realizations_but_not_windows():
    """Windows are never an independent resampling unit.

    Two cell sets built from the *same* window-averaged error Gram but from
    different per-window arrangements must give identical bootstrap draws.  A
    bootstrap that resampled individual windows could not have that property.
    """
    settings = _tiny_settings(n_mc=12, n_evaluation_seeds=3, ensemble_sizes=(5, 10, 20))
    rng = np.random.default_rng(5)
    M, W, n = 12, 7, 6

    errors = rng.standard_normal((M, W, n))
    shuffled = errors[:, rng.permutation(W), :]  # same gram, different window order
    gram_a = rvc.summarize_cell(errors, np.zeros((M, W), bool))["error_gram"]
    gram_b = rvc.summarize_cell(shuffled, np.zeros((M, W), bool))["error_gram"]
    assert np.allclose(gram_a, gram_b, atol=1e-12)

    grid = [
        (regime, name, N)
        for regime in rvc.REGIMES
        for name in settings.filter_names
        for N in settings.ensemble_sizes
    ]
    draws_a = rvc.hierarchical_bootstrap_regimes(
        _synthetic_cells(settings, {cell: gram_a for cell in grid}),
        settings.evaluation_seeds, settings, n_bootstrap=64, n_inner=16,
    )
    draws_b = rvc.hierarchical_bootstrap_regimes(
        _synthetic_cells(settings, {cell: gram_b for cell in grid}),
        settings.evaluation_seeds, settings, n_bootstrap=64, n_inner=16,
    )
    for cell in grid:
        assert np.allclose(draws_a["draws"][cell], draws_b["draws"][cell])

    # The outer level resamples whole truth seeds: with identical grams in every
    # cell, restart and cycled must be paired exactly, so A_cycle == 1 in every
    # single draw. That only holds if both regimes share outer and inner indices.
    for name in settings.filter_names:
        for N in settings.ensemble_sizes:
            assert np.allclose(draws_a["amplification"][(name, int(N))], 1.0)

    # And every truth seed supplied is used -- none are dropped.
    assert sorted(draws_a["truth_seeds"]) == sorted(settings.evaluation_seeds)


def test_variance_slope_recovers_the_exponent_on_synthetic_data():
    sizes = [5, 10, 20, 40]
    for beta in (0.5, 1.0, 1.7):
        for C in (0.3, 12.0):
            variances = [C * N**-beta for N in sizes]
            alpha, fitted = rvc.fit_variance_slope(sizes, variances)
            assert np.isclose(fitted, beta, atol=1e-10)
            assert np.isclose(np.exp(alpha), C, rtol=1e-10)

    # Two ensemble sizes are refused outright rather than silently interpolated.
    with pytest.raises(ValueError, match="at least three ensemble sizes"):
        rvc.fit_variance_slope([10, 20], [1.0, 0.5])

    # Non-positive or non-finite variances cannot be logged; the fit reports
    # NaN rather than raising or quietly dropping the point.
    alpha, fitted = rvc.fit_variance_slope(sizes, [1.0, 0.0, -1.0, np.nan])
    assert np.isnan(alpha) and np.isnan(fitted)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
