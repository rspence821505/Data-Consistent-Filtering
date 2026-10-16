"""Correctness checks for the ``(N, kappa)`` bias--variance response-surface experiment.

Three classes of check, in the order they matter.

1. **The filter is unchanged.**  The kernel here adds diagnostics to QPCA-EnDCF;
   it must not add arithmetic that reaches the analysis.  The analysis means are
   asserted bit-identical against ``restart_vs_cycled.run_one_window``, which is
   itself pinned against ``bias_variance_mc``'s reference kernels.  Without this
   the two implementations drift and the response surface stops describing the
   filter the rest of the paper measures.

2. **The diagnostics mean what the docstrings say.**  The projector
   identities -- ``kappa - ||Pbar||_F^2`` as a Frobenius variance, and as the
   mean over realization pairs of the sum of squared sines of the principal
   angles -- are checked against a brute-force computation rather than trusted.

3. **The design is what it claims.**  The rank constraint, the kappa-independent
   ensemble streams that make the kappa comparison paired, the restart regime's
   refusal to carry an ensemble between windows, and the configuration hash.

Run:  python -m pytest tests/test_n_kappa_bias_variance.py -v
  or: python tests/test_n_kappa_bias_variance.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import n_kappa_bias_variance as nk  # noqa: E402
from src.experiments import restart_vs_cycled as rvc  # noqa: E402

PUBLISHED_RESTART_LAW = (
    PROJECT_ROOT / "results" / "published" / "lorenz96" / "restart_law_2fbc05cb06f87eaa.npz"
)


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
        "n_kappa": {},
    }


def _tiny_settings(**overrides) -> nk.NKappaSettings:
    base = dict(
        ensemble_sizes=(5, 10),
        kappa_values=(1, 2),
        n_mc=6,
        n_windows=4,
        window_stride=2,
        spinup_windows=2,
        n_evaluation_seeds=2,
        label="test",
        stacked_obs_dimension=100,
    )
    base.update(overrides)
    return nk.NKappaSettings(**base)


def _window_inputs(seed: int = 7, M: int = 5, N: int = 10):
    """One window's worth of forecast ensembles, observations and operator."""
    config = _config()
    settings = _tiny_settings()
    truth_data = nk.generate_truth_batch(config, [settings.evaluation_seeds[0]])
    truth, observations, H = truth_data[settings.evaluation_seeds[0]]
    rng = np.random.default_rng(seed)
    n = config["model"]["n"]
    block = truth[0][:, None] + 0.5 * rng.standard_normal((n, M * N))
    return config, block, observations[:5], H, M, N


# --------------------------------------------------------------------------- #
# 1. The filter is unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kappa", [1, 2, 4])
def test_kernel_matches_the_reference_qpca_window_bit_for_bit(kappa):
    """The diagnostics kernel and ``run_one_window`` are the same filter.

    Bit-identical, not merely close: both call the same imported helpers in the
    same order on the same inputs, so any difference at all would mean the
    diagnostics changed the arithmetic that feeds the analysis.
    """
    config, block, obs_window, H, M, N = _window_inputs()
    context = rvc.build_window_context(config, kappa)

    reference_block, reference_means, reference_diverged = rvc.run_one_window(
        "QPCA-EnDCF", block.copy(), obs_window, H, context, M, N, None
    )
    (
        new_block,
        new_means,
        new_diverged,
        realization_diagnostics,
        window_diagnostics,
    ) = nk.run_qpca_window(
        block.copy(), obs_window, H, context, M, N, kappa, collect_diagnostics=True
    )

    assert np.array_equal(reference_means, new_means)
    assert np.array_equal(reference_block, new_block)
    assert np.array_equal(reference_diverged, new_diverged)
    assert realization_diagnostics.shape == (M, len(nk.REALIZATION_DIAGNOSTICS))
    assert np.isfinite(realization_diagnostics).all()
    assert set(window_diagnostics) == set(nk.PROJECTOR_DIAGNOSTICS)


def test_collecting_diagnostics_does_not_change_the_analysis():
    """The flag adds a dict, not arithmetic."""
    config, block, obs_window, H, M, N = _window_inputs()
    context = rvc.build_window_context(config, 2)
    _b_on, means_on, _d_on, realization, window = nk.run_qpca_window(
        block.copy(), obs_window, H, context, M, N, 2, collect_diagnostics=True
    )
    _b_off, means_off, _d_off, none_realization, none_window = nk.run_qpca_window(
        block.copy(), obs_window, H, context, M, N, 2, collect_diagnostics=False
    )
    assert np.array_equal(means_on, means_off)
    assert realization is not None and window is not None
    assert none_realization is None and none_window is None


# --------------------------------------------------------------------------- #
# 2. The diagnostics mean what the docstrings say
# --------------------------------------------------------------------------- #


def _random_projectors(d: int, kappa: int, M: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((M, d, d))
    for t in range(M):
        basis, _ = np.linalg.qr(rng.standard_normal((d, kappa)))
        out[t] = basis @ basis.T
    return out


@pytest.mark.parametrize("kappa", [1, 3])
def test_projector_variability_is_a_frobenius_variance(kappa):
    """``kappa - ||Pbar||_F^2`` equals ``M^-1 sum_t ||Pi_t - Pbar||_F^2``."""
    d, M = 12, 9
    projectors = _random_projectors(d, kappa, M)
    summary = nk._projector_summary(
        projectors.sum(axis=0), np.zeros(d), kappa, M
    )

    mean_projector = projectors.mean(axis=0)
    brute_force = np.mean(
        [np.linalg.norm(p - mean_projector, "fro") ** 2 for p in projectors]
    )
    assert summary["projector_variability"] == pytest.approx(brute_force, rel=1e-10)
    assert summary["projector_variability_normalized"] == pytest.approx(
        brute_force / kappa, rel=1e-10
    )


@pytest.mark.parametrize("kappa", [1, 3])
def test_projector_variability_is_the_mean_squared_sine_of_principal_angles(kappa):
    """The same number, read as subspace geometry.

    ``||Pi_1 - Pi_2||_F^2 = 2 sum_i sin^2(theta_i)`` for two rank-kappa
    projectors, so the pairwise mean of ``sum_i sin^2(theta_i)`` is exactly
    ``kappa - ||Pbar||_F^2``.  That is what makes the normalized value a number
    in [0, 1] rather than an arbitrary scale.
    """
    d, M = 12, 9
    projectors = _random_projectors(d, kappa, M, seed=11)
    summary = nk._projector_summary(projectors.sum(axis=0), np.zeros(d), kappa, M)

    total = 0.0
    for a in range(M):
        for b in range(M):
            # Principal angles between range(Pi_a) and range(Pi_b): the singular
            # values of U_a^T U_b are cos(theta_i).
            u_a = np.linalg.eigh(projectors[a])[1][:, -kappa:]
            u_b = np.linalg.eigh(projectors[b])[1][:, -kappa:]
            cosines = np.clip(np.linalg.svd(u_a.T @ u_b, compute_uv=False), 0.0, 1.0)
            total += float(np.sum(1.0 - cosines**2))
    assert summary["projector_variability"] == pytest.approx(total / M**2, abs=1e-9)
    assert 0.0 <= summary["projector_variability_normalized"] <= 1.0


def test_identical_projectors_have_zero_variability():
    d, kappa, M = 8, 2, 5
    basis, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((d, kappa)))
    projector = basis @ basis.T
    summary = nk._projector_summary(
        np.repeat(projector[None], M, axis=0).sum(axis=0), np.zeros(d), kappa, M
    )
    assert summary["projector_variability"] == pytest.approx(0.0, abs=1e-12)
    assert summary["projector_error_vs_mean_subspace"] == pytest.approx(0.0, abs=1e-10)


def test_projector_error_vs_mean_subspace_matches_the_brute_force_definition():
    d, kappa, M = 10, 2, 7
    projectors = _random_projectors(d, kappa, M, seed=5)
    summary = nk._projector_summary(projectors.sum(axis=0), np.zeros(d), kappa, M)

    values, vectors = np.linalg.eigh(projectors.mean(axis=0))
    leading = vectors[:, np.argsort(values)[::-1][:kappa]]
    mean_subspace = leading @ leading.T
    brute_force = np.mean(
        [np.linalg.norm(p - mean_subspace, "fro") ** 2 for p in projectors]
    )
    assert summary["projector_error_vs_mean_subspace"] == pytest.approx(
        brute_force, rel=1e-10
    )


def test_pooled_truncation_energy_is_the_discarded_component_of_the_pooled_mean():
    """``||(I - Pibar_kappa) muhat_E||^2`` and its fraction of ``||muhat_E||^2``."""
    d, kappa, M = 10, 3, 6
    projectors = _random_projectors(d, kappa, M, seed=17)
    rng = np.random.default_rng(23)
    residual_means = rng.standard_normal((M, d))
    summary = nk._projector_summary(
        projectors.sum(axis=0), residual_means.sum(axis=0), kappa, M
    )

    values, vectors = np.linalg.eigh(projectors.mean(axis=0))
    leading = vectors[:, np.argsort(values)[::-1][:kappa]]
    pooled = residual_means.mean(axis=0)
    discarded = pooled - leading @ (leading.T @ pooled)
    assert summary["pooled_truncation_energy"] == pytest.approx(
        float(discarded @ discarded), rel=1e-10
    )
    assert summary["pooled_mean_residual_energy"] == pytest.approx(
        float(pooled @ pooled), rel=1e-12
    )
    assert summary["pooled_discarded_mean_mismatch_fraction"] == pytest.approx(
        float(discarded @ discarded) / float(pooled @ pooled), rel=1e-10
    )


def test_a_dead_realization_is_excluded_from_the_projector_statistics():
    """A realization that already diverged never rejoins the across-realization set.

    ``_sanitize`` zeroes a blown-up realization's columns so LAPACK is not handed
    NaNs on the healthy realizations' behalf, and a zero Lorenz-96 state is
    perfectly finite -- it relaxes toward F. A window that re-tested from scratch
    would find it "recovered" and fold a fabricated trajectory into the projector
    mean. The cycled runner passes its accumulated mask instead; this asserts the
    kernel honours it.
    """
    config, block, obs_window, H, M, N = _window_inputs(M=6)
    context = rvc.build_window_context(config, 2)
    exclude = np.zeros(M, dtype=bool)
    exclude[[1, 4]] = True

    _b, means, diverged, realization, window = nk.run_qpca_window(
        block.copy(), obs_window, H, context, M, N, 2,
        collect_diagnostics=True, exclude=exclude,
    )
    assert diverged[exclude].all() and not diverged[~exclude].any()
    assert np.isnan(means[exclude]).all() and np.isfinite(means[~exclude]).all()
    assert np.isnan(realization[exclude]).all()
    assert np.isfinite(realization[~exclude]).all()
    assert window["n_valid_realizations"] == M - 2

    # And the surviving realizations are unaffected by who was excluded.
    _b2, means2, _d2, _r2, _w2 = nk.run_qpca_window(
        block.copy(), obs_window, H, context, M, N, 2, collect_diagnostics=True
    )
    assert np.array_equal(means[~exclude], means2[~exclude])


def test_a_single_realization_yields_no_projector_variability_estimate():
    """One realization is recorded as NaN, never as a perfectly stable subspace."""
    summary = nk._projector_summary(np.zeros((6, 6)), np.zeros(6), 1, 1)
    assert np.isnan(summary["projector_variability"])
    assert summary["n_valid_realizations"] == 1.0


def test_retained_and_discarded_mean_mismatch_fractions_sum_to_one():
    config, block, obs_window, H, M, N = _window_inputs()
    context = rvc.build_window_context(config, 3)
    _b, _m, _d, realization, _w = nk.run_qpca_window(
        block, obs_window, H, context, M, N, 3, collect_diagnostics=True
    )
    retained = realization[:, nk.REALIZATION_DIAGNOSTICS.index("retained_mean_mismatch_fraction")]
    discarded = realization[
        :, nk.REALIZATION_DIAGNOSTICS.index("discarded_mean_mismatch_fraction")
    ]
    assert np.allclose(retained + discarded, 1.0)
    assert ((retained >= 0.0) & (retained <= 1.0)).all()


def test_retained_mean_mismatch_fraction_is_monotone_in_kappa():
    """A larger retained subspace cannot capture less of the mean mismatch.

    This is the geometric content of the truncation term
    ``||(I - Pi_kappa) mu_E||^2``: nested subspaces, so the discarded energy is
    non-increasing in kappa.  If this ever fails, the eigenvector ordering is
    wrong somewhere.
    """
    config, block, obs_window, H, M, N = _window_inputs(N=10)
    index = nk.REALIZATION_DIAGNOSTICS.index("retained_mean_mismatch_fraction")
    previous = np.zeros(M)
    for kappa in (1, 2, 4, 8):
        context = rvc.build_window_context(config, kappa)
        _b, _m, _d, realization, _w = nk.run_qpca_window(
            block.copy(), obs_window, H, context, M, N, kappa, collect_diagnostics=True
        )
        current = realization[:, index]
        assert (current >= previous - 1e-12).all()
        previous = current


def test_cutoff_gap_is_measured_at_the_retained_discarded_boundary():
    """The gap is ``lambda_kappa - lambda_{kappa+1}``, not ``lambda_1 - lambda_2``.

    ``bias_variance_mc`` stores the latter under the same name because it only
    ever runs kappa = 1.  Confusing the two would make the stability factor
    kappa/delta_kappa^2 wrong at every kappa > 1, which is most of this grid.
    """
    config, block, obs_window, H, M, N = _window_inputs(N=10)
    gap_index = nk.REALIZATION_DIAGNOSTICS.index("cutoff_gap")
    lead_index = nk.REALIZATION_DIAGNOSTICS.index("leading_eigenvalue")
    normalized_index = nk.REALIZATION_DIAGNOSTICS.index("normalized_cutoff_gap")
    stability_index = nk.REALIZATION_DIAGNOSTICS.index("stability_factor")

    context_1 = rvc.build_window_context(config, 1)
    _b, _m, _d, one, _w = nk.run_qpca_window(
        block.copy(), obs_window, H, context_1, M, N, 1, collect_diagnostics=True
    )
    context_3 = rvc.build_window_context(config, 3)
    _b, _m, _d, three, _w = nk.run_qpca_window(
        block.copy(), obs_window, H, context_3, M, N, 3, collect_diagnostics=True
    )

    # Recompute the spectrum independently of the kernel.
    mL = config["observations"]["m"] * config["assimilation"]["window_length"]
    from src.experiments.bias_variance_mc import _advance

    advanced = block.copy()
    Y = np.empty((5, config["observations"]["m"], M, N))
    for ell in range(5):
        advanced = _advance(advanced, 10, 0.01, 8.0)
        Y[ell] = (H @ advanced).reshape(config["observations"]["m"], M, N)
    z_stack = obs_window.ravel()
    for i in range(M):
        Y_stack = Y[:, :, i, :].reshape(mL, N)
        E = context_1.whiten @ (Y_stack - z_stack[:, None])
        E_c = E - E.mean(axis=1, keepdims=True)
        eigenvalues = np.linalg.svd(E_c, compute_uv=False) ** 2 / (N - 1)
        assert one[i, gap_index] == pytest.approx(eigenvalues[0] - eigenvalues[1])
        assert three[i, gap_index] == pytest.approx(eigenvalues[2] - eigenvalues[3])
        assert one[i, lead_index] == pytest.approx(eigenvalues[0])
        assert three[i, normalized_index] == pytest.approx(
            (eigenvalues[2] - eigenvalues[3]) / eigenvalues[0]
        )
        assert three[i, stability_index] == pytest.approx(
            3.0 / (eigenvalues[2] - eigenvalues[3]) ** 2
        )


# --------------------------------------------------------------------------- #
# 3. The design is what it claims
# --------------------------------------------------------------------------- #


def test_grid_is_triangular_under_the_rank_constraint():
    points = nk.valid_design_points((5, 10, 20, 40), (1, 2, 4, 8), 100)
    assert (5, 8) not in points  # kappa = 8 > N - 1 = 4
    assert (5, 4) in points
    assert len(points) == 15
    for N, kappa in points:
        assert 1 <= kappa <= N - 1
    # Also rejected by the shared validator, so a stray cell cannot slip through.
    with pytest.raises(ValueError):
        rvc.validate_kappa(8, 5, 100)


def test_stacked_observation_dimension_also_caps_kappa():
    assert nk.valid_design_points((10,), (1, 2, 4), 3) == [(10, 1), (10, 2)]


def test_restart_ensembles_are_identical_across_kappa_at_fixed_N():
    """The kappa comparison is paired at the level of the ensemble draw.

    Nothing in the stream keys depends on kappa, and QPCA-EnDCF draws no
    observation perturbations, so at a fixed N every kappa restarts from exactly
    the same forecast ensembles.  That is why this experiment can resolve small
    differences across kappa with a modest number of truth seeds.
    """
    law = rvc.bootstrap_restart_distribution(40)
    drawn = [
        law.draw(
            np.zeros(40),
            10,
            4,
            rvc.restart_mean_generator(12345, 7),
            rvc.restart_anomaly_generator(12345, 7, 10),
            shared_mean_across_realizations=True,
        )
        for _ in range(2)
    ]
    assert np.array_equal(drawn[0], drawn[1])


def test_restart_regime_never_reuses_an_analysis_ensemble(monkeypatch):
    """Every evaluated restart window must start from a freshly drawn ensemble.

    Asserted by recording the window-initial block of every window and checking
    that none of them is the previous window's analysis, rather than by trusting
    the loop body.
    """
    config = _config()
    settings = _tiny_settings()
    seed = int(settings.evaluation_seeds[0])
    evaluation_config = rvc._config_for_windows(config, settings.total_cycled_windows)
    truth, observations, H = nk.generate_truth_batch(evaluation_config, [seed])[seed]
    context = rvc.build_window_context(evaluation_config, 1)
    law = rvc.bootstrap_restart_distribution(config["model"]["n"])

    starts, ends = [], []
    original = nk.run_qpca_window

    def recording(block, *args, **kwargs):
        starts.append(block.copy())
        result = original(block, *args, **kwargs)
        ends.append(result[0].copy())
        return result

    monkeypatch.setattr(nk, "run_qpca_window", recording)
    nk.run_restart_cell(
        truth,
        observations,
        H,
        context,
        law,
        seed,
        10,
        1,
        settings.n_mc,
        [int(w) for w in settings.evaluation_windows],
    )

    assert len(starts) == settings.n_windows
    for previous_end, next_start in zip(ends[:-1], starts[1:]):
        assert not np.array_equal(previous_end, next_start)


def test_cycled_regime_carries_the_previous_analysis_ensemble(monkeypatch):
    config = _config()
    settings = _tiny_settings()
    seed = int(settings.evaluation_seeds[0])
    evaluation_config = rvc._config_for_windows(config, settings.total_cycled_windows)
    truth, observations, H = nk.generate_truth_batch(evaluation_config, [seed])[seed]
    context = rvc.build_window_context(evaluation_config, 1)
    law = rvc.bootstrap_restart_distribution(config["model"]["n"])

    starts, ends = [], []
    original = nk.run_qpca_window

    def recording(block, *args, **kwargs):
        starts.append(block.copy())
        result = original(block, *args, **kwargs)
        ends.append(result[0].copy())
        return result

    monkeypatch.setattr(nk, "run_qpca_window", recording)
    nk.run_cycled_cell(
        truth,
        observations,
        H,
        context,
        law,
        seed,
        10,
        1,
        settings.n_mc,
        [int(w) for w in settings.evaluation_windows],
    )

    assert len(starts) == settings.total_cycled_windows
    for previous_end, next_start in zip(ends[:-1], starts[1:]):
        assert np.array_equal(previous_end, next_start)


def test_config_hash_tracks_scientifically_relevant_changes():
    config = _config()
    settings = _tiny_settings()
    base = nk.config_hash(nk.scientific_config(config, settings, "law-id"))

    # The label is presentation only.
    assert base == nk.config_hash(
        nk.scientific_config(config, _tiny_settings(label="other"), "law-id")
    )
    # The restart law is not.
    assert base != nk.config_hash(nk.scientific_config(config, settings, "other-law"))
    # Nor is any part of the design.
    for changed in (
        _tiny_settings(kappa_values=(1, 2, 4)),
        _tiny_settings(ensemble_sizes=(5, 10, 20)),
        _tiny_settings(n_mc=7),
        _tiny_settings(n_windows=5),
        _tiny_settings(window_stride=3),
        _tiny_settings(spinup_windows=4),
        _tiny_settings(mean_error_mode="per_realization"),
        _tiny_settings(regimes=("cycled",)),
    ):
        assert base != nk.config_hash(nk.scientific_config(config, changed, "law-id"))

    changed_model = _config()
    changed_model["model"]["F"] = 8.5
    assert base != nk.config_hash(nk.scientific_config(changed_model, settings, "law-id"))


def test_calibration_spec_matches_the_published_section_7_run():
    """The reused law must be one this module would have computed itself.

    ``CALIBRATION_SPEC`` exists so that reusing section 7's restart distribution
    is an identity rather than an approximation.  If section 7's defaults ever
    move, this fails here rather than silently putting the two experiments on
    different forecast laws.
    """
    reference = rvc.ExperimentSettings()
    assert reference.calibration_ensemble_size == nk.CALIBRATION_SPEC["calibration_ensemble_size"]
    assert reference.calibration_n_mc == nk.CALIBRATION_SPEC["calibration_n_mc"]
    assert reference.calibration_n_windows == nk.CALIBRATION_SPEC["calibration_n_windows"]
    assert (
        reference.calibration_spinup_windows
        == nk.CALIBRATION_SPEC["calibration_spinup_windows"]
    )
    assert reference.eigenvalue_floor == nk.CALIBRATION_SPEC["eigenvalue_floor"]
    assert reference.shrinkage == nk.CALIBRATION_SPEC["shrinkage"]
    assert reference.restart_mode == nk.CALIBRATION_SPEC["restart_mode"]
    assert reference.n_calibration_seeds == nk.CALIBRATION_SPEC["n_calibration_seeds"]
    assert tuple(rvc.RESTART_SUPPORTED_FILTERS) == nk.CALIBRATION_SPEC["filter_names"]


def test_evaluation_seeds_match_the_published_experiments():
    """Design points sit on the same truth trajectories as sections 1-7.

    Same seed family, same evaluation/calibration split by index, so a
    ``(N=10, kappa=1)`` cell is on the trajectories the hierarchical run and
    section 7 already used, and the calibration seeds remain disjoint.
    """
    settings = nk.NKappaSettings(n_evaluation_seeds=20)
    reference = rvc.make_seed_sets(10, 20, rvc.TRUTH_SEED_BASE, rvc.CALIBRATION_SEED_OFFSET)
    assert settings.evaluation_seeds == tuple(reference[1])
    assert settings.calibration_seeds == tuple(reference[0])
    assert not set(settings.evaluation_seeds) & set(settings.calibration_seeds)


def test_decomposition_satisfies_the_plugin_and_debiased_identities():
    """``MSE = Bias^2 + Var`` exactly, and ``T = Bias^2_db + Var = MSE - Var/(M-1)``."""
    rng = np.random.default_rng(4)
    M, W, n = 8, 5, 40
    errors = rng.standard_normal((M, W, n))
    summary = rvc.summarize_cell(errors, np.zeros((M, W), dtype=bool))
    plug_in = summary["bias_squared_trace"] + summary["variance_trace"]
    assert np.allclose(plug_in, summary["mse_trace"], atol=1e-12)
    total = summary["bias_squared_debiased_trace"] + summary["variance_trace"]
    assert np.allclose(total, summary["mse_trace"] - summary["variance_trace"] / (M - 1))


def test_smoke_experiment_completes_and_produces_finite_surfaces(tmp_path):
    """End-to-end: run a tiny grid, bootstrap it, and check the surfaces.

    Also checks the shape of the response surface itself: the ``(N=5, kappa=8)``
    corner must be NaN rather than filled in, so a masked colormap leaves a
    design point that does not exist blank instead of interpolating across it.
    """
    config = nk.load_experiment_config(nk.DEFAULT_CONFIG_PATH)
    settings = nk.settings_from_config(
        config,
        ensemble_sizes=(5, 10),
        kappa_values=(1, 2, 8),
        n_mc=4,
        n_windows=3,
        window_stride=2,
        spinup_windows=2,
        n_evaluation_seeds=2,
        label="pytest",
    )
    # The test must not spend time calibrating a restart law. A clean clone has
    # no restart_vs_cycled run on disk to borrow one from, so seed the driver's
    # calibration cache with the published law instead.
    cache = nk._calibration_cache_path(config, settings, tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PUBLISHED_RESTART_LAW, cache)
    outcome = nk.run_n_kappa_experiment(
        config, settings, n_jobs=1, results_root=tmp_path, allow_calibration=False
    )
    report = nk.complete_truth_seeds(outcome["cells"], settings)
    assert len(report["common"]) == 2

    bootstrap = nk.bootstrap_grid(
        outcome["cells"], report["common"], settings, n_bootstrap=50, n_inner=10
    )
    table = nk.diagnostic_table(outcome["cells"], report["common"], settings)
    for metric in nk.METRIC_NAMES:
        grid = nk.surface(bootstrap, settings, "cycled", metric)
        assert grid.shape == (2, 3)
        assert np.isnan(grid[0, 2])  # (N=5, kappa=8) does not exist
        assert np.isfinite(grid[0, :2]).all() and np.isfinite(grid[1]).all()

    variability = nk.diagnostic_surface(
        table, settings, "restart", "projector_variability_normalized"
    )
    assert np.isnan(variability[0, 2])
    finite = variability[np.isfinite(variability)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all()

    # The decomposition-consistent total is what the surface ranks on.
    for cell, draws in bootstrap["draws"].items():
        columns = {name: draws[:, j] for j, name in enumerate(nk.METRIC_NAMES)}
        assert np.allclose(
            columns["total"], columns["bias_squared_debiased"] + columns["variance"]
        )

    csv_path, npz_path = nk.save_summary(
        outcome["results_dir"], bootstrap, table, settings, report
    )
    assert csv_path.exists() and npz_path.exists()

    # Resuming re-reads the checkpoints instead of recomputing them.
    again = nk.run_n_kappa_experiment(
        config, settings, n_jobs=1, results_root=tmp_path, allow_calibration=False
    )
    assert set(again["cells"]) == set(outcome["cells"])
    for key in outcome["cells"]:
        assert np.array_equal(
            outcome["cells"][key]["mse_trace"], again["cells"][key]["mse_trace"]
        )


def test_existing_experiments_are_untouched():
    """This module adds a filter grid; it must not redefine section 7 or the 100x100 run."""
    from src.experiments import bias_variance_mc as bv

    assert rvc.RESTART_SUPPORTED_FILTERS == ("Seq-EnKF", "4D-EnKF", "QPCA-EnDCF")
    assert rvc.REGIMES == ("restart", "cycled")
    assert bv.FILTER_NAMES == (
        "Seq-EnKF",
        "Seq-ETKF",
        "4D-EnKF",
        "4D-ETKF",
        "QPCA-EnDCF",
    )
    assert bv.SPECTRAL_DIAGNOSTIC_NAMES == (
        "leading_eigenvalue_fraction",
        "normalized_cutoff_gap",
        "retained_mean_mismatch_fraction",
    )
    # The results roots are distinct, so nothing here can overwrite a published
    # checkpoint.
    assert nk.DEFAULT_RESULTS_ROOT != rvc.DEFAULT_RESULTS_ROOT
    assert nk.DEFAULT_RESULTS_ROOT.name == "n_kappa_bias_variance"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
